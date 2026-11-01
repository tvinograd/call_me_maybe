"""Turn one request into a validated function call."""
import json

from pydantic import BaseModel

from .decoder import ConstrainedDecoder
from .grammar import ValueConstraint
from .models import FunctionCallResult, FunctionDefinition, ParamType
from .prompt import build_prompt

STRING_MAX_LENGTH = 200
OUTPUT_START = '{"name":"'
NAME_SUFFIX = '","parameters":{'
OUTPUT_END = "}}"


def key_text(name: str, param_type: ParamType) -> str:
    """Return the fixed text written before a parameter's value.

    Args:
        name: The parameter name.
        param_type: The parameter type.

    Returns:
        Example: '"s":"' (for strings) or '"a":'.
    """
    text = json.dumps(name) + ":"
    if param_type == "string":
        text += '"'
    return text


def value_constraint(param_type: ParamType, suffix: str) -> ValueConstraint:
    """Build the grammar rules for one parameter value.

    Args:
        param_type: The parameter type from the definition.
        suffix: The fixed text that follows the value.

    Returns:
        The matching constraint.
    """
    if param_type == "string":
        return ValueConstraint(
            kind="string", suffix='"' + suffix, max_length=STRING_MAX_LENGTH
            )
    if param_type == "boolean":
        return ValueConstraint(
            kind="boolean", suffix=suffix, options=["true", "false"]
            )
    return ValueConstraint(kind=param_type, suffix=suffix)


class FunctionCaller(BaseModel):
    """Generate function calls for requests with constrained decoding.

    Attributes:
        decoder: The constrained decoder.
        functions: Every function the model may call.
    """

    decoder: ConstrainedDecoder
    functions: list[FunctionDefinition]

    def call(self, request: str) -> FunctionCallResult:
        """Generate the function call for one request.

        Args:
            request: The user's natural-language request.

        Returns:
            The validated function call.
        """
        ids = self.decoder.encode(build_prompt(self.functions, request))
        ids.extend(self.decoder.encode(OUTPUT_START))
        output = OUTPUT_START

        name, written = self.decoder.generate_value(
            ids, ValueConstraint(
                kind="enum",
                suffix=NAME_SUFFIX,
                options=[function.name for function in self.functions]
                )
            )
        output += written
        function = next(f for f in self.functions if f.name == name)
        params = list(function.parameters.items())

        if params:
            fixed = key_text(params[0][0], params[0][1].type)
        else:
            fixed = OUTPUT_END
        ids.extend(self.decoder.encode(fixed))
        output += fixed

        for index, (_, spec) in enumerate(params):
            if index + 1 < len(params):
                next_name, next_spec = params[index + 1]
                suffix = ", " + key_text(next_name, next_spec.type)
            else:
                suffix = OUTPUT_END
            _, written = self.decoder.generate_value(
                ids, value_constraint(spec.type, suffix))
            output += written

        parsed = json.loads(output)
        return FunctionCallResult(
            prompt=request, name=name, parameters=parsed["parameters"]
            )

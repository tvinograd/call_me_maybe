"""Build the text the model reads before it writes its answer."""
from .models import FunctionDefinition

SYSTEM_TEXT = (
    "You turn a user request into one function call in JSON.\n"
    "Pick the function whose description matches the request "
    "and fill in its arguments.\n"
    "Available functions:\n"
)
EXAMPLE_TEXT = (
    'Example: "Multiply 3 by 4" -> '
    '{"name":"fn_multiply","parameters":{"x":3,"y":4}}\n'
)


def describe_function(function: FunctionDefinition) -> str:
    """Describe one function on a single line.

    Args:
        function: The function definition.

    Returns:
        A line like "- fn_greet(name: string): Greet a person."
    """
    params = ", ".join(f"{name}: {spec.type}"
                       for name, spec in function.parameters.items())
    return f"- {function.name}({params}): {function.description}\n"


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    """Build the full chat prompt for one request.

    Args:
        functions: Every function the model may call.
        request: The user's natural-language request.

    Returns:
        The prompt text, ending where the assistant's answer begins.
    """
    listing = "".join(describe_function(f) for f in functions)
    return (
        "<|im_start|>system\n" + SYSTEM_TEXT + listing + EXAMPLE_TEXT
        + "<|im_end|>\n"
        + "<|im_start|>user\n" + request + "<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

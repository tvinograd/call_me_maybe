"""Data models for the input files and the output results."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ParamType = Literal["number", "integer", "string", "boolean"]


class ParameterSpec(BaseModel):
    """Type of one parameter or of a return value."""

    type: ParamType


class FunctionDefinition(BaseModel):
    """One function the model may call."""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        """Reject names the name grammar cannot generate."""
        if not name.strip() or '"' in name or "\\" in name:
            raise ValueError(f"Invalid function name: {name!r}")
        return name


class PromptItem(BaseModel):
    """One natural-language request from the input file."""

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry of the output file."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, bool | int | float | str]

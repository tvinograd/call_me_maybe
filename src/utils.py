"""Read the input files and write the output file."""
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .models import FunctionCallResult, FunctionDefinition, PromptItem


def describe(error: ValidationError) -> str:
    """Summarize a pydantic validation error, one problem per line.

    Args:
        error: The validation error.

    Returns:
        Lines like "0.parameters.a.type: Input should be ...".
    """
    lines = []
    for problem in error.errors():
        location = ".".join(str(part) for part in problem["loc"])
        lines.append(f"  {location or 'file'}: {problem['msg']}")
    return "\n".join(lines)


def read_json(path: Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: The file to read.

    Returns:
        The parsed JSON data.

    Raises:
        ValueError: If the file cannot be read or is not valid JSON.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except OSError as error:
        raise ValueError(f"Cannot read {path}: {error.strerror}") from None
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not a UTF-8 text file") from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} is not valid JSON (line {error.lineno}, "
            f"column {error.colno}): {error.msg}"
        ) from None


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions.

    Args:
        path: The function definitions file.

    Returns:
        The validated function definitions.

    Raises:
        ValueError: If the file is unreadable, invalid or empty, or if
            two functions share a name.
    """
    data = read_json(path)
    try:
        functions = TypeAdapter(list[FunctionDefinition]).validate_python(
            data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid function definitions in {path}:\n{describe(error)}"
        ) from None
    if not functions:
        raise ValueError(f"{path} contains no functions")
    names = [function.name for function in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate function names in {path}: {', '.join(duplicates)}"
        )
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate the prompts.

    Args:
        path: The prompts file.

    Returns:
        The validated prompts.

    Raises:
        ValueError: If the file is unreadable or invalid.
    """
    data = read_json(path)
    try:
        return TypeAdapter(list[PromptItem]).validate_python(data)
    except ValidationError as error:
        raise ValueError(
            f"Invalid prompts in {path}:\n{describe(error)}"
        ) from None


def save_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the results as a JSON array.

    Args:
        path: The output file.
        results: The function calls to write.

    Raises:
        ValueError: If the file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump([result.model_dump() for result in results], file,
                      indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as error:
        raise ValueError(f"Cannot write {path}: {error.strerror}") from None

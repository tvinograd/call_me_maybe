"""Rules that decide which text may come next while generating a value."""
import re
from typing import Literal

from pydantic import BaseModel

ValueKind = Literal["number", "integer", "string", "boolean", "enum"]

NUMBER_RE = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
INTEGER_RE = re.compile(r"-?(0|[1-9][0-9]*)")
ESCAPABLE = set('"\\/bfnrt')


def scan_string(text: str) -> tuple[bool, bool]:
    """Check JSON string content.

    Returns (valid, ends_inside_escape).
    """
    escaped = False
    for char in text:
        if escaped:
            if char not in ESCAPABLE:
                return False, False
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"' or ord(char) < 0x20:
            return False, False
    return True, escaped


def find_end(text: str, end_char: str) -> int:
    """Return the index of the first unescaped end_char, or -1."""
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == end_char:
            return index
    return -1


class ValueConstraint(BaseModel):
    """Constraint for one JSON value followed by a fixed suffix."""

    kind: ValueKind
    suffix: str
    options: list[str] = []
    max_length: int = 64

    def is_complete(self, value: str) -> bool:
        """Return True if value is a finished, valid value."""
        if self.kind == "number":
            return NUMBER_RE.fullmatch(value) is not None
        if self.kind == "integer":
            return INTEGER_RE.fullmatch(value) is not None
        if self.kind == "string":
            valid, in_escape = scan_string(value)
            return valid and not in_escape
        return value in self.options

    def is_valid_prefix(self, value: str) -> bool:
        """Return True if value can still be extended to a complete value."""
        if self.kind in ("number", "integer"):
            return len(value) < self.max_length and (
                self.is_complete(value) or self.is_complete(value + "0"))
        if self.kind == "string":
            valid, in_escape = scan_string(value)
            if in_escape:
                return valid and len(value) < self.max_length
            return valid and len(value) <= self.max_length
        return any(option.startswith(value) for option in self.options)

    def accepts(self, generated: str, token: str) -> bool:
        """Return True if token may be appended to the generated value."""
        text = generated + token
        end = find_end(text, self.suffix[0])
        if end == -1:
            return self.is_valid_prefix(text)
        value, rest = text[:end], text[end:]
        return self.is_complete(value) and self.suffix.startswith(rest)

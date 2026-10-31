"""The Python-name contract shared by catalog loaders and code generation."""

import keyword
from typing import Any


def require_python_identifier(value: Any, context: str) -> str:
    """Return *value* if it is a Python identifier and not a keyword.

    Raise :class:`ValueError` otherwise, naming *context* and the rejected
    value. Uses Python's Unicode identifier rules; soft keywords are allowed.
    This validates names, without translating or normalizing them.
    """
    if (not isinstance(value, str)
            or not value.isidentifier()
            or keyword.iskeyword(value)):
        raise ValueError(
            f"{context} {value!r} must be a valid Python identifier and "
            f"must not be a Python keyword"
        )
    return value

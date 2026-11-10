
from __future__ import annotations

from typing import Optional, Tuple, Union

__all__ = ["Span", "SYSTEM_PROMPT"]

Span = Union[str, Tuple[int, int], None]

SYSTEM_PROMPT = (
    "You are a memory assistant. Use the provided memory excerpt if it is given, "
    "and reply with only the requested value (no explanations)."
)

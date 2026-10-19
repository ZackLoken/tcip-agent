"""The platform's non-empty text type."""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Text that must say something: surrounding whitespace is stripped and an empty value refuses."""

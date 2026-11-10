"""The platform's non-empty text type."""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""Text that must say something: surrounding whitespace is stripped and an empty value refuses."""

Crop = Text
"""A crop's name as a record holds it, refused empty as :data:`Text` is; which names exist is
crops.yml's, checked where a record is registered."""

DatasetId = Text
"""A dataset's minted id as a record holds it, refused empty as :data:`Text` is and read back
stripped."""

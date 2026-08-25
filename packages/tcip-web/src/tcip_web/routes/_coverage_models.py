"""The view-coverage record's viewing context, declared once so ``routes/coverage.py`` (the
record's writer and reader) and ``routes/images.py`` (the header values the browser folds into
that context) agree on its shape, and so ``scripts/generate_frontend_types.py`` can render it for
the browser instead of the browser hand-transcribing it.

Every model here forbids an undeclared key: a viewing context or a stored record carrying one is
refused by name rather than silently accepted and later misread.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from tcip_mcp.pipelines.band_stats import STRETCH_MODES

_STRETCH_LITERAL = ("minmax", "percent_clip", "none")
if _STRETCH_LITERAL != STRETCH_MODES:
    raise AssertionError(
        "CoverageViewing.stretch's literal vocabulary has drifted from band_stats.STRETCH_MODES; "
        "update both together"
    )


class GridGeometry(BaseModel):
    """The reference-grid geometry a coverage or completeness record was accumulated against.

    ``overlap`` must be 0: the exact-partition contract puts every native pixel in exactly one
    cell, which an overlapping lattice breaks.
    """

    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    tile_size: int
    overlap: float
    cols: int
    rows: int

    @field_validator("overlap")
    @classmethod
    def _overlap_is_zero(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError(
                "a coverage record's grid requires overlap 0: the exact-partition contract puts "
                "every native pixel in exactly one cell, which overlapping cells break"
            )
        return value


class StatsSource(BaseModel):
    """Where a served image's display-stretch bounds came from, named for the read that produced
    them.

    ``window_sample`` and ``overview`` both read part or a reduced version of the raster rather
    than the whole array a whole view served: ``pixel_fraction`` is the share of the raster's
    pixels a seeded window sample covered (1.0 means the budget happened to cover all of them
    anyway), and ``overview_scale`` is the served/native resolution ratio a reduced read came off.
    """

    model_config = ConfigDict(extra="forbid")

    read: Literal["none", "dtype_full_scale", "served_array", "window_sample", "overview"]
    seed: Optional[int] = None
    pixel_fraction: Optional[float] = None
    overview_scale: Optional[float] = None


class WorkingScaleBar(BaseModel):
    """The coarsest view scale at which annotations were committed on an image and subject this
    session, and how that scale was measured."""

    model_config = ConfigDict(extra="forbid")

    value: float
    source: str


class CoverageViewing(BaseModel):
    """The display context a coverage record carries, so a reviewer can reconstruct what was on
    screen.

    ``bands`` and ``stretch`` are the two keys a plain RGB view (no composite selected) legitimately
    omits. The other four are always present, ``None`` where nothing applies: ``display_bounds`` is
    positional against the displayed bands, in the same order ``bands`` names them, and a
    one-element list is one bound applied to every displayed band.
    """

    model_config = ConfigDict(extra="forbid")

    bands: Optional[str] = None
    stretch: Optional[Literal["minmax", "percent_clip", "none"]] = None
    stats_source: Optional[StatsSource]
    display_bounds: Optional[list[tuple[float, float]]]
    base_served_size: Optional[str]
    working_scale_bar: Optional[WorkingScaleBar]


class CoverageRecord(BaseModel):
    """The stored per-image view-coverage record, as ``get_coverage`` serves it and
    ``post_coverage`` writes it."""

    model_config = ConfigDict(extra="forbid")

    grid: GridGeometry
    cells_served_at_native: list[str]
    cells_swept: list[str]
    viewing: CoverageViewing
    updated_at: str

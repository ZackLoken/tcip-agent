"""The view-coverage record's viewing context, declared once so ``routes/coverage.py`` (the
record's writer and reader) and ``routes/images.py`` (the header values the browser folds into
that context) agree on its shape, and so ``tools/generate_frontend_types.py`` can render it for
the browser instead of the browser hand-transcribing it.

Every model here forbids an undeclared key: a viewing context or a stored record carrying one is
refused by name rather than silently accepted and later misread.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

from tcip_mcp.pipelines.band_stats import StretchMode


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


class WorkingScale(BaseModel):
    """The scale a coverage sweep is judged against: the breeder's own set inspection zoom for a
    subject (``coverage_grid_zoom``), never derived from annotation content. ``zoom`` names what
    the number is (the grid zoom the breeder set, not a bar derived from anything); ``source``
    states who set it and when ("set by user:<name> at <time>"), so a stamped comparison always
    carries its own provenance rather than a bare number."""

    model_config = ConfigDict(extra="forbid")

    zoom: float
    source: str


class CoverageViewing(BaseModel):
    """The display context a coverage record carries, so a reviewer can reconstruct what was on
    screen.

    ``bands`` and ``stretch`` are the two keys a plain RGB view (no composite selected) legitimately
    omits. The other four are always present, ``None`` where nothing applies: ``display_bounds`` is
    positional against the displayed bands, in the same order ``bands`` names them, and a
    one-element list is one bound applied to every displayed band. Each low/high value in a pair is
    a finite number or ``null``; a ``null`` bound means the stretch found no finite value for that
    half of that band's range (a raster whose sampled or served pixels held a NaN or an infinity).
    A plain view of a non-``uint8`` raster, and a band composite under ``stretch="none"``, both
    report the low as always zero: a band with no positive data still divides by a positive number
    (the magnitude of its sampled minimum), so its pair holds no pixel of the band between its low
    and high.
    """

    model_config = ConfigDict(extra="forbid")

    bands: Optional[str] = None
    stretch: Optional[StretchMode] = None
    stats_source: Optional[StatsSource]
    display_bounds: Optional[list[tuple[Optional[float], Optional[float]]]]
    base_served_size: Optional[str]


class CoverageRecord(BaseModel):
    """The stored per-image view-coverage record, as ``get_coverage`` serves it and
    ``post_coverage`` writes it.

    ``cells_seen_at_scale`` is a bound, not an attention claim: a recorded value means every
    sub-cell of that cell has, across any number of viewport moments this session or an earlier
    one, been fully on screen at or above it (``coverageTracker.ts``'s ``subCellScale``, the
    tightest bound the tracker knows). Whether a cell counts as "swept" is derived against a
    working-scale bar the record no longer carries (see ``routes/coverage.py``'s docstring and
    ``lib/coverage.ts``'s ``meetsBar``), never stored here.
    """

    model_config = ConfigDict(extra="forbid")

    grid: GridGeometry
    cells_served_at_native: list[str]
    cells_seen_at_scale: dict[str, float]
    viewing: CoverageViewing
    updated_at: str

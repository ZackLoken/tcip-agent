"""The tile geometry a pass runs at, resolved from what a caller states and what a checkpoint
records of its own training geometry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tcip_mcp.pipelines.execution import DEFAULT_OVERLAP

logger = logging.getLogger(__name__)


def _native_ratio_tile_size(train_native_size: Any) -> int | None:
    """The tile edge a checkpoint's own uniform untiled training frame justifies: the frame's edge
    when every training frame shared one square size (``train_native_size``, stamped ``[width,
    height]``), else ``None``."""
    if not isinstance(train_native_size, (list, tuple)) or len(train_native_size) != 2:
        return None
    try:
        width, height = int(train_native_size[0]), int(train_native_size[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    if width != height:
        logger.info(
            "no tile edge derivable from this checkpoint's untiled training frame (%dx%d): tile "
            "geometry is a single square edge, and no square edge reproduces a rectangular frame's "
            "scale on both axes. Pass tile_size explicitly, or run untiled.", width, height)
        return None
    return width


class TileEdgeContradiction(ValueError):
    """A caller-stated tile edge that differs from the checkpoint's own recorded tile geometry."""


@dataclass(frozen=True)
class TileGeometry:
    """A pass's tile edge and overlap, each beside the source it came from, and the resize each
    tile runs through.

    ``tile_size_source`` is ``"explicit"``, ``"derived"`` (the checkpoint's persisted training
    geometry), ``"native_ratio"`` (the square frame the checkpoint trained untiled at) or
    ``"unavailable"`` (``tile_size`` ``None``); ``tile_size_derived_from`` says why a stated edge on
    a tiled pass is trusted and is ``None`` otherwise. ``overlap_source`` is ``"explicit"``,
    ``"derived"`` or ``"default"``.
    """

    tile_size: int | None
    tile_size_source: str
    tile_size_derived_from: str | None
    overlap: float
    overlap_source: str
    tile_resize: tuple[int, int] | None


def resolve_tile_geometry(
    predictor: Any, *, tiled: bool, tile_size: int | None, overlap: float | None,
) -> TileGeometry:
    """The tile geometry a pass runs at, each value by precedence: stated, then the checkpoint's
    persisted training geometry (``train_tile_size``/``train_overlap``), then for the edge the
    square frame it trained untiled at (``train_native_size``), else the edge ``None`` and the
    overlap ``DEFAULT_OVERLAP``.

    A stated edge on a tiled pass that differs from the checkpoint's recorded edge (persisted, else
    native) raises :class:`TileEdgeContradiction` naming both. A tiled pass whose edge came from
    the native frame runs each tile through the resize the run's recorded augmentation chain
    applied (:func:`~tcip_mcp.pipelines.data.augmentations.recorded_resize`); every other pass
    runs none.
    """
    persisted = getattr(predictor, "train_tile_size", None)
    native = _native_ratio_tile_size(getattr(predictor, "train_native_size", None))
    recorded = int(persisted) if persisted is not None else native
    edge: int | None = None
    derived_from = None
    if tile_size is not None:
        edge, source = int(tile_size), "explicit"
        if tiled and recorded is None:
            derived_from = "stated on a checkpoint that records no tile geometry"
        elif tiled:
            kind = ("persisted training tile geometry" if persisted is not None
                    else "recorded untiled training frame")
            if recorded != edge:
                raise TileEdgeContradiction(
                    f"stated tile_size {edge} contradicts this checkpoint's own {kind} of "
                    f"{recorded}. Pass tile_size {recorded} to match the checkpoint, or leave "
                    "tile_size unset to derive it from the checkpoint.")
            derived_from = (
                "equal to the checkpoint's persisted training tile geometry"
                if persisted is not None else
                "equal to the edge the checkpoint's recorded untiled training frame yields, run "
                "without that frame's own recorded resize")
    elif persisted is not None:
        edge, source = int(persisted), "derived"
    elif native is not None:
        edge, source = native, "native_ratio"
    else:
        source = "unavailable"

    train_overlap = getattr(predictor, "train_overlap", None)
    if overlap is not None:
        resolved_overlap, overlap_source = float(overlap), "explicit"
    elif train_overlap is not None:
        resolved_overlap, overlap_source = float(train_overlap), "derived"
    else:
        resolved_overlap, overlap_source = DEFAULT_OVERLAP, "default"

    tile_resize = None
    if tiled and source == "native_ratio":
        from tcip_mcp.pipelines.data.augmentations import recorded_resize

        tile_resize = recorded_resize(getattr(predictor, "train_augmentation", None))
    return TileGeometry(edge, source, derived_from, resolved_overlap, overlap_source, tile_resize)

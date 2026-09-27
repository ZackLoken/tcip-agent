"""A tiled pass's execution regime as the platform's own producers state it, for tests that hand
records to a resolver, and a prepared pass around a stub predictor for tests that calibrate one."""

from __future__ import annotations

from typing import Any

from tcip_mcp.pipelines.resolution import (
    DEFAULT_OVERLAP, DEFAULT_POSTPROCESS, resolve_cross_tile_nms,
)
from tcip_mcp.pipelines.slicing import slicing_record


def tiled_regime(cross_tile_nms: float | None = None, *, postprocess: str = DEFAULT_POSTPROCESS,
                 overlap: float = DEFAULT_OVERLAP) -> dict:
    """``resolve_operating_point``'s ``slicing`` and ``cross_tile_nms`` for a tiled pass at
    ``overlap`` under ``postprocess``, its threshold stated or the documented default."""
    slicing = slicing_record(overlap, None, postprocess)
    return {"slicing": slicing,
            "cross_tile_nms": resolve_cross_tile_nms(cross_tile_nms, slicing).to_provenance()}


def stub_pass(predictor: Any, *, tile_size: int | None = None, postprocess: str | None = None,
              cross_tile_nms: float | None = None, tile_batch_size: int = 8) -> Any:
    """A prepared pass (``inference_tools._PreparedPass``) around ``predictor``, untiled unless a
    ``postprocess`` is named, its geometry and merge threshold resolved by the platform's own
    resolvers; identity and scope are placeholders a calibration never reads."""
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.inference.predictor import resolve_tile_geometry
    from tcip_mcp.tools.inference_tools import _PreparedPass

    tiled = postprocess is not None
    geometry = resolve_tile_geometry(predictor, tiled=tiled, tile_size=tile_size, overlap=None)
    slicing = slicing_record(geometry.overlap, geometry.tile_resize, postprocess) if tiled else None
    return _PreparedPass(
        checkpoint_path="stub.pt", predictor=predictor, images_dir=None, paths=[],
        identity={"sha256": "stub", "experiment_id": None},
        scope=ClassScope.recorded_in(getattr(predictor, "config", {}).get("data") or {}),
        id_map=None, geometry=geometry, slicing=slicing,
        cross_tile_nms=resolve_cross_tile_nms(cross_tile_nms, slicing), conf=0.5, max_dets=None,
        conf_stated=False, max_dets_stated=False, tile_batch_size=tile_batch_size)

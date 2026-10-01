"""The execution record a pass runs under, and the pass prepared from a checkpoint and that record.

One record states everything that decides which predictions a pass produces: the confidence
threshold and the full-frame cap, and for a tiled pass the tile edge, overlap, per-tile resize,
the cross-tile merge (its SAHI type and match metric, resolved once) and the threshold it merges
at. The record a pass executes is the record a bucket or an assessment stamps.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.inference.predictor import TileGeometry

logger = logging.getLogger(__name__)

DEFAULT_CONF = 0.5
"""The confidence a pass runs at when its caller states none and no assessment supplies one."""
DEFAULT_NMS_IOU = 0.3
"""The cross-tile merge threshold when none is stated and the ground truth in hand derives none."""
DEFAULT_OVERLAP = 0.2
"""The tile overlap when none is stated and the checkpoint records none."""
DEFAULT_MAX_DETS = 1000
"""The full-frame detection cap when none is stated, high enough that a dense scene is not
truncated at a framework default."""
DEFAULT_TILE_BATCH_SIZE = 96
"""Tiles per forward batch: a throughput setting that changes no prediction."""
DEFAULT_IMAGE_BATCH_SIZE = 16
"""Untiled images per forward batch: a throughput setting that changes no prediction."""
DEFAULT_POSTPROCESS = "nms"
"""The cross-tile merge when a caller names none: suppress overlaps rather than union them."""
CROSS_TILE_MERGES: dict[str, tuple[str, str]] = {
    "nms": ("NMS", "IOU"), "nmm": ("NMM", "IOS"), "greedynmm": ("GREEDYNMM", "IOS"),
}
"""Every cross-tile merge a caller may name, as the SAHI postprocess type it runs and the match
metric that type compares over (intersection over union, or over the smaller box)."""


def cross_tile_merge_rule(postprocess: str) -> tuple[str, str]:
    """The ``(SAHI postprocess type, match metric)`` ``postprocess`` names; any other name refuses
    with the vocabulary."""
    try:
        return CROSS_TILE_MERGES[postprocess]
    except KeyError:
        raise ValueError(
            f"postprocess {postprocess!r} names no cross-tile merge; choose one of "
            f"{sorted(CROSS_TILE_MERGES)}") from None


class ExecutionRefused(ValueError):
    """A pass that cannot run as stated: a tile edge the checkpoint contradicts, a tiled pass with
    no basis for its scale, or a stated value that differs from the record being restored."""


@dataclass(frozen=True)
class Execution:
    """One pass's complete execution record.

    ``conf`` and ``max_dets`` are ``None`` for a checkpoint whose head is not a detector. The tiled
    fields are ``None`` for an untiled pass. ``sources`` names, per field that holds a value, where
    the value came from, in :class:`~tcip_mcp.pipelines.inference.predictor.TileGeometry`'s words
    (``explicit`` for a value the caller stated, ``derived``, ``native_ratio``, ``default``), or the
    derivation or fit that produced it, by name.
    """

    conf: float | None
    max_dets: int | None
    tile_size: int | None
    overlap: float | None
    tile_resize: tuple[int, int] | None
    postprocess: str | None
    merge_type: str | None
    match_metric: str | None
    cross_tile_nms: float | None
    sahi_version: str | None
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def tiled(self) -> bool:
        return self.tile_size is not None

    def record(self) -> dict[str, Any]:
        """The record as JSON values."""
        out = asdict(self)
        out["tile_resize"] = list(self.tile_resize) if self.tile_resize is not None else None
        return out

    @classmethod
    def of(cls, record: dict[str, Any]) -> Execution:
        """The execution a :meth:`record` states. A record missing a field refuses naming it."""
        missing = sorted(set(cls.__dataclass_fields__) - set(record))
        if missing:
            raise ValueError(f"the execution record carries no {missing}: it states only part of "
                             "a pass, so no pass can be restored from it.")
        resize = record["tile_resize"]
        return cls(**{**{k: record[k] for k in cls.__dataclass_fields__},
                      "tile_resize": tuple(resize) if resize is not None else None,
                      "sources": dict(record["sources"])})

    def with_value(self, name: str, value: Any, source: str) -> Execution:
        """This record with ``name`` set to ``value`` from ``source``."""
        return replace(self, **{name: value}, sources={**self.sources, name: source})


class Stated(BaseModel):
    """What a caller states of an execution record, every field ``None`` when unstated: whether
    the pass is tiled, its tile edge and overlap, the cross-tile merge and its threshold, the
    confidence threshold and the full-frame cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tile: bool | None = None
    tile_size: int | None = None
    overlap: float | None = None
    postprocess: str | None = None
    cross_tile_nms: float | None = None
    conf: float | None = None
    max_dets: int | None = None

    def differences(self, restored: Execution) -> list[str]:
        """Each stated value, the tiled flag included, that ``restored`` states differently."""
        differing = ([f"tile stated {self.tile!r}, recorded {restored.tiled!r}"]
                     if self.tile is not None and self.tile != restored.tiled else [])
        for name in Stated.model_fields.keys() - {"tile"}:
            value = getattr(self, name)
            if value is not None and value != getattr(restored, name):
                differing.append(f"{name} stated {value!r}, recorded {getattr(restored, name)!r}")
        return sorted(differing)


@dataclass
class Pass:
    """A predictor built from a verified checkpoint and running under one execution record, the
    images it runs over and the class scope its labels decode under."""

    checkpoint: VerifiedCheckpoint
    predictor: Any
    scope: ClassScope
    execution: Execution
    tile_batch_size: int
    paths: list[Any] = field(default_factory=list)

    def predict(self, sources: list[str | BandGroupRef | Any],
                execution: Execution | None = None) -> list[dict]:
        """One result per source under the pass's own execution record, or under ``execution``
        where one is given."""
        return self.predictor.predict_batch(
            sources, execution=execution or self.execution, tile_batch_size=self.tile_batch_size)

    def derive_merge(self, gt_boxes_per_image: list[list[list[float]]]) -> None:
        """For a tiled pass whose merge threshold was not stated, the threshold the ground truth's
        neighbor-overlap tail derives in the record's own match metric (``gt_boxes_per_image`` one
        list of xywh boxes per image); the record is left as it is when nothing derives."""
        from tcip_mcp.pipelines.derivations import (
            CROSS_TILE_NMS_DERIVATIONS, derive_cross_tile_nms,
        )

        record = self.execution
        if not record.tiled or record.sources.get("cross_tile_nms") == "explicit":
            return
        assert record.match_metric is not None
        value = derive_cross_tile_nms(gt_boxes_per_image, metric=record.match_metric)
        if value is not None:
            self.execution = record.with_value(
                "cross_tile_nms", value, CROSS_TILE_NMS_DERIVATIONS[record.match_metric])


def prepare_pass(
    checkpoint: VerifiedCheckpoint, stated: Stated = Stated(), *, images_dir: str | None = None,
    device: str | None = None, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    restored: Execution | None = None,
) -> Pass:
    """The pass ``checkpoint`` runs, over ``images_dir``'s logical images (none for a raster pass).

    With ``restored`` the pass runs exactly that record, and a value ``stated`` states differently
    refuses (:class:`ExecutionRefused`) naming each. Otherwise each value is the stated one, else
    the checkpoint's own recorded geometry, else a documented default, its source recorded; a
    stated tile edge the checkpoint's geometry contradicts, and a tiled pass with no basis for its
    edge, refuse.
    """
    import sahi

    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.inference.predictor import TileEdgeContradiction, resolve_tile_geometry

    logical = list_logical_images(images_dir) if images_dir is not None else {}
    paths = [logical[stem] for stem in sorted(logical)]

    if restored is not None:
        differing = stated.differences(restored)
        if restored.tiled and restored.sahi_version != sahi.__version__:
            differing.append(f"sahi {sahi.__version__} installed, {restored.sahi_version} recorded")
        if differing:
            raise ExecutionRefused(
                f"the recorded execution differs from this call: {'; '.join(differing)}. A pass "
                "restored from a record runs exactly that record; drop the differing argument "
                "or start a new pass.")
        execution = restored
    else:
        execution = untiled_execution(checkpoint, conf=stated.conf, max_dets=stated.max_dets)

    predictor = GenericPredictor(checkpoint, device=device)
    if restored is None and (getattr(predictor, "train_tile_size", None) is not None
                             if stated.tile is None else stated.tile):
        try:
            geometry = resolve_tile_geometry(predictor, tiled=True, tile_size=stated.tile_size,
                                             overlap=stated.overlap)
        except TileEdgeContradiction as exc:
            raise ExecutionRefused(str(exc)) from exc
        if geometry.tile_size is None:
            raise ExecutionRefused(
                f"tile_size could not be resolved for {checkpoint.path}: this checkpoint carries "
                "no persisted training tile geometry, no tile_size was given explicitly, and its "
                "untiled training frame yields no square tile edge, so tiled inference has no "
                "basis to run at. Pass tile_size explicitly, or run untiled.")
        execution = tiled_execution(execution, geometry, postprocess=stated.postprocess,
                                    cross_tile_nms=stated.cross_tile_nms)
    return Pass(checkpoint=checkpoint, predictor=predictor,
                scope=ClassScope.of(checkpoint.data_config), execution=execution,
                tile_batch_size=tile_batch_size, paths=paths)


def tiled_execution(base: Execution, geometry: TileGeometry, *, postprocess: str | None,
                    cross_tile_nms: float | None) -> Execution:
    """``base`` tiled at ``geometry``, merging across tiles by ``postprocess`` at
    ``cross_tile_nms``, each the documented default when ``None``, every value's source recorded,
    under the installed SAHI version."""
    import sahi

    merge = postprocess or DEFAULT_POSTPROCESS
    merge_type, match_metric = cross_tile_merge_rule(merge)
    return replace(
        base, tile_size=geometry.tile_size, overlap=geometry.overlap,
        tile_resize=geometry.tile_resize, postprocess=merge, merge_type=merge_type,
        match_metric=match_metric,
        cross_tile_nms=float(cross_tile_nms) if cross_tile_nms is not None else DEFAULT_NMS_IOU,
        sahi_version=sahi.__version__,
        sources={**base.sources, "tile_size": geometry.tile_size_source,
                 "overlap": geometry.overlap_source,
                 "postprocess": "explicit" if postprocess else "default",
                 "cross_tile_nms": "explicit" if cross_tile_nms is not None else "default"})


def untiled_execution(checkpoint: VerifiedCheckpoint, *, conf: float | None,
                      max_dets: int | None) -> Execution:
    """The untiled record a fresh pass of ``checkpoint`` starts from: for a detector the stated
    conf and cap, else the documented defaults, each with its source; for any other head neither,
    and a stated conf or cap refuses (:class:`ExecutionRefused`)."""
    sources: dict[str, str] = {}
    if checkpoint.task in ("detection", "instance_seg"):
        sources = {"conf": "explicit" if conf is not None else "default",
                   "max_dets": "explicit" if max_dets is not None else "default"}
        conf = float(conf) if conf is not None else DEFAULT_CONF
        max_dets = int(max_dets) if max_dets is not None else DEFAULT_MAX_DETS
    elif conf is not None or max_dets is not None:
        raise ExecutionRefused(
            f"{checkpoint.path} is a {checkpoint.task!r} checkpoint: a confidence threshold and a "
            "detection cap govern a detector's boxes and nothing this head produces.")
    return Execution(conf=conf, max_dets=max_dets, tile_size=None, overlap=None, tile_resize=None,
                     postprocess=None, merge_type=None, match_metric=None, cross_tile_nms=None,
                     sahi_version=None, sources=sources)

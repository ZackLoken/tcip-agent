"""The execution record a pass runs under, and the pass prepared from a checkpoint and that record.

One record states everything that decides which predictions a pass produces: the confidence
threshold and the object density each frame's detection cap scales by, and for a tiled pass the
tile edge, overlap, per-tile resize, the cross-tile merge by name (its SAHI type and match
metric resolved from the name where they are read) and the threshold it merges at. The record a
pass executes is the record a bucket or an assessment stamps.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.derivations import Region
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.slicing import TileGeometry

logger = logging.getLogger(__name__)

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


def cross_tile_merge_of(name: str) -> tuple[str, str]:
    """The SAHI postprocess type and match metric the cross-tile merge ``name`` runs
    (:data:`CROSS_TILE_MERGES`); a name it does not hold refuses (``ValueError``) naming the
    ones it does."""
    from tcip_mcp.pipelines.model_build import resolve_named

    return resolve_named(name, CROSS_TILE_MERGES, kind="cross-tile merge")


class ExecutionRefusedError(ValueError):
    """A pass that cannot run as stated: a tile edge the checkpoint contradicts, a pass with no
    basis for its scale or its density, or a stated value that differs from the record being
    restored."""


@dataclass(frozen=True)
class Execution:
    """One pass's complete execution record.

    ``conf`` and ``density`` (objects per pixel, which ``derivations.detection_cap`` scales to a
    frame) are ``None`` for a checkpoint whose head is not a detector. The tiled fields are
    ``None`` for an untiled pass. A record stating a tile edge without an overlap, a conf without
    a density, or either reverse, refuses (``ValueError``) at construction. ``sources`` names, per
    field that holds a value, where the value came from, in
    :class:`~tcip_mcp.pipelines.slicing.TileGeometry`'s words (``explicit`` for a value the caller
    stated, ``derived`` for the checkpoint's own recorded one, ``native_ratio``), or the
    derivation or fit that produced it, by name.
    """

    conf: float | None
    density: float | None
    tile_size: int | None
    overlap: float | None
    tile_resize: tuple[int, int] | None
    postprocess: str | None
    cross_tile_nms: float | None
    sahi_version: str | None
    sources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for first, second, whole in (("tile_size", "overlap", "a tiled pass runs on one lattice"),
                                     ("conf", "density", "a detector's pass keeps each frame's "
                                      "detections by both")):
            if (getattr(self, first) is None) != (getattr(self, second) is None):
                raise ValueError(
                    f"the execution record states {first} {getattr(self, first)!r} and {second} "
                    f"{getattr(self, second)!r}: {whole}, so no pass runs from a record stating "
                    "one alone.")

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
    the pass is tiled, its tile edge and overlap, the cross-tile merge and its threshold, and the
    confidence threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tile: bool | None = None
    tile_size: int | None = None
    overlap: float | None = None
    postprocess: str | None = None
    cross_tile_nms: float | None = None
    conf: float | None = None

    def differences(self, restored: Execution) -> list[str]:
        """Each stated value, the tiled flag included, that ``restored`` states differently."""
        differing = ([f"tile stated {self.tile!r}, recorded {restored.tiled!r}"]
                     if self.tile is not None and self.tile != restored.tiled else [])
        for name in Stated.model_fields.keys() - {"tile"}:
            value = getattr(self, name)
            if value is not None and value != getattr(restored, name):
                differing.append(f"{name} stated {value!r}, recorded {getattr(restored, name)!r}")
        return sorted(differing)


@dataclass(frozen=True)
class Reference:
    """The ground truth a caller holding a calibration reference supplies a pass's unstated
    values from: its ``regions`` (``derivations.Region``, each its objects and its pixel area),
    the basis of the density, the cross-tile merge threshold and a tile geometry the checkpoint
    does not record."""

    regions: list[Region]


@dataclass
class Pass:
    """A predictor built from a verified checkpoint and running under one complete execution
    record, the images it runs over and the class scope its labels decode under."""

    checkpoint: VerifiedCheckpoint
    predictor: GenericPredictor
    scope: ClassScope
    execution: Execution
    tile_batch_size: int
    paths: list[Any] = field(default_factory=list)

    def predict(self, sources: list[Any],
                execution: Execution | None = None) -> list[dict]:
        """One result per source under the pass's own execution record, or under ``execution``
        where one is given."""
        return self.predictor.predict_batch(
            sources, execution=execution or self.execution, tile_batch_size=self.tile_batch_size)


@dataclass
class Preparation:
    """A checkpoint readied for a pass and holding no execution record yet: its predictor, the
    class scope its labels decode under, the tile geometry stated or recorded for a tiled pass
    (``None`` untiled; a value neither states left for :func:`execution_record`), what the caller
    stated, the record being restored where one is, and the images the
    pass runs over. :meth:`runnable` constructs the pass."""

    checkpoint: VerifiedCheckpoint
    predictor: GenericPredictor
    scope: ClassScope
    geometry: TileGeometry | None
    stated: Stated
    restored: Execution | None
    tile_batch_size: int
    paths: list[Any]

    def runnable(self, reference: Reference | None = None) -> Pass:
        """The pass under the restored record, or under the record :func:`execution_record`
        resolves from what was stated and ``reference``."""
        execution = (self.restored if self.restored is not None else
                     execution_record(self.checkpoint, self.stated, self.geometry, reference))
        return Pass(checkpoint=self.checkpoint, predictor=self.predictor, scope=self.scope,
                    execution=execution, tile_batch_size=self.tile_batch_size, paths=self.paths)


def prepare(
    checkpoint: VerifiedCheckpoint, stated: Stated = Stated(), *, images_dir: str | None = None,
    device: str | None = None, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    restored: Execution | None = None,
) -> Preparation:
    """``checkpoint`` readied for a pass over ``images_dir``'s logical images (none for a raster
    pass). With ``restored`` the pass will run exactly that record, and a value ``stated`` states
    differently refuses (:class:`ExecutionRefusedError`) naming each. Otherwise the pass is tiled
    as stated, else as the checkpoint trained; its tile edge and overlap are the stated ones,
    else the checkpoint's own recorded geometry (:func:`~tcip_mcp.pipelines.slicing.
    resolve_tile_geometry`), else left for :func:`execution_record` to derive, a stated edge the
    checkpoint's geometry contradicts refusing."""
    import sahi

    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.pipelines.slicing import TileEdgeContradictionError, resolve_tile_geometry

    logical = list_logical_images(images_dir) if images_dir is not None else {}
    paths = [logical[stem] for stem in sorted(logical)]

    if restored is not None:
        differing = stated.differences(restored)
        if restored.tiled and restored.sahi_version != sahi.__version__:
            differing.append(f"sahi {sahi.__version__} installed, {restored.sahi_version} recorded")
        if differing:
            raise ExecutionRefusedError(
                f"the recorded execution differs from this call: {'; '.join(differing)}. A pass "
                "restored from a record runs exactly that record; drop the differing argument "
                "or start a new pass.")

    predictor = GenericPredictor(checkpoint, device=device)
    geometry = None
    if restored is None and (predictor.train_tile_size is not None
                             if stated.tile is None else stated.tile):
        try:
            geometry = resolve_tile_geometry(predictor, tiled=True, tile_size=stated.tile_size,
                                             overlap=stated.overlap)
        except TileEdgeContradictionError as exc:
            raise ExecutionRefusedError(str(exc)) from exc
    return Preparation(checkpoint=checkpoint, predictor=predictor,
                       scope=checkpoint.spec.data.recorded_scope, geometry=geometry,
                       stated=stated, restored=restored, tile_batch_size=tile_batch_size,
                       paths=paths)


def execution_record(checkpoint: VerifiedCheckpoint, stated: Stated,
                     geometry: TileGeometry | None, reference: Reference | None) -> Execution:
    """The one complete execution record a fresh pass of ``checkpoint`` runs under, each value's
    source recorded. A detector's ``conf`` is the stated one; its ``density`` the one
    ``reference``'s regions derive (``derivations.derive_object_density``), else the one
    the checkpoint recorded of its training regions (``data.train_object_density``).
    Tiled at ``geometry``, an edge or overlap it leaves unresolved derives from ``reference``'s
    regions (``derivations.derive_tile_geometry``), each recorded by the derivation that
    produced it, and it merges across tiles by ``postprocess`` (the documented default when
    unstated) at the stated ``cross_tile_nms``, else, for a merge comparing by IoU, the
    threshold ``reference``'s regions derive (``derivations.derive_cross_tile_nms``). Every
    value left with no statement and no basis
    refuses (:class:`ExecutionRefusedError`) naming it, and a conf stated for a head that is no
    detector refuses too."""
    import sahi

    from tcip_mcp.pipelines.derivations import (
        CROSS_TILE_NMS_DERIVATION, OBJECT_DENSITY_DERIVATION, TILE_EDGE_DERIVATION,
        TILE_OVERLAP_DERIVATION, derive_cross_tile_nms, derive_object_density,
        derive_tile_geometry,
    )
    from tcip_mcp.pipelines.model_contract import DETECTION_TASKS

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    if checkpoint.task in DETECTION_TASKS:
        if stated.conf is None:
            raise ExecutionRefusedError(
                f"{checkpoint.path} is a detector and this pass states no conf: state the "
                "confidence its boxes are kept at, or run under an assessment's record")
        values["conf"], sources["conf"] = float(stated.conf), "explicit"
        if reference is not None:
            values["density"] = derive_object_density(reference.regions)
            sources["density"] = OBJECT_DENSITY_DERIVATION
        elif checkpoint.spec.data.train_object_density is not None:
            values["density"] = checkpoint.spec.data.train_object_density
            sources["density"] = "derived"
        else:
            raise ExecutionRefusedError(
                f"{checkpoint.path} records no object density of its training frames, so no "
                "frame's detection cap has a basis: train it with launch_training over label "
                "documents, or run under an assessment's record")
    elif stated.conf is not None:
        raise ExecutionRefusedError(
            f"{checkpoint.path} is a {checkpoint.task!r} checkpoint: a confidence threshold "
            "governs a detector's boxes and nothing this head produces.")
    if geometry is None:
        return Execution(conf=values.get("conf"), density=values.get("density"),
                         tile_size=None, overlap=None, tile_resize=None, postprocess=None,
                         cross_tile_nms=None, sahi_version=None, sources=sources)

    edge, overlap = geometry.tile_size, geometry.overlap
    if edge is None or overlap is None:
        if reference is None:
            raise ExecutionRefusedError(
                f"a tiled pass of {checkpoint.path} has no "
                f"{'tile_size' if edge is None else 'overlap'}: state it, train the checkpoint "
                "tiled so it records its lattice, or run under an assessment's reference whose "
                "ground truth derives it")
        edge, overlap = derive_tile_geometry(reference.regions, tile_size=edge, overlap=overlap)
    sources["tile_size"] = (geometry.tile_size_source if geometry.tile_size is not None
                            else TILE_EDGE_DERIVATION)
    sources["overlap"] = (geometry.overlap_source if geometry.overlap is not None
                          else TILE_OVERLAP_DERIVATION)

    merge = DEFAULT_POSTPROCESS if stated.postprocess is None else stated.postprocess
    match_metric = cross_tile_merge_of(merge)[1]
    threshold = stated.cross_tile_nms
    if threshold is not None:
        sources["cross_tile_nms"] = "explicit"
    elif match_metric != "IOU":
        raise ExecutionRefusedError(
            f"a tiled pass merging by {merge!r} compares by {match_metric} and has no "
            "cross_tile_nms: state it, since no derivation stands behind an IoS threshold")
    elif reference is None:
        raise ExecutionRefusedError(
            f"a tiled pass merging by {merge!r} has no cross_tile_nms: state it, since this pass "
            "holds no reference ground truth to derive it from")
    else:
        threshold = derive_cross_tile_nms(reference.regions)
        if threshold is None:
            raise ExecutionRefusedError(
                "the reference's ground truth derives no cross-tile merge threshold, since no two "
                "of its objects in one region overlap; state cross_tile_nms")
        sources["cross_tile_nms"] = CROSS_TILE_NMS_DERIVATION
    return Execution(
        conf=values.get("conf"), density=values.get("density"), tile_size=edge, overlap=overlap,
        tile_resize=geometry.tile_resize, postprocess=merge, cross_tile_nms=float(threshold),
        sahi_version=sahi.__version__,
        sources={**sources, "postprocess": "explicit" if stated.postprocess else "default"})

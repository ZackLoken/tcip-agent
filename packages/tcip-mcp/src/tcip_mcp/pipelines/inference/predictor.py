"""Model-kind contract + the predictor factory.

Inference dispatches on a model kind sniffed from the checkpoint; a tcip checkpoint (a bespoke
model built by an agent-written importable builder) is the one kind implemented.

Kind travels three ways: stamped on tcip checkpoints at save time (a top-level ``kind`` key),
recorded on the registry entry, and, for a foreign ``.pt`` this platform never wrote, sniffed from
the checkpoint's top-level keys. An undeterminable kind raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY, STATE_DICT_KEY
from tcip_mcp.pipelines.resolution import (
    DEFAULT_IMAGE_BATCH_SIZE, DEFAULT_NMS_IOU, DEFAULT_OVERLAP, DEFAULT_POSTPROCESS,
    DEFAULT_TILE_BATCH_SIZE,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.inference.generic_predictor import WindowedRasterReader

logger = logging.getLogger(__name__)

# A bespoke, from-scratch model built by an agent-written importable builder (model_source).
# Reproduced by re-importing that builder, never by exec.
KIND_TCIP_MODULE = "tcip_module"
DEFAULT_KIND = KIND_TCIP_MODULE


@runtime_checkable
class Predictor(Protocol):
    """The inference surface every model kind exposes (``GenericPredictor``).

    Detection result dict: ``{image, width, height, boxes[[x1,y1,x2,y2] px], scores[], labels[],
    count, cap_hit}`` (sliced adds ``tiles``), untiled and sliced alike. For ``task ==
    "instance_seg"`` it carries ``masks`` too (sliced only with ``require_masks``): one
    ``{"segmentation"}`` of SAHI polygons per detection, in full-image pixels, same order as
    ``boxes``/``scores``/``labels``. Every coordinate in a result is in the source's real pixel
    space. Labels are 1-indexed foreground (background = 0).
    """

    task: str
    in_chans: int
    # The loaded module and its device, read by the count calibrator and the tiled regime, and
    # the operating point the callers set on a predictor after resolving it against the data.
    model: torch.nn.Module
    device: torch.device
    score_threshold: float | None
    max_dets: int | None

    # Each image argument may be a plain path/string or a BandGroupRef (a band-grouped capture,
    # see pipelines.data.band_groups), the same image sources image_utils.list_logical_images/
    # resolve_image_source hand every other reader in this platform.
    def predict(self, image_path: str | Path | BandGroupRef) -> dict: ...

    def predict_batch(
        self, image_paths: list[str | Path | BandGroupRef], tile: bool = False,
        tile_size: int | None = None, overlap: float = DEFAULT_OVERLAP, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
        cross_tile_nms: float = DEFAULT_NMS_IOU, batch_size: int = DEFAULT_IMAGE_BATCH_SIZE, postprocess: str = DEFAULT_POSTPROCESS,
        *, require_masks: bool = True, tile_resize: tuple[int, int] | None = None,
    ) -> list[dict]: ...

    def predict_sliced(
        self, source: "str | Path | BandGroupRef | WindowedRasterReader", *, tile_size: int,
        overlap: float, postprocess: str, cross_tile_nms: float, tile_batch_size: int,
        tile_resize: tuple[int, int] | None, require_masks: bool, source_label: str = "",
        prior: dict | None = None, progress: "Callable[[int, int, dict], None] | None" = None,
    ) -> dict: ...


def _require_dict_payload(ckpt: Any, checkpoint_path: str) -> dict:
    """``ckpt`` as a payload dict, or refuse naming the structural markers it lacks."""
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Cannot determine model kind for {checkpoint_path}: checkpoint did not unpickle to a "
            f"dict (got {type(ckpt).__name__}), so it carries none of the markers ('kind', "
            f"{STATE_DICT_KEY!r}) this platform's own checkpoints do."
        )
    return ckpt


def _kind_from_ckpt(ckpt: Any, checkpoint_path: str) -> str:
    """Resolve the kind from an already-loaded checkpoint object (no second disk read): its stamped
    ``kind``, else a tcip module when its config names a model source and it carries weights."""
    ckpt = _require_dict_payload(ckpt, checkpoint_path)
    stamped = ckpt.get("kind")
    if stamped:
        return str(stamped)
    if (ckpt.get("config") or {}).get(MODEL_SOURCE_KEY) and STATE_DICT_KEY in ckpt:
        return KIND_TCIP_MODULE
    raise ValueError(
        f"Cannot determine model kind for {checkpoint_path}: no 'kind' and no config "
        f"{MODEL_SOURCE_KEY!r} beside {STATE_DICT_KEY!r} (tcip). "
        f"Top-level keys: {sorted(ckpt)[:12]}"
    )


def detect_kind(checkpoint_path: str) -> str:
    """Return the model kind for a checkpoint on disk (loads it once to sniff)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _kind_from_ckpt(ckpt, checkpoint_path)


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


def build_predictor(
    checkpoint: "VerifiedCheckpoint", *, kind: str | None = None, **kwargs: Any,
) -> "Predictor":
    """Construct the right predictor for a checkpoint's kind.

    ``checkpoint`` is a :class:`~tcip_mcp.model_registry.VerifiedCheckpoint`, the object
    :func:`~tcip_mcp.model_registry.load_registered_checkpoint` returns; this function reads no
    file itself. Pass ``kind`` to skip detection (e.g. the registry already recorded it); the
    payload it sniffs from is ``checkpoint.payload``. Predictor kwargs (``device``,
    ``score_threshold``, ``max_dets``) pass through; the model's own NMS stays as its builder
    constructed it.
    """
    if kind is None:
        kind = _kind_from_ckpt(checkpoint.payload, checkpoint.path)

    if kind == KIND_TCIP_MODULE:
        # A tcip checkpoint GenericPredictor rebuilds by re-importing the bespoke model_source
        # builder through build_model, never exec.
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        return GenericPredictor(checkpoint, **kwargs)
    raise ValueError(f"Unsupported model kind {kind!r} for {checkpoint.path}")

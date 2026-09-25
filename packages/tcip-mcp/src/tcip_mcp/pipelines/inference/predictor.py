"""Model-kind contract + the predictor factory.

Inference dispatches on a model kind sniffed from the checkpoint; a tcip checkpoint (a bespoke
model built by an agent-written importable builder) is the one kind implemented.

Kind travels three ways: stamped on tcip checkpoints at save time (a top-level ``kind`` key),
recorded on the registry entry, and, for a foreign ``.pt`` this platform never wrote, sniffed from
the checkpoint's top-level keys. An undeterminable kind raises.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Protocol, runtime_checkable

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
    count}`` (tiled adds ``tiles``). For ``task == "instance_seg"``, ``predict``/``predict_batch``
    (untiled) additionally carry ``masks``, one soft (unbinarized) ``[H, W]`` probability array per
    surviving detection, already in full-image coordinates, same order as
    ``boxes``/``scores``/``labels``. ``predict_tiled`` also carries ``masks`` by default
    (``require_masks=True``), but tile-local: a list of ``{"mask_patch", "offset_x", "offset_y"}``
    dicts, never interchangeable with the untiled shape; ``require_masks=False`` returns no
    ``masks`` key. ``predict_tiled(..., tile_resize=(w, h))`` resizes each tile to ``(w, h)``
    before the forward pass and maps every returned box (and mask) back into the tile's own native
    pixel space; every coordinate in a result is in the source's real pixel space. Labels are
    1-indexed foreground (background = 0).
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
        global_nms_iou: float = DEFAULT_NMS_IOU, batch_size: int = DEFAULT_IMAGE_BATCH_SIZE, postprocess: str = DEFAULT_POSTPROCESS,
        *, require_masks: bool = True, tile_resize: tuple[int, int] | None = None,
    ) -> list[dict]: ...

    def predict_tiled(
        self, source: "str | Path | BandGroupRef | WindowedRasterReader",
        tile_size: int | None = None, overlap: float = DEFAULT_OVERLAP, tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
        global_nms_iou: float = DEFAULT_NMS_IOU, postprocess: str = DEFAULT_POSTPROCESS, *,
        require_masks: bool = True, source_label: str = "",
        tile_resize: tuple[int, int] | None = None, prior: dict | None = None,
        progress: "Callable[[int, int, dict], None] | None" = None,
    ) -> dict: ...


def _require_dict_payload(ckpt: Any, checkpoint_path: str) -> dict:
    """The isinstance check every checkpoint reader needs before trusting ``ckpt`` as a payload
    dict, shared so a verified load and the kind sniff raise the identical fact one way."""
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Cannot determine model kind for {checkpoint_path}: checkpoint did not unpickle to a "
            f"dict (got {type(ckpt).__name__}), so it carries none of the structural markers "
            f"('kind' / {MODEL_SOURCE_KEY!r}+{STATE_DICT_KEY!r}) this platform's own "
            "checkpoints do."
        )
    return ckpt


def _kind_from_ckpt(ckpt: Any, checkpoint_path: str) -> str:
    """Resolve the kind from an already-loaded checkpoint object (no second disk read)."""
    ckpt = _require_dict_payload(ckpt, checkpoint_path)
    stamped = ckpt.get("kind")
    if stamped:
        return str(stamped)
    # Structural fallback for tcip checkpoints whose kind wasn't stamped.
    if MODEL_SOURCE_KEY in ckpt and STATE_DICT_KEY in ckpt:
        return KIND_TCIP_MODULE
    raise ValueError(
        f"Cannot determine model kind for {checkpoint_path}: no 'kind'/{MODEL_SOURCE_KEY!r} "
        f"(tcip). "
        f"Top-level keys: {sorted(ckpt)[:12]}"
    )


def detect_kind(checkpoint_path: str) -> str:
    """Return the model kind for a checkpoint on disk (loads it once to sniff)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _kind_from_ckpt(ckpt, checkpoint_path)


def _native_ratio_tile_size(train_native_size: Any) -> tuple[int | None, str]:
    """The tile edge a checkpoint's own uniform untiled training frame justifies, if any.

    A checkpoint trained untiled on frames that all shared one square size (``train_native_size``,
    stamped ``[width, height]``) justifies that edge. A rectangular frame yields ``"unavailable"``.
    """
    if not isinstance(train_native_size, (list, tuple)) or len(train_native_size) != 2:
        return None, "unavailable"
    try:
        width, height = int(train_native_size[0]), int(train_native_size[1])
    except (TypeError, ValueError):
        return None, "unavailable"
    if width <= 0 or height <= 0:
        return None, "unavailable"
    if width != height:
        logger.info(
            "no tile edge derivable from this checkpoint's untiled training frame (%dx%d): tile "
            "geometry is a single square edge, and no square edge reproduces a rectangular frame's "
            "scale on both axes. Pass tile_size explicitly, or run untiled.", width, height)
        return None, "unavailable"
    return width, "native_ratio"


def resolve_tile_geometry(
    predictor: Any, *, tile_size: int | None, overlap: float | None,
) -> tuple[int | None, str, float, str]:
    """Resolve tile_size/overlap by precedence: explicit caller value > the checkpoint's own
    persisted training geometry (``predictor.train_tile_size``/``train_overlap``, source
    ``"derived"``) > a native-size ratio tier (source ``"native_ratio"``, the square frame size a
    checkpoint trained untiled at a uniform ``predictor.train_native_size`` records, see
    :func:`_native_ratio_tile_size`) > no real basis at all (source ``"unavailable"``,
    ``tile_size`` itself ``None``).

    A ``"native_ratio"`` edge is a geometry reference
    ``resolution.accepted_references("geometry")`` accepts, ranked weaker than the persisted tier
    and stronger than a caller's explicit edge when a delivery mixes buckets across them (see
    :func:`~tcip_mcp.pipelines.resolution.reconcile_tile_size_validity`). The train-time resize is
    not returned; :func:`native_ratio_tile_resize` or :func:`resolve_tile_regime` gives it.

    Never raises. Returns ``(tile_size, tile_size_source, overlap, overlap_source)`` where
    ``tile_size`` is ``None`` only when its source is ``"unavailable"`` and each ``*_source`` is
    one of ``"explicit"``/``"derived"``/``"native_ratio"``/``"unavailable"``.
    """
    resolved_tile: int | None
    if tile_size is not None:
        resolved_tile, tile_source = int(tile_size), "explicit"
    elif getattr(predictor, "train_tile_size", None) is not None:
        resolved_tile, tile_source = int(predictor.train_tile_size), "derived"
    else:
        resolved_tile, tile_source = _native_ratio_tile_size(
            getattr(predictor, "train_native_size", None))

    if overlap is not None:
        resolved_overlap, overlap_source = float(overlap), "explicit"
    elif getattr(predictor, "train_overlap", None) is not None:
        resolved_overlap, overlap_source = float(predictor.train_overlap), "derived"
    else:
        resolved_overlap, overlap_source = DEFAULT_OVERLAP, "default"

    return resolved_tile, tile_source, resolved_overlap, overlap_source


class TileEdgeContradiction(ValueError):
    """A caller-stated tile edge that differs from the checkpoint's own recorded tile geometry."""


def _raise_tile_edge_contradiction(edge: int, recorded: int, kind: str) -> NoReturn:
    raise TileEdgeContradiction(
        f"stated tile_size {int(edge)} contradicts this checkpoint's own {kind} of "
        f"{recorded}. Pass tile_size {recorded} to match the checkpoint, or leave "
        "tile_size unset to derive it from the checkpoint."
    )


def _recorded_geometry_edge(predictor: Any) -> tuple[int | None, str]:
    """The checkpoint's own recorded tile edge a stated edge is checked against, and what kind of
    record it came from: the persisted training tile geometry when the checkpoint carries one, else
    the edge its recorded uniform untiled training frame yields, else ``None`` (the checkpoint
    records no tile geometry at all, the foreign-checkpoint case).
    """
    train_tile = getattr(predictor, "train_tile_size", None)
    if train_tile is not None:
        return int(train_tile), "persisted training tile geometry"
    native_edge, _native_tier = _native_ratio_tile_size(
        getattr(predictor, "train_native_size", None))
    if native_edge is not None:
        return native_edge, "recorded untiled training frame"
    return None, "no persisted geometry"


def explicit_edge_provenance(predictor: Any, edge: int) -> str:
    """The ``derived_from`` sentence for a stated tile edge that cleared
    :func:`resolve_tile_regime`'s contradiction check, from the facts the door holds: equal to the
    checkpoint's persisted training tile geometry, equal to the edge its recorded untiled training
    frame yields (run without that frame's own recorded resize, which the stated-edge path never
    applies), or stated on a checkpoint that records no tile geometry at all.

    Raises :class:`TileEdgeContradiction` if ``edge`` differs from a recorded geometry the
    checkpoint does carry.
    """
    recorded, kind = _recorded_geometry_edge(predictor)
    if recorded is None:
        return "stated on a checkpoint that records no tile geometry"
    if int(recorded) == int(edge):
        if kind == "persisted training tile geometry":
            return "equal to the checkpoint's persisted training tile geometry"
        return (
            "equal to the edge the checkpoint's recorded untiled training frame yields, run "
            "without that frame's own recorded resize"
        )
    _raise_tile_edge_contradiction(int(edge), recorded, kind)


def native_ratio_tile_resize(predictor: Any, tile_size_source: str) -> tuple[int, int] | None:
    """The ``(width, height)`` a native-size tile must be resized to before the forward pass, for a
    run whose tile edge came from the native-size ratio tier; ``None`` for every other tier.

    The resize is the one the run's recorded augmentation chain applied to a training frame
    (:func:`~tcip_mcp.pipelines.data.augmentations.recorded_resize`); a chain pinning no size
    returns ``None``. An augmentation config that cannot be built raises.
    """
    if tile_size_source != "native_ratio":
        return None
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    return recorded_resize(getattr(predictor, "train_augmentation", None))


def resolve_tile_regime(
    predictor: Any, *, tiled: bool, tile_size: int | None, overlap: float | None,
) -> tuple[int | None, str, float, str, tuple[int, int] | None]:
    """Resolve tile_size/overlap plus the resize a native-ratio edge must run each tile through,
    composing :func:`resolve_tile_geometry` and :func:`native_ratio_tile_resize`.

    When ``tiled`` and the caller states an edge, that edge is checked against the checkpoint's own
    recorded geometry (:func:`_recorded_geometry_edge`, the persisted training tile geometry, or,
    absent one, the edge its recorded uniform untiled frame yields) and
    :class:`TileEdgeContradiction` is raised, naming both edges and the recorded geometry's own
    kind, when they differ. A checkpoint recording neither has nothing to contradict. An untiled
    call with a stated edge is never checked.

    The resize is resolved only when ``tiled``.

    Returns ``(tile_size, tile_size_source, overlap, overlap_source, tile_resize)``.
    """
    if tiled and tile_size is not None:
        recorded, kind = _recorded_geometry_edge(predictor)
        if recorded is not None and int(recorded) != int(tile_size):
            _raise_tile_edge_contradiction(int(tile_size), recorded, kind)
    resolved_tile, tile_size_source, resolved_overlap, overlap_source = resolve_tile_geometry(
        predictor, tile_size=tile_size, overlap=overlap)
    tile_resize = native_ratio_tile_resize(predictor, tile_size_source) if tiled else None
    return resolved_tile, tile_size_source, resolved_overlap, overlap_source, tile_resize


def build_predictor(
    checkpoint: "VerifiedCheckpoint", *, kind: str | None = None, **kwargs: Any,
) -> "Predictor":
    """Construct the right predictor for a checkpoint's kind.

    ``checkpoint`` is a :class:`~tcip_mcp.model_registry.VerifiedCheckpoint`, the object
    :func:`~tcip_mcp.model_registry.load_registered_checkpoint` returns; this function reads no
    file itself. Pass ``kind`` to skip detection (e.g. the registry already recorded it); the
    payload it sniffs from is ``checkpoint.payload``. Predictor kwargs (``device``,
    ``score_threshold``, ``nms_iou``, ``max_dets``, ...) pass through.
    """
    if kind is None:
        kind = _kind_from_ckpt(checkpoint.payload, checkpoint.path)

    if kind == KIND_TCIP_MODULE:
        # A tcip checkpoint GenericPredictor rebuilds by re-importing the bespoke model_source
        # builder through build_model, never exec.
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        return GenericPredictor(checkpoint, **kwargs)
    raise ValueError(f"Unsupported model kind {kind!r} for {checkpoint.path}")

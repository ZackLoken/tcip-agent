"""Model-kind contract + the predictor factory.

Inference dispatches on a model kind so the platform can run more than one framework of
detector without special-casing at every call site. A tcip checkpoint (a bespoke model built
by an agent-written importable builder) is the only kind today; the dispatch stays
kind-aware, sniffed from the checkpoint, not hardcoded to one implementation, because
``.kind`` is a real field other code already reads generically (``require_composed_detector``,
the model registry), not scaffolding for a single implementation.

Kind travels three ways: stamped on tcip checkpoints at save time (a top-level ``kind`` key),
recorded on the registry entry, and, for a foreign ``.pt`` this platform never wrote,
sniffed from the checkpoint's top-level keys. An undeterminable kind raises rather than
guessing and running a wrong forward, which would silently corrupt the count (and the count
is the phenotype).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

logger = logging.getLogger(__name__)

# A bespoke, from-scratch model built by an agent-written importable builder (model_source).
# Reproduced by re-importing that builder, never by exec.
KIND_TCIP_MODULE = "tcip_module"
DEFAULT_KIND = KIND_TCIP_MODULE


@runtime_checkable
class Predictor(Protocol):
    """The inference surface every model kind exposes (today: ``GenericPredictor``; the shape
    stays open to a future foreign kind).

    Detection result dict:
    ``{image, width, height, boxes[[x1,y1,x2,y2] px], scores[], labels[], count}`` (tiled adds
    ``tiles``). For ``task == "instance_seg"``, ``predict``/``predict_batch`` (untiled) additionally
    carry ``masks``, one soft (unbinarized) ``[H, W]`` probability array per surviving detection,
    already in full-image coordinates, same order as ``boxes``/``scores``/``labels``.
    ``predict_tiled`` (whichever source kind, a path/``BandGroupRef`` or a windowed reader) also
    carries ``masks`` by default (``require_masks=True``), but in a different, tile-local shape:
    a list of ``{"mask_patch",
    "offset_x", "offset_y"}`` dicts (a small patch plus its full-image-space origin, never a dense
    full-image-sized array, since a tile source can be a raster too large to afford one), never
    interchangeable with the untiled shape above. A caller that never reads masks opts out
    deliberately with ``predict_tiled(..., require_masks=False)`` and gets ordinary boxes-only tiled
    inference with no ``masks`` key at all. ``predict_tiled(..., tile_resize=(w, h))`` additionally
    resizes each tile to ``(w, h)`` before the forward pass and maps every returned box (and mask)
    back into the tile's own native pixel space, so a run can reproduce the input geometry a
    checkpoint trained at; every coordinate in a result is in the source's real pixel space either
    way. Labels are 1-indexed foreground (background = 0), the
    torchvision convention the rest of the pipeline (JSON prediction export, Review, CSV) already
    assumes; a future kind that is natively 0-indexed would shift to it at its own boundary,
    so downstream code never has to know which kind produced a result.
    """

    task: str
    in_chans: int

    # Each image argument may be a plain path/string or a BandGroupRef (a band-grouped capture,
    # see pipelines.data.band_groups), the same image sources image_utils.list_logical_images/
    # resolve_image_source hand every other reader in this platform.
    def predict(self, image_path: str | Path | BandGroupRef) -> dict: ...
    def predict_batch(self, image_paths: list[str | Path | BandGroupRef], **kwargs: Any) -> list[dict]: ...
    def predict_tiled(self, source: str | Path | BandGroupRef, **kwargs: Any) -> dict: ...


def _kind_from_ckpt(ckpt: Any, checkpoint_path: str) -> str:
    """Resolve the kind from an already-loaded checkpoint object (no second disk read)."""
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Cannot determine model kind for {checkpoint_path}: checkpoint did not unpickle to a "
            f"dict (got {type(ckpt).__name__}), so it carries none of the structural markers "
            "('kind' / 'model_source'+'model_state_dict') this platform's own checkpoints do."
        )
    stamped = ckpt.get("kind")
    if stamped:
        return str(stamped)
    # Structural fallback for tcip checkpoints whose kind wasn't stamped.
    if "model_source" in ckpt and "model_state_dict" in ckpt:
        return KIND_TCIP_MODULE
    raise ValueError(
        f"Cannot determine model kind for {checkpoint_path}: no 'kind'/'model_source' (tcip). "
        f"Top-level keys: {sorted(ckpt)[:12]}"
    )


def detect_kind(checkpoint_path: str) -> str:
    """Return the model kind for a checkpoint on disk (loads it once to sniff)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _kind_from_ckpt(ckpt, checkpoint_path)


def _native_ratio_tile_size(train_native_size: Any) -> tuple[int | None, str]:
    """The tile edge a checkpoint's own uniform untiled training frame justifies, if any.

    A checkpoint trained untiled on frames that all shared one size (``train_native_size``, stamped
    ``[width, height]``) does justify one tile edge: a tile cut at that frame's own size presents an
    object to the model at the pixel scale a whole training frame did, which is the entire point of
    matching tile geometry to training geometry.

    That only holds for a square frame. Tile geometry is one edge everywhere it travels in this
    platform (``resolve_tile_geometry`` -> the operating point's ``tile_size`` -> the delivery gate
    -> the tiling core's own ``tile_size``/stride), and no single edge reproduces a ``W x H`` frame
    when ``W != H``: whichever edge is chosen, one axis presents objects at the wrong scale, and a
    detector's own internal resize is isotropic so it cannot undo the difference either. A
    rectangular frame therefore yields no edge here (``"unavailable"``) rather than an edge that
    silently mis-scales one axis; the caller states a ``tile_size`` explicitly, or runs untiled.
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
    :func:`_native_ratio_tile_size`) > no real basis at all (source ``"unavailable"``, ``tile_size``
    itself ``None``). One implementation both ``run_inference`` and the delivery-grade
    ``run_full_frame_evaluation`` call, so they can't silently disagree on which regime a model
    actually ran in.

    A ``"native_ratio"`` edge is a real basis to tile at all, never an independently validated one:
    it says what the model saw a frame at, not that tiling this raster at that edge was checked
    against anything. It stays outside ``resolution.accepted_references("geometry")``, so any gated
    door requires ``acknowledge_unvalidated=True`` to write a result that rests on it, and
    ``run_full_frame_evaluation`` refuses it outright. Reproducing the training input geometry also
    takes the recorded train-time resize, which this function does not return; the caller pairs it
    with :func:`native_ratio_tile_resize`.

    Pure fact-return, never raises: this is a capability, not a policy. Returns
    ``(tile_size, tile_size_source, overlap, overlap_source)`` where ``tile_size`` is ``None`` only
    when its source is ``"unavailable"`` (nothing to derive a scale from) and each ``*_source`` is
    one of ``"explicit"``/``"derived"``/``"native_ratio"``/``"unavailable"``. A caller with a stake
    in the source (e.g. a delivery gate that must not silently fabricate a number) inspects the
    returned source itself and decides whether to refuse, warn, or proceed, that policy does not
    belong here, since an exploratory caller (``run_inference``) and a certifying caller
    (``run_full_frame_evaluation``) legitimately make different calls on the same fact.
    """
    from tcip_mcp.pipelines.resolution import DEFAULT_OVERLAP

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


def native_ratio_tile_resize(predictor: Any, tile_size_source: str) -> tuple[int, int] | None:
    """The ``(width, height)`` a native-size tile must be resized to before the forward pass, for a
    run whose tile edge came from the native-size ratio tier; ``None`` for every other tier.

    The tier reproduces the input geometry the checkpoint trained at: a tile cut at the training
    frame's own size, then the same resize the run's recorded augmentation chain applied to a
    training frame (:func:`~tcip_mcp.pipelines.data.augmentations.recorded_resize`, which resolves a
    preset-name config through the builder rather than re-reading it). The recorded chain pinning no
    size (common) returns ``None`` and the tier reduces to tiling at the native size with no resize.

    Only for a ``"native_ratio"`` tile edge: an explicit or persisted-geometry edge feeds the model
    the tile as it stands, which is what those tiers mean and what every existing count was produced
    at. An augmentation config that cannot be built raises from the builder rather than being
    reported as "no resize", so the door about to tile refuses instead of running at a geometry it
    could not confirm.
    """
    if tile_size_source != "native_ratio":
        return None
    from tcip_mcp.pipelines.data.augmentations import recorded_resize

    return recorded_resize(getattr(predictor, "train_augmentation", None))


def build_predictor(checkpoint_path: str, *, kind: str | None = None, **kwargs: Any) -> "Predictor":
    """Construct the right predictor for a checkpoint's kind (the one inference entry point).

    Pass ``kind`` to skip detection (e.g. the registry already recorded it). Predictor
    kwargs (``device``, ``score_threshold``, ``nms_iou``, ``max_dets``, …) pass through.
    """
    ckpt = None
    if kind is None:
        # Sniff by loading once; hand the loaded checkpoint to the predictor so the weights aren't
        # read from disk twice. If the file can't be read (missing / corrupt / a test stub), fall
        # back to the default tcip kind, GenericPredictor then surfaces the real load error rather
        # than us masking it here.
        import torch

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            logger.debug("build_predictor: could not read %s to sniff kind; assuming %s",
                         checkpoint_path, DEFAULT_KIND, exc_info=True)
            kind = DEFAULT_KIND
        else:
            kind = _kind_from_ckpt(ckpt, checkpoint_path)

    if kind == KIND_TCIP_MODULE:
        # A tcip checkpoint GenericPredictor rebuilds by re-importing the bespoke model_source
        # builder through build_model, never exec.
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        extra = {"checkpoint": ckpt} if ckpt is not None else {}
        return GenericPredictor(checkpoint_path=checkpoint_path, **extra, **kwargs)
    raise ValueError(f"Unsupported model kind {kind!r} for {checkpoint_path}")

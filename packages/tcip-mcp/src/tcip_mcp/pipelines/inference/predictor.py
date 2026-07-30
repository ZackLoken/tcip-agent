"""Model-kind contract + the predictor factory.

Inference dispatches on a model KIND so the platform can run more than one framework of
detector without special-casing at every call site. A tcip checkpoint (a bespoke model built
by an agent-written importable builder) is the only kind today; the dispatch stays
kind-aware — sniffed from the checkpoint, not hardcoded to one implementation — because
``.kind`` is a real field other code already reads generically (``require_composed_detector``,
the model registry), not scaffolding for a single implementation.

Kind travels three ways: stamped on tcip checkpoints at save time (a top-level ``kind`` key),
recorded on the registry entry, and — for a foreign ``.pt`` this platform never wrote —
sniffed from the checkpoint's top-level keys. An undeterminable kind raises rather than
guessing and running a wrong forward, which would silently corrupt the count (and the count
is the phenotype).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# A bespoke, from-scratch model built by an agent-written importable builder (model_source).
# Reproduced by re-importing that builder — never by exec.
KIND_TCIP_MODULE = "tcip_module"
DEFAULT_KIND = KIND_TCIP_MODULE


@runtime_checkable
class Predictor(Protocol):
    """The inference surface every model kind exposes (today: ``GenericPredictor``; the shape
    stays open to a future foreign kind).

    Detection result dict:
    ``{image, width, height, boxes[[x1,y1,x2,y2] px], scores[], labels[], count}`` (tiled adds
    ``tiles``). For ``task == "instance_seg"``, ``predict``/``predict_batch`` (untiled) additionally
    carry ``masks`` — one soft (unbinarized) ``[H, W]`` probability array per surviving detection,
    same order as ``boxes``/``scores``/``labels``. ``predict_tiled`` does not yet thread masks
    through cross-tile reconstruction/merge and raises ``NotImplementedError`` for ``instance_seg``
    rather than silently dropping them; a caller that never reads masks opts out deliberately with
    ``predict_tiled(..., require_masks=False)`` and gets ordinary boxes-only tiled inference.
    Labels are **1-indexed foreground** (background = 0) — the
    torchvision convention the rest of the pipeline (JSON prediction export, Review, CSV) already
    assumes; a future kind that is natively 0-indexed would shift to it at its own boundary,
    so downstream code never has to know which kind produced a result.
    """

    task: str
    in_chans: int

    def predict(self, image_path: str) -> dict: ...
    def predict_batch(self, image_paths: list[str], **kwargs: Any) -> list[dict]: ...
    def predict_tiled(self, image_path: str, **kwargs: Any) -> dict: ...


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
    """Return the model KIND for a checkpoint on disk (loads it once to sniff)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _kind_from_ckpt(ckpt, checkpoint_path)


def resolve_tile_geometry(
    predictor: Any, *, tile_size: int | None, overlap: float | None,
) -> tuple[int, str, float, str]:
    """Resolve tile_size/overlap by precedence: explicit caller value > the checkpoint's own
    persisted training geometry (``predictor.train_tile_size``/``train_overlap``) > a documented
    default (K10 CV2/CV3 — one implementation both ``run_inference`` and the delivery-grade
    ``run_full_frame_evaluation`` call, so they can't silently disagree on which regime a model
    actually ran in).

    Pure fact-return, never raises: this is a capability, not a policy. Returns
    ``(tile_size, tile_size_source, overlap, overlap_source)`` where each ``*_source`` is one of
    ``"explicit"``/``"derived"``/``"default"``. A caller with a stake in the source (e.g. a
    delivery gate that must not silently fabricate a number) inspects the returned source itself
    and decides whether to refuse, warn, or proceed — that policy does not belong here, since an
    exploratory caller (``run_inference``) and a certifying caller (``run_full_frame_evaluation``)
    legitimately make different calls on the same fact.
    """
    from tcip_mcp.pipelines.resolution import DEFAULT_OVERLAP, DEFAULT_TILE_SIZE

    if tile_size is not None:
        resolved_tile, tile_source = int(tile_size), "explicit"
    elif getattr(predictor, "train_tile_size", None) is not None:
        resolved_tile, tile_source = int(predictor.train_tile_size), "derived"
    else:
        resolved_tile, tile_source = DEFAULT_TILE_SIZE, "default"

    if overlap is not None:
        resolved_overlap, overlap_source = float(overlap), "explicit"
    elif getattr(predictor, "train_overlap", None) is not None:
        resolved_overlap, overlap_source = float(predictor.train_overlap), "derived"
    else:
        resolved_overlap, overlap_source = DEFAULT_OVERLAP, "default"

    return resolved_tile, tile_source, resolved_overlap, overlap_source


def build_predictor(checkpoint_path: str, *, kind: str | None = None, **kwargs: Any) -> "Predictor":
    """Construct the right predictor for a checkpoint's kind (the one inference entry point).

    Pass ``kind`` to skip detection (e.g. the registry already recorded it). Predictor
    kwargs (``device``, ``score_threshold``, ``nms_iou``, ``max_dets``, …) pass through.
    """
    ckpt = None
    if kind is None:
        # Sniff by loading once; hand the loaded checkpoint to the predictor so the weights aren't
        # read from disk twice. If the file can't be read (missing / corrupt / a test stub), fall
        # back to the default tcip kind — GenericPredictor then surfaces the real load error rather
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
        # builder through build_model — never exec.
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        extra = {"checkpoint": ckpt} if ckpt is not None else {}
        return GenericPredictor(checkpoint_path=checkpoint_path, **extra, **kwargs)
    raise ValueError(f"Unsupported model kind {kind!r} for {checkpoint_path}")

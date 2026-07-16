"""Model-kind contract + the predictor factory.

Inference dispatches on a model KIND so the platform can run more than one framework of
detector without special-casing at every call site. A tcip-composed checkpoint (a
torchvision detector built from a ``model_spec``) is the default and historical kind; a
pretrained ultralytics YOLO checkpoint is a foreign artifact we sniff.

Kind travels three ways: stamped on tcip checkpoints at save time (a top-level ``kind`` key,
mirrored into ``model_spec``), recorded on the registry entry, and — for a foreign ``.pt``
we never wrote — sniffed from the checkpoint's top-level keys. An undeterminable kind raises
rather than guessing and running a wrong forward, which would silently corrupt the count
(and the count is the phenotype).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

KIND_TORCHVISION_COMPOSED = "torchvision_composed"
KIND_ULTRALYTICS = "ultralytics"
DEFAULT_KIND = KIND_TORCHVISION_COMPOSED


@runtime_checkable
class Predictor(Protocol):
    """The inference surface every model kind exposes (torchvision, ultralytics, future kinds).

    Detection result dict:
    ``{image, width, height, boxes[[x1,y1,x2,y2] px], scores[], labels[], count}`` (tiled adds
    ``tiles``). Labels are **1-indexed foreground** (background = 0) — the torchvision
    convention the rest of the pipeline (JSON prediction export, Review, CSV) already
    assumes; a kind that is natively 0-indexed (ultralytics) shifts to it at its own boundary,
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
        # ultralytics can pickle a bare nn.Module in some export paths.
        return KIND_ULTRALYTICS
    stamped = ckpt.get("kind") or (ckpt.get("model_spec") or {}).get("kind")
    if stamped:
        return str(stamped)
    if "model_spec" in ckpt and "model_state_dict" in ckpt:
        return KIND_TORCHVISION_COMPOSED
    if "model" in ckpt and ("train_args" in ckpt or "names" in ckpt or "nc" in ckpt):
        return KIND_ULTRALYTICS
    raise ValueError(
        f"Cannot determine model kind for {checkpoint_path}: no 'kind'/'model_spec' (tcip) and "
        f"no ultralytics markers ('model' + 'train_args'/'names'/'nc'). "
        f"Top-level keys: {sorted(ckpt)[:12]}"
    )


def detect_kind(checkpoint_path: str) -> str:
    """Return the model KIND for a checkpoint on disk (loads it once to sniff)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return _kind_from_ckpt(ckpt, checkpoint_path)


def build_predictor(checkpoint_path: str, *, kind: str | None = None, **kwargs: Any) -> "Predictor":
    """Construct the right predictor for a checkpoint's kind (the one inference entry point).

    Pass ``kind`` to skip detection (e.g. the registry already recorded it). Predictor
    kwargs (``device``, ``score_threshold``, ``nms_iou``, ``max_dets``, …) pass through.
    """
    if kind == KIND_ULTRALYTICS:
        from tcip_mcp.pipelines.inference.yolo_predictor import YoloPredictor
        return YoloPredictor(checkpoint_path, **kwargs)

    ckpt = None
    if kind is None:
        # Sniff by loading once; hand the loaded checkpoint to the composed predictor so the
        # weights aren't read from disk twice. If the file can't be read (missing / corrupt /
        # a test stub), fall back to the historical composed kind — GenericPredictor then
        # surfaces the real load error rather than us masking it here.
        import torch

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            logger.debug("build_predictor: could not read %s to sniff kind; assuming %s",
                         checkpoint_path, DEFAULT_KIND, exc_info=True)
            kind = DEFAULT_KIND
        else:
            kind = _kind_from_ckpt(ckpt, checkpoint_path)

    if kind == KIND_TORCHVISION_COMPOSED:
        from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
        extra = {"checkpoint": ckpt} if ckpt is not None else {}
        return GenericPredictor(checkpoint_path=checkpoint_path, **extra, **kwargs)
    if kind == KIND_ULTRALYTICS:
        from tcip_mcp.pipelines.inference.yolo_predictor import YoloPredictor
        return YoloPredictor(checkpoint_path, **kwargs)
    raise ValueError(f"Unsupported model kind {kind!r} for {checkpoint_path}")

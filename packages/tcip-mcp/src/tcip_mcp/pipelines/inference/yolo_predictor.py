"""First-class predictor for a pretrained ultralytics YOLO checkpoint.

Matches :class:`GenericPredictor`'s public surface (``predict`` / ``predict_batch`` /
``predict_tiled`` + the identical result dict) so every inference call site and
``result_to_yolo_lines`` work unchanged regardless of model kind. Tiled inference goes
through SAHI (native ultralytics support, cross-tile NMS/NMM merge, EXIF-upright), which is
also how the baseline was run offline — this makes that governed rather than a hand-script.

Two boundary conversions keep the rest of the pipeline kind-agnostic:
  * **Labels** — ultralytics is 0-indexed; the shared result dict is 1-indexed foreground
    (background = 0, the torchvision convention ``result_to_yolo_lines`` assumes). We add 1.
  * **Coordinates** — boxes come back as pixel xyxy in the EXIF-upright frame (we hand SAHI
    the already-oriented array + normalize by upright dims), so YOLO predictions land in the
    same frame the GT labels are authored in.
"""

from __future__ import annotations

import logging

from tcip_mcp.pipelines.inference.predictor import KIND_ULTRALYTICS

logger = logging.getLogger(__name__)

# SAHI merge across tile seams: NMS suppresses overlaps, NMM (Non-Max Merging) unions the
# fragments of one object split by a seam. Default to NMS for parity with the validated
# baseline; NMM is available per-call for objects that straddle tiles.
DEFAULT_POSTPROCESS = "NMS"
DEFAULT_MATCH_THRESHOLD = 0.5


def build_result(object_predictions, image_path: str, w: int, h: int,
                 max_dets: int | None = None, tiles: int | None = None) -> dict:
    """SAHI object predictions -> the shared detection result dict (module-level so the
    label/max_dets logic is unit-testable without loading a model).

    Labels shift 0-indexed ultralytics -> 1-indexed foreground (background = 0). A full-frame
    ``max_dets`` cap is applied AFTER any tiled merge, highest score first, so dense scenes
    aren't silently truncated per-tile at the library default.
    """
    preds = sorted(object_predictions, key=lambda p: float(p.score.value), reverse=True)
    if max_dets is not None:
        preds = preds[:max_dets]

    boxes, scores, labels = [], [], []
    for p in preds:
        bb = p.bbox
        boxes.append([float(bb.minx), float(bb.miny), float(bb.maxx), float(bb.maxy)])
        scores.append(float(p.score.value))
        labels.append(int(p.category.id) + 1)

    result = {
        "image": image_path, "width": w, "height": h,
        "boxes": boxes, "scores": scores, "labels": labels, "count": len(boxes),
    }
    if tiles is not None:
        result["tiles"] = tiles
    return result


def _training_imgsz(checkpoint_path: str, default: int = 640) -> int:
    """The size the YOLO model was trained at (from the checkpoint), so a tile tracks the
    model instead of a guessed constant. Falls back to ``default``."""
    import torch

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        imgsz = (ckpt.get("train_args") or {}).get("imgsz", default) if isinstance(ckpt, dict) else default
        return int(imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz)
    except Exception:
        return default


class YoloPredictor:
    """Load a pretrained ultralytics YOLO checkpoint and run (optionally tiled) inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        score_threshold: float = 0.5,
        nms_iou: float | None = None,
        max_dets: int | None = None,
    ) -> None:
        import torch
        from sahi import AutoDetectionModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_dets = max_dets
        self.kind = KIND_ULTRALYTICS
        self.task = "detection"
        self.in_chans = 3  # ultralytics YOLO is RGB
        self.checkpoint_path = checkpoint_path
        self.training_imgsz = _training_imgsz(checkpoint_path)

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=checkpoint_path,
            confidence_threshold=score_threshold,
            device=str(self.device),
        )
        # Make in-model NMS IoU + the detection cap part of the operating point when the caller
        # derives one, rather than leaving ultralytics' library defaults (iou=0.7, max_det=300)
        # silently in charge of which boxes survive. Without the max_det override every forward
        # truncates at 300 and the post-hoc build_result cap can only remove boxes, never restore
        # ones ultralytics already dropped — a silent undercount on a dense frame (the phenotype).
        try:
            if nms_iou is not None:
                self.model.model.overrides["iou"] = float(nms_iou)
            if max_dets is not None:
                self.model.model.overrides["max_det"] = int(max_dets)
        except Exception:
            logger.debug("YoloPredictor: could not set model iou/max_det overrides", exc_info=True)

        # The class map travels with the model so labels resolve to names for the deliverable
        # (a bare integer label is a measurement-integrity hazard downstream).
        self.class_map = self._read_class_map()

    def _read_class_map(self) -> dict[int, str]:
        for getter in (
            lambda: {int(k): str(v) for k, v in self.model.category_mapping.items()},
            lambda: {int(k): str(v) for k, v in self.model.model.names.items()},
        ):
            try:
                m = getter()
                if m:
                    return m
            except Exception:
                continue
        return {}

    # ── coordinate helpers ────────────────────────────────────────────────

    def _upright_array(self, image_path: str):
        """Return (upright RGB ndarray, upright width, upright height) — the GT's frame."""
        import numpy as np
        from PIL import Image

        from tcip_annotation.utils import auto_orient_image, get_image_dimensions

        w, h = get_image_dimensions(image_path)
        arr = np.asarray(auto_orient_image(Image.open(image_path)).convert("RGB"))
        return arr, w, h

    def _to_result(self, object_predictions, image_path: str, w: int, h: int,
                   tiles: int | None = None) -> dict:
        return build_result(object_predictions, image_path, w, h, self.max_dets, tiles)

    # ── prediction surface (mirrors GenericPredictor) ─────────────────────

    def predict(self, image_path: str) -> dict:
        """Full-image detection (no tiling)."""
        from sahi.predict import get_prediction

        arr, w, h = self._upright_array(image_path)
        result = get_prediction(image=arr, detection_model=self.model)
        return self._to_result(result.object_prediction_list, image_path, w, h)

    def predict_tiled(
        self, image_path: str, tile_size: int | None = None, overlap: float = 0.2,
        tile_batch_size: int = 96, global_nms_iou: float | None = None,
        postprocess_type: str = DEFAULT_POSTPROCESS, **_: object,
    ) -> dict:
        """Tiled (SAHI) detection: slice at ``tile_size`` -> per-tile predict -> cross-tile
        merge -> full-image detections. ``tile_size`` defaults to the model's training size."""
        from sahi.predict import get_sliced_prediction
        from sahi.slicing import get_slice_bboxes

        tile = int(tile_size or self.training_imgsz)
        arr, w, h = self._upright_array(image_path)
        match = global_nms_iou if global_nms_iou is not None else DEFAULT_MATCH_THRESHOLD
        result = get_sliced_prediction(
            image=arr,
            detection_model=self.model,
            slice_height=tile,
            slice_width=tile,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match,
            verbose=0,
        )
        try:
            n_tiles = len(get_slice_bboxes(
                image_height=h, image_width=w, slice_height=tile, slice_width=tile,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap))
        except Exception:
            n_tiles = None
        return self._to_result(result.object_prediction_list, image_path, w, h, tiles=n_tiles)

    def predict_batch(
        self, image_paths: list[str], tile: bool = False, tile_size: int | None = None,
        overlap: float = 0.2, tile_batch_size: int = 96, global_nms_iou: float | None = None,
        batch_size: int = 16, postprocess: str = "nms", **_: object,
    ) -> list[dict]:
        """Run inference over many images (optionally tiled). SAHI processes one image at a
        time, so this loops; the signature mirrors :class:`GenericPredictor` for the factory."""
        if tile:
            ptype = "NMM" if str(postprocess).lower() == "nmm" else "NMS"
            return [
                self.predict_tiled(p, tile_size=tile_size, overlap=overlap,
                                   tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
                                   postprocess_type=ptype)
                for p in image_paths
            ]
        return [self.predict(p) for p in image_paths]

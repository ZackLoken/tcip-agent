"""Generic predictor for any bespoke ``model_source`` checkpoint.

Auto-detects task from the saved ``model_source``. Supports single image,
batch, and ONNX export.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from tcip_mcp.pipelines.model_build import build_model
from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE
from tcip_mcp.pipelines.resolution import DEFAULT_NMS_IOU, DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)

# Detection task names that format outputs as boxes/scores/labels. A bespoke model_source declares
# the task type ``detection`` / ``instance_seg`` — both route through the detection formatter.
_DETECTION_TASKS = frozenset({"detection", "instance_seg"})


from tcip_mcp.pipelines.image_utils import crop_pad_tile as _crop_pad_tile  # noqa: E402


class GenericPredictor:
    """Load any bespoke ``model_source`` checkpoint and run inference.

    The checkpoint must contain 'model_source' and 'model_state_dict'.
    Task type is read from the model_source.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        score_threshold: float = 0.5,
        nms_iou: float | None = None,
        max_dets: int | None = None,
        *,
        checkpoint: dict | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_threshold = score_threshold
        self.max_dets = max_dets

        # ``build_predictor`` may hand us the already-loaded checkpoint (it read it to sniff the
        # kind) so the weights aren't read from disk twice; fall back to loading it ourselves.
        ckpt = checkpoint if checkpoint is not None else torch.load(
            checkpoint_path, map_location=self.device, weights_only=False)
        # A bespoke checkpoint carries the importable-builder ref; build_model re-imports it.
        self.model_source = ckpt.get("model_source")
        self.kind = KIND_TCIP_MODULE
        self.config = ckpt.get("config", {})

        # Training tile geometry, so inference can derive the tile scale from the checkpoint instead
        # of a mismatched default (CV2). None when this checkpoint carried no tiling geometry.
        _tiling = (self.config.get("data") or {}).get("tiling") or {}
        self.train_tile_size = _tiling.get("tile_size")
        self.train_overlap = _tiling.get("overlap")

        self.model = build_model(ckpt)  # re-imported bespoke builder (no exec)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Make the operating point govern which boxes exist (in-model thresholds), not just a
        # post-hoc filter that can never recover a box the model already discarded (the audit's
        # finding). No-op for non-detection models. See pipelines/operating_point.py.
        from tcip_mcp.pipelines.operating_point import set_detector_operating_point
        set_detector_operating_point(self.model, score_thresh=score_threshold,
                                     nms_thresh=nms_iou, detections_per_img=max_dets)

        # Task + input channels come from the bespoke model_source's declared ``task`` / ``in_chans``.
        src = self.model_source or {}
        self.task = src.get("task", "unknown")
        self.in_chans = int(src.get("in_chans", 3))

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """Run inference on a single image."""
        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        tensor = pil_to_tensor(img).to(self.device)

        if self.task in _DETECTION_TASKS:
            outputs = self.model([tensor])
            if isinstance(outputs, list):
                outputs = outputs[0]
            return self._format_detection(outputs, image_path, w, h)
        else:
            outputs = self.model(tensor.unsqueeze(0))
            return self._format_other(outputs, image_path, w, h)

    @torch.no_grad()
    def predict_batch(
        self, image_paths: list[str], tile: bool = False, tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = 0.2, tile_batch_size: int = 96, global_nms_iou: float = DEFAULT_NMS_IOU,
        batch_size: int = 16, postprocess: str = "nms",
    ) -> list[dict]:
        """Run inference on multiple images (optionally tiled for small objects).

        For detection, images are run through the detector in batches of ``batch_size``
        (one GPU forward per batch — torchvision detectors take a list of variable-size
        images), instead of one forward per image. Non-detection heads stay per-image
        since their inputs are native-resolution (can't be stacked without resizing).
        """
        if tile:
            return [
                self.predict_tiled(p, tile_size=tile_size, overlap=overlap,
                                   tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou,
                                   postprocess=postprocess)
                for p in image_paths
            ]
        if self.task in _DETECTION_TASKS:
            return self._predict_batch_detection(image_paths, batch_size)
        return [self.predict(p) for p in image_paths]

    @torch.no_grad()
    def _predict_batch_detection(self, image_paths: list[str], batch_size: int) -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(image_paths), max(1, batch_size)):
            chunk = image_paths[start:start + max(1, batch_size)]
            tensors, meta = [], []
            for p in chunk:
                img = load_image(p, self.in_chans)
                w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
                tensors.append(pil_to_tensor(img).to(self.device))
                meta.append((p, w, h))
            outputs = self.model(tensors)  # one forward over the whole chunk
            for (p, w, h), out in zip(meta, outputs):
                results.append(self._format_detection(out, p, w, h))
        return results

    @torch.no_grad()
    def predict_tiled(
        self, image_path: str, tile_size: int = DEFAULT_TILE_SIZE, overlap: float = 0.2,
        tile_batch_size: int = 96, global_nms_iou: float = DEFAULT_NMS_IOU, postprocess: str = "nms",
    ) -> dict:
        """Tiled (SAHI-style) detection: sliding-window tiles -> per-tile predict ->
        core-region reconstruction -> cross-tile merge -> full-image detections.

        ``postprocess`` selects the cross-tile merge: ``"nms"`` suppresses overlaps, ``"nmm"``
        unions boxes split across a seam. Falls back to :meth:`predict` for non-detection heads.
        """
        if self.task not in _DETECTION_TASKS:
            return self.predict(image_path)

        import numpy as np
        from tcip_mcp.pipelines.data.tiling import (
            compute_stride, tile_positions, reconstruct_core, global_nms, global_merge,
        )

        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        stride = compute_stride(tile_size, overlap)
        positions = tile_positions(h, w, tile_size, stride)

        min_size = ((self.model_source or {}).get("builder_kwargs") or {}).get("min_size")
        if min_size and abs(int(min_size) - tile_size) > tile_size:
            logger.warning("predict_tiled: model min_size=%s differs greatly from tile_size=%s "
                           "(tiles will be rescaled).", min_size, tile_size)

        per_tile_boxes, per_tile_scores, per_tile_labels, tile_info = [], [], [], []
        batch_tiles: list = []
        batch_meta: list = []

        def _flush() -> None:
            if not batch_tiles:
                return
            outputs = self.model(batch_tiles)
            for out, meta in zip(outputs, batch_meta):
                keep = out["scores"] >= self.score_threshold
                per_tile_boxes.append(out["boxes"][keep].cpu().numpy())
                per_tile_scores.append(out["scores"][keep].cpu().numpy())
                per_tile_labels.append(out["labels"][keep].cpu().numpy())
                tile_info.append(meta)
            batch_tiles.clear()
            batch_meta.clear()

        for tile_x, tile_y in positions:
            crop = _crop_pad_tile(img, tile_x, tile_y, tile_size, w, h)
            batch_tiles.append(pil_to_tensor(crop).to(self.device))
            batch_meta.append({"tile_x": tile_x, "tile_y": tile_y, "original_width": w, "original_height": h})
            if len(batch_tiles) >= tile_batch_size:
                _flush()
        _flush()

        boxes, scores, labels = reconstruct_core(
            per_tile_boxes, per_tile_scores, per_tile_labels, tile_info, tile_size, stride)
        if len(boxes) == 0:
            pass
        elif postprocess == "nmm":
            boxes, scores, labels = global_merge(boxes, scores, labels, global_nms_iou)
        else:
            keep = global_nms(boxes, scores, labels, global_nms_iou)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Enforce the full-frame detection cap after the cross-tile merge (highest score first);
        # the in-model detections_per_img only caps per tile, so a dense image can exceed it.
        if self.max_dets is not None and len(scores) > self.max_dets:
            top = np.argsort(scores)[::-1][: self.max_dets]
            boxes, scores, labels = boxes[top], scores[top], labels[top]

        return {
            "image": image_path,
            "width": w,
            "height": h,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "count": int(len(boxes)),
            "tiles": len(positions),
        }

    def export_onnx(self, output_path: str, opset: int = 17) -> str:
        """Export model to ONNX format with dynamic batch size."""
        dummy = torch.randn(1, 3, 640, 640).to(self.device)
        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy,
            output_path,
            opset_version=opset,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
        )
        logger.info("ONNX model exported to %s", output_path)
        return output_path

    def _format_detection(self, outputs: dict, image_path: str, w: int, h: int) -> dict:
        keep = outputs["scores"] >= self.score_threshold
        return {
            "image": image_path,
            "width": w,
            "height": h,
            "boxes": outputs["boxes"][keep].cpu().tolist(),
            "scores": outputs["scores"][keep].cpu().tolist(),
            "labels": outputs["labels"][keep].cpu().tolist(),
            "count": int(keep.sum()),
        }

    def _format_other(self, outputs: dict, image_path: str, w: int, h: int) -> dict:
        result: dict = {"image": image_path, "width": w, "height": h}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu().tolist()
                else:
                    result[k] = v
        elif isinstance(outputs, torch.Tensor):
            result["output"] = outputs.cpu().tolist()
        return result

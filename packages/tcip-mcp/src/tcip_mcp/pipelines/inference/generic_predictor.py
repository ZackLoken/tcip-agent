"""Generic predictor for any ComposedModel checkpoint.

Auto-detects task from saved model_spec. Supports single image,
batch, and ONNX export.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from tcip_mcp.pipelines.composer import compose_model
from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

logger = logging.getLogger(__name__)


class GenericPredictor:
    """Load any ComposedModel checkpoint and run inference.

    The checkpoint must contain 'model_spec' and 'model_state_dict'.
    Task type is inferred from model_spec heads.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_threshold = score_threshold

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_spec = ckpt["model_spec"]
        self.config = ckpt.get("config", {})

        self.model = compose_model(self.model_spec)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Infer task from first head
        heads = self.model_spec.get("heads", [])
        self.task = heads[0]["name"] if heads else "unknown"

        # Input channels the model expects (matches the backbone's in_chans).
        bb = self.model_spec.get("backbone", {})
        self.in_chans = bb.get("in_chans", 3) if isinstance(bb, dict) else 3

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """Run inference on a single image."""
        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        tensor = pil_to_tensor(img).to(self.device)

        if self.task in ("anchor_detection", "anchor_free_detection"):
            outputs = self.model([tensor])
            if isinstance(outputs, list):
                outputs = outputs[0]
            return self._format_detection(outputs, image_path, w, h)
        else:
            outputs = self.model(tensor.unsqueeze(0))
            return self._format_other(outputs, image_path, w, h)

    @torch.no_grad()
    def predict_batch(
        self, image_paths: list[str], tile: bool = False, tile_size: int = 224,
        overlap: float = 0.2, tile_batch_size: int = 96, global_nms_iou: float = 0.3,
        batch_size: int = 16,
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
                                   tile_batch_size=tile_batch_size, global_nms_iou=global_nms_iou)
                for p in image_paths
            ]
        if self.task in ("anchor_detection", "anchor_free_detection"):
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
        self, image_path: str, tile_size: int = 224, overlap: float = 0.2,
        tile_batch_size: int = 96, global_nms_iou: float = 0.3,
    ) -> dict:
        """Tiled (SAHI-style) detection: sliding-window tiles -> per-tile predict ->
        core-region reconstruction -> single global NMS -> full-image detections.

        Falls back to :meth:`predict` for non-detection heads.
        """
        if self.task not in ("anchor_detection", "anchor_free_detection"):
            return self.predict(image_path)

        import numpy as np
        from tcip_mcp.pipelines.data.tiling import (
            compute_stride, tile_positions, reconstruct_core, global_nms,
        )

        img = load_image(image_path, self.in_chans)
        w, h = img.size if isinstance(img, Image.Image) else (img.shape[1], img.shape[0])
        stride = compute_stride(tile_size, overlap)
        positions = tile_positions(h, w, tile_size, stride)

        min_size = self.model_spec.get("heads", [{}])[0].get("min_size")
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
            crop = img.crop((tile_x, tile_y, min(tile_x + tile_size, w), min(tile_y + tile_size, h)))
            if crop.size != (tile_size, tile_size):
                padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                padded.paste(crop, (0, 0))  # zero-pad bottom/right
                crop = padded
            batch_tiles.append(pil_to_tensor(crop).to(self.device))
            batch_meta.append({"tile_x": tile_x, "tile_y": tile_y, "original_width": w, "original_height": h})
            if len(batch_tiles) >= tile_batch_size:
                _flush()
        _flush()

        boxes, scores, labels = reconstruct_core(
            per_tile_boxes, per_tile_scores, per_tile_labels, tile_info, tile_size, stride)
        keep = global_nms(boxes, scores, labels, global_nms_iou) if len(boxes) else np.zeros((0,), dtype=np.int64)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

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

"""Model inference — load a checkpoint and run predictions on images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from tcip_mcp.pipelines.models.builder import build_model
from tcip_mcp.pipelines.data.classes import ClassMap

logger = logging.getLogger(__name__)


class Predictor:
    """Loads a trained detection model and runs inference on images."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        score_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.score_threshold = score_threshold

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = ckpt["config"]
        self.model = build_model(config["model"])
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.config = config

        # Load class map from checkpoint if available
        self.class_map: ClassMap | None = None
        if "class_map" in ckpt:
            self.class_map = ClassMap.from_dict(ckpt["class_map"])
        elif "class_map" in config:
            self.class_map = ClassMap.from_dict(config["class_map"])

    @torch.no_grad()
    def predict_image(self, image_path: str) -> dict:
        """Run detection on a single image.

        Returns:
            Dict with 'boxes', 'scores', 'labels' (int), 'class_names' (str, if available),
            and image metadata.
        """
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        tensor = self._to_tensor(img).to(self.device)
        outputs = self.model([tensor])[0]

        # Filter by score
        keep = outputs["scores"] >= self.score_threshold
        boxes = outputs["boxes"][keep].cpu()
        scores = outputs["scores"][keep].cpu()
        labels = outputs["labels"][keep].cpu()

        result = {
            "image": image_path,
            "width": w,
            "height": h,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "count": int(keep.sum()),
        }

        # Add class names if class map is available
        if self.class_map is not None:
            names = []
            for lbl in labels.tolist():
                yolo_id = lbl - 1  # torchvision 1-indexed → YOLO 0-indexed
                if 0 <= yolo_id < self.class_map.num_classes:
                    names.append(self.class_map.name(yolo_id))
                else:
                    names.append(f"unknown_{lbl}")
            result["class_names"] = names

        return result

    @torch.no_grad()
    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        """Run detection on a batch of images."""
        return [self.predict_image(p) for p in image_paths]

    @torch.no_grad()
    def predict_tta(
        self,
        image_path: str,
        transforms: Sequence[str] = ("original", "hflip", "vflip"),
        nms_iou: float = 0.5,
    ) -> dict:
        """Run test-time augmentation (TTA) on an image.

        Applies multiple geometric transforms, runs inference on each, merges
        with NMS.  Transforms available: 'original', 'hflip', 'vflip',
        'rotate90', 'rotate180', 'rotate270'.

        Args:
            image_path: Path to the image.
            transforms: Sequence of transform names to apply.
            nms_iou: IoU threshold for NMS on merged boxes.

        Returns:
            Dict with merged boxes, scores, labels, and count.
        """
        from torchvision.ops import nms

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for tname in transforms:
            aug_img = self._apply_tta_transform(img, tname)
            aw, ah = aug_img.size
            tensor = self._to_tensor(aug_img).to(self.device)
            out = self.model([tensor])[0]
            keep = out["scores"] >= self.score_threshold
            boxes = out["boxes"][keep].cpu().float()
            scores = out["scores"][keep].cpu()
            labels = out["labels"][keep].cpu()

            # Map boxes back to original image geometry
            boxes = self._reverse_tta_boxes(boxes, tname, w, h, aw, ah)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not all_boxes or all(b.numel() == 0 for b in all_boxes):
            return {
                "image": image_path, "width": w, "height": h,
                "boxes": [], "scores": [], "labels": [], "count": 0,
                "tta_transforms": list(transforms),
            }

        merged_boxes = torch.cat(all_boxes)
        merged_scores = torch.cat(all_scores)
        merged_labels = torch.cat(all_labels)

        # Per-class NMS
        final_keep: list[int] = []
        for cid in merged_labels.unique():
            mask = merged_labels == cid
            idx = mask.nonzero(as_tuple=True)[0]
            kept = nms(merged_boxes[idx], merged_scores[idx], nms_iou)
            final_keep.extend(idx[kept].tolist())

        final_keep_t = torch.tensor(final_keep, dtype=torch.long)
        result = {
            "image": image_path,
            "width": w,
            "height": h,
            "boxes": merged_boxes[final_keep_t].tolist(),
            "scores": merged_scores[final_keep_t].tolist(),
            "labels": merged_labels[final_keep_t].tolist(),
            "count": len(final_keep),
            "tta_transforms": list(transforms),
        }

        if self.class_map is not None:
            names = []
            for lbl in result["labels"]:
                yolo_id = lbl - 1
                if 0 <= yolo_id < self.class_map.num_classes:
                    names.append(self.class_map.name(yolo_id))
                else:
                    names.append(f"unknown_{lbl}")
            result["class_names"] = names

        return result

    @staticmethod
    def _apply_tta_transform(img: Image.Image, name: str) -> Image.Image:
        if name == "original":
            return img
        if name == "hflip":
            return TF.hflip(img)
        if name == "vflip":
            return TF.vflip(img)
        if name == "rotate90":
            return img.transpose(Image.Transpose.ROTATE_90)
        if name == "rotate180":
            return img.transpose(Image.Transpose.ROTATE_180)
        if name == "rotate270":
            return img.transpose(Image.Transpose.ROTATE_270)
        raise ValueError(f"Unknown TTA transform: {name}")

    @staticmethod
    def _reverse_tta_boxes(
        boxes: torch.Tensor, name: str, orig_w: int, orig_h: int,
        aug_w: int, aug_h: int,
    ) -> torch.Tensor:
        """Map boxes from augmented coords back to original image coords."""
        if boxes.numel() == 0:
            return boxes
        b = boxes.clone()
        if name == "original":
            return b
        if name == "hflip":
            b[:, [0, 2]] = aug_w - boxes[:, [2, 0]]
            return b
        if name == "vflip":
            b[:, [1, 3]] = aug_h - boxes[:, [3, 1]]
            return b
        if name == "rotate90":
            # (x,y) in rot90 → (y, W-x) in original
            return torch.stack([
                boxes[:, 1], orig_w - boxes[:, 2],
                boxes[:, 3], orig_w - boxes[:, 0],
            ], dim=1)
        if name == "rotate180":
            return torch.stack([
                aug_w - boxes[:, 2], aug_h - boxes[:, 3],
                aug_w - boxes[:, 0], aug_h - boxes[:, 1],
            ], dim=1)
        if name == "rotate270":
            return torch.stack([
                orig_h - boxes[:, 3], boxes[:, 0],
                orig_h - boxes[:, 1], boxes[:, 2],
            ], dim=1)
        return b

    def export_yolo(self, result: dict, output_path: str) -> None:
        """Write prediction results as a YOLO-format text file.

        Format: class_id confidence cx cy w h (normalized)
        """
        w, h = result["width"], result["height"]
        lines = []
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # torchvision 1-indexed → YOLO 0-indexed
            cid = label - 1
            lines.append(f"{cid} {score:.6f} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def export_onnx(
        self,
        output_path: str,
        input_size: tuple[int, int] = (800, 800),
        opset_version: int = 17,
        dynamic_axes: bool = True,
    ) -> str:
        """Export model to ONNX format for portable deployment.

        Args:
            output_path: Path for the .onnx file.
            input_size: (H, W) of the dummy input.
            opset_version: ONNX opset version. Default 17.
            dynamic_axes: Whether to allow dynamic batch/spatial dims.

        Returns:
            Path to the exported ONNX file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)

        axes = None
        if dynamic_axes:
            axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
            }

        torch.onnx.export(
            self.model,
            (dummy_input,),
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["boxes", "labels", "scores"],
            dynamic_axes=axes,
        )
        logger.info("Exported ONNX model to %s", output_path)
        return output_path

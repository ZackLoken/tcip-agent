"""Generic predictor for any ComposedModel checkpoint.

Auto-detects task from saved model_spec. Supports single image,
batch, and ONNX export.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tcip_mcp.pipelines.composer import compose_model

logger = logging.getLogger(__name__)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


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

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """Run inference on a single image."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        tensor = _pil_to_tensor(img).to(self.device)

        if self.task in ("anchor_detection", "anchor_free_detection"):
            outputs = self.model([tensor])
            if isinstance(outputs, list):
                outputs = outputs[0]
            return self._format_detection(outputs, image_path, w, h)
        else:
            outputs = self.model(tensor.unsqueeze(0))
            return self._format_other(outputs, image_path, w, h)

    @torch.no_grad()
    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        """Run inference on multiple images."""
        return [self.predict(p) for p in image_paths]

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

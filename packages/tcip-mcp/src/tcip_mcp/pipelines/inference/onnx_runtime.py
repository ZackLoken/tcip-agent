"""ONNX Runtime inference for edge deployment.

Same predict() interface as GenericPredictor but uses ONNX Runtime
instead of PyTorch, for fast CPU/GPU inference on deployed models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OnnxPredictor:
    """Run inference on an ONNX-exported model via ONNX Runtime."""

    def __init__(self, onnx_path: str, score_threshold: float = 0.5) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime-gpu")

        self.score_threshold = score_threshold
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info("ONNX model loaded from %s (providers=%s)", onnx_path, self.session.get_providers())

    def predict(self, image_path: str) -> dict:
        """Run inference on a single image."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))  # HWC → CHW
        tensor = np.expand_dims(tensor, 0)  # add batch dim

        outputs = self.session.run(self.output_names, {self.input_name: tensor})

        return {
            "image": image_path,
            "width": w,
            "height": h,
            "outputs": {name: out.tolist() for name, out in zip(self.output_names, outputs)},
        }

    def predict_batch(self, image_paths: list[str]) -> list[dict]:
        return [self.predict(p) for p in image_paths]

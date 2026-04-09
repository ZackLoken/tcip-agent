"""PyTorch Dataset for YOLO-format object detection data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    """A dataset that loads YOLO-format detection labels.

    Each item returns (image_tensor, target_dict) matching torchvision's
    FasterRCNN / RetinaNet expected format:
        target = {
            "boxes": FloatTensor[N, 4] in xyxy pixel coords,
            "labels": Int64Tensor[N],
            "image_id": int,
        }
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        if stems is not None:
            self.stems = stems
        else:
            self.stems = sorted(
                f.stem
                for f in self.images_dir.iterdir()
                if f.suffix.lower() in image_exts
            )

    def __len__(self) -> int:
        return len(self.stems)

    def _find_image(self, stem: str) -> Path:
        for ext in (".jpg", ".JPG", ".jpeg", ".png", ".tif"):
            p = self.images_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for stem: {stem}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img_path = self._find_image(stem)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Parse YOLO label
        label_path = self.labels_dir / f"{stem}.txt"
        boxes = []
        labels = []
        if label_path.is_file():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cid + 1)  # YOLO 0-indexed → torchvision 1-indexed (0 = background)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = _pil_to_tensor(img)

        return img, target


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to float tensor [C, H, W] in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_split_stems(split_path: str) -> list[str]:
    """Load image stems from a split JSON file."""
    with open(split_path) as f:
        return json.load(f)

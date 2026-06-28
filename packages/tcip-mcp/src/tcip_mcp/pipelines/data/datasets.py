"""Multi-task datasets with standardized interfaces.

Each dataset type returns (image_tensor, target_dict) where the target
format is task-specific but always dict-based. A factory function
`build_dataset` dispatches to the correct class by task type.
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tcip_mcp.pipelines.image_utils import pil_to_tensor

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_image(images_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".tif"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for stem: {stem}")


def _read_det_boxes_px(label_path, w: int, h: int) -> tuple[list[list[float]], list[int]]:
    """YOLO ``cls cx cy w h`` -> pixel xyxy boxes + 1-indexed labels (cid+1, background 0)."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    lp = Path(label_path)
    if lp.is_file():
        for line in lp.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            labels.append(cid + 1)  # 0-indexed -> 1-indexed (background = 0)
    return boxes, labels


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def num_samples(self) -> int: ...

    @property
    def class_distribution(self) -> dict[int, int]:
        """Class ID → count. Subclasses should override for efficiency."""
        return {}

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]: ...

    def __len__(self) -> int:
        return self.num_samples


# ====================================================================
# Detection
# ====================================================================

class DetectionDataset(BaseDataset):
    """YOLO-format detection: ``cls cx cy w h`` per line."""

    task_type = "detection"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        if stems is not None:
            self.stems = stems
        else:
            self.stems = sorted(
                f.stem for f in self.images_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for stem in self.stems:
            lp = self.labels_dir / f"{stem}.txt"
            if lp.is_file():
                for line in lp.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        counts[int(parts[0])] += 1
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = Image.open(_find_image(self.images_dir, stem)).convert("RGB")
        w, h = img.size
        boxes, labels = _read_det_boxes_px(self.labels_dir / f"{stem}.txt", w, h)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": idx,
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)
        return img, target


# ====================================================================
# Tiled Detection (SAHI-style sliding window)
# ====================================================================

class TiledDetectionDataset(BaseDataset):
    """Wrap a ``DetectionDataset`` and expand each source image into native-resolution
    tiles with labels clipped/remapped to tile space (W3).

    Tile membership is computed at ``__init__`` (header-only image sizes + the YOLO
    txt) so the dataset can return one sample per tile index. ``__getitem__`` decodes
    the source image once, crops the tile, zero-pads border crops to ``tile_size``,
    and emits the same target dict shape as ``DetectionDataset``.
    """

    task_type = "detection"

    def __init__(
        self,
        base: "DetectionDataset",
        tile_size: int = 224,
        overlap: float = 0.2,
        min_edge_keep_frac: float = 0.35,
        dedup_iou: float = 0.8,
        skip_empty: bool = True,
        transforms: Any = None,
    ) -> None:
        from tcip_mcp.pipelines.data.tiling import (
            compute_stride, tile_positions, clip_boxes_to_tile, dedup_boxes,
        )

        self.base = base
        self.tile_size = tile_size
        self.overlap = overlap
        self.transforms = transforms
        self.stride = compute_stride(tile_size, overlap)
        self._index: list[dict] = []

        for stem in base.stems:
            img_path = _find_image(base.images_dir, stem)
            with Image.open(img_path) as im:
                w, h = im.size  # header-only, no pixel decode
            full_boxes, full_labels = _read_det_boxes_px(base.labels_dir / f"{stem}.txt", w, h)
            fb = np.asarray(full_boxes, dtype=np.float32).reshape(-1, 4)
            fl = np.asarray(full_labels, dtype=np.int64)
            for tile_x, tile_y in tile_positions(h, w, tile_size, self.stride):
                tb, tl = clip_boxes_to_tile(fb, fl, tile_x, tile_y, tile_size, min_edge_keep_frac)
                if len(tb) > 1:
                    tb, tl = dedup_boxes(tb, tl, dedup_iou)
                if skip_empty and len(tb) == 0:
                    continue
                self._index.append({"stem": stem, "tile_x": tile_x, "tile_y": tile_y, "boxes": tb, "labels": tl})

    @property
    def num_classes(self) -> int:
        return self.base.num_classes

    @property
    def num_samples(self) -> int:
        return len(self._index)

    @property
    def stems(self) -> list[str]:
        return [e["stem"] for e in self._index]

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for e in self._index:
            for lab in e["labels"].tolist():
                counts[int(lab) - 1] += 1  # 0-indexed cid, matching DetectionDataset
        return dict(counts)

    def _crop_tile(self, img: "Image.Image", tile_x: int, tile_y: int) -> "Image.Image":
        w, h = img.size
        crop = img.crop((tile_x, tile_y, min(tile_x + self.tile_size, w), min(tile_y + self.tile_size, h)))
        if crop.size != (self.tile_size, self.tile_size):
            padded = Image.new("RGB", (self.tile_size, self.tile_size), (0, 0, 0))
            padded.paste(crop, (0, 0))  # zero-pad bottom/right
            crop = padded
        return crop

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        e = self._index[idx]
        img = Image.open(_find_image(self.base.images_dir, e["stem"])).convert("RGB")
        tile = self._crop_tile(img, e["tile_x"], e["tile_y"])
        target = {
            "boxes": torch.tensor(e["boxes"], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(e["labels"], dtype=torch.int64),
            "image_id": idx,
        }
        if self.transforms is not None:
            tile, target = self.transforms(tile, target)
        else:
            tile = pil_to_tensor(tile)
        return tile, target


# ====================================================================
# Instance Segmentation
# ====================================================================

class InstanceSegDataset(BaseDataset):
    """YOLO polygon format: ``cls x1 y1 x2 y2 ...`` per line."""

    task_type = "instance_seg"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.stems = stems or sorted(
            f.stem for f in self.images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = Image.open(_find_image(self.images_dir, stem)).convert("RGB")
        w, h = img.size

        boxes, labels, masks = [], [], []
        lp = self.labels_dir / f"{stem}.txt"
        if lp.is_file():
            for line in lp.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                xs = [coords[i] * w for i in range(0, len(coords), 2)]
                ys = [coords[i] * h for i in range(1, len(coords), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                boxes.append([x1, y1, x2, y2])
                labels.append(cid + 1)

                # Rasterize polygon to binary mask
                mask = np.zeros((h, w), dtype=np.uint8)
                try:
                    from PIL import ImageDraw
                    poly_img = Image.new("L", (w, h), 0)
                    draw = ImageDraw.Draw(poly_img)
                    poly_pts = list(zip(xs, ys))
                    draw.polygon(poly_pts, fill=1)
                    mask = np.array(poly_img)
                except Exception:
                    pass
                masks.append(mask)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks) if masks else np.zeros((0, h, w)), dtype=torch.uint8),
            "image_id": idx,
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)
        return img, target


# ====================================================================
# Semantic Segmentation
# ====================================================================

class SemanticSegDataset(BaseDataset):
    """PNG mask images where pixel values are class IDs."""

    task_type = "semantic_seg"

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 2,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.stems = stems or sorted(
            f.stem for f in self.images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = Image.open(_find_image(self.images_dir, stem)).convert("RGB")
        mask_path = self.masks_dir / f"{stem}.png"
        mask = np.array(Image.open(mask_path).convert("L")) if mask_path.exists() else np.zeros(img.size[::-1], dtype=np.int64)
        # Key matches the SemanticSegHead loss contract.
        target = {"masks": torch.tensor(mask, dtype=torch.int64)}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)
        return img, target


# ====================================================================
# Classification
# ====================================================================

class ClassificationDataset(BaseDataset):
    """Image classification from CSV (image_stem, label) or folder structure."""

    task_type = "classification"

    def __init__(
        self,
        images_dir: str,
        csv_path: str | None = None,
        stems: list[str] | None = None,
        labels: list[int] | None = None,
        transforms: Any = None,
        num_classes: int = 2,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        if csv_path is not None:
            self._stems, self._labels = self._load_csv(csv_path)
        elif stems is not None and labels is not None:
            self._stems = stems
            self._labels = labels
        else:
            # Folder-based: images_dir/<class_name>/<image>
            self._stems, self._labels = self._load_folder_structure()

    def _load_csv(self, path: str) -> tuple[list[str], list[int]]:
        stems, labels = [], []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    stems.append(row[0].strip())
                    labels.append(int(row[1].strip()))
        return stems, labels

    def _load_folder_structure(self) -> tuple[list[str], list[int]]:
        stems, labels = [], []
        class_dirs = sorted(d for d in self.images_dir.iterdir() if d.is_dir())
        for cid, cdir in enumerate(class_dirs):
            for f in cdir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    stems.append(str(f))
                    labels.append(cid)
        return stems, labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._labels))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        if Path(stem).is_absolute() or Path(stem).exists():
            img = Image.open(stem).convert("RGB")
        else:
            img = Image.open(_find_image(self.images_dir, stem)).convert("RGB")
        target = {"labels": self._labels[idx]}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)
        return img, target


# ====================================================================
# Ordinal
# ====================================================================

class OrdinalDataset(BaseDataset):
    """Ordinal regression from CSV (image_stem, rank). E.g., disease severity 0-4."""

    task_type = "ordinal"

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transforms: Any = None,
        num_ranks: int = 5,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._num_ranks = num_ranks
        self._stems: list[str] = []
        self._ranks: list[int] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self._stems.append(row[0].strip())
                    self._ranks.append(int(row[1].strip()))

    @property
    def num_classes(self) -> int:
        return self._num_ranks

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._ranks))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = Image.open(_find_image(self.images_dir, stem)).convert("RGB")
        # Key matches the OrdinalHead loss contract (plural, like "labels"/"masks").
        target = {"ranks": self._ranks[idx], "num_ranks": self._num_ranks}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)
        return img, target


# ====================================================================
# Regression
# ====================================================================

class RegressionDataset(BaseDataset):
    """Continuous-value regression from CSV (image_stem, value)."""

    task_type = "regression"

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transforms: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._stems: list[str] = []
        self._values: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self._stems.append(row[0].strip())
                    self._values.append(float(row[1].strip()))

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = Image.open(_find_image(self.images_dir, stem)).convert("RGB")
        # Key matches the RegressionHead loss contract.
        target = {"values": self._values[idx]}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)
        return img, target


# ====================================================================
# Factory
# ====================================================================

_DATASET_MAP = {
    "detection": DetectionDataset,
    "instance_seg": InstanceSegDataset,
    "semantic_seg": SemanticSegDataset,
    "classification": ClassificationDataset,
    "ordinal": OrdinalDataset,
    "regression": RegressionDataset,
}


def build_dataset(task: str, **kwargs) -> BaseDataset:
    """Factory: build a dataset by task type.

    An optional ``tiling`` dict (``{enabled, tile_size, overlap, min_edge_keep_frac,
    dedup_iou, skip_empty}``) wraps the detection dataset in a
    :class:`TiledDetectionDataset` (W3). Ignored for non-detection tasks.
    """
    tiling = kwargs.pop("tiling", None)
    cls = _DATASET_MAP.get(task)
    if cls is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_DATASET_MAP.keys())}")

    if tiling and tiling.get("enabled", True):
        if task != "detection":
            logger.warning("tiling is only supported for task='detection'; ignoring for task=%r", task)
        else:
            transforms = kwargs.pop("transforms", None)
            base = cls(**kwargs)
            tile_kwargs = {k: tiling[k] for k in
                           ("tile_size", "overlap", "min_edge_keep_frac", "dedup_iou", "skip_empty")
                           if k in tiling}
            return TiledDetectionDataset(base, transforms=transforms, **tile_kwargs)

    return cls(**kwargs)

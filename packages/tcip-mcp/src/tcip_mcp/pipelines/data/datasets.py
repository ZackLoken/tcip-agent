"""Multi-task datasets with standardized interfaces.

Each dataset type returns (image_tensor, target_dict) where the target
format is task-specific but always dict-based. A factory function
`build_dataset` dispatches to the correct class by task type.
"""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tcip_annotation.utils import get_image_dimensions

from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_image(images_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".tif"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for stem: {stem}")


def _read_det_boxes_format(stem, labels_dir, fmt, coco, w, h, file_name):
    """Pixel-xyxy boxes + 1-indexed labels for ``stem``.

    The canonical on-disk format is per-image ``json`` (json_io); ``coco`` is the assembled
    dataset view of that JSON used for training; ``voc`` / ``labelme`` are external formats parsed
    via tcip_annotation.format_io (all pixel-space). YOLO is not a dataset read format — it is
    import-only, converted to JSON at ingest.
    """
    fmt = (fmt or "json").lower()
    if fmt == "json":
        from tcip_annotation import json_io
        bboxes = json_io.read_detect(str(labels_dir / f"{stem}.json"))[0]
    else:
        from tcip_annotation import format_io
        bboxes = []
        if fmt == "voc":
            p = labels_dir / f"{stem}.xml"
            if p.is_file():
                bboxes = format_io.parse_voc_detect(str(p))[0]
        elif fmt == "labelme":
            p = labels_dir / f"{stem}.json"
            if p.is_file():
                bboxes = format_io.parse_labelme_detect(str(p))[0]
        elif fmt == "coco":
            if coco is not None:
                bboxes = format_io.parse_coco_detect(coco, file_name=file_name)[0]
        else:
            raise ValueError(
                f"Unknown detection label_format {fmt!r} (use json/coco/voc/labelme); "
                "YOLO .txt is import-only — convert it to JSON first "
                "(scripts/migrate_labels_to_json.py)")
    boxes = [[b.x1, b.y1, b.x2, b.y2] for b in bboxes]
    labels = [b.class_id + 1 for b in bboxes]  # 0-indexed cid -> 1-indexed (background 0)
    return boxes, labels


def dir_label_format(labels_dir) -> str | None:
    """Best-effort on-disk format of a per-image label dir: ``"json"`` (canonical json_io
    schema), ``"yolo"``, or ``None`` (empty / unrecognized). Used to route a JSON label store
    onto the COCO training path; a ``.json`` that isn't our schema (e.g. LabelMe) is not claimed."""
    d = Path(labels_dir)
    if not d.is_dir():
        return None
    for jp in sorted(d.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "objects" in data:
            return "json"
        break
    return "yolo" if any(d.glob("*.txt")) else None


def confirmed_negative_names(labels_dir) -> set[str]:
    """Image names a human explicitly marked negative (empty + Complete) for this label dir.

    Walks up from ``labels_dir`` to the project's ``.tcip/state/image_status.json`` (the GUI's
    completion store). An empty label file alone is never a negative — someone may have just
    emptied it mid-work — so training trusts only this human confirmation. No store → empty set.
    """
    d = Path(labels_dir).resolve()
    for parent in (d, *d.parents):
        status_file = parent / ".tcip" / "state" / "image_status.json"
        if status_file.is_file():
            try:
                statuses = json.loads(status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return set()
            return {name for name, s in statuses.items() if s == "negative"}
    return set()


def assemble_coco(labels_dir, images_dir, stems=None, categories=None) -> dict:
    """Assemble a dataset-level COCO dict from per-image JSON labels (the json_io schema).

    Pairs each stem's ``<labels_dir>/<stem>.json`` with its image's on-disk file name — the same
    name the dataset resolves at read time — so the COCO ``file_name`` keys line up. Stems whose
    image is missing are skipped; ``json_io.to_coco_dataset`` skips missing label files
    (unannotated is not a training negative). This is how per-image JSON reaches training: a COCO
    the existing ``label_format='coco'`` path consumes.
    """
    from tcip_annotation import json_io

    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    if stems is None:
        stems = sorted(p.stem for p in labels_dir.glob("*.json"))
    entries: list[tuple[str, str]] = []
    for stem in stems:
        try:
            file_name = _find_image(images_dir, stem).name
        except FileNotFoundError:
            continue
        entries.append((str(labels_dir / f"{stem}.json"), file_name))
    return json_io.to_coco_dataset(
        entries, categories=categories,
        confirmed_negative_names=confirmed_negative_names(labels_dir),
    )


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""
    expected_channels: int = 3  # input channels the dataset yields (3=RGB; set by build_dataset)

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


class BaseImageDataset(BaseDataset):
    """Base for image datasets — centralizes channel-aware loading + finalization.

    Subclasses set ``self.images_dir`` and ``self.transforms`` (and inherit
    ``expected_channels`` from build_dataset), then build only the task-specific target.
    """

    images_dir: Path
    transforms: Any = None

    def _resolve_path(self, stem: str) -> Path:
        """A ``stem`` may be a literal path (classification folder mode) or a stem in images_dir."""
        p = Path(stem)
        if p.is_absolute() or p.exists():
            return p
        return _find_image(self.images_dir, stem)

    def _open_image(self, stem: str):
        """Open an image honoring ``expected_channels`` (PIL for 1/3/4 ch, else ndarray)."""
        return load_image(self._resolve_path(stem), self.expected_channels)

    @staticmethod
    def _image_size(img) -> tuple[int, int]:
        """Return ``(width, height)`` for a PIL image or an ``[H, W, C]`` array."""
        if isinstance(img, Image.Image):
            return img.size
        return int(img.shape[1]), int(img.shape[0])

    def _finalize(self, img, target: dict) -> tuple[torch.Tensor, dict]:
        """Apply PIL transforms (when applicable) or convert straight to a tensor."""
        if self.transforms is not None and isinstance(img, Image.Image):
            return self.transforms(img, target)
        return pil_to_tensor(img), target


# ====================================================================
# Detection
# ====================================================================

class DetectionDataset(BaseImageDataset):
    """Object detection. ``label_format`` selects the on-disk label format:

    - ``json`` (default): canonical per-image ``<labels_dir>/<stem>.json`` (json_io schema)
    - ``coco``: a single COCO JSON at ``coco_json`` — the assembled dataset view of the per-image
      JSON, used for training (annotations matched by file name)
    - ``voc``: one PASCAL VOC ``<stem>.xml`` per image (external import format)
    - ``labelme``: one LabelMe ``<stem>.json`` per image (external import format)

    YOLO ``.txt`` is not a read format — import it to JSON first (migrate_labels_to_json).
    """

    task_type = "detection"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
        label_format: str = "json",
        coco_json: str | None = None,
        coco_data: dict | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.label_format = (label_format or "json").lower()
        self._coco = None
        if coco_data is not None:  # in-memory COCO assembled from per-image JSON (train/eval)
            self._coco = coco_data
            self.label_format = "coco"
        elif self.label_format == "coco":
            if not coco_json:
                raise ValueError("label_format='coco' requires coco_json (path to the COCO JSON).")
            self._coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
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
        if self.label_format == "coco" and self._coco:
            for ann in self._coco.get("annotations", []):
                counts[ann.get("category_id", 0)] += 1
        else:  # json / voc / labelme: parse each image's annotation
            for stem in self.stems:
                _, labels = _read_det_boxes_format(stem, self.labels_dir, self.label_format, None, 0, 0, "")
                for lab in labels:
                    counts[lab - 1] += 1  # back to 0-indexed cid
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)
        file_name = _find_image(self.images_dir, stem).name if self.label_format == "coco" else ""
        boxes, labels = _read_det_boxes_format(
            stem, self.labels_dir, self.label_format, self._coco, w, h, file_name)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": idx,
        }
        return self._finalize(img, target)


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
        sliver_frac: float = 0.5,
        dedup_iou: float = 0.8,
        skip_empty: bool = False,
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

        # Pass 1: read every image's upright dims + full-image-px boxes, and accumulate GT box sizes
        # so the seam-sliver cutoff is derived from this dataset's class-average object size, not a
        # fixed fraction (Q5 / derive-don't-pin). skip_empty defaults False: empty tiles are valid
        # negatives (the invariant the old skip_empty=True default violated).
        stems_data: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
        char_sizes: list[float] = []
        for stem in base.stems:
            img_path = _find_image(base.images_dir, stem)
            w, h = get_image_dimensions(str(img_path))  # EXIF-oriented, shares the label frame
            # Format-aware read: json (canonical) / coco (assembled) / voc / labelme all go through
            # the shared reader; only coco needs the image file name to match its annotations.
            file_name = img_path.name if base.label_format == "coco" else ""
            full_boxes, full_labels = _read_det_boxes_format(
                stem, base.labels_dir, base.label_format, base._coco, w, h, file_name)
            fb = np.asarray(full_boxes, dtype=np.float32).reshape(-1, 4)
            fl = np.asarray(full_labels, dtype=np.int64)
            if len(fb):
                char_sizes.extend((((fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1])).clip(min=0) ** 0.5).tolist())
            stems_data.append((stem, fb, fl, w, h))

        self.class_avg_size = float(np.mean(char_sizes)) if char_sizes else 0.0
        self.min_box_size = sliver_frac * self.class_avg_size

        # Pass 2: tile using the derived sliver cutoff.
        for stem, fb, fl, w, h in stems_data:
            for tile_x, tile_y in tile_positions(h, w, tile_size, self.stride):
                tb, tl = clip_boxes_to_tile(fb, fl, tile_x, tile_y, tile_size, self.min_box_size)
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
        # EXIF-oriented (via load_image) so cropped pixels align with the tile geometry and
        # labels computed from get_image_dimensions() in __init__.
        img = load_image(_find_image(self.base.images_dir, e["stem"]), 3)
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

class InstanceSegDataset(BaseImageDataset):
    """Instance masks from per-image polygons: canonical per-image JSON ``<stem>.json`` (default),
    or a COCO dict / assembled per-image JSON (``label_format='coco'`` / ``coco_data``)."""

    task_type = "instance_seg"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
        label_format: str = "json",
        coco_json: str | None = None,
        coco_data: dict | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.label_format = (label_format or "json").lower()
        self._coco = None
        if coco_data is not None:
            self._coco = coco_data
            self.label_format = "coco"
        elif self.label_format == "coco":
            if not coco_json:
                raise ValueError("label_format='coco' requires coco_json (path to the COCO JSON).")
            self._coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        self.stems = stems or sorted(
            f.stem for f in self.images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        )

    def _read_polys(self, stem: str, w: int, h: int) -> list[tuple[list[tuple[float, float]], int]]:
        """(pixel polygon points, 1-indexed label) per instance — from the COCO dict or the
        canonical per-image ``<stem>.json``. Both are already pixel-space."""
        if self.label_format == "coco":
            from tcip_annotation import format_io
            file_name = _find_image(self.images_dir, stem).name
            polys, _ = format_io.parse_coco_segment(self._coco, file_name=file_name)
        else:
            from tcip_annotation import json_io
            polys = json_io.read_segment(str(self.labels_dir / f"{stem}.json"))[0]
        return [(list(p.points), p.class_id + 1) for p in polys]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)

        boxes, labels, masks = [], [], []
        for pts, lab in self._read_polys(stem, w, h):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if not xs:
                continue
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(lab)

            # Rasterize polygon to binary mask
            mask = np.zeros((h, w), dtype=np.uint8)
            try:
                from PIL import ImageDraw
                poly_img = Image.new("L", (w, h), 0)
                ImageDraw.Draw(poly_img).polygon(list(zip(xs, ys)), fill=1)
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
        return self._finalize(img, target)


# ====================================================================
# Semantic Segmentation
# ====================================================================

class SemanticSegDataset(BaseImageDataset):
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
        img = self._open_image(stem)
        w, h = self._image_size(img)
        mask_path = self.masks_dir / f"{stem}.png"
        # load_image EXIF-orients so the mask shares the image's upright frame (no-op for a
        # plain PNG mask; matters only if a mask ever carries EXIF orientation).
        mask = np.array(load_image(mask_path, 1)) if mask_path.exists() else np.zeros((h, w), dtype=np.int64)
        # Key matches the SemanticSegHead loss contract.
        target = {"masks": torch.tensor(mask, dtype=torch.int64)}
        return self._finalize(img, target)


# ====================================================================
# Classification
# ====================================================================

class ClassificationDataset(BaseImageDataset):
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
        img = self._open_image(stem)
        target = {"labels": self._labels[idx]}
        return self._finalize(img, target)


# ====================================================================
# Ordinal
# ====================================================================

class OrdinalDataset(BaseImageDataset):
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
        img = self._open_image(stem)
        # Key matches the OrdinalHead loss contract (plural, like "labels"/"masks").
        target = {"ranks": self._ranks[idx], "num_ranks": self._num_ranks}
        return self._finalize(img, target)


# ====================================================================
# Regression
# ====================================================================

class RegressionDataset(BaseImageDataset):
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
        img = self._open_image(stem)
        # Key matches the RegressionHead loss contract.
        target = {"values": self._values[idx]}
        return self._finalize(img, target)


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


def _autoresolve_json_labels(kwargs: dict) -> None:
    """Route a canonical per-image-JSON label dir onto the assembled-COCO path for training/eval.
    No-op when the caller pinned a format or already supplied COCO data. A legacy YOLO ``.txt`` dir
    is rejected loudly (it is import-only) so it can never be read as all-empty negatives."""
    if kwargs.get("coco_data") is not None or kwargs.get("coco_json") or kwargs.get("label_format"):
        return
    labels_dir = kwargs.get("labels_dir", "")
    images_dir = kwargs.get("images_dir", "")
    if not labels_dir:
        return
    fmt = dir_label_format(labels_dir)
    if fmt == "json" and images_dir:
        kwargs["coco_data"] = assemble_coco(labels_dir, images_dir, stems=kwargs.get("stems"))
        kwargs["label_format"] = "coco"
    elif fmt == "yolo":
        raise ValueError(
            f"{labels_dir} holds legacy YOLO .txt labels — convert them to canonical per-image "
            "JSON first (scripts/migrate_labels_to_json.py). YOLO is import-only.")


def build_dataset(task: str, **kwargs) -> BaseDataset:
    """Factory: build a dataset by task type.

    An optional ``tiling`` dict (``{enabled, tile_size, overlap, sliver_frac,
    dedup_iou, skip_empty}``) wraps the detection dataset in a
    :class:`TiledDetectionDataset` (W3). Ignored for non-detection tasks.
    """
    tiling = kwargs.pop("tiling", None)
    num_channels = kwargs.pop("num_channels", 3)
    cls = _DATASET_MAP.get(task)
    if cls is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_DATASET_MAP.keys())}")

    if task in ("detection", "instance_seg"):
        _autoresolve_json_labels(kwargs)

    if tiling and tiling.get("enabled", True) and task == "detection":
        transforms = kwargs.pop("transforms", None)
        base = cls(**kwargs)
        tile_kwargs = {k: tiling[k] for k in
                       ("tile_size", "overlap", "sliver_frac", "dedup_iou", "skip_empty")
                       if k in tiling}
        ds = TiledDetectionDataset(base, transforms=transforms, **tile_kwargs)
    else:
        if tiling and tiling.get("enabled", True) and task != "detection":
            logger.warning("tiling is only supported for task='detection'; ignoring for task=%r", task)
        ds = cls(**kwargs)

    ds.expected_channels = num_channels
    return ds

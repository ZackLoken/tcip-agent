"""Class name ↔ ID mapping for detection datasets.

Provides a unified ClassMap that is used across:
  - Dataset loading (YOLO label class IDs → names)
  - Model building (num_classes derived from map)
  - Training (class names saved in checkpoint)
  - Inference (predictions include class names)
  - GUI (ClassSelector populated from map)
  - Export (classes.txt and data.yaml generation)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class ClassMap:
    """Bidirectional mapping between class names and integer IDs.

    IDs are 0-indexed (YOLO convention). When used with torchvision
    models that reserve 0 for background, call `torchvision_label(id)`
    which returns id + 1.

    Examples:
        >>> cm = ClassMap(["catkin", "leaf", "branch"])
        >>> cm.id("catkin")
        0
        >>> cm.name(1)
        'leaf'
        >>> cm.num_classes
        3
        >>> cm.torchvision_label(0)
        1
    """

    def __init__(self, names: list[str]) -> None:
        if not names:
            raise ValueError("ClassMap requires at least one class name")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate class names: {names}")
        self._names = list(names)
        self._name_to_id = {n: i for i, n in enumerate(names)}

    @property
    def num_classes(self) -> int:
        """Number of classes (excluding background)."""
        return len(self._names)

    @property
    def names(self) -> list[str]:
        """Ordered list of class names."""
        return list(self._names)

    def id(self, name: str) -> int:
        """Get 0-indexed ID for a class name."""
        if name not in self._name_to_id:
            raise KeyError(f"Unknown class: '{name}'. Known: {self._names}")
        return self._name_to_id[name]

    def name(self, class_id: int) -> str:
        """Get name for a 0-indexed class ID."""
        if class_id < 0 or class_id >= len(self._names):
            raise IndexError(f"Class ID {class_id} out of range [0, {len(self._names)})")
        return self._names[class_id]

    def torchvision_label(self, yolo_id: int) -> int:
        """Convert YOLO 0-indexed → torchvision 1-indexed (0 = background)."""
        return yolo_id + 1

    def from_torchvision(self, tv_label: int) -> int:
        """Convert torchvision 1-indexed → YOLO 0-indexed."""
        return tv_label - 1

    def to_dict(self) -> dict:
        """Serialize for embedding in configs/checkpoints."""
        return {"names": self._names}

    @classmethod
    def from_dict(cls, d: dict) -> ClassMap:
        """Reconstruct from serialized dict."""
        return cls(d["names"])

    # --- File I/O ---

    def write_classes_txt(self, path: str | Path) -> Path:
        """Write classes.txt (one class name per line, ordered by ID)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(self._names) + "\n", encoding="utf-8")
        logger.info("Wrote %d classes to %s", len(self._names), p)
        return p

    def write_data_yaml(self, path: str | Path, train_dir: str = "", val_dir: str = "") -> Path:
        """Write data.yaml compatible with YOLO training tools.

        Format:
            train: <train_dir>
            val: <val_dir>
            nc: <num_classes>
            names: [class1, class2, ...]
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train": str(train_dir),
            "val": str(val_dir),
            "nc": self.num_classes,
            "names": self._names,
        }
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info("Wrote data.yaml with %d classes to %s", self.num_classes, p)
        return p

    @classmethod
    def from_classes_txt(cls, path: str | Path) -> ClassMap:
        """Load from a classes.txt file (one name per line)."""
        p = Path(path)
        names = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not names:
            raise ValueError(f"No class names found in {p}")
        return cls(names)

    @classmethod
    def from_data_yaml(cls, path: str | Path) -> ClassMap:
        """Load from a data.yaml file."""
        p = Path(path)
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if not names:
            raise ValueError(f"No 'names' field in {p}")
        return cls(names)

    @classmethod
    def from_labels_dir(cls, labels_dir: str | Path) -> ClassMap:
        """Infer class count from label files (names will be generic: class_0, class_1, ...).

        Scans all .txt files in labels_dir to find the maximum class ID,
        then generates names. Useful as a fallback when no classes.txt exists.
        """
        labels_path = Path(labels_dir)
        max_id = -1
        for txt in labels_path.glob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    max_id = max(max_id, cid)

        if max_id < 0:
            raise ValueError(f"No class IDs found in {labels_path}")

        names = [f"class_{i}" for i in range(max_id + 1)]
        logger.warning("Inferred %d classes from labels (generic names). Consider providing classes.txt.", len(names))
        return cls(names)

    def __repr__(self) -> str:
        return f"ClassMap({self._names})"

    def __len__(self) -> int:
        return len(self._names)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ClassMap):
            return self._names == other._names
        return NotImplemented

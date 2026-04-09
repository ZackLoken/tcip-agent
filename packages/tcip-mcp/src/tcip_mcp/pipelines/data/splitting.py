"""Dataset splitting utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path


def create_splits(
    image_stems: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Divide image stems into train/val/test splits.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to lists of stems.
    """
    rng = random.Random(seed)
    stems = list(image_stems)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }


def save_splits(splits: dict[str, list[str]], output_dir: str) -> dict[str, str]:
    """Write split manifests as JSON files.

    Returns:
        Dict mapping split name to written file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, stems in splits.items():
        path = out / f"{name}.json"
        with open(path, "w") as f:
            json.dump(stems, f, indent=2)
        written[name] = str(path)
    return written


def load_splits(splits_dir: str) -> dict[str, list[str]]:
    """Load split manifests from JSON files."""
    d = Path(splits_dir)
    result = {}
    for name in ("train", "val", "test"):
        path = d / f"{name}.json"
        if path.is_file():
            with open(path) as f:
                result[name] = json.load(f)
    return result

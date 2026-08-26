"""Characterization goldens for the splits merge.

Freezes ``make_splits`` stats and ``split_dataset`` on-disk tree/return against current code, so
folding materialization into ``make_splits(materialize=...)`` is provably behavior-preserving:
``make_splits(materialize=False)`` reproduces the stats and ``make_splits(materialize=True)``
reproduces the tree plus ``output_dir``/``structure``.
"""

from __future__ import annotations

import json
from pathlib import Path

import tcip_store as ts
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox


# 4 source prefixes (srcA..srcD) x 3 tiles x 1 GT box each: 4 leakage groups, uniform density.
# No test partition: make_splits' default 0.8/0.2 train/val split carries every group into one.
GOLDEN_MAKE_SPLITS = {
    "splits": {"train": 9, "val": 3},
    "foreground_annotations": {"train": 9, "val": 3},
    "total_stems": 12,
    "total_annotations": 12,
    "groups": 4,
    "seed": 1,
    "group_by": "tile_prefix",
    "stratified": True,
}

# split_dataset (seed=1) and make_splits (seed=1) assign the same groups per split.
GOLDEN_TREE = sorted([
    "train.json", "val.json", "split_manifest.json",
    "train/images/srcB_0_0.jpg", "train/images/srcB_1_0.jpg", "train/images/srcB_2_0.jpg",
    "train/images/srcC_0_0.jpg", "train/images/srcC_1_0.jpg", "train/images/srcC_2_0.jpg",
    "train/images/srcD_0_0.jpg", "train/images/srcD_1_0.jpg", "train/images/srcD_2_0.jpg",
    "train/labels/srcB_0_0.json", "train/labels/srcB_1_0.json", "train/labels/srcB_2_0.json",
    "train/labels/srcC_0_0.json", "train/labels/srcC_1_0.json", "train/labels/srcC_2_0.json",
    "train/labels/srcD_0_0.json", "train/labels/srcD_1_0.json", "train/labels/srcD_2_0.json",
    "val/images/srcA_0_0.jpg", "val/images/srcA_1_0.jpg", "val/images/srcA_2_0.jpg",
    "val/labels/srcA_0_0.json", "val/labels/srcA_1_0.json", "val/labels/srcA_2_0.json",
])


def _multi_source_dataset(root: Path) -> Path:
    date = "2-11-26"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = root / "annotations" / date
    labels_dir.mkdir(parents=True)
    for pref in ("srcA", "srcB", "srcC", "srcD"):
        for t in range(3):
            stem = f"{pref}_{t}_0"
            Image.new("RGB", (64, 64), (128, 128, 128)).save(images_dir / f"{stem}.jpg")
            json_io.write_annotations(str(labels_dir / f"{stem}.json"),
                                      [Annotation(subject="catkin", geometry=BBox(10, 10, 30, 30))], 64, 64)
    return root


def _tree(out_dir: Path) -> list[str]:
    # Lock files outlive writes on POSIX; the golden tree lists the split's real artifacts.
    return sorted(
        str(p.relative_to(out_dir)).replace("\\", "/")
        for p in out_dir.rglob("*")
        if p.is_file() and p.suffix != ".lock"
    )


def test_make_splits_stats_golden(tmp_path: Path):
    from tcip_mcp.tools.data_tools import make_splits

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = make_splits(str(root), output_path=str(out), seed=1, subject="catkin")
    result.pop("manifest_dir")
    assert result.pop("subject") == "catkin"
    assert result.pop("attribute") is None
    admission_counts = result.pop("admission_counts")
    assert admission_counts["annotated"] == 12
    assert result == GOLDEN_MAKE_SPLITS


def test_make_splits_materialize_tree_golden(tmp_path: Path):
    """Bound to the file backend on purpose: the golden lists the split's own record documents
    (train.json, split_manifest.json, ...) as sibling files, a fact about the file layout a
    database backend does not reproduce."""
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.tools.data_tools import make_splits

    ts.bind(FileBackend())
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "s"
    result = make_splits(str(root), output_path=str(out), seed=1, materialize=True, subject="catkin")
    assert result["splits"] == {"train": 9, "val": 3}
    assert result["total_stems"] == 12
    assert result["seed"] == 1
    assert result["output_dir"] == str(out)
    assert result["structure"] == f"{out}/{{train,val}}/{{images,labels}}/"
    assert _tree(out) == GOLDEN_TREE
    for split in ("train", "val"):
        assert json.loads((out / f"{split}.json").read_text())

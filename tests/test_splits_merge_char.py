"""Characterization goldens for the splits merge.

Freezes ``draw_splits`` stats and ``split_dataset`` on-disk tree/return against current code, so
folding materialization into ``draw_splits(materialize=...)`` is provably behavior-preserving:
``draw_splits(materialize=False)`` reproduces the stats and ``draw_splits(materialize=True)``
reproduces the tree plus ``output_dir``/``structure``.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox


# 4 source prefixes (srcA..srcD) x 3 tiles x 1 GT box each: 4 leakage groups, uniform density,
# exactly meeting the manifest floor (one group each for train/val, two for calibration).
GOLDEN_DRAW_SPLITS = {
    "splits": {"train": 3, "val": 3, "calibration": 6},
    "foreground_annotations": {"train": 3, "val": 3, "calibration": 6},
    "total_stems": 12,
    "total_annotations": 12,
    "groups": 4,
    "seed": 1,
    "group_by": "tile_prefix",
    "stratified": True,
}
GOLDEN_CALIBRATION_FOREGROUND_GROUPS_BY_DATE = {"2-11-26": 2}
GOLDEN_REALIZED_RATIOS = {"train": 0.25, "val": 0.25, "calibration": 0.5}

# split_dataset (seed=1) and draw_splits (seed=1) assign the same groups per split.
GOLDEN_TREE = sorted([
    "split_manifest.json",
    "train/images/srcD_0_0.jpg", "train/images/srcD_1_0.jpg", "train/images/srcD_2_0.jpg",
    "train/labels/srcD_0_0.json", "train/labels/srcD_1_0.json", "train/labels/srcD_2_0.json",
    "val/images/srcA_0_0.jpg", "val/images/srcA_1_0.jpg", "val/images/srcA_2_0.jpg",
    "val/labels/srcA_0_0.json", "val/labels/srcA_1_0.json", "val/labels/srcA_2_0.json",
    "calibration/images/srcB_0_0.jpg", "calibration/images/srcB_1_0.jpg",
    "calibration/images/srcB_2_0.jpg", "calibration/images/srcC_0_0.jpg",
    "calibration/images/srcC_1_0.jpg", "calibration/images/srcC_2_0.jpg",
    "calibration/labels/srcB_0_0.json", "calibration/labels/srcB_1_0.json",
    "calibration/labels/srcB_2_0.json", "calibration/labels/srcC_0_0.json",
    "calibration/labels/srcC_1_0.json", "calibration/labels/srcC_2_0.json",
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
                                      [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))], 64, 64)
    return root


def _tree(out_dir: Path) -> list[str]:
    # Lock files outlive writes on POSIX; the golden tree lists the split's real artifacts.
    return sorted(
        str(p.relative_to(out_dir)).replace("\\", "/")
        for p in out_dir.rglob("*")
        if p.is_file() and p.suffix != ".lock"
    )


def test_draw_splits_stats_golden(tmp_path: Path):
    from tcip_mcp.tools.data_tools import draw_splits

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    result.pop("manifest_dir")
    assert result.pop("subject") == "bud"
    assert result.pop("attribute") is None
    admission_counts = result.pop("admission_counts")
    assert admission_counts["annotated"] == 12
    hashes = result.pop("dataset_hashes_by_date")
    assert list(hashes) == ["2-11-26"] and hashes["2-11-26"]
    assert result.pop("calibration_foreground_groups_by_date") == \
        GOLDEN_CALIBRATION_FOREGROUND_GROUPS_BY_DATE
    assert result.pop("realized_ratios") == GOLDEN_REALIZED_RATIOS
    assert result == GOLDEN_DRAW_SPLITS


def test_draw_splits_materialize_tree_golden(tmp_path: Path):
    """Bound to the file backend on purpose: the golden lists the split's own record document
    (split_manifest.json) as a sibling file, a fact about the file layout a database backend
    does not reproduce."""
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key

    ts.bind(FileBackend())
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "s"
    result = draw_splits(str(root), output_path=str(out), seed=1, materialize=True, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert result["splits"] == {"train": 3, "val": 3, "calibration": 6}
    assert result["total_stems"] == 12
    assert result["seed"] == 1
    assert result["output_dir"] == str(out)
    assert result["structure"] == f"{out}/{{train,val,calibration}}/{{images,labels}}/"
    assert _tree(out) == GOLDEN_TREE
    manifest = ts.read(split_manifest_key(out))
    for split in ("train", "val", "calibration"):
        assert manifest["splits"][split]

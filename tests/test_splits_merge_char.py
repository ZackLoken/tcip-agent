"""Characterization goldens for ``draw_splits``.

Freezes the statistics ``draw_splits`` returns and the selection document it writes beside
them, so a change to either shape fails here.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox


# 4 source prefixes (srcA..srcD) x 3 tiles x 1 GT box each: 4 leakage groups, uniform density,
# exactly meeting the draw's floor (one group each for train/val, two for calibration).
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
GOLDEN_CALIBRATION_FOREGROUND_GROUPS = 2
GOLDEN_REALIZED_RATIOS = {"train": 0.25, "val": 0.25, "calibration": 0.5}

# The side each group's crops land on at seed 1: the draw places whole groups, never crops. A
# group key carries its capture date, so two dates' same-named sources are two groups.
GOLDEN_SIDE_BY_GROUP = {
    "2-11-26/srcD": "train", "2-11-26/srcA": "val",
    "2-11-26/srcB": "calibration", "2-11-26/srcC": "calibration",
}


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


def test_draw_splits_stats_golden(tmp_path: Path):
    from tcip_mcp.tools.data_tools import draw_splits

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    result.pop("selection_dir")
    assert result.pop("subject") == "bud"
    assert result.pop("attribute") is None
    admission_counts = result.pop("admission_counts")
    assert admission_counts["annotated"] == 12
    assert result.pop("calibration_foreground_groups") == GOLDEN_CALIBRATION_FOREGROUND_GROUPS
    assert result.pop("realized_ratios") == GOLDEN_REALIZED_RATIOS
    assert result == GOLDEN_DRAW_SPLITS


def test_draw_splits_selection_document_golden(tmp_path: Path):
    """Bound to the file backend on purpose: the selection's own document sits beside the draw
    as ``selection.json``, a fact about the file layout a database backend does not reproduce.
    Every sample names its own source and label; the draw copies nothing."""
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    ts.bind(FileBackend())
    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "s"
    result = draw_splits(str(root), output_path=str(out), seed=1, subject="bud",
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert result["splits"] == {"train": 3, "val": 3, "calibration": 6}
    assert result["total_stems"] == 12
    assert result["seed"] == 1
    assert result["selection_dir"] == str(out)

    written = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*")
                     if p.is_file() and p.suffix != ".lock")
    assert written == ["selection.json"]

    drawn = read_selection(out)
    assert drawn.counts() == {"train": 3, "val": 3, "calibration": 6}
    assert {s.group: s.side for s in drawn.samples} == GOLDEN_SIDE_BY_GROUP
    for sample in drawn.samples:
        assert Path(sample.source).parent == root / "images" / "2-11-26"
        assert Path(sample.ground_truth).parent == root / "annotations" / "2-11-26"

"""Characterization goldens for ``draw_splits``.

Freezes the statistics ``draw_splits`` returns and the selection document it writes beside
them.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import image_dir
from tests._producer_fixtures import image_label_key, label_image, write_image
from tests._training_values import RATIONALE, drawn_split

_DRAW = drawn_split(
    seed=1,
    val_ratio=0.25,
    calibration_ratio=0.125,
    holdout_ratio=0.125,
    grouping="tile_prefix",
)
"""The draw both goldens are of."""

GOLDEN_REQUESTED_RATIOS = {
    "train": 0.5,
    "val": 0.25,
    "calibration": 0.125,
    "holdout": 0.125,
}
"""The shares :data:`_DRAW` requests, as its selection records them."""

# 4 source prefixes (srcA..srcD) x 3 tiles x 1 GT box each: 4 leakage groups, uniform density,
# exactly meeting the draw's floor (one group each for train/val, two for the reference).
GOLDEN_DRAW_SPLITS = {
    "splits": {"train": 3, "val": 3, "calibration": 3, "holdout": 3},
    "foreground_annotations": {
        "train": 3,
        "val": 3,
        "calibration": 3,
        "holdout": 3,
    },
    "total_stems": 12,
    "total_annotations": 12,
    "groups": 4,
    "seed": 1,
    "group_by": "tile_prefix",
    "stratify_foreground": True,
}
GOLDEN_CALIBRATION_FOREGROUND_GROUPS = 2
GOLDEN_REALIZED_RATIOS = {
    "train": 0.25,
    "val": 0.25,
    "calibration": 0.25,
    "holdout": 0.25,
}

# The side each group's crops land on at seed 1: the draw places whole groups, never crops. A
# group key carries its capture date, so two dates' same-named sources are two groups.
GOLDEN_SIDE_BY_GROUP = {
    "2-11-26/srcD": "train",
    "2-11-26/srcA": "val",
    "2-11-26/srcB": "holdout",
    "2-11-26/srcC": "calibration",
}


def _multi_source_dataset(root: Path) -> Path:
    date = "2-11-26"
    images_dir = image_dir(root, date)
    for pref in ("srcA", "srcB", "srcC", "srcD"):
        for t in range(3):
            image = write_image(
                images_dir / f"{pref}_{t}_0.jpg", (64, 64), (128, 128, 128)
            )
            label_image(
                image,
                [Annotation(subject="bud", geometry=BBox(10, 10, 30, 30))],
                64,
                64,
            )
    return root


def test_draw_splits_stats_golden(tmp_path: Path) -> None:
    from tcip_mcp.tools.data_tools import draw_splits

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    result = draw_splits(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="bud",
        **_DRAW,
        actor=None,
    )
    result.pop("selection_dir")
    assert result.pop("scope") == {"subject": "bud", "attributes": []}
    assert result.pop("tallies") == {"partial": 12}
    assert (
        result.pop("calibration_foreground_groups")
        == GOLDEN_CALIBRATION_FOREGROUND_GROUPS
    )
    assert result.pop("realized_ratios") == GOLDEN_REALIZED_RATIOS
    assert result == GOLDEN_DRAW_SPLITS


def test_draw_splits_selection_document_golden(tmp_path: Path) -> None:
    """The selection is the draw's one record, in the database under its directory.

    Every sample's source lies in the dataset's capture directory and its ground truth is that
    image's label key, and the selection directory holds ``.tcip/store.db`` alone.
    """
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_store.file_backend import is_bookkeeping

    root = _multi_source_dataset(tmp_path / "ds")
    out = tmp_path / "s"
    result = draw_splits(
        tmp_path,
        str(root),
        output_path=str(out),
        subject="bud",
        **_DRAW,
        actor=None,
    )
    assert result["splits"] == GOLDEN_DRAW_SPLITS["splits"]
    assert result["total_stems"] == 12
    assert result["seed"] == 1
    assert result["selection_dir"] == str(out)

    written = sorted(
        str(p.relative_to(out)).replace("\\", "/")
        for p in out.rglob("*")
        if p.is_file() and not is_bookkeeping(p.name)
    )
    assert written == [".tcip/store.db"]

    drawn = read_selection(out, project=tmp_path)
    assert drawn.counts() == GOLDEN_DRAW_SPLITS["splits"]
    assert {s.group: s.side for s in drawn.samples} == GOLDEN_SIDE_BY_GROUP
    assert drawn.ratios == GOLDEN_REQUESTED_RATIOS
    assert (drawn.stratify_foreground, drawn.rationale) == (True, RATIONALE)
    for sample in drawn.samples:
        assert Path(sample.source).parent == image_dir(root, "2-11-26")
        assert sample.ground_truth == image_label_key(sample.source)

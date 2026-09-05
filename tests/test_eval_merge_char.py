"""Characterization goldens for the eval-on-disk merge (Merge A).

Freezes the exact return dicts of the on-disk evaluators against current code, so the
``evaluate_detections`` + ``evaluate_dataset`` → ``score_predictions`` consolidation is
provably behavior-preserving. Before the merge these assert the two original tools; after it
they assert ``score_predictions`` reproduces the same dicts keyed on file-vs-dir input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tcip_mcp.tools.annotation_tools")


# Frozen at conf_threshold=0.5 on the canonical ``data_dir`` fixture (1 TP + 1 FP + 1 FN/image).
GOLDEN_DETECTIONS = {
    "tp": 1,
    "fp": 1,
    "fn": 1,
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.5,
    "map50": 0.505,
    "iou_type": "bbox",
    "iou_threshold": 0.5,
    "conf_threshold": 0.5,
    "matches": {
        "tp": [{"gt_idx": 0, "pred_idx": 0, "iou": 1.0, "class_name": "bud", "conf": 0.9}],
        "fp": [{"pred_idx": 1, "class_name": "bud", "conf": 0.7}],
        "fn": [{"gt_idx": 1, "class_name": "bud"}],
    },
    "img_w": 640,
    "img_h": 480,
    "detections": [
        {"tag": "tp", "class_name": "bud", "iou": 1.0, "confidence": 0.9,
         "gt_idx": 0, "pred_idx": 0, "box": [288.0, 216.0, 64.0, 48.0]},
        {"tag": "fp", "class_name": "bud", "confidence": 0.7, "pred_idx": 1,
         "box": [496.0, 372.0, 32.0, 24.0]},
        {"tag": "fn", "class_name": "bud", "confidence": 0, "gt_idx": 1,
         "box": [176.0, 132.0, 32.0, 24.0]},
    ],
}

GOLDEN_DATASET = {
    "image_count": 3,
    "map": 0.505,
    "map50": 0.505,
    "total_tp": 3,
    "total_fp": 3,
    "total_fn": 3,
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.5,
    "iou_type": "bbox",
    "per_image": [
        {"image": "img_001.jpg", "tp": 1, "fp": 1, "fn": 1},
        {"image": "img_002.jpg", "tp": 1, "fp": 1, "fn": 1},
        {"image": "img_003.jpg", "tp": 1, "fp": 1, "fn": 1},
    ],
}


def test_evaluate_single_image_golden(data_dir: Path):
    from tcip_mcp.tools.annotation_tools import score_predictions

    img = data_dir / "images" / "2-11-26" / "img_001.jpg"
    result = score_predictions(str(img), iou_threshold=0.5, conf_threshold=0.5, detail=True)
    assert result.pop("image") == str(img)
    assert result == GOLDEN_DETECTIONS


def test_evaluate_folder_golden(data_dir: Path):
    from tcip_mcp.tools.annotation_tools import score_predictions

    result = score_predictions(str(data_dir), iou_threshold=0.5, conf_threshold=0.5)
    assert result.pop("path") == str(data_dir)
    assert result == GOLDEN_DATASET

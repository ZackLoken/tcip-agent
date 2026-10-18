"""tcip score-predictions: the scoring door's command-line entry point.

A plain score (no --trait) scores a single image with no project named, and --trait refuses
without --project; neither writes a ``.tcip`` directory under the working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import write_image

SCORED_AT = ["--iou-threshold", "0.5", "--conf-threshold", "0.25"]
"""A sample IoU and confidence each scoring here states."""


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    """Write an image, its label document and a published bucket; return image and bucket name."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published
    from tests._producer_fixtures import label_image

    img = tmp_path / "images" / UNDATED_BUCKET / "IMG_0000.jpg"
    write_image(img)
    label_image(
        img, [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))], 100, 80
    )
    published(
        tmp_path,
        "baseline",
        [
            {
                "image": str(img),
                "width": 100,
                "height": 80,
                "boxes": [[1.0, 1.0, 40.0, 30.0]],
                "scores": [0.9],
                "labels": [1],
                "cap": 2,
            }
        ],
        scope={"subject": "bud"},
    )
    return img, "baseline"


def test_refuses_a_trait_scoped_run_naming_no_project(tmp_path):
    img, bucket = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "score-predictions",
        [
            "--path",
            str(img),
            "--bucket",
            bucket,
            *SCORED_AT,
            "--trait",
            "bud_count",
        ],
        cwd=cwd,
    )

    assert result.returncode != 0, result.stdout
    assert "--trait requires --project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_scores_a_single_image_with_no_project_named(tmp_path):
    img, bucket = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "score-predictions",
        ["--path", str(img), "--bucket", bucket, *SCORED_AT],
        cwd=cwd,
    )

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert body["matches"] == [[0, 0]]
    assert body["tp"] == 1
    assert not (cwd / ".tcip").exists()

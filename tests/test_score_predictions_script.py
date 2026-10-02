"""tcip score-predictions: the demoted door's own command-line entry point.

A plain score (no --trait) needs no project at all, since it only reads the image and its
label/prediction files by path; --trait requires --project, since resolving a trait's derived
localization criterion reads that project's own confirmed revision.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """An image, its label document, and a published bucket holding its document."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    img = tmp_path / "images" / "IMG_0000.jpg"
    _write_image(img)
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))], 100, 80)
    preds = published(tmp_path, tmp_path / "predictions" / "baseline", [
        {"image": str(img), "width": 100, "height": 80, "boxes": [[1.0, 1.0, 40.0, 30.0]],
         "scores": [0.9], "labels": [1]}],
        scope={"subject": "bud"}).path
    return img, preds


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "score-predictions", *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=60,
    )


def test_refuses_a_trait_scoped_run_naming_no_project(tmp_path):
    img, preds = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--path", str(img), "--predictions-dir", str(preds), "--trait", "bud_count"],
                  cwd=cwd)

    assert result.returncode != 0, result.stdout
    assert "--trait requires --project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_scores_a_single_image_with_no_project_named(tmp_path):
    img, preds = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--path", str(img), "--predictions-dir", str(preds)], cwd=cwd)

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert body["matches"] == [[0, 0]]
    assert body["tp"] == 1
    assert not (cwd / ".tcip").exists()

"""tcip score-predictions: the demoted door's own command-line entry point.

A plain score (no --trait) needs no platform root at all, since it only reads the image and its
label/prediction files by path; --trait requires one, since resolving a trait's derived
localization criterion reads the project's own trait registry, matching the shared
require_and_pin_platform_root mechanism test_platform_root_pinning.py covers directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(120, 120, 120)).save(path)


def _fixture(tmp_path: Path) -> Path:
    img = tmp_path / "images" / "IMG_0000.jpg"
    _write_image(img)
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(labels / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30))], 100, 80)
    preds = tmp_path / "predictions" / "baseline"
    preds.mkdir(parents=True)
    write_annotations(preds / "IMG_0000.json",
                      [Annotation(subject="bud", geometry=BBox(1, 1, 40, 30), score=0.9)],
                      100, 80)
    return img


def _run(args: list[str], cwd: Path, platform_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if platform_root is None:
        env.pop("TCIP_STATE_ROOT", None)
    else:
        env["TCIP_STATE_ROOT"] = platform_root
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "score-predictions", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def test_refuses_a_trait_scoped_run_from_an_unpinned_cwd(tmp_path):
    img = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--path", str(img), "--trait", "bud_count"], cwd=cwd, platform_root=None)

    assert result.returncode != 0, result.stdout
    assert "TCIP_STATE_ROOT" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_scores_a_single_image_over_a_fixture_root_with_no_project_pinned(tmp_path):
    img = _fixture(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--path", str(img)], cwd=cwd, platform_root=None)

    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert "error" not in body
    assert len(body["matches"]["tp"]) == 1
    assert body["matches"]["tp"][0]["class_name"] == "bud"
    assert not (cwd / ".tcip").exists()

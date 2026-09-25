"""Tests for tcip check-dataset-identity's outcomes: OK, CHANGED, VERSION-REFUSED,
NEVER-RECORDED and MOVED."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_mcp.dataset_layout import dataset_identity_key, require_dataset_identity
from tcip_mcp.tools.project_tools import register_dataset, upsert_dataset


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "check-dataset-identity", *args],
        capture_output=True, text=True, timeout=60,
    )


def _real_dataset(root: Path) -> None:
    images = root / "images" / "2024-01-01"
    labels = root / "annotations" / "2024-01-01"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    (labels / "a.json").write_text(
        '{"image": "a", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )


def test_a_matching_fingerprint_reports_ok(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    completed = _run_script(str(root))

    assert completed.returncode == 0, completed.stdout
    assert "OK" in completed.stdout


def test_a_real_content_change_still_reports_changed(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    (root / "annotations" / "2024-01-01" / "b.json").write_text(
        '{"image": "b", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )
    Image.new("RGB", (10, 10), (4, 5, 6)).save(root / "images" / "2024-01-01" / "b.png")

    completed = _run_script(str(root))

    assert completed.returncode == 2, completed.stdout
    assert "CHANGED" in completed.stdout


def test_a_version_refused_identity_reports_its_own_outcome_not_a_crash(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    identity = require_dataset_identity(root)
    document = {**identity, "schema_version": 2}
    ts.put_blob(dataset_identity_key(root), ts.RECORD_JSON.encode(document))

    completed = _run_script(str(root))

    assert completed.returncode == 5, completed.stdout + completed.stderr
    assert "VERSION-REFUSED" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_a_never_recorded_fingerprint_is_its_own_outcome(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    assert result["fingerprint"] is None

    # Real content shows up after the fingerprint-less registration: recorded stays None while
    # a fresh recompute now finds real content.
    _real_dataset(root)

    completed = _run_script(str(root))

    assert completed.returncode == 4, completed.stdout
    assert "NEVER-RECORDED" in completed.stdout


def test_a_moved_dataset_is_reported_moved(tmp_path):
    root = tmp_path / "dataset"
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(tmp_path))
    assert "error" not in result, result

    # The registry now names a different, no-longer-existing path for this same id/fingerprint,
    # standing in for the dataset having been moved without ever touching root's own live files.
    stale_path = str((tmp_path / "gone_now").resolve())
    upsert_dataset(tmp_path, {"id": result["id"], "path": stale_path,
                              "crop": "chestnut", "fingerprint": result["fingerprint"]})

    completed = _run_script(str(root), "--project", str(tmp_path))

    assert f"MOVED: id {result['id']}" in completed.stdout, completed.stdout



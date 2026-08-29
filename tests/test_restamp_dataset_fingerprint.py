"""Tests for scripts/restamp_dataset_fingerprint.py, run by subprocess against a project.

register_dataset now stamps a formula-versioned fingerprint, so the bare legacy shape this
script restamps cannot be produced by any writer any more; each fixture registers a dataset for
real and then hand-edits its identity record to a bare value, standing in for a dataset
registered before this family the way test_conform_dataset_registry_paths.py's fixtures stand in
for a project registered before its own family.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_mcp.dataset_layout import dataset_identity_key, require_dataset_identity
from tcip_mcp.pipelines.resolution import dataset_fingerprint
from tcip_mcp.tools.project_tools import register_dataset

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "restamp_dataset_fingerprint.py"


def _run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
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


def _bare_the_identity(root: Path) -> str:
    """Overwrite a real registration's prefixed fingerprint with its bare hex, standing in for a
    dataset registered before the formula-version prefix existed."""
    identity = require_dataset_identity(root)
    prefixed = identity["fingerprint"]
    bare = prefixed.split(":", 1)[1]
    document = {**identity, "fingerprint": bare}
    ts.put_blob(dataset_identity_key(root), ts.RECORD_JSON.encode(document))
    return bare


def test_a_bare_matching_fingerprint_is_restamped_through_register_datasets_own_path(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    bare = _bare_the_identity(root)

    completed = _run_script(str(root), "--project", str(root))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RESTAMPED" in completed.stdout
    identity = require_dataset_identity(root)
    assert identity["fingerprint"] == f"v1:{bare}"


def test_an_already_prefixed_fingerprint_is_left_alone(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    before = require_dataset_identity(root)["fingerprint"]

    completed = _run_script(str(root), "--project", str(root))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout
    assert require_dataset_identity(root)["fingerprint"] == before


def test_a_bare_fingerprint_that_no_longer_matches_the_recompute_refuses_and_reports(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    bare = _bare_the_identity(root)

    # A real content change after the identity was bare-ified: the recompute no longer agrees.
    (root / "annotations" / "2024-01-01" / "b.json").write_text(
        '{"image": "b", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )
    Image.new("RGB", (10, 10), (4, 5, 6)).save(root / "images" / "2024-01-01" / "b.png")
    assert dataset_fingerprint(root).split(":", 1)[1] != bare

    completed = _run_script(str(root), "--project", str(root))

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "MISMATCH" in completed.stdout
    assert require_dataset_identity(root)["fingerprint"] == bare  # nothing was written

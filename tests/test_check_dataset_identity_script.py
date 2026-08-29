"""Tests for scripts/check_dataset_identity.py's CHANGED branch: a bare recorded fingerprint
compared against a prefixed recompute must report formula-unrecorded with the re-register
remedy, never CHANGED, since the two were never computed under the same formula.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_mcp.dataset_layout import dataset_identity_key, require_dataset_identity
from tcip_mcp.tools.project_tools import register_dataset

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_dataset_identity.py"


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


def test_a_matching_prefixed_fingerprint_reports_ok(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    completed = _run_script(str(root))

    assert completed.returncode == 0, completed.stdout
    assert "OK" in completed.stdout


def test_a_bare_fingerprint_reports_formula_unrecorded_never_changed(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    identity = require_dataset_identity(root)
    bare = identity["fingerprint"].split(":", 1)[1]
    document = {**identity, "fingerprint": bare}
    ts.put_blob(dataset_identity_key(root), ts.RECORD_JSON.encode(document))

    completed = _run_script(str(root))

    assert completed.returncode == 3, completed.stdout
    assert "FORMULA-UNRECORDED" in completed.stdout
    assert "CHANGED" not in completed.stdout
    assert "register_dataset" in completed.stdout


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
    assert "FORMULA-UNRECORDED" not in completed.stdout

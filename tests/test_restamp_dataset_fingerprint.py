"""Tests for scripts/restamp_dataset_fingerprint.py, run by subprocess against a project.

register_dataset now stamps a formula-versioned fingerprint, so the bare legacy shape this
script restamps cannot be produced by any writer any more; each fixture registers a dataset for
real and then hand-edits its identity record to a bare value, standing in for a dataset
registered before this family the way test_conform_dataset_registry_paths.py's fixtures stand in
for a project registered before its own family.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_mcp.dataset_layout import dataset_identity_key, require_dataset_identity
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tcip_mcp.tools.project_tools import read_datasets, register_dataset, upsert_dataset

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


def test_a_never_recorded_fingerprint_is_nothing_to_restamp(tmp_path):
    # No images/labels: dataset_fingerprint is None, so both records agree at None already.
    root = tmp_path / "dataset"
    root.mkdir()
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    assert result["fingerprint"] is None

    completed = _run_script(str(root), "--project", str(root))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "never recorded a fingerprint" in completed.stdout
    assert require_dataset_identity(root)["fingerprint"] is None  # nothing was written


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
    assert "formula shift" in completed.stdout  # names both possibilities, not just a real change
    assert require_dataset_identity(root)["fingerprint"] == bare  # nothing was written


def test_a_stale_bare_registry_entry_is_restamped_even_when_identity_is_already_prefixed(
    tmp_path,
):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    bare = result["fingerprint"].split(":", 1)[1]
    # Stands in for a registry entry a prior restamp run never reached (item 5a): the dataset's
    # own identity record already carries the prefix, but this project's registry entry does not.
    upsert_dataset(root, {"id": result["id"], "path": ".", "crop": "chestnut", "fingerprint": bare})

    completed = _run_script(str(root), "--project", str(root))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RESTAMPED" in completed.stdout
    matches = [r for r in read_datasets(root) if r.get("id") == result["id"]]
    assert matches[0]["fingerprint"] == result["fingerprint"]


def test_a_registry_entry_disagreeing_with_identity_refuses_rather_than_pick_one(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    bare = _bare_the_identity(root)
    upsert_dataset(root, {"id": result["id"], "path": ".", "crop": "chestnut",
                          "fingerprint": "f" * 16})

    completed = _run_script(str(root), "--project", str(root))

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "DISAGREE" in completed.stdout
    assert require_dataset_identity(root)["fingerprint"] == bare  # nothing was written


def test_restamp_reports_a_post_write_mismatch_instead_of_trusting_its_own_recompute(
    tmp_path, monkeypatch, capsys,
):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    _bare_the_identity(root)

    spec = importlib.util.spec_from_file_location("restamp_dataset_fingerprint_under_test", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _wrong_register_dataset(dataset_root: str, crop: str, project_root: str) -> dict:
        # Stands in for register_dataset landing a different value than this run just verified
        # (item 5b): the write happens, but what actually landed disagrees with the comparison.
        wrong = "v1:" + "0" * 16
        document = {**require_dataset_identity(Path(dataset_root)), "fingerprint": wrong}
        ts.put_blob(dataset_identity_key(Path(dataset_root)), ts.RECORD_JSON.encode(document))
        return {"dataset_root": dataset_root, **document}

    monkeypatch.setattr(module, "register_dataset", _wrong_register_dataset)

    code = module.restamp(str(root), str(root))
    out = capsys.readouterr().out

    assert code == 2
    assert "WROTE-BUT-MISMATCH" in out
    assert require_dataset_identity(root)["fingerprint"] == "v1:" + "0" * 16  # the fake did write

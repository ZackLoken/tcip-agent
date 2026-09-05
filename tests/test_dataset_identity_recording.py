"""The experiment immutably records the dataset identity it trained on.

The content end of the reproduce-a-number chain: id + fingerprint are written into the lineage and
split.json at creation, are never backfilled/changed via update_lineage (identity, not a mutable
edge), and compare_experiments surfaces whether two runs share a dataset so a metric comparison across
different data is not read as apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
import tcip_mcp.experiments as exp
from tcip_mcp.experiments import compare_experiments, create_experiment, update_lineage


@pytest.fixture
def exp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIMENTS_DIR", tmp_path / "experiments")
    return tmp_path / "experiments"


def _make_dataset(root: Path) -> None:
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    (root / "images" / "2-11-26").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2-11-26" / "img_000.jpg")
    (root / "annotations" / "2-11-26").mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(str(root / "annotations" / "2-11-26" / "img_000.json"),
                              [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    class_registry.write_registry(root / "classes.json",
                                  ClassRegistry(subjects=(Subject(name="bud"),)))


def test_create_experiment_records_identity_in_lineage(exp_dir):
    create_experiment("e1", {}, dataset_id="abc123", dataset_fingerprint="ff00")
    lin = ts.read(exp.lineage_key("e1"))
    assert lin["dataset_id"] == "abc123" and lin["dataset_fingerprint"] == "ff00"


def test_update_lineage_cannot_change_or_backfill_identity(exp_dir):
    create_experiment("e1", {}, dataset_id="abc123", dataset_fingerprint="ff00")
    # a recorded identity is immutable; a legitimate edge (predictions) still updates
    res = update_lineage("e1", dataset_fingerprint="DIFFERENT", predictions="w.pt")
    assert res["lineage"]["dataset_fingerprint"] == "ff00"
    assert res["lineage"]["predictions"] == "w.pt"
    # a run that recorded None identity stays None, never silently backfilled
    create_experiment("e2", {})
    update_lineage("e2", dataset_fingerprint="sneaky")
    assert ts.read(exp.lineage_key("e2"))["dataset_fingerprint"] is None


def test_update_lineage_still_applies_legitimate_updates_when_identity_audit_fails(exp_dir, monkeypatch):
    """A dropped identity update is audited before the transaction that applies the call's other,
    legitimate updates; when that audit append itself fails, the transaction must still run (the
    legitimate update must still land) and the append failure is raised at the end, not lost."""
    create_experiment("e3", {}, dataset_id="abc123", dataset_fingerprint="ff00")

    def _boom(*a, **k):
        raise OSError("simulated audit append failure")

    monkeypatch.setattr("tcip_mcp.audit.record_event_or_raise", _boom)

    with pytest.raises(OSError):
        update_lineage("e3", dataset_fingerprint="DIFFERENT", predictions="w.pt")

    # The legitimate update landed despite the identity-refusal's own audit line failing.
    assert ts.read(exp.lineage_key("e3"))["predictions"] == "w.pt"
    assert ts.read(exp.lineage_key("e3"))["dataset_fingerprint"] == "ff00"  # still not backfilled


def test_compare_experiments_surfaces_shared_fingerprint(exp_dir):
    create_experiment("a", {}, dataset_id="1", dataset_fingerprint="v1:ff")
    create_experiment("b", {}, dataset_id="1", dataset_fingerprint="v1:ff")
    assert compare_experiments(["a", "b"])["same_dataset_fingerprint"] is True
    create_experiment("c", {}, dataset_id="2", dataset_fingerprint="v1:ee")
    assert compare_experiments(["a", "c"])["same_dataset_fingerprint"] is False


def test_compare_experiments_mixed_none_fingerprint_is_unknown_not_same(exp_dir):
    """One run with a known fingerprint compared against a bespoke/imageless run (None) must
    report unknown identity, not a false apples-to-apples True: the two demonstrably did not
    train on the same (known) data."""
    create_experiment("a", {}, dataset_id="1", dataset_fingerprint="ff")
    create_experiment("b", {})  # bespoke/imageless -> no recorded fingerprint
    assert compare_experiments(["a", "b"])["same_dataset_fingerprint"] is None


def test_dataset_identity_helper_registered_vs_bespoke(tmp_path):
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    _make_dataset(tmp_path)
    reg = register_dataset(str(tmp_path), crop="currant")
    ds_id, fp = dataset_identity({"images_dir": str(tmp_path / "images" / "2-11-26")})
    assert ds_id == reg["id"] and fp == reg["fingerprint"]
    # bespoke / imageless run -> no fabricated identity
    assert dataset_identity({}) == (None, None)


def test_dataset_identity_fingerprint_io_error_degrades_to_none(tmp_path, monkeypatch):
    """A fingerprint read failure (a locked/removed image mid-scan) must degrade to an honest
    None, not raise: raising here propagates out of launch_training's tracking try/except and
    silently drops the whole experiment record (lineage/status/split.json) for a run that still
    trains, which is strictly worse than losing only the fingerprint."""
    import tcip_mcp.pipelines.data.dataset_fingerprint as dataset_fingerprint_mod
    from tcip_mcp.pipelines.data.split_construction import dataset_identity

    _make_dataset(tmp_path)

    def _raise(_root):
        raise OSError("simulated I/O error mid-scan")

    # dataset_identity does `from tcip_mcp.pipelines.data.dataset_fingerprint import
    # dataset_fingerprint` locally at call time, so it must be patched at the source module.
    monkeypatch.setattr(dataset_fingerprint_mod, "dataset_fingerprint", _raise)
    ds_id, fp = dataset_identity({"images_dir": str(tmp_path / "images" / "2-11-26")})
    assert fp is None
    assert ds_id is None  # no dataset.json registered in this fixture


def test_persist_split_manifest_records_identity(exp_dir):
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    create_experiment("e1", {})

    class _DS:
        stems = ["a", "b"]

    persist_split_manifest("e1", _DS(), None, {"labels_dir": ""},
                            dataset_id="x", dataset_fingerprint="yz")
    split = ts.read(exp.split_key("e1"))
    assert split["dataset_id"] == "x" and split["dataset_fingerprint"] == "yz"
    assert split["train"] == ["a", "b"]  # membership still recorded beside the identity

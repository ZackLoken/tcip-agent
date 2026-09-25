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
from tcip_mcp.pipelines.data.split_construction import recorded_side


@pytest.fixture
def exp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "EXPERIMENTS_DIR", tmp_path / "experiments")
    return tmp_path / "experiments"


SUBJECT = "bud"


def _make_dataset(root: Path) -> tuple[Path, Path]:
    """One labeled image under one capture date; answers ``(images_dir, labels_dir)``."""
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import subject_registry
    from tcip_mcp.subject_registry import SubjectRegistry, Subject

    images_dir, labels_dir = root / "images" / "2-11-26", root / "annotations" / "2-11-26"
    images_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(images_dir / "img_000.jpg")
    labels_dir.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(str(labels_dir / "img_000.json"),
                              [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))], 32, 32)
    subject_registry.write_registry(root / "subjects.json",
                                  SubjectRegistry(subjects=(Subject(name=SUBJECT),)))
    return images_dir, labels_dir


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


def test_update_lineage_names_a_refused_identity_field_and_applies_the_legitimate_updates(exp_dir):
    """A dropped identity update is named under ``refused`` in the result, writes no audit line,
    and the call's other, legitimate updates still land."""
    create_experiment("e3", {}, dataset_id="abc123", dataset_fingerprint="ff00")

    result = update_lineage("e3", dataset_fingerprint="DIFFERENT", predictions="w.pt")

    assert result["refused"] == ["dataset_fingerprint"]
    assert ts.read(exp.lineage_key("e3"))["predictions"] == "w.pt"
    assert ts.read(exp.lineage_key("e3"))["dataset_fingerprint"] == "ff00"  # still not backfilled


def test_compare_experiments_surfaces_shared_fingerprint(exp_dir):
    create_experiment("a", {}, dataset_id="1", dataset_fingerprint="v1:ff")
    create_experiment("b", {}, dataset_id="1", dataset_fingerprint="v1:ff")
    assert compare_experiments(["a", "b"], stale_seconds=600.0)["same_dataset_fingerprint"] is True
    create_experiment("c", {}, dataset_id="2", dataset_fingerprint="v1:ee")
    assert compare_experiments(["a", "c"], stale_seconds=600.0)["same_dataset_fingerprint"] is False


def test_compare_experiments_mixed_none_fingerprint_is_unknown_not_same(exp_dir):
    """One run with a known fingerprint compared against a bespoke/imageless run (None) must
    report unknown identity, not a false apples-to-apples True: the two demonstrably did not
    train on the same (known) data."""
    create_experiment("a", {}, dataset_id="1", dataset_fingerprint="ff")
    create_experiment("b", {})  # bespoke/imageless -> no recorded fingerprint
    assert compare_experiments(["a", "b"], stale_seconds=600.0)["same_dataset_fingerprint"] is None


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


def test_persist_run_partition_records_identity(tmp_path, exp_dir):
    """The dataset identity rides beside the membership the producer named, in one record.

    The membership comes from ``auto_train_val``'s own partition, the only thing this writer
    records members from: it reads no loader, so a run's members are what a producer admitted.
    """
    from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

    images_dir, labels_dir = _make_dataset(tmp_path)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                "subject": SUBJECT}
    create_experiment("e1", {"data": data_cfg})
    _train_ds, _val_ds, partition = auto_train_val("detection", data_cfg, None)

    persist_run_partition("e1", data_cfg, dataset_id="x", dataset_fingerprint="yz",
                          partition=partition)
    split = ts.read(exp.split_key("e1"))
    assert split["dataset_id"] == "x" and split["dataset_fingerprint"] == "yz"
    assert split["members"] == partition  # membership beside the identity
    assert recorded_side(split["members"], "train"), (
        "the fixture admits samples, so the record names them")

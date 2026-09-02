"""freeze_split_manifest: a finished run's own drawn train/val partition, frozen into a
split_manifest record a later run can bind to.

Reuses test_split_manifest_binding.py's dataset fixture and builds a real drawn split through
the same producers that file's own tests exercise directly (auto_train_val + persist_split_
manifest), rather than restating either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import create_experiment
from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map
from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_split_manifest

from tests.test_split_manifest_binding import DATES, SUBJECT, _two_subject_two_date_dataset

# freeze_split_manifest is imported inside each test body, not here, so a tree without it
# still collects this module and reaches each test's own assertion.

BUILDER = "tests.bespoke_models:build_bespoke_detection"


def _real_drawn_experiment(
    root: Path, experiment_id: str, *, date: str = DATES[0], subject: str = SUBJECT,
    attribute: str | None = None, val_images_dir: str | None = None, auto_val: bool = True,
) -> dict:
    """Draws a real train/val split over ``root``'s own fixture dataset (through auto_train_val,
    the identical function a training run's own draw calls) and persists it as ``experiment_id``'s
    ``split.json`` (through persist_split_manifest, the one writer), plus a durable config the
    real subprocess worker would have stamped ``subject``/``labels_dir``/``images_dir``/``id_map``
    onto. Returns the resolved ``data`` section used.
    """
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    data_cfg: dict = {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                      "subject": subject, "attribute": attribute, "auto_val": auto_val}
    if val_images_dir is not None:
        data_cfg["val_images_dir"] = val_images_dir
    # data_cfg keeps the caller's raw attribute; the registry lookup needs "no attribute" as None.
    _reg, id_map = resolve_registry_id_map(str(labels_dir), subject, attribute or None)

    config = {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {**data_cfg, "id_map": id_map},
    }
    create_experiment(experiment_id, config)

    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg,
                           dataset_id=None, dataset_fingerprint=None, label_digests=label_digests)
    return data_cfg


# -- admits valid work: freeze, read back, bind a second run ------------------------


def test_freeze_split_manifest_round_trips_through_a_real_bind(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.tools.data_tools import freeze_split_manifest, read_split_manifest_dir
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-src")

    result = freeze_split_manifest("exp-src")

    assert "error" not in result, result
    manifest_dir = result["manifest_dir"]
    assert manifest_dir == str(root / "splits" / "frozen-exp-src")
    assert "calibration" in result["note"] and "refuse" in result["note"]

    manifest = read_split_manifest_dir(manifest_dir)
    assert manifest["origin"]["experiment_id"] == "exp-src"
    date_block = manifest["members"][DATES[0]]
    assert date_block["images_root"] == str(root / "images" / DATES[0])
    assert manifest["splits"]["calibration"] == []
    train_stems = {member_identity(DATES[0], s) for s in ("a", "b", "c", "d", "e", "f")}
    assert set(manifest["splits"]["train"]) | set(manifest["splits"]["val"]) <= train_stems
    assert manifest["splits"]["train"] and manifest["splits"]["val"]

    second_cfg: dict[str, Any] = {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {"images_dir": str(root / "images" / DATES[0]),
                 "labels_dir": str(root / "annotations" / DATES[0]), "subject": SUBJECT,
                 "split": {"manifest_dir": manifest_dir}},
    }
    assert manifest_compatibility(second_cfg, manifest, manifest_dir) == []

    from tcip_mcp.pipelines.data.splits import manifest_scope_issues

    scope_issues, scope_narrowing = manifest_scope_issues(
        manifest, subject=SUBJECT, attribute=None, date=DATES[0],
        images_dir=str(root / "images" / DATES[0]), label="data.images_dir",
        manifest_dir=manifest_dir,
    )
    assert scope_issues == []
    assert scope_narrowing is not None

    second_data_cfg = dict(second_cfg["data"])
    train_ds, val_ds, label_digests = auto_train_val("detection", second_data_cfg, None)
    assert label_digests is not None
    assert len(train_ds) > 0 and len(val_ds) > 0


def test_freeze_split_manifest_from_an_empty_string_attribute_run_binds(tmp_path: Path):
    """A run whose durable config carries ``data.attribute=""`` (an explicit empty string, not
    ``None``) freezes a manifest a later, attribute-unscoped run still binds to: the frozen
    ``attribute`` is normalized on write, and the scope check normalizes both sides of the
    comparison, so neither reads the checkpoint's own ``""`` as a distinct scope from the
    manifest's ``None``."""
    from tcip_mcp.tools.data_tools import freeze_split_manifest, read_split_manifest_dir
    from tcip_mcp.tools.training_tools import manifest_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-empty-attribute", attribute="")

    result = freeze_split_manifest("exp-empty-attribute")
    assert "error" not in result, result

    manifest = read_split_manifest_dir(result["manifest_dir"])
    assert manifest["attribute"] is None

    second_cfg: dict[str, Any] = {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {"images_dir": str(root / "images" / DATES[0]),
                 "labels_dir": str(root / "annotations" / DATES[0]), "subject": SUBJECT,
                 "split": {"manifest_dir": result["manifest_dir"]}},
    }
    assert manifest_compatibility(second_cfg, manifest, result["manifest_dir"]) == []

    train_ds, val_ds, label_digests = auto_train_val(
        "detection", dict(second_cfg["data"]), None)
    assert label_digests is not None
    assert len(train_ds) > 0 and len(val_ds) > 0


def test_freeze_split_manifest_rekeys_an_explicit_group_key_map_to_identities(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT,
               "split": {"group_key_map": {s: "g1" for s in ("a", "b", "c")}
                        | {s: "g2" for s in ("d", "e", "f")}}}
    _reg, id_map = resolve_registry_id_map(str(labels_dir), SUBJECT, None)
    create_experiment("exp-explicit-map", {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {**data_cfg, "id_map": id_map},
    })
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    persist_split_manifest("exp-explicit-map", train_ds, val_ds, data_cfg,
                           label_digests=label_digests)

    result = freeze_split_manifest("exp-explicit-map")
    assert "error" not in result, result

    from tcip_mcp.tools.data_tools import read_split_manifest_dir
    manifest = read_split_manifest_dir(result["manifest_dir"])
    assert manifest["group_by"] == "explicit_map"
    for identity in manifest["group_key_map"]:
        assert identity.startswith(f"{DATES[0]}/")


# -- refusals -----------------------------------------------------------------------


def test_freeze_split_manifest_refuses_no_split_record(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    create_experiment("exp-no-split", {"data": {"images_dir": str(root / "images" / DATES[0])}})

    result = freeze_split_manifest("exp-no-split")
    assert "error" in result and "no split record" in result["error"]


def test_freeze_split_manifest_refuses_a_bound_run(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest, make_splits

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    bound_manifest_dir = tmp_path / "src-manifest"
    make_result = make_splits(str(root), output_path=str(bound_manifest_dir), subject=SUBJECT,
                              seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in make_result, make_result

    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT,
               "split": {"manifest_dir": str(bound_manifest_dir)}}
    _reg, id_map = resolve_registry_id_map(str(labels_dir), SUBJECT, None)
    create_experiment("exp-bound", {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {**data_cfg, "id_map": id_map},
    })
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    persist_split_manifest("exp-bound", train_ds, val_ds, data_cfg, label_digests=label_digests)

    result = freeze_split_manifest("exp-bound")
    assert "error" in result and "bound" in result["error"]


def test_freeze_split_manifest_refuses_a_spatial_split(tmp_path: Path):
    """The record's own ``group_by`` names a spatial split (``spatial_strip``, region
    identities, never bare stems): the one field freeze_split_manifest's spatial refusal reads,
    hand-set here rather than through a tiled single-source fixture, since that field is the
    whole of what the refusal inspects."""
    import tcip_store as ts
    from tcip_mcp.experiments import split_key
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-spatial")

    split = ts.read(split_key("exp-spatial"))
    ts.replace(split_key("exp-spatial"), {**split, "group_by": "spatial_strip"})

    result = freeze_split_manifest("exp-spatial")
    assert "error" in result and "spatial" in result["error"]


def test_freeze_split_manifest_refuses_no_group_by(tmp_path: Path):
    """A split record with no group_by at all (predating the field): freeze_split_manifest
    never defaults it to 'stem'."""
    import tcip_store as ts
    from tcip_mcp.experiments import split_key
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-no-group-by")

    split = ts.read(split_key("exp-no-group-by"))
    ts.replace(split_key("exp-no-group-by"), {**split, "group_by": None})

    result = freeze_split_manifest("exp-no-group-by")
    assert "error" in result and "group_by" in result["error"]


def test_freeze_split_manifest_refuses_no_dataset_hash(tmp_path: Path):
    """A split record with no dataset_hash at all: freeze_split_manifest cannot check the
    labels have not moved, so it refuses rather than skipping the check."""
    import tcip_store as ts
    from tcip_mcp.experiments import split_key
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-no-hash")

    split = ts.read(split_key("exp-no-hash"))
    ts.replace(split_key("exp-no-hash"), {**split, "dataset_hash": None})

    result = freeze_split_manifest("exp-no-hash")
    assert "error" in result and "dataset_hash" in result["error"]


def test_freeze_split_manifest_refuses_an_external_validation_run(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    val_root = tmp_path / "external_val"
    (val_root).mkdir()
    from PIL import Image
    Image.new("RGB", (64, 64)).save(val_root / "z.jpg")

    _real_drawn_experiment(root, "exp-external", val_images_dir=str(val_root))

    result = freeze_split_manifest("exp-external")
    assert "error" in result and "external" in result["error"].lower() \
        or "val_images_dir" in result["error"]


def test_freeze_split_manifest_refuses_an_empty_val_side(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-no-val", auto_val=False)

    result = freeze_split_manifest("exp-no-val")
    assert "error" in result and "validation" in result["error"]


def test_freeze_split_manifest_refuses_a_task_it_does_not_admit(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT}
    create_experiment("exp-wrong-task", {
        "model_source": {"builder": BUILDER, "task": "semantic_seg"}, "data": data_cfg,
    })
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    persist_split_manifest("exp-wrong-task", train_ds, val_ds, data_cfg,
                           label_digests=label_digests)

    result = freeze_split_manifest("exp-wrong-task")
    assert "error" in result and "semantic_seg" in result["error"]


def test_freeze_split_manifest_refuses_a_config_missing_id_map(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT}
    create_experiment("exp-no-id-map", {
        "model_source": {"builder": BUILDER, "task": "detection"}, "data": data_cfg,
    })
    train_ds, val_ds, label_digests = auto_train_val("detection", data_cfg, None)
    persist_split_manifest("exp-no-id-map", train_ds, val_ds, data_cfg,
                           label_digests=label_digests)

    result = freeze_split_manifest("exp-no-id-map")
    assert "error" in result and "id_map" in result["error"]


def test_freeze_split_manifest_refuses_labels_changed_since_the_run(tmp_path: Path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-stale-labels")

    labels_dir = root / "annotations" / DATES[0]
    json_io.write_annotations(
        labels_dir / "a.json", [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))], 64, 64,
        keep_empty=True,
    )

    result = freeze_split_manifest("exp-stale-labels")
    assert "error" in result and "changed" in result["error"]


def test_freeze_split_manifest_refuses_when_a_manifest_already_exists_at_the_output(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_split_manifest

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-first")
    first = freeze_split_manifest("exp-first")
    assert "error" not in first, first

    _real_drawn_experiment(root, "exp-second")
    second = freeze_split_manifest("exp-second", output_path=first["manifest_dir"])
    assert "error" in second and "already exists" in second["error"]

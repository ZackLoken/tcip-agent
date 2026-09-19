"""freeze_selection: a finished run's own drawn train/val partition, frozen into a selection a
later run can bind to.

Reuses test_selection_binding.py's dataset fixture and builds a real drawn split through the same
producers that file's own tests exercise directly (auto_train_val + persist_run_partition),
rather than restating either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import create_experiment
from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map
from tcip_mcp.pipelines.data.selection import read_selection
from tcip_mcp.pipelines.data.split_construction import auto_train_val, persist_run_partition

from tests.test_selection_binding import DATES, SUBJECT, _two_subject_two_date_dataset

BUILDER = "tests.bespoke_models:build_bespoke_detection"


def _real_drawn_experiment(
    root: Path, experiment_id: str, *, date: str = DATES[0], subject: str = SUBJECT,
    attribute: str | None = None, val_images_dir: str | None = None, auto_val: bool = True,
) -> dict:
    """Draws a real train/val split over ``root``'s own fixture dataset (through auto_train_val,
    the identical function a training run's own draw calls) and persists it as ``experiment_id``'s
    ``split.json`` (through persist_run_partition, the one writer), plus a durable config the
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

    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    persist_run_partition(experiment_id, train_ds, val_ds, data_cfg,
                          dataset_id=None, dataset_fingerprint=None, partition=partition)
    return data_cfg


# -- admits valid work: freeze, read back, bind a second run -------------------


def test_freeze_selection_round_trips_through_a_real_bind(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import selection_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-src")

    result = freeze_selection("exp-src")

    assert "error" not in result, result
    selection_dir = result["selection_dir"]
    assert selection_dir == str(root / "splits" / "frozen-exp-src")
    assert "calibration" in result["note"] and "refuse" in result["note"]

    frozen = read_selection(selection_dir)
    assert frozen.origin["experiment_id"] == "exp-src"
    assert frozen.counts()["calibration"] == 0
    assert frozen.counts()["train"] and frozen.counts()["val"]
    assert {Path(s.ground_truth).stem for s in frozen.samples} <= set("abcdef")
    for sample in frozen.samples:
        assert Path(sample.source).parent == root / "images" / DATES[0]
        assert Path(sample.ground_truth).is_file()

    second_cfg: dict[str, Any] = {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {"split": {"selection_dir": selection_dir}},
    }
    assert selection_compatibility(second_cfg, frozen, selection_dir) == []

    train_ds, val_ds, partition = auto_train_val(
        "detection", dict(second_cfg["data"]), None)
    assert partition is not None
    assert len(train_ds) > 0 and len(val_ds) > 0


def test_freeze_selection_from_an_empty_string_attribute_run_binds(tmp_path: Path):
    """A run whose durable config carries ``data.attribute=""`` (an explicit empty string, not
    ``None``) freezes a selection a later, attribute-unscoped run still binds to: the frozen
    ``attribute`` is normalized on write, so no reader has to read one form as the other."""
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import selection_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-empty-attribute", attribute="")

    result = freeze_selection("exp-empty-attribute")
    assert "error" not in result, result

    frozen = read_selection(result["selection_dir"])
    assert frozen.attribute is None

    second_cfg: dict[str, Any] = {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {"split": {"selection_dir": result["selection_dir"]}},
    }
    assert selection_compatibility(second_cfg, frozen, result["selection_dir"]) == []

    train_ds, val_ds, partition = auto_train_val("detection", dict(second_cfg["data"]), None)
    assert partition is not None
    assert len(train_ds) > 0 and len(val_ds) > 0


def test_freeze_selection_carries_an_explicit_group_key_map_onto_its_samples(tmp_path: Path):
    """The run drew under an agent-supplied map; the frozen selection records each sample's own
    key from it, so a later bind groups by what was drawn rather than re-resolving a policy.

    The map is keyed by member identity, the key every producer of one uses (``draw_splits`` and
    ``tcip plant-aware-group-splits``), since a stem names one image only within one capture date.
    """
    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT,
               "split": {"group_key_map":
                        {member_identity(DATES[0], s): "g1" for s in ("a", "b", "c")}
                        | {member_identity(DATES[0], s): "g2" for s in ("d", "e", "f")}}}
    _reg, id_map = resolve_registry_id_map(str(labels_dir), SUBJECT, None)
    create_experiment("exp-explicit-map", {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {**data_cfg, "id_map": id_map},
    })
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    persist_run_partition("exp-explicit-map", train_ds, val_ds, data_cfg, partition=partition)

    result = freeze_selection("exp-explicit-map")
    assert "error" not in result, result

    frozen = read_selection(result["selection_dir"])
    assert frozen.group_by == "explicit_map"
    assert {s.group for s in frozen.samples} <= {"g1", "g2"}
    by_group = {s.group: s.side for s in frozen.samples}
    assert len(by_group) == len({s.group for s in frozen.samples})


def test_freeze_selection_carries_the_plain_paths_own_explicit_map(tmp_path: Path):
    """The plain split path draws whatever the drawn geometry path does not: a run naming its own
    assembled COCO takes it. Its explicit map is resolved and recorded through the same derivation
    every producer uses, so freezing such a run reads the keys it wrote rather than a second
    spelling of them."""
    import json

    from tcip_mcp.pipelines.data.splits import member_identity
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    stems = sorted(p.stem for p in labels_dir.glob("*.json"))
    coco = {
        "images": [{"id": i, "file_name": f"{stem}.jpg", "width": 64, "height": 64}
                   for i, stem in enumerate(stems)],
        "annotations": [{"id": i, "image_id": i, "category_id": 0, "iscrowd": 0,
                         "bbox": [4, 4, 16, 16], "area": 256} for i, _stem in enumerate(stems)],
        "categories": [{"id": 0, "name": SUBJECT}],
    }
    coco_path = tmp_path / "assembled.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    _reg, id_map = resolve_registry_id_map(str(labels_dir), SUBJECT, None)
    group_key_map = {member_identity(DATES[0], stem): f"g{index % 2}"
                     for index, stem in enumerate(stems)}
    data_cfg: dict[str, Any] = {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT,
        "label_format": "coco", "coco_json": str(coco_path),
        "split": {"val_ratio": 0.5, "seed": 1, "group_key_map": group_key_map},
    }
    create_experiment("exp-plain-map", {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {**data_cfg, "id_map": id_map},
    })
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    assert val_ds is not None
    persist_run_partition("exp-plain-map", train_ds, val_ds, data_cfg, partition=partition)

    frozen = freeze_selection("exp-plain-map")
    assert "error" not in frozen, frozen

    selection = read_selection(frozen["selection_dir"])
    assert selection.group_by == "explicit_map"
    assert {s.group for s in selection.samples} <= {"g0", "g1"}


# -- refusals ------------------------------------------------------------------


def test_freeze_selection_refuses_no_split_record(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    create_experiment("exp-no-split", {"data": {"images_dir": str(root / "images" / DATES[0])}})

    result = freeze_selection("exp-no-split")
    assert "error" in result and "no split record" in result["error"]


def _bound_run(root: Path, tmp_path: Path, experiment_id: str, **split_extra) -> None:
    from tcip_mcp.tools.data_tools import draw_splits

    selection_dir = tmp_path / f"src-{experiment_id}"
    drawn = draw_splits(str(root), output_path=str(selection_dir), subject=SUBJECT,
                        seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in drawn, drawn

    labels_dir = root / "annotations" / DATES[0]
    data_cfg = {"split": {"selection_dir": str(selection_dir), **split_extra}}
    _reg, id_map = resolve_registry_id_map(str(labels_dir), SUBJECT, None)
    create_experiment(experiment_id, {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {**data_cfg, "id_map": id_map},
    })
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    persist_run_partition(experiment_id, train_ds, val_ds, data_cfg, partition=partition)


def test_freeze_selection_refuses_a_bound_run(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _bound_run(root, tmp_path, "exp-bound")

    result = freeze_selection("exp-bound")
    assert "error" in result and "bound" in result["error"]


def test_freeze_selection_refuses_a_redrawn_bound_run_naming_the_reproduction(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _bound_run(root, tmp_path, "exp-redrawn-bound",
               redraw_within_selection=True, seed=11)

    result = freeze_selection("exp-redrawn-bound")
    assert "error" in result
    assert "redrew" in result["error"]
    assert "redraw_within_selection" in result["error"]


def test_freeze_selection_refuses_a_spatial_split(tmp_path: Path):
    """The record's own ``group_by`` names a spatial split (``spatial_strip``, region
    identities, never bare stems): the one field freeze_selection's spatial refusal reads,
    hand-set here rather than through a tiled single-source fixture, since that field is the
    whole of what the refusal inspects."""
    import tcip_store as ts
    from tcip_mcp.experiments import split_key
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-spatial")

    split = ts.read(split_key("exp-spatial"))
    ts.replace(split_key("exp-spatial"), {**split, "group_by": "spatial_strip"})

    result = freeze_selection("exp-spatial")
    assert "error" in result and "spatial" in result["error"]


def test_freeze_selection_refuses_no_group_by(tmp_path: Path):
    """A split record with no group_by at all: freeze_selection never defaults it to 'stem'."""
    import tcip_store as ts
    from tcip_mcp.experiments import split_key
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-no-group-by")

    split = ts.read(split_key("exp-no-group-by"))
    ts.replace(split_key("exp-no-group-by"), {**split, "group_by": None})

    result = freeze_selection("exp-no-group-by")
    assert "error" in result and "group_by" in result["error"]


def test_freeze_selection_refuses_no_dataset_hash(tmp_path: Path):
    """A split record with no dataset_hash at all: freeze_selection cannot check the labels have
    not moved, so it refuses rather than skipping the check."""
    import tcip_store as ts
    from tcip_mcp.experiments import split_key
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-no-hash")

    split = ts.read(split_key("exp-no-hash"))
    ts.replace(split_key("exp-no-hash"), {**split, "dataset_hash": None})

    result = freeze_selection("exp-no-hash")
    assert "error" in result and "dataset_hash" in result["error"]


def test_freeze_selection_refuses_an_external_validation_run(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    val_root = tmp_path / "external_val"
    val_root.mkdir()
    from PIL import Image
    Image.new("RGB", (64, 64)).save(val_root / "z.jpg")

    _real_drawn_experiment(root, "exp-external", val_images_dir=str(val_root))

    result = freeze_selection("exp-external")
    assert "error" in result
    assert "external" in result["error"].lower() or "val_images_dir" in result["error"]


def test_freeze_selection_refuses_an_empty_val_side(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-no-val", auto_val=False)

    result = freeze_selection("exp-no-val")
    assert "error" in result and "validation" in result["error"]


def test_freeze_selection_refuses_a_task_it_does_not_admit(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT}
    create_experiment("exp-wrong-task", {
        "model_source": {"builder": BUILDER, "task": "semantic_seg"}, "data": data_cfg,
    })
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    persist_run_partition("exp-wrong-task", train_ds, val_ds, data_cfg, partition=partition)

    result = freeze_selection("exp-wrong-task")
    assert "error" in result and "semantic_seg" in result["error"]


def test_freeze_selection_refuses_a_config_missing_id_map(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    images_dir, labels_dir = root / "images" / DATES[0], root / "annotations" / DATES[0]
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": SUBJECT}
    create_experiment("exp-no-id-map", {
        "model_source": {"builder": BUILDER, "task": "detection"}, "data": data_cfg,
    })
    train_ds, val_ds, partition = auto_train_val("detection", data_cfg, None)
    persist_run_partition("exp-no-id-map", train_ds, val_ds, data_cfg, partition=partition)

    result = freeze_selection("exp-no-id-map")
    assert "error" in result and "id_map" in result["error"]


def test_freeze_selection_refuses_labels_changed_since_the_run(tmp_path: Path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-stale-labels")

    labels_dir = root / "annotations" / DATES[0]
    json_io.write_annotations(
        labels_dir / "a.json", [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))], 64, 64,
        keep_empty=True,
    )

    result = freeze_selection("exp-stale-labels")
    assert "error" in result and "changed" in result["error"]


def test_freeze_selection_refuses_when_a_selection_already_exists_at_the_output(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(root, "exp-first")
    first = freeze_selection("exp-first")
    assert "error" not in first, first

    _real_drawn_experiment(root, "exp-second")
    second = freeze_selection("exp-second", output_path=first["selection_dir"])
    assert "error" in second and "already exists" in second["error"]

"""Membership for the tasks a per-image label document does not answer for.

A semantic segmentation run's ground truth is a mask raster beside its image; a classification,
ordinal or regression run's is one row of a table. Both are named by the platform's own producer,
drawn into a selection the same way a label-document draw is, and read back off the loaders the
run actually builds rather than off a record written beside them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.data.selection import read_selection  # noqa: E402
from tcip_mcp.pipelines.data.split_construction import (  # noqa: E402
    auto_train_val, persist_run_partition, recorded_side as _side,
)
from tcip_mcp.tools.data_tools import draw_splits  # noqa: E402

STEMS = ("a", "b", "c", "d", "e", "f", "g", "h")



def _mask_dataset(root: Path) -> tuple[Path, Path]:
    """A dataset whose ground truth is one ``<stem>.png`` mask per image, each carrying a distinct
    foreground block so a mask read back can be told from its neighbor's."""
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{stem}.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[: index + 1, : index + 1] = 1
        Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")
    return images_dir, masks_dir


def _table_dataset(root: Path) -> tuple[Path, Path]:
    """A dataset whose ground truth is one table row per image, each row a distinct label."""
    images_dir, csv_path = root / "images", root / "labels.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, stem in enumerate(STEMS):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{stem}.png")
        rows.append((stem, index % 3))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)
    return images_dir, csv_path


def _drawn(root: Path, ground_truth: Path, out: Path, *, seed: int = 4):
    result = draw_splits(str(root), output_path=str(out), ground_truth=str(ground_truth),
                         seed=seed, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25,
                         group_by="stem")
    assert "error" not in result, result
    return read_selection(out)


# -- a bound run trains over exactly the selection's own samples ---------------


def test_a_bound_semantic_seg_run_trains_over_exactly_its_selections_samples(tmp_path: Path):
    """The loaders hold exactly the samples the selection recorded on each side, each reading the
    mask it names, and the held-out calibration side builds neither loader."""
    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, masks_dir, out)

    data_cfg = {"split": {"selection_dir": str(out)}, "num_classes": 2}
    train_ds, val_ds, partition = auto_train_val("semantic_seg", data_cfg, None)

    assert sorted(train_ds.stems) == sorted(s.identity for s in drawn.on("train"))
    assert sorted(val_ds.stems) == sorted(s.identity for s in drawn.on("val"))
    held_out = {s.identity for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    assert _side(partition, "train") == sorted(s.member for s in drawn.on("train"))


def test_a_bound_classification_run_trains_over_exactly_its_selections_samples(tmp_path: Path):
    """The same for a table: each side holds the rows the selection put on it, by row key."""
    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    train_ds, val_ds, partition = auto_train_val("classification", data_cfg, None)

    assert sorted(train_ds.stems) == sorted(s.identity for s in drawn.on("train"))
    assert sorted(val_ds.stems) == sorted(s.identity for s in drawn.on("val"))
    held_out = {s.identity for s in drawn.on("calibration")}
    assert held_out and not held_out & set(train_ds.stems + val_ds.stems)
    assert sorted(partition) == [str(csv_path)]


def test_a_directory_named_like_a_mask_is_admitted_by_neither_read_of_it(tmp_path: Path):
    """What a mask is is one predicate, so the admission that enumerates a directory and the
    re-admission that checks a recorded sample's own path cannot disagree: a directory named
    ``a.1.png`` is a mask to neither."""
    from tcip_mcp.pipelines.data.label_queries import admit, refuse_inadmissible_samples

    images_dir, masks_dir = tmp_path / "images", tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    for stem in ("a.1", "b"):
        Image.new("RGB", (16, 16), (10, 20, 30)).save(images_dir / f"{stem}.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(masks_dir / "b.png")
    (masks_dir / "a.1.png").mkdir()

    admitted = admit(images_dir, masks_dir)
    assert [record.member for record in admitted.records] == ["b"]
    # Admits valid work: what it did admit still admits when it is checked again.
    refuse_inadmissible_samples(
        admitted.samples({record.member: "train" for record in admitted.records}, lambda m: m))


# -- each sample reaches its own ground truth ----------------------------------


def test_a_mask_sample_reaches_its_own_mask(tmp_path: Path):
    """The mask a loader serves for one sample is the file that sample names, not one rebuilt from
    a directory and a stem: each fixture mask carries a distinct foreground area, so a mask served
    from the wrong sample reads as the wrong area."""
    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, masks_dir, out)

    train_ds, _val_ds, _partition = auto_train_val(
        "semantic_seg", {"split": {"selection_dir": str(out)}, "num_classes": 2}, None)

    by_identity = {s.identity: s for s in drawn.samples}
    for index, key in enumerate(train_ds.stems):
        sample = by_identity[key]
        expected = STEMS.index(sample.member) + 1
        _image, target = train_ds[index]
        assert int(target["masks"].sum()) == expected * expected
        assert Path(sample.ground_truth) == masks_dir / f"{sample.member}.png"


def test_a_row_key_sample_reaches_the_row_it_names(tmp_path: Path):
    """The label a loader serves for one sample is the row its ``row_key`` names in the table it
    names: each fixture row carries a distinct label, so a row read by position instead of by key
    reads as the wrong label."""
    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    train_ds, _val_ds, _partition = auto_train_val(
        "classification", {"split": {"selection_dir": str(out)}}, None)

    by_identity = {s.identity: s for s in drawn.samples}
    for index, key in enumerate(train_ds.stems):
        sample = by_identity[key]
        assert sample.row_key == sample.member
        assert Path(sample.ground_truth) == csv_path
        _image, target = train_ds[index]
        assert target["labels"] == STEMS.index(sample.row_key) % 3


# -- an unbound run's loaders answer for their own membership -------------------


def test_an_unbound_semantic_seg_run_reads_its_membership_off_its_own_samples(tmp_path: Path):
    """No selection: the run's own producer admits the masks beside the images and every loader is
    built from those samples, so membership is read off the loaders rather than off a record."""
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir), "num_classes": 2,
                "split": {"val_ratio": 0.25, "seed": 7}}

    train_ds, val_ds, partition = auto_train_val("semantic_seg", data_cfg, None)

    assert val_ds is not None
    members = {ds: {ds.member_of(key) for key in ds.stems} for ds in (train_ds, val_ds)}
    assert members[train_ds].isdisjoint(members[val_ds])
    assert members[train_ds] | members[val_ds] == set(STEMS)
    assert _side(partition, "train") == sorted(members[train_ds])
    assert _side(partition, "val") == sorted(members[val_ds])
    # The maps themselves, not what a fallback could answer: each loader indexes by its samples'
    # own source identities and holds that sample's own source, ground truth and member name.
    for ds in (train_ds, val_ds):
        assert ds.sample_sources is not None and ds.sample_ground_truth is not None
        assert ds.sample_members is not None
        assert set(ds.stems) == set(ds.sample_sources) == set(ds.sample_ground_truth)
        for key in ds.stems:
            assert ds.sample_sources[key] == key, "a sample is indexed by its own source"
            assert Path(ds.sample_sources[key]) == images_dir / f"{ds.sample_members[key]}.png"
            assert Path(ds.sample_ground_truth[key]) == masks_dir / \
                f"{ds.sample_members[key]}.png"


def test_an_unbound_regression_run_reads_its_membership_off_its_own_samples(tmp_path: Path):
    """The same for a table: the run's own producer admits the rows naming an image that exists,
    and each loader indexes exactly the rows it was handed."""
    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                "split": {"val_ratio": 0.25, "seed": 7}}

    train_ds, val_ds, partition = auto_train_val("regression", data_cfg, None)

    assert val_ds is not None
    members = {ds: {ds.member_of(key) for key in ds.stems} for ds in (train_ds, val_ds)}
    assert members[train_ds].isdisjoint(members[val_ds])
    assert members[train_ds] | members[val_ds] == set(STEMS)
    assert _side(partition, "train") == sorted(members[train_ds])
    assert partition[str(csv_path)]["val"] == sorted(members[val_ds])
    # The maps themselves, not what ``member_of``'s fallback could answer for a bare key.
    for ds in (train_ds, val_ds):
        assert ds.sample_sources is not None and ds.sample_ground_truth is not None
        assert ds.sample_members is not None
        assert set(ds.stems) == set(ds.sample_sources) == set(ds.sample_members)
        for key in ds.stems:
            assert ds.sample_sources[key] == key, "a sample is indexed by its own source"
            assert Path(ds.sample_ground_truth[key]) == csv_path
            assert Path(key) == images_dir / f"{ds.sample_members[key]}.png"


# -- a member's name is its row key, dots and all ------------------------------


def test_a_dotted_row_key_names_one_member_end_to_end(tmp_path: Path):
    """A row key is already the member's own name. Stripping a suffix from it would read ``a.1``
    and ``a.2`` as one member, collapsing two rows in the loaders' own membership, in the
    persisted partition, and in the leakage check that joins a calibration image against the
    training side by that name.
    """
    from tcip_mcp.experiments import create_experiment, read_run_partition
    from tcip_mcp.pipelines.data.datasets import record_stems_of
    from tcip_mcp.pipelines.operating_point import _train_disjointness

    root = tmp_path / "ds"
    images_dir, csv_path = root / "images", root / "labels.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    dotted = [f"a.{index}" for index in range(8)]
    rows = []
    for index, key in enumerate(dotted):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{key}.png")
        rows.append((key, index % 3))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        writer.writerows(rows)

    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                "split": {"val_ratio": 0.5, "seed": 3, "group_by": "stem"}}
    train_ds, val_ds, partition = auto_train_val("classification", data_cfg, None)

    # The loaders name every row apart, not one collapsed member.
    named = set(record_stems_of(train_ds) or []) | set(record_stems_of(val_ds) or [])
    assert named == set(dotted)
    assert len(record_stems_of(train_ds) or []) == train_ds.num_samples

    create_experiment("exp-dotted", {"data": data_cfg})
    persist_run_partition("exp-dotted", data_cfg, partition=partition)
    record = read_run_partition("exp-dotted")
    scope = record["members"][str(csv_path)]
    assert sorted(scope["train"] + scope["val"]) == sorted(dotted)
    assert sorted(scope["group_key_map"]) == sorted(dotted)

    # A calibration over one training row is that row leaking, and the check names it: with the
    # names collapsed, every row would answer for every other one instead.
    leaked = _train_disjointness(
        "exp-dotted", {scope["train"][0]}, set(), calibration_labels_dir=str(csv_path))
    assert leaked["leaked_groups"] == [scope["train"][0]]
    clean = _train_disjointness(
        "exp-dotted", {scope["val"][0]}, set(), calibration_labels_dir=str(csv_path))
    assert clean["leaked_groups"] == [] and clean["leaked_stems"] == []


# -- freezing a run whose ground truth is not a label document -----------------


def _frozen(experiment_id: str, task: str, data_cfg: dict) -> dict:
    """Train a run through the platform's own producer, persist its partition, and freeze it."""
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import freeze_selection

    create_experiment(experiment_id, {"model_source": {"builder": "m:f", "task": task},
                                      "data": data_cfg})
    _train, _val, partition = auto_train_val(task, data_cfg, None)
    persist_run_partition(experiment_id, data_cfg, partition=partition)
    return freeze_selection(experiment_id)


def test_a_mask_run_freezes_into_a_selection_its_bind_accepts(tmp_path: Path):
    """The staleness check freezing rests on is per member, against the digest the run's own
    producer recorded, so a mask run freezes: the frozen selection names each member's own mask
    and a later run binds to exactly that partition."""
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir), "num_classes": 2,
                "split": {"val_ratio": 0.25, "seed": 7}}

    result = _frozen("exp-mask-freeze", "semantic_seg", data_cfg)

    assert "error" not in result, result
    frozen = read_selection(result["selection_dir"])
    assert frozen.subject is None and frozen.id_map == {}
    assert {Path(s.ground_truth).suffix for s in frozen.samples} == {".png"}
    for sample in frozen.samples:
        assert Path(sample.ground_truth).parent == masks_dir
        assert Path(sample.ground_truth).is_file()

    train_ds, val_ds, partition = auto_train_val(
        "semantic_seg", {"split": {"selection_dir": result["selection_dir"]}, "num_classes": 2},
        None)
    assert sorted(train_ds.stems) == sorted(s.identity for s in frozen.on("train"))
    assert val_ds is not None
    assert result["train"] == len(frozen.on("train"))
    assert _side(partition, "train") == sorted(s.member for s in frozen.on("train"))


def test_a_table_run_freezes_into_a_selection_its_bind_accepts(tmp_path: Path):
    """The same for a table: the frozen selection names each member's own row in the table the
    run read, and a later run binds to exactly that partition."""
    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                "split": {"val_ratio": 0.25, "seed": 7}}

    result = _frozen("exp-table-freeze", "classification", data_cfg)

    assert "error" not in result, result
    frozen = read_selection(result["selection_dir"])
    assert {s.ground_truth for s in frozen.samples} == {str(csv_path)}
    assert all(s.row_key is not None for s in frozen.samples)

    train_ds, val_ds, _partition = auto_train_val(
        "classification", {"split": {"selection_dir": result["selection_dir"]}}, None)
    assert sorted(train_ds.stems) == sorted(s.identity for s in frozen.on("train"))
    assert val_ds is not None


def test_a_mask_edited_after_the_run_refuses_the_freeze_by_name(tmp_path: Path):
    """A member's mask replaced after the run digests differently from the ``at_run`` the record
    holds, so freezing refuses and names it rather than composing a selection over ground truth
    the run never saw."""
    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir), "num_classes": 2,
                "split": {"val_ratio": 0.25, "seed": 7}}
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.data_tools import freeze_selection

    create_experiment("exp-mask-moved", {"model_source": {"builder": "m:f",
                                                          "task": "semantic_seg"},
                                         "data": data_cfg})
    _train, _val, partition = auto_train_val("semantic_seg", data_cfg, None)
    persist_run_partition("exp-mask-moved", data_cfg, partition=partition)

    edited = np.zeros((16, 16), dtype=np.uint8)
    edited[:12, :12] = 1
    Image.fromarray(edited, mode="L").save(masks_dir / f"{STEMS[0]}.png")

    result = freeze_selection("exp-mask-moved")

    assert "error" in result and "changed since" in result["error"]
    assert STEMS[0] in result["error"]


# -- the calibration readers speak the scope the record states ------------------


def test_an_unchanged_table_reports_no_member_as_moved(tmp_path: Path):
    """The label-movement window recomputes each member's digest under the scope the run recorded
    it in. A table scope holds one file answering for every member, so reconstructing a per-image
    document path there would report every calibration member as changed the moment it is read.
    """
    from tcip_mcp.experiments import create_experiment, read_run_partition
    from tcip_mcp.pipelines.operating_point import _resolve_label_movement

    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    _train, _val, partition = auto_train_val("classification", data_cfg, None)
    create_experiment("exp-table-movement", {"data": data_cfg})
    persist_run_partition("exp-table-movement", data_cfg, partition=partition)
    record = read_run_partition("exp-table-movement")

    held_out = {s.member for s in drawn.on("calibration")}
    movement = _resolve_label_movement(
        record["members"][str(csv_path)]["label_digests"], held_out, str(csv_path), None,
        record["selection_binding"]["selection_sha256"])

    assert movement["labels_moved_draw_to_run"] == []
    assert movement["labels_moved_run_to_now"] == []
    assert movement["calibration_labels_moved"] == []


def test_an_edited_table_reports_its_members_as_moved(tmp_path: Path):
    """The same window still catches a real edit: the table rewritten after the run digests
    differently, so every member it answers for reads as moved."""
    from tcip_mcp.experiments import create_experiment, read_run_partition
    from tcip_mcp.pipelines.operating_point import _resolve_label_movement

    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    _train, _val, partition = auto_train_val("classification", data_cfg, None)
    create_experiment("exp-table-edited", {"data": data_cfg})
    persist_run_partition("exp-table-edited", data_cfg, partition=partition)
    record = read_run_partition("exp-table-edited")

    with open(csv_path, "a", newline="") as handle:
        csv.writer(handle).writerow(("later", 2))

    held_out = sorted(s.member for s in drawn.on("calibration"))
    movement = _resolve_label_movement(
        record["members"][str(csv_path)]["label_digests"], set(held_out), str(csv_path),
        None, record["selection_binding"]["selection_sha256"])

    assert movement["labels_moved_run_to_now"] == held_out


def test_a_scalar_calibration_names_the_table_scope_its_run_recorded(tmp_path, monkeypatch):
    """A scalar calibration reads its ground truth from a table, and the run's own partition
    records its members under that table as their scope. The door states that scope, which is what
    lets the selection check see the bound run's own validation rows; naming none would answer
    that no scope was given and let a calibration over the checkpoint's own selection side pass
    unseen. The predictor is a stand-in returning each row's recorded value, so what is exercised
    is the door's own provenance rather than a model's accuracy.
    """
    import torch

    import tcip_mcp.traits as traits
    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point
    from tcip_mcp.tools.model_tools import register_model
    from tcip_mcp.traits import TraitSpec

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    # One registered trait, so what this exercises is the door's provenance, not trait authoring.
    monkeypatch.setattr(
        traits, "get_trait_for",
        lambda name, project_root=None: TraitSpec(name=name, regression_skill_floor=0.0))

    root = tmp_path / "ds"
    images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    data_cfg = {"split": {"selection_dir": str(out)}}
    _train, _val, partition = auto_train_val("regression", data_cfg, None)
    create_experiment("exp-scalar-scope", {"data": data_cfg})
    persist_run_partition("exp-scalar-scope", data_cfg, partition=partition)

    checkpoint = tmp_path / "model_best.pt"
    torch.save({"model_state_dict": {}, "kind": "tcip_module"}, checkpoint)
    assert "error" not in register_model(
        name="table-scope", checkpoint_path=str(checkpoint), config={},
        project_path=str(tmp_path))

    by_row = {key: float(value) for key, value in
              (row.split(",") for row in
               csv_path.read_text(encoding="utf-8").splitlines()[1:])}

    class _RecordedValues:
        def predict_batch(self, sources):
            return [{"head0_values": [by_row[Path(s).stem]]} for s in sources]

    monkeypatch.setattr("tcip_mcp.pipelines.inference.predictor.build_predictor",
                        lambda *a, **kw: _RecordedValues())

    result = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="regression", checkpoint_path=str(checkpoint),
        images_dir=str(images_dir), csv_path=str(csv_path), criterion="r_squared",
        output_dir=str(tmp_path / "calib"), dataset_root=str(root),
        experiment_id="exp-scalar-scope", group_by="stem",
    )

    assert "error" not in result, result
    from tcip_mcp.pipelines.resolution import read_regression_operating_point_sidecar

    sidecar = read_regression_operating_point_sidecar(tmp_path / "calib")
    selection_check = sidecar["gate_evidence"]["selection_disjointness"]
    assert selection_check["applicable"] is True, selection_check
    # The calibration draws over every row, so the run's own validation rows are in it and named.
    assert selection_check["leaked_groups"] == sorted(s.member for s in drawn.on("val"))


def test_a_table_calibration_universe_holds_the_rows_the_draw_held_out(tmp_path: Path):
    """The universe is the selection's own calibration samples, whatever shape their ground truth
    is. Narrowing by the parent of a sample's ground-truth path would match none of a table's rows
    and report an empty universe; the scope the sample itself records matches, so a table's held-out
    rows are the universe a door measuring rows reads."""
    from tcip_mcp.pipelines.data.splits import selection_calibration_universe

    root = tmp_path / "ds"
    _images_dir, csv_path = _table_dataset(root)
    out = tmp_path / "m"
    drawn = _drawn(root, csv_path, out)

    stems, group_by, group_key_map, excluded, _counts, samples = \
        selection_calibration_universe(drawn, str(csv_path), min_foreground_groups={})

    held_out = sorted(s.member for s in drawn.on("calibration"))
    assert stems == held_out and sorted(samples) == held_out
    assert group_by == "explicit_map"
    assert sorted(group_key_map) == held_out
    assert drawn.scope.subject is None and drawn.scope.attribute is None
    assert excluded["excluded_training_stems"] == sorted(
        s.member for s in drawn.on("train"))
    assert all(Path(samples[stem].ground_truth) == csv_path for stem in stems)
    assert all(samples[stem].row_key == stem for stem in stems)


def _three_class_masks(root: Path) -> tuple[Path, Path]:
    """Four images and their masks, one of which alone reaches class 2: a run sizing each loader
    by the classes its own half happens to hold would size the two halves differently."""
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS[:4]):
        Image.new("RGB", (16, 16), (10 * index, 20, 30)).save(images_dir / f"{stem}.png")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:8, :8] = 1
        if index == 0:
            mask[8:, 8:] = 2
        Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")
    return images_dir, masks_dir


def test_a_runs_stated_class_count_reaches_both_its_loaders(tmp_path: Path):
    """The count a config states is the count both of one run's loaders are built at, and the
    loader's own refusal fires on the run path: a count that never left the config would let a
    train loader and a validation loader over one run's own halves disagree about the class
    space, and would leave a configured count that is too small to reach the refusal."""
    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    base = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
            "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}

    train_ds, val_ds, _partition = auto_train_val(
        "semantic_seg", {**base, "num_classes": 3, "split": dict(base["split"])}, None)
    assert val_ds is not None
    assert train_ds.num_classes == val_ds.num_classes == 3

    with pytest.raises(ValueError, match="num_classes"):
        auto_train_val(
            "semantic_seg", {**base, "num_classes": 2, "split": dict(base["split"])}, None)


def test_a_runs_unstated_class_count_is_read_once_for_both_its_loaders(tmp_path: Path):
    """A class reaching one side only still sizes both loaders: the count a config states none of
    is read off every sample the run was handed, before the split, so a training loader and a
    validation loader over one run's own halves cannot be built in two vocabularies."""
    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}

    train_ds, val_ds, partition = auto_train_val("semantic_seg", data_cfg, None)

    # The one mask reaching class 2 is on the validation side, which is the disagreement a
    # per-loader count would produce here.
    assert _side(partition, "val") == [STEMS[0]]
    assert val_ds is not None
    assert train_ds.num_classes == val_ds.num_classes == 3
    # And the run records what it resolved, so a reader after training takes the count from there.
    assert data_cfg["num_classes"] == 3
    assert data_cfg["num_channels"] == train_ds.expected_channels


def test_a_runs_metrics_are_reported_over_the_class_space_it_trains_in(tmp_path: Path):
    """The half being scored is never asked how many classes there are: a training half whose
    masks reach only class 1 is still measured over the three the run trains in, so the two halves
    of one run report metrics on one scale."""
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.evaluation import evaluate
    from tests import bespoke_models

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}

    train_ds, _val_ds, _partition = auto_train_val("semantic_seg", data_cfg, None)
    held = {int(np.array(Image.open(masks_dir / f"{stem}.png")).max())
            for stem in STEMS[1:4]}
    assert held == {1}, "the training half reaches only class 1, which is what this measures"

    model = bespoke_models.build_bespoke_semantic_seg(num_classes=train_ds.num_classes)
    loader = DataLoader(train_ds, batch_size=1, collate_fn=task_collate("semantic_seg"))
    result = evaluate(model, loader, torch.device("cpu"), "semantic_seg")

    assert sorted(result["per_class_iou"]) == [0, 1, 2]


def test_a_head_that_states_no_class_count_stops_the_measurement(tmp_path: Path):
    """The scale metrics are reported on is the head's own count and nothing else: a model whose
    head states none stops the evaluation where the count is read, rather than falling back to
    whatever the scored half happens to carry and reporting that as the run's scale."""
    import torch
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.evaluation import evaluate

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 3}}
    train_ds, _val_ds, _partition = auto_train_val("semantic_seg", data_cfg, None)

    class _CountlessSegModel(torch.nn.Module):
        """A segmentation model whose head states no class count at all."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, train_ds.num_classes, 1)
            self.heads = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, images, targets=None):
            logits = self.conv(images)
            if self.training and targets is not None:
                return {"head0_loss": logits.mean()}
            return {"head0_masks": logits.argmax(1)}

    loader = DataLoader(train_ds, batch_size=1, collate_fn=task_collate("semantic_seg"))

    with pytest.raises(AttributeError, match="num_classes"):
        evaluate(_CountlessSegModel(), loader, torch.device("cpu"), "semantic_seg")


def test_the_place_and_the_record_read_one_ground_truth_shape(tmp_path: Path):
    """The sniff over a place a config names and the read over a sample's own recorded ground
    truth answer with one shape, because the place asks the record's own rule of what it finds
    there: a table, a directory of masks and a directory of documents each read the same way from
    either side."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape
    from tcip_mcp.pipelines.data.selection import shape_of

    root = tmp_path / "ds"
    _images_dir, masks_dir = _mask_dataset(root)
    _table_images, csv_path = _table_dataset(tmp_path / "table")
    documents = root / "annotations"
    documents.mkdir(parents=True)
    (documents / "a.json").write_text("{}", encoding="utf-8")

    for place, member in ((csv_path, csv_path),
                          (masks_dir, masks_dir / f"{STEMS[0]}.png"),
                          (documents, documents / "a.json")):
        assert ground_truth_shape(str(place)) == shape_of(str(member), None), place


def test_a_mask_run_records_no_class_space_over_a_stale_config_one(tmp_path: Path):
    """A mask raster carries its own classes and no registry scopes it, so the run records no
    subject and no map: a stale scope a relaunched config carried would otherwise be stamped onto
    the checkpoint as the vocabulary this run trained in."""
    images_dir, masks_dir = _mask_dataset(tmp_path / "ds")
    data_cfg = {"images_dir": str(images_dir), "labels_dir": str(masks_dir), "num_classes": 2,
                "subject": "leaf", "attribute": "condition", "id_map": {"leaf": 0},
                "split": {"group_by": "stem", "val_ratio": 0.25, "seed": 5}}

    auto_train_val("semantic_seg", data_cfg, None)

    assert data_cfg["subject"] is None and data_cfg["attribute"] is None
    assert data_cfg["id_map"] is None


def test_a_mask_selections_held_out_side_redraws_without_a_subject(tmp_path: Path):
    """A subject scopes a document admission and nothing else, so a redraw over a mask
    selection's held-out side reads the selection's own scope and draws: requiring a subject
    ahead of the read would refuse a selection whose universe this door can measure."""
    from tcip_mcp.pipelines.data.splits import cal_holdout_scope_root
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    drawn = _drawn(root, masks_dir, tmp_path / "m")

    result = redraw_calibration_holdout(
        dataset_root=str(cal_holdout_scope_root(masks_dir)), labels_dir=str(masks_dir),
        images_dir=str(images_dir), selection_dir=str(tmp_path / "m"), reason="test redraw",
    )

    assert "error" not in result, result
    held_out = {s.member for s in drawn.on("calibration")}
    assert set(result["new_membership"]["calibration"]) | set(
        result["new_membership"]["holdout"]) == held_out


# -- two producers of one record, compared against each other ------------------


def test_the_bound_and_drawn_mask_routes_record_one_directory_the_same_way(tmp_path: Path):
    """Two producers of one fact, compared against each other rather than against a fixture.

    The same mask directory is admitted twice, once by a run bound to a selection drawn over it
    and once by a run that drew its own split. The two partition it differently, which is what
    each route is for; what they may not do is disagree about which members that directory holds,
    where it is, or what each member's ground truth digests to now.
    """
    from tcip_mcp.experiments import create_experiment, read_run_partition

    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    out = tmp_path / "m"
    _drawn(root, masks_dir, out)

    def persisted(experiment_id: str, data_cfg: dict) -> dict:
        _train, _val, partition = auto_train_val("semantic_seg", data_cfg, None)
        create_experiment(experiment_id, {"data": data_cfg})
        persist_run_partition(experiment_id, data_cfg, partition=partition)
        return read_run_partition(experiment_id)

    bound = persisted("exp-mask-bound",
                      {"split": {"selection_dir": str(out)}, "num_classes": 2})
    drawn = persisted("exp-mask-drawn",
                      {"images_dir": str(images_dir), "labels_dir": str(masks_dir),
                       "num_classes": 2, "split": {"val_ratio": 0.25, "seed": 7}})

    scope = str(masks_dir)
    assert sorted(bound["members"]) == sorted(drawn["members"]) == [scope]
    bound_block, drawn_block = bound["members"][scope], drawn["members"][scope]
    # The bound run holds out a calibration side the drawn one trains on, so the two agree on
    # every member they both name rather than on the union.
    bound_named = set(bound_block["train"]) | set(bound_block["val"])
    drawn_named = set(drawn_block["train"]) | set(drawn_block["val"])
    assert bound_named and drawn_named and bound_named <= drawn_named
    shared = bound_named & drawn_named
    assert {k: v for k, v in bound_block["label_digests"]["at_run"].items() if k in shared} == \
        {k: v for k, v in drawn_block["label_digests"]["at_run"].items() if k in shared}
    assert {k: v for k, v in bound_block["sources"].items() if k in shared} == \
        {k: v for k, v in drawn_block["sources"].items() if k in shared}


def test_a_document_run_and_a_mask_run_record_the_same_images_the_same_way(tmp_path: Path):
    """Two shapes of one producer, compared against each other rather than against a fixture.

    The same images are trained twice, once over a per-image label document each and once over a
    ``<stem>.png`` mask each, at the same seed and grouping policy. The two ground truths are
    genuinely different files, so the digests and the scopes differ; what the one producer may not
    do is name a different membership, a different partition, a differently shaped record or a
    different image for one shape than for the other.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.experiments import create_experiment, read_run_partition
    from tcip_mcp.subject_registry import SubjectRegistry, Subject, write_registry

    root = tmp_path / "ds"
    images_dir, masks_dir = _mask_dataset(root)
    labels_dir = root / "annotations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_registry(root / "subjects.json", SubjectRegistry(subjects=(Subject(name="leaf"),)))
    for stem in STEMS:
        json_io.write_annotations(labels_dir / f"{stem}.json",
                                  [Annotation(subject="leaf", geometry=BBox(2, 2, 8, 8))],
                                  16, 16, keep_empty=True)

    split = {"group_by": "stem", "val_ratio": 0.25, "seed": 11}

    def persisted(experiment_id: str, task: str, data_cfg: dict) -> tuple[dict, dict, dict]:
        """The persisted record, and what the loaders this run actually built say each member's
        own ground truth and source are."""
        train_ds, val_ds, partition = auto_train_val(task, data_cfg, None)
        create_experiment(experiment_id, {"data": data_cfg})
        persist_run_partition(experiment_id, data_cfg, partition=partition)
        served_truth = {ds.sample_members[key]: ds.sample_ground_truth[key]
                        for ds in (train_ds, val_ds) for key in ds.sample_ground_truth}
        served_sources = {ds.sample_members[key]: ds.sample_sources[key]
                          for ds in (train_ds, val_ds) for key in ds.sample_sources}
        return read_run_partition(experiment_id), served_truth, served_sources

    document_run, document_served, document_sources = persisted(
        "exp-shape-document", "detection", {
            "images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "leaf",
            "split": dict(split)})
    mask_run, mask_served, mask_sources = persisted(
        "exp-shape-mask", "semantic_seg", {
            "images_dir": str(images_dir), "labels_dir": str(masks_dir), "num_classes": 2,
            "split": dict(split)})

    assert set(document_run) == set(mask_run), "one record shape, whatever the ground truth is"
    assert document_run["group_by"] == mask_run["group_by"] == "stem"
    assert sorted(document_run["members"]) == [str(labels_dir)]
    assert sorted(mask_run["members"]) == [str(masks_dir)]

    document_block = document_run["members"][str(labels_dir)]
    mask_block = mask_run["members"][str(masks_dir)]
    assert set(document_block) == set(mask_block)
    for field in ("train", "val", "group_key_map"):
        assert document_block[field] == mask_block[field], field

    # One assertion over both shapes: each path the record names for a member is the path the
    # loader that trained on it read, so a record naming any other file fails here.
    for block, served, sources in ((document_block, document_served, document_sources),
                                   (mask_block, mask_served, mask_sources)):
        assert block["label_digests"]["ground_truth"] == served
        assert block["sources"] == sources
        assert set(served) == set(block["train"]) | set(block["val"])

    # Two runs over one set of images name the same pixels per member: a source wrong the same
    # way in a producer and in the record it writes agrees with itself, and fails only here.
    assert document_block["sources"] == mask_block["sources"]

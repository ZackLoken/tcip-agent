"""freeze_selection: a finished run's own drawn train/val partition, frozen into a selection a
later run can bind to.

Reuses test_selection_binding.py's dataset fixture and builds a real drawn split through the
launcher's own resolution, rather than restating it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import (  # noqa: E402
    RUN_FILE,
    experiment_dir,
    read_record,
    run_resolution,
)
from tcip_mcp.pipelines.data.selection import read_selection  # noqa: E402
from tcip_mcp.pipelines.data.split_construction import partition_samples, resolve_run  # noqa: E402

from tests._verified_checkpoint_fixtures import opened_run, resolved_run  # noqa: E402
from tests.test_selection_binding import DATES, SUBJECT, _two_subject_two_date_dataset  # noqa: E402

BUILDER = "tests.bespoke_models:build_bespoke_detection"


def _real_drawn_experiment(
    project: Path, root: Path, experiment_id: str, *, date: str = DATES[0],
    subject: str = SUBJECT, attribute: str | None = None, auto_val: bool = True,
) -> dict:
    """Draws a real train/val split over ``root``'s own fixture dataset through the launcher's
    own resolution, recorded in ``experiment_id``'s launch record under ``project``; a run with
    ``auto_val`` off selects on its training loss. Returns the resolved ``data`` section."""
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    opened_run(project, {
        "model_source": {"task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": subject, "attribute": attribute}, "auto_val": auto_val},
        **({} if auto_val else {"evaluation": {"selection_metric": "loss"}}),
    }, experiment_id=experiment_id)
    return run_resolution(experiment_id, project=project)["data"]


def _damage_resolved(project: Path, experiment_id: str, change) -> None:
    """Rewrite what a run's launch record under ``project`` says it resolved, on disk past its
    one writer, ``change`` applied to it."""
    from tcip_store import RECORD_JSON

    path = experiment_dir(experiment_id, project=project) / RUN_FILE
    record = read_record(path)
    change(record["resolved"])
    path.write_bytes(RECORD_JSON.encode(record))


# -- admits valid work: freeze, read back, bind a second run -------------------


def test_freeze_selection_round_trips_through_a_real_bind(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import selection_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-src")

    result = freeze_selection(tmp_path, "exp-src")

    assert "error" not in result, result
    selection_dir = result["selection_dir"]
    assert selection_dir == str(root / "splits" / "frozen-exp-src")
    assert "reference sides are empty" in result["note"] and "refuse" in result["note"]

    frozen = read_selection(selection_dir, project=tmp_path)
    assert frozen.counts()["calibration"] == frozen.counts()["holdout"] == 0
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

    resolution = resolve_run(second_cfg, project=tmp_path)
    assert len(resolution.train_ds) > 0 and len(resolution.val_ds) > 0


def test_freeze_selection_names_the_sources_the_run_read_not_the_launch_input(tmp_path: Path):
    """The frozen selection is composed from what the run resolved alone. A launch record whose
    stated config is edited to name another images directory holding identically named files
    changes nothing: freezing a partition against pixels the run never saw would bind a later run
    to a different dataset."""
    from PIL import Image
    from tcip_store import RECORD_JSON

    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    data_cfg = _real_drawn_experiment(tmp_path, root, "exp-elsewhere")
    trained_images = Path(data_cfg["images_dir"])

    other = root / "images" / "other"
    other.mkdir(parents=True)
    for image in trained_images.iterdir():
        Image.new("RGB", (16, 16), (200, 10, 10)).save(other / image.name)
    run_dir = experiment_dir("exp-elsewhere", project=tmp_path)
    launch = read_record(run_dir / RUN_FILE)
    launch["config"]["data"]["images_dir"] = str(other)
    (run_dir / RUN_FILE).write_bytes(RECORD_JSON.encode(launch))

    result = freeze_selection(tmp_path, "exp-elsewhere", output_path=str(tmp_path / "frozen"))

    assert "error" not in result, result
    frozen = read_selection(tmp_path / "frozen", project=tmp_path)
    assert frozen.samples
    for sample in frozen.samples:
        assert Path(sample.source).parent == trained_images


def test_freeze_selection_from_an_empty_string_attribute_run_binds(tmp_path: Path):
    """A run whose data section carries ``data.attribute=""`` (an explicit empty string, not
    ``None``) freezes a selection a later, attribute-unscoped run still binds to: the frozen
    ``attribute`` is normalized on write, so no reader has to read one form as the other."""
    from tcip_mcp.tools.data_tools import freeze_selection
    from tcip_mcp.tools.training_tools import selection_compatibility

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-empty-attribute", attribute="")

    result = freeze_selection(tmp_path, "exp-empty-attribute")
    assert "error" not in result, result

    frozen = read_selection(result["selection_dir"], project=tmp_path)
    assert frozen.scope.attribute is None

    second_cfg: dict[str, Any] = {
        "model_source": {"builder": BUILDER, "task": "detection"},
        "data": {"split": {"selection_dir": result["selection_dir"]}},
    }
    assert selection_compatibility(second_cfg, frozen, result["selection_dir"]) == []

    resolution = resolve_run(second_cfg, project=tmp_path)
    assert len(resolution.train_ds) > 0 and len(resolution.val_ds) > 0


def test_freeze_selection_keeps_two_scopes_same_named_members_apart(tmp_path: Path):
    """A record holding two ground-truth scopes holds one member name twice, and the frozen
    selection carries both: composing the scopes into one map keyed by bare name would drop one
    date's member and bind a later run to half the partition its record describes. The partition
    is composed through the resolution's own partition producer over the run's samples and set
    into a real run's launch record past its writer."""
    from tcip_mcp.dataset_layout import status_bucket
    from tcip_mcp.pipelines.data.selection import Sample
    from tcip_mcp.pipelines.data.split_construction import _partition_record
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    train, val = [], []
    for date in DATES:
        images_dir, labels_dir = root / "images" / date, root / "annotations" / date
        stems = sorted(p.stem for p in labels_dir.glob("*.json"))[:2]
        for index, stem in enumerate(stems):
            sample = Sample(member=stem, source=str(images_dir / f"{stem}.jpg"),
                            ground_truth=str(labels_dir / f"{stem}.json"), group=stem,
                            side="train" if index == 0 else "val",
                            confirmation_bucket=status_bucket(SUBJECT, date))
            (train if index == 0 else val).append(sample)
    assert {s.member for s in train} == {s.member for s in train[:1]}, (
        "both dates must contribute the same member name for this to bite")

    _real_drawn_experiment(tmp_path, root, "exp-two-scope")
    _damage_resolved(tmp_path, "exp-two-scope", lambda resolved: resolved.update(
        partition=_partition_record(train + val, seed=0, group_by="stem", selection=None)))

    result = freeze_selection(tmp_path, "exp-two-scope", output_path=str(tmp_path / "frozen"))

    assert "error" not in result, result
    assert (result["train"], result["val"]) == (len(train), len(val))
    frozen = read_selection(tmp_path / "frozen", project=tmp_path)
    assert len(frozen.samples) == len(train) + len(val)
    assert {s.ground_truth for s in frozen.samples} == {
        s.ground_truth for s in train + val}


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
    resolved_run(tmp_path, {
        "images_dir": str(images_dir), "labels_dir": str(labels_dir),
        "scope": {"subject": SUBJECT},
        "split": {"group_key_map":
                  {member_identity(DATES[0], s): "g1" for s in ("a", "b", "c")}
                  | {member_identity(DATES[0], s): "g2" for s in ("d", "e", "f")}}},
        experiment_id="exp-explicit-map")

    result = freeze_selection(tmp_path, "exp-explicit-map")
    assert "error" not in result, result

    frozen = read_selection(result["selection_dir"], project=tmp_path)
    assert frozen.group_by == "explicit_map"
    assert {s.group for s in frozen.samples} <= {"g1", "g2"}
    by_group = {s.group: s.side for s in frozen.samples}
    assert len(by_group) == len({s.group for s in frozen.samples})


# -- refusals ------------------------------------------------------------------


def test_freeze_selection_refuses_an_id_naming_no_run(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    result = freeze_selection(tmp_path, "exp-no-run")
    assert "error" in result and "no run directory" in result["error"]


def _bound_run(root: Path, tmp_path: Path, experiment_id: str, **split_extra) -> None:
    from tcip_mcp.tools.data_tools import draw_splits

    selection_dir = tmp_path / f"src-{experiment_id}"
    drawn = draw_splits(tmp_path, str(root), output_path=str(selection_dir), subject=SUBJECT,
                        seed=2, train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in drawn, drawn

    resolved_run(tmp_path, {"split": {"selection_dir": str(selection_dir), **split_extra}},
                 experiment_id=experiment_id)


@pytest.mark.parametrize("split_extra", [{}, {"redraw_within_selection": True, "seed": 11}],
                         ids=["bound", "redrawn"])
def test_a_bound_run_freezes_and_a_later_run_rebinds_to_its_membership(
    tmp_path: Path, split_extra: dict,
):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _bound_run(root, tmp_path, "exp-bound", **split_extra)
    frozen = freeze_selection(tmp_path, "exp-bound", output_path=str(tmp_path / "frozen"))
    assert "error" not in frozen, frozen

    resolved_run(tmp_path, {"split": {"selection_dir": frozen["selection_dir"]}},
                 experiment_id="exp-rebound")

    def _sides(experiment_id: str) -> dict:
        samples = partition_samples(run_resolution(experiment_id, project=tmp_path)["partition"])
        return {side: sorted(s.ground_truth for s in samples if s.side == side)
                for side in ("train", "val")}

    assert _sides("exp-rebound") == _sides("exp-bound")


def test_freeze_selection_refuses_a_spatial_split(tmp_path: Path):
    """A within-image spatial split resolves region identities, which its data section records as
    ``split.spatial_manifest``, the one field freeze_selection's spatial refusal reads, set here
    past the writer rather than through a tiled single-source fixture, since that field is the
    whole of what the refusal inspects."""
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-spatial")
    _damage_resolved(tmp_path, "exp-spatial",
                     lambda resolved: resolved["data"]["split"].update(spatial_manifest={}))

    result = freeze_selection(tmp_path, "exp-spatial")
    assert "error" in result and "spatial" in result["error"]


def test_freeze_selection_refuses_a_member_whose_ground_truth_moved(tmp_path: Path):
    """The staleness check is per member, against the digest the run's own producer recorded for
    it: a label edited after the run names that member and refuses, since a selection composed
    from it would bind a later run to ground truth this run never saw."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    data_cfg = _real_drawn_experiment(tmp_path, root, "exp-moved")

    moved = Path(data_cfg["labels_dir"]) / "a.json"
    json_io.write_annotations(moved, [
        Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20)),
        Annotation(subject=SUBJECT, geometry=BBox(30, 30, 50, 50)),
    ], 64, 64, keep_empty=True)

    result = freeze_selection(tmp_path, "exp-moved")
    assert "error" in result and "changed since" in result["error"]
    assert "'a'" in result["error"]


def test_freeze_selection_reads_a_member_replaced_by_another_extension_as_moved(tmp_path: Path):
    """The record names the file each member's ground truth was, so a member's document replaced
    by one of another extension carrying the identical bytes reads as moved rather than as
    unchanged: a later bind would otherwise be composed from a path nothing holds."""
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    data_cfg = _real_drawn_experiment(tmp_path, root, "exp-renamed")

    document = Path(data_cfg["labels_dir"]) / "a.json"
    document.with_suffix(".txt").write_bytes(document.read_bytes())
    document.unlink()

    result = freeze_selection(tmp_path, "exp-renamed")
    assert "error" in result and "changed since" in result["error"]
    assert "'a'" in result["error"]


def test_freeze_selection_refuses_an_empty_val_side(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-no-val", auto_val=False)

    result = freeze_selection(tmp_path, "exp-no-val")
    assert "error" in result and "validation" in result["error"]


def test_freeze_selection_refuses_a_resolved_scope_missing_id_map(tmp_path: Path):
    """A resolved record edited past its writer to drop its scope's map composes a selection the
    selection writer refuses, and the door answers with that refusal."""
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-no-id-map")
    _damage_resolved(tmp_path, "exp-no-id-map",
                     lambda resolved: resolved["data"]["scope"].update(id_map=None))

    result = freeze_selection(tmp_path, "exp-no-id-map")
    assert "error" in result and "id_map" in result["error"]


def test_freeze_selection_refuses_labels_changed_since_the_run(tmp_path: Path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-stale-labels")

    labels_dir = root / "annotations" / DATES[0]
    json_io.write_annotations(
        labels_dir / "a.json", [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))], 64, 64,
        keep_empty=True,
    )

    result = freeze_selection(tmp_path, "exp-stale-labels")
    assert "error" in result and "changed" in result["error"]


def test_freeze_selection_refuses_when_a_selection_already_exists_at_the_output(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-first")
    first = freeze_selection(tmp_path, "exp-first")
    assert "error" not in first, first

    _real_drawn_experiment(tmp_path, root, "exp-second")
    second = freeze_selection(tmp_path, "exp-second", output_path=first["selection_dir"])
    assert "error" in second and "already exists" in second["error"]

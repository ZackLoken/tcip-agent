"""freeze_selection: a finished run's own drawn train/val partition, frozen into a selection a
later run can bind to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import tcip_store as ts  # noqa: E402
from tcip_mcp.dataset_layout import capture_label_keys, label_key  # noqa: E402
from tcip_mcp.experiments import (  # noqa: E402
    RUN_FILE,
    experiment_dir,
    read_record,
    run_resolution,
)
from tcip_mcp.pipelines.data.selection import read_selection  # noqa: E402
from tcip_mcp.pipelines.data.split_construction import partition_samples, resolve_run  # noqa: E402
from tcip_mcp.pipelines.schemas import train_config  # noqa: E402

from tests._chain_fixtures import BESPOKE_DETECTION, run_config, training_config  # noqa: E402
from tests._training_values import evaluation_block  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    opened_run, resolved_run, unbuilt_source,
)
from tests.test_selection_binding import DATES, SUBJECT, _two_subject_two_date_dataset  # noqa: E402


def _real_drawn_experiment(
    project: Path, root: Path, experiment_id: str, *, date: str = DATES[0],
    subject: str = SUBJECT, auto_val: bool = True,
) -> dict:
    """Draws a real train/val split over ``root``'s own fixture dataset through the launcher's
    own resolution, recorded in ``experiment_id``'s launch record under ``project``; a run with
    ``auto_val`` off selects on its training loss. Returns the resolved ``data`` block."""
    images_dir = root / "images" / date
    opened_run(project, training_config(
        unbuilt_source("detection"),
        {"images_dir": str(images_dir), "scope": {"subject": subject},
         "auto_val": auto_val, "split": {"seed": 0, "val_ratio": 0.15}},
        **({} if auto_val else {"evaluation": evaluation_block(selection_metric="loss")})),
        experiment_id=experiment_id)
    return run_resolution(experiment_id, project=project).data


def _damage_resolved(project: Path, experiment_id: str, change) -> None:
    """Rewrite what a run's launch record under ``project`` says it resolved, on disk past its
    one writer, ``change`` applied to it."""
    from tcip_store import encode_record

    path = experiment_dir(experiment_id, project=project) / RUN_FILE
    record = read_record(path)
    change(record["resolved"])
    path.write_bytes(encode_record(record))


# -- admits valid work: freeze, read back, bind a second run -------------------


def test_freeze_selection_round_trips_through_a_real_bind(tmp_path: Path):
    from tcip_mcp.pipelines.data.split_construction import selection_compatibility
    from tcip_mcp.tools.data_tools import freeze_selection

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
    assert {s.ground_truth.parts[-1] for s in frozen.samples} <= set("abcdef")
    for sample in frozen.samples:
        assert Path(sample.source).parent == root / "images" / DATES[0]
        assert ts.exists(sample.ground_truth)

    second = train_config(run_config(Path(selection_dir),
                                     {"builder": BESPOKE_DETECTION, "task": "detection"}))
    assert selection_compatibility(second.data, frozen, selection_dir) == []

    resolution = resolve_run(second, project=tmp_path)
    assert len(resolution.train_ds) > 0 and len(resolution.val_ds) > 0


def test_freeze_selection_names_the_sources_the_run_read_not_the_launch_input(tmp_path: Path):
    """The frozen selection is composed from what the run resolved alone. A launch record whose
    stated config is edited to name another images directory holding identically named files
    changes nothing: freezing a partition against pixels the run never saw would bind a later run
    to a different dataset."""
    from PIL import Image
    from tcip_store import encode_record

    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    data_cfg = _real_drawn_experiment(tmp_path, root, "exp-elsewhere")
    trained_images = Path(str(data_cfg.images_dir))

    other = root / "images" / "other"
    other.mkdir(parents=True)
    for image in trained_images.iterdir():
        Image.new("RGB", (16, 16), (200, 10, 10)).save(other / image.name)
    run_dir = experiment_dir("exp-elsewhere", project=tmp_path)
    launch = read_record(run_dir / RUN_FILE)
    launch["config"]["data"]["images_dir"] = str(other)
    (run_dir / RUN_FILE).write_bytes(encode_record(launch))

    result = freeze_selection(tmp_path, "exp-elsewhere", output_path=str(tmp_path / "frozen"))

    assert "error" not in result, result
    frozen = read_selection(tmp_path / "frozen", project=tmp_path)
    assert frozen.samples
    for sample in frozen.samples:
        assert Path(sample.source).parent == trained_images


def test_freeze_selection_keeps_two_scopes_same_named_members_apart(tmp_path: Path):
    """A record holding two ground-truth scopes holds one member name twice, and the frozen
    selection carries both: composing the scopes into one map keyed by bare name would drop one
    date's member and bind a later run to half the partition its record describes. The partition
    is composed through the resolution's own partition producer over the run's samples and set
    into a real run's launch record past its writer."""
    from tcip_mcp.pipelines.data.selection import Sample, ground_truth_digest
    from tcip_mcp.pipelines.data.split_construction import _partition_record
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    train, val = [], []
    for date in DATES:
        images_dir = root / "images" / date
        stems = [key.parts[-1] for key in capture_label_keys(root, date)][:2]
        for index, stem in enumerate(stems):
            key = label_key(root, date, stem)
            sample = Sample(member=stem, source=str(images_dir / f"{stem}.jpg"),
                            ground_truth=key, group=stem,
                            side="train" if index == 0 else "val",
                            ground_truth_digest=ground_truth_digest(key))
            (train if index == 0 else val).append(sample)
    assert {s.member for s in train} == {s.member for s in train[:1]}, (
        "both dates must contribute the same member name for this to bite")

    _real_drawn_experiment(tmp_path, root, "exp-two-scope")
    _damage_resolved(tmp_path, "exp-two-scope", lambda resolved: resolved.update(
        partition=_partition_record(train + val, seed=0, group_by="stem", selection=None,
                                    spatial=None)))

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
    images_dir = root / "images" / DATES[0]
    resolved_run(tmp_path, {
        "images_dir": str(images_dir), "scope": {"subject": SUBJECT},
        "split": {"seed": 0, "val_ratio": 0.15, "group_key_map":
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
                        seed=2, val_ratio=0.25, calibration_ratio=0.125, holdout_ratio=0.125)
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
        samples = partition_samples(run_resolution(experiment_id, project=tmp_path).partition)
        return {side: {s.ground_truth for s in samples if s.side == side}
                for side in ("train", "val")}

    assert _sides("exp-rebound") == _sides("exp-bound")


def test_freeze_selection_refuses_a_spatial_split(tmp_path: Path):
    """A within-image spatial split resolves region identities, which its partition records as
    its ``spatial`` split, and freeze_selection binds only a stem-keyed partition."""
    from tcip_mcp.tools.data_tools import freeze_selection
    from tests.test_spatial_region_containment import _mosaic_dataset

    images_dir = _mosaic_dataset(tmp_path / "ds")
    resolved_run(tmp_path, {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "tiling": {"enabled": True, "tile_size": 128, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "seed": 1}}, experiment_id="exp-spatial")

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
    _real_drawn_experiment(tmp_path, root, "exp-moved")

    json_io.write_label_document(label_key(root, DATES[0], "a"), [
        Annotation(subject=SUBJECT, geometry=BBox(4, 4, 20, 20)),
        Annotation(subject=SUBJECT, geometry=BBox(30, 30, 50, 50)),
    ], 64, 64, keep_empty=True)

    result = freeze_selection(tmp_path, "exp-moved")
    assert "error" in result and "changed since" in result["error"]
    assert "'a'" in result["error"]


def test_freeze_selection_reads_a_members_removed_document_as_moved(tmp_path: Path):
    """The record names the document each member's ground truth was, so a member whose document
    was removed since reads as moved rather than as unchanged: a later bind would otherwise be
    composed from a document nothing holds."""
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-renamed")

    ts.delete(label_key(root, DATES[0], "a"))

    result = freeze_selection(tmp_path, "exp-renamed")
    assert "error" in result and "changed since" in result["error"]
    assert "'a'" in result["error"]


def test_freeze_selection_refuses_an_empty_val_side(tmp_path: Path):
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-no-val", auto_val=False)

    result = freeze_selection(tmp_path, "exp-no-val")
    assert "error" in result and "validation" in result["error"]


def test_freeze_selection_refuses_a_resolved_scope_missing_its_attributes(tmp_path: Path):
    """A resolved record edited past its writer to drop its scope's attributes composes a
    selection the selection writer refuses, and the door answers with that refusal."""
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-no-attributes")
    _damage_resolved(tmp_path, "exp-no-attributes",
                     lambda resolved: resolved["data"]["scope"].update(attributes=None))

    result = freeze_selection(tmp_path, "exp-no-attributes")
    assert "error" in result and "records no attributes" in result["error"]


def test_freeze_selection_refuses_labels_changed_since_the_run(tmp_path: Path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.data_tools import freeze_selection

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    _real_drawn_experiment(tmp_path, root, "exp-stale-labels")

    json_io.write_label_document(
        label_key(root, DATES[0], "a"), [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))],
        64, 64, keep_empty=True)

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

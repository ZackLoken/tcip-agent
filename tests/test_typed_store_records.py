"""Each stored record read back through its own model.

Every record is written through the platform's own producer and read back through the reader
that validates it into its model; the read equals what the producer was given, and a stored
document missing a required field is refused naming the field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.model_registry import Producer

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.experiments import SweepRecord
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.execution import Pass

DATE = "2026-02-11"


def _sweep_record() -> SweepRecord:
    """Return a one-trial :class:`~tcip_mcp.experiments.SweepRecord` over an empty space."""
    from tcip_mcp.experiments import SweepInput, SweepRecord

    return SweepRecord(
        created="2026-02-11T00:00:00+00:00",
        trait_revision=None,
        input=SweepInput(
            n_trials=1,
            search_alg="random",
            scheduler={"name": "fifo"},
            max_concurrent=1,
            baseline_params=None,
            resources_per_trial=None,
            param_space={},
            base_config={"evaluation": {"selection_metric": "val_loss"}},
            relaunched_from=None,
            split_draw_seeds=None,
            search_seed=0,
            trial_budget=None,
        ),
    )


def _sweep_dir(project: Path, name: str) -> Path:
    """Open a sweep directory under ``project`` holding :func:`_sweep_record`'s record."""
    from tcip_mcp.experiments import SWEEP_FILE, experiment_dir, write_record

    run_dir = experiment_dir(name, project=project)
    run_dir.mkdir(parents=True)
    write_record(run_dir / SWEEP_FILE, _sweep_record())
    return run_dir


def test_a_sweep_record_reads_back_as_written_and_one_missing_a_field_refuses(
    tmp_path: Path,
) -> None:
    """A sweep record written once reads back equal; one stored without its input refuses."""
    from tcip_mcp.experiments import SWEEP_FILE, observe_sweep

    sweep_dir = _sweep_dir(tmp_path, "hpo_a")

    assert observe_sweep(sweep_dir).record == _sweep_record()
    path = sweep_dir / SWEEP_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["input"]
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"input\s+Field required"):
        observe_sweep(sweep_dir)


def test_a_run_record_reads_back_as_its_launch_wrote_it_and_one_missing_a_field_refuses(
    tmp_path: Path,
) -> None:
    """A launched run's record reads back whole; one stored without its dataset refuses."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, read_run
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path)
    record = read_run(run_dir)

    assert record.resolved is not None
    assert read_run(run_dir) == record
    path = run_dir / RUN_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["dataset"]
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"dataset\s+Field required"):
        read_run(run_dir)


def test_a_run_record_whose_partition_sample_states_no_member_refuses_at_the_read(
    tmp_path: Path,
) -> None:
    """A launched run reads back; one whose partition's first sample lacks ``member`` refuses."""
    pytest.importorskip("torch")
    from tcip_mcp.experiments import RUN_FILE, observe_run
    from tests._verified_checkpoint_fixtures import finished_run

    run_dir = finished_run(tmp_path)
    assert observe_run(run_dir).record.resolution.partition.samples
    path = run_dir / RUN_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["resolved"]["partition"]["samples"][0]["member"]
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValidationError) as refused:
        observe_run(run_dir)

    (error,) = refused.value.errors()
    assert error["loc"][-4:] == ("partition", "samples", 0, "member")


def test_an_execution_record_stating_no_sources_refuses_naming_them() -> None:
    """A pass's execution record reads back whole; one stored without ``sources`` refuses."""
    from tcip_mcp.pipelines.execution import EXECUTION_RECORD, Execution

    execution = Execution(
        conf=None,
        density=None,
        masks=False,
        mask_cut=None,
        tile_size=None,
        overlap=None,
        tile_resize=None,
        postprocess=None,
        cross_tile_nms=None,
        sahi_version=None,
        rationale=None,
        sources={},
    )
    stored = EXECUTION_RECORD.dump_python(execution, mode="json")
    assert EXECUTION_RECORD.validate_python(stored) == execution
    del stored["sources"]

    with pytest.raises(ValidationError) as refused:
        EXECUTION_RECORD.validate_python(stored)

    (error,) = refused.value.errors()
    assert (error["loc"], error["type"]) == (("sources",), "missing")


def test_a_tuple_refuses_at_both_encoders_naming_its_type(
    tmp_path: Path,
) -> None:
    """A tuple, which JSON reads back as a list, refuses at a log append and a record write."""
    key = ts.Key("probe", str(tmp_path), ("one",))
    tupled: dict = {"pair": (1, 2)}

    with pytest.raises(ts.StoreError, match="tuple"):
        ts.append(key, tupled)
    with pytest.raises(ts.StoreError, match="tuple"):
        ts.replace(key, tupled)
    ts.append(key, {"pair": [1, 2]})
    assert ts.read_log(key).records == [{"pair": [1, 2]}]


def _run_checkpoint(project: Path, **kwargs: object) -> Path:
    """Return the checkpoint a run of ``project`` selected by completing, with ``kwargs``."""
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import finished_run

    checkpoint = observe(finished_run(project, **kwargs)).checkpoint
    assert checkpoint is not None
    return Path(checkpoint.path)


def _foreign_checkpoint(tmp_path: Path) -> Path:
    """Return a run's checkpoint copied into the project at ``tmp_path``, which took no run."""
    ckpt = tmp_path / "weights" / "m.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(
        _run_checkpoint(tmp_path.parent / "producer").read_bytes()
    )
    return ckpt


def test_a_registered_checkpoint_reads_back_as_the_entry_it_was_registered_as(
    tmp_path: Path,
) -> None:
    """A registered foreign checkpoint reads back with its name, digest, metrics and tags."""
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry,
        admitted_digest,
        read_registry_index,
    )

    ckpt = _foreign_checkpoint(tmp_path)
    returned = (
        ModelRegistry(str(tmp_path))
        .register_model(
            "m", str(ckpt), metrics={"map50": 0.5}, tags=["field"], actor=None
        )
        .entry
    )

    (entry,) = read_registry_index(tmp_path)
    assert (
        entry.name,
        entry.checkpoint_path,
        entry.sha256,
        entry.metrics,
        entry.tags,
        entry.experiment_id,
    ) == (
        "m",
        "weights/m.pt",
        admitted_digest(ckpt),
        {"map50": 0.5},
        ["field"],
        None,
    )
    assert (returned.sha256, returned.registered_at) == (
        entry.sha256,
        entry.registered_at,
    )


def test_a_registry_index_missing_a_field_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """An index entry stored without its ``sha256`` refuses where the index is read."""
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import (
        ModelRegistry,
        read_registry_index,
        registry_index_key,
    )

    ckpt = _foreign_checkpoint(tmp_path)
    ModelRegistry(str(tmp_path)).register_model("m", str(ckpt), actor=None)
    (entry,) = read_registry_index(tmp_path)
    ts.replace(
        registry_index_key(tmp_path),
        {"entries": [entry.model_dump(mode="json", exclude={"sha256"})]},
    )

    with pytest.raises(
        ValidationError, match=r"entries\.0\.sha256\s+Field required"
    ):
        read_registry_index(tmp_path)


def _published(project: Path) -> tuple[Bucket, Pass, VerifiedCheckpoint, Path]:
    """Publish one image's predictions under a registered checkpoint's untiled pass.

    Returns the bucket read back, the pass it was published under, the checkpoint and the
    image's path.
    """
    from tcip_mcp.buckets import pass_documents, publish, source_root
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tests._chain_fixtures import RATIONALE, predicted
    from tests._producer_fixtures import write_image
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    image = write_image(image_dir(project / "ds", DATE) / "a.png", (64, 64))
    checkpoint = load_registered_checkpoint(
        _run_checkpoint(
            project, data={"num_channels": 3, "scope": {"subject": "bud"}}
        ),
        project=project,
    )
    p = prepare(
        checkpoint, Stated(tile=False, conf=SAMPLE_CONF, rationale=RATIONALE)
    ).runnable()
    bucket = publish(
        project,
        source_root([str(image)]),
        f"m/{DATE}",
        pass_documents(p, [predicted(image, ["bud"])]),
        producer=checkpoint.producer,
        scope=p.scope,
        execution=p.execution,
        raster_path=None,
        raster_identity=None,
        assessment_id=None,
        actor=None,
    )
    return bucket, p, checkpoint, image


def test_a_published_bucket_reads_back_as_the_pass_that_published_it(
    tmp_path: Path,
) -> None:
    """A published bucket reads back with the pass's scope, execution and producer."""
    pytest.importorskip("torch")
    bucket, p, checkpoint, image = _published(tmp_path)

    assert (bucket.scope, bucket.execution, bucket.producer) == (
        p.scope,
        p.execution,
        checkpoint.producer,
    )
    assert type(bucket.producer) is Producer
    assert (bucket.documents, bucket.date, bucket.dropped_boxes) == (
        {"a": "a.png"},
        DATE,
        0,
    )


def test_a_proposal_bucket_reads_back_naming_what_proposed_it(
    tmp_path: Path,
) -> None:
    """A bucket published as proposals reads back with its proposer, never a checkpoint."""
    from tcip_mcp.buckets import (
        Document,
        Proposed,
        publish,
        read_bucket,
        source_root,
    )
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tests._producer_fixtures import write_image

    image = write_image(image_dir(tmp_path / "ds", DATE) / "a.png", (64, 64))
    root = source_root([str(image)])
    publish(
        tmp_path,
        root,
        f"engine/{DATE}/a",
        [Document(str(image), {"annotations": []})],
        producer=Proposed(proposed_by="engine"),
        scope=ClassScope(),
        execution=None,
        raster_path=None,
        raster_identity=None,
        assessment_id=None,
        actor=None,
    )

    producer = read_bucket(root, f"engine/{DATE}/a").producer

    assert type(producer) is Proposed
    assert producer == Proposed(proposed_by="engine")


def test_a_bucket_record_missing_a_field_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """A bucket record stored without its ``execution`` refuses where the bucket is read."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import bucket_key

    bucket, _p, _checkpoint, _image = _published(tmp_path)
    key = bucket_key(bucket.root, bucket.name)
    stored = ts.read(key)
    assert isinstance(stored, dict)
    del stored["execution"]
    ts.replace(key, stored)

    with pytest.raises(ValidationError, match=r"execution\s+Field required"):
        read_bucket(bucket.root, bucket.name)


def test_a_final_status_reads_back_as_it_was_written(tmp_path: Path) -> None:
    """A run's final status reads back with its state, error and checkpoint."""
    from tcip_mcp.audit import now
    from tcip_mcp.experiments import RunCheckpoint, observe, write_final_status

    run_dir = _sweep_dir(tmp_path, "run_a")
    path = str(run_dir / "model_final.pt")
    write_final_status(
        run_dir,
        "completed",
        None,
        ended=now(),
        checkpoint=RunCheckpoint(path=path, sha256="ab"),
    )

    final = observe(run_dir).final
    assert final is not None
    assert (final.state, final.status_error, final.checkpoint) == (
        "completed",
        None,
        RunCheckpoint(path=path, sha256="ab"),
    )
    with pytest.raises(ValidationError, match="state"):
        write_final_status(
            _sweep_dir(tmp_path, "run_b"), "running", None, ended=now()
        )


def test_a_final_status_missing_a_field_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """A final status stored without its ``ended`` refuses where the run is observed."""
    from tcip_mcp.audit import now
    from tcip_mcp.experiments import (
        FINAL_STATUS_FILE,
        observe,
        write_final_status,
    )

    run_dir = _sweep_dir(tmp_path, "run_a")
    write_final_status(run_dir, "failed", "stopped", ended=now())
    path = run_dir / FINAL_STATUS_FILE
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["ended"]
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValidationError, match=r"ended\s+Field required"):
        observe(run_dir)


def test_a_cancellation_reads_back_as_its_first_request(
    tmp_path: Path,
) -> None:
    """A second cancellation request answers the first one's time, read through its record."""
    from tcip_mcp.experiments import (
        CANCEL_FILE,
        cancel_requested_at,
        request_cancel,
    )

    run_dir = _sweep_dir(tmp_path, "run_a")
    first = request_cancel(run_dir)

    assert request_cancel(run_dir) == first == cancel_requested_at(run_dir)
    (run_dir / CANCEL_FILE).write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"requested\s+Field required"):
        cancel_requested_at(run_dir)


def test_a_raised_flag_reads_back_as_raised_and_one_missing_a_field_refuses(
    tmp_path: Path,
) -> None:
    """A flag raised through the save reads back as raised; one stored without ``by`` refuses."""
    from tcip_annotation.flags import (
        FlagRequest,
        PointTarget,
        flag_key,
        flags_record,
        read_flags,
    )
    from tcip_mcp.dataset_layout import Gestures, save_label_document
    from tests._producer_fixtures import image_label_key, write_image

    target = PointTarget(point=(50.0, 40.0), subject="bud")

    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (100, 80))
    key = image_label_key(image)
    save_label_document(
        tmp_path,
        key,
        [{"subject": "bud", "bbox": [40.0, 30.0, 60.0, 50.0]}],
        width=100,
        height=80,
        author="user:breeder",
        actor="user:breeder",
        gestures=Gestures(flag=(FlagRequest(text=" open? ", target=target),)),
    )

    (flag,) = read_flags(flag_key(key))
    assert (flag.request.text, flag.by, flag.request.target, flag.open) == (
        "open?",
        "user:breeder",
        target,
        True,
    )
    entries = flags_record([flag])["flags"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    entry = entries[0]
    del entry["by"]
    ts.replace(flag_key(key), {"flags": [entry]})
    with pytest.raises(
        ValidationError, match=r"flags\.0\.by\s+Field required"
    ):
        read_flags(flag_key(key))


def test_a_subject_registry_reads_back_as_written_and_one_missing_a_field_refuses(
    tmp_path: Path,
) -> None:
    """A saved registry reads back whole; a subject lacking a description refuses."""
    from tcip_mcp import subject_registry as sr
    from tests._producer_fixtures import registry_over

    registry = sr.SubjectRegistry(
        subjects=(
            sr.Subject(
                name="bud",
                description="one bud",
                attributes=(
                    sr.Attribute(
                        name="opening",
                        type="ordinal",
                        values=("closed", "open"),
                    ),
                ),
            ),
            sr.Subject(name="bush", description="one bush"),
        )
    )
    registry_over(tmp_path, registry)

    assert sr.read_registry(tmp_path) == registry
    with pytest.raises(sr.RegistryError, match="description"):
        sr.registry_from_dict({"bud": {"attributes": None}})


def test_a_scale_assessment_reads_back_as_its_producer_answered_it(
    tmp_path: Path,
) -> None:
    """A scale assessment reads back as calibrated, covering the one capture its samples span.

    One stored without failures refuses, and so does one carrying a ``passed`` beside its failures
    or a reference carrying ``captures`` beside its samples.
    """
    from tcip_mcp.assessment import (
        ASSESSMENT_FILE,
        assessment_dir,
        read_assessment,
    )
    from tcip_mcp.dataset_layout import read_dataset_identity
    from tests import _trait_fixtures as fx
    from tests.test_physical_scale_assessment import (
        DATE as SCALE_DATE,
        _author_tolerance,
        _calibrate,
        _reference,
    )

    fx.seed_delivery_traits(tmp_path)
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    produced = _calibrate(tmp_path, selection, csv_path)

    assessment = read_assessment(tmp_path, produced["assessment_id"])
    assert (assessment.passed, assessment.failures) == (True, [])
    assert assessment.scale is not None
    assert assessment.scale["unit"] == "mm"
    assert assessment.scale["value"] == pytest.approx(0.1)
    assert assessment.reference.selection_dir is not None
    assert Path(assessment.reference.selection_dir) == selection.resolve()
    n_samples = produced["reference"]["n_samples"]
    assert len(assessment.reference.samples) == n_samples
    identity = read_dataset_identity(tmp_path / "ds")
    assert identity is not None
    assert assessment.reference.dataset_id == identity.id
    other_date = "2026-02-01"
    assert {
        (dataset_id, date): assessment.reference.covers(
            dataset_id=dataset_id, date=date, raster_identity=None
        )
        for dataset_id in (identity.id, None)
        for date in (SCALE_DATE, other_date)
    } == {
        (identity.id, SCALE_DATE): True,
        (identity.id, other_date): False,
        (None, SCALE_DATE): False,
        (None, other_date): False,
    }

    path = (
        assessment_dir(tmp_path, produced["assessment_id"]) / ASSESSMENT_FILE
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    for record, match in (
        (
            {**stored, "passed": True},
            r"passed\s+Unexpected keyword argument",
        ),
        (
            {
                **stored,
                "reference": {
                    **stored["reference"],
                    "captures": [[identity.id, SCALE_DATE]],
                },
            },
            r"reference\.captures\s+Unexpected keyword argument",
        ),
        (
            {k: v for k, v in stored.items() if k != "failures"},
            r"failures\s+Field required",
        ),
    ):
        path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValidationError, match=match):
            read_assessment(tmp_path, produced["assessment_id"])

"""The checkpoint digest rail: every door that loads a checkpoint recomputes the sha256 of the bytes
it loaded and refuses one no completed run of the project and no registry entry names, before
anything in it is unpickled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import METRICS_KEY, STATE_DICT_KEY  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    finished_run,
    foreign_checkpoint,
    registered_checkpoint,
)


def _unregistered(tmp_path: Path, **kwargs) -> str:
    """A checkpoint a completed run of another project produced, which nothing in this one
    names."""
    return registered_checkpoint(tmp_path.parent / "other_project", **kwargs)


def _images(tmp_path: Path, n: int = 1, size: int = 100):
    from PIL import Image

    images_dir = tmp_path / "images" / "2026-01-01"
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = images_dir / f"img{i}.png"
        Image.new("RGB", (size, size), (100, 100, 100)).save(p)
        paths.append(str(p))
    return images_dir, paths


def _register(tmp_path: Path, ckpt_path: str, *, name: str = "rail-model",
              tags: list[str] | None = None) -> dict:
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(tmp_path, name=name, checkpoint_path=ckpt_path, tags=tags)
    assert "error" not in result, result
    return result


def _infer(tmp_path: Path, ckpt: str, bucket: str = "preds") -> dict:
    """``run_inference`` of ``ckpt`` over one image, untiled, publishing the bucket named
    ``bucket`` under ``tmp_path``."""
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF, SAMPLE_MAX_DETS

    images_dir, _ = _images(tmp_path)
    return run_inference(tmp_path, ckpt, images_dir=str(images_dir), bucket=bucket,
                         device="cpu",
                         stated=Stated(tile=False, conf=SAMPLE_CONF, max_dets=SAMPLE_MAX_DETS))


def _published(tmp_path: Path, bucket: str) -> bool:
    import tcip_store

    from tcip_mcp.dataset_layout import bucket_key

    return tcip_store.exists(bucket_key(tmp_path, bucket))


# Rail 1: an unregistered checkpoint the platform's own producer wrote is refused by name, at
# every door, writing nothing.

def test_run_inference_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path):
    r = _infer(tmp_path, _unregistered(tmp_path))

    assert "register_model" in r["error"]
    assert repr(str(tmp_path)) in r["error"]
    assert not _published(tmp_path, "preds")


def test_evaluate_model_refuses_an_unregistered_checkpoint_by_bare_path(tmp_path):
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.training_tools import evaluate_model

    r = evaluate_model(tmp_path, ckpt, str(images_dir))
    assert "register_model" in r["error"]


def test_assess_checkpoint_refuses_an_unregistered_checkpoint(tmp_path):
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tests import _trait_fixtures as fx

    fx.seed_confirmed_count(tmp_path)

    r = assess_checkpoint(tmp_path, checkpoint_path=_unregistered(tmp_path),
                          trait=fx.COUNT_TRAIT, delivery_kind="per_image_count",
                          selection_dir=str(tmp_path / "selection"))

    assert "register_model" in r["error"]
    assert not (tmp_path / ".tcip" / "assessments").exists()


def test_web_inference_worker_refuses_an_unregistered_checkpoint(tmp_path):
    pytest.importorskip("fastapi")
    from tcip_mcp.pipelines.execution import Stated
    from tcip_web.routes.inference import InferenceJob, _worker
    from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS

    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    job = InferenceJob(job_id="rail1", actor="user:tester", checkpoint_path=ckpt,
                       images_dir=str(images_dir), dataset_root=str(tmp_path), bucket="out",
                       project=str(tmp_path), stated=Stated(tile=False, **SAMPLE_DETECTOR_PASS))
    _worker(job)
    assert job.status == "failed"
    assert "register_model" in job.error
    assert not _published(tmp_path, "out")
    assert job.done == 0


def test_triage_predictions_refuses_an_unregistered_checkpoint(tmp_path):
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    r = triage_predictions(tmp_path, ckpt, str(images_dir))
    assert "register_model" in r["error"]


def test_triage_predictions_refuses_by_the_project_it_acts_on(tmp_path):
    """The project the call acts on is where the load looks: registered under a project the call
    does not name, the checkpoint still refuses, naming the project it did name."""
    registered_root = tmp_path / "registered"
    registered_root.mkdir()
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    ckpt = _unregistered(tmp_path)
    _register(registered_root, ckpt)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    r = triage_predictions(other_root, ckpt, str(images_dir))
    assert "register_model" in r["error"]
    assert repr(str(other_root)) in r["error"]
    assert str(registered_root) not in r["error"]


def test_triage_predictions_admits_a_checkpoint_registered_under_the_project_it_acts_on(
    tmp_path,
):
    """The admitting direction, roles reversed: acting on the project the checkpoint really is
    registered under lets the same call through."""
    registered_root = tmp_path / "registered"
    registered_root.mkdir()
    ckpt = _unregistered(tmp_path)
    _register(registered_root, ckpt)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS

    r = triage_predictions(registered_root, ckpt, str(images_dir), max_dets=SAMPLE_MAX_DETS)
    assert "error" not in r, r


def test_review_priority_route_worker_fails_the_job_on_an_unregistered_checkpoint(tmp_path):
    """The review-priority route's own worker, driven directly, fails its job on a checkpoint
    nothing registered."""
    pytest.importorskip("fastapi")
    from tcip_web.routes.annotate import PriorityQueueJob, _pq_worker

    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    job = PriorityQueueJob(job_id="rail1-pq", checkpoint_path=ckpt, images_dir=str(images_dir),
                           subject=None, method="combined", budget=10, project=str(tmp_path))
    _pq_worker(job)
    assert job.status == "failed"
    assert "register_model" in job.error
    assert job.queue == []


def test_review_priority_route_worker_completes_the_job_with_a_registered_checkpoint(
    tmp_path, monkeypatch,
):
    """The admitting half: a checkpoint a run of the job's own project completed runs
    _pq_worker to a completed job rather than a failed one."""
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    import tcip_mcp.pipelines.active_learning.scorer as al_scorer
    from tcip_web.routes.annotate import PriorityQueueJob, _pq_worker

    ckpt = registered_checkpoint(tmp_path)
    images_dir, _ = _images(tmp_path)

    monkeypatch.setattr(
        al_scorer, "resolve_scorer",
        lambda method, task: SimpleNamespace(score=lambda sources, predictor: []))

    job = PriorityQueueJob(job_id="rail7-pq", checkpoint_path=ckpt, images_dir=str(images_dir),
                           subject=None, method="combined", budget=10, project=str(tmp_path))
    _pq_worker(job)
    assert job.status == "completed", job.error
    assert job.queue == []


# Rail 2: a registered checkpoint whose bytes are replaced (in place, or by rename) after
# registration is refused: the digest of the bytes actually loaded names no entry.

def test_run_inference_refuses_a_checkpoint_overwritten_in_place_after_registration(tmp_path):
    ckpt = Path(foreign_checkpoint(tmp_path))
    ckpt.write_bytes(Path(_unregistered(tmp_path)).read_bytes())

    assert "register_model" in _infer(tmp_path, str(ckpt))["error"]


def test_run_inference_refuses_a_registered_checkpoint_replaced_by_rename(tmp_path):
    ckpt = Path(foreign_checkpoint(tmp_path))
    other = tmp_path / "other.pt"
    other.write_bytes(Path(_unregistered(tmp_path)).read_bytes())
    ckpt.unlink()
    other.rename(ckpt)

    assert "register_model" in _infer(tmp_path, str(ckpt))["error"]


# Rail 4: one sha256 resolves to exactly one owner: the run whose completion names those bytes
# (the first to complete), whatever order foreign registrations of the same bytes came in.

def _identical_runs_data(tmp_path) -> dict:
    """One data section two runs at one seed train identical bytes over, outside every project
    the runs are in, so each checkpoint spells its locations the same way."""
    from tests._verified_checkpoint_fixtures import SCOPED_DATA, detection_images

    shared = tmp_path.parent / f"{tmp_path.name}-shared"
    return {**detection_images(shared, SCOPED_DATA["scope"]), **SCOPED_DATA}


def test_two_runs_whose_final_statuses_name_one_digest_resolve_to_the_first_to_complete(
        tmp_path):
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint, registered_entries

    data = _identical_runs_data(tmp_path)
    first = finished_run(tmp_path, experiment_id="expA", seed=7, data=data)
    second = finished_run(tmp_path, experiment_id="expB", seed=7, data=data)
    checkpoint_a, checkpoint_b = observe(first).checkpoint, observe(second).checkpoint
    assert checkpoint_a is not None and checkpoint_b is not None
    assert checkpoint_a["sha256"] == checkpoint_b["sha256"]

    assert [e["experiment_id"] for e in registered_entries(tmp_path)] == ["expA"]
    for path in (checkpoint_a["path"], checkpoint_b["path"]):
        assert load_registered_checkpoint(path, project=tmp_path).experiment_id == "expA"


def test_a_foreign_registration_after_its_run_completed_leaves_the_run_its_one_owner(tmp_path):
    from tcip_mcp.model_registry import registered_entries
    from tcip_mcp.tools.model_tools import register_model

    ckpt = registered_checkpoint(tmp_path, experiment_id="expA")
    copy = tmp_path / "copy.pt"
    copy.write_bytes(Path(ckpt).read_bytes())

    owner = register_model(tmp_path, name="entry-untagged", checkpoint_path=str(copy))

    assert owner["experiment_id"] == "expA"
    assert [e["experiment_id"] for e in registered_entries(tmp_path)] == ["expA"]


def test_a_foreign_registration_before_its_run_completed_leaves_the_run_its_one_owner(tmp_path):
    """The same bytes registered as foreign first and then named by a run's completion resolve
    to one entry, the run's."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint, registered_entries
    from tcip_mcp.tools.model_tools import register_model

    data = _identical_runs_data(tmp_path)
    elsewhere = observe(finished_run(tmp_path.parent / f"{tmp_path.name}-elsewhere",
                                     experiment_id="expX", seed=7, data=data)).checkpoint
    assert elsewhere is not None
    registered = register_model(tmp_path, name="foreign-first", checkpoint_path=elsewhere["path"])
    assert registered["experiment_id"] is None

    run = observe(finished_run(tmp_path, experiment_id="expA", seed=7, data=data)).checkpoint
    assert run is not None and run["sha256"] == elsewhere["sha256"]

    (entry,) = registered_entries(tmp_path)
    assert entry["experiment_id"] == "expA"
    loaded = load_registered_checkpoint(elsewhere["path"], project=tmp_path)
    assert loaded.experiment_id == "expA"


def test_two_foreign_registrations_of_one_sha256_resolve_to_one_entry(tmp_path):
    """The registry is keyed by a checkpoint's bytes: the same foreign bytes registered twice,
    from two paths under two names, leave one entry, the later registration's, and a load of
    either copy resolves to that one entry."""
    from tcip_mcp.model_registry import load_registered_checkpoint, registered_entries
    from tcip_mcp.tools.model_tools import register_model

    ckpt = foreign_checkpoint(tmp_path, name="first-name")
    copy = tmp_path / "copy.pt"
    copy.write_bytes(Path(ckpt).read_bytes())
    assert "error" not in register_model(tmp_path, name="second-name", checkpoint_path=str(copy))

    (entry,) = registered_entries(tmp_path)
    assert entry["name"] == "second-name"
    for path in (ckpt, str(copy)):
        loaded = load_registered_checkpoint(path, project=tmp_path)
        assert loaded.entry["name"] == "second-name"
        assert loaded.experiment_id is None


# Rail 5: an unregistered checkpoint is refused without being unpickled.

def _touch_marker(marker_path: str) -> "_SideEffectOnUnpickle":
    Path(marker_path).write_text("unpickled", encoding="utf-8")
    return _SideEffectOnUnpickle.__new__(_SideEffectOnUnpickle)


class _SideEffectOnUnpickle:
    """Its unpickling writes a marker file, so a load that unpickles is visible on disk."""

    def __init__(self, marker_path: str) -> None:
        self._marker_path = marker_path

    def __reduce__(self):
        return (_touch_marker, (self._marker_path,))


def test_run_inference_refuses_without_unpickling_a_side_effect_payload(tmp_path):
    marker = tmp_path / "unpickled.marker"
    ckpt = tmp_path / "m.pt"
    torch.save({STATE_DICT_KEY: {},
                "carries_side_effect": _SideEffectOnUnpickle(str(marker))}, str(ckpt))

    assert "register_model" in _infer(tmp_path, str(ckpt))["error"]
    assert not marker.exists()  # the payload was never unpickled


# Rail 6: valid work the rail admits.

def test_run_inference_admits_a_registered_checkpoint_and_records_its_digest(tmp_path):
    from tcip_mcp.buckets import read_bucket

    ckpt = foreign_checkpoint(tmp_path)

    r = _infer(tmp_path, ckpt)

    assert "error" not in r, r
    digest = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert r["checkpoint_sha256"] == digest
    bucket = read_bucket(tmp_path, "preds")
    assert bucket.producer["checkpoint_sha256"] == digest and bucket.assessment_id is None


def test_run_inference_admits_the_same_checkpoint_copied_to_another_path(tmp_path):
    ckpt = foreign_checkpoint(tmp_path)
    copy = tmp_path / "copy.pt"
    copy.write_bytes(Path(ckpt).read_bytes())

    r = _infer(tmp_path, str(copy))

    assert "error" not in r, r
    assert r["checkpoint_sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()


def _best_and_final(ctx) -> None:
    """A body saving two checkpoints: the model as built as model_best, and a second build as
    model_final."""
    ctx.save_checkpoint({STATE_DICT_KEY: ctx.build_model().state_dict()}, "model_best")
    ctx.save_checkpoint({STATE_DICT_KEY: ctx.build_model().state_dict()}, "model_final")


def test_run_inference_admits_a_second_checkpoint_of_a_run_registered_under_a_distinct_name(
    tmp_path,
):
    """A run's model_final beside the model_best its completion registered is registered
    explicitly under a distinct name, and admitted."""
    from tcip_mcp.pipelines.training.generic_trainer import checkpoint_path

    run_dir = finished_run(tmp_path, experiment_id="expTwo",
                           training_source=f"{__name__}:_best_and_final")
    final = checkpoint_path(run_dir, "model_final")
    _register(tmp_path, str(final), name="run-final")

    r = _infer(tmp_path, str(final))
    assert "error" not in r, r


# Rail 7: a checkpoint a completed envelope run registered by completing runs with no further
# step, and carries the run as its producer.

def test_a_completed_runs_weights_run_through_run_inference_with_no_further_step(tmp_path):
    ckpt = registered_checkpoint(tmp_path, experiment_id="exp-rail7")

    r = _infer(tmp_path, ckpt)

    assert "error" not in r, r
    assert r["checkpoint_sha256"] == hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert r["experiment_id"] == "exp-rail7"


# Rail 10: register_model and load_registered_checkpoint agree on one file's digest.

def test_registration_digest_and_load_digest_agree(tmp_path):
    ckpt = _unregistered(tmp_path)
    reg = _register(tmp_path, ckpt)

    from tcip_mcp.model_registry import load_registered_checkpoint

    checkpoint = load_registered_checkpoint(ckpt, project=tmp_path)
    assert checkpoint.sha256 == reg["sha256"]


def test_a_completed_runs_final_status_and_the_load_agree_on_its_digest(tmp_path):
    """The completion's recorded digest and the load's recomputed one are one digest, and the
    metrics a ranking reads are the ones the checkpoint's own payload carries."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import ModelRegistry, load_registered_checkpoint

    run_dir = finished_run(tmp_path, experiment_id="exp-digest", metrics={"map": 0.9})
    checkpoint = observe(run_dir).checkpoint
    assert checkpoint is not None
    loaded = load_registered_checkpoint(checkpoint["path"], project=tmp_path)
    assert loaded.sha256 == checkpoint["sha256"]
    [entry] = ModelRegistry(str(tmp_path)).list_models()
    assert entry["metrics"] == {"map": 0.9} == loaded.payload[METRICS_KEY]
    assert entry["metrics_source"] == "training_source"


def test_ctx_save_checkpoint_admits_a_state_naming_no_reserved_key(tmp_path):
    """An ordinary bespoke state, through a real ctx.save_checkpoint call."""
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import unbuilt_source
    from tests.tiny_trainer_fixtures import trainer_run

    run_dir = tmp_path / "out"
    run_dir.mkdir()
    run = trainer_run(training_config(unbuilt_source("regression"), {}), run_dir,
                      project=tmp_path, has_val_loader=False, id="auto-run-4")
    ctx = TrainContext(run=run, train_loader=None)

    path = ctx.save_checkpoint({STATE_DICT_KEY: {}})
    assert Path(path).is_file()


# Rail 8: the doctor lists a published bucket whose record names a digest no entry names, and
# stays silent on one whose digest an entry names.

def test_doctor_lists_a_bucket_naming_an_unregistered_digest_and_stays_silent_on_a_registered_one(
    tmp_path,
):
    import tcip_store

    from tcip_mcp.cli import doctor as doctor_module
    from tcip_mcp.dataset_layout import bucket_key

    ckpt = _unregistered(tmp_path)
    reg = _register(tmp_path, ckpt, name="good-model")
    good = _infer(tmp_path, ckpt, "baseline/2026-01-01")
    stale = _infer(tmp_path, ckpt, "stale/2026-01-01")
    # A bucket is any name its caller gives, a date segment or none.
    undated = _infer(tmp_path, ckpt, "undated-model")
    assert "error" not in good and "error" not in stale and "error" not in undated
    assert good["checkpoint_sha256"] == reg["sha256"]
    for bucket, digest in (("stale/2026-01-01", "a" * 64), ("undated-model", "b" * 64)):
        key = bucket_key(tmp_path, bucket)
        record = tcip_store.read(key)
        tcip_store.replace(key, {**record, "producer": {**record["producer"],
                                                        "checkpoint_sha256": digest}})

    findings: list = []
    doctor_module.check_registry(tmp_path, findings)

    messages = [m for _, m in findings]
    assert len([m for m in messages if "a" * 64 in m]) == 1, messages
    (undated_finding,) = [m for m in messages if "b" * 64 in m]
    assert "'undated-model'" in undated_finding
    assert not any(reg["sha256"] in m for m in messages)

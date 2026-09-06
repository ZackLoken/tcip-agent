"""The checkpoint digest rail: every delivery door recomputes the sha256 of the checkpoint bytes
it loaded and refuses one no registry entry names, before anything in it is unpickled.

See docs/audit/remediation/milestone-s/checkpoint-digest-design.md, sections 3-5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import build_model  # noqa: E402

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _bespoke_checkpoint(path: Path, *, stamp: dict | None = None, tile_size: int = 64) -> str:
    """A real, unpicklable tcip checkpoint at path, the platform's own producer's shape."""
    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": tile_size,
                                       "max_size": tile_size * 2},
                    "task": "detection"}
    payload = {"model_source": model_source,
              "model_state_dict": build_model({"model_source": model_source}).state_dict()}
    if stamp:
        payload.update(stamp)
    torch.save(payload, str(path))
    return str(path)


def _images(tmp_path: Path, n: int = 1, size: int = 100):
    from PIL import Image

    images_dir = tmp_path / "images"
    images_dir.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = images_dir / f"img{i}.png"
        Image.new("RGB", (size, size), (100, 100, 100)).save(p)
        paths.append(str(p))
    return images_dir, paths


def _register(tmp_path: Path, ckpt_path: str, *, name: str = "rail-model",
             tags: list[str] | None = None) -> dict:
    from tcip_mcp.tools.model_tools import register_model

    result = register_model(name=name, checkpoint_path=ckpt_path, config={},
                            project_path=str(tmp_path), tags=tags)
    assert "error" not in result, result
    return result


# Rail 1: an unregistered checkpoint the platform's own producer wrote is refused by name, at
# every door, writing nothing.

def test_run_inference_refuses_an_unregistered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(ckpt, images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]
    assert str(tmp_path) in r["error"]


def test_run_inference_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)
    out = tmp_path / "preds"

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" in r
    assert "register_model" in r["error"]
    assert not out.exists()


def test_deliver_per_image_counts_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path, monkeypatch):
    from tests import _operationalization_fixtures as fx

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    fx.seed_confirmed_count(tmp_path)
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)
    out_csv = tmp_path / "o.csv"

    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    r = deliver_per_image_counts(ckpt, str(images_dir), str(out_csv), trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "register_model" in r["error"]
    assert not out_csv.exists()


def test_evaluate_model_refuses_an_unregistered_checkpoint_by_bare_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.training_tools import evaluate_model

    r = evaluate_model(ckpt, str(images_dir), str(images_dir), task="detection")
    assert "error" in r
    assert "register_model" in r["error"]


def test_web_inference_worker_refuses_an_unregistered_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from tcip_web.routes.inference import InferenceJob, _worker

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(job_id="rail1", checkpoint_path=ckpt, images_dir=str(images_dir),
                       output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
                       slice_hw=(224, 224), overlap=0.2)
    _worker(job)
    assert job.status == "failed"
    assert "register_model" in job.error
    assert not out_dir.exists()
    assert job.done == 0


def test_triage_predictions_refuses_an_unregistered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    r = triage_predictions(ckpt, str(images_dir))
    assert "error" in r
    assert "register_model" in r["error"]


def test_triage_predictions_refuses_by_the_stated_project_path(tmp_path):
    """project_path, not just the process root, is where the load looks: registered under a
    root the call does not name, the checkpoint still refuses, naming the root it did name."""
    registered_root = tmp_path / "registered"
    registered_root.mkdir()
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    _register(registered_root, ckpt)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    r = triage_predictions(ckpt, str(images_dir), project_path=str(other_root))
    assert "error" in r
    assert "register_model" in r["error"]
    assert repr(str(other_root)) in r["error"]
    assert str(registered_root) not in r["error"]


def test_triage_predictions_admits_a_checkpoint_registered_under_the_stated_project_path(
    tmp_path,
):
    """The admitting direction, roles reversed: project_path naming the root the checkpoint
    really is registered under lets the same call through."""
    registered_root = tmp_path / "registered"
    registered_root.mkdir()
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    _register(registered_root, ckpt)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    r = triage_predictions(ckpt, str(images_dir), project_path=str(registered_root))
    assert "error" not in r, r


def test_calibrate_operating_point_script_refuses_an_unregistered_checkpoint(tmp_path):
    """Coverage, not a guard: --project-root and this refusal landed in the same change, so no
    baseline exists that parses the flag but lacks the check; prove_test_fails_before against
    f4413a14 reports SystemExit(2) from argparse, never this test's own assertion."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path, n=3)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for i in range(3):
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 100, 100)

    from tcip_mcp.cli.calibrate_operating_point import main

    rc = main([
        "--checkpoint", ckpt, "--trait", "bud",
        "--labels-dir", str(labels_dir), "--images-dir", str(images_dir),
        "--dataset-root", str(tmp_path), "--project-root", str(tmp_path),
    ])
    assert rc == 2


def test_calibrate_scalar_operating_point_refuses_an_unregistered_checkpoint(
    tmp_path, monkeypatch,
):
    """The checkpoint load runs before the cal/holdout split is locked, so a refused calibration
    leaves no lock record for the CSV's identity behind."""
    import tcip_store as ts

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path, n=4)
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text(
        "stem,rank\n" + "".join(f"img{i},{i % 3}\n" for i in range(4)), encoding="utf-8")
    out = tmp_path / "calib"

    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_key, cal_holdout_scope_root
    from tcip_mcp.pipelines.resolution import csv_dataset_hash
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point

    r = calibrate_scalar_operating_point(
        trait_name="bud_opening", task="ordinal", checkpoint_path=ckpt,
        images_dir=str(images_dir), csv_path=str(csv_path),
        criterion="quadratic_weighted_kappa", output_dir=str(out),
        dataset_root=str(tmp_path),
    )
    assert "error" in r
    assert "register_model" in r["error"]
    assert not (out / "ordinal_operating_point.json").exists()

    lock_key = cal_holdout_lock_key(
        csv_dataset_hash(str(csv_path)), scope_root=cal_holdout_scope_root(str(tmp_path)))
    assert not ts.exists(lock_key)


def test_review_priority_route_worker_fails_the_job_on_an_unregistered_checkpoint(tmp_path):
    """Drives the review-priority route's own worker directly, the way
    tests/test_inference_route_write_order.py:69 drives the inference worker."""
    pytest.importorskip("fastapi")
    from tcip_web.routes.review import PriorityQueueJob, _pq_worker

    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    job = PriorityQueueJob(job_id="rail1-pq", checkpoint_path=ckpt, images_dir=str(images_dir),
                          dataset_root=str(tmp_path), method="combined", budget=10)
    _pq_worker(job)
    assert job.status == "failed"
    assert "register_model" in job.error
    assert job.queue == []


def test_review_priority_route_worker_completes_the_job_with_a_registered_checkpoint(
    tmp_path, monkeypatch,
):
    """The admitting half: a checkpoint registered against the job's own platform_root runs
    _pq_worker to a completed job rather than a failed one."""
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    import tcip_mcp.pipelines.active_learning.helpers as al_helpers
    from tcip_web.routes.review import PriorityQueueJob, _pq_worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images_dir, _ = _images(tmp_path)

    monkeypatch.setattr(
        al_helpers, "build_scorer",
        lambda method, task: SimpleNamespace(score=lambda sources, model, device: []))

    job = PriorityQueueJob(job_id="rail7-pq", checkpoint_path=ckpt, images_dir=str(images_dir),
                          dataset_root=str(tmp_path), method="combined", budget=10,
                          platform_root=str(tmp_path))
    _pq_worker(job)
    assert job.status == "completed", job.error
    assert job.queue == []


# Rail 2: a registered checkpoint whose bytes are replaced (in place, or by rename) after
# registration is refused: the digest of the bytes actually loaded names no entry.

def test_run_inference_refuses_a_checkpoint_overwritten_in_place_after_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))

    # Replace the bytes in place, as a second torch.save over the same path.
    _bespoke_checkpoint(ckpt, tile_size=96)
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]


def test_run_inference_refuses_a_registered_checkpoint_replaced_by_rename(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))

    # A different checkpoint's bytes moved into the registered name by rename.
    other = _bespoke_checkpoint(tmp_path / "other.pt", tile_size=96)
    ckpt.unlink()
    Path(other).rename(ckpt)
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]


# Rail 4: registry entries naming one digest with disagreeing producers refuse the load by
# name; entries that agree, or one naming none beside one that does, admit it.

def test_two_entries_naming_one_digest_with_disagreeing_producers_refuse(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    data = ckpt.read_bytes()
    for exp_id, name in (("expA", "entry-a"), ("expB", "entry-b")):
        run_ckpt = tmp_path / f"{name}.pt"
        run_ckpt.write_bytes(data)
        create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
        assert "error" not in complete_run(exp_id, str(run_ckpt))
        reg = register_model_from_experiment(exp_id, str(run_ckpt), name=name)
        assert "error" not in reg, reg
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "expA" in r["error"] and "expB" in r["error"]


def test_two_entries_naming_one_digest_with_agreeing_producers_admit_it(tmp_path, monkeypatch):
    """Coverage: the admitting half of rail 4. The same run's weights registered under two
    distinct names both name the run's own experiment_id, agreeing by construction."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    create_experiment("expA", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("expA", str(ckpt))
    assert "error" not in register_model_from_experiment("expA", str(ckpt), name="entry-a")
    assert "error" not in register_model_from_experiment("expA", str(ckpt), name="entry-b")
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["experiment_id"] == "expA"


def test_one_entry_naming_none_beside_one_that_does_admits_the_named_producer(tmp_path, monkeypatch):
    """Coverage: an explicit-mode entry (experiment_id null) is not a vote for producer=None;
    it is simply ignored."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt), name="entry-untagged")
    create_experiment("expA", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("expA", str(ckpt))
    assert "error" not in register_model_from_experiment("expA", str(ckpt), name="entry-tagged")
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["experiment_id"] == "expA"


# Rail 5: an unregistered checkpoint is refused without being unpickled.

def _touch_marker(marker_path: str) -> "_SideEffectOnUnpickle":
    Path(marker_path).write_text("unpickled", encoding="utf-8")
    return _SideEffectOnUnpickle.__new__(_SideEffectOnUnpickle)


class _SideEffectOnUnpickle:
    """Its unpickling writes a marker file; the design read's own probe shape."""

    def __init__(self, marker_path: str) -> None:
        self._marker_path = marker_path

    def __reduce__(self):
        return (_touch_marker, (self._marker_path,))


def test_run_inference_refuses_without_unpickling_a_side_effect_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    marker = tmp_path / "unpickled.marker"
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": {},
               "carries_side_effect": _SideEffectOnUnpickle(str(marker))}, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    try:
        run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    except Exception:
        pass
    assert not marker.exists()  # the payload was never unpickled


# Rail 6: valid work the rail admits, through the doors that gate on measurement.

def test_run_inference_admits_a_registered_checkpoint_and_carries_its_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == reg["sha256"]


def test_run_inference_admits_the_same_checkpoint_copied_to_another_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt))
    copy = tmp_path / "copy.pt"
    copy.write_bytes(ckpt.read_bytes())
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(copy), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == reg["sha256"]


def test_run_inference_admits_a_raw_run_with_no_trait_and_stamps_unvalidated(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["validated"] is False


def test_run_inference_admits_a_second_checkpoint_of_a_run_registered_under_a_distinct_name(
    tmp_path, monkeypatch,
):
    """model_final beside model_best, registered explicit mode under a distinct name, is admitted:
    experiment mode would have replaced the run's own registered entry by name instead."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    best = tmp_path / "model_best.pt"
    _bespoke_checkpoint(best)
    _register(tmp_path, str(best), name="run-best")
    final = tmp_path / "model_final.pt"
    _bespoke_checkpoint(final, tile_size=96)
    _register(tmp_path, str(final), name="run-final")
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(final), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r


# Rail 7: a checkpoint a completed envelope run registered on completion (the platform's own
# producer, register_model_from_experiment) runs with no further step.

def test_a_completed_runs_registered_weights_run_through_run_inference_with_no_further_step(
    tmp_path, monkeypatch,
):
    from tcip_mcp.experiments import (
        complete_run, create_experiment, register_model_from_experiment, update_status,
    )

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail7"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    _bespoke_checkpoint(ckpt)
    assert "error" not in complete_run(exp_id, str(ckpt))
    reg = register_model_from_experiment(exp_id, str(ckpt))
    assert "error" not in reg, reg
    images_dir, _ = _images(tmp_path)

    import hashlib

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()


# Rail 11: a registration that fails after completion appends model_registration_failed.

def test_registration_failure_after_completion_is_recorded_in_the_audit_log(tmp_path, monkeypatch):
    import tcip_mcp.experiments as experiments_mod
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training.envelope import TrainContext, _finalize_run
    from tcip_mcp.pipelines.training.run_registry import create_run

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail11"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    run = create_run({"data": {}}, str(tmp_path / "out"))
    run.status = "completed"
    ckpt = tmp_path / "out" / "model_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    _bespoke_checkpoint(ckpt)

    def _boom(*a, **kw):
        raise ValueError("registration exploded")

    monkeypatch.setattr(experiments_mod, "register_model_from_experiment", _boom)

    ctx = TrainContext(run=run, train_loader=None, experiment_id=exp_id, final_weights=str(ckpt))
    _finalize_run(ctx)

    import tcip_store
    from tcip_mcp.audit import audit_log_key

    page = tcip_store.read_log(audit_log_key(tmp_path))
    events = [r for r in page.records if r["tool"] == "model_registration_failed"]
    assert len(events) == 1, events
    assert events[0]["arguments"]["weights_path"] == str(ckpt)
    assert "registration exploded" in events[0]["arguments"]["reason"]


# Rail 10: register_model and load_registered_checkpoint agree on one file's digest.

def test_registration_digest_and_load_digest_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt))

    from tcip_mcp.model_registry import load_registered_checkpoint

    checkpoint = load_registered_checkpoint(str(ckpt), project_path=str(tmp_path))
    assert checkpoint.sha256 == reg["sha256"]


# The version field: register_model_from_experiment routes through the same unpickle+version
# check load_registered_checkpoint uses, and the load-time refusal is a class doors catch.

def test_register_model_from_experiment_applies_the_same_version_check_as_load_registered_checkpoint(
    tmp_path, monkeypatch,
):
    """Before this fix, register_model_from_experiment ran its own torch.load(weights_only=False)
    with no version check, so a payload above the ceiling would register with its real metrics.
    Routed through the shared _load_verified_payload, this payload's metrics are read no
    differently than any other payload this reader cannot act on: empty, never fabricated."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt, stamp={"schema_version": 999, "metrics": {"map": 0.9}})
    create_experiment("exp-version", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("exp-version", str(ckpt))

    reg = register_model_from_experiment("exp-version", str(ckpt))
    assert "error" not in reg, reg
    assert reg["metrics"] == {}
    assert reg["metrics_source"] is None


def test_register_model_from_experiment_reads_metrics_through_the_shared_verified_load(
    tmp_path, monkeypatch,
):
    """The admitting half: an ordinary checkpoint (no schema_version key) still has its stamped
    metrics read and registered, through the platform's own producers (complete_run,
    register_model_from_experiment)."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt, stamp={"metrics": {"map": 0.9}})
    create_experiment("exp-plain", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("exp-plain", str(ckpt))

    reg = register_model_from_experiment("exp-plain", str(ckpt))
    assert "error" not in reg, reg
    assert reg["metrics"] == {"map": 0.9}
    assert reg["metrics_source"] == "trainer"


def test_ctx_save_checkpoint_refuses_a_state_naming_the_reserved_schema_version_key(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import create_run

    run = create_run({"data": {}}, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None)

    with pytest.raises(ValueError, match="schema_version"):
        ctx.save_checkpoint({"model_state_dict": {}, "schema_version": 2})


def test_ctx_save_checkpoint_admits_a_state_naming_no_reserved_key(tmp_path, monkeypatch):
    """The admitting half: an ordinary bespoke state, through a real ctx.save_checkpoint call."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tcip_mcp.pipelines.training.run_registry import create_run

    run = create_run({"data": {}}, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None)

    path = ctx.save_checkpoint({"model_state_dict": {}})
    assert Path(path).is_file()


def test_load_registered_checkpoints_version_refusal_is_caught_by_a_door(tmp_path, monkeypatch):
    """The load-time version refusal is UnregisteredCheckpoint, the class every checkpoint door
    already catches, not a bare ValueError that would surface as an unhandled 500."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt, stamp={"schema_version": 999})
    _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r


# Rail 3: a sweep record edited after the run is refused by _calibration_evidence through
# run_inference, naming both digests.

def _stand_in_calibration(monkeypatch, calibration_pipeline, labels_dir):
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tests._dense_op_fixtures import dense_records

    n_images, objects = 20, 80
    inputs = {
        "dataset_hash": "H",
        "calibration_records": dense_records(n_images=n_images, objects_per_image=objects,
                                             id_prefix="c", fp_pattern=[1] * n_images, score=0.9,
                                             fp_score=0.05),
        "holdout_records": dense_records(n_images=n_images, objects_per_image=objects,
                                         id_prefix="h", shift=5.0, fp_pattern=[1] * n_images,
                                         score=0.9, fp_score=0.05),
        "tiled": False, "tile_size": None, "tile_size_source": "default",
        "staged_conf_floor": 0.01,
    }
    bundle = resolve_operating_point("bud_opening", experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(labels_dir)}}}
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))


def test_run_inference_refuses_a_sweep_record_edited_after_the_run(tmp_path, monkeypatch):
    """Coverage, not a fail-before guard: the spy stubs _run_inference_verified, a symbol the
    family's baseline (f4413a14) does not carry, so a run against that baseline dies in setup
    rather than on the assertion this test names. See the driven-through-real-doors version
    below for the guard."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)
    _stand_in_calibration(monkeypatch, calibration_pipeline, tmp_path)

    real_verified = itools._run_inference_verified
    captured: dict = {}

    def _spy(*a, **kw):
        result = real_verified(*a, **kw)
        captured.clear()
        captured.update(result)
        return result

    monkeypatch.setattr(itools, "_run_inference_verified", _spy)

    out = tmp_path / "preds"
    r = itools.run_inference(str(ckpt), str(images_dir), output_dir=str(out), trait="bud_opening",
                             calibration_labels_dir=str(tmp_path))
    assert "error" not in r, r

    from tcip_store import store

    identity = captured["calibration_evidence_key"]
    key = itools.calibration_curve_key(identity)
    body = store.read(key)
    body["calibration_evidence"]["inputs"]["dataset_hash"] = "tampered"
    store.replace(key, body)

    monkeypatch.setattr(itools, "_run_inference_verified", lambda *a, **kw: dict(captured))
    out2 = tmp_path / "preds2"
    refused = itools.run_inference(str(ckpt), str(images_dir), output_dir=str(out2), trait="bud_opening",
                                   calibration_labels_dir=str(tmp_path))
    assert "error" in refused
    assert identity in refused["error"]
    assert not out2.exists()


def test_run_inference_refuses_a_sweep_record_edited_after_the_run_through_real_doors(
    tmp_path, monkeypatch,
):
    """The fail-before guard for rail 3, driven through real doors with no stub of
    _run_inference_verified (a symbol the family's baseline, f4413a14, does not carry): the
    calibration itself is the same deterministic stand-in the coverage test above uses (a real
    model pass is not reproducible byte for byte across two separate calls, see
    _stand_in_calibration), so two real run_inference calls over it agree
    on one identity. store.replace (a baseline-old primitive) is patched to skip only the
    confidence-sweep write on the second call, so the first call's tampered record is the one
    run_inference reads back and refuses on."""
    import tcip_store.store as store_mod

    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    from tests._verified_checkpoint_fixtures import run_inference_verified

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)
    _stand_in_calibration(monkeypatch, calibration_pipeline, tmp_path)

    r1 = run_inference_verified(str(ckpt), images_dir=str(images_dir), trait="bud_opening",
                                calibration_labels_dir=str(tmp_path))
    assert "error" not in r1, r1
    identity = r1["calibration_evidence_key"]

    from tcip_store import store

    key = itools.calibration_curve_key(identity)
    body = store.read(key)
    body["calibration_evidence"]["inputs"]["dataset_hash"] = "tampered"
    store.replace(key, body)

    real_replace = store_mod.replace

    def _skip_the_sweep_write(k, value, **kw):
        if k.store == itools.CONFIDENCE_SWEEP_STORE:
            return None
        return real_replace(k, value, **kw)

    monkeypatch.setattr(store_mod, "replace", _skip_the_sweep_write)

    out = tmp_path / "preds"
    refused = itools.run_inference(str(ckpt), str(images_dir), output_dir=str(out), trait="bud_opening",
                                   calibration_labels_dir=str(tmp_path))
    assert "error" in refused
    assert identity in refused["error"]
    assert not out.exists()


def test_run_inference_refuses_a_sweep_whose_evidence_the_codec_cannot_carry(
    tmp_path, monkeypatch,
):
    """A body the codec refuses (RECORD_JSON's allow_nan=False; a NaN in the resolver's inputs
    is the natural one) makes the door return its own error and write no bucket, rather than
    the swallowed warning the pre-family code left behind. The admitting half of this branch is
    already covered: test_a_bespoke_module_exposing_its_own_knob_reaches_a_validated_point in
    test_detector_operating_point_holder.py is an ordinary calibrated run surviving it."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    _register(tmp_path, ckpt)

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for i, size in enumerate((32, 40, 48, 56, 64, 72)):
        Image.new("RGB", (size, size), (100, 100, 100)).save(images_dir / f"img{i}.png")
        box = BBox(size * 0.25, size * 0.25, size * 0.75, size * 0.75)
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=box)], size, size)

    real_calibrate = calibration_pipeline.calibrate_operating_point

    def _nan_evidence(*a, **kw):
        bundle, dh, n_excluded, evidence = real_calibrate(*a, **kw)
        evidence["inputs"]["staged_conf_floor"] = float("nan")
        return bundle, dh, n_excluded, evidence

    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point", _nan_evidence)

    out = tmp_path / "preds"
    refused = itools.run_inference(ckpt, str(images_dir), output_dir=str(out), trait="bud_opening",
                                   calibration_labels_dir=str(labels_dir))
    assert "error" in refused
    assert "could not be kept" in refused["error"]
    assert not out.exists()


# Rail 8: the doctor lists a prediction bucket whose stamp digest no entry names, and stays
# silent on one whose digest an entry names.

def test_doctor_lists_a_prerail_bucket_and_stays_silent_on_a_registered_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt), name="good-model")

    stale_ckpt = tmp_path / "stale.pt"
    _bespoke_checkpoint(stale_ckpt, tile_size=96)
    _register(tmp_path, str(stale_ckpt), name="stale-model")

    images_dir, _ = _images(tmp_path)
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.tools.inference_tools import run_inference

    good_dir = tmp_path / "predictions" / "baseline" / "2026-01-01"
    r_good = run_inference(str(ckpt), str(images_dir), output_dir=str(good_dir), tile=False)
    assert "error" not in r_good, r_good
    assert r_good["checkpoint_sha256"] == reg["sha256"]

    stale_dir = tmp_path / "predictions" / "stale" / "2026-01-01"
    r_stale = run_inference(str(stale_ckpt), str(images_dir), output_dir=str(stale_dir), tile=False)
    assert "error" not in r_stale, r_stale

    # An undated bucket (prediction_dir(root, model, None), no date segment) is a real platform
    # shape (a bare-path export, the web tab's default), not only the dated ones above.
    undated_ckpt = tmp_path / "undated.pt"
    _bespoke_checkpoint(undated_ckpt, tile_size=112)
    _register(tmp_path, str(undated_ckpt), name="undated-model")
    undated_dir = prediction_dir(tmp_path, "undated-model", None)
    r_undated = run_inference(str(undated_ckpt), str(images_dir), output_dir=str(undated_dir), tile=False)
    assert "error" not in r_undated, r_undated

    # Supersede stale-model's and undated-model's entries under the same name: their digests no
    # longer name any entry, the same pre-rail state a bucket already on disk can be in.
    replacement = tmp_path / "replacement.pt"
    _bespoke_checkpoint(replacement, tile_size=128)
    _register(tmp_path, str(replacement), name="stale-model")

    undated_replacement = tmp_path / "undated_replacement.pt"
    _bespoke_checkpoint(undated_replacement, tile_size=144)
    _register(tmp_path, str(undated_replacement), name="undated-model")

    from tcip_mcp.cli import doctor as doctor_module

    findings: list = []
    doctor_module.check_registry(tmp_path, findings)
    messages = [m for _, m in findings]
    stale_findings = [m for m in messages if r_stale["checkpoint_sha256"] in m]
    assert len(stale_findings) == 1, messages
    undated_findings = [m for m in messages if r_undated["checkpoint_sha256"] in m]
    assert len(undated_findings) == 1, messages
    assert str(undated_dir.relative_to(tmp_path)) in undated_findings[0]
    assert not any(r_good["checkpoint_sha256"] in m for m in messages)

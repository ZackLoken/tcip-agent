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

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


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
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(ckpt, images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]
    assert str(tmp_path) in r["error"]


def test_export_predictions_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)
    out = tmp_path / "preds"

    from tcip_mcp.tools.inference_tools import export_predictions

    r = export_predictions(ckpt, str(images_dir), str(out), tile=False)
    assert "error" in r
    assert "register_model" in r["error"]
    assert not out.exists()


def test_tabulate_counts_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path, monkeypatch):
    from tests import _operationalization_fixtures as fx

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    fx.seed_confirmed_count(tmp_path)
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)
    out_csv = tmp_path / "o.csv"

    from tcip_mcp.tools.inference_tools import tabulate_counts

    r = tabulate_counts(ckpt, str(images_dir), str(out_csv), trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "register_model" in r["error"]
    assert not out_csv.exists()


def test_evaluate_model_refuses_an_unregistered_checkpoint_by_bare_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.training_tools import evaluate_model

    r = evaluate_model(ckpt, str(images_dir), str(images_dir), task="detection")
    assert "error" in r
    assert "register_model" in r["error"]


def test_web_inference_worker_refuses_an_unregistered_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from tcip_web.routes.inference import InferenceJob, _worker

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(job_id="rail1", checkpoint_path=ckpt, images_dir=str(images_dir),
                       output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
                       slice_hw=(224, 224), overlap=0.2)
    _worker(job)
    assert job.status == "failed"
    assert "register_model" in job.error
    assert not (out_dir / "img0.json").exists()
    assert job.done == 0


def test_prioritize_review_queue_refuses_an_unregistered_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import prioritize_review_queue

    r = prioritize_review_queue(ckpt, str(images_dir), strategy="confidence_triage")
    assert "error" in r
    assert "register_model" in r["error"]


def test_prioritize_review_queue_refuses_by_the_stated_project_path(tmp_path):
    """Coverage: project_path, not just the process root, is where the load looks."""
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import prioritize_review_queue

    r = prioritize_review_queue(ckpt, str(images_dir), strategy="confidence_triage",
                                project_path=str(tmp_path))
    assert "error" in r
    assert "register_model" in r["error"]


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
            [Annotation(subject="catkin", geometry=BBox(10, 10, 40, 40))], 100, 100)

    from scripts.calibrate_operating_point import main

    rc = main([
        "--checkpoint", ckpt, "--trait", "catkin",
        "--labels-dir", str(labels_dir), "--images-dir", str(images_dir),
        "--dataset-root", str(tmp_path), "--project-root", str(tmp_path),
    ])
    assert rc == 2


def test_calibrate_ordinal_regression_operating_point_refuses_an_unregistered_checkpoint(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = _bespoke_checkpoint(tmp_path / "m.pt")
    images_dir, _ = _images(tmp_path, n=4)
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text(
        "stem,rank\n" + "".join(f"img{i},{i % 3}\n" for i in range(4)), encoding="utf-8")
    out = tmp_path / "calib"

    from tcip_mcp.tools.phenology_tools import calibrate_ordinal_regression_operating_point

    r = calibrate_ordinal_regression_operating_point(
        trait_name="catkin", task="ordinal", checkpoint_path=ckpt,
        images_dir=str(images_dir), csv_path=str(csv_path),
        criterion="quadratic_weighted_kappa", output_dir=str(out),
        dataset_root=str(tmp_path),
    )
    assert "error" in r
    assert "register_model" in r["error"]
    assert not (out / "ordinal_operating_point.json").exists()


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


# Rail 2: a registered checkpoint whose bytes are replaced (in place, or by rename) after
# registration is refused: the digest of the bytes actually loaded names no entry.

def test_run_inference_refuses_a_checkpoint_overwritten_in_place_after_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))

    # Replace the bytes in place, as a second torch.save over the same path.
    _bespoke_checkpoint(ckpt, tile_size=96)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]


def test_run_inference_refuses_a_registered_checkpoint_replaced_by_rename(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))

    # A different checkpoint's bytes moved into the registered name by rename.
    other = _bespoke_checkpoint(tmp_path / "other.pt", tile_size=96)
    ckpt.unlink()
    Path(other).rename(ckpt)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]


# Rail 4: registry entries naming one digest with disagreeing producers refuse the load by
# name; entries that agree, or one naming none beside one that does, admit it.

def test_two_entries_naming_one_digest_with_disagreeing_producers_refuse(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
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

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "expA" in r["error"] and "expB" in r["error"]


def test_two_entries_naming_one_digest_with_agreeing_producers_admit_it(tmp_path, monkeypatch):
    """Coverage: the admitting half of rail 4. The same run's weights registered under two
    distinct names both name the run's own experiment_id, agreeing by construction."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    create_experiment("expA", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("expA", str(ckpt))
    assert "error" not in register_model_from_experiment("expA", str(ckpt), name="entry-a")
    assert "error" not in register_model_from_experiment("expA", str(ckpt), name="entry-b")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["experiment_id"] == "expA"


def test_one_entry_naming_none_beside_one_that_does_admits_the_named_producer(tmp_path, monkeypatch):
    """Coverage: an explicit-mode entry (experiment_id null) is not a vote for producer=None;
    it is simply ignored."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment

    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt), name="entry-untagged")
    create_experiment("expA", {"model_source": {"builder": "x:y"}})
    assert "error" not in complete_run("expA", str(ckpt))
    assert "error" not in register_model_from_experiment("expA", str(ckpt), name="entry-tagged")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

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
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    marker = tmp_path / "unpickled.marker"
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state_dict": {},
               "carries_side_effect": _SideEffectOnUnpickle(str(marker))}, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    try:
        run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    except Exception:
        pass
    assert not marker.exists()  # the payload was never unpickled


# Rail 6: valid work the rail admits, through the doors that gate on measurement.

def test_run_inference_admits_a_registered_checkpoint_and_carries_its_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == reg["sha256"]


def test_run_inference_admits_the_same_checkpoint_copied_to_another_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt))
    copy = tmp_path / "copy.pt"
    copy.write_bytes(ckpt.read_bytes())
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(copy), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == reg["sha256"]


def test_run_inference_admits_a_raw_run_with_no_trait_and_stamps_unvalidated(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["validated"] is False


def test_run_inference_admits_a_second_checkpoint_of_a_run_registered_under_a_distinct_name(
    tmp_path, monkeypatch,
):
    """model_final beside model_best, registered explicit mode under a distinct name, is admitted:
    experiment mode would have replaced the run's own registered entry by name instead."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    best = tmp_path / "model_best.pt"
    _bespoke_checkpoint(best)
    _register(tmp_path, str(best), name="run-best")
    final = tmp_path / "model_final.pt"
    _bespoke_checkpoint(final, tile_size=96)
    _register(tmp_path, str(final), name="run-final")
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.inference_tools import run_inference

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

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
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

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()


# Rail 11: a registration that fails after completion appends model_registration_failed.

def test_registration_failure_after_completion_is_recorded_in_the_audit_log(tmp_path, monkeypatch):
    import tcip_mcp.experiments as experiments_mod
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training.envelope import TrainContext, _finalize_run
    from tcip_mcp.pipelines.training.generic_trainer import create_run

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
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
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt))

    from tcip_mcp.model_registry import load_registered_checkpoint

    checkpoint = load_registered_checkpoint(str(ckpt), project_path=str(tmp_path))
    assert checkpoint.sha256 == reg["sha256"]


# Rail 3: a sweep record edited after the run is refused by _calibration_evidence through
# export_predictions, naming both digests.

def _stand_in_calibration(monkeypatch, itools, labels_dir):
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
    bundle = resolve_operating_point("catkin", experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(labels_dir)}}}
    monkeypatch.setattr(itools, "_calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))


def test_export_predictions_refuses_a_sweep_record_edited_after_the_run(tmp_path, monkeypatch):
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    _register(tmp_path, str(ckpt))
    images_dir, _ = _images(tmp_path)
    _stand_in_calibration(monkeypatch, itools, tmp_path)

    real_verified = itools._run_inference_verified
    captured: dict = {}

    def _spy(*a, **kw):
        result = real_verified(*a, **kw)
        captured.clear()
        captured.update(result)
        return result

    monkeypatch.setattr(itools, "_run_inference_verified", _spy)

    out = tmp_path / "preds"
    r = itools.export_predictions(str(ckpt), str(images_dir), str(out), trait="catkin",
                                  calibration_labels_dir=str(tmp_path))
    assert "error" not in r, r

    from tcip_store import store

    identity = captured["calibration_evidence_key"]
    key = itools.confidence_sweep_key(identity)
    body = store.read(key)
    body["calibration_evidence"]["inputs"]["dataset_hash"] = "tampered"
    store.replace(key, body)

    monkeypatch.setattr(itools, "_run_inference_verified", lambda *a, **kw: dict(captured))
    out2 = tmp_path / "preds2"
    refused = itools.export_predictions(str(ckpt), str(images_dir), str(out2), trait="catkin",
                                        calibration_labels_dir=str(tmp_path))
    assert "error" in refused
    assert identity in refused["error"]
    assert not out2.exists()


# Rail 8: the doctor lists a prediction bucket whose stamp digest no entry names, and stays
# silent on one whose digest an entry names.

def test_doctor_lists_a_prerail_bucket_and_stays_silent_on_a_registered_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    ckpt = tmp_path / "m.pt"
    _bespoke_checkpoint(ckpt)
    reg = _register(tmp_path, str(ckpt), name="good-model")

    stale_ckpt = tmp_path / "stale.pt"
    _bespoke_checkpoint(stale_ckpt, tile_size=96)
    _register(tmp_path, str(stale_ckpt), name="stale-model")

    images_dir, _ = _images(tmp_path)
    from tcip_mcp.tools.inference_tools import export_predictions

    good_dir = tmp_path / "predictions" / "baseline" / "2026-01-01"
    r_good = export_predictions(str(ckpt), str(images_dir), str(good_dir), tile=False)
    assert "error" not in r_good, r_good
    assert r_good["checkpoint_sha256"] == reg["sha256"]

    stale_dir = tmp_path / "predictions" / "stale" / "2026-01-01"
    r_stale = export_predictions(str(stale_ckpt), str(images_dir), str(stale_dir), tile=False)
    assert "error" not in r_stale, r_stale

    # Supersede stale-model's entry under the same name: its digest no longer names any entry,
    # the same pre-rail state a bucket already on disk can be in.
    replacement = tmp_path / "replacement.pt"
    _bespoke_checkpoint(replacement, tile_size=128)
    _register(tmp_path, str(replacement), name="stale-model")

    import importlib.util

    doctor_path = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"
    spec = importlib.util.spec_from_file_location("tcip_digest_rail_doctor", doctor_path)
    doctor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doctor_module)

    findings: list = []
    doctor_module.check_registry(tmp_path, findings)
    messages = [m for _, m in findings]
    stale_findings = [m for m in messages if r_stale["checkpoint_sha256"] in m]
    assert len(stale_findings) == 1, messages
    assert not any(r_good["checkpoint_sha256"] in m for m in messages)

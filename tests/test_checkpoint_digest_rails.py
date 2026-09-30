"""The checkpoint digest rail: every delivery door recomputes the sha256 of the checkpoint bytes
it loaded and refuses one no completed run of the project and no registry entry names, before
anything in it is unpickled.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tests._verified_checkpoint_fixtures import (  # noqa: E402
    finished_run,
    foreign_checkpoint,
    registered_checkpoint,
)

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def _unregistered(tmp_path: Path, **kwargs) -> str:
    """A checkpoint a completed run of another project produced, which nothing in this one
    names."""
    return registered_checkpoint(tmp_path.parent / "other_project", **kwargs)


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

    result = register_model(tmp_path, name=name, checkpoint_path=ckpt_path, config={}, tags=tags)
    assert "error" not in result, result
    return result


# Rail 1: an unregistered checkpoint the platform's own producer wrote is refused by name, at
# every door, writing nothing.

def test_run_inference_refuses_an_unregistered_checkpoint(tmp_path):
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, ckpt, images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]
    assert repr(str(tmp_path)) in r["error"]


def test_run_inference_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path):
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)
    out = tmp_path / "preds"

    from tcip_mcp.tools.inference_tools import run_inference

    r = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), tile=False)
    assert "error" in r
    assert "register_model" in r["error"]
    assert not out.exists()


def test_deliver_per_image_counts_refuses_an_unregistered_checkpoint_and_writes_nothing(tmp_path):
    from tests import _trait_fixtures as fx

    fx.seed_confirmed_count(tmp_path)
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)
    out_csv = tmp_path / "o.csv"

    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    r = deliver_per_image_counts(tmp_path, ckpt, str(images_dir), str(out_csv), trait=fx.COUNT_TRAIT)
    assert "error" in r
    assert "register_model" in r["error"]
    assert not out_csv.exists()


def test_evaluate_model_refuses_an_unregistered_checkpoint_by_bare_path(tmp_path):
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.training_tools import evaluate_model

    r = evaluate_model(tmp_path, ckpt, str(images_dir), str(images_dir))
    assert "error" in r
    assert "register_model" in r["error"]


def test_web_inference_worker_refuses_an_unregistered_checkpoint(tmp_path):
    pytest.importorskip("fastapi")
    from tcip_web.routes.inference import InferenceJob, _worker

    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)
    out_dir = tmp_path / "out"

    job = InferenceJob(job_id="rail1", checkpoint_path=ckpt, images_dir=str(images_dir),
                       output_dir=str(out_dir), tile=False, conf=0.25, cross_tile_nms=0.7,
                       overlap=0.2, project=str(tmp_path))
    _worker(job)
    assert job.status == "failed"
    assert "register_model" in job.error
    assert not out_dir.exists()
    assert job.done == 0


def test_triage_predictions_refuses_an_unregistered_checkpoint(tmp_path):
    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tcip_mcp.tools.feedback_tools import triage_predictions

    r = triage_predictions(tmp_path, ckpt, str(images_dir))
    assert "error" in r
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
    assert "error" in r
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

    r = triage_predictions(registered_root, ckpt, str(images_dir))
    assert "error" not in r, r


def test_calibrate_operating_point_script_refuses_an_unregistered_checkpoint(tmp_path, project):
    """Coverage, not a guard: no baseline separates ``--project`` from the refusal it carries, so
    nothing can be observed failing without the check."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    ckpt = _unregistered(tmp_path)
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
        "--dataset-root", str(tmp_path), "--project", str(project),
    ])
    assert rc == 2


def test_calibrate_scalar_operating_point_refuses_an_unregistered_checkpoint(tmp_path):
    """The checkpoint load runs before the cal/holdout split is locked, so a refused calibration
    leaves no lock record for the CSV's identity behind."""
    import tcip_store as ts

    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path, n=4)
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text(
        "stem,rank\n" + "".join(f"img{i},{i % 3}\n" for i in range(4)), encoding="utf-8")
    out = tmp_path / "calib"

    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_key, cal_holdout_scope_root
    from tcip_mcp.pipelines.resolution import csv_dataset_hash
    from tcip_mcp.tools.calibration_tools import calibrate_scalar_operating_point

    r = calibrate_scalar_operating_point(
        tmp_path, trait_name="bud_opening", checkpoint_path=ckpt,
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
    tests/test_inference_route_write_order.py drives the inference worker."""
    pytest.importorskip("fastapi")
    from tcip_web.routes.review import PriorityQueueJob, _pq_worker

    ckpt = _unregistered(tmp_path)
    images_dir, _ = _images(tmp_path)

    job = PriorityQueueJob(job_id="rail1-pq", checkpoint_path=ckpt, images_dir=str(images_dir),
                          dataset_root=str(tmp_path), method="combined", budget=10,
                          project=str(tmp_path))
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

    import tcip_mcp.pipelines.active_learning.helpers as al_helpers
    from tcip_web.routes.review import PriorityQueueJob, _pq_worker

    ckpt = registered_checkpoint(tmp_path)
    images_dir, _ = _images(tmp_path)

    monkeypatch.setattr(
        al_helpers, "build_scorer",
        lambda method, task: SimpleNamespace(score=lambda sources, predictor: []))

    job = PriorityQueueJob(job_id="rail7-pq", checkpoint_path=ckpt, images_dir=str(images_dir),
                          dataset_root=str(tmp_path), method="combined", budget=10,
                          project=str(tmp_path))
    _pq_worker(job)
    assert job.status == "completed", job.error
    assert job.queue == []


# Rail 2: a registered checkpoint whose bytes are replaced (in place, or by rename) after
# registration is refused: the digest of the bytes actually loaded names no entry.

def test_run_inference_refuses_a_checkpoint_overwritten_in_place_after_registration(tmp_path):
    ckpt = Path(foreign_checkpoint(tmp_path))

    # Replace the bytes in place with another run's checkpoint.
    ckpt.write_bytes(Path(_unregistered(tmp_path)).read_bytes())
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]


def test_run_inference_refuses_a_registered_checkpoint_replaced_by_rename(tmp_path):
    ckpt = Path(foreign_checkpoint(tmp_path))

    # A different checkpoint's bytes moved into the registered name by rename.
    other = tmp_path / "other.pt"
    other.write_bytes(Path(_unregistered(tmp_path)).read_bytes())
    ckpt.unlink()
    other.rename(ckpt)
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" in r
    assert "register_model" in r["error"]


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

    owner = register_model(tmp_path, name="entry-untagged", checkpoint_path=str(copy), config={})

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
    registered = register_model(tmp_path, name="foreign-first", checkpoint_path=elsewhere["path"],
                                config={})
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
    assert "error" not in register_model(tmp_path, name="second-name", checkpoint_path=str(copy),
                                         config={})

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
    torch.save({"model_state_dict": {},
               "carries_side_effect": _SideEffectOnUnpickle(str(marker))}, str(ckpt))
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    try:
        run_inference(tmp_path, str(ckpt), images_dir=str(images_dir), device="cpu", tile=False)
    except Exception:
        pass
    assert not marker.exists()  # the payload was never unpickled


# Rail 6: valid work the rail admits, through the doors that gate on measurement.

def test_run_inference_admits_a_registered_checkpoint_and_carries_its_digest(tmp_path):
    ckpt = foreign_checkpoint(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, ckpt, images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()


def test_run_inference_admits_the_same_checkpoint_copied_to_another_path(tmp_path):
    ckpt = foreign_checkpoint(tmp_path)
    copy = tmp_path / "copy.pt"
    copy.write_bytes(Path(ckpt).read_bytes())
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, str(copy), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["checkpoint_sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()


def test_run_inference_admits_a_raw_run_with_no_trait_and_stamps_unvalidated(tmp_path):
    ckpt = foreign_checkpoint(tmp_path)
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, ckpt, images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r
    assert r["validated"] is False


def _best_and_final(ctx) -> None:
    """A body saving two checkpoints: the model as built as model_best, and a second build as
    model_final."""
    ctx.save_checkpoint({"model_state_dict": ctx.build_model().state_dict()}, "model_best")
    ctx.save_checkpoint({"model_state_dict": ctx.build_model().state_dict()}, "model_final")


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
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, str(final), images_dir=str(images_dir), device="cpu", tile=False)
    assert "error" not in r, r


# Rail 7: a checkpoint a completed envelope run registered by completing runs with no further
# step, and carries the run as its producer.

def test_a_completed_runs_weights_run_through_run_inference_with_no_further_step(tmp_path):
    ckpt = registered_checkpoint(tmp_path, experiment_id="exp-rail7")
    images_dir, _ = _images(tmp_path)

    from tests._verified_checkpoint_fixtures import run_inference_verified as run_inference

    r = run_inference(tmp_path, ckpt, images_dir=str(images_dir), device="cpu", tile=False)
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
    assert entry["metrics"] == {"map": 0.9} == loaded.payload["metrics"]
    assert entry["metrics_source"] == "training_source"


def test_ctx_save_checkpoint_admits_a_state_naming_no_reserved_key(tmp_path):
    """An ordinary bespoke state, through a real ctx.save_checkpoint call."""
    from tcip_mcp.pipelines.training.envelope import TrainContext
    from tests.tiny_trainer_fixtures import trainer_run

    run_dir = tmp_path / "out"
    run_dir.mkdir()
    run = trainer_run({"model_source": {"task": "regression"}, "data": {}}, run_dir,
                      project=tmp_path, has_val_loader=False, id="auto-run-4")
    ctx = TrainContext(run=run, train_loader=None)

    path = ctx.save_checkpoint({"model_state_dict": {}})
    assert Path(path).is_file()


# Rail 3: a sweep record edited after the run is refused by _calibration_evidence through
# run_inference, naming both digests.

def _stand_in_calibration(project, monkeypatch, calibration_pipeline, labels_dir):
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
        "slicing": None, "tile_size": None, "tile_size_source": "default",
        "staged_conf_floor": 0.01,
    }
    bundle = resolve_operating_point("bud_opening", project=project, experiment_id=None, **inputs)
    evidence = {"resolver": "resolve_operating_point", "inputs": inputs,
                "reference_inputs": {"label_dirs": {"calibration": str(labels_dir)}}}
    monkeypatch.setattr(calibration_pipeline, "calibrate_operating_point",
                        lambda *a, **k: (bundle, "H", 0, evidence))


def test_run_inference_refuses_a_sweep_record_edited_after_the_run(tmp_path, monkeypatch):
    """Coverage: the spy stubs ``_run_inference_verified``, so this exercises the refusal
    without a real model pass. The version below drives the same refusal through real doors."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools

    ckpt = foreign_checkpoint(tmp_path)
    images_dir, _ = _images(tmp_path)
    _stand_in_calibration(tmp_path, monkeypatch, calibration_pipeline, tmp_path)

    real_verified = itools._run_inference_verified
    captured: dict = {}

    def _spy(*a, **kw):
        result = real_verified(*a, **kw)
        captured.clear()
        captured.update(result)
        return result

    monkeypatch.setattr(itools, "_run_inference_verified", _spy)

    out = tmp_path / "preds"
    r = itools.run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), trait="bud_opening",
                             calibration_labels_dir=str(tmp_path))
    assert "error" not in r, r

    from tcip_store import store

    identity = captured["calibration_evidence_key"]
    key = itools.calibration_curve_key(tmp_path, identity)
    body = store.read(key)
    body["calibration_evidence"]["inputs"]["dataset_hash"] = "tampered"
    store.replace(key, body)

    monkeypatch.setattr(itools, "_run_inference_verified", lambda *a, **kw: dict(captured))
    out2 = tmp_path / "preds2"
    refused = itools.run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out2), trait="bud_opening",
                                   calibration_labels_dir=str(tmp_path))
    assert "error" in refused
    assert identity in refused["error"]
    assert not out2.exists()


def test_run_inference_refuses_a_sweep_record_edited_after_the_run_through_real_doors(
    tmp_path, monkeypatch,
):
    """Rail 3 driven through real doors, with no stub of ``_run_inference_verified``. The
    calibration is the same deterministic stand-in the coverage test above uses, since a real
    model pass is not reproducible byte for byte across two separate calls (see
    ``_stand_in_calibration``), so two real run_inference calls over it agree on one identity.
    ``store.replace`` is patched to skip only the confidence-sweep write on the second call, so
    the first call's tampered record is the one run_inference reads back and refuses on."""
    import tcip_store.store as store_mod

    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    from tests._verified_checkpoint_fixtures import run_inference_verified

    ckpt = foreign_checkpoint(tmp_path)
    images_dir, _ = _images(tmp_path)
    _stand_in_calibration(tmp_path, monkeypatch, calibration_pipeline, tmp_path)

    r1 = run_inference_verified(tmp_path, ckpt, images_dir=str(images_dir), trait="bud_opening",
                                calibration_labels_dir=str(tmp_path))
    assert "error" not in r1, r1
    identity = r1["calibration_evidence_key"]

    from tcip_store import store

    key = itools.calibration_curve_key(tmp_path, identity)
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
    refused = itools.run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), trait="bud_opening",
                                   calibration_labels_dir=str(tmp_path))
    assert "error" in refused
    assert identity in refused["error"]
    assert not out.exists()


def test_run_inference_refuses_a_sweep_whose_evidence_the_codec_cannot_carry(
    tmp_path, monkeypatch,
):
    """A body the codec refuses (RECORD_JSON's allow_nan=False; a NaN in the resolver's inputs
    is the natural one) makes the door return its own error and write no bucket, never a
    swallowed warning. The admitting half of this branch is already covered:
    test_a_bespoke_module_exposing_its_own_knob_reaches_a_validated_point in
    test_detector_operating_point_holder.py is an ordinary calibrated run surviving it."""
    import tcip_mcp.pipelines.calibration as calibration_pipeline
    import tcip_mcp.tools.inference_tools as itools
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    ckpt = foreign_checkpoint(tmp_path)

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
    refused = itools.run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(out), trait="bud_opening",
                                   calibration_labels_dir=str(labels_dir))
    assert "error" in refused
    assert "could not be kept" in refused["error"]
    assert not out.exists()


# Rail 8: the doctor lists a prediction bucket whose stamp digest no entry names, and stays
# silent on one whose digest an entry names.

def test_doctor_lists_a_prerail_bucket_and_stays_silent_on_a_registered_one(tmp_path):
    ckpt = _unregistered(tmp_path)
    reg = _register(tmp_path, ckpt, name="good-model")

    stale_ckpt = _unregistered(tmp_path)
    _register(tmp_path, stale_ckpt, name="stale-model")

    images_dir, _ = _images(tmp_path)
    from tcip_mcp.dataset_layout import prediction_dir
    from tcip_mcp.tools.inference_tools import run_inference

    good_dir = tmp_path / "predictions" / "baseline" / "2026-01-01"
    r_good = run_inference(tmp_path, ckpt, str(images_dir), output_dir=str(good_dir), tile=False)
    assert "error" not in r_good, r_good
    assert r_good["checkpoint_sha256"] == reg["sha256"]

    stale_dir = tmp_path / "predictions" / "stale" / "2026-01-01"
    r_stale = run_inference(tmp_path, stale_ckpt, str(images_dir), output_dir=str(stale_dir), tile=False)
    assert "error" not in r_stale, r_stale

    # An undated bucket (prediction_dir(root, model, None), no date segment) is a real platform
    # shape (a bare-path export, the web tab's default), not only the dated ones above.
    undated_ckpt = _unregistered(tmp_path)
    _register(tmp_path, undated_ckpt, name="undated-model")
    undated_dir = prediction_dir(tmp_path, "undated-model", None)
    r_undated = run_inference(tmp_path, undated_ckpt, str(images_dir), output_dir=str(undated_dir), tile=False)
    assert "error" not in r_undated, r_undated

    # Damage the stale and undated buckets' stamps to name a digest no entry names, the pre-rail
    # state a bucket already on disk can be in.
    import tcip_store as ts
    from tcip_mcp.pipelines.resolution import sidecar_key

    for bucket, digest in ((stale_dir, "a" * 64), (undated_dir, "b" * 64)):
        key = sidecar_key(bucket, "operating_point")
        stamp = ts.read_versioned(key)
        ts.replace(key, {**stamp.value, "checkpoint_sha256": digest}, expect=stamp.version)
    r_stale["checkpoint_sha256"], r_undated["checkpoint_sha256"] = "a" * 64, "b" * 64

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

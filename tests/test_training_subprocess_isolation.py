"""K24 — concurrent-execution isolation: subprocess isolation for launch_training + resource
visibility/caps. Each test pins the specific gap a design-review round found (docs/decisions/
k24-design.md) rather than re-testing the whole subprocess path end to end (that's
test_audit_cv_fixes.py::test_cv2_launch_training_persists_effective_tile_geometry)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ── K25/K13.5-2c: persist the training run's class id_map ──────────────────────────


def _write_classes_json(dataset_root, subject="catkin", attribute=None, values=None):
    # classes.json lives at the DATASET ROOT, the parent of the canonical labels/images/annotations
    # segment (dataset_layout.py's _DATASET_SEGMENTS) — not inside the labels dir itself.
    from pathlib import Path

    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry

    attrs = ()
    if attribute:
        attrs = (Attribute(name=attribute, type="categorical", values=tuple(values)),)
    write_registry(Path(dataset_root) / "classes.json",
                   ClassRegistry((Subject(subject, attributes=attrs),)))


def test_resolve_run_id_map_works_with_no_dataset_object_at_all(tmp_path):
    """Stage-6 review round 1 (MUST-FIX 1): the OLD version of this hook read
    ``train_ds.id_map``/``.subject``, which is silently absent for the COCO-assembled ``auto_val``
    default AND for every ``TiledDetectionDataset`` build — the shipped Phase-1 catkin path. The
    fix resolves independently of any dataset object; this test proves that directly, passing no
    dataset at all (only the ``data_cfg`` a real run always has), for BOTH the plain-subject and
    the attribute-scoped case."""
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    proj1 = tmp_path / "proj1"
    (proj1 / "labels").mkdir(parents=True)
    _write_classes_json(proj1, subject="catkin")
    data_cfg = {"images_dir": str(proj1 / "images"), "labels_dir": str(proj1 / "labels"),
               "subject": "catkin"}
    result = _resolve_run_id_map("detection", data_cfg)
    assert result == ("catkin", None, {"catkin": 0})

    proj2 = tmp_path / "proj2"
    (proj2 / "labels").mkdir(parents=True)
    _write_classes_json(proj2, subject="catkin", attribute="elongation",
                        values=["dormant", "elongated"])
    data_cfg2 = {"images_dir": str(proj2 / "images"), "labels_dir": str(proj2 / "labels"),
                "subject": "catkin", "attribute": "elongation"}
    result2 = _resolve_run_id_map("detection", data_cfg2)
    assert result2 == ("catkin", "elongation", {"dormant": 0, "elongated": 1})


def test_resolve_run_id_map_none_for_non_detection_task_or_no_subject(tmp_path):
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    assert _resolve_run_id_map("classification", {"subject": "catkin", "labels_dir": "x"}) is None
    assert _resolve_run_id_map("detection", {"labels_dir": "x"}) is None  # no subject


def test_resolve_run_id_map_none_for_coco_or_bespoke_source(tmp_path):
    """Stage-6 review round 2, N1: a run trained from a pre-built COCO file or a bespoke
    dataset_source doesn't necessarily get its targets from (labels_dir, subject, attribute) at
    all — a COCO file's own category ids can be authored in any order, and a bespoke builder owns
    its class space entirely. Re-deriving via the registry anyway could stamp a map that is the
    WRONG id space for what the run actually trained on and record it as an authoritative fact —
    worse than recording nothing. Must return None for both, even with a real, resolvable registry
    present (build_dataset itself never reaches the registry resolution on this same predicate,
    datasets.py's has_coco/dataset_source branch)."""
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    proj = tmp_path / "proj"
    (proj / "labels").mkdir(parents=True)
    _write_classes_json(proj, subject="catkin")

    coco_cfg = {"images_dir": str(proj / "images"), "labels_dir": str(proj / "labels"),
               "subject": "catkin", "coco_json": str(proj / "coco.json")}
    assert _resolve_run_id_map("detection", coco_cfg) is None

    coco_fmt_cfg = {"images_dir": str(proj / "images"), "labels_dir": str(proj / "labels"),
                    "subject": "catkin", "label_format": "coco"}
    assert _resolve_run_id_map("detection", coco_fmt_cfg) is None

    bespoke_cfg = {"images_dir": str(proj / "images"), "labels_dir": str(proj / "labels"),
                   "subject": "catkin", "dataset_source": "tests.bespoke_models:build_dataset"}
    assert _resolve_run_id_map("detection", bespoke_cfg) is None


def test_resolve_run_id_map_none_for_attribute_scope_with_no_registry(tmp_path):
    """The one legitimate degraded case _resolve_registry_id_map itself names — must not raise."""
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()  # no classes.json
    data_cfg = {"labels_dir": str(labels_dir), "subject": "catkin", "attribute": "elongation"}
    assert _resolve_run_id_map("detection", data_cfg) is None


def test_patch_experiment_config_id_map_merges_into_durable_config(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.experiments import experiments_dir
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_id_map

    exp_dir = experiments_dir() / "exp1"
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.json").write_text(
        json.dumps({"model_source": {"builder": "x:y"}, "data": {"images_dir": "img"}}),
        encoding="utf-8")

    _patch_experiment_config_id_map("exp1", "catkin", "elongation", {"dormant": 0, "elongated": 1})

    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["data"]["id_map"] == {"dormant": 0, "elongated": 1}
    assert cfg["data"]["subject"] == "catkin"
    assert cfg["data"]["attribute"] == "elongation"
    assert cfg["data"]["images_dir"] == "img"  # a merge, not a rewrite
    assert cfg["model_source"] == {"builder": "x:y"}  # untouched sibling key


def test_patch_experiment_config_id_map_never_sinks_a_run_with_no_experiment_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_id_map

    # No experiments/<id>/config.json exists at all — best-effort, must not raise.
    _patch_experiment_config_id_map("no_such_exp", "catkin", None, {"catkin": 0})


# ── attach_run (finding 2) ──────────────────────────────────────────────────────────


def test_attach_run_preserves_given_run_id():
    from tcip_mcp.pipelines.training.generic_trainer import attach_run, create_run, get_run

    run = attach_run("run_fixed_id", {"model_source": {"builder": "x:y"}}, "out")
    assert run.run_id == "run_fixed_id"
    assert get_run("run_fixed_id") is run

    # Distinct from create_run, which always mints its own id regardless of what's in config.
    minted = create_run({"model_source": {"builder": "x:y"}}, "out2")
    assert minted.run_id != "run_fixed_id"


def test_launch_training_child_receives_resolved_experiment_id(tmp_path, monkeypatch):
    """Design-mandated test for finding 1 (critical): the child must receive experiment_id as an
    explicit --experiment-id CLI arg resolved by the parent — including the K12 fresh-id conflict
    branch — never inferred from launch_config.json (which is written before that resolution is
    known in the fresh-id branch). Mocks subprocess.Popen to capture argv without spawning a real
    child; preflight_config's smoke check still runs for real in this process, so a real (tiny)
    bespoke model/dataset is needed."""
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    import subprocess

    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools import training_tools

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

    captured_argv: list[list[str]] = []

    class _FakeProc:
        pid = 424242

    def _fake_popen(argv, **kwargs):
        captured_argv.append(argv)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / "img0.png")
    json_io.write_annotations(str(labels_dir / "img0.json"),
                              [Annotation(subject="catkin", geometry=BBox(1, 1, 9, 9))], 32, 32)

    def _cfg(experiment_id: str) -> dict:
        return {
            "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                             "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                             "task": "detection"},
            "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "catkin"},
            "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                         "mixed_precision": False, "device": "cpu"},
            "experiment_id": experiment_id,
        }

    def _argv_experiment_id(argv: list[str]) -> str:
        return argv[argv.index("--experiment-id") + 1]

    # Fresh-creation branch: --experiment-id is the plain resolved id.
    res1 = training_tools.launch_training(_cfg("exp_fresh"), str(tmp_path / "out1"))
    assert _argv_experiment_id(captured_argv[-1]) == "exp_fresh" == res1["experiment_id"]

    # K12 fresh-id conflict branch: pre-populate "exp_reused" with real recorded history so
    # _ensure_experiment mints "exp_reused_<run_id>" instead of reusing it.
    create_experiment("exp_reused", {"a": 1})
    update_status("exp_reused", "running")
    log_metrics("exp_reused", 1, {"loss": 0.1})

    res2 = training_tools.launch_training(_cfg("exp_reused"), str(tmp_path / "out2"))
    expected_fresh_id = f"exp_reused_{res2['run_id']}"
    assert res2["experiment_id"] == expected_fresh_id
    argv2_experiment_id = _argv_experiment_id(captured_argv[-1])
    assert argv2_experiment_id == expected_fresh_id
    # Not what launch_config.json alone would carry (the caller's original, pre-resolution id) —
    # confirms the CLI arg is the parent's resolved value, never read back from that file.
    assert argv2_experiment_id != "exp_reused"


# ── should_cancel() / the sentinel file (findings 3, 10) ────────────────────────────


def test_cancel_sentinel_written_and_polled(tmp_path):
    from tcip_mcp.pipelines.training.generic_trainer import attach_run, cancel_run

    run = attach_run("run_sentinel", {"model_source": {"builder": "x:y"}}, str(tmp_path))
    run.pid = 12345  # subprocess-delegated
    assert run.should_cancel() is False

    assert cancel_run("run_sentinel") is True
    assert (tmp_path / ".cancel_requested").is_file()
    assert run.should_cancel() is True


def test_ctx_should_cancel_and_dispatch_classification_honor_sentinel(tmp_path):
    """The gap round 1 found: envelope.py's own two cancel_event.is_set() reads bypassed the
    sentinel-aware should_cancel(). A bespoke train(ctx) that calls ctx.should_cancel() (the taught
    pattern) must see a sentinel-only cancellation, and dispatch_train_body must classify the
    resulting run as 'cancelled', not 'completed'."""
    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.generic_trainer import attach_run

    run = attach_run("run_ctx_cancel", {"training_source": "tests.test_training_subprocess_isolation:_bespoke_loop"},
                     str(tmp_path))
    (tmp_path / ".cancel_requested").touch()  # no cancel_event set anywhere — sentinel only

    ctx = TrainContext(run=run, train_loader=None, experiment_id=None)
    assert ctx.should_cancel() is True

    dispatch_train_body(ctx)
    assert run.status == "cancelled"


def _bespoke_loop(ctx) -> None:
    """Referenced by dotted path in the test above — a minimal train(ctx) that only checks
    ctx.should_cancel(), the exact taught pattern this test is pinning."""
    if ctx.should_cancel():
        return
    raise AssertionError("should_cancel() did not see the sentinel-only cancellation")


# ── cancel_run's disk fallback (findings 6, 9, 11) ──────────────────────────────────


def test_cancel_run_falls_back_to_disk_when_not_in_local_registry(tmp_path, monkeypatch):
    """A run this process never held in _RUNS (launched by a different process) can still be
    cancelled, provided its experiment identity was stamped (K24) — the failure mode round 2 found
    was a silent no-op write to a guessed path nobody polls; this confirms the real path instead."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity
    from tcip_mcp.pipelines.training.generic_trainer import cancel_run

    real_output_dir = tmp_path / "real_run_dir"
    real_output_dir.mkdir()
    create_experiment("exp_cross_proc", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_cross_proc", "run_cross_proc", str(real_output_dir))

    assert cancel_run("run_cross_proc") is True
    assert (real_output_dir / ".cancel_requested").is_file()


def test_cancel_run_unknown_run_refuses_honestly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.generic_trainer import cancel_run

    assert cancel_run("no-such-run-anywhere") is False


def test_ensure_experiment_pristine_reuse_stamps_identity(tmp_path, monkeypatch):
    """Finding 11: the pristine pre-created-experiment reuse branch (a real, tested workflow —
    test_ensure_experiment_attaches_to_precreated) must stamp run_id/output_dir too, not only the
    fresh-creation branch — otherwise a run launched against a pre-named experiment is permanently
    unresolvable by resolve_experiment_dir_for_run from a different process."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, resolve_experiment_dir_for_run
    from tcip_mcp.tools.training_tools import _ensure_experiment

    create_experiment("precreated", {"a": 1})  # agent pre-creates before any run_id exists
    eid = _ensure_experiment("precreated", {"a": 1}, None, resume_from="", run_id="run_later",
                             output_dir=str(tmp_path / "out"))
    assert eid == "precreated"

    resolved = resolve_experiment_dir_for_run("run_later")
    assert resolved is not None
    assert resolved.name == "precreated"


# ── resolve_experiment_dir_for_run / reconstruct_run_status (finding 9) ─────────────


def test_resolve_experiment_dir_for_run_handles_fresh_id_suffix(tmp_path, monkeypatch):
    """The K12 fresh-id format (f'{experiment_id}_{run_id}') means experiment_id != run_id — the
    resolver must not assume they're equal (the bug the original fix-6 shape had)."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, resolve_experiment_dir_for_run, stamp_run_identity

    create_experiment("exp1_run_9_0", {"model_source": {"builder": "x:y"}},
                      parent_experiment="exp1")
    stamp_run_identity("exp1_run_9_0", "run_9_0", "out")

    resolved = resolve_experiment_dir_for_run("run_9_0")
    assert resolved is not None and resolved.name == "exp1_run_9_0"


def test_resolve_experiment_dir_for_run_refuses_unresolvable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import resolve_experiment_dir_for_run

    assert resolve_experiment_dir_for_run("nope") is None


def test_reconstruct_run_status_from_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        create_experiment, log_metrics, reconstruct_run_status, stamp_run_identity, update_status,
    )

    create_experiment("exp_disk", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_disk", "run_disk", "out_dir")
    update_status("exp_disk", "running")
    log_metrics("exp_disk", 3, {"loss": 0.1})

    result = reconstruct_run_status("run_disk")
    assert result is not None
    assert result["status"] == "running"
    assert result["current_epoch"] == 3
    assert result["output_dir"] == "out_dir"
    assert result["best_metric"] is None  # not fabricated from the metrics log


def test_reconstruct_run_status_surfaces_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, reconstruct_run_status, stamp_run_identity, update_status

    create_experiment("exp_err", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_err", "run_err", "out_dir")
    update_status("exp_err", "failed", error="exceeded max_wall_clock_seconds (10)")

    result = reconstruct_run_status("run_err")
    assert result["status"] == "failed"
    assert result["error"] == "exceeded max_wall_clock_seconds (10)"


def test_reconstruct_run_status_reports_cancelled_not_running_or_interrupted(tmp_path, monkeypatch):
    """Stage-6 review finding: reconstruct_run_status originally re-derived state from heartbeat
    freshness for anything not in experiments.py's _TERMINAL_STATES ({"completed", "failed"} —
    deliberately excludes "cancelled" so a cancelled run's record stays reopenable/resumable). But
    that set is for a DIFFERENT purpose (the update_status mutation-lock); reusing it here meant a
    gracefully cancelled run (real state, fresh heartbeat from its own final update_status call)
    reported as "running", then permanently "interrupted" once the heartbeat went stale — cancel_
    training's own documented "status flips to cancelled" contract was false for every real launch."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, reconstruct_run_status, stamp_run_identity, update_status

    create_experiment("exp_cancelled", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_cancelled", "run_cancelled", "out_dir")
    update_status("exp_cancelled", "running")
    update_status("exp_cancelled", "cancelled")  # stamps a fresh heartbeat, same as any update_status call

    result = reconstruct_run_status("run_cancelled")
    assert result["status"] == "cancelled"

    # And it must not flip to "interrupted" once the heartbeat goes stale — a cancelled run is
    # already a known, final outcome, not a liveness question.
    result_stale = reconstruct_run_status("run_cancelled", stale_seconds=-1)  # heartbeat always "stale"
    assert result_stale["status"] == "cancelled"


def test_update_status_error_is_keyword_only_and_backward_compatible(tmp_path, monkeypatch):
    """Finding 8: update_status's original signature had no error param at all — the wall-clock
    watcher's call would have raised TypeError on first real use. Every existing 2-positional-arg
    call site must still work unchanged."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, get_experiment, update_status

    create_experiment("exp_compat", {"a": 1})
    assert update_status("exp_compat", "running") == {"experiment_id": "exp_compat", "state": "running"}
    update_status("exp_compat", "failed", error="boom")
    status = get_experiment("exp_compat")["status"]
    assert status["error"] == "boom"


# ── check_training_status / list_training_runs disk fallback ───────────────────────


def test_check_training_status_falls_back_to_disk_for_delegated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, log_metrics, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.generic_trainer import attach_run
    from tcip_mcp.tools.training_tools import check_training_status

    run = attach_run("run_delegated", {"model_source": {"builder": "x:y"}}, "out_dir")
    run.pid = 999  # subprocess-delegated — in-memory fields below are now stale by design

    create_experiment("run_delegated", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("run_delegated", "run_delegated", "out_dir")
    update_status("run_delegated", "running")
    log_metrics("run_delegated", 7, {"loss": 0.2})

    result = check_training_status("run_delegated")
    assert result["epoch"] == 7  # not the stale in-memory 0
    assert result["status"] == "running"


def test_list_training_runs_leaves_in_process_runs_untouched(tmp_path, monkeypatch):
    """A run with no pid (every existing synchronous test) is reported from the live in-memory
    record exactly as before — the disk overlay must never touch it."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.generic_trainer import create_run
    from tcip_mcp.tools.training_tools import list_training_runs

    run = create_run({"model_source": {"builder": "x:y"}}, "out_dir")
    run.status = "running"
    run.current_epoch = 5

    runs = list_training_runs()["runs"]
    entry = next(r for r in runs if r["run_id"] == run.run_id)
    assert entry["status"] == "running"
    assert entry["current_epoch"] == 5


# ── GPU device pinning (finding 4) ───────────────────────────────────────────────────


def test_gpu_device_pinning_round_robins(monkeypatch):
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    seen = set()
    for _ in range(4):
        env = training_tools._child_env_for_launch({})
        seen.add(env["CUDA_VISIBLE_DEVICES"])
    assert seen == {"0", "1"}


def test_gpu_pinning_skipped_when_device_explicit(monkeypatch):
    """Finding 4: CUDA_VISIBLE_DEVICES remaps indices inside the child — pinning it when the
    config already names an explicit device would ask for an ordinal invalid in the child's own
    remapped view. Must be a no-op whenever the config already knows which device it wants."""
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    env = training_tools._child_env_for_launch({"device": "cuda:1"})
    assert "CUDA_VISIBLE_DEVICES" not in env

    env = training_tools._child_env_for_launch({"training": {"device": "cuda:0"}})
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_gpu_pinning_noop_with_single_gpu(monkeypatch):
    from tcip_mcp.tools import training_tools

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    env = training_tools._child_env_for_launch({})
    assert "CUDA_VISIBLE_DEVICES" not in env


# ── wall-clock timeout (findings 7, 8) ──────────────────────────────────────────────


def test_max_wall_clock_seconds_terminates_hung_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import subprocess
    import sys
    import time

    from tcip_mcp.experiments import create_experiment, experiments_dir, update_status
    from tcip_mcp.pipelines.training.generic_trainer import attach_run
    from tcip_mcp.tools.training_tools import _watch_wall_clock
    from tcip_mcp.utils.atomic_io import read_json

    create_experiment("exp_timeout", {"model_source": {"builder": "x:y"}})
    update_status("exp_timeout", "running")
    run = attach_run("run_timeout", {"model_source": {"builder": "x:y"}}, str(tmp_path))

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _watch_wall_clock(proc, run, "exp_timeout", 0.2)
        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, "watcher did not terminate the hung process"
        assert run.status == "failed"

        # The failure reason must land through the real status channel (finding 8), not only the
        # in-memory mark check_training_status ignores for a pid-bearing run. Polls the status.json
        # file directly via read_json (already OSError-tolerant, unlike get_experiment's raw read)
        # since a Windows AV/indexer can transiently hold the file mid-write.
        status_path = experiments_dir() / "exp_timeout" / "status.json"
        status: dict = {}
        for _ in range(50):
            status = read_json(status_path, default={})
            if status.get("error"):
                break
            time.sleep(0.1)
        assert "max_wall_clock_seconds" in status.get("error", "")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ── inspect_compute_resources ────────────────────────────────────────────────────────


def test_inspect_compute_resources_degrades_without_psutil(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated absent psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)

    from tcip_mcp.tools.training_tools import inspect_compute_resources
    result = inspect_compute_resources()
    assert result["cpu"]["percent_used"] is None
    assert result["memory"]["total_bytes"] is None
    assert result["memory"]["available_bytes"] is None
    assert isinstance(result["gpus"], list)  # still populates via torch alone
    assert "active_training_runs" in result


def test_inspect_compute_resources_reports_gpu_free_memory(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(idx):
            return (1_000, 4_000)

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    from tcip_mcp.tools.training_tools import inspect_compute_resources
    result = inspect_compute_resources()
    assert result["gpus"] == [{"index": 0, "free_bytes": 1_000, "total_bytes": 4_000}]


def test_inspect_compute_resources_counts_subprocess_delegated_running_runs(tmp_path, monkeypatch):
    """Stage-6 review finding (found independently by both lenses): active_training_runs called
    the raw in-memory generic_trainer.list_runs() instead of the disk-aware list_training_runs()
    two functions above it in the same file — so it always reported 0 for a subprocess-delegated
    run, since the parent's own TrainRun.status never leaves its create_run-time "created" default
    once the child starts mutating its own separate copy. This is the common case post-K24, not an
    edge case, and it's the exact number the tool exists to give the agent before it decides
    whether to launch another concurrent run."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.generic_trainer import attach_run
    from tcip_mcp.tools.training_tools import inspect_compute_resources

    # _RUNS is a process-global registry other tests in this session also populate — compare a
    # delta, not an absolute count, so this test doesn't depend on being run in isolation.
    baseline = inspect_compute_resources()["active_training_runs"]

    run = attach_run("run_active", {"model_source": {"builder": "x:y"}}, str(tmp_path))
    run.pid = 555  # subprocess-delegated — mirrors what launch_training does after Popen
    assert run.status == "created"  # the parent-side placeholder never advances

    create_experiment("run_active", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("run_active", "run_active", str(tmp_path))
    update_status("run_active", "running")  # only the child's disk write reflects reality

    assert inspect_compute_resources()["active_training_runs"] == baseline + 1


# ── Part A: HPO resource caps ─────────────────────────────────────────────────────────


def test_default_trial_resources_derives_fractional_gpu_share(monkeypatch):
    from tcip_mcp.pipelines.training import hpo

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

    monkeypatch.setattr(torch, "cuda", _FakeCuda)

    assert hpo._default_trial_resources(max_concurrent=1) == {"cpu": 1.0, "gpu": 1.0}
    assert hpo._default_trial_resources(max_concurrent=4) == {"cpu": 1.0, "gpu": 0.5}


def test_default_trial_resources_no_gpu(monkeypatch):
    from tcip_mcp.pipelines.training import hpo

    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    monkeypatch.setattr(torch, "cuda", _FakeCuda)
    assert hpo._default_trial_resources(max_concurrent=1) == {"cpu": 1.0, "gpu": 0.0}


def test_tune_search_accepts_explicit_resources_per_trial(tmp_path):
    """A real, lightweight Ray sweep (matching test_imbalance_aug_hpo.py's own established
    pattern for this function — a pure-math objective, no training) still finds the minimum when
    an explicit resources_per_trial is supplied, proving it's accepted end to end rather than only
    unit-testing the derivation helper in isolation."""
    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import tune_search

    def obj(config, report):
        report((config["x"] - 2.0) ** 2)

    result = tune_search(
        obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        metric="objective", mode="min", num_samples=4,
        search_alg="random", scheduler="none",
        resources_per_trial={"cpu": 1.0, "gpu": 0.0},
    )
    assert result["n_trials"] == 4
    assert result["best_value"] is not None

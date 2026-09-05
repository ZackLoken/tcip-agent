"""Subprocess isolation tests for launch_training and resource visibility/caps. Each test pins a
specific concurrency/isolation gap rather than re-testing the whole subprocess path end to end
(that's test_detection_measurement_integrity.py::test_launch_training_persists_effective_tile_geometry)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ── persist the training run's class id_map ──────────────────────────


def _write_classes_json(dataset_root, subject="bud", attribute=None, values=None):
    # classes.json lives at the dataset root, the parent of the canonical labels/images/annotations
    # segment (dataset_layout.py's _DATASET_SEGMENTS), not inside the labels dir itself.
    from pathlib import Path

    from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject, write_registry

    attrs = ()
    if attribute:
        attrs = (Attribute(name=attribute, type="categorical", values=tuple(values)),)
    write_registry(Path(dataset_root) / "classes.json",
                   ClassRegistry((Subject(subject, attributes=attrs),)))


def test_resolve_run_id_map_works_with_no_dataset_object_at_all(tmp_path):
    """id_map resolution must not depend on ``train_ds.id_map``/``.subject``, which is silently
    absent for the COCO-assembled ``auto_val`` default and for every ``TiledDetectionDataset``
    build (the shipped Phase-1 bud path). This test passes no dataset object at all (only the
    ``data_cfg`` a real run always has), for both the plain-subject and the attribute-scoped
    case."""
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    proj1 = tmp_path / "proj1"
    (proj1 / "labels").mkdir(parents=True)
    _write_classes_json(proj1, subject="bud")
    data_cfg = {"images_dir": str(proj1 / "images"), "labels_dir": str(proj1 / "labels"),
               "subject": "bud"}
    result = _resolve_run_id_map("detection", data_cfg)
    assert result == ("bud", None, {"bud": 0})

    proj2 = tmp_path / "proj2"
    (proj2 / "labels").mkdir(parents=True)
    _write_classes_json(proj2, subject="bud", attribute="opening",
                        values=["closed", "open"])
    data_cfg2 = {"images_dir": str(proj2 / "images"), "labels_dir": str(proj2 / "labels"),
                "subject": "bud", "attribute": "opening"}
    result2 = _resolve_run_id_map("detection", data_cfg2)
    assert result2 == ("bud", "opening", {"closed": 0, "open": 1})


def test_resolve_run_id_map_none_for_non_detection_task_or_no_subject(tmp_path):
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    assert _resolve_run_id_map("classification", {"subject": "bud", "labels_dir": "x"}) is None
    assert _resolve_run_id_map("detection", {"labels_dir": "x"}) is None  # no subject


def test_resolve_run_id_map_none_for_coco_or_bespoke_source(tmp_path):
    """A run trained from a pre-built COCO file or a bespoke dataset_source doesn't necessarily
    get its targets from (labels_dir, subject, attribute) at all: a COCO file's own category ids
    can be authored in any order, and a bespoke builder owns its class space entirely. Re-deriving
    via the registry anyway could stamp a map that is the wrong id space for what the run actually
    trained on and record it as an authoritative fact, worse than recording nothing. Must return
    None for both, even with a real, resolvable registry present (build_dataset itself never
    reaches the registry resolution on this same predicate, datasets.py's has_coco/dataset_source
    branch)."""
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    proj = tmp_path / "proj"
    (proj / "labels").mkdir(parents=True)
    _write_classes_json(proj, subject="bud")

    coco_cfg = {"images_dir": str(proj / "images"), "labels_dir": str(proj / "labels"),
               "subject": "bud", "coco_json": str(proj / "coco.json")}
    assert _resolve_run_id_map("detection", coco_cfg) is None

    coco_fmt_cfg = {"images_dir": str(proj / "images"), "labels_dir": str(proj / "labels"),
                    "subject": "bud", "label_format": "coco"}
    assert _resolve_run_id_map("detection", coco_fmt_cfg) is None

    bespoke_cfg = {"images_dir": str(proj / "images"), "labels_dir": str(proj / "labels"),
                   "subject": "bud", "dataset_source": "tests.bespoke_models:build_dataset"}
    assert _resolve_run_id_map("detection", bespoke_cfg) is None


def test_resolve_run_id_map_none_for_attribute_scope_with_no_registry(tmp_path):
    """The one legitimate degraded case resolve_registry_id_map itself names: must not raise."""
    from tcip_mcp.pipelines.training.subprocess_worker import _resolve_run_id_map

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()  # no classes.json
    data_cfg = {"labels_dir": str(labels_dir), "subject": "bud", "attribute": "opening"}
    assert _resolve_run_id_map("detection", data_cfg) is None


def test_patch_experiment_config_id_map_merges_into_durable_config(tmp_path, monkeypatch):
    import tcip_store as ts

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_id_map

    create_experiment("exp1", {"model_source": {"builder": "x:y"}, "data": {"images_dir": "img"}})

    _patch_experiment_config_id_map("exp1", "bud", "opening", {"closed": 0, "open": 1})

    cfg = ts.read(config_key("exp1"))
    assert cfg["data"]["id_map"] == {"closed": 0, "open": 1}
    assert cfg["data"]["subject"] == "bud"
    assert cfg["data"]["attribute"] == "opening"
    assert cfg["data"]["images_dir"] == "img"  # a merge, not a rewrite
    assert cfg["model_source"] == {"builder": "x:y"}  # untouched sibling key


def test_patch_experiment_config_id_map_never_sinks_a_run_with_no_experiment_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.pipelines.training.subprocess_worker import _patch_experiment_config_id_map

    # No experiments/<id>/config.json exists at all: best-effort, must not raise.
    _patch_experiment_config_id_map("no_such_exp", "bud", None, {"bud": 0})


def test_is_manifest_bound_split_only_true_for_a_manifest_binding():
    """A spatial or auto-split run's own resolved ``data.split`` block (member identities that
    stay out of the durable config) must not qualify, only a manifest-bound run's block does."""
    from tcip_mcp.pipelines.training.subprocess_worker import _is_manifest_bound_split

    assert _is_manifest_bound_split({"manifest_binding": {"date": "2-11-26"}}) is True
    assert _is_manifest_bound_split(
        {"resolved_group_by": "spatial_strip", "spatial_manifest": {}}) is False
    assert _is_manifest_bound_split({"resolved_group_by": "tile_prefix"}) is False
    assert _is_manifest_bound_split({}) is False
    assert _is_manifest_bound_split(None) is False


def test_worker_leaves_a_spatial_runs_identities_out_of_the_durable_config(tmp_path, monkeypatch):
    """Only a manifest-bound run's resolved split block is mirrored into the durable experiment
    record; a spatial run's own per-region member identities stay with its checkpoints."""
    import tcip_store as ts

    import tcip_mcp.tools.training_tools as ttools
    from tcip_mcp.experiments import config_key, create_experiment
    from tcip_mcp.pipelines.data import split_construction as sc
    from tcip_mcp.pipelines.training import subprocess_worker as worker

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    out = tmp_path / "run"
    out.mkdir()
    config = {"model_source": {"builder": "x:y", "task": "detection"},
              "data": {"images_dir": "img", "split": {"group_by": "spatial_strip"}}}
    create_experiment("exp1", config)
    ts.replace(ttools.launch_config_key(out), config)

    def stub_auto_train_val(task, data_cfg, transforms):
        data_cfg["split"].update({
            "resolved_group_by": "spatial_strip",
            "spatial_manifest": {"regions": {"r0": ["a"], "r1": ["b"]}},
        })
        return None, None, None

    class StopAfterSplit(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopAfterSplit

    monkeypatch.setattr(sc, "auto_train_val", stub_auto_train_val)
    monkeypatch.setattr(worker, "_resolve_run_id_map", stop)
    with pytest.raises(StopAfterSplit):
        worker.run("run1", "exp1", str(out), "")

    assert "spatial_manifest" not in ts.read(config_key("exp1"))["data"].get("split", {})


# ── attach_run ──────────────────────────────────────────────────────────


def test_attach_run_preserves_given_run_id():
    from tcip_mcp.pipelines.training.run_registry import attach_run, create_run, get_run

    run = attach_run("run_fixed_id", {"model_source": {"builder": "x:y"}}, "out")
    assert run.run_id == "run_fixed_id"
    assert get_run("run_fixed_id") is run

    # Distinct from create_run, which always mints its own id regardless of what's in config.
    minted = create_run({"model_source": {"builder": "x:y"}}, "out2")
    assert minted.run_id != "run_fixed_id"


def test_launch_training_child_receives_resolved_experiment_id(tmp_path, monkeypatch):
    """The child must receive experiment_id as an explicit --experiment-id CLI arg resolved by the
    parent (including the fresh-id conflict branch), never inferred from launch_config.json
    (which is written before that resolution is known in the fresh-id branch). Mocks
    subprocess.Popen to capture argv without spawning a real child; preflight_config's smoke check
    still runs for real in this process, so a real (tiny) bespoke model/dataset is needed."""
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
                              [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)

    def _cfg(experiment_id: str) -> dict:
        return {
            "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                             "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                             "task": "detection"},
            "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud"},
            "training": {"batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                         "mixed_precision": False, "device": "cpu"},
            "experiment_id": experiment_id,
        }

    def _argv_experiment_id(argv: list[str]) -> str:
        return argv[argv.index("--experiment-id") + 1]

    # Fresh-creation branch: --experiment-id is the plain resolved id.
    res1 = training_tools.launch_training(_cfg("exp_fresh"), str(tmp_path / "out1"))
    assert _argv_experiment_id(captured_argv[-1]) == "exp_fresh" == res1["experiment_id"]

    # fresh-id conflict branch: pre-populate "exp_reused" with real recorded history so
    # _ensure_experiment mints "exp_reused_<run_id>" instead of reusing it.
    create_experiment("exp_reused", {"a": 1})
    update_status("exp_reused", "running")
    log_metrics("exp_reused", 1, {"loss": 0.1})

    res2 = training_tools.launch_training(_cfg("exp_reused"), str(tmp_path / "out2"))
    expected_fresh_id = f"exp_reused_{res2['run_id']}"
    assert res2["experiment_id"] == expected_fresh_id
    argv2_experiment_id = _argv_experiment_id(captured_argv[-1])
    assert argv2_experiment_id == expected_fresh_id
    # Not what launch_config.json alone would carry (the caller's original, pre-resolution id).
    # Confirms the CLI arg is the parent's resolved value, never read back from that file.
    assert argv2_experiment_id != "exp_reused"


# ── should_cancel() / the sentinel file ────────────────────────────


def test_cancel_sentinel_written_and_polled(tmp_path):
    from tcip_mcp.pipelines.training.run_registry import attach_run, cancel_run

    run = attach_run("run_sentinel", {"model_source": {"builder": "x:y"}}, str(tmp_path))
    run.pid = 12345  # subprocess-delegated
    assert run.should_cancel() is False

    assert cancel_run("run_sentinel") is True
    assert (tmp_path / ".cancel_requested").is_file()
    assert run.should_cancel() is True


def test_ctx_should_cancel_and_dispatch_classification_honor_sentinel(tmp_path):
    """envelope.py's own two cancel_event.is_set() reads bypass the sentinel-aware
    should_cancel(). A bespoke train(ctx) that calls ctx.should_cancel() (the taught pattern) must
    see a sentinel-only cancellation, and dispatch_train_body must classify the resulting run as
    'cancelled', not 'completed'."""
    from tcip_mcp.pipelines.training.envelope import TrainContext, dispatch_train_body
    from tcip_mcp.pipelines.training.run_registry import attach_run

    run = attach_run("run_ctx_cancel", {"training_source": "tests.test_training_subprocess_isolation:_bespoke_loop"},
                     str(tmp_path))
    (tmp_path / ".cancel_requested").touch()  # no cancel_event set anywhere, sentinel only

    ctx = TrainContext(run=run, train_loader=None, experiment_id=None)
    assert ctx.should_cancel() is True

    dispatch_train_body(ctx)
    assert run.status == "cancelled"


def _bespoke_loop(ctx) -> None:
    """Referenced by dotted path in the test above: a minimal train(ctx) that only checks
    ctx.should_cancel(), the exact taught pattern this test is pinning."""
    if ctx.should_cancel():
        return
    raise AssertionError("should_cancel() did not see the sentinel-only cancellation")


# ── cancel_run's disk fallback ──────────────────────────────────


def test_cancel_run_falls_back_to_disk_when_not_in_local_registry(tmp_path, monkeypatch):
    """A run this process never held in _RUNS (launched by a different process) can still be
    cancelled, provided its experiment identity was stamped: otherwise cancellation silently
    writes to a guessed path nobody polls; this confirms the real path instead."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity
    from tcip_mcp.pipelines.training.run_registry import cancel_run

    real_output_dir = tmp_path / "real_run_dir"
    real_output_dir.mkdir()
    create_experiment("exp_cross_proc", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_cross_proc", "run_cross_proc", str(real_output_dir))

    assert cancel_run("run_cross_proc") is True
    assert (real_output_dir / ".cancel_requested").is_file()


def test_cancel_run_unknown_run_refuses_honestly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.run_registry import cancel_run

    assert cancel_run("no-such-run-anywhere") is False


def test_ensure_experiment_pristine_reuse_stamps_identity(tmp_path, monkeypatch):
    """The pristine pre-created-experiment reuse branch (a real, tested workflow,
    test_ensure_experiment_attaches_to_precreated) must stamp run_id/output_dir too, not only the
    fresh-creation branch. Otherwise a run launched against a pre-named experiment is permanently
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


# ── resolve_experiment_dir_for_run / reconstruct_run_status ─────────────


def test_resolve_experiment_dir_for_run_handles_fresh_id_suffix(tmp_path, monkeypatch):
    """The fresh-id format (f'{experiment_id}_{run_id}') means experiment_id != run_id: the
    resolver must not assume they're equal."""
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
    """A row with no stamped ``selection_metric`` (a bespoke loop that never called through
    ``generic_trainer.train()``) carries no best: the best is derived from the trainer's own
    stamped rows, never fabricated from a metric name this log never recorded."""
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
    assert result["best_metric"] is None
    assert result["best_metric_name"] is None


def test_reconstruct_run_status_derives_best_from_stamped_rows(tmp_path, monkeypatch):
    """A row shaped the way ``generic_trainer.train()`` actually writes it (``selection`` +
    ``selection_metric``) does carry a name and a best, read back rather than re-derived from
    config."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        create_experiment, log_metrics, reconstruct_run_status, stamp_run_identity, update_status,
    )

    create_experiment("exp_stamped", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_stamped", "run_stamped", "out_dir")
    update_status("exp_stamped", "running")
    log_metrics("exp_stamped", 1, {"selection": 0.5, "selection_metric": "map50"})
    log_metrics("exp_stamped", 2, {"selection": 0.7, "selection_metric": "map50"})

    result = reconstruct_run_status("run_stamped")
    assert result is not None
    assert result["best_metric_name"] == "map50"
    assert result["best_metric"] == 0.7


def test_best_selection_from_log_withholds_on_no_name_or_undeclared_direction():
    from tcip_mcp.experiments import best_selection_from_log

    # No row stamps a selection metric at all.
    assert best_selection_from_log([{"epoch": 1, "loss": 0.1}]) == (None, None)

    # A stamped name with no declared ranking direction is never guessed at.
    assert best_selection_from_log(
        [{"epoch": 1, "selection": 0.1, "selection_metric": "not_a_real_metric"}],
    ) == (None, None)


def test_best_selection_from_log_ranks_in_the_metrics_declared_direction():
    from tcip_mcp.experiments import best_selection_from_log

    # loss: lower is better.
    name, best = best_selection_from_log([
        {"epoch": 1, "selection": 0.9, "selection_metric": "loss"},
        {"epoch": 2, "selection": 0.4, "selection_metric": "loss"},
        {"epoch": 3, "selection": 0.6, "selection_metric": "loss"},
    ])
    assert (name, best) == ("loss", 0.4)

    # map50: higher is better.
    name, best = best_selection_from_log([
        {"epoch": 1, "selection": 0.4, "selection_metric": "map50"},
        {"epoch": 2, "selection": 0.9, "selection_metric": "map50"},
        {"epoch": 3, "selection": 0.6, "selection_metric": "map50"},
    ])
    assert (name, best) == ("map50", 0.9)

    # A non-finite row is skipped rather than winning the comparison.
    name, best = best_selection_from_log([
        {"epoch": 1, "selection": 0.5, "selection_metric": "loss"},
        {"epoch": 2, "selection": float("nan"), "selection_metric": "loss"},
    ])
    assert (name, best) == ("loss", 0.5)


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
    """reconstruct_run_status must not re-derive a "cancelled" run's state from heartbeat
    freshness: experiments.py's _TERMINAL_STATES ({"completed", "failed"}) deliberately excludes
    "cancelled" so a cancelled run's record stays reopenable/resumable, but that set exists for a
    different purpose (the update_status mutation lock). Treating "cancelled" as non-terminal
    here would report a gracefully cancelled run as "running", then permanently "interrupted" once
    its heartbeat goes stale."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, reconstruct_run_status, stamp_run_identity, update_status

    create_experiment("exp_cancelled", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp_cancelled", "run_cancelled", "out_dir")
    update_status("exp_cancelled", "running")
    update_status("exp_cancelled", "cancelled")  # stamps a fresh heartbeat, same as any update_status call

    result = reconstruct_run_status("run_cancelled")
    assert result["status"] == "cancelled"

    # And it must not flip to "interrupted" once the heartbeat goes stale: a cancelled run is
    # already a known, final outcome, not a liveness question.
    result_stale = reconstruct_run_status("run_cancelled", stale_seconds=-1)  # heartbeat always "stale"
    assert result_stale["status"] == "cancelled"


def test_update_status_error_is_keyword_only_and_backward_compatible(tmp_path, monkeypatch):
    """update_status's error param must be keyword-only so every existing 2-positional-arg call
    site keeps working unchanged, while the wall-clock watcher can still pass error= on failure."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, get_experiment, update_status

    create_experiment("exp_compat", {"a": 1})
    assert update_status("exp_compat", "running") == {"experiment_id": "exp_compat", "state": "running"}
    update_status("exp_compat", "failed", error="boom")
    status = get_experiment("exp_compat")["status"]
    assert status["error"] == "boom"


# ── monitor_training / list_experiments(launched_only=True) disk fallback ─────


def test_monitor_training_falls_back_to_disk_for_delegated_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, log_metrics, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.run_registry import attach_run
    from tcip_mcp.tools.training_tools import monitor_training

    run = attach_run("run_delegated", {"model_source": {"builder": "x:y"}}, "out_dir")
    run.pid = 999  # subprocess-delegated, in-memory fields below are now stale by design

    create_experiment("run_delegated", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("run_delegated", "run_delegated", "out_dir")
    update_status("run_delegated", "running")
    log_metrics("run_delegated", 7, {"loss": 0.2})

    result = monitor_training("run_delegated")
    assert result["epoch"] == 7  # not the stale in-memory 0
    assert result["status"] == "running"


def test_launched_runs_view_reconstructs_without_the_full_scan_resolver(tmp_path, monkeypatch):
    """Each row is reconstructed from the record the enumeration already holds; it must never
    round-trip a custom-named (id != run_id) experiment through resolve_experiment_for_run's
    full-scan fallback the way the old per-run reconstruct_run_status call did."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    import tcip_mcp.experiments as exp_mod
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.experiment_tools import list_experiments

    create_experiment("exp-001-chestnut-burr-det", {"model_source": {"builder": "my_models:burr_det"}})
    update_status("exp-001-chestnut-burr-det", "running")

    calls = {"n": 0}
    real = exp_mod.resolve_experiment_for_run

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(exp_mod, "resolve_experiment_for_run", _counting)

    runs = list_experiments(launched_only=True)["runs"]
    assert any(r["run_id"] == "exp-001-chestnut-burr-det" for r in runs)
    assert calls["n"] == 0


def test_launched_runs_view_lists_a_launched_experiment_this_process_never_held(tmp_path, monkeypatch):
    """A run another process launched (never in this process's _RUNS at all, not merely
    pid-bearing) still lists, reconstructed straight from its own disk record."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.experiment_tools import list_experiments

    create_experiment("exp-no-stamp", {"model_source": {"builder": "my_models:chestnut_burr_det"}})
    update_status("exp-no-stamp", "running")

    by_id = {r["run_id"]: r for r in list_experiments(launched_only=True)["runs"]}
    assert "exp-no-stamp" in by_id
    assert by_id["exp-no-stamp"]["external"] is True


def test_launched_runs_view_overlays_a_pid_bearing_entry_from_disk(tmp_path, monkeypatch):
    """A pid-bearing in-memory entry (subprocess-delegated) takes the disk overlay for its
    status/current_epoch: its own in-memory copy is a stale launch-time placeholder once the
    child starts mutating its own separate copy on disk. The status/current_epoch overlay is
    pre-existing coverage; the new assertion this test adds is external=False, unlike a run this
    process never held at all."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.experiments import create_experiment, log_metrics, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.run_registry import attach_run
    from tcip_mcp.tools.experiment_tools import list_experiments

    run = attach_run("run_pid_overlay", {"model_source": {"builder": "my_models:burr_det"}}, "out_dir")
    run.pid = 4242

    create_experiment("run_pid_overlay", {"model_source": {"builder": "my_models:burr_det"}})
    stamp_run_identity("run_pid_overlay", "run_pid_overlay", "out_dir")
    update_status("run_pid_overlay", "running")
    log_metrics("run_pid_overlay", 9, {"loss": 0.1})

    by_id = {r["run_id"]: r for r in list_experiments(launched_only=True)["runs"]}
    assert by_id["run_pid_overlay"]["status"] == "running"
    assert by_id["run_pid_overlay"]["current_epoch"] == 9
    assert by_id["run_pid_overlay"]["external"] is False


def test_launched_runs_view_leaves_in_process_runs_untouched(tmp_path, monkeypatch):
    """A run with no pid (every existing synchronous test) is reported from the live in-memory
    record exactly as before. The disk overlay must never touch it."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.experiment_tools import list_experiments

    run = create_run({"model_source": {"builder": "x:y"}}, "out_dir")
    run.status = "running"
    run.current_epoch = 5

    runs = list_experiments(launched_only=True)["runs"]
    entry = next(r for r in runs if r["run_id"] == run.run_id)
    assert entry["status"] == "running"
    assert entry["current_epoch"] == 5


# ── GPU device pinning ───────────────────────────────────────────────────


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
    """CUDA_VISIBLE_DEVICES remaps indices inside the child: pinning it when the config already
    names an explicit device would ask for an ordinal invalid in the child's own remapped view.
    Must be a no-op whenever the config already knows which device it wants."""
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


# ── wall-clock timeout ──────────────────────────────────────────────


def test_max_wall_clock_seconds_terminates_hung_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import subprocess
    import sys
    import time

    from tcip_mcp.experiments import create_experiment, status_key, update_status
    from tcip_mcp.pipelines.training.run_registry import attach_run
    from tcip_mcp.tools.training_tools import _watch_wall_clock
    from tcip_store import DecodeError, read

    create_experiment("exp_timeout", {"model_source": {"builder": "x:y"}})
    update_status("exp_timeout", "running")
    run = attach_run("run_timeout", {"model_source": {"builder": "x:y"}}, str(tmp_path))

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _watch_wall_clock(proc, run, "exp_timeout", 0.2, root=tmp_path)
        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, "watcher did not terminate the hung process"
        assert run.status == "failed"

        # The failure reason must land through the real status channel, not only the
        # in-memory mark monitor_training ignores for a pid-bearing run. Polls the record
        # itself, tolerating a read that lands mid-write, since a Windows AV/indexer can
        # transiently hold the file.
        status: dict = {}
        for _ in range(50):
            try:
                status = read(status_key("exp_timeout"), default={})
            except DecodeError:
                status = {}
            if status.get("error"):
                break
            time.sleep(0.1)
        assert "max_wall_clock_seconds" in status.get("error", "")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_max_wall_clock_seconds_writes_to_the_launch_root_after_an_adopt(tmp_path, monkeypatch):
    """The watchdog's write belongs to the project the run launched under, not wherever this
    process's platform root has since moved to: a repin between the launch and the timeout
    must not drop the failure or leave the launch project's record stuck at ``running``."""
    monkeypatch.chdir(tmp_path)
    import subprocess
    import sys
    import time

    from tcip_mcp import workspace
    from tcip_mcp.experiments import create_experiment, status_key, update_status
    from tcip_mcp.pipelines.training.run_registry import attach_run
    from tcip_mcp.tools.training_tools import _watch_wall_clock
    from tcip_store import DecodeError, read

    launch_root = tmp_path
    create_experiment("exp_timeout_other_root", {"model_source": {"builder": "x:y"}})
    update_status("exp_timeout_other_root", "running")
    run = attach_run("run_timeout_other_root", {"model_source": {"builder": "x:y"}}, str(tmp_path))

    other_proj = workspace.project_path("chestnut_burr_other")
    (other_proj / ".tcip").mkdir(parents=True)

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _watch_wall_clock(proc, run, "exp_timeout_other_root", 0.2, root=launch_root)
        workspace.activate_project("chestnut_burr_other")  # repins this process elsewhere

        deadline = time.monotonic() + 10
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, "watcher did not terminate the hung process"
        assert run.status == "failed"

        status: dict = {}
        for _ in range(50):
            try:
                status = read(status_key("exp_timeout_other_root", root=launch_root), default={})
            except DecodeError:
                status = {}
            if status.get("error"):
                break
            time.sleep(0.1)
        assert status.get("state") == "failed"
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
    """active_training_runs must count from the disk-aware _all_training_runs() (the internal
    function list_experiments(launched_only=True) and inspect_compute_resources() both build
    on), not the raw
    in-memory run_registry.list_runs(): a subprocess-delegated run's parent-side
    TrainRun.status never leaves its create_run-time "created" default once the child starts
    mutating its own separate copy, so the raw call always reports 0 for it. This is the common
    case for a subprocess-launched run, not an edge case, and it's the exact number the tool
    exists to give the agent before it decides whether to launch another concurrent run."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity, update_status
    from tcip_mcp.pipelines.training.run_registry import attach_run
    from tcip_mcp.tools.training_tools import inspect_compute_resources

    # _RUNS is a process-global registry other tests in this session also populate. Compare a
    # delta, not an absolute count, so this test doesn't depend on being run in isolation.
    baseline = inspect_compute_resources()["active_training_runs"]

    run = attach_run("run_active", {"model_source": {"builder": "x:y"}}, str(tmp_path))
    run.pid = 555  # subprocess-delegated, mirrors what launch_training does after Popen
    assert run.status == "created"  # the parent-side placeholder never advances

    create_experiment("run_active", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("run_active", "run_active", str(tmp_path))
    update_status("run_active", "running")  # only the child's disk write reflects reality

    assert inspect_compute_resources()["active_training_runs"] == baseline + 1


def test_inspect_compute_resources_counts_a_run_another_process_recorded(tmp_path, monkeypatch):
    """A run this process never held in memory at all, one another process launched and is
    still updating on disk with a fresh heartbeat, still counts: the disk record alone, with no
    in-memory _RUNS entry, is enough."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.tools.training_tools import inspect_compute_resources

    baseline = inspect_compute_resources()["active_training_runs"]

    create_experiment("run_other_process", {"model_source": {"builder": "x:y"}})
    update_status("run_other_process", "running")  # fresh heartbeat, no in-memory entry anywhere

    assert inspect_compute_resources()["active_training_runs"] == baseline + 1


# ── HPO resource caps ─────────────────────────────────────────────────────────


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
    pattern for this function: a pure-math objective, no training) still finds the minimum when
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
        storage_path=str(tmp_path),
    )
    assert result["n_trials"] == 4
    assert result["best_value"] is not None


def test_tune_search_runs_despite_deprecated_ray_result_dir_variables(tmp_path, monkeypatch):
    """Ray Tune refuses to run while TUNE_RESULT_DIR or RAY_AIR_LOCAL_CACHE_DIR is set anywhere
    in the environment, even though storage_path alone decides where trial results land. A
    machine that still carries such a redirect must get a working sweep, stored under the
    caller's storage_path, with the variables left in the environment exactly as they were."""
    import os

    pytest.importorskip("ray")
    from tcip_mcp.pipelines.training.hpo import tune_search

    machine_scratch = str(tmp_path / "machine_scratch")
    monkeypatch.setenv("TUNE_RESULT_DIR", machine_scratch)
    monkeypatch.setenv("RAY_AIR_LOCAL_CACHE_DIR", machine_scratch)

    def obj(config, report):
        report((config["x"] - 2.0) ** 2)

    result = tune_search(
        obj,
        param_space={"x": {"type": "uniform", "low": -5.0, "high": 5.0}},
        metric="objective", mode="min", num_samples=2,
        search_alg="random", scheduler="none",
        resources_per_trial={"cpu": 1.0, "gpu": 0.0},
        storage_path=str(tmp_path / "sweep_store"),
    )
    assert result["n_trials"] == 2
    assert result["best_value"] is not None
    assert (tmp_path / "sweep_store").is_dir()
    assert os.environ["TUNE_RESULT_DIR"] == machine_scratch
    assert os.environ["RAY_AIR_LOCAL_CACHE_DIR"] == machine_scratch


def test_tune_search_refuses_to_run_without_a_storage_path():
    """Trial results land where the caller says; with no storage_path Ray would fall back to a
    home-directory default outside any project, so the call refuses and names the resolver."""
    from tcip_mcp.pipelines.training.hpo import tune_search

    with pytest.raises(ValueError, match="storage_path"):
        tune_search(
            objective_fn=lambda config, report: report(0.0),
            param_space={"x": {"type": "uniform", "low": 0.0, "high": 1.0}},
        )

"""The audited training envelope: audit-around-body, custom train(ctx) dispatch, and ctx sinks.

Proves the envelope guarantees hold around any training body (default trainer or a custom
``train(ctx)``): the run is bracketed by audit events, env provenance is snapshotted, checkpoints
saved through ``ctx`` are stamped, and completion registers the model + lineage + artifact.
"""

from __future__ import annotations

import pytest

import tcip_store as ts

torch = pytest.importorskip("torch")

from tcip_mcp.audit import audit_log_key  # noqa: E402
from tcip_mcp.experiments import artifacts_key, env_key, lineage_key  # noqa: E402
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope  # noqa: E402
from tcip_mcp.pipelines.training.run_registry import create_run  # noqa: E402


def _audit_events(root, tool="training_run"):
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == tool]


# --------------------------------------------------------------------------
# Custom train(ctx) dispatch: the agent's own loop drives training via ctx.
# --------------------------------------------------------------------------

def _agent_train(ctx):
    """A minimal custom loop: uses ctx sinks, leaves status for the envelope to mark completed."""
    assert ctx.should_cancel() is False
    ctx.log_metrics(1, {"train_loss": 0.5, "val_loss": 0.4})
    ctx.save_checkpoint(
        {"model_state_dict": {}, "metrics": {"val_loss": 0.4, "epoch": 1}}, "model_best")


def test_envelope_dispatches_to_custom_train_and_guarantees_provenance(tmp_path):
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    out = tmp_path / "out"
    config = {
        "model_source": {"builder": "x:y", "task": "detection", "in_chans": 3},
        "training_source": f"{__name__}:_agent_train",
        "device": "cpu",
    }
    create_experiment("expE", config, data_source="imgs")
    update_status("expE", "running")
    run = create_run(config, str(out))

    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="detection",
                       experiment_id="expE")
    run_training_envelope(ctx)

    # Custom loop ran via ctx: its rows land on the experiment's own metrics log (the record
    # that tracks the run), not beside the weights in a separately-computed output dir.
    assert run.status == "completed"
    assert not (out / "metrics.jsonl").exists()
    from tcip_mcp.experiments import read_metrics

    assert [row["epoch"] for row in read_metrics("expE")] == [1]
    best = torch.load(out / "model_best.pt", weights_only=False)
    assert best["kind"] == KIND_TCIP_MODULE
    assert best["model_source"] == config["model_source"]

    # Body is bracketed on the append-only audit log (open running + close completed).
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]
    assert events[-1]["arguments"]["run_id"] == run.run_id

    # Env/source provenance snapshotted into the immutable experiment dir.
    env = ts.read(env_key("expE"))
    assert env["model_kind"] == KIND_TCIP_MODULE
    assert env["env"]["torch"]

    # Completion registered the model with the bespoke kind + recorded lineage + artifact.
    entry = ModelRegistry(str(tmp_path)).get_model("expE")
    assert entry is not None and entry["kind"] == KIND_TCIP_MODULE
    lineage = ts.read(lineage_key("expE"))
    assert lineage["model_weights"].endswith("model_best.pt")
    artifacts = ts.read(artifacts_key("expE"))
    assert "model_weights" in artifacts
    # The digest completion recorded is the same fact in both members: complete_run's one
    # transaction takes one hash of the one file and writes it into both.
    assert lineage["model_weights_sha256"] == artifacts["model_weights"]["sha256"]
    # The entry's own experiment_id is the run's, the binding registration wrote.
    assert entry["sha256"] == lineage["model_weights_sha256"]
    assert entry["experiment_id"] == "expE"

    from tcip_mcp.pipelines.resolution import corroborated_producer

    assert corroborated_producer(entry["sha256"], "expE") == (entry["sha256"], "expE")


# --------------------------------------------------------------------------
# Default path: no training_source, so ctx.default_train() (today's trainer) runs, still audited.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# A phantom deliverable (no discoverable weights) must fail the run, not register a
# nonexistent path. An explicit ctx.set_final_weights() override still works.
# --------------------------------------------------------------------------

def _agent_train_default_tag_no_override(ctx):
    """Saves under the default tag ("checkpoint"), not model_best/model_final, and never
    calls set_final_weights. A loop like this produces no discoverable deliverable."""
    ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}})


def test_envelope_default_tag_with_no_override_fails_run_and_registers_nothing(tmp_path):
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    out = tmp_path / "out"
    config = {
        "model_source": {"builder": "x:y", "task": "detection", "in_chans": 3},
        "training_source": f"{__name__}:_agent_train_default_tag_no_override",
        "device": "cpu",
    }
    create_experiment("expF", config, data_source="imgs")
    update_status("expF", "running")
    run = create_run(config, str(out))

    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="detection",
                       experiment_id="expF")
    run_training_envelope(ctx)

    assert run.status == "failed"
    assert "final weights" in (run.error or "")
    assert ModelRegistry(str(tmp_path)).get_model("expF") is None
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "failed"]


def _agent_train_declares_a_path_it_never_wrote(ctx):
    """Declares its deliverable through set_final_weights at a path this loop never wrote."""
    import os

    ctx.set_final_weights(os.path.join(ctx.run.output_dir, "never_written.pt"))


def test_envelope_declared_deliverable_never_written_fails_run_and_registers_nothing(tmp_path):
    """The refusal partner of rail 16: a declared path this run cannot read is refused by
    complete_run, and the envelope marks the run failed rather than completing with an
    unrecorded digest, naming the path."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    out = tmp_path / "out"
    config = {
        "model_source": {"builder": "x:y", "task": "detection", "in_chans": 3},
        "training_source": f"{__name__}:_agent_train_declares_a_path_it_never_wrote",
        "device": "cpu",
    }
    create_experiment("expUnwritten", config, data_source="imgs")
    update_status("expUnwritten", "running")
    run = create_run(config, str(out))

    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="detection",
                       experiment_id="expUnwritten")
    run_training_envelope(ctx)

    assert run.status == "failed"
    assert "never_written.pt" in (run.error or "")
    assert ModelRegistry(str(tmp_path)).get_model("expUnwritten") is None
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "failed"]


def _agent_train_explicit_override(ctx):
    """Saves under a non-conventional tag, but explicitly declares it the deliverable."""
    path = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}}, "custom_tag")
    ctx.set_final_weights(path)


def test_envelope_explicit_set_final_weights_overrides_convention(tmp_path):
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    out = tmp_path / "out"
    config = {
        "model_source": {"builder": "x:y", "task": "detection", "in_chans": 3},
        "training_source": f"{__name__}:_agent_train_explicit_override",
        "device": "cpu",
    }
    create_experiment("expG", config, data_source="imgs")
    update_status("expG", "running")
    run = create_run(config, str(out))

    ctx = TrainContext(run=run, train_loader=None, val_loader=None, task="detection",
                       experiment_id="expG")
    run_training_envelope(ctx)

    assert run.status == "completed"
    entry = ModelRegistry(str(tmp_path)).get_model("expG")
    assert entry is not None
    assert entry["checkpoint_path"].endswith("custom_tag.pt")


# --------------------------------------------------------------------------
# Resume provenance: env.json records the resume request + whether RNG state was
# actually restored, refreshed after dispatch (not just the pre-dispatch request).
# --------------------------------------------------------------------------

def test_envelope_records_resume_provenance_in_env_json(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    import csv
    from PIL import Image
    from torch.utils.data import DataLoader
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.training.generic_trainer import task_collate, train
    from tcip_mcp.pipelines.training.run_registry import create_run as gt_create_run
    from tcip_mcp.experiments import create_experiment, update_status

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(6):
        Image.new("RGB", (32, 32), (40 * (i % 5), 50, 60)).save(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)

    def build_loader():
        ds = build_dataset("classification", images_dir=str(images_dir), csv_path=str(csv_path), num_classes=2)
        return DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "builder_kwargs": {"num_classes": 2}, "task": "classification"},
        "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 2}], "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False}, "checkpoint_every_n_epochs": 1,
    }
    # Generate the resumable checkpoint directly (not through the envelope).
    train(gt_create_run(dict(cfg), str(tmp_path / "out")), build_loader(), task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    # Resume through the full audited envelope: env.json must reflect the real outcome.
    create_experiment("expH", cfg)
    update_status("expH", "running")
    run = gt_create_run(dict(cfg), str(tmp_path / "out2"))
    ctx = TrainContext(run=run, train_loader=build_loader(), val_loader=None, task="classification",
                       experiment_id="expH", resume_from=str(ckpt))
    run_training_envelope(ctx)

    assert run.status == "completed"
    env = ts.read(env_key("expH"))
    assert env["resumed_from"] == str(ckpt)
    assert env["rng_state_restored"] is True


# --------------------------------------------------------------------------
# ctx.report_objective: a bespoke train(ctx)'s explicit primitive for reporting HPO
# trial progress, independent of the automatic epoch_hook/metric-key-guessing path.
# --------------------------------------------------------------------------

def test_report_objective_calls_trial_report_when_attached(tmp_path):
    run = create_run({"model_source": {"builder": "x:y"}}, str(tmp_path / "out"))
    reported: list = []
    ctx = TrainContext(run=run, train_loader=None, trial_report=reported.append)
    ctx.report_objective(3.14)
    assert reported == [3.14]


def test_report_objective_is_noop_outside_hpo(tmp_path):
    run = create_run({"model_source": {"builder": "x:y"}}, str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None)  # no trial_report, not an HPO trial
    ctx.report_objective(3.14)  # must not raise


def test_envelope_default_path_runs_default_train_and_audits(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.generic_trainer as gt
    from tcip_mcp.experiments import create_experiment, update_status

    out = tmp_path / "out"
    out.mkdir(parents=True)
    captured = {}

    def _stub_train(run, train_loader, val_loader=None, task="detection",
                    epoch_callback=None, resume_from=""):
        captured["epoch_callback"] = epoch_callback
        captured["called"] = True
        (out / "model_final.pt").write_bytes(b"stub")
        run.status = "completed"
        return run

    monkeypatch.setattr(gt, "train", _stub_train)

    config = {"model_source": {"builder": "x:y", "task": "classification"}, "device": "cpu"}
    create_experiment("expD", config)
    update_status("expD", "running")
    run = create_run(config, str(out))

    ctx = TrainContext(run=run, train_loader=None, experiment_id="expD", task="classification")
    run_training_envelope(ctx)

    assert captured.get("called") is True                     # dispatched to default_train
    assert captured["epoch_callback"] == ctx._epoch_sink       # experiment logging wired in
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]

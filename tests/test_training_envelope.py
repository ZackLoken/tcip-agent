"""The audited training envelope: audit-around-body, custom train(ctx) dispatch, and ctx sinks.

Proves the envelope guarantees hold around any training body (default trainer or a custom
``train(ctx)``): the run is bracketed by audit events, env provenance is snapshotted, checkpoints
saved through ``ctx`` are stamped, and completion registers the model + lineage + artifact.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run  # noqa: E402


def _audit_events(root, tool="training_run"):
    path = root / ".tcip" / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if json.loads(x).get("tool") == tool]


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

    # Custom loop ran via ctx: metrics.jsonl written, checkpoint stamped as a bespoke module.
    assert run.status == "completed"
    assert (out / "metrics.jsonl").is_file()
    best = torch.load(out / "model_best.pt", weights_only=False)
    assert best["kind"] == KIND_TCIP_MODULE
    assert best["model_source"] == config["model_source"]

    # Body is bracketed on the append-only audit log (open running + close completed).
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]
    assert events[-1]["arguments"]["run_id"] == run.run_id

    # Env/source provenance snapshotted into the immutable experiment dir.
    env = json.loads((tmp_path / ".tcip" / "experiments" / "expE" / "env.json").read_text())
    assert env["model_kind"] == KIND_TCIP_MODULE
    assert env["env"]["torch"]

    # Completion registered the model with the bespoke kind + recorded lineage + artifact.
    entry = ModelRegistry(str(tmp_path)).get_model("expE")
    assert entry is not None and entry["kind"] == KIND_TCIP_MODULE
    lineage = json.loads((tmp_path / ".tcip" / "experiments" / "expE" / "lineage.json").read_text())
    assert lineage["model_weights"].endswith("model_best.pt")
    artifacts = json.loads((tmp_path / ".tcip" / "experiments" / "expE" / "artifacts.json").read_text())
    assert "model_weights" in artifacts


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
    from tcip_mcp.pipelines.training.generic_trainer import create_run as gt_create_run, task_collate, train
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
    env = json.loads((tmp_path / ".tcip" / "experiments" / "expH" / "env.json").read_text())
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

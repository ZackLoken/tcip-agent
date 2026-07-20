"""S2 — the audited training envelope: audit-around-body + custom train(ctx) dispatch + ctx sinks.

Proves the envelope guarantees hold around ANY training body (default trainer or a custom
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
# Custom train(ctx) dispatch — the agent's own loop drives training via ctx.
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
# Default path — no training_source → ctx.default_train() (today's trainer), still audited.
# --------------------------------------------------------------------------

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

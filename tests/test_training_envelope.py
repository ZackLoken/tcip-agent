"""The audited training envelope: audit-around-body, custom train(ctx) dispatch, and ctx sinks.

Proves the envelope guarantees hold around any training body (default trainer or a custom
``train(ctx)``): the run is bracketed by audit events, checkpoints saved through ``ctx`` are
stamped, and completion writes the final status whose checkpoint the registry reads.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

import tcip_store as ts

torch = pytest.importorskip("torch")

from tcip_mcp.audit import audit_log_key  # noqa: E402
from tcip_mcp.experiments import observe  # noqa: E402
from tcip_mcp.pipelines.model_build import CONFIG_KEY, METRICS_KEY, STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext, run_training_envelope  # noqa: E402
from tests._producer_fixtures import dataset_over, run_over  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    completed_checkpoint,
    detection_config,
)


def _audit_events(root, tool="training_run"):
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == tool]


def _context(tmp_path, config: dict, **kwargs) -> tuple[TrainContext, Path]:
    """A context over a run directory the launcher's own writer opened over ``config``, its run
    training under the data section the launch resolved, as the child's own entry trains it."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(tmp_path, config, resume_from=kwargs.get("resume_from"))
    return TrainContext(run=observed_run(observe(run_dir)),
                        **{"train_loader": None, **kwargs}), run_dir


def _bespoke(tmp_path, body: str) -> dict:
    """A detector run over two frames of its own whose training body is ``body`` of this
    module."""
    return detection_config(tmp_path / "data", training_source=f"{__name__}:{body}",
                            device="cpu")


def _agent_train(ctx):
    """A minimal custom loop: uses ctx sinks, leaves status for the envelope to mark completed."""
    assert ctx.should_cancel() is False
    ctx.log_metrics(1, {"train_loss": 0.5, "val_loss": 0.4})
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4, "epoch": 1}}, "model_best")


def test_envelope_dispatches_to_custom_train_and_guarantees_provenance(tmp_path):
    from tcip_mcp.experiments import METRICS_FILE, read_rows
    from tcip_mcp.model_registry import load_registered_checkpoint

    config = _bespoke(tmp_path, "_agent_train")
    ctx, run_dir = _context(tmp_path, config)
    run_training_envelope(ctx)

    assert ctx.run.status == "completed"
    assert [row["epoch"] for row in read_rows(run_dir / METRICS_FILE)[0]] == [1]
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    assert best[CONFIG_KEY]["model_source"] == config["model_source"]
    assert "model_source" not in best

    # Body is bracketed on the append-only audit log (open running + close completed).
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]
    assert events[-1]["arguments"]["experiment_id"] == run_dir.name

    # Completion names the checkpoint, and the registry reads that one record for its producer.
    checkpoint = completed_checkpoint(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("model_best.pt")
    verified = load_registered_checkpoint(checkpoint["path"], project=tmp_path)
    assert verified.sha256 == checkpoint["sha256"]
    assert verified.experiment_id == run_dir.name


def _refuse_the_closing_audit_line(monkeypatch) -> None:
    """Every ``audit.append`` but a ``running`` line raises a full-disk ``OSError``."""
    import tcip_mcp.audit as audit

    real_append = audit.append

    def refuse_the_closing_line(key, entry):
        if entry["status"] != "running":
            raise OSError("the log's disk is full")
        real_append(key, entry)

    monkeypatch.setattr(audit, "append", refuse_the_closing_line)


def test_a_run_whose_closing_audit_line_is_refused_ends_failed_naming_it(tmp_path, monkeypatch):
    """The body completes and saves its deliverable; the closing ``training_run`` append is
    refused, so the final status reads ``failed`` naming the unwritten line, names no checkpoint,
    and the log holds the opening line alone."""
    _refuse_the_closing_audit_line(monkeypatch)
    ctx, run_dir = _context(tmp_path, _bespoke(tmp_path, "_agent_train"))
    run_training_envelope(ctx)

    final = observe(run_dir).final
    assert final["state"] == "failed"
    assert "audit entry could not be written" in final["status_error"]
    assert completed_checkpoint(run_dir) is None
    assert [e["status"] for e in _audit_events(tmp_path)] == ["running"]


def _agent_train_raises(ctx):
    """A body that fails on its own."""
    raise RuntimeError("the body exploded")


def test_a_failed_run_whose_closing_audit_line_is_refused_names_both_causes(
    tmp_path, monkeypatch,
):
    """The body fails and the closing append is refused: the final status names the body's
    error and the unwritten line, neither hiding the other."""
    _refuse_the_closing_audit_line(monkeypatch)
    ctx, run_dir = _context(tmp_path, _bespoke(tmp_path, "_agent_train_raises"))
    run_training_envelope(ctx)

    final = observe(run_dir).final
    assert final["state"] == "failed"
    assert "the body exploded" in final["status_error"]
    assert "audit entry could not be written" in final["status_error"]


def _agent_train_default_tag_no_override(ctx):
    """Saves under the default tag ("checkpoint"), not model_best/model_final, and never
    calls set_final_weights. A loop like this produces no discoverable deliverable."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}})


def test_envelope_default_tag_with_no_override_fails_run_and_completes_nothing(tmp_path):
    """A phantom deliverable (no discoverable weights) fails the run rather than completing with
    a nonexistent path."""
    ctx, run_dir = _context(tmp_path, _bespoke(tmp_path, "_agent_train_default_tag_no_override"))
    run_training_envelope(ctx)

    assert ctx.run.status == "failed"
    assert "final weights" in observe(run_dir).final["status_error"]
    assert completed_checkpoint(run_dir) is None
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "failed"]


def _agent_train_declares_a_path_it_never_wrote(ctx):
    """Declares its deliverable through set_final_weights under a tag this loop never saved, the
    file under that name written behind the context's back."""
    (ctx.run_dir / "never_written.pt").write_bytes(b"")
    ctx.set_final_weights("never_written")


def test_envelope_declared_deliverable_never_written_fails_run_and_completes_nothing(tmp_path):
    """A declared tag this run's own saves never wrote ends the run failed rather than completing
    with a file the run did not save, naming the tag."""
    ctx, run_dir = _context(
        tmp_path, _bespoke(tmp_path, "_agent_train_declares_a_path_it_never_wrote"))
    run_training_envelope(ctx)

    assert ctx.run.status == "failed"
    assert "'never_written'" in observe(run_dir).final["status_error"]
    assert completed_checkpoint(run_dir) is None
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "failed"]


def _agent_train_explicit_override(ctx):
    """Saves under a non-conventional tag, but explicitly declares it the deliverable."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "custom_tag")
    ctx.set_final_weights("custom_tag")


def test_envelope_explicit_set_final_weights_overrides_convention(tmp_path):
    ctx, run_dir = _context(tmp_path, _bespoke(tmp_path, "_agent_train_explicit_override"))
    run_training_envelope(ctx)

    assert ctx.run.status == "completed"
    checkpoint = completed_checkpoint(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("custom_tag.pt")


def test_a_resumed_run_records_its_resume_checkpoint_and_completes(tmp_path):
    """The run resumed from a checkpoint names it in its launch record and completes."""
    pytest.importorskip("torchvision")
    import csv

    from PIL import Image
    from torch.utils.data import DataLoader

    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.registry_paths import stored_path
    from tests.tiny_trainer_fixtures import trainer_run

    images_dir = tmp_path / "images" / UNDATED_BUCKET
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
        ds = dataset_over("classification", str(images_dir), str(csv_path))
        return DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))

    _ds, data = run_over("classification", str(images_dir), str(csv_path))
    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_classifier",
                         "task": "classification"},
        "data": data,
        "device": "cpu", "stages": [{"freeze_to": -1, "epochs": 2}], "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False}, "checkpoint_every_n_epochs": 1,
        "seed": 3,
    }
    # Generate the resumable checkpoint directly (not through the envelope).
    train(trainer_run(dict(cfg), tmp_path / "out", project=tmp_path, has_val_loader=False,
                      id="resume-source"),
          build_loader())
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    launched = {**cfg, "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path),
                                "split": {"seed": 3, "val_ratio": 0.15}}}
    ctx, run_dir = _context(tmp_path, launched, train_loader=build_loader(), val_loader=None,
                            resume_from=str(ckpt))
    run_training_envelope(ctx)

    assert ctx.run.status == "completed"
    assert read_record(run_dir / RUN_FILE)["resume_from"] == stored_path(ckpt, tmp_path)
    assert completed_checkpoint(run_dir) is not None


def test_report_objective_records_a_selection_row_that_reaches_the_epoch_hook(tmp_path):
    """A bespoke body's reported objective is one metrics row stamping it the ``selection`` under
    the run's objective, at the last epoch logged, and fires the epoch hook like any row."""
    from tcip_mcp.experiments import METRICS_FILE, read_rows

    seen: list = []
    ctx, run_dir = _context(tmp_path, detection_config(tmp_path / "data"),
                            epoch_hook=lambda epoch, metrics: seen.append((epoch, metrics)))
    ctx.log_metrics(2, {"train_loss": 0.5})
    ctx.report_objective(3.14)

    metric = ctx.run.objective["selection_metric"]
    assert seen[-1] == (2, {"selection": 3.14, "selection_metric": metric})
    last = read_rows(run_dir / METRICS_FILE)[0][-1]
    assert (last["epoch"], last["selection"], last["selection_metric"]) == (2, 3.14, metric)


def test_envelope_default_path_runs_default_train_and_audits(tmp_path, monkeypatch):
    import tcip_mcp.pipelines.training.generic_trainer as gt

    captured = {}

    from tests._verified_checkpoint_fixtures import checkpoint_file

    def _stub_train(run, train_loader, val_loader=None,
                    epoch_callback=None, batch_callback=None, resume_from=""):
        captured["epoch_callback"] = epoch_callback
        captured["batch_callback"] = batch_callback
        captured["called"] = True
        run.saved["model_final"] = checkpoint_file(Path(run.output_dir) / "model_final.pt", "stub")
        run.status = "completed"
        return run

    monkeypatch.setattr(gt, "train", _stub_train)

    ctx, _ = _context(tmp_path, detection_config(tmp_path / "data", device="cpu"))
    run_training_envelope(ctx)

    assert captured.get("called") is True                     # dispatched to default_train
    # The run's metrics log and its one TensorBoard writer, wired in through the same sinks a
    # custom loop uses.
    assert captured["epoch_callback"] == ctx.log_metrics
    assert captured["batch_callback"] == ctx.log_batch
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]

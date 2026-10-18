"""The audited training envelope: audit-around-body, custom train(ctx) dispatch, and ctx sinks.

The envelope's guarantees are checked around the default trainer and custom ``train(ctx)``
bodies: the run is bracketed by audit events, checkpoints saved through ``ctx`` are stamped,
and completion writes the final status whose checkpoint the registry reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts

torch = pytest.importorskip("torch")

from tcip_mcp.audit import audit_log_key  # noqa: E402
from tcip_mcp.experiments import observe  # noqa: E402
from tcip_mcp.pipelines.model_build import (
    CONFIG_KEY,
    METRICS_KEY,
    STATE_DICT_KEY,
)  # noqa: E402
from tcip_mcp.pipelines.training.envelope import (
    TrainContext,
    run_training_envelope,
)  # noqa: E402
from tests._producer_fixtures import dataset_over  # noqa: E402
from tests._training_values import (  # noqa: E402
    adamw_optimizer,
    drawn_split,
    evaluation_block,
    schedule,
    stage,
)
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    completed_checkpoint,
    detection_config,
    detector_declaring,
)


def _audit_events(root, tool="training_run"):
    events = ts.read_log(audit_log_key(root)).records
    return [e for e in events if e.get("tool") == tool]


def _context(tmp_path, config: dict, **kwargs) -> tuple[TrainContext, Path]:
    """Return a context over a run directory the launcher's own writer opened over ``config``.

    Its run is the record ``observed_run`` reads back from that directory, and its train_loader
    is None unless ``kwargs`` give one.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import opened_run

    run_dir = opened_run(
        tmp_path, config, resume_from=kwargs.get("resume_from")
    )
    return TrainContext(
        run=observed_run(observe(run_dir)), **{"train_loader": None, **kwargs}
    ), run_dir


def _bespoke(tmp_path, body: str) -> dict:
    """Return a detector run config over two frames whose training body is ``body`` here."""
    return detection_config(
        tmp_path / "data",
        project=tmp_path,
        model_source=detector_declaring(__file__),
        training_source=f"{Path(__file__).stem}:{body}",
    )


def _agent_train(ctx):
    """Assert no cancel, log one metrics row and save ``model_best`` through ctx, no status."""
    assert ctx.should_cancel() is False
    ctx.log_metrics(1, {"train_loss": 0.5, "val_loss": 0.4})
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4, "epoch": 1}},
        "model_best",
    )


def test_envelope_dispatches_to_custom_train_and_guarantees_provenance(
    tmp_path,
):
    from tcip_mcp.experiments import METRICS_FILE, read_rows
    from tcip_mcp.model_registry import load_registered_checkpoint

    config = _bespoke(tmp_path, "_agent_train")
    ctx, run_dir = _context(tmp_path, config)
    run_training_envelope(ctx)

    assert ctx.run.status == "completed"
    assert [row["epoch"] for row in read_rows(run_dir / METRICS_FILE)[0]] == [
        1
    ]
    from tcip_mcp.experiments import CONFIG_PATHS
    from tcip_mcp.registry_paths import runtime_paths

    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    stored = best[CONFIG_KEY]["model_source"]
    resolved = runtime_paths(best[CONFIG_KEY], CONFIG_PATHS, tmp_path)
    assert {**resolved["model_source"], "source_files": None} == {
        **config["model_source"],
        "source_files": None,
    }
    # Each declared file is named by its copy in the run's own snapshot.
    copies = [tmp_path / name for name in stored["source_files"]]
    assert [copy.read_bytes() for copy in copies] == [
        Path(name).read_bytes()
        for name in config["model_source"]["source_files"]
    ]
    assert all(copy.is_relative_to(run_dir / "model_src") for copy in copies)
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
    """Make every ``audit.append`` but a ``running`` line raise a full-disk ``OSError``."""
    import tcip_mcp.audit as audit

    real_append = audit.append

    def refuse_the_closing_line(key, entry):
        if entry["status"] != "running":
            raise OSError("the log's disk is full")
        real_append(key, entry)

    monkeypatch.setattr(audit, "append", refuse_the_closing_line)


def test_a_run_whose_closing_audit_line_is_refused_ends_failed_naming_it(
    tmp_path, monkeypatch
):
    """A run whose closing audit line is refused ends failed, naming the line.

    The body completes and saves its deliverable; the closing ``training_run`` append is
    refused, so the final status reads ``failed`` naming the unwritten line, names no checkpoint,
    and the log holds the opening line alone.
    """
    _refuse_the_closing_audit_line(monkeypatch)
    ctx, run_dir = _context(tmp_path, _bespoke(tmp_path, "_agent_train"))
    run_training_envelope(ctx)

    final = observe(run_dir).final
    assert final["state"] == "failed"
    assert "audit entry could not be written" in final["status_error"]
    assert completed_checkpoint(run_dir) is None
    assert [e["status"] for e in _audit_events(tmp_path)] == ["running"]


def _agent_train_raises(ctx):
    """Raise an error of the body's own."""
    raise RuntimeError("the body exploded")


def test_a_failed_run_whose_closing_audit_line_is_refused_names_both_causes(
    tmp_path,
    monkeypatch,
):
    """A failed run whose closing audit line is refused names both causes.

    The final status names the body's error and the unwritten line, neither hiding the other.
    """
    _refuse_the_closing_audit_line(monkeypatch)
    ctx, run_dir = _context(
        tmp_path, _bespoke(tmp_path, "_agent_train_raises")
    )
    run_training_envelope(ctx)

    final = observe(run_dir).final
    assert final["state"] == "failed"
    assert "the body exploded" in final["status_error"]
    assert "audit entry could not be written" in final["status_error"]


def _agent_train_default_tag_no_override(ctx):
    """Save one checkpoint under the default tag, ``checkpoint``, declaring no final weights."""
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}})


def test_envelope_default_tag_with_no_override_fails_run_and_completes_nothing(
    tmp_path,
):
    """A run saving under the default tag alone fails naming final weights and completes nothing."""
    ctx, run_dir = _context(
        tmp_path, _bespoke(tmp_path, "_agent_train_default_tag_no_override")
    )
    run_training_envelope(ctx)

    assert ctx.run.status == "failed"
    assert "final weights" in observe(run_dir).final["status_error"]
    assert completed_checkpoint(run_dir) is None
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "failed"]


def _agent_train_declares_a_path_it_never_wrote(ctx):
    """Write ``never_written.pt`` directly and declare it through set_final_weights."""
    (ctx.run_dir / "never_written.pt").write_bytes(b"")
    ctx.set_final_weights("never_written")


def test_envelope_declared_deliverable_never_written_fails_run_and_completes_nothing(
    tmp_path,
):
    """A deliverable declared but not saved through the context fails the run, naming its tag."""
    ctx, run_dir = _context(
        tmp_path,
        _bespoke(tmp_path, "_agent_train_declares_a_path_it_never_wrote"),
    )
    run_training_envelope(ctx)

    assert ctx.run.status == "failed"
    assert "'never_written'" in observe(run_dir).final["status_error"]
    assert completed_checkpoint(run_dir) is None
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "failed"]


def _agent_train_explicit_override(ctx):
    """Save under the tag ``custom_tag`` and declare it the final weights."""
    ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}, "custom_tag"
    )
    ctx.set_final_weights("custom_tag")


def test_envelope_explicit_set_final_weights_overrides_convention(tmp_path):
    ctx, run_dir = _context(
        tmp_path, _bespoke(tmp_path, "_agent_train_explicit_override")
    )
    run_training_envelope(ctx)

    assert ctx.run.status == "completed"
    checkpoint = completed_checkpoint(run_dir)
    assert checkpoint is not None
    assert checkpoint["path"].endswith("custom_tag.pt")


def test_a_resumed_run_records_its_resume_checkpoint_and_completes(tmp_path):
    """The run resumed from a checkpoint names it in its launch record and completes."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.pipelines.training.generic_trainer import run_loaders, train
    from tcip_mcp.registry_paths import stored_path
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [(i % 5) / 5 for i in range(6)],
        [i % 2 for i in range(6)],
        height=32,
        width=32,
    )

    def build_loader(run):
        """Return ``run``'s training loader, built by ``run_loaders`` over the six-frame dataset."""
        ds = dataset_over("classification", str(images_dir), str(csv_path))
        return run_loaders(run, ds, None)[0]

    from tests._chain_fixtures import CLASSIFIER_SOURCE, training_config

    launched = training_config(
        tmp_path,
        CLASSIFIER_SOURCE,
        {
            "images_dir": str(images_dir),
            "labels_dir": str(csv_path),
            "split": drawn_split(seed=3, val_ratio=0.15),
        },
        stages=[stage(-1)],
        scheduler=schedule("cosine", horizon_epochs=2),
        optimizer=adamw_optimizer(),
        evaluation=evaluation_block(selection_metric="loss"),
        seed=3,
    )
    # Generate the resumable checkpoint by training a first run directly (not through the
    # envelope).
    source, source_dir = _context(tmp_path, launched)
    train(source.run, build_loader(source.run))
    ckpt = source_dir / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    ctx, run_dir = _context(
        tmp_path, launched, val_loader=None, resume_from=str(ckpt)
    )
    ctx.train_loader = build_loader(ctx.run)
    run_training_envelope(ctx)

    assert ctx.run.status == "completed"
    assert read_record(run_dir / RUN_FILE)["resume_from"] == stored_path(
        ckpt, tmp_path
    )
    assert completed_checkpoint(run_dir) is not None


def test_report_objective_records_a_selection_row_that_reaches_the_epoch_hook(
    tmp_path,
):
    """A reported objective records a selection row that reaches the epoch hook.

    A bespoke body's reported value is one metrics row stamping it the ``selection`` under
    the config's selection metric, at the last epoch logged, and fires the epoch hook.
    """
    from tcip_mcp.experiments import METRICS_FILE, read_rows

    seen: list = []
    ctx, run_dir = _context(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        epoch_hook=lambda epoch, metrics: seen.append((epoch, metrics)),
    )
    ctx.log_metrics(2, {"train_loss": 0.5})
    ctx.report_objective(3.14)

    metric = ctx.spec.evaluation.selection_metric
    assert seen[-1] == (2, {"selection": 3.14, "selection_metric": metric})
    last = read_rows(run_dir / METRICS_FILE)[0][-1]
    assert (last["epoch"], last["selection"], last["selection_metric"]) == (
        2,
        3.14,
        metric,
    )


def test_ctx_evaluate_counts_at_the_runs_validated_conf_and_takes_none_of_its_own(
    tmp_path, monkeypatch
):
    """A bespoke body's evaluation is handed the run's validated conf and takes none of its own.

    The stubbed ``evaluate`` receives the conf the run's validated evaluation block states, and
    a call naming another conf raises TypeError naming ``conf_threshold``.
    """
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tests._training_values import VALIDATION_CONF

    handed: dict = {}
    monkeypatch.setattr(
        evaluation, "evaluate", lambda *a, **k: handed.update(k) or {}
    )
    ctx, _run_dir = _context(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )

    ctx.evaluate(object(), loader=[])
    assert handed["conf_threshold"] == VALIDATION_CONF
    with pytest.raises(TypeError, match="conf_threshold"):
        ctx.evaluate(object(), loader=[], conf_threshold=0.1)


def test_envelope_default_path_runs_default_train_and_audits(
    tmp_path, monkeypatch
):
    import tcip_mcp.pipelines.training.generic_trainer as gt

    captured = {}

    from tests._verified_checkpoint_fixtures import produced_checkpoint

    def _stub_train(
        run,
        train_loader,
        val_loader=None,
        epoch_callback=None,
        batch_callback=None,
        resume_from="",
    ):
        captured["epoch_callback"] = epoch_callback
        captured["batch_callback"] = batch_callback
        captured["called"] = True
        run.saved["model_final"] = produced_checkpoint(
            Path(run.output_dir) / "model_final.pt", "stub"
        )
        run.status = "completed"
        return run

    monkeypatch.setattr(gt, "train", _stub_train)

    ctx, _ = _context(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )
    run_training_envelope(ctx)

    assert captured.get("called") is True  # dispatched to default_train
    # The run's metrics log and its one TensorBoard writer, wired in through the same sinks a
    # custom loop uses.
    assert captured["epoch_callback"] == ctx.log_metrics
    assert captured["batch_callback"] == ctx.log_batch
    events = _audit_events(tmp_path)
    assert [e["status"] for e in events] == ["running", "completed"]

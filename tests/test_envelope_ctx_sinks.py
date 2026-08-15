"""The envelope-owned ``ctx`` sinks a hand-rolled ``train(ctx)`` routes through.

These are the promises ``TrainContext`` makes to a training body independent of dispatch: the
per-epoch signal reaches an HPO trial's pruner even with no experiment record in scope,
``metrics.jsonl`` accumulates rather than truncates, only real scalars reach TensorBoard, a
checkpoint lands under the tag it was asked for without stamping the caller's own live state,
cancellation is visible through the cross-process sentinel, and the calibration seam defaults
to this run's experiment without overriding one the caller named.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run  # noqa: E402

CONFIG = {"model_source": {"builder": "x:y", "task": "detection", "in_chans": 5}, "device": "cpu"}


class _RecordingWriter:
    """Stand-in summary writer recording exactly which scalars the sink routes to it."""

    def __init__(self):
        self.scalars: list[tuple] = []
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1


def test_epoch_signal_reaches_the_trial_hook_without_an_experiment(tmp_path):
    """An HPO trial runs with ``experiment_id=None``; its pruning signal must still fire."""
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    seen: list[tuple] = []
    ctx = TrainContext(run=run, train_loader=None, experiment_id=None,
                       epoch_hook=lambda epoch, metrics: seen.append((epoch, dict(metrics))))

    ctx.log_metrics(4, {"val_loss": 0.25})

    assert seen == [(4, {"val_loss": 0.25})]


def test_metrics_file_accumulates_one_row_per_epoch(tmp_path):
    """Every logged epoch survives; the file is a history, not a slot holding the last row."""
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, experiment_id=None)

    ctx.log_metrics(3, {"val_loss": 0.75, "map50": 0.10})
    ctx.log_metrics(7, {"val_loss": 0.25, "map50": 0.60})

    lines = (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    assert len(rows) == 2
    assert [r["epoch"] for r in rows] == [3, 7]
    assert [r["val_loss"] for r in rows] == [0.75, 0.25]
    assert [r["map50"] for r in rows] == [0.10, 0.60]


def test_only_real_scalars_reach_the_summary_writer(tmp_path):
    """A boolean flag or a text label is not a curve; plotting one misreads it as a number."""
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    writer = _RecordingWriter()
    ctx = TrainContext(run=run, train_loader=None, experiment_id=None, _tb=writer)

    ctx.log_metrics(9, {"val_loss": 0.25, "lr": 0.001, "early_stopped": True, "stage_name": "head"})

    assert sorted(writer.scalars) == [("lr", 0.001, 9), ("val_loss", 0.25, 9)]
    assert writer.flushes == 1


def test_checkpoint_lands_under_the_tag_it_was_asked_for(tmp_path):
    """Distinct tags are distinct files; a periodic save never overwrites the best one."""
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, experiment_id="expTagged")

    best = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.2}}, "model_best")
    periodic = ctx.save_checkpoint(
        {"model_state_dict": {}, "metrics": {"val_loss": 0.9}}, "checkpoint_epoch_3")

    assert best.endswith("model_best.pt")
    assert periodic.endswith("checkpoint_epoch_3.pt")
    saved_best = torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)
    saved_periodic = torch.load(tmp_path / "out" / "checkpoint_epoch_3.pt", weights_only=False)
    assert saved_best["metrics"]["val_loss"] == 0.2
    assert saved_periodic["metrics"]["val_loss"] == 0.9
    assert saved_best["experiment_id"] == "expTagged"
    assert saved_best["config"]["model_source"]["in_chans"] == 5


def test_a_checkpoint_tag_cannot_walk_out_of_the_run_directory(tmp_path):
    """A tag is a name inside the run, so one spelled as a path leaves nothing outside it.

    A bespoke loop names its own tags, and a checkpoint landing beside the run rather than in
    it is a weight file no provenance points at.
    """
    from tcip_store import BadKey

    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, experiment_id="expEscape")

    with pytest.raises(BadKey):
        ctx.save_checkpoint({"model_state_dict": {}}, "../escaped")

    assert not (tmp_path / "escaped.pt").exists()


def test_checkpoint_stamping_leaves_the_callers_state_untouched(tmp_path):
    """The stamp goes onto the saved payload, never back into the loop's own live state dict."""
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, experiment_id="expTagged")
    state = {"model_state_dict": {}, "metrics": {"val_loss": 0.2}}

    ctx.save_checkpoint(state, "model_best")

    assert set(state) == {"model_state_dict", "metrics"}
    assert torch.load(tmp_path / "out" / "model_best.pt", weights_only=False)["experiment_id"] == "expTagged"


def test_cancellation_is_seen_through_the_cross_process_sentinel(tmp_path):
    """A stop requested by another process must reach a loop polling ``ctx.should_cancel()``."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    run = create_run(dict(CONFIG), str(out))
    ctx = TrainContext(run=run, train_loader=None)

    assert ctx.should_cancel() is False
    (out / ".cancel_requested").write_text("")
    assert ctx.should_cancel() is True


def _spy_on_operating_point(monkeypatch):
    import tcip_mcp.pipelines.operating_point as op

    seen: dict = {}

    def _record(trait_name, **kwargs):
        seen.clear()
        seen.update({"trait_name": trait_name, **kwargs})
        return {"conf": 0.5}

    monkeypatch.setattr(op, "resolve_operating_point", _record)
    return seen


def test_calibration_defaults_to_the_experiment_this_run_belongs_to(tmp_path, monkeypatch):
    """The train-disjointness gate must check against the split this exact run drew."""
    seen = _spy_on_operating_point(monkeypatch)
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, experiment_id="expOwn")

    ctx.calibrate("catkin", calibration_records=[], holdout_records=[], tiled=False,
                  staged_conf_floor=0.05)

    assert seen["trait_name"] == "catkin"
    assert seen["experiment_id"] == "expOwn"
    assert seen["staged_conf_floor"] == 0.05


def test_calibration_keeps_an_experiment_the_caller_named(tmp_path, monkeypatch):
    """Calibrating against a different run's split is a caller decision, not one to overwrite."""
    seen = _spy_on_operating_point(monkeypatch)
    run = create_run(dict(CONFIG), str(tmp_path / "out"))
    ctx = TrainContext(run=run, train_loader=None, experiment_id="expOwn")

    ctx.calibrate("catkin", experiment_id="expOther", calibration_records=[], holdout_records=[],
                  tiled=False, staged_conf_floor=0.05)

    assert seen["experiment_id"] == "expOther"

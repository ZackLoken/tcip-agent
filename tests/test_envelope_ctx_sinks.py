"""The envelope-owned ``ctx`` sinks a hand-rolled ``train(ctx)`` routes through.

These are the promises ``TrainContext`` makes to a training body independent of dispatch: the
per-epoch signal reaches an HPO trial's pruner, the run's ``metrics.jsonl`` accumulates rather
than truncates, only real scalars reach TensorBoard, a checkpoint lands once under the tag it was
asked for without stamping the caller's own live state, an artifact is a file in the run's
directory, and cancellation is visible through the run's cancellation record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, METRICS_KEY, STATE_DICT_KEY  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402


def _context(tmp_path, **kwargs) -> tuple[TrainContext, Path]:
    """A context over a run directory the launcher's own writer opened under ``tmp_path``."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data", device="cpu"))
    return TrainContext(run=observed_run(observe(run_dir)), train_loader=None, **kwargs), run_dir


def _rows(run_dir: Path) -> list[dict]:
    """The run's metrics rows without the instant each was written."""
    from tcip_mcp.experiments import METRICS_FILE, read_rows

    return [{k: v for k, v in row.items() if k != "timestamp"}
            for row in read_rows(run_dir / METRICS_FILE)[0]]


class _RecordingWriter:
    """Stand-in summary writer recording exactly which scalars the sink routes to it."""

    def __init__(self):
        self.scalars: list[tuple] = []
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1


def test_epoch_signal_reaches_the_trial_hook(tmp_path):
    """An HPO trial's pruning signal fires with each epoch the body logs."""
    seen: list[tuple] = []
    ctx, _ = _context(tmp_path,
                      epoch_hook=lambda epoch, metrics: seen.append((epoch, dict(metrics))))

    ctx.log_metrics(4, {"val_loss": 0.25})

    assert seen == [(4, {"val_loss": 0.25})]


def test_the_sink_is_the_one_stamp_of_a_rows_epoch_and_instant(tmp_path):
    """Metrics carrying the epoch or the instant refuse naming the key and write nothing; the
    same metrics without it record the epoch the sink was handed."""
    from tcip_mcp.experiments import EPOCH_KEY, TIMESTAMP_KEY

    ctx, run_dir = _context(tmp_path)

    for reserved in (EPOCH_KEY, TIMESTAMP_KEY):
        with pytest.raises(ValueError, match=reserved):
            ctx.log_metrics(2, {reserved: 99, "val_loss": 0.5})
    assert _rows(run_dir) == []

    ctx.log_metrics(2, {"val_loss": 0.5})
    assert _rows(run_dir) == [{"val_loss": 0.5, EPOCH_KEY: 2}]


def test_metrics_file_accumulates_one_row_per_epoch(tmp_path):
    """Every logged epoch survives; the file is a history, not a slot holding the last row."""
    ctx, run_dir = _context(tmp_path)

    ctx.log_metrics(3, {"val_loss": 0.75, "map50": 0.10})
    ctx.log_metrics(7, {"val_loss": 0.25, "map50": 0.60})

    rows = _rows(run_dir)
    assert [r["epoch"] for r in rows] == [3, 7]
    assert [r["val_loss"] for r in rows] == [0.75, 0.25]
    assert [r["map50"] for r in rows] == [0.10, 0.60]


def test_a_diverged_metric_is_logged_as_null_beside_the_state_that_names_it(tmp_path):
    """A diverged loss is real information the run has to record, and NaN is not JSON: the
    row keeps the epoch, states the value is absent, and says why."""
    ctx, run_dir = _context(tmp_path)

    ctx.log_metrics(2, {"val_loss": float("nan"), "map50": 0.0})

    assert _rows(run_dir) == [{"epoch": 2, "val_loss": None, "val_loss_state": "nan", "map50": 0.0}]


def test_a_diverged_metric_still_reaches_the_pruning_hook_as_the_number_it_was(tmp_path):
    """The stored row cannot carry a non-finite value, but a pruner compares numbers, so the
    hook sees what the training body produced rather than the record's representation."""
    seen: list[dict] = []
    ctx, _ = _context(tmp_path, epoch_hook=lambda epoch, metrics: seen.append(dict(metrics)))

    ctx.log_metrics(1, {"val_loss": float("inf")})

    assert seen[0]["val_loss"] == float("inf")


def test_only_real_scalars_reach_the_summary_writer(tmp_path):
    """A boolean flag or a text label is not a curve; plotting one misreads it as a number."""
    writer = _RecordingWriter()
    ctx, _ = _context(tmp_path, _tb=writer)

    ctx.log_metrics(9, {"val_loss": 0.25, "lr": 0.001, "early_stopped": True, "stage_name": "head"})

    assert sorted(writer.scalars) == [("lr", 0.001, 9), ("val_loss", 0.25, 9)]
    assert writer.flushes == 1


def test_checkpoint_lands_under_the_tag_it_was_asked_for(tmp_path):
    """Distinct tags are distinct files; a periodic save never overwrites the best one."""
    ctx, run_dir = _context(tmp_path)

    best = ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2}}, "model_best")
    periodic = ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.9}}, "checkpoint_epoch_3")

    assert best.endswith("model_best.pt")
    assert periodic.endswith("checkpoint_epoch_3.pt")
    saved_best = torch.load(run_dir / "model_best.pt", weights_only=False)
    saved_periodic = torch.load(run_dir / "checkpoint_epoch_3.pt", weights_only=False)
    assert saved_best[METRICS_KEY]["val_loss"] == 0.2
    assert saved_periodic[METRICS_KEY]["val_loss"] == 0.9
    assert saved_best[CONFIG_KEY]["data"]["num_channels"] == 3


def test_a_checkpoint_name_written_twice_refuses_and_keeps_the_first(tmp_path):
    """A checkpoint is written once: a second save under one tag refuses and the file under that
    name stays the bytes the first save published."""
    ctx, run_dir = _context(tmp_path)
    ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2}}, "model_best")
    first = (run_dir / "model_best.pt").read_bytes()

    with pytest.raises(FileExistsError):
        ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.1}}, "model_best")

    assert (run_dir / "model_best.pt").read_bytes() == first
    assert not [p.name for p in run_dir.iterdir() if p.name.endswith(".staging")]


def test_a_checkpoint_tag_cannot_walk_out_of_the_run_directory(tmp_path):
    """A tag is a name inside the run, so one spelled as a path leaves nothing outside it.

    A bespoke loop names its own tags, and a checkpoint landing beside the run rather than in
    it is a weight file no provenance points at.
    """
    from tcip_store import BadKeyError

    ctx, run_dir = _context(tmp_path)

    with pytest.raises(BadKeyError):
        ctx.save_checkpoint({STATE_DICT_KEY: {}}, "../escaped")

    assert not (run_dir.parent / "escaped.pt").exists()


def test_checkpoint_stamping_leaves_the_callers_state_untouched(tmp_path):
    """The stamp goes onto the saved payload, never back into the loop's own live state dict."""
    ctx, run_dir = _context(tmp_path)
    state = {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.2}}

    ctx.save_checkpoint(state, "model_best")

    assert set(state) == {STATE_DICT_KEY, METRICS_KEY}
    assert CONFIG_KEY in torch.load(run_dir / "model_best.pt", weights_only=False)


def test_record_artifact_copies_the_file_into_the_run(tmp_path):
    """An artifact is a file in the run's own directory and never the run's deliverable, and a
    second recording under one name refuses."""
    ctx, run_dir = _context(tmp_path)
    source = tmp_path / "stderr.txt"
    source.write_text("trace")

    ctx.record_artifact("failure_log", str(source))

    assert ctx.run.deliverable is None
    assert (run_dir / "artifacts" / "failure_log").read_text() == "trace"
    with pytest.raises(FileExistsError):
        ctx.record_artifact("failure_log", str(source))


def test_an_artifact_name_cannot_walk_out_of_the_run_directory(tmp_path):
    """An artifact name is a name inside the run, so one spelled as a path copies nothing."""
    from tcip_store import BadKeyError

    ctx, run_dir = _context(tmp_path)
    source = tmp_path / "stderr.txt"
    source.write_text("trace")

    with pytest.raises(BadKeyError):
        ctx.record_artifact("../../escaped", str(source))

    assert not (run_dir.parent / "escaped").exists()


def test_cancellation_is_seen_through_the_runs_cancellation_record(tmp_path):
    """A stop requested by another process must reach a loop polling ``ctx.should_cancel()``."""
    from tcip_mcp.experiments import request_cancel

    ctx, run_dir = _context(tmp_path)

    assert ctx.should_cancel() is False
    request_cancel(run_dir)
    assert ctx.should_cancel() is True

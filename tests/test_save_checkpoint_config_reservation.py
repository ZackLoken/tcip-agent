"""``TrainContext.save_checkpoint`` reserves the ``config`` key: the checkpoint's ``config`` is
always this run's own launch config, the record every publishing door reads a run's ``scope``
from, so a bespoke ``train(ctx)`` loop's own ``state`` carrying that key would silently displace
it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402


def _ctx(tmp_path) -> tuple[TrainContext, dict]:
    """A context over a run the launcher's own writer opened under ``tmp_path``, and the config
    its launch record states."""
    from tcip_mcp.experiments import RUN_FILE, read_record
    from tcip_mcp.pipelines.training.run_registry import TrainRun
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"))
    record = read_record(run_dir / RUN_FILE)
    run = TrainRun(id=run_dir.name, config=record["config"],
                   objective=record["resolved"]["objective"], output_dir=str(run_dir))
    return TrainContext(run=run, train_loader=None, val_loader=None), record["config"]


def test_save_checkpoint_refuses_a_state_carrying_its_own_config_key(tmp_path) -> None:
    ctx, _ = _ctx(tmp_path)

    with pytest.raises(ValueError, match="reserved for this run's own"):
        ctx.save_checkpoint({"model_state_dict": {},
                             "config": {"data": {"scope": {"subject": "shoot"}}}})


def test_save_checkpoint_writes_the_launch_config_never_the_loops_own(tmp_path) -> None:
    ctx, launched = _ctx(tmp_path)

    path = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}})

    payload = torch.load(path, weights_only=False)
    assert payload["config"] == ctx.config
    assert payload["config"]["data"] == launched["data"]

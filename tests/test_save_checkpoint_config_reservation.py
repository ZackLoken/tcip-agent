"""``TrainContext.save_checkpoint`` reserves the ``config`` key the way it reserves
``schema_version``: the checkpoint's ``config`` is always this run's own launch config, the record
every publishing door reads a run's ``(subject, attribute, id_map)`` scope from, so a bespoke
``train(ctx)`` loop's own ``state`` carrying that key would silently displace it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tcip_mcp.pipelines.training.run_registry import create_run  # noqa: E402


def _ctx(tmp_path, config: dict) -> TrainContext:
    run = create_run(config, str(tmp_path / "out"))
    return TrainContext(run=run, train_loader=None, val_loader=None, task="detection")


def test_save_checkpoint_refuses_a_state_carrying_its_own_config_key(tmp_path) -> None:
    config = {"model_source": {"builder": "x:y"}, "data": {"subject": "bud", "attribute": "bud_opening"}}
    ctx = _ctx(tmp_path, config)

    with pytest.raises(ValueError, match="reserved for this run's own"):
        ctx.save_checkpoint({"model_state_dict": {}, "config": {"data": {"subject": "shoot"}}})


def test_save_checkpoint_writes_the_launch_config_never_the_loops_own(tmp_path) -> None:
    config = {"model_source": {"builder": "x:y"}, "data": {"subject": "bud", "attribute": "bud_opening"}}
    ctx = _ctx(tmp_path, config)

    path = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}})

    payload = torch.load(path, weights_only=False)
    assert payload["config"] == config
    assert payload["config"]["data"] == {"subject": "bud", "attribute": "bud_opening"}


def test_save_checkpoint_still_refuses_the_schema_version_reservation(tmp_path) -> None:
    """Coverage, alongside the new ``config`` reservation: the two rails share one shape, checked
    before either payload assembly runs."""
    config = {"model_source": {"builder": "x:y"}}
    ctx = _ctx(tmp_path, config)

    with pytest.raises(ValueError, match="reserved for this platform's own checkpoint"):
        ctx.save_checkpoint({"model_state_dict": {}, "schema_version": 99})

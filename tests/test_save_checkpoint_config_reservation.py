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
    its run trains under, read from its launch record."""
    from tcip_mcp.experiments import RUN_FILE, observe, read_record
    from tcip_mcp.pipelines.training.run_registry import TrainRun, trained_config
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run_dir = opened_run(tmp_path, detection_config(tmp_path / "data"))
    record = observe(run_dir).record
    run = TrainRun(id=run_dir.name, config=trained_config(record),
                   objective=record["resolved"]["objective"], project=tmp_path,
                   output_dir=str(run_dir))
    stored = trained_config(read_record(run_dir / RUN_FILE))
    return TrainContext(run=run, train_loader=None, val_loader=None), stored


def test_save_checkpoint_refuses_a_state_carrying_its_own_config_key(tmp_path) -> None:
    ctx, launched = _ctx(tmp_path)

    with pytest.raises(ValueError, match="reserved for this run's own"):
        ctx.save_checkpoint({"model_state_dict": {}, "config": launched})


def test_save_checkpoint_writes_the_launch_config_never_the_loops_own(tmp_path) -> None:
    """The checkpoint carries the config the run trains under, its launch config with the
    resolved data section, every field but the data locations only the run's record keeps."""
    ctx, launched = _ctx(tmp_path)

    path = ctx.save_checkpoint({"model_state_dict": {}, "metrics": {"val_loss": 0.4}})

    payload = torch.load(path, weights_only=False)
    locations = {"images_dir", "labels_dir"}
    assert {k: v for k, v in payload["config"].items() if k != "data"} == {
        k: v for k, v in launched.items() if k != "data"}
    assert payload["config"]["data"] == {
        k: v for k, v in launched["data"].items() if k not in locations}
    assert "images_dir" in launched["data"]

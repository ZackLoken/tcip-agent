"""``TrainContext.save_checkpoint`` reserves the ``config`` key: the checkpoint's ``config`` is
always this run's own launch config, the record every publishing door reads a run's ``scope``
from, so a bespoke ``train(ctx)`` loop's own ``state`` carrying that key would silently displace
it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import (  # noqa: E402
    CONFIG_KEY, METRICS_KEY, SNAPSHOT_KEY, STATE_DICT_KEY,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402


def _ctx(tmp_path) -> tuple[TrainContext, dict]:
    """A context over a run the launcher's own writer opened under ``tmp_path``, and the config
    its run trains under as its spec records it."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import detection_config, opened_run

    run = observed_run(observe(opened_run(tmp_path, detection_config(tmp_path / "data"))))
    return TrainContext(run=run, train_loader=None, val_loader=None), run.spec.record()


def test_save_checkpoint_refuses_a_state_carrying_its_own_config_key(tmp_path) -> None:
    ctx, launched = _ctx(tmp_path)

    with pytest.raises(ValueError, match="reserved for this run's own"):
        ctx.save_checkpoint({STATE_DICT_KEY: {}, CONFIG_KEY: launched})


def test_save_checkpoint_writes_the_launch_config_never_the_loops_own(tmp_path) -> None:
    """The checkpoint carries the config the run trains under, its launch config with the
    resolved data section and its source files bound to the run's snapshot, every field but the
    data locations only the run's record keeps, its paths read back against the project."""
    from tcip_mcp.experiments import CONFIG_PATHS
    from tcip_mcp.registry_paths import runtime_paths

    ctx, launched = _ctx(tmp_path)

    path = ctx.save_checkpoint({STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}})

    stored = runtime_paths(torch.load(path, weights_only=False)[CONFIG_KEY], CONFIG_PATHS,
                           tmp_path)
    locations = {"images_dir", "labels_dir"}
    assert {k: v for k, v in stored.items() if k != "data"} == {
        k: v for k, v in launched.items() if k != "data"}
    assert stored["data"] == {k: v for k, v in launched["data"].items() if k not in locations}
    assert "images_dir" in launched["data"]


def test_the_default_trainer_and_a_bespoke_loop_stamp_one_checkpoint_config(tmp_path) -> None:
    """The default trainer and a bespoke loop's ``ctx.save_checkpoint``, over one run the
    launcher's own writer opened, write one stamp: one ``config``, the model's and the dataset
    builder's declared files each this run's own snapshot copy of it, as the run's record names
    it, and no field of the run's data locations (``experiments.DATA_PATHS``), and one
    ``SNAPSHOT_KEY``, the run's own snapshot digest."""
    from tcip_mcp.experiments import DATA_PATHS, observe
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.registry_paths import _each_path, stored_path
    from tests._verified_checkpoint_fixtures import opened_run
    from tests.test_dataset_source_seam import BESPOKE_DS, BESPOKE_DS_FILE
    from tests.tiny_trainer_fixtures import (
        opposed_regression_loaders,
        regressor_config,
        write_regression_dataset,
    )

    intensities = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities, [2.0 * c for c in intensities])
    config = regressor_config(seed=3, data={
        "num_channels": 1, "scope": {}, "images_dir": str(images_dir),
        "labels_dir": str(csv_path), "split": {"seed": 1, "val_ratio": 0.15},
        "dataset_source": {"builder": BESPOKE_DS, "source_files": [BESPOKE_DS_FILE]}})
    observation = observe(opened_run(tmp_path, config))
    run = observed_run(observation)
    train_loader, val_loader = opposed_regression_loaders(run, [0.1, 0.4, 0.7], [0.2, 0.6])
    trained = train(run, train_loader, val_loader=val_loader)
    assert trained.status == "completed", trained.status_error
    path = TrainContext(run=run, train_loader=None).save_checkpoint(
        {STATE_DICT_KEY: {}}, "model_bespoke")

    stamped = {CONFIG_KEY, SNAPSHOT_KEY}
    by_trainer, by_loop = ({key: value for key, value in torch.load(
        saved, weights_only=False).items() if key in stamped}
        for saved in (trained.saved["model_final"], path))
    assert by_trainer == by_loop
    assert set(by_loop) == stamped
    stored = by_loop[CONFIG_KEY]

    snapshot = observation.record["source"]["files"]

    def copies(declared: list[str]) -> list[str]:
        return [stored_path(observation.directory / snapshot[stored_path(file, tmp_path)]["file"],
                            tmp_path) for file in declared]

    assert config["model_source"]["source_files"]
    assert stored["model_source"]["source_files"] == copies(
        config["model_source"]["source_files"])
    assert stored["data"]["dataset_source"]["source_files"] == copies([BESPOKE_DS_FILE])
    located: list[str] = []

    def found(value: str) -> str:
        located.append(value)
        return value

    for field in DATA_PATHS:
        _each_path(stored["data"], field, found)
    assert located == [], located

"""``TrainContext.save_checkpoint`` reserves the ``config`` key: the checkpoint's ``config`` is
always this run's own launch config, the record every publishing door reads a run's ``scope``
from, so a bespoke ``train(ctx)`` loop's own ``state`` carrying that key would silently displace
it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import (  # noqa: E402
    CONFIG_KEY,
    GEOMETRY_KEY,
    METRICS_KEY,
    SNAPSHOT_KEY,
    STATE_DICT_KEY,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402


def _ctx(tmp_path) -> tuple[TrainContext, dict]:
    """The context of a run the launcher's own writer opened under ``tmp_path``, and the config
    its run trains under as its spec records it.
    """
    from tests._verified_checkpoint_fixtures import detection_config
    from tests.tiny_trainer_fixtures import run_context

    ctx = run_context(tmp_path, detection_config(tmp_path / "data"))
    return ctx, ctx.run.spec.record()


def test_save_checkpoint_refuses_a_state_carrying_its_own_config_key(
    tmp_path,
) -> None:
    ctx, launched = _ctx(tmp_path)

    with pytest.raises(ValueError, match="reserved for this run's own"):
        ctx.save_checkpoint({STATE_DICT_KEY: {}, CONFIG_KEY: launched})


@pytest.mark.parametrize("key", [CONFIG_KEY, SNAPSHOT_KEY, GEOMETRY_KEY])
def test_a_checkpoint_written_past_the_stamp_refuses_naming_the_key_it_lacks(
    tmp_path, key
) -> None:
    """A produced checkpoint's payload saved back without one key the stamp writes refuses where
    the checkpoint is read, naming that key and the stamp, never on whatever the absent value
    leaves its reader to guess at.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import (
        register_checkpoint,
        registered_checkpoint,
    )

    payload = torch.load(
        registered_checkpoint(tmp_path), map_location="cpu", weights_only=False
    )
    del payload[key]
    unstamped = tmp_path / "unstamped.pt"
    torch.save(payload, unstamped)
    register_checkpoint(tmp_path, str(unstamped), name="unstamped")
    checkpoint = load_registered_checkpoint(str(unstamped), project=tmp_path)

    with pytest.raises(ValueError, match=rf"carries no '{key}'.*checkpoint_stamp"):
        (checkpoint.spec, checkpoint.layout, checkpoint.geometry)


def test_save_checkpoint_writes_the_launch_config_never_the_loops_own(
    tmp_path,
) -> None:
    """The checkpoint carries the config the run trains under, its launch config with the
    resolved data section and its source files bound to the run's snapshot, every field but the
    data locations only the run's record keeps, its paths read back against the project.
    """
    from tcip_mcp.experiments import CONFIG_PATHS
    from tcip_mcp.registry_paths import runtime_paths

    ctx, launched = _ctx(tmp_path)

    path = ctx.save_checkpoint(
        {STATE_DICT_KEY: {}, METRICS_KEY: {"val_loss": 0.4}}
    )

    stored = runtime_paths(
        torch.load(path, weights_only=False)[CONFIG_KEY],
        CONFIG_PATHS,
        tmp_path,
    )
    locations = {"images_dir", "labels_dir"}
    assert {k: v for k, v in stored.items() if k != "data"} == {
        k: v for k, v in launched.items() if k != "data"
    }
    assert stored["data"] == {
        k: v for k, v in launched["data"].items() if k not in locations
    }
    assert "images_dir" in launched["data"]


def test_the_default_trainer_and_a_bespoke_loop_stamp_one_checkpoint_config(
    tmp_path,
) -> None:
    """The default trainer and a bespoke loop's ``ctx.save_checkpoint``, over one run the
    launcher's own writer opened, write one stamp: one ``config``, the model's and the dataset
    builder's declared files each this run's own snapshot copy of it, as the run's record names
    it, and no field of the run's data locations (``experiments.DATA_PATHS``), one
    ``SNAPSHOT_KEY``, the run's own snapshot digest, and one ``GEOMETRY_KEY``.
    """
    from tcip_mcp.experiments import DATA_PATHS, observe
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.registry_paths import _each_path, stored_path
    from tests.test_dataset_source_seam import BESPOKE_DS, BESPOKE_DS_FILE
    from tests.tiny_trainer_fixtures import (
        opposed_regression_data,
        regressor_config,
        run_context,
    )

    config = regressor_config(
        seed=3,
        data={
            **opposed_regression_data(
                tmp_path / "ds",
                [0.1, 0.25, 0.4, 0.55, 0.7, 0.85],
                [0.2, 0.6, 0.9],
            ),
            "dataset_source": {
                "builder": BESPOKE_DS,
                "source_files": [BESPOKE_DS_FILE],
            },
        },
    )
    ctx = run_context(tmp_path, config)
    observation = observe(ctx.run_dir)
    trained = train(ctx.run, ctx.train_loader, val_loader=ctx.val_loader)
    assert trained.status == "completed", trained.status_error
    path = ctx.save_checkpoint({STATE_DICT_KEY: {}}, "model_bespoke")

    stamped = {CONFIG_KEY, SNAPSHOT_KEY, GEOMETRY_KEY}
    by_trainer, by_loop = (
        {
            key: value
            for key, value in torch.load(saved, weights_only=False).items()
            if key in stamped
        }
        for saved in (trained.saved["model_final"], path)
    )
    assert by_trainer == by_loop
    assert set(by_loop) == stamped
    stored = by_loop[CONFIG_KEY]

    snapshot = observation.record["source"]["files"]

    def copies(declared: list[str]) -> list[str]:
        return [
            stored_path(
                observation.directory
                / snapshot[stored_path(file, tmp_path)]["file"],
                tmp_path,
            )
            for file in declared
        ]

    assert config["model_source"]["source_files"]
    assert stored["model_source"]["source_files"] == copies(
        config["model_source"]["source_files"]
    )
    assert stored["data"]["dataset_source"]["source_files"] == copies(
        [BESPOKE_DS_FILE]
    )
    located: list[str] = []

    def found(value: str) -> str:
        located.append(value)
        return value

    for field in DATA_PATHS:
        _each_path(stored["data"], field, found)
    assert located == [], located

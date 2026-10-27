"""``TrainContext.save_checkpoint`` reserves the ``config`` key.

The checkpoint's ``config`` is this run's own launch config, the record a run's ``scope`` is read
from, and a bespoke ``train(ctx)`` loop's own ``state`` carrying that key is refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import (  # noqa: E402
    CONFIG_KEY,
    GEOMETRY_KEY,
    METRICS_KEY,
    SNAPSHOT_KEY,
    STATE_DICT_KEY,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402


def _ctx(tmp_path: Path) -> tuple[TrainContext, dict]:
    """Return a run's context and the config its spec records.

    The run is one the launcher's own writer opened under ``tmp_path``.
    """
    from tests._verified_checkpoint_fixtures import detection_config
    from tests.tiny_trainer_fixtures import run_context

    ctx = run_context(
        tmp_path, detection_config(tmp_path / "data", project=tmp_path)
    )
    return ctx, ctx.run.spec.record()


def test_save_checkpoint_refuses_a_state_carrying_its_own_config_key(
    tmp_path: Path,
) -> None:
    ctx, launched = _ctx(tmp_path)

    with pytest.raises(ValueError, match="reserved for this run's own"):
        ctx.save_checkpoint({STATE_DICT_KEY: {}, CONFIG_KEY: launched})


@pytest.mark.parametrize("key", [CONFIG_KEY, SNAPSHOT_KEY, GEOMETRY_KEY])
def test_a_checkpoint_written_past_the_stamp_refuses_naming_the_key_it_lacks(
    tmp_path: Path, key: str
) -> None:
    """A checkpoint lacking a key the stamp writes refuses where it is read, naming the key.

    A produced checkpoint's payload saved back without one key the stamp writes refuses where
    the checkpoint is read, naming that key and the stamp.
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

    with pytest.raises(
        ValueError, match=rf"carries no '{key}'.*checkpoint_stamp"
    ):
        (checkpoint.spec, checkpoint.layout, checkpoint.geometry)


def test_save_checkpoint_writes_the_launch_config_never_the_loops_own(
    tmp_path: Path,
) -> None:
    """``save_checkpoint`` writes the run's launch config.

    A checkpoint saved through an untrained run's context carries the config its spec records,
    paths read back against the project: each field of that config, and its data section but
    ``images_dir`` and ``labels_dir``.
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
    assert isinstance(stored, dict)
    locations = {"images_dir", "labels_dir"}
    assert {k: v for k, v in stored.items() if k != "data"} == {
        k: v for k, v in launched.items() if k != "data"
    }
    assert stored["data"] == {
        k: v for k, v in launched["data"].items() if k not in locations
    }
    assert "images_dir" in launched["data"]


def test_the_default_trainer_and_a_bespoke_loop_stamp_one_checkpoint_config(
    tmp_path: Path,
) -> None:
    """The default trainer and a bespoke loop stamp one checkpoint config.

    The default trainer and a bespoke loop's ``ctx.save_checkpoint``, over one run the
    launcher's own writer opened, write one stamp: one ``config``, the model's and the dataset
    builder's declared files each this run's own snapshot copy of it, as the run's record names
    it, and no field of the run's data locations (``experiments.DATA_PATHS``), one
    ``SNAPSHOT_KEY``, the run's own snapshot digest, and one ``GEOMETRY_KEY``.
    """
    from tcip_mcp.experiments import DATA_PATHS, observe_run
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
        project=tmp_path,
    )
    ctx = run_context(tmp_path, config)
    observation = observe_run(ctx.run_dir)
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

    snapshot = observation.record.source["files"]

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


def test_a_bound_runs_checkpoint_carries_its_split_whole_and_reads_it_back_bound(
    tmp_path: Path,
) -> None:
    """A bound run's checkpoint stamps its selection directory and reads it back bound.

    The stamped split's ``selection_dir`` is the selection's directory stored against the
    project, and the checkpoint's spec reads it back as a ``BoundSplit`` naming that directory
    absolutely.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.schemas import BoundSplit
    from tcip_mcp.registry_paths import stored_path
    from tests.test_feedback_tools import _bound_checkpoint
    from tests.test_selection_ground_truth_digests import _dataset, _draw

    manifest_dir = tmp_path / "manifest"
    root = tmp_path / "data"
    _dataset(root)
    _draw(tmp_path, root, manifest_dir)
    _run_dir, path = _bound_checkpoint(
        tmp_path, manifest_dir, "exp-bound-stamp"
    )

    stamped = torch.load(path, weights_only=False)[CONFIG_KEY]["data"]["split"]
    assert stamped["selection_dir"] == stored_path(manifest_dir, tmp_path)
    split = load_registered_checkpoint(path, project=tmp_path).spec.data.split
    assert isinstance(split, BoundSplit), split
    assert split.selection_dir == str(manifest_dir)

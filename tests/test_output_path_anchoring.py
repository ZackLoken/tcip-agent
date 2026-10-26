"""Output artifacts anchor to the project they are produced for.

A run's weights and logs, and a sweep's trials, land under the project's own ``.tcip`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET


def test_launch_training_defaults_into_the_projects_experiment_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched run's directory is its own under the project's experiments store.

    The run directory is ``.tcip/experiments/<experiment_id>`` under the project, the run
    completes, and its weights land there.
    """
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)

    from tcip_mcp.tools import training_tools
    from tests._chain_fixtures import BLOB_BUILDER
    from tests._producer_fixtures import (
        no_tensorboard,
        seed_bud_images,
        small_detection_config,
    )
    from tests._verified_checkpoint_fixtures import run_to_end

    images_dir = seed_bud_images(tmp_path / "images" / UNDATED_BUCKET, n=2)
    no_tensorboard(monkeypatch)

    res = training_tools.launch_training(
        tmp_path,
        small_detection_config(images_dir, BLOB_BUILDER, project=tmp_path),
        actor=None,
    )
    assert isinstance(res, training_tools.RunLaunched), res
    run_dir = Path(res.output_dir)
    assert run_dir == tmp_path / ".tcip" / "experiments" / res.experiment_id

    # The subprocess writes into this test's tmp root until it ends.
    assert run_to_end(tmp_path, res.experiment_id)["state"] == "completed"
    assert (run_dir / "model_best.pt").is_file() or (
        run_dir / "model_final.pt"
    ).is_file()

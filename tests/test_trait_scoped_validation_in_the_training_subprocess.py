"""A trait-scoped validation through the real training subprocess.

A training run launched under an evaluation naming a confirmed trait completes in its own
process, and the trait's record keeps its one revision.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_the_subprocess_resolves_the_authored_criterion_and_leaves_the_trait_entry_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training child resolves the trait's authored criterion and leaves the entry alone.

    The child runs in its own process under an evaluation naming the trait and completes, and
    the trait's record still holds the one revision confirmed at setup.
    """
    pytest.importorskip("torchvision")
    monkeypatch.chdir(tmp_path)
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools import training_tools
    from tcip_mcp.traits import read_trait
    from tests._chain_fixtures import BLOB_BUILDER, training_config
    from tests._producer_fixtures import no_tensorboard, seed_labeled_images
    from tests._training_values import drawn_split, evaluation_block
    from tests._trait_fixtures import entry, propose_and_confirm

    no_tensorboard(monkeypatch)

    images_dir = seed_labeled_images(
        tmp_path / "ds" / "images" / "train",
        [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30))],
        n=2,
        width=128,
        height=128,
    )
    proposed = propose_and_confirm(
        tmp_path, entry("leaf", ("leaf_length",), localization="center_match")
    )

    cfg = training_config(
        tmp_path,
        BLOB_BUILDER,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15),
        },
        evaluation=evaluation_block(trait="leaf"),
    )
    res = training_tools.launch_training(tmp_path, cfg, actor=None)
    assert "error" not in res, res
    assert res["pid"] != os.getpid()

    from tests._verified_checkpoint_fixtures import run_to_end

    assert (
        run_to_end(tmp_path, res["experiment_id"], seconds=180)["state"]
        == "completed"
    )

    assert read_trait("leaf", tmp_path).revisions == (proposed,)

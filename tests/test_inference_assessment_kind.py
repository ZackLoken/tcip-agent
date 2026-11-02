"""An inference pass publishes under a checkpoint's assessment and refuses a physical scale's.

Both assessments are made by their own producers: the checkpoint's by ``assess`` over a trained
detector's held-out reference, the scale's by ``calibrate_physical_scale``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_a_pass_publishes_under_the_checkpoint_assessment_it_names(
    tmp_path: Path,
) -> None:
    """A pass naming a passed checkpoint assessment publishes its bucket under that assessment."""
    from tcip_mcp.buckets import read_bucket
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-assessed-pass")

    assert (
        read_bucket(chain.root, chain.bucket).assessment_id
        == chain.assessment["assessment_id"]
    )


def test_a_pass_naming_a_physical_scale_assessment_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """A pass naming a physical-scale assessment is refused, naming the checkpoint's to use."""
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests import _trait_fixtures as fx
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tests.test_physical_scale_assessment import (
        DATE,
        _author_tolerance,
        _calibrate,
        _reference,
    )

    fx.seed_delivery_traits(tmp_path)
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    scale = _calibrate(tmp_path, selection, csv_path)

    refused = run_inference(
        tmp_path,
        registered_checkpoint(tmp_path),
        str(image_dir(tmp_path / "ds", DATE)),
        bucket=f"scaled/{DATE}",
        stated=Stated(),
        assessment_id=scale["assessment_id"],
        actor=None,
    )

    assert "measured a physical scale, not a checkpoint" in refused["error"]
    assert "name the checkpoint's assessment" in refused["error"]

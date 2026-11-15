"""The delivery's ordinal kappa and evaluate's ordinal kappa score one set of nearest ranks.

A model whose ordinal head emits fractional ranks is trained over a drawn selection of a
table-labeled dataset. Its checkpoint is assessed for a per-plant ordinal delivery, and the same
checkpoint is evaluated over the selection's holdout side through the evaluation pipeline; the two
routes' own records are compared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from tests._chain_fixtures import BESPOKE_FRACTIONAL_ORDINAL  # noqa: E402
from tests._training_values import (  # noqa: E402
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    IOU_THRESHOLD,
    RATIONALE,
    adamw_optimizer,
    drawn_split,
    schedule,
)

if TYPE_CHECKING:
    from pathlib import Path

FRAMES = 48
GRADES = (0, 2, 3)
"""The ranks the frames take, one band of frames each in order of brightness; the run's rank count
is the largest plus one."""
EPOCHS = 4
"""Epochs the head trains, enough to move it off a constant rank over these frames."""


def test_the_delivery_kappa_equals_the_evaluate_kappa_for_a_head_emitting_fractional_ranks(
    tmp_path: Path,
) -> None:
    """The delivery's kappa equals evaluate's over the holdout side, at the stored six places.

    The head emits each rank a fixed fraction below its count, so truncating it and rounding it
    to the nearest rank give different ranks; the test also holds that the truncated ranks would
    have scored a different kappa, so a delivery that truncated would be told apart.
    """
    from torch.utils.data import DataLoader

    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.data.split_construction import predictor_dataset
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.eval_runners import run_test_evaluation
    from tcip_mcp.pipelines.training.evaluation import (
        quadratic_weighted_kappa,
        rounded,
    )
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.traits import PER_PLANT_ORDINAL_AGGREGATE
    from tests import _trait_fixtures as fx
    from tests._producer_fixtures import register_dataset_at
    from tests._verified_checkpoint_fixtures import (
        bespoke_source,
        train_config,
        worker_run,
    )
    from tests.tiny_trainer_fixtures import write_regression_dataset

    fx.seed_delivery_traits(tmp_path)
    revision = fx.seed_confirmed_aggregate(
        tmp_path,
        "astringency",
        value_keys=["astringency"],
        delivery_kind=PER_PLANT_ORDINAL_AGGREGATE,
    )
    root = tmp_path / "ds"
    _images, table = write_regression_dataset(
        root,
        [i / FRAMES for i in range(FRAMES)],
        [GRADES[i * len(GRADES) // FRAMES] for i in range(FRAMES)],
        height=48,
        width=64,
    )
    register_dataset_at(tmp_path, root, crop="persimmon")
    selection = tmp_path / "selection"
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(selection),
        ground_truth=str(table),
        **drawn_split(
            seed=2, val_ratio=0.1, calibration_ratio=0.1, holdout_ratio=0.3
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    checkpoint = observe(
        worker_run(
            tmp_path,
            train_config(
                tmp_path,
                bespoke_source(BESPOKE_FRACTIONAL_ORDINAL, "ordinal"),
                {"split": {"selection_dir": str(selection)}},
                2,
                optimizer=adamw_optimizer(backbone_lr=1e-2, head_lr=1e-1),
                scheduler=schedule("cosine", horizon_epochs=EPOCHS),
            ),
            experiment_id="exp-fractional-ordinal",
        )
    ).checkpoint
    assert checkpoint is not None

    answered = assess_checkpoint(
        tmp_path,
        checkpoint_path=checkpoint.path,
        trait=revision.entry.name,
        delivery_kind=PER_PLANT_ORDINAL_AGGREGATE,
        selection_dir=str(selection),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=Stated(),
        actor=None,
    )
    assert "error" not in answered, answered
    delivered = read_assessment(
        tmp_path, answered["assessment_id"]
    ).criterion.scalar
    assert delivered is not None
    assert delivered.criterion == "quadratic_weighted_kappa"

    registered = load_registered_checkpoint(checkpoint.path, project=tmp_path)
    pass_ = prepare(
        registered, Stated().model_copy(update={"tile": False})
    ).runnable()
    loader = DataLoader(
        predictor_dataset(
            "ordinal",
            read_selection(selection, project=tmp_path).on("holdout"),
            registered.spec.data.recorded_scope,
            pass_.predictor,
            None,
        ),
        batch_size=4,
        collate_fn=task_collate("ordinal"),
    )
    evaluated = run_test_evaluation(
        pass_, loader, iou_threshold=IOU_THRESHOLD, trait=None
    )
    kappa = evaluated["quadratic_weighted_kappa"]
    assert kappa is not None
    assert delivered.score is not None
    assert rounded(delivered.score) == kappa

    model = pass_.predictor.model.eval()
    with torch.no_grad():
        emitted = torch.cat(
            [
                model(images.to(pass_.predictor.device))["head0_ranks"].cpu()
                for images, _targets in loader
            ]
        )
    truth = torch.cat([targets["ranks"] for _images, targets in loader])
    assert (emitted != emitted.round()).any()
    truncated = quadratic_weighted_kappa(
        emitted.trunc(), truth, pass_.predictor.dims["num_ranks"]
    )
    assert rounded(truncated) != kappa

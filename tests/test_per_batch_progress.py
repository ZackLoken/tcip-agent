"""Tests of per-batch progress rows.

The trainer emits them at a stated cadence, on the run's one metrics log, and no reader of that
log takes one for an epoch row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from torch import device, nn
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.execution import MaskCut
    from tcip_mcp.pipelines.schemas import ScoreWeights, TrainerReads
    from tcip_mcp.traits import TraitEntry, TraitRecord

torch = pytest.importorskip("torch")

from tcip_mcp.experiments import (  # noqa: E402
    EPOCH_KEY,
    METRICS_FILE,
    STEP_KEY,
    best_selection,
    epoch_rows,
    observe,
    partition_rows,
    read_rows,
    run_summary,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tests._training_values import evaluation_block  # noqa: E402
from tests.tiny_trainer_fixtures import (  # noqa: E402
    TRAIN_INTENSITIES,
    VAL_INTENSITIES,
    frames_data,
    opposed_regression_data,
    regressor_config,
    run_context,
)

TRAIT_SCOPED = evaluation_block(
    selection_metric="loss", iou_threshold=0.6, trait="bud_opening"
)
"""A regression run's evaluation block scoring under the confirmed bud-opening trait."""


def batch_rows(rows: list[dict]) -> list[dict]:
    """Return the per-batch rows among ``rows`` (``experiments.partition_rows``)."""
    return partition_rows(rows)[1]


def _context(
    tmp_path: Path,
    hook_calls: list,
    intensities: list[float] = TRAIN_INTENSITIES,
    *,
    validated: bool = True,
    **extra: object,
) -> TrainContext:
    """Return the context of a regression run opened by the launcher's own producer.

    The run is over opposed frames when ``validated``, else over its training frames alone with no
    validation side; its training frames are at ``intensities``, its epoch hook appends each epoch
    to ``hook_calls``, and ``extra`` are config keys.
    """
    root = tmp_path / "ds"
    data = (
        opposed_regression_data(root, intensities, VAL_INTENSITIES)
        if validated
        else frames_data(root, intensities, [2.0 * c for c in intensities])
    )
    config = regressor_config(2, data=data, project=tmp_path, **extra)
    return run_context(
        tmp_path,
        config,
        epoch_hook=lambda epoch, metrics: hook_calls.append(epoch),
    )


def test_a_run_admits_what_its_training_reads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building, training and twice evaluating a run read one admission of its training reads."""
    from tcip_mcp.pipelines.schemas import TrainConfigSchema

    calls: list[int] = []
    real = TrainConfigSchema.trainer_reads

    def counted(self: TrainConfigSchema) -> TrainerReads:
        calls.append(1)
        return real(self)

    monkeypatch.setattr(TrainConfigSchema, "trainer_reads", counted)
    ctx = _context(tmp_path, [])
    assert ctx.default_train().status == "completed"
    model = ctx.build_model()
    ctx.evaluate(model)
    ctx.evaluate(model)
    assert calls == [1]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_run_resolves_what_its_evaluation_reads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training and twice evaluating a trait-stating run read the trait's entry once.

    The run states a confirmed trait, and the registry's ``read_trait`` answers one call.
    """
    import tcip_mcp.traits as traits

    calls: list[int] = []
    real = traits.read_trait

    def counted(trait: str, project: str | Path) -> TraitRecord:
        calls.append(1)
        return real(trait, project)

    ctx = _context(tmp_path, [], evaluation=TRAIT_SCOPED)
    monkeypatch.setattr(traits, "read_trait", counted)
    assert ctx.default_train().status == "completed"
    model = ctx.build_model()
    ctx.evaluate(model)
    ctx.evaluate(model)
    assert calls == [1]


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_run_scores_under_the_trait_revision_its_launch_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's evaluations score under the trait revision its launch resolved.

    The run is launched against a trait's confirmed revision, then trained after the breeder
    confirms a newer one.
    """
    import tcip_mcp.pipelines.training.generic_trainer as generic_trainer
    from tests import _trait_fixtures as fx

    handed: list = []
    real = generic_trainer.evaluate

    def recorded(
        *args: nn.Module | DataLoader | device | str,
        **kwargs: Mapping[str, object]
        | float
        | ScoreWeights
        | TraitEntry
        | MaskCut
        | None,
    ) -> dict:
        handed.append(kwargs["trait"])
        return real(*args, **kwargs)

    monkeypatch.setattr(generic_trainer, "evaluate", recorded)
    ctx = _context(tmp_path, [], evaluation=TRAIT_SCOPED)
    launched = fx.latest("bud_opening", tmp_path)
    superseding = fx.propose_and_confirm(
        tmp_path,
        fx.with_fields(launched, notes="A newer statement of the trait."),
    )
    recorded = ctx.run.trait_revision
    assert recorded is not None
    assert recorded.trait_revision == superseding.number - 1
    assert ctx.default_train().status == "completed"
    assert handed and all(trait == launched for trait in handed)


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_a_run_whose_recorded_revision_holds_another_entry_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A run whose recorded revision holds another entry refuses by name.

    The body reads the trait entry at the revision its launch recorded, the entry's hash included:
    a record naming a hash that revision does not hold refuses with ``RevisionMovedError`` naming
    the revision's own hash.
    """
    from tcip_mcp.traits import RevisionMovedError

    ctx = _context(tmp_path, [], evaluation=TRAIT_SCOPED)
    recorded = ctx.run.trait_revision
    assert recorded is not None
    ctx.run.trait_revision = recorded.model_copy(
        update={"trait_revision_sha256": "0" * 64}
    )

    with pytest.raises(
        RevisionMovedError, match=recorded.trait_revision_sha256
    ):
        ctx.run.evaluation_arguments


@pytest.mark.usefixtures("seed_bud_trait_spec")
def test_the_trainers_validation_and_a_bodys_evaluation_hand_evaluate_one_keyword_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer's validation and ``ctx.evaluate`` hand ``evaluate`` the same keywords.

    The run states its own IoU threshold and a confirmed trait, and the trait's entry is among the
    keywords.
    """
    import tcip_mcp.pipelines.training.evaluation as evaluation
    import tcip_mcp.pipelines.training.generic_trainer as generic_trainer

    handed: dict[str, dict] = {}
    real = evaluation.evaluate

    def recorded(route: str) -> Callable[..., dict]:
        def evaluate(
            *args: nn.Module | DataLoader | device | str,
            **kwargs: Mapping[str, object]
            | float
            | ScoreWeights
            | TraitEntry
            | MaskCut
            | None,
        ) -> dict:
            handed[route] = kwargs
            return real(*args, **kwargs)

        return evaluate

    monkeypatch.setattr(generic_trainer, "evaluate", recorded("trainer"))
    monkeypatch.setattr(evaluation, "evaluate", recorded("body"))
    ctx = _context(tmp_path, [], evaluation=TRAIT_SCOPED)
    assert ctx.default_train().status == "completed"
    ctx.evaluate(ctx.build_model())

    assert handed["trainer"] == handed["body"]
    assert handed["body"]["iou_threshold"] == 0.6
    assert handed["body"]["trait"].name == "bud_opening"


def test_the_trainer_logs_a_batch_row_at_the_stated_cadence_and_no_reader_takes_it_for_an_epoch(
    tmp_path: Path,
) -> None:
    hook_calls: list = []
    ctx = _context(tmp_path, hook_calls, log_every_n_batches=2)
    run = ctx.default_train()
    assert run.status == "completed", run.status_error

    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    batches = batch_rows(rows)
    assert [(row[STEP_KEY], row[EPOCH_KEY]) for row in batches] == [
        (2, 1),
        (4, 2),
        (6, 2),
    ]
    assert all(isinstance(row["batch_loss"], float) for row in batches)

    epochs = epoch_rows(rows)
    assert [row[EPOCH_KEY] for row in epochs] == [1, 2]
    assert not any(STEP_KEY in row or "batch_loss" in row for row in epochs)
    assert hook_calls == [1, 2]
    summary = run_summary(observe(ctx.run_dir), rows, None)
    assert summary.current_epoch == 2


@pytest.mark.parametrize("n_batches", [3, 19, 29, 30, 31])
def test_a_config_stating_no_cadence_logs_ten_batch_rows_every_epoch(
    tmp_path: Path, n_batches: int
) -> None:
    """A config stating no cadence logs up to ten batch rows each epoch.

    Each of two epochs logs ``min(10, n_batches)`` distinct rows inside the epoch, the gaps between
    them differing by at most one batch, the last at the epoch's own final batch.
    """
    intensities = [0.1 + 0.6 * i / n_batches for i in range(n_batches)]
    ctx = _context(tmp_path, [], intensities, validated=False, batch_size=1)
    assert ctx.default_train().status == "completed"
    batches = batch_rows(read_rows(ctx.run_dir / METRICS_FILE)[0])
    for epoch in (1, 2):
        steps = [row[STEP_KEY] for row in batches if row[EPOCH_KEY] == epoch]
        assert len(steps) == len(set(steps)) == min(10, n_batches)
        assert all(
            (epoch - 1) * n_batches < step <= epoch * n_batches
            for step in steps
        )
        assert steps[-1] == epoch * n_batches
        gaps = [
            b - a for a, b in zip([(epoch - 1) * n_batches, *steps], steps)
        ]
        assert max(gaps) - min(gaps) <= 1, gaps


def test_the_best_selection_reads_epoch_rows_and_never_a_batch_row_s_selection(
    tmp_path: Path,
) -> None:
    """A batch row's ``selection``, better than every epoch's, does not become the best selection.

    The batch row is a bespoke loop's.
    """
    ctx = _context(tmp_path, [])
    ctx.log_metrics(1, {"selection": 0.5})
    ctx.log_batch(3, 1, {"selection": -100.0})
    ctx.log_metrics(2, {"selection": 0.7})

    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    for higher_is_better, best in ((False, 0.5), (True, 0.7)):
        assert best_selection(rows, higher_is_better) == best


def test_a_bespoke_loop_logs_a_batch_row_and_neither_sink_takes_the_other_s_stamp(
    tmp_path: Path,
) -> None:
    hook_calls: list = []
    ctx = _context(tmp_path, hook_calls)

    ctx.log_batch(5, 1, {"batch_loss": 0.3})
    ctx.log_metrics(1, {"train_loss": 0.4})
    with pytest.raises(ValueError, match="step"):
        ctx.log_batch(6, 1, {"step": 3})
    with pytest.raises(ValueError, match="step"):
        ctx.log_metrics(1, {"step": 3})

    rows = read_rows(ctx.run_dir / METRICS_FILE)[0]
    (batch,) = batch_rows(rows)
    assert (batch[STEP_KEY], batch[EPOCH_KEY], batch["batch_loss"]) == (
        5,
        1,
        0.3,
    )
    (epoch,) = epoch_rows(rows)
    assert epoch["train_loss"] == 0.4 and "batch_loss" not in epoch
    assert hook_calls == [1]

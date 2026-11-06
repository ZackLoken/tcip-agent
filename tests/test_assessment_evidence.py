"""An assessment's evidence reads back as the records its producers built, its failures named once.

Each case assesses through the platform's own doors over the chain fixtures and reads the record
back through ``read_assessment``.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from tests._chain_fixtures import assess, run_the_chain
from tests._training_values import (
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    RATIONALE,
    collection_pass,
    drawn_split,
)
from tools.generate_delivery_fixture import UNMET_CONF

pytest.importorskip("torch")


def _vocabulary() -> tuple[str, ...]:
    """Return the failure names an assessment may record."""
    from tcip_mcp.assessment_evidence import FailureName

    return typing.get_args(FailureName.__value__)


@pytest.fixture
def unmet(tmp_path: Path) -> tuple[Path, str]:
    """Return a chain project and the id of its checkpoint's assessment at :data:`UNMET_CONF`."""
    project = tmp_path
    chain = run_the_chain(project, experiment_id="exp-typed-evidence")
    answered = assess(
        project, chain.checkpoint_path, chain.selection_dir, conf=UNMET_CONF
    )
    assert "error" not in answered, answered
    return project, answered["assessment_id"]


def test_a_count_assessment_reads_back_as_its_evidence_record_its_failures_named_once(
    unmet: tuple[Path, str],
) -> None:
    """A failed count assessment reads back typed, each failure a vocabulary name, named once.

    The criterion is one evidence record holding the count member alone, and the per-class
    failure is stamped exactly when the record's own derivation finds a class whose bias is not
    equivalent to zero.
    """
    from tcip_mcp.assessment import read_assessment

    record = read_assessment(*unmet)
    criterion = record.criterion
    assert type(criterion).__name__ == "AssessmentEvidence", criterion
    count = criterion.count
    assert count is not None
    assert (
        criterion.classifier,
        criterion.scalar,
        criterion.scale,
        criterion.reserved_region,
    ) == (None, None, None, None)

    assert record.failures, record
    assert len(record.failures) == len(set(record.failures)), record.failures
    assert set(record.failures) <= set(_vocabulary()), record.failures
    failing = {
        cid
        for cid in count.per_class_typical_count
        if not count.scope_equivalence(cid)[0]
    }
    assert bool(failing) == (
        "count_bias_exceeds_tolerance_per_class" in record.failures
    ), (failing, record.failures)
    assert (not count.scope_equivalence()[0]) == (
        "count_bias_exceeds_tolerance" in record.failures
    ), record.failures


def test_each_scopes_derived_pass_agrees_with_the_bias_equivalence_over_its_curve_row(
    unmet: tuple[Path, str],
) -> None:
    """Each scope's derived pass and tolerance is the bias equivalence over its stored row.

    The equivalence runs at the trait's own tolerance fraction, the scope's typical count and
    the assessment's stated ``equivalence_z``, for the pooled holdout and every class.
    """
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.operationalization import latest_confirmed
    from tests import _trait_fixtures as fx

    count = read_assessment(*unmet).criterion.count
    assert count is not None
    from tcip_mcp.assessment_evidence import bias_equivalence

    frac = latest_confirmed(
        fx.COUNT_TRAIT, unmet[0]
    ).entry.count_bias_tolerance_frac
    assert frac is not None and count.count_bias_tolerance_frac == frac
    rows = count.holdout_at_conf["per_class"]
    assert isinstance(rows, dict) and rows
    scopes = {
        None: (count.holdout_at_conf, count.typical_count),
        **{
            cid: (rows[cid], count.per_class_typical_count[cid])
            for cid in count.per_class_typical_count
        },
    }
    for scope, (row, typical) in scopes.items():
        assert isinstance(row, dict)
        expected = bias_equivalence(
            row["count_bias_mean_present"],
            row["count_bias_std_present"],
            row["n_present"],
            tolerance_frac=frac,
            typical_count=typical,
            equivalence_z=EQUIVALENCE_Z,
        )
        assert count.scope_equivalence(scope) == expected, scope


def test_a_state_crossing_assessment_records_both_criteria_and_names_a_shared_failure_once(
    tmp_path: Path,
) -> None:
    """A state-crossing assessment over a one-image holdout records both members, a failure once.

    The count and the classifier criterion each find the holdout's single image too few, and
    the record names ``insufficient_holdout_images`` once.
    """
    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._chain_fixtures import SUBJECT, attributed_series

    series = attributed_series(
        tmp_path,
        fractions=(0.0, 1.0),
        assessed=False,
        experiment_id="exp-crossing-thin",
    )
    thin = tmp_path / "thin-selection"
    drawn = draw_splits(
        tmp_path,
        str(series.root),
        output_path=str(thin),
        subject=SUBJECT,
        **drawn_split(
            seed=2,
            val_ratio=0.2,
            calibration_ratio=0.3,
            holdout_ratio=0.025,
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    answered = assess_checkpoint(
        tmp_path,
        checkpoint_path=series.checkpoint_path,
        trait=series.trait,
        delivery_kind="state_crossing_dates",
        selection_dir=str(thin),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=collection_pass(),
        actor=None,
    )
    assert "error" not in answered, answered

    record = read_assessment(tmp_path, answered["assessment_id"])
    assert record.failures.count("insufficient_holdout_images") == 1, (
        record.failures
    )
    assert record.criterion.count is not None
    assert record.criterion.classifier is not None
    assert record.criterion.count.holdout_at_conf["n_present"] < 2
    assert record.criterion.classifier.positive_count_bias_images < 2

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


def test_a_regression_assessment_records_its_scalar_evidence_alone_and_pairs_by_convention(
    tmp_path: Path,
) -> None:
    """A regression checkpoint's assessment reads back with its scalar evidence alone.

    A regressor trained over a drawn selection of a table-labeled dataset is assessed for a
    per-plant regression delivery: the record reads back holding the scalar member and no other,
    its failures vocabulary names, its stored score without a state companion. A bucket
    published under it measured no count, so it pairs under the comparability convention: an
    annotation and a proposal whose boxes overlap above ``COMPARABILITY_IOU`` pair, index 0 to 0.
    """
    import json

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.assessment import (
        ASSESSMENT_FILE,
        assessment_dir,
        read_assessment,
    )
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import proposal_pairs
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.traits import PER_PLANT_REGRESSION_AGGREGATE
    from tests import _trait_fixtures as fx
    from tests._verified_checkpoint_fixtures import worker_run
    from tests.tiny_trainer_fixtures import (
        regressor_config,
        write_regression_dataset,
    )

    fx.seed_delivery_traits(tmp_path)
    revision = fx.seed_confirmed_aggregate(
        tmp_path,
        "plant_surface_area",
        value_keys=["area_mm2"],
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
    )
    root = tmp_path / "ds"
    intensities = [0.05 + 0.09 * i for i in range(10)]
    images_dir, table = write_regression_dataset(
        root, intensities, [2.0 * v for v in intensities]
    )
    selection = tmp_path / "selection"
    drawn = draw_splits(
        tmp_path,
        str(root),
        output_path=str(selection),
        ground_truth=str(table),
        **drawn_split(
            seed=2, val_ratio=0.2, calibration_ratio=0.2, holdout_ratio=0.3
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    checkpoint = observe(
        worker_run(
            tmp_path,
            regressor_config(
                1,
                data={"split": {"selection_dir": str(selection)}},
                project=tmp_path,
            ),
            experiment_id="exp-regression-assessed",
        )
    ).checkpoint
    assert checkpoint is not None

    answered = assess_checkpoint(
        tmp_path,
        checkpoint_path=checkpoint.path,
        trait=revision.entry.name,
        delivery_kind=PER_PLANT_REGRESSION_AGGREGATE,
        selection_dir=str(selection),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=Stated(),
        actor=None,
    )
    assert "error" not in answered, answered

    record = read_assessment(tmp_path, answered["assessment_id"])
    criterion = record.criterion
    assert criterion.scalar is not None, criterion
    assert (
        criterion.count,
        criterion.classifier,
        criterion.scale,
        criterion.reserved_region,
    ) == (None, None, None, None)
    assert set(record.failures) <= set(_vocabulary()), record.failures
    stored = json.loads(
        (
            assessment_dir(tmp_path, record.assessment_id) / ASSESSMENT_FILE
        ).read_text(encoding="utf-8")
    )
    assert set(stored["criterion"]["scalar"]) == {
        "criterion",
        "score",
        "floor",
        "n_holdout",
    }, stored["criterion"]

    published = run_inference(
        tmp_path,
        checkpoint_path=checkpoint.path,
        images_dir=str(images_dir),
        bucket="regression",
        assessment_id=record.assessment_id,
        stated=Stated(),
        actor=None,
    )
    assert "error" not in published, published
    bucket = read_bucket(root, "regression")
    assert bucket.assessment_id == record.assessment_id
    annotation = Annotation(
        subject="bud", geometry=BBox(10.0, 10.0, 30.0, 30.0)
    )
    proposal = Annotation(
        subject="bud", geometry=BBox(11.0, 11.0, 31.0, 31.0), score=0.9
    )
    assert proposal_pairs(tmp_path, bucket, [annotation], [proposal]) == {0: 0}


def test_a_state_crossing_assessment_refuses_below_the_floor_and_records_both_criteria_at_it(
    tmp_path: Path,
) -> None:
    """A state-crossing assessment refuses a holdout below the floor and records both at it.

    Over a holdout of one ground-truth-bearing image the assessment refuses naming the holdout
    side; over a holdout at least at the floor it records the count and the classifier
    criterion.
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

    def assessed(holdout_ratio: float) -> tuple[dict, dict]:
        selection = tmp_path / f"selection-{holdout_ratio}"
        drawn = draw_splits(
            tmp_path,
            str(series.root),
            output_path=str(selection),
            subject=SUBJECT,
            **drawn_split(
                seed=2,
                val_ratio=0.2,
                calibration_ratio=0.3,
                holdout_ratio=holdout_ratio,
            ),
            actor=None,
        )
        assert "error" not in drawn, drawn
        return drawn, assess_checkpoint(
            tmp_path,
            checkpoint_path=series.checkpoint_path,
            trait=series.trait,
            delivery_kind="state_crossing_dates",
            selection_dir=str(selection),
            equivalence_z=EQUIVALENCE_Z,
            dispersion_quantile=DISPERSION_QUANTILE,
            rationale=RATIONALE,
            stated=collection_pass(),
            actor=None,
        )

    thin, refused = assessed(0.025)
    assert thin["foreground_images"]["holdout"] == 1, thin
    assert "the holdout side holds 1" in refused.get("error", ""), refused

    wide, answered = assessed(0.1)
    assert wide["foreground_images"]["holdout"] >= wide["reference_floor"]
    assert "error" not in answered, answered
    record = read_assessment(tmp_path, answered["assessment_id"])
    assert record.criterion.count is not None
    assert record.criterion.classifier is not None

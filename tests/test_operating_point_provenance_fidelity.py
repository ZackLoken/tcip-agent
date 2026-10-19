"""What the count criterion's evidence says about itself matches what ran.

Covered: which derivation produced the conf, and what the reference's own scores looked like
against the floor it was collected at.
"""

from __future__ import annotations

import pytest

from tests import _trait_fixtures as fx
from tests._training_values import COUNT_TEST

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import count_criterion  # noqa: E402

N_IMAGES = 4
OBJECTS_PER_IMAGE = 8
ENTRY = fx.with_fields(
    fx.BUD_OPENING, count_bias_tolerance_frac=0.1, count_error_tolerance=1.0
)


def _records(
    prefix: str, offset: float, *, stray: tuple[dict, ...] = ()
) -> list[dict]:
    """Return ``N_IMAGES`` records built by ``evaluation.prediction_record``.

    Each image's 40 px GT boxes, 100 px apart from ``offset``, each get a detection at the same
    box scored 0.9, and ``stray`` is appended unchanged to the first image's detections.
    """
    from tcip_mcp.pipelines.training.evaluation import (
        dt_record,
        gt_record,
        prediction_record,
    )

    recs = []
    for i in range(N_IMAGES):
        boxes = [
            [offset + 100.0 * k, 50.0 + 10.0 * i, 40.0, 40.0]
            for k in range(OBJECTS_PER_IMAGE)
        ]
        dt = [dt_record(box, 1, 0.9) for box in boxes] + list(
            stray if i == 0 else ()
        )
        recs.append(
            prediction_record(
                dt,
                [gt_record(box, 1, 0) for box in boxes],
                width=4000,
                height=1000,
                cap=len(dt) + 1,
                count=len(dt),
                image_id=f"{prefix}{i}",
            )
        )
    return recs


def test_every_conf_label_the_registered_pickers_can_record_has_a_registered_implementation() -> (
    None
):
    """Every conf label a registered count picker records names a registered implementation.

    Each registered count objective is driven, and the two labels those runs produced are each
    found in the derivation registry.
    """
    from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    labels = set()
    for objective in sorted(COUNT_OBJECTIVE_PICKERS):
        _conf, evidence, _failures = count_criterion(
            _records("c", 0.0),
            _records("h", 100000.0),
            fx.with_fields(ENTRY, count_objective=objective),
            staged_conf_floor=0.05,
            staged_conf_floor_attribute_path=None,
            **COUNT_TEST,
        )
        labels.add(evidence["conf_derived_from"])

    assert (
        sorted(lbl for lbl in labels if lbl not in DERIVATION_IMPLEMENTATIONS)
        == []
    )
    assert len(labels) == 2  # two pickers


def test_the_lowest_score_across_the_reference_is_surfaced_and_never_fails_it() -> (
    None
):
    """The lowest score across the reference is surfaced as evidence and does not fail it.

    One holdout detection scores 0.05, the floor the reference was collected at, and the observed
    minimum is 0.05 with no failure; with that detection absent the minimum is 0.9.
    """
    from tcip_mcp.pipelines.training.evaluation import dt_record

    cal = _records("c", 0.0)
    hold = _records(
        "h",
        100000.0,
        stray=(dt_record([900000.0, 50.0, 40.0, 40.0], 1, 0.05),),
    )

    _conf, evidence, failures = count_criterion(
        cal,
        hold,
        ENTRY,
        staged_conf_floor=0.05,
        staged_conf_floor_attribute_path=None,
        **COUNT_TEST,
    )

    assert evidence["observed_min_score"] == pytest.approx(0.05)
    assert evidence["staged_conf_floor"] == pytest.approx(0.05)
    assert failures == []

    _conf, evidence, _failures = count_criterion(
        cal,
        _records("h", 100000.0),
        ENTRY,
        staged_conf_floor=0.05,
        staged_conf_floor_attribute_path=None,
        **COUNT_TEST,
    )
    assert evidence["observed_min_score"] == pytest.approx(0.9)

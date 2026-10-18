"""The derivation registry checked against the labels and numbers the calibration path produces.

A label the calibration path can stamp is registered under the text that path produces, and the
callable registered for a label reproduces the number the label was stamped on.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip(
    "torch"
)  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS  # noqa: E402
from tests._dense_op_fixtures import gt_only  # noqa: E402
from tests._trait_fixtures import IOU_JITTER_PX, IOU_MARGIN, confirm_bare  # noqa: E402


def _resolve(label: str):
    target = str(DERIVATION_IMPLEMENTATIONS[label])
    module, _, attr = target.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def test_every_count_objective_label_the_calibration_path_can_stamp_is_registered():
    """Every count-objective label the calibration path can stamp is registered.

    Every label the count-objective picker registry holds is a key of the derivation registry.
    """
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    labels = {label for _, label in COUNT_OBJECTIVE_PICKERS.values()}
    assert labels, "expected at least one registered count-objective picker"
    for label in labels:
        assert label in DERIVATION_IMPLEMENTATIONS, (
            f"picker label {label!r} is stamped on conf but is not registered in "
            "DERIVATION_IMPLEMENTATIONS"
        )


def test_the_registered_callable_for_an_iou_threshold_stamp_reproduces_the_stamped_number(
    tmp_path,
):
    """The callable registered for an IoU threshold's label reproduces the stamped number.

    The IoU threshold a match criterion reports carries a derivation label, and the callable that
    label resolves to, run on the same GT, returns that number.
    """
    trait = confirm_bare(
        tmp_path,
        "leaf",
        localization="iou_match",
        iou_jitter_px=IOU_JITTER_PX,
        iou_margin=IOU_MARGIN,
    ).entry
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion

    boxes = [(0, 0, 60, 60), (500, 0, 60, 60)]
    result = resolve_match_criterion(trait, gt_only(boxes), class_id=0)
    assert result["kind"] == "iou_match"

    label = result["derived_from"]
    assert label in DERIVATION_IMPLEMENTATIONS
    assert _resolve(label)(
        [list(boxes)], jitter_px=IOU_JITTER_PX, margin=IOU_MARGIN
    ) == (pytest.approx(result["iou_threshold"]))


def test_registering_a_picker_registers_the_labels_it_can_stamp(monkeypatch):
    """Registering a picker registers the labels it can stamp.

    A picker added to the picker registry under a new label finds that label in the derivation
    registry, at the implementation an existing picker's label names.
    """
    from tcip_mcp.pipelines import derivations, operating_point

    label = "a sweep registered by this test"
    monkeypatch.setitem(
        operating_point.COUNT_OBJECTIVE_PICKERS,
        "objective_registered_by_this_test",
        (lambda records: None, label),
    )

    live = derivations.DERIVATION_IMPLEMENTATIONS
    _picker, existing = next(
        iter(operating_point.COUNT_OBJECTIVE_PICKERS.values())
    )
    assert live[label] == live[existing]


def test_the_registered_callable_for_a_center_match_stamp_reproduces_the_stamped_tolerance(
    tmp_path,
):
    """The callable registered for a center-match label reproduces the stamped tolerance.

    On the center-match branch the reported tolerance is the registered callable's fraction
    scaled by the GT's own average characteristic size.
    """
    trait = confirm_bare(tmp_path, "leaf", localization="center_match").entry
    from tcip_mcp.pipelines.training.evaluation import (
        gt_class_avg_size,
        resolve_match_criterion,
    )

    boxes = [(0, 0, 20, 20), (40, 0, 20, 20), (80, 0, 20, 20)]
    per_image = gt_only(boxes)
    result = resolve_match_criterion(trait, per_image, class_id=0)
    assert result["kind"] == "center_match"

    label = result["derived_from"]
    assert label in DERIVATION_IMPLEMENTATIONS
    expected = _resolve(label)([list(boxes)]) * gt_class_avg_size(
        per_image, class_id=0
    )
    assert result["tolerance"] == pytest.approx(expected)

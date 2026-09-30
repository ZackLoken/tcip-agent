"""The derivation registry checked against what actually runs, not only against itself.

``DERIVATION_IMPLEMENTATIONS`` is the record that every ``derived_from`` label has a real
computation behind it. Reading it statically proves only that its own entries import. These tests
hold it to the two things a reader of an audit trail relies on: that a label the calibration path
can stamp is registered under the exact text that path produces, and that the callable registered
for a label is the one that produced the number the label was stamped on.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.derivations import DERIVATION_IMPLEMENTATIONS  # noqa: E402
from tests._dense_op_fixtures import gt_only  # noqa: E402
from tests._trait_fixtures import confirm_bare  # noqa: E402


def _resolve(label: str):
    target = str(DERIVATION_IMPLEMENTATIONS[label])
    module, _, attr = target.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def test_every_count_objective_label_the_calibration_path_can_stamp_is_registered():
    """The conf label is built from whichever picker ran, so the objective registry is the source
    of that text. Every base label it holds, and the review-verdict variant the same picker earns
    when it runs over confirmed review verdicts, must be registered against the same
    implementation. The label is assembled into a variable before it is stamped, so a registry key
    that drifts from the text the picker registry names is invisible to a scan of stamp sites."""
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    base_labels = {label for _, label in COUNT_OBJECTIVE_PICKERS.values()}
    assert base_labels, "expected at least one registered count-objective picker"
    for base in base_labels:
        assert base in DERIVATION_IMPLEMENTATIONS, (
            f"picker label {base!r} is stamped on conf but is not registered in "
            "DERIVATION_IMPLEMENTATIONS")
        variants = [k for k in DERIVATION_IMPLEMENTATIONS if k.startswith(base) and k != base]
        assert variants, (
            f"no review-reference variant of {base!r} is registered, so the label that same picker "
            "earns over review verdicts has no implementation recorded behind it")
        for variant in variants:
            assert DERIVATION_IMPLEMENTATIONS[variant] == DERIVATION_IMPLEMENTATIONS[base], (
                f"{variant!r} and {base!r} name the same picker running over different references "
                "and must record the same implementation")


def test_the_registered_callable_for_an_iou_threshold_stamp_reproduces_the_stamped_number(
    tmp_path,
):
    """The IoU threshold a match criterion reports carries a derivation label; the callable that
    label resolves to, run on the same GT, has to return that same number. A registry entry that
    names a different-but-importable derivation passes every static check while pointing an
    auditor at code that never produced the value."""
    trait = confirm_bare(tmp_path, "leaf", localization="iou_match").entry
    from tcip_mcp.pipelines.training.evaluation import resolve_match_criterion

    boxes = [(0, 0, 60, 60), (500, 0, 60, 60)]
    result = resolve_match_criterion(trait, gt_only(boxes), class_id=0)
    assert result["kind"] == "iou_match"

    label = result["derived_from"]
    assert label in DERIVATION_IMPLEMENTATIONS
    assert _resolve(label)([list(boxes)]) == pytest.approx(result["iou_threshold"])


def test_registering_a_picker_registers_the_labels_it_can_stamp(monkeypatch):
    """A picker's label, and the variant it earns over confirmed review verdicts, are the picker
    registry's own text. A second list of them would let a registered picker stamp a label no
    implementation is recorded for, which is the auditing gap this registry exists to close, so
    the labels are read from the registry rather than restated beside it."""
    from tcip_mcp.pipelines import derivations, operating_point

    label = "a sweep registered by this test"
    monkeypatch.setitem(
        operating_point.COUNT_OBJECTIVE_PICKERS, "objective_registered_by_this_test",
        (lambda records: None, label))

    live = derivations.DERIVATION_IMPLEMENTATIONS
    _picker, existing = next(iter(operating_point.COUNT_OBJECTIVE_PICKERS.values()))
    assert live[label] == live[existing]
    assert live[label + operating_point.REVIEW_VERDICT_LABEL_SUFFIX] == live[existing]


def test_the_registered_callable_for_a_center_match_stamp_reproduces_the_stamped_tolerance(
    tmp_path,
):
    """Same demand on the center-match branch, where the reported tolerance is the derived
    fraction scaled by the GT's own average characteristic size: both sides of the comparison run
    the real implementations, so the label is checked against the computation rather than against
    a number copied out of one of them."""
    trait = confirm_bare(tmp_path, "leaf", localization="center_match").entry
    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size, resolve_match_criterion

    boxes = [(0, 0, 20, 20), (40, 0, 20, 20), (80, 0, 20, 20)]
    per_image = gt_only(boxes)
    result = resolve_match_criterion(trait, per_image, class_id=0)
    assert result["kind"] == "center_match"

    label = result["derived_from"]
    assert label in DERIVATION_IMPLEMENTATIONS
    expected = _resolve(label)([list(boxes)]) * gt_class_avg_size(per_image, class_id=0)
    assert result["tolerance"] == pytest.approx(expected)

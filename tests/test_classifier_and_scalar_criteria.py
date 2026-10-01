"""The classifier and scalar criteria an assessment judges a reference by, at their own boundary.

The classifier criterion holds a state call to chance-corrected agreement above the trait's own
floor and to a present-scoped per-image positive-count bias equivalent to zero at a fraction of
the typical positive count; the scalar criterion holds a rank or value prediction to its
statistic above the trait's floor. Each refuses a holdout too thin to judge.
"""

from __future__ import annotations

import pytest

from tests import _trait_fixtures as fx

torch = pytest.importorskip("torch")

ENTRY = fx.with_fields(fx.BUD_OPENING, count_bias_tolerance_frac=0.01,
                       classifier_agreement_floor=0.41)


def _flipped_items(n: int, flips: set[int]) -> list[dict]:
    """``n`` classifier items one per image, the first half true positives, each index in
    ``flips`` predicted the other way."""
    return [{"image_id": f"i{i}", "is_true_positive": i < n // 2,
             "is_pred_positive": (i >= n // 2) if i in flips else (i < n // 2)} for i in range(n)]


def _classifier_items(prefix, n_images, pos_per_image, *, miscall_images=()):
    """``pos_per_image`` correctly called positives per image and one token negative (so kappa
    stays defined), plus one extra positive-called negative on each of ``miscall_images``: the
    same absolute miscall whatever the density."""
    items = []
    for i in range(n_images):
        image = f"{prefix}{i}"
        items += [{"image_id": image, "is_true_positive": True, "is_pred_positive": True}
                  for _ in range(pos_per_image)]
        if i in miscall_images:
            items.append({"image_id": image, "is_true_positive": False, "is_pred_positive": True})
        items.append({"image_id": image, "is_true_positive": False, "is_pred_positive": False})
    return items


def test_a_single_image_holdout_cannot_pass_on_its_zero_dispersion():
    """One holdout image has no images to vary its bias across, so the dispersion the equivalence
    test discounts by vanishes; it refuses as too thin rather than passing at the tolerance."""
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    cal = _flipped_items(20, set())
    hold = [{"image_id": "h0", "is_true_positive": i < 10, "is_pred_positive": i < 11}
            for i in range(20)]

    _evidence, failures = classifier_criterion(cal, hold, ENTRY)

    assert "insufficient_holdout_images" in failures


def test_the_positive_count_bias_is_scoped_to_images_carrying_a_positive():
    """An image with no true and no predicted positive contributes no certain zero to the bias,
    or forty such images would dilute a real two-per-image over-call toward nothing."""
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    hold = []
    for i in range(10):
        hold += [{"image_id": f"h{i}", "is_true_positive": True, "is_pred_positive": True}] * 100
        hold += [{"image_id": f"h{i}", "is_true_positive": False, "is_pred_positive": True}] * 2
    hold += [{"image_id": f"empty{i}", "is_true_positive": False, "is_pred_positive": False}
             for i in range(40)]

    evidence, failures = classifier_criterion(_flipped_items(20, set()), hold, ENTRY)

    assert evidence["typical_positive_count"] == pytest.approx(100.0)
    assert evidence["positive_count_bias_images"] == 10
    assert evidence["positive_count_bias"] == pytest.approx(2.0)
    assert "positive_count_bias_exceeds_tolerance" in failures


def test_the_same_miscall_fails_a_sparse_class_and_passes_a_dense_one():
    """The tolerance is a fraction of the typical positive count: one extra positive call on one
    holdout image of twenty refuses at one positive per image and passes at a hundred and fifty."""
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    entry = fx.with_fields(ENTRY, count_bias_tolerance_frac=0.1)
    sparse, sparse_failures = classifier_criterion(
        _classifier_items("c", 20, 1), _classifier_items("h", 20, 1, miscall_images=[0]), entry)
    dense, dense_failures = classifier_criterion(
        _classifier_items("c", 20, 150), _classifier_items("h", 20, 150, miscall_images=[0]),
        entry)

    assert "positive_count_bias_exceeds_tolerance" in sparse_failures
    assert dense["positive_count_bias"] == pytest.approx(sparse["positive_count_bias"])
    assert dense["positive_count_bias_std"] == pytest.approx(sparse["positive_count_bias_std"])
    assert dense["positive_count_bias_tolerance"] > sparse["positive_count_bias_tolerance"]
    assert "positive_count_bias_exceeds_tolerance" not in dense_failures


def test_the_trait_authored_agreement_floor_is_the_floor_applied():
    """A kappa of 0.8 clears a 0.41 floor and fails the 0.9 floor a trait authors."""
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    cal, hold = _flipped_items(20, set()), _flipped_items(100, set(range(10)))

    lenient, lenient_failures = classifier_criterion(cal, hold, ENTRY)
    strict, strict_failures = classifier_criterion(
        cal, hold, fx.with_fields(ENTRY, classifier_agreement_floor=0.9))

    assert 0.75 < lenient["kappa"] < 0.85
    assert "classifier_agreement_below_floor" not in lenient_failures
    assert strict["kappa_floor"] == 0.9
    assert "classifier_agreement_below_floor" in strict_failures


def _qwk(num_ranks: int):
    from tcip_mcp.pipelines.training.evaluation import quadratic_weighted_kappa

    return lambda pred, gt: quadratic_weighted_kappa(pred, gt, num_ranks)


def _items(true, predicted):
    return [{"image_id": f"h{i}", "true": t, "predicted": p}
            for i, (t, p) in enumerate(zip(true, predicted))]


def test_an_agreeing_ordinal_holdout_passes_and_a_cyclic_miscall_fails_below_the_floor():
    from tcip_mcp.pipelines.operating_point import scalar_criterion

    ranks = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1] * 2
    evidence, failures = scalar_criterion(_items(ranks, ranks), score=_qwk(3), floor=0.6,
                                          criterion="quadratic_weighted_kappa")
    assert failures == []
    assert evidence["score"] == pytest.approx(1.0)

    true = [0, 1, 2] * 8
    evidence, failures = scalar_criterion(_items(true, [2, 0, 1] * 8), score=_qwk(3), floor=0.6,
                                          criterion="quadratic_weighted_kappa")
    assert evidence["score"] <= evidence["floor"]
    assert failures == ["criterion_below_floor"]


def test_an_empty_holdout_refuses_as_too_thin_and_undefined():
    from tcip_mcp.pipelines.operating_point import scalar_criterion

    _evidence, failures = scalar_criterion([], score=_qwk(3), floor=0.6,
                                           criterion="quadratic_weighted_kappa")
    assert failures == ["insufficient_holdout_items", "criterion_undefined"]


@pytest.mark.parametrize("criterion", ["r_squared", "concordance_correlation_coefficient"])
def test_each_regression_statistic_is_the_one_recorded_and_a_skill_free_prediction_fails(
    criterion,
):
    from tcip_mcp.pipelines.operating_point import REGRESSION_CRITERIA, scalar_criterion

    values = [float(i) for i in range(20)]
    evidence, failures = scalar_criterion(_items(values, values),
                                          score=REGRESSION_CRITERIA[criterion], floor=0.5,
                                          criterion=criterion)
    assert (evidence["criterion"], failures) == (criterion, [])
    assert evidence["score"] == pytest.approx(1.0)

    evidence, failures = scalar_criterion(_items(values, [10.0 + v / 100 for v in values]),
                                          score=REGRESSION_CRITERIA[criterion], floor=0.5,
                                          criterion=criterion)
    assert failures == ["criterion_below_floor"]

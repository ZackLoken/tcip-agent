"""Each class's count-bias tolerance is scaled by that class's own typical per-image count.

The relative count-bias tolerance is a fraction of a derived density; on a multi-class reference
the pooled per-image count is the sum over every class. The fixtures here pair a dense class with a
rare one.
"""

from __future__ import annotations

import pytest

from tests import _trait_fixtures as fx
from tests._training_values import COUNT_TEST

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import count_criterion  # noqa: E402

ENTRY = fx.with_fields(
    fx.BUD_OPENING, count_bias_tolerance_frac=0.01, count_error_tolerance=100.0
)
N_IMAGES = 6
DENSE_PER_IMAGE = 250
RARE_PER_IMAGE = 10
SPURIOUS_PER_IMAGE = 2


def _records(prefix: str, offset: float, *, spurious: bool) -> list[dict]:
    """Return ``N_IMAGES`` records of ``DENSE_PER_IMAGE`` class-1 and ``RARE_PER_IMAGE`` class-2 GT.

    Each GT box gets a detection at the same box, scored 0.9. With ``spurious``, each image also
    gets ``SPURIOUS_PER_IMAGE`` class-2 detections in a row of their own. The 20 px boxes sit
    50 px apart from ``offset`` along each row.
    """
    recs = []
    for i in range(N_IMAGES):
        gt, dt = [], []
        for k in range(DENSE_PER_IMAGE):
            box = [offset + 50.0 * k, 100.0 + 10.0 * i, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 1, "iscrowd": 0})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        for k in range(RARE_PER_IMAGE):
            box = [offset + 50.0 * k, 900.0 + 10.0 * i, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 2, "iscrowd": 0})
            dt.append({"bbox": box, "category_id": 2, "score": 0.9})
        if spurious:
            for k in range(SPURIOUS_PER_IMAGE):
                dt.append(
                    {
                        "bbox": [
                            offset + 50.0 * k,
                            1500.0 + 10.0 * i,
                            20.0,
                            20.0,
                        ],
                        "category_id": 2,
                        "score": 0.9,
                    }
                )
        recs.append(
            {
                "image_id": f"{prefix}{i}",
                "width": 20000,
                "height": 3000,
                "gt": gt,
                "dt": dt,
                "count": len(dt),
                "cap": len(dt) + 1,
            }
        )
    return recs


def _criterion(*, spurious: bool):
    return count_criterion(
        _records("c", 0.0, spurious=spurious),
        _records("h", 100000.0, spurious=spurious),
        ENTRY,
        staged_conf_floor=0.05,
        staged_conf_floor_attribute_path=None,
        **COUNT_TEST,
    )


def test_a_rare_class_is_not_granted_the_dense_class_density_as_its_tolerance():
    """A rare class is judged against its own density, not the dense class's, as its tolerance.

    Class 2 carries 10 objects an image and is over-counted by 2 on every one of them, a 20%
    systematic miscount of that class. Class 1 carries 250 an image with no bias, so the pooled
    per-image density is 260: the pooled check does not fail, the rare class's tolerance is under a
    tenth of the pooled one, and the per-class check fails.
    """
    _conf, evidence, failures = _criterion(spurious=True)
    rare = evidence["per_class"]["2"]

    assert evidence["pooled_typical_count"] == pytest.approx(260.0)
    assert rare["typical_count"] == pytest.approx(float(RARE_PER_IMAGE))
    assert rare["n_present"] == N_IMAGES
    assert rare["bias"] == pytest.approx(float(SPURIOUS_PER_IMAGE))
    assert evidence["per_class"]["1"]["bias"] == pytest.approx(0.0)

    # The pooled scope is blind here, so nothing but the per-class scope can fail it.
    assert "count_bias_exceeds_tolerance" not in failures
    assert rare["tolerance"] < evidence["pooled_count_bias_tolerance"] / 10
    assert rare["bias"] > rare["tolerance"]
    assert (
        rare["passed"] is False
        and evidence["per_class"]["1"]["passed"] is True
    )
    assert "count_bias_exceeds_tolerance_per_class" in failures


def test_the_same_dense_and_rare_class_pairing_passes_when_every_class_is_honest():
    """A dense and rare class pairing counted correctly passes, both classes in the evidence."""
    _conf, evidence, failures = _criterion(spurious=False)

    assert set(evidence["per_class"]) == {"1", "2"}
    assert failures == []

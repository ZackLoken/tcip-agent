"""The derived per-image detection cap covers the dense tail of the reference it came from.

``derive_max_dets_from_counts`` is the one formula behind both the record-collection cap and the
resolved ``max_dets``, and it exists so that crowded scenes are not truncated. It therefore has to
read the tail of the per-image object-count distribution, not its bulk: on a skewed density
distribution the two are an order of magnitude apart, and a cap read off the bulk silently trims
detections from exactly the images the cap was meant to protect. The truncation lands on delivered
counts, never on an error.

The fixtures here are deliberately skewed. On a uniform per-image count distribution every quantile
of the distribution is the same number, so no fixture of that shape can distinguish the two.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.operating_point import (  # noqa: E402
    derive_max_dets_from_counts,
    resolve_operating_point,
)

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


def test_cap_admits_the_crowded_decile_of_a_skewed_count_distribution():
    """Ninety ordinary scenes at 100 objects and ten crowded ones at 1000: the cap has to clear the
    crowded ones, which sit ten times above the bulk of the distribution.
    """
    counts = [100] * 90 + [1000] * 10
    cap = derive_max_dets_from_counts(counts)

    assert cap == 1500
    assert cap > max(counts)  # no image in this reference is truncated by its own cap


def test_cap_on_a_uniform_count_distribution_is_unchanged_by_the_tail_rule():
    """The companion obligation: reading the tail must not inflate the cap for an ordinary
    reference whose images all carry about the same number of objects.
    """
    assert derive_max_dets_from_counts([100] * 100) == 150
    assert derive_max_dets_from_counts([2] * 20) == 100  # the documented floor still governs


def _skewed_calibration_records() -> list[dict]:
    """Nine sparse images (5 objects each) and one crowded image (300), each object exactly matched
    by one detection. Boxes sit 50 px apart, well outside any tolerance this GT derives.
    """
    recs = []
    per_image = [5] * 9 + [300]
    for i, n in enumerate(per_image):
        gt, dt = [], []
        for k in range(n):
            box = [50.0 * k, 100.0 + 10.0 * i, 20.0, 20.0]
            gt.append({"bbox": box, "category_id": 1})
            dt.append({"bbox": box, "category_id": 1, "score": 0.9})
        recs.append({"image_id": f"c{i}", "width": 20000, "height": 2000, "gt": gt, "dt": dt})
    return recs


def test_resolved_max_dets_covers_the_densest_calibration_image():
    """End to end through the real door: the cap stamped on the bundle has to admit the reference's
    own crowded image, not just its typical one.
    """
    recs = _skewed_calibration_records()
    densest = max(len(r["gt"]) for r in recs)
    assert densest == 300  # the fixture really is skewed, not uniform

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h", calibration_records=recs)
    max_dets = b.params["max_dets"]

    assert max_dets._raw == 411
    assert max_dets._raw > densest
    assert "p99" in max_dets.derived_from

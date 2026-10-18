"""The derived object density is the 0.99 quantile of the occupied counted regions' densities.

The quantile is taken over the counted regions holding at least one object. Each case reads the
density as a cap, the density times one frame's pixels.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.derivations import derive_object_density  # noqa: E402

FRAME = 512.0 * 512.0
"""A sample frame area in pixels, counted over and capped at alike."""


def _frames(counts: list[int]) -> list:
    """Return one region of :data:`FRAME` pixels per count, holding that many objects."""
    from tests._verified_checkpoint_fixtures import objects_over

    return [objects_over([[0.0, 0.0, 1.0, 1.0]] * n, FRAME) for n in counts]


def _cap(counts: list[int]) -> float:
    """Return the detection count the derived density gives one frame of :data:`FRAME` pixels."""
    return derive_object_density(_frames(counts)) * FRAME


def test_the_density_admits_the_crowded_decile_of_a_skewed_count_distribution():
    """The density admits the crowded decile of a skewed count distribution.

    With ninety ordinary scenes at 100 objects and ten crowded ones at 1000, one frame's cap is
    1000, ten times the bulk of the distribution.
    """
    counts = [100] * 90 + [1000] * 10

    assert _cap(counts) == pytest.approx(1000)


def test_the_density_on_a_uniform_count_distribution_is_unchanged_by_the_tail_rule():
    """The density of a uniform count distribution is unchanged by the tail rule.

    Reading the tail does not inflate the density of a reference whose images each carry 100
    objects, and no floor raises one whose images each carry 2.
    """
    assert _cap([100] * 100) == pytest.approx(100)
    assert _cap([2] * 20) == pytest.approx(2)


def test_the_density_reaches_toward_the_densest_image_of_a_skewed_reference():
    """The density reaches toward the densest image of a skewed reference, short of it.

    Over nine sparse images (5 objects each) and one crowded one (300), the density is the 0.99
    quantile of the ten, most of the way toward the crowded image and short of it.
    """
    counts = [5] * 9 + [300]

    assert _cap(counts) == pytest.approx(5 + 0.91 * 295)


def test_an_empty_reference_derives_no_density_and_refuses():
    """An empty reference derives no density and refuses, having no object-bearing region."""
    with pytest.raises(
        ValueError, match="none of the 0 counted regions is object-bearing"
    ):
        derive_object_density([])

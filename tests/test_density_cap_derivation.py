"""The derived object density is the 0.99 quantile of the densities of the counted regions
holding at least one object.

``derive_object_density`` is the one formula behind every frame's detection cap (the density
times the frame's pixels, rounded up). It reads the tail of the reference's density distribution,
not its bulk: on a skewed distribution the two are an order of magnitude apart. A frame denser
than the quantile can still exceed its cap.

The fixtures here are deliberately skewed. On a uniform distribution every quantile is the same
number, so no fixture of that shape can distinguish the two.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.derivations import derive_object_density  # noqa: E402

FRAME = 512.0 * 512.0
"""A sample frame area in pixels, counted over and capped at alike."""


def _frames(counts: list[int]) -> list:
    """One region of :data:`FRAME` pixels per count, holding that many objects."""
    from tests._verified_checkpoint_fixtures import objects_over

    return [objects_over([[0.0, 0.0, 1.0, 1.0]] * n, FRAME) for n in counts]


def _cap(counts: list[int]) -> float:
    """The detection count the derived density gives one frame of :data:`FRAME` pixels."""
    return derive_object_density(_frames(counts)) * FRAME


def test_the_density_admits_the_crowded_decile_of_a_skewed_count_distribution():
    """Ninety ordinary scenes at 100 objects and ten crowded ones at 1000: a frame's cap has to
    clear the crowded ones, which sit ten times above the bulk of the distribution."""
    counts = [100] * 90 + [1000] * 10

    assert _cap(counts) == pytest.approx(1000)


def test_the_density_on_a_uniform_count_distribution_is_unchanged_by_the_tail_rule():
    """The companion obligation: reading the tail must not inflate the density for an ordinary
    reference whose images all carry about the same number of objects; no floor raises a sparse
    one."""
    assert _cap([100] * 100) == pytest.approx(100)
    assert _cap([2] * 20) == pytest.approx(2)


def test_the_density_reaches_toward_the_densest_image_of_a_skewed_reference():
    """Nine sparse images (5 objects each) and one crowded one (300): the density is the 0.99
    quantile of the ten, most of the way toward the crowded image and short of it."""
    counts = [5] * 9 + [300]

    assert _cap(counts) == pytest.approx(5 + 0.91 * 295)


def test_an_empty_reference_derives_no_density_and_refuses():
    with pytest.raises(ValueError, match="none were counted"):
        derive_object_density([])

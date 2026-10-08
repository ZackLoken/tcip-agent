"""The derived detection cap covers the dense tail of the reference it came from.

``derive_max_dets`` is the one formula behind an assessment's unstated ``max_dets``, and it exists
so that crowded scenes are not truncated. It therefore has to read the tail of the reference's
density distribution, not its bulk: on a skewed distribution the two are an order of magnitude
apart, and a cap read off the bulk silently trims detections from exactly the images the cap was
meant to protect. The truncation lands on delivered counts, never on an error.

The fixtures here are deliberately skewed. On a uniform distribution every quantile is the same
number, so no fixture of that shape can distinguish the two.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tcip_mcp.pipelines.derivations import derive_max_dets  # noqa: E402

FRAME = 512.0 * 512.0
"""A sample frame area in pixels, counted over and published at alike."""


def _frames(counts: list[int]) -> list[tuple[int, float]]:
    return [(n, FRAME) for n in counts]


def test_cap_admits_the_crowded_decile_of_a_skewed_count_distribution():
    """Ninety ordinary scenes at 100 objects and ten crowded ones at 1000: the cap has to clear the
    crowded ones, which sit ten times above the bulk of the distribution.
    """
    counts = [100] * 90 + [1000] * 10
    cap = derive_max_dets(_frames(counts), FRAME)

    assert cap == 1500
    assert cap > max(counts)  # no image in this reference is truncated by its own cap


def test_cap_on_a_uniform_count_distribution_is_unchanged_by_the_tail_rule():
    """The companion obligation: reading the tail must not inflate the cap for an ordinary
    reference whose images all carry about the same number of objects; no floor raises a sparse
    one.
    """
    assert derive_max_dets(_frames([100] * 100), FRAME) == 150
    assert derive_max_dets(_frames([2] * 20), FRAME) == 3


def test_the_cap_covers_the_densest_image_of_a_skewed_reference():
    """Nine sparse images (5 objects each) and one crowded one (300): the cap the assessment
    records has to admit the reference's own crowded image, not just its typical one."""
    counts = [5] * 9 + [300]

    cap = derive_max_dets(_frames(counts), FRAME)

    assert cap == 411
    assert cap > max(counts)


def test_an_empty_reference_derives_no_cap_and_refuses():
    with pytest.raises(ValueError, match="holds none"):
        derive_max_dets([], FRAME)

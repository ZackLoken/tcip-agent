"""The mask threshold and ring simplification an instance segmentation assessment derives over its
calibration side's matched detections, checked against shapely's own region IoU, a threshold known
in advance and objects drawn at a known vertex count.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("sklearn")

KNOWN_CUT = 0.6
"""The probability every synthetic object below is its soft mask cut at."""
FAR_CUT = 0.2
"""A cut far from :data:`KNOWN_CUT`, whose rings spill past every object."""


def _soft_disk(radius: int) -> np.ndarray:
    """A soft mask falling from 1 at its center to 0 at ``radius``, in steps of 0.1 a few pixels
    wide, so neighboring cuts differ by a band of pixels.
    """
    size = 2 * radius + 1
    yy, xx = np.mgrid[:size, :size]
    distance = np.hypot(xx - radius, yy - radius)
    return np.clip(np.round(1.0 - distance / radius, 1), 0.0, 1.0)


def _pairs() -> list:
    """Three detections at their frame origins, each matched to the object its own soft mask cut
    at :data:`KNOWN_CUT` is, that object's rings drawn by the platform's contour extractor.
    """
    from tcip_annotation.mask_contours import mask_to_polygon_rings
    from tests._training_values import EPSILON_FRAC

    pairs = []
    for radius, (x, y) in ((40, (10, 20)), (30, (200, 15)), (45, (90, 160))):
        soft = _soft_disk(radius)
        rings = [
            [(px + x, py + y) for px, py in ring]
            for ring in mask_to_polygon_rings(
                soft >= KNOWN_CUT, threshold=None, epsilon_frac=EPSILON_FRAC
            )
        ]
        pairs.append((soft, (x, y), rings))
    return pairs


def _shapely_mean_iou(pairs: list, cut: float) -> float:
    """Each pair's object against its soft mask cut at ``cut``, by shapely's own polygon areas."""
    from shapely.geometry import Polygon

    from tcip_annotation.mask_contours import mask_to_polygon_rings

    ious = []
    for soft, (x, y), rings in pairs:
        (ring,) = rings
        (cut_ring,) = mask_to_polygon_rings(
            soft, threshold=cut, epsilon_frac=0.0
        )
        truth = Polygon(ring)
        predicted = Polygon([(px + x, py + y) for px, py in cut_ring])
        overlap = truth.intersection(predicted).area
        ious.append(overlap / (truth.area + predicted.area - overlap))
    return float(np.mean(ious))


def test_the_derived_mask_threshold_is_the_cut_the_objects_were_drawn_at():
    """Agreement. Objects that are their soft masks cut at a known probability derive that
    probability, its mean rings IoU is shapely's own over the same rings, and a far cut scores
    lower, as shapely scores it.
    """
    from tcip_mcp.pipelines.derivations import (
        derive_mask_binarize,
        mask_rings_iou,
    )

    pairs = _pairs()

    assert derive_mask_binarize(pairs) == pytest.approx(KNOWN_CUT)
    assert mask_rings_iou(pairs, KNOWN_CUT) == pytest.approx(
        _shapely_mean_iou(pairs, KNOWN_CUT)
    )
    assert mask_rings_iou(pairs, FAR_CUT) == pytest.approx(
        _shapely_mean_iou(pairs, FAR_CUT)
    )
    assert mask_rings_iou(pairs, FAR_CUT) < mask_rings_iou(pairs, KNOWN_CUT)


def _empty_pair() -> tuple:
    """A detection whose soft mask holds no positive pixel, as the collection crops one (an empty
    part at origin zero), matched to an object of its own.
    """
    _soft, _origin, rings = _pairs()[0]
    return np.zeros((0, 0)), (0, 0), [[(px + 300, py) for px, py in rings[0]]]


def test_a_matched_mask_holding_no_positive_value_offers_no_cut_and_still_counts():
    """Guard. A matched detection whose soft mask is empty contributes no candidate, so the known
    cut still derives, and its zero overlap counts in every candidate's mean; a reference whose
    every matched mask is empty, or that matches none, refuses naming ``mask_binarize``.
    """
    from tcip_mcp.pipelines.derivations import (
        derive_mask_binarize,
        mask_rings_iou,
    )

    pairs = _pairs()
    with_empty = [*pairs, _empty_pair()]

    assert derive_mask_binarize(with_empty) == pytest.approx(KNOWN_CUT)
    assert mask_rings_iou(with_empty, KNOWN_CUT) == pytest.approx(
        mask_rings_iou(pairs, KNOWN_CUT) * len(pairs) / len(with_empty)
    )
    for unusable in ([], [_empty_pair()]):
        with pytest.raises(ValueError, match="mask_binarize"):
            derive_mask_binarize(unusable)


DISK = (80, 80)
"""The half-axes, in pixels, of the round objects below: an octagon drawn on a radius-80 disk."""
OVAL = (120, 95)
"""The half-axes of the oval object below: a 12-gon drawn on a 120 by 95 pixel ellipse, chosen
because the fraction that leaves the disk eight vertices leaves this ellipse twelve while 0.02
leaves it fewer."""


def _polygon_pair(
    origin: tuple[int, int],
    vertices: int,
    turn: float,
    axes: tuple[int, int] = DISK,
):
    """A detection at ``origin`` whose soft mask is a smooth filled ellipse of half-axes
    ``axes``, matched to the polygon of ``vertices`` corners inscribed in the same ellipse at
    evenly spaced angles turned by ``turn`` radians.
    """
    import cv2

    x, y = origin
    a, b = axes
    soft = np.zeros((2 * b + 1, 2 * a + 1), dtype=np.float32)
    cv2.ellipse(soft, (a, b), (a, b), 0, 0, 360, 1.0, -1)
    angles = turn + 2 * np.pi * np.arange(vertices) / vertices
    ring = [
        (x + a + a * float(np.cos(t)), y + b + b * float(np.sin(t)))
        for t in angles
    ]
    return soft, (x, y), [ring]


def _polygon_pairs() -> list:
    """Three octagon objects on disks, each turned its own way, and one 12-gon on an oval."""
    return [
        *(
            _polygon_pair(origin, 8, turn)
            for origin, turn in (
                ((10, 20), 0.0),
                ((300, 15), 0.3),
                ((90, 300), 1.1),
            )
        ),
        _polygon_pair((500, 40), 12, 0.2, OVAL),
    ]


def test_the_derived_simplification_leaves_each_prediction_its_objects_vertex_count():
    """Agreement. Objects of two shapes drawn at two known vertex counts, each predicted as a
    smooth contour of the same object, derive the one fraction at which
    ``mask_to_polygon_rings`` simplifies every prediction to its own object's count; a pair whose
    count no fraction leaves contributes nothing, and references where none does refuse naming
    ``epsilon_frac``.
    """
    from tcip_annotation.mask_contours import mask_to_polygon_rings
    from tcip_mcp.pipelines.derivations import derive_epsilon_frac

    pairs = _polygon_pairs()
    derived = derive_epsilon_frac(pairs, KNOWN_CUT)

    assert [
        [
            len(ring)
            for ring in mask_to_polygon_rings(
                soft, threshold=KNOWN_CUT, epsilon_frac=derived
            )
        ]
        for soft, _origin, _rings in pairs
    ] == [[len(rings[0])] for _soft, _origin, rings in pairs]
    # A symmetric disk's simplification keeps its vertices in pairs, so no fraction leaves seven.
    unreached = _polygon_pair((700, 700), 7, 0.0)
    assert derive_epsilon_frac([*pairs, unreached], KNOWN_CUT) == derived
    for unusable in ([], [unreached], [_empty_pair()]):
        with pytest.raises(ValueError, match="epsilon_frac"):
            derive_epsilon_frac(unusable, KNOWN_CUT)


def test_an_untiled_instance_segmentation_assessment_records_the_mask_threshold_it_derives(
    tmp_path,
):
    """An instance segmentation assessment stating no mask threshold or ring simplification
    derives both over its calibration side and records each, by its derivation's label, on the
    execution record a published pass restores.
    """
    from tcip_mcp import assessment
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.derivations import (
        EPSILON_FRAC_DERIVATION,
        MASK_BINARIZE_DERIVATION,
    )
    from tcip_mcp.pipelines.execution import prepare
    from tests import _trait_fixtures as fx
    from tests._producer_fixtures import samples_over
    from tests._training_values import COUNT_TEST, collection_pass
    from tests._verified_checkpoint_fixtures import fixture_data_dir
    from tests.test_sliced_inference import _checkpoint

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)
    images = fixture_data_dir(tmp_path, "blobs") / "images"
    (bucket,) = images.iterdir()
    samples = samples_over(bucket, subject="bud")

    p, _measured, _failures = assessment._detection(
        prepare(
            checkpoint, collection_pass(tile=False, masks=True), device="cpu"
        ),
        samples[:1],
        samples[1:],
        fx.COUNT_SPEC,
        "per_image_count",
        source_digests(samples),
        **COUNT_TEST,
    )

    cut = p.execution.mask_cut
    assert cut is not None
    assert p.execution.sources["mask_binarize"] == MASK_BINARIZE_DERIVATION
    assert cut.mask_binarize == pytest.approx(1.0)
    assert p.execution.sources["epsilon_frac"] == EPSILON_FRAC_DERIVATION
    # Each blob's mask cuts to the rectangle its object is, whose corners need no simplification.
    assert cut.epsilon_frac == 0.0


@pytest.mark.parametrize("missing", ["mask_binarize", "epsilon_frac"])
def test_a_stated_mask_cut_missing_either_value_is_refused_naming_it(missing):
    """Guard. A statement's ``mask_cut`` carries both of its values or none is stated: one
    missing either is refused by the type, naming the missing value, and one holding both is
    admitted.
    """
    from pydantic import ValidationError

    from tcip_mcp.pipelines.execution import Stated
    from tests._training_values import MASK_CUT, RATIONALE

    whole = {
        "mask_binarize": MASK_CUT.mask_binarize,
        "epsilon_frac": MASK_CUT.epsilon_frac,
    }
    statement = {"masks": True, "rationale": RATIONALE}

    with pytest.raises(ValidationError, match=missing):
        Stated.model_validate(
            {
                **statement,
                "mask_cut": {k: v for k, v in whole.items() if k != missing},
            }
        )
    assert (
        Stated.model_validate({**statement, "mask_cut": whole}).mask_cut
        == MASK_CUT
    )

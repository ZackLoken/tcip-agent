"""The mask threshold and vertex spacing an instance segmentation assessment derives over its
calibration side's matched detections, checked against shapely's own region IoU, a threshold known
in advance and objects of several sizes drawn at one known spacing.
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
    from tests._training_values import VERTEX_SPACING

    pairs = []
    for radius, (x, y) in ((40, (10, 20)), (30, (200, 15)), (45, (90, 160))):
        soft = _soft_disk(radius)
        rings = [
            [(px + x, py + y) for px, py in ring]
            for ring in mask_to_polygon_rings(
                soft >= KNOWN_CUT,
                threshold=None,
                vertex_spacing=VERTEX_SPACING,
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
            soft, threshold=cut, vertex_spacing=0.0
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


OBJECT_AXES = ((40, 40), (80, 80), (120, 120), (120, 95))
"""The half-axes, in pixels, of the objects below: disks of three radii and one oval."""


def _drawn_pair(
    origin: tuple[int, int],
    axes: tuple[int, int],
    turn: float,
    spacing: int,
):
    """A detection at ``origin`` whose soft mask is a smooth filled ellipse of half-axes
    ``axes``, matched to the polygon inscribed in the same ellipse at evenly spaced angles turned
    by ``turn`` radians, drawn as a person draws an outline at ``spacing`` pixels per vertex: its
    count the nearest integer to the ellipse's circumference over ``spacing``.
    """
    import cv2

    x, y = origin
    a, b = axes
    soft = np.zeros((2 * b + 1, 2 * a + 1), dtype=np.float32)
    cv2.ellipse(soft, (a, b), (a, b), 0, 0, 360, 1.0, -1)
    dense = np.linspace(0, 2 * np.pi, 4096)
    circumference = np.hypot(
        np.diff(a * np.cos(dense)), np.diff(b * np.sin(dense))
    ).sum()
    vertices = round(circumference / spacing)
    angles = turn + 2 * np.pi * np.arange(vertices) / vertices
    ring = [
        (x + a + a * float(np.cos(t)), y + b + b * float(np.sin(t)))
        for t in angles
    ]
    return soft, (x, y), [ring]


def _closed_length(points: np.ndarray) -> float:
    """The length of the closed polyline through ``points``, by numpy alone."""
    return float(
        np.hypot(*(np.roll(points, -1, axis=0) - points).T.astype(float)).sum()
    )


@pytest.mark.parametrize("spacing", [15, 30, 60])
def test_the_derived_vertex_spacing_leaves_each_prediction_its_objects_vertex_count(
    spacing,
):
    """Agreement. Objects of four sizes drawn at one spacing, each predicted as a smooth contour
    of the same object, derive a spacing at which ``mask_to_polygon_rings`` leaves every
    prediction its own object's count within one, whatever its size; the spacing is the median of
    each contour's perimeter over its object's count, numpy's own over the same contours, and the
    raster contour runs longer than the polygon it traces, so it lies at or above the drawn one
    and within 15 percent of it.
    """
    from tcip_annotation.mask_contours import (
        mask_contours,
        mask_to_polygon_rings,
    )
    from tcip_mcp.pipelines.derivations import derive_vertex_spacing

    pairs = [
        _drawn_pair((10 + 300 * i, 20), axes, 0.1 * i, spacing)
        for i, axes in enumerate(OBJECT_AXES)
    ]
    derived = derive_vertex_spacing(pairs, KNOWN_CUT)
    kept = [
        len(ring)
        for soft, _origin, _rings in pairs
        for ring in mask_to_polygon_rings(
            soft, threshold=KNOWN_CUT, vertex_spacing=derived
        )
    ]
    drawn = [len(rings[0]) for _soft, _origin, rings in pairs]

    assert len(kept) == len(drawn)
    assert all(abs(k - d) <= 1 for k, d in zip(kept, drawn)), (kept, drawn)
    assert derived == pytest.approx(
        np.median(
            [
                _closed_length(
                    mask_contours(soft, threshold=KNOWN_CUT)[0].reshape(-1, 2)
                )
                / len(rings[0])
                for soft, _origin, rings in pairs
            ]
        )
    )
    assert spacing <= derived <= 1.15 * spacing


def test_a_ring_at_vertex_spacing_zero_keeps_every_contour_vertex():
    """Guard. At spacing zero ``mask_to_polygon_rings`` returns each contour's own vertices."""
    from tcip_annotation.mask_contours import (
        mask_contours,
        mask_to_polygon_rings,
    )

    soft, _origin, _rings = _drawn_pair((0, 0), (80, 80), 0.0, 30)

    assert mask_to_polygon_rings(
        soft, threshold=KNOWN_CUT, vertex_spacing=0.0
    ) == [
        [(float(px), float(py)) for px, py in contour.reshape(-1, 2)]
        for contour in mask_contours(soft, threshold=KNOWN_CUT)
    ]


def test_a_ring_already_under_its_count_at_any_admitted_spacing_is_kept():
    """Guard. A spacing the block admits, however small, leaves a contour already holding no more
    vertices than its perimeter implies as the contour gives it.
    """
    from tcip_annotation.mask_contours import mask_to_polygon_rings
    from tcip_mcp.pipelines.execution import MaskCut

    cut = MaskCut(mask_binarize=KNOWN_CUT, vertex_spacing=1e-310)
    square = np.zeros((7, 7), dtype=np.uint8)
    square[2:5, 2:5] = 1

    assert mask_to_polygon_rings(
        square, threshold=None, vertex_spacing=cut.vertex_spacing
    ) == mask_to_polygon_rings(square, threshold=None, vertex_spacing=0.0)


def test_a_pair_with_no_contour_or_no_ring_contributes_nothing_to_the_spacing():
    """Guard. A matched object holding no ring and a detection whose mask cuts to nothing leave
    the derived spacing as it was, and a reference where no pair contributes refuses naming
    ``vertex_spacing``.
    """
    from tcip_mcp.pipelines.derivations import derive_vertex_spacing

    pairs = [
        _drawn_pair((10 + 300 * i, 20), axes, 0.1 * i, 30)
        for i, axes in enumerate(OBJECT_AXES)
    ]
    ringless = (pairs[0][0], pairs[0][1], [])
    derived = derive_vertex_spacing(pairs, KNOWN_CUT)

    assert (
        derive_vertex_spacing([*pairs, ringless, _empty_pair()], KNOWN_CUT)
        == derived
    )
    for unusable in ([], [ringless], [_empty_pair()]):
        with pytest.raises(ValueError, match="vertex_spacing"):
            derive_vertex_spacing(unusable, KNOWN_CUT)


def test_an_untiled_instance_segmentation_assessment_records_the_mask_threshold_it_derives(
    tmp_path,
):
    """An instance segmentation assessment stating no mask threshold or vertex spacing derives
    both over its calibration side and records each, by its derivation's label, on the
    execution record a published pass restores.
    """
    from tcip_mcp import assessment
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.derivations import (
        MASK_BINARIZE_DERIVATION,
        VERTEX_SPACING_DERIVATION,
    )
    from tcip_mcp.pipelines.execution import prepare
    from tests import _trait_fixtures as fx
    from tests._producer_fixtures import samples_over
    from tests._training_values import COUNT_TEST, collection_pass
    from tests._verified_checkpoint_fixtures import fixture_data_dir
    from tests.test_sliced_inference import BLOBS, _checkpoint

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
    assert p.execution.sources["vertex_spacing"] == VERTEX_SPACING_DERIVATION
    # Each blob's mask cuts to its rectangle, whose contour runs through pixel centers one pixel
    # inside its box, against its object's four corners.
    assert cut.vertex_spacing == pytest.approx(
        np.median(
            [2 * (x1 - x0 + y1 - y0 - 2) / 4 for x0, y0, x1, y1 in BLOBS]
        )
    )


def test_a_mask_collecting_pass_stating_no_block_and_no_reference_names_the_blocks_fields(
    tmp_path,
):
    """Guard. An instance segmentation pass collecting masks that states no ``mask_cut`` and has
    no reference to derive one over is refused naming each of the block's fields; the same pass
    stating the block runs, each field recorded as stated.
    """
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, prepare
    from tests._training_values import MASK_CUT, collection_pass
    from tests.test_sliced_inference import _checkpoint

    _path, checkpoint = _checkpoint(tmp_path, with_masks=True)

    with pytest.raises(ExecutionRefusedError) as refused:
        prepare(
            checkpoint, collection_pass(tile=False, masks=True), device="cpu"
        ).runnable()
    assert "mask_binarize" in str(refused.value)
    assert "vertex_spacing" in str(refused.value)
    p = prepare(
        checkpoint,
        collection_pass(tile=False, masks=True, mask_cut=MASK_CUT),
        device="cpu",
    ).runnable()
    assert p.execution.mask_cut == MASK_CUT
    assert {
        p.execution.sources[k] for k in ("mask_binarize", "vertex_spacing")
    } == {"explicit"}


@pytest.mark.parametrize("missing", ["mask_binarize", "vertex_spacing"])
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
        "vertex_spacing": MASK_CUT.vertex_spacing,
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

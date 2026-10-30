"""Data and model derivations: channels, classes and anchors from the data in hand."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.pipelines.derivations import (
    derive_block_scale_px,
    derive_cross_tile_nms,
    derive_iou_match_threshold,
    derive_localization_tolerance_frac,
    derive_sliver_frac,
    gt_aspect_ratios,
    num_classes_from_distribution,
    probe_channels,
)
from tests import _trait_fixtures as fx
from tests._producer_fixtures import write_image

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from tifffile import TiffTags

    from tcip_mcp.pipelines.derivations import Region
    from tcip_mcp.pipelines.raster_source import _TiffHeader


def test_probe_channels_from_raster(tmp_path: Path) -> None:
    gray = tmp_path / "gray.png"
    Image.new("L", (8, 8)).save(gray)
    assert probe_channels(write_image(tmp_path / "rgb.png", (8, 8))) == 3
    assert probe_channels(gray) == 1
    np.save(tmp_path / "ms.npy", np.zeros((8, 8, 5), dtype=np.float32))
    assert probe_channels(tmp_path / "ms.npy") == 5


def test_num_classes_from_distribution() -> None:
    assert (
        num_classes_from_distribution({0: 5, 1: 3}) == 2
    )  # ids 0..1 -> 2 classes
    assert num_classes_from_distribution({0: 10}) == 1
    assert num_classes_from_distribution({}) == 0


def test_gt_aspect_ratios_covers_open() -> None:
    # tall boxes (h/w ~ 4) -> the derived ratio set must include a tall ratio the default
    # (0.5,1,2) lacks
    boxes = [(10.0, 40.0)] * 20
    ratios = gt_aspect_ratios(boxes)
    assert max(ratios) >= 3.0


def _regions(
    boxes_per_image: Sequence[Sequence[tuple[float, float, float, float]]],
) -> list[Region]:
    """Return one region per image of the xywh ``boxes_per_image``, each of area 200 by 200 px."""
    from tests._verified_checkpoint_fixtures import objects_over

    return [
        objects_over([[x, y, x + w, y + h] for x, y, w, h in boxes], 200 * 200)
        for boxes in boxes_per_image
    ]


def test_derive_cross_tile_nms_dense_cluster_exceeds_sparse() -> None:
    # Dense boxes (20px, offset 4px -> neighbor IoU ~0.667) push the threshold up so genuinely-
    # overlapping dense objects aren't merged; sparse boxes (offset 16px -> IoU ~0.111) sit lower.
    dense = [[(0, 0, 20, 20), (4, 0, 20, 20), (8, 0, 20, 20), (12, 0, 20, 20)]]
    sparse = [[(0, 0, 20, 20), (16, 0, 20, 20), (32, 0, 20, 20)]]
    t_dense = derive_cross_tile_nms(_regions(dense))
    t_sparse = derive_cross_tile_nms(_regions(sparse))
    assert t_dense is not None and t_sparse is not None
    assert t_dense > t_sparse
    # p99 of the neighbor-IoU tail + margin, at either density.
    assert t_dense == pytest.approx(0.6667 + 0.05, abs=1e-2)
    assert t_sparse == pytest.approx(0.1111 + 0.05, abs=1e-3)


def test_derive_cross_tile_nms_no_overlap_returns_none() -> None:
    # No genuine neighbor overlap anywhere -> underivable -> the caller states a threshold.
    boxes = [[(0, 0, 20, 20), (100, 100, 20, 20)], [(0, 0, 20, 20)]]
    assert derive_cross_tile_nms(_regions(boxes)) is None
    assert derive_cross_tile_nms([]) is None


def test_derive_cross_tile_nms_answers_near_duplicates_unbounded_and_refuses_full_overlap() -> (
    None
):
    # Near-duplicate neighbors (IoU 19/21) answer their own tail plus the margin; neighbors that
    # overlap fully leave the margin nowhere below 1 to land.
    near = [[(0, 0, 20, 20), (1, 0, 20, 20)]]
    assert derive_cross_tile_nms(_regions(near)) == pytest.approx(
        19 / 21 + 0.05
    )
    with pytest.raises(ValueError, match="cross_tile_nms"):
        derive_cross_tile_nms(_regions([[(0, 0, 20, 20), (0, 0, 20, 20)]]))


def test_derive_cross_tile_nms_refuses_where_the_margin_exhausts_the_interval() -> (
    None
):
    # Two 100 px boxes 2 px apart overlap at IoU 98/102, past 0.95: the margin carries the
    # threshold past 1 and refuses naming it, while 3 px apart (IoU 97/103) answers below 1.
    with pytest.raises(ValueError, match="margin 0.05"):
        derive_cross_tile_nms(_regions([[(0, 0, 100, 100), (2, 0, 100, 100)]]))
    assert derive_cross_tile_nms(
        _regions([[(0, 0, 100, 100), (3, 0, 100, 100)]])
    ) == (pytest.approx(97 / 103 + 0.05))


def test_derive_cross_tile_nms_reads_the_neighbor_iou_tail() -> None:
    # A box nested in another: IoU 0.25, so the threshold sits a margin above it.
    nested = [[(0, 0, 20, 20), (5, 5, 10, 10)]]
    assert derive_cross_tile_nms(_regions(nested)) == pytest.approx(
        0.25 + 0.05
    )


def test_derive_localization_tolerance_frac_tight_spacing_stays_tighter_than_loose() -> (
    None
):
    # Same box size (20x20, char size 20) both times; only neighbor spacing differs. Tight spacing
    # (10px between centers) must derive a smaller fraction than loose spacing (100px): the
    # tolerance has to stay well inside how close real neighbors actually get, or two distinct
    # nearby objects start double-matching to one detection.
    tight = [
        [
            (0, 0, 20, 20),
            (10, 0, 20, 20),
            (20, 0, 20, 20),
            (30, 0, 20, 20),
            (40, 0, 20, 20),
        ]
    ]
    loose = [[(0, 0, 20, 20), (100, 0, 20, 20), (200, 0, 20, 20)]]
    t_tight = derive_localization_tolerance_frac(tight)
    t_loose = derive_localization_tolerance_frac(loose)
    assert t_tight is not None and t_loose is not None
    assert t_tight < t_loose
    # half the p10 nn-dist over char_size (20): 10 -> 0.25, 100 -> 2.5.
    assert t_tight == pytest.approx(0.25)
    assert t_loose == pytest.approx(2.5)


def test_derive_localization_tolerance_frac_refuses_a_zero_spacing_percentile_only() -> (
    None
):
    """The localization tolerance refuses a zero spacing percentile and admits one stray pair.

    Boxes all sharing one center refuse, and one coincident pair among thirty spaced boxes
    leaves the percentile positive.
    """
    stacked = [[(0, 0, 20, 20), (0, 0, 20, 20), (0, 0, 20, 20)]]
    with pytest.raises(ValueError, match="spacing is 0 px"):
        derive_localization_tolerance_frac(stacked)
    spaced = [(100.0 * i, 0, 20, 20) for i in range(1, 29)]
    assert (
        derive_localization_tolerance_frac([spaced + [(0, 0, 20, 20)] * 2]) > 0
    )


def _broadcast_nearest_other(centers: np.ndarray) -> np.ndarray:
    """Return each center's distance to its nearest other center over the all-pairs matrix."""
    n = len(centers)
    dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    return dist[~np.eye(n, dtype=bool)].reshape(n, n - 1).min(axis=1)


def test_nearest_neighbor_center_distances_agree_with_the_all_pairs_matrix() -> (
    None
):
    """Each box's nearest-neighbor center distance equals the all-pairs matrix's row minimum.

    Twenty seeded boxes and a set holding one coincident pair beside a distinct box each agree
    with the matrix built here by broadcasting, the pair answering zero for both; two boxes answer
    their one spacing each, and one box answers nothing.
    """
    from tcip_annotation.matching import box_centers
    from tcip_mcp.pipelines.derivations import _neighbor_min_center_distances

    rng = np.random.default_rng(0)
    corner = rng.uniform(0.0, 1000.0, size=(20, 2))
    size = rng.uniform(5.0, 50.0, size=(20, 2))
    scattered = np.hstack([corner, corner + size])
    paired = np.array(
        [
            [0.0, 0.0, 20.0, 20.0],
            [5.0, 5.0, 15.0, 15.0],
            [30.0, 40.0, 50.0, 60.0],
        ]
    )
    for boxes in (scattered, paired):
        np.testing.assert_allclose(
            _neighbor_min_center_distances(boxes),
            _broadcast_nearest_other(box_centers(boxes)),
            rtol=0.0,
            atol=1e-9,
        )
    assert _neighbor_min_center_distances(paired)[:2] == [0.0, 0.0]
    two = np.array([[0.0, 0.0, 10.0, 10.0], [30.0, 40.0, 40.0, 50.0]])
    assert _neighbor_min_center_distances(two) == pytest.approx([50.0, 50.0])
    assert _neighbor_min_center_distances(two[:1]) == []


def test_derive_localization_tolerance_frac_no_same_image_neighbor_returns_none() -> (
    None
):
    # Every image holds at most one box of this class -> no neighbor spacing to measure from.
    assert (
        derive_localization_tolerance_frac(
            [[(0, 0, 20, 20)], [(0, 0, 10, 10)]]
        )
        is None
    )
    assert derive_localization_tolerance_frac([]) is None


def test_derive_sliver_frac_wide_spread_lower_than_tight_spread() -> None:
    # A class with wide natural size variation (e.g. across a growth/bloom stage) needs a lower
    # cutoff, or it discards real small-but-complete instances as tile-seam slivers; a tightly-sized
    # class can use a higher one without losing anything real.
    tight = list(np.linspace(38, 42, 20))
    wide = list(np.linspace(10, 90, 20))
    f_tight = derive_sliver_frac(tight)
    f_wide = derive_sliver_frac(wide)
    assert f_tight is not None and f_wide is not None
    assert f_wide < f_tight
    assert f_wide == pytest.approx(0.36, abs=1e-2)
    assert f_tight == pytest.approx(0.96, abs=1e-2)


def test_derive_sliver_frac_can_exceed_one_when_small_boxes_drag_the_mean_down() -> (
    None
):
    # One tiny box among ten large ones: the low percentile sits near 100 and the mean near 91.
    sizes = [1.0] + [100.0] * 10
    assert derive_sliver_frac(sizes) == pytest.approx(
        float(np.percentile(sizes, 10)) / float(np.mean(sizes))
    )
    assert derive_sliver_frac(sizes) > 1


def test_derive_sliver_frac_no_boxes_returns_none() -> None:
    assert derive_sliver_frac([]) is None
    assert derive_sliver_frac([0.0, 0.0]) is None


def test_derive_sliver_frac_too_few_samples_returns_none() -> None:
    # A single box's ratio to itself is trivially ~1.0 regardless of the class's real variation:
    # not a spread, just noise. Below five sizes it must refuse rather than derive from it.
    assert derive_sliver_frac([25.6]) is None
    assert (
        derive_sliver_frac([10.0, 20.0, 30.0, 40.0]) is None
    )  # 4 sizes, fewer than five


def _label_percentile(label: str) -> float:
    """Return the percentile a derivation label names, as its ``p<N>``."""
    match = re.search(r"\bp(\d+)\b", label)
    assert match, label
    return float(match.group(1))


def test_the_localization_tolerance_is_the_method_its_label_names() -> None:
    """The localization tolerance is the method its label names.

    The criterion's tolerance and label, as an assessment resolves them, agree: half the labeled
    percentile of a spread of nearest-neighbor spacings over the characteristic size.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.training.evaluation import localization_frac

    spacings = [5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0]
    frac, label = localization_frac(
        fx.COUNT_SPEC,
        [[(0.0, 0.0, 20.0, 20.0), (d, 0.0, 20.0, 20.0)] for d in spacings],
    )
    assert label.startswith("half the p"), label
    nearest = [d for d in spacings for _ in range(2)]
    assert frac == pytest.approx(
        0.5 * np.percentile(nearest, _label_percentile(label)) / 20
    )


def test_the_sliver_cutoff_is_the_method_its_label_names() -> None:
    """The sliver cutoff is the method its label names.

    The registered label's percentile over the mean is the cutoff derived from a spread of
    characteristic sizes.
    """
    from tcip_mcp.pipelines.derivations import (
        _STATIC_DERIVATION_IMPLEMENTATIONS,
    )

    label = next(
        name
        for name, impl in _STATIC_DERIVATION_IMPLEMENTATIONS.items()
        if impl == "tcip_mcp.pipelines.derivations.derive_sliver_frac"
    )
    sizes = [3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]
    assert derive_sliver_frac(sizes) == pytest.approx(
        np.percentile(sizes, _label_percentile(label)) / np.mean(sizes)
    )


JITTER = {"jitter_px": fx.IOU_JITTER_PX, "margin": fx.IOU_MARGIN}
"""The sample trait-authored jitter and margin the IoU derivation takes."""


def test_derive_iou_match_threshold_exact_value() -> None:
    # char size 60 -> modeled IoU (60-12)/(60+12) = 2/3 -> threshold 2/3 - margin 0.05.
    boxes = [[(0, 0, 60, 60)]]
    assert derive_iou_match_threshold(boxes, **JITTER) == pytest.approx(
        2 / 3 - 0.05
    )


def test_derive_iou_match_threshold_scales_with_object_size() -> None:
    # Larger characteristic size -> higher achievable IoU under the same jitter -> higher threshold.
    small = derive_iou_match_threshold([[(0, 0, 60, 60)]], **JITTER)
    large = derive_iou_match_threshold([[(0, 0, 300, 300)]], **JITTER)
    assert small is not None and large is not None
    assert large > small


@pytest.mark.parametrize(
    "size, jitter_px, margin",
    [(1000, 12.0, 0.05), (60, 0.0, 0.0), (60, 50.0, 0.0)],
)
def test_derive_iou_match_threshold_answers_the_modeled_value_at_either_end(
    size: int, jitter_px: float, margin: float
) -> None:
    """The IoU-match threshold answers the modeled value at either end, with no cutoff over it.

    A large box, an exact repeat and a near-disjoint one each answer the modeled IoU less the
    margin.
    """
    expected = (size - jitter_px) / (size + jitter_px) - margin
    assert derive_iou_match_threshold(
        [[(0, 0, size, size)]], jitter_px=jitter_px, margin=margin
    ) == pytest.approx(expected)


@pytest.mark.parametrize("jitter_px", [60.0, 90.0])
def test_derive_iou_match_threshold_refuses_a_jitter_at_or_past_the_box_size(
    jitter_px: float,
) -> None:
    with pytest.raises(ValueError, match="iou_jitter_px"):
        derive_iou_match_threshold(
            [[(0, 0, 60, 60)]], jitter_px=jitter_px, margin=0.0
        )


@pytest.mark.parametrize("margin", [0.5, 0.8])
def test_derive_iou_match_threshold_refuses_a_margin_that_swallows_the_modeled_iou(
    margin: float,
) -> None:
    # Boxes of 60 px 20 px apart model IoU 0.5: a margin of 0.5 or more leaves no threshold.
    with pytest.raises(ValueError, match="iou_margin"):
        derive_iou_match_threshold(
            [[(0, 0, 60, 60)]], jitter_px=20.0, margin=margin
        )


def test_derive_iou_match_threshold_no_boxes_returns_none() -> None:
    assert derive_iou_match_threshold([], **JITTER) is None
    assert derive_iou_match_threshold([[], []], **JITTER) is None
    assert derive_iou_match_threshold([[(0, 0, 0, 0)]], **JITTER) is None


def test_derive_block_scale_px_gt_object_spacing_floored_at_tile_size() -> (
    None
):
    # Objects 200px apart, tile_size 50: the derived scale (median NN spacing) is 200, well above
    # the floor, so the floor never engages.
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    px, source = derive_block_scale_px(
        tile_size=50, objects=_regions([boxes])[0]
    )
    assert px == 200
    assert "GT object-spacing" in source


def test_derive_block_scale_px_floors_at_tile_size_when_spacing_is_smaller() -> (
    None
):
    boxes = [(x, 0, 5, 5) for x in range(0, 100, 10)]  # 10px spacing
    px, source = derive_block_scale_px(
        tile_size=64, objects=_regions([boxes])[0]
    )
    assert (
        px == 64
    )  # the tile_size floor wins over the smaller measured spacing
    assert "floored at tile_size" in source


def test_derive_block_scale_px_no_data_refuses_named() -> None:
    with pytest.raises(ValueError, match="no block scale is derivable"):
        derive_block_scale_px(tile_size=64, objects=_regions([[]])[0])


def test_derive_block_scale_px_insufficient_plant_registry_refuses_named() -> (
    None
):
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    one_plant = [PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0)]
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    with pytest.raises(ValueError, match="plant grid pitch is underivable"):
        derive_block_scale_px(
            tile_size=50,
            objects=_regions([boxes])[0],
            plants=one_plant,
            raster_path=None,
        )


def test_derive_block_scale_px_plant_pitch_via_projected_geotransform(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    # Two plants about 100m apart (99.80m on the WGS84 geodesic), on a
    # real UTM 15N GeoTIFF at the fixture's default 0.5 m/px.
    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),  # ~100m north
    ]
    raster_path = tmp_path / "mosaic.tif"
    write_geotiff(raster_path)
    boxes = [
        (x, 0, 20, 20) for x in range(0, 40, 20)
    ]  # sparse GT: the plant path must win
    px, source = derive_block_scale_px(
        tile_size=16,
        objects=_regions([boxes])[0],
        plants=plants,
        raster_path=str(raster_path),
    )
    assert "plant grid pitch" in source
    assert "EPSG:32615" in source
    assert px == pytest.approx(200, rel=0.05)  # ~100m / 0.5 m-per-px = ~200px


def test_derive_block_scale_px_reads_the_rasters_header_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The block-scale derivation reads the raster's header once.

    The pixel size the plant pitch converts through comes off one header read.
    """
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]
    raster_path = tmp_path / "mosaic.tif"
    write_geotiff(raster_path)
    reads: list = []
    real = raster_source.tiff_header

    def counting(tif: Path) -> _TiffHeader:
        reads.append(tif)
        return real(tif)

    monkeypatch.setattr(raster_source, "tiff_header", counting)
    _px, source = derive_block_scale_px(
        tile_size=16,
        objects=_regions([[(0, 0, 20, 20)]])[0],
        plants=plants,
        raster_path=str(raster_path),
    )
    assert "plant grid pitch" in source
    assert len(reads) == 1


def test_derive_block_scale_px_converts_a_foot_unit_raster_through_its_crs(
    tmp_path: Path,
) -> None:
    """A foot-unit raster converts the plant pitch through its CRS's own unit factor.

    A raster in US survey feet (EPSG 2264) converts the plant-pitch meters through 1200/3937 m
    per foot.
    """
    from pyproj import Geod

    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord(
            "p1", "a1", 0, 0, 0, 45.000898, -93.0
        ),  # ~99.80m north (the WGS84 geodesic)
    ]
    raster_path = tmp_path / "mosaic.tif"
    write_geotiff(
        raster_path, pixel_scale=(1.0, 1.0, 0.0), projected_epsg=2264
    )
    boxes = [
        (x, 0, 20, 20) for x in range(0, 40, 20)
    ]  # sparse GT: the plant path must win
    px, source = derive_block_scale_px(
        tile_size=16,
        objects=_regions([boxes])[0],
        plants=plants,
        raster_path=str(raster_path),
    )
    assert "plant grid pitch" in source
    assert "EPSG:2264" in source
    _fwd, _back, pitch_m = Geod(ellps="WGS84").inv(
        -93.0, 45.0, -93.0, 45.000898
    )
    # US survey feet (EPSG 2264): 1200/3937 m per foot; a meter-blind read would give 100.
    assert px == round(pitch_m / (1200 / 3937))


def test_derive_block_scale_px_photographic_raster_path_refuses_named(
    tmp_path: Path,
) -> None:
    """A photographic image given as ``raster_path`` is refused by name.

    A ``raster_path`` that is a photographic plot image is refused naming its path.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]
    photo_path = write_image(tmp_path / "plot.jpg", (8, 8))
    boxes = [(x, 0, 20, 20) for x in range(0, 40, 20)]
    with pytest.raises(ValueError) as exc_info:
        derive_block_scale_px(
            tile_size=16,
            objects=_regions([boxes])[0],
            plants=plants,
            raster_path=str(photo_path),
        )
    assert str(photo_path) in str(exc_info.value)


def test_derive_block_scale_px_anisotropic_raster_falls_back_to_gt_spacing(
    tmp_path: Path,
) -> None:
    """An anisotropic raster falls back to GT-object spacing for the block scale.

    A raster whose axes carry differing pixel scales gives the GT-object-spacing scale, 200 px.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]
    raster_path = tmp_path / "mosaic.tif"
    write_geotiff(raster_path, pixel_scale=(0.5, 0.6, 0.0))
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    px, source = derive_block_scale_px(
        tile_size=50,
        objects=_regions([boxes])[0],
        plants=plants,
        raster_path=str(raster_path),
    )
    assert "GT object-spacing" in source  # fell back, not a refusal
    assert px == 200


def test_derive_block_scale_px_unprojected_raster_falls_back_to_gt_spacing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tcip_mcp.pipelines.postprocessing import orthomosaic_mapping
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]

    def _boom(tags: TiffTags, path: Path) -> NoReturn:
        raise orthomosaic_mapping.GeoreferencingError("no geokeys")

    monkeypatch.setattr(orthomosaic_mapping, "geotransform_of", _boom)
    raster_path = tmp_path / "mosaic.tif"
    # A georeferenced raster, so the plant path would win without the stub above.
    write_geotiff(raster_path)
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    px, source = derive_block_scale_px(
        tile_size=50,
        objects=_regions([boxes])[0],
        plants=plants,
        raster_path=str(raster_path),
    )
    assert "GT object-spacing" in source  # fell back, not a refusal
    assert px == 200


def test_derive_block_scale_px_truncated_raster_refuses_named(
    tmp_path: Path,
) -> None:
    """A truncated raster is refused by name.

    A raster_path with a raster suffix that cannot be opened is refused as not opening as a
    raster.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]
    raster_path = tmp_path / "mosaic.tif"
    raster_path.write_bytes(b"not a real tiff")
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    with pytest.raises(ValueError, match="could not be opened as a raster"):
        derive_block_scale_px(
            tile_size=50,
            objects=_regions([boxes])[0],
            plants=plants,
            raster_path=str(raster_path),
        )


def test_derive_block_scale_px_npy_raster_refuses_named_for_no_georeference(
    tmp_path: Path,
) -> None:
    """An .npy raster is refused by name for carrying no georeference.

    An .npy array container is a raster by suffix but carries no georeferencing tags, and the
    refusal says it carries none.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]
    raster_path = tmp_path / "mosaic.npy"
    np.save(str(raster_path), np.zeros((20, 20, 3), dtype=np.uint8))
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    with pytest.raises(ValueError, match="carries no georeferencing tags"):
        derive_block_scale_px(
            tile_size=50,
            objects=_regions([boxes])[0],
            plants=plants,
            raster_path=str(raster_path),
        )


def test_write_subject_registry(tmp_path: Path) -> None:
    import json

    from tcip_mcp import subject_registry
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    out = tmp_path / "subjects.json"
    # The expert authors the nested registry: two ordered values of a categorical attribute.
    res = write_subject_registry(
        tmp_path,
        str(tmp_path),
        subjects={
            "bud": {
                "description": "a currant bud",
                "attributes": {
                    "opening": {
                        "type": "categorical",
                        "values": ["closed", "open"],
                    }
                },
            }
        },
        actor=None,
    )
    assert "error" not in res
    assert res["subjects"] == ["bud"]
    assert res["subjects_path"] == str(out)
    # Declared order is the id order (a value's id is its position): 0=closed, 1=open, and the
    # on-disk nested shape carries the same value order.
    reg = subject_registry.read_registry(tmp_path)
    assert reg.subjects[0].attribute("opening").values == ("closed", "open")  # type: ignore[union-attr]
    assert json.loads(out.read_text())["bud"]["attributes"]["opening"][
        "values"
    ] == ["closed", "open"]


def test_write_subject_registry_no_labels(tmp_path: Path) -> None:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    # An empty registry mapping is not authorable: the tool refuses rather than writing nothing.
    res = write_subject_registry(
        tmp_path, str(tmp_path), subjects={}, actor=None
    )
    assert "error" in res

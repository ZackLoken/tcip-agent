"""Tier-A data/model derivations (channels/classes/anchors from the data in hand)."""

from __future__ import annotations

import functools

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


def test_probe_channels_from_raster(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "rgb.png")
    Image.new("L", (8, 8)).save(tmp_path / "gray.png")
    assert probe_channels(tmp_path / "rgb.png") == 3
    assert probe_channels(tmp_path / "gray.png") == 1
    np.save(tmp_path / "ms.npy", np.zeros((8, 8, 5), dtype=np.float32))
    assert probe_channels(tmp_path / "ms.npy") == 5


def test_num_classes_from_distribution():
    assert num_classes_from_distribution({0: 5, 1: 3}) == 2  # ids 0..1 -> 2 classes
    assert num_classes_from_distribution({0: 10}) == 1
    assert num_classes_from_distribution({}) == 0


def test_gt_aspect_ratios_covers_open():
    # tall boxes (h/w ~ 4) -> the derived ratio set must include a tall ratio the default
    # (0.5,1,2) lacks
    boxes = [(10.0, 40.0)] * 20
    ratios = gt_aspect_ratios(boxes)
    assert max(ratios) >= 3.0


def _regions(boxes_per_image):
    """One region per image of the xywh ``boxes_per_image``, over a 200 px square frame."""
    from tests._verified_checkpoint_fixtures import objects_over

    return [objects_over([[x, y, x + w, y + h] for x, y, w, h in boxes], 200 * 200)
            for boxes in boxes_per_image]


def test_derive_cross_tile_nms_dense_cluster_exceeds_sparse():
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


def test_derive_cross_tile_nms_no_overlap_returns_none():
    # No genuine neighbor overlap anywhere -> underivable -> the caller states a threshold.
    boxes = [[(0, 0, 20, 20), (100, 100, 20, 20)], [(0, 0, 20, 20)]]
    assert derive_cross_tile_nms(_regions(boxes)) is None
    assert derive_cross_tile_nms([]) is None


def test_derive_cross_tile_nms_answers_near_duplicates_unbounded_and_refuses_full_overlap():
    # Near-duplicate neighbors (IoU 19/21) answer their own tail plus the margin; neighbors that
    # overlap fully leave the margin nowhere below 1 to land.
    near = [[(0, 0, 20, 20), (1, 0, 20, 20)]]
    assert derive_cross_tile_nms(_regions(near)) == pytest.approx(19 / 21 + 0.05)
    with pytest.raises(ValueError, match="cross_tile_nms"):
        derive_cross_tile_nms(_regions([[(0, 0, 20, 20), (0, 0, 20, 20)]]))


def test_derive_cross_tile_nms_refuses_where_the_margin_exhausts_the_interval():
    # Two 100 px boxes 2 px apart overlap at IoU 98/102: the 0.05 margin carries the threshold
    # past 1 and refuses naming the margin, while a 0.01 margin answers below it.
    boxes = [[(0, 0, 100, 100), (2, 0, 100, 100)]]
    with pytest.raises(ValueError, match="margin 0.05"):
        derive_cross_tile_nms(_regions(boxes))
    assert derive_cross_tile_nms(_regions(boxes), margin=0.01) == pytest.approx(98 / 102 + 0.01)


def test_derive_cross_tile_nms_reads_the_neighbor_iou_tail():
    # A box nested in another: IoU 0.25, so the threshold sits a margin above it.
    nested = [[(0, 0, 20, 20), (5, 5, 10, 10)]]
    assert derive_cross_tile_nms(_regions(nested)) == pytest.approx(0.25 + 0.05)


def test_derive_localization_tolerance_frac_tight_spacing_stays_tighter_than_loose():
    # Same box size (20x20, char size 20) both times; only neighbor spacing differs. Tight spacing
    # (10px between centers) must derive a smaller fraction than loose spacing (100px): the
    # tolerance has to stay well inside how close real neighbors actually get, or two distinct
    # nearby objects start double-matching to one detection.
    tight = [[(0, 0, 20, 20), (10, 0, 20, 20), (20, 0, 20, 20), (30, 0, 20, 20), (40, 0, 20, 20)]]
    loose = [[(0, 0, 20, 20), (100, 0, 20, 20), (200, 0, 20, 20)]]
    t_tight = derive_localization_tolerance_frac(tight)
    t_loose = derive_localization_tolerance_frac(loose)
    assert t_tight is not None and t_loose is not None
    assert t_tight < t_loose
    # p10 nn-dist * margin_frac (0.5) / char_size (20): 10 -> 0.25, 100 -> 2.5.
    assert t_tight == pytest.approx(0.25)
    assert t_loose == pytest.approx(2.5)


def test_derive_localization_tolerance_frac_refuses_a_zero_spacing_percentile_only():
    """The refusal is the selected spacing percentile at zero: boxes all sharing one center
    refuse, and one coincident pair among thirty spaced boxes leaves the percentile positive."""
    stacked = [[(0, 0, 20, 20), (0, 0, 20, 20), (0, 0, 20, 20)]]
    with pytest.raises(ValueError, match="spacing is 0 px"):
        derive_localization_tolerance_frac(stacked)
    spaced = [(100.0 * i, 0, 20, 20) for i in range(1, 29)]
    assert derive_localization_tolerance_frac([spaced + [(0, 0, 20, 20)] * 2]) > 0


def test_derive_localization_tolerance_frac_no_same_image_neighbor_returns_none():
    # Every image holds at most one box of this class -> no neighbor spacing to measure from.
    assert derive_localization_tolerance_frac([[(0, 0, 20, 20)], [(0, 0, 10, 10)]]) is None
    assert derive_localization_tolerance_frac([]) is None


def test_derive_sliver_frac_wide_spread_lower_than_tight_spread():
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


def test_derive_sliver_frac_can_exceed_one_when_small_boxes_drag_the_mean_down():
    # One tiny box among ten large ones: the low percentile sits near 100 and the mean near 91.
    sizes = [1.0] + [100.0] * 10
    assert derive_sliver_frac(sizes) == pytest.approx(
        float(np.percentile(sizes, 10)) / float(np.mean(sizes)))
    assert derive_sliver_frac(sizes) > 1


def test_derive_sliver_frac_no_boxes_returns_none():
    assert derive_sliver_frac([]) is None
    assert derive_sliver_frac([0.0, 0.0]) is None


def test_derive_sliver_frac_too_few_samples_returns_none():
    # A single box's ratio to itself is trivially ~1.0 regardless of the class's real variation:
    # not a spread, just noise. Below min_samples must refuse rather than derive from it.
    assert derive_sliver_frac([25.6]) is None
    assert derive_sliver_frac([10.0, 20.0, 30.0, 40.0]) is None  # 4 < default min_samples=5


JITTER = {"jitter_px": fx.IOU_JITTER_PX, "margin": fx.IOU_MARGIN}
"""The sample trait-authored jitter and margin the IoU derivation takes."""


def test_derive_iou_match_threshold_exact_value():
    # char size 60 -> modeled IoU (60-12)/(60+12) = 2/3 -> threshold 2/3 - margin 0.05.
    boxes = [[(0, 0, 60, 60)]]
    assert derive_iou_match_threshold(boxes, **JITTER) == pytest.approx(2 / 3 - 0.05)


def test_derive_iou_match_threshold_scales_with_object_size():
    # Larger characteristic size -> higher achievable IoU under the same jitter -> higher threshold.
    small = derive_iou_match_threshold([[(0, 0, 60, 60)]], **JITTER)
    large = derive_iou_match_threshold([[(0, 0, 300, 300)]], **JITTER)
    assert small is not None and large is not None
    assert large > small


@pytest.mark.parametrize("size, jitter_px, margin", [(1000, 12.0, 0.05), (60, 0.0, 0.0),
                                                     (60, 50.0, 0.0)])
def test_derive_iou_match_threshold_answers_the_modeled_value_at_either_end(size, jitter_px,
                                                                           margin):
    """No cutoff stands over the model: a large box, an exact repeat and a near-disjoint one each
    answer the modeled IoU less the margin."""
    expected = (size - jitter_px) / (size + jitter_px) - margin
    assert derive_iou_match_threshold([[(0, 0, size, size)]], jitter_px=jitter_px,
                                      margin=margin) == pytest.approx(expected)


@pytest.mark.parametrize("jitter_px", [60.0, 90.0])
def test_derive_iou_match_threshold_refuses_a_jitter_at_or_past_the_box_size(jitter_px):
    with pytest.raises(ValueError, match="iou_jitter_px"):
        derive_iou_match_threshold([[(0, 0, 60, 60)]], jitter_px=jitter_px, margin=0.0)


@pytest.mark.parametrize("margin", [0.5, 0.8])
def test_derive_iou_match_threshold_refuses_a_margin_that_swallows_the_modeled_iou(margin):
    # Boxes of 60 px 20 px apart model IoU 0.5: a margin of 0.5 or more leaves no threshold.
    with pytest.raises(ValueError, match="iou_margin"):
        derive_iou_match_threshold([[(0, 0, 60, 60)]], jitter_px=20.0, margin=margin)


def test_derive_iou_match_threshold_no_boxes_returns_none():
    assert derive_iou_match_threshold([], **JITTER) is None
    assert derive_iou_match_threshold([[], []], **JITTER) is None
    assert derive_iou_match_threshold([[(0, 0, 0, 0)]], **JITTER) is None


@pytest.mark.parametrize("fn", [
    derive_localization_tolerance_frac,
    functools.partial(derive_iou_match_threshold, **JITTER),
])
def test_derive_box_functions_raise_valueerror_on_malformed_gt_boxes(fn):
    # A bare Python operation on malformed input raises whatever exception type it happens to hit
    # (TypeError on None, ValueError on an unpack mismatch, IndexError from numpy); every derive_*
    # function in this module raises one consistent ValueError instead, with a message naming what
    # was actually wrong.
    with pytest.raises(ValueError, match="gt_boxes_per_image"):
        fn(None)
    with pytest.raises(ValueError, match="4-element"):
        fn([[(1, 2, 3)]])  # a box with only 3 coordinates
    with pytest.raises(ValueError, match="non-numeric"):
        fn([[("a", "b", "c", "d")]])
    with pytest.raises(ValueError, match="gt_boxes_per_image"):
        fn("not a sequence of boxes")


def test_derive_sliver_frac_raises_valueerror_on_malformed_char_sizes():
    with pytest.raises(ValueError, match="char_sizes"):
        derive_sliver_frac(None)
    with pytest.raises(ValueError, match="not numeric"):
        derive_sliver_frac(["abc", 1.0, 2.0, 3.0, 4.0])  # type: ignore[list-item]  # the non-numeric entry is the subject of the refusal


def test_derive_block_scale_px_gt_object_spacing_floored_at_tile_size():
    # Objects 200px apart, tile_size 50: the derived scale (median NN spacing) is 200, well above
    # the floor, so the floor never engages.
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    px, source = derive_block_scale_px(tile_size=50, objects=_regions([boxes])[0])
    assert px == 200
    assert "GT object-spacing" in source


def test_derive_block_scale_px_floors_at_tile_size_when_spacing_is_smaller():
    boxes = [(x, 0, 5, 5) for x in range(0, 100, 10)]  # 10px spacing
    px, source = derive_block_scale_px(tile_size=64, objects=_regions([boxes])[0])
    assert px == 64  # the tile_size floor wins over the smaller measured spacing
    assert "floored at tile_size" in source


def test_derive_block_scale_px_no_data_refuses_named():
    with pytest.raises(ValueError, match="no block scale is derivable"):
        derive_block_scale_px(tile_size=64, objects=_regions([[]])[0])


def test_derive_block_scale_px_insufficient_plant_registry_refuses_named():
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    one_plant = [PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0)]
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    with pytest.raises(ValueError, match="plant grid pitch is underivable"):
        derive_block_scale_px(
            tile_size=50, objects=_regions([boxes])[0], plants=one_plant, raster_path=None)


def test_derive_block_scale_px_plant_pitch_via_projected_geotransform(tmp_path):
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    # Two plants 100m apart (haversine-ish at this latitude close enough for a unit test), on a
    # real UTM 15N GeoTIFF at the fixture's default 0.5 m/px.
    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),  # ~100m north
    ]
    raster_path = tmp_path / "mosaic.tif"
    write_geotiff(raster_path)
    boxes = [(x, 0, 20, 20) for x in range(0, 40, 20)]  # sparse GT: the plant path must win
    px, source = derive_block_scale_px(
        tile_size=16, objects=_regions([boxes])[0], plants=plants,
        raster_path=str(raster_path))
    assert "plant grid pitch" in source
    assert "EPSG:32615" in source
    assert px == pytest.approx(200, rel=0.05)  # ~100m / 0.5 m-per-px = ~200px


def test_derive_block_scale_px_reads_the_rasters_header_once(tmp_path, monkeypatch):
    """The refusal of an unreadable raster and the pixel size the plant pitch converts through
    come off one header read."""
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

    def counting(tif):
        reads.append(tif.filehandle.path)
        return real(tif)

    monkeypatch.setattr(raster_source, "tiff_header", counting)
    _px, source = derive_block_scale_px(
        tile_size=16, objects=_regions([[(0, 0, 20, 20)]])[0], plants=plants,
        raster_path=str(raster_path))
    assert "plant grid pitch" in source
    assert len(reads) == 1


def test_derive_block_scale_px_converts_a_foot_unit_raster_through_its_crs(tmp_path):
    """A raster in US survey feet (EPSG 2264) converts the plant-pitch meters through the CRS's
    own unit conversion factor, not a naive meter-blind pixel-scale division."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),  # ~99.96m north (haversine_m)
    ]
    raster_path = tmp_path / "mosaic.tif"
    write_geotiff(raster_path, pixel_scale=(1.0, 1.0, 0.0), projected_epsg=2264)
    boxes = [(x, 0, 20, 20) for x in range(0, 40, 20)]  # sparse GT: the plant path must win
    px, source = derive_block_scale_px(
        tile_size=16, objects=_regions([boxes])[0], plants=plants,
        raster_path=str(raster_path))
    assert "plant grid pitch" in source
    assert "EPSG:2264" in source
    assert px == 328  # 99.96m / 0.3048006 ft-per-m factor; a meter-blind read would give 100


def test_derive_block_scale_px_photographic_raster_path_refuses_named(tmp_path):
    """A ``raster_path`` that is not a raster file at all (a photographic plot image) is refused
    by name rather than silently downgraded to the GT-object-spacing fallback."""
    from PIL import Image

    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]
    photo_path = tmp_path / "plot.jpg"
    Image.new("RGB", (8, 8)).save(photo_path)
    boxes = [(x, 0, 20, 20) for x in range(0, 40, 20)]
    with pytest.raises(ValueError) as exc_info:
        derive_block_scale_px(
            tile_size=16, objects=_regions([boxes])[0], plants=plants,
            raster_path=str(photo_path))
    assert str(photo_path) in str(exc_info.value)


def test_derive_block_scale_px_anisotropic_raster_falls_back_to_gt_spacing(tmp_path):
    """A raster whose axes carry differing pixel scales has no single pixel size to convert the
    plant pitch through, so the derivation falls back to GT-object-spacing rather than average
    the two axes."""
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
        tile_size=50, objects=_regions([boxes])[0], plants=plants,
        raster_path=str(raster_path))
    assert "GT object-spacing" in source  # fell back, not a refusal
    assert px == 200


def test_derive_block_scale_px_unprojected_raster_falls_back_to_gt_spacing(monkeypatch, tmp_path):
    from tcip_mcp.pipelines.postprocessing import orthomosaic_mapping
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
    from tests._geotiff_fixtures import write_geotiff

    plants = [
        PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0),
        PlantRecord("p1", "a1", 0, 0, 0, 45.000898, -93.0),
    ]

    def _boom(tags, path):
        raise orthomosaic_mapping.GeoreferencingError("no geokeys")

    monkeypatch.setattr(orthomosaic_mapping, "geotransform_of", _boom)
    raster_path = tmp_path / "mosaic.tif"
    # a real georeferenced raster: without the stub above the plant path would win, so this
    # test exercises the GeoreferencingError fallback rather than an empty file's own decline
    write_geotiff(raster_path)
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    px, source = derive_block_scale_px(
        tile_size=50, objects=_regions([boxes])[0], plants=plants,
        raster_path=str(raster_path))
    assert "GT object-spacing" in source  # fell back, not a refusal
    assert px == 200


def test_derive_block_scale_px_truncated_raster_refuses_named(tmp_path):
    """A raster_path with a raster suffix that cannot be opened at all (truncated/corrupt) is a
    file-level problem, refused by name rather than falling back to GT-object-spacing."""
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
            tile_size=50, objects=_regions([boxes])[0], plants=plants,
            raster_path=str(raster_path))


def test_derive_block_scale_px_npy_raster_refuses_named_for_no_georeference(tmp_path):
    """guard. An .npy array container is a raster by suffix but carries no georeferencing tags at
    all; it is refused by name, distinct from the truncated-file words above, rather than tried
    through the header's georeference read."""
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
            tile_size=50, objects=_regions([boxes])[0], plants=plants,
            raster_path=str(raster_path))


def test_write_subject_registry(tmp_path):
    import json

    from tcip_mcp import subject_registry
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    out = tmp_path / "subjects.json"
    # The expert authors the nested registry: two ordered values of a categorical attribute.
    res = write_subject_registry(
        tmp_path, str(tmp_path),
        subjects={"bud": {"description": "a currant bud",
                             "attributes": {"opening": {"type": "categorical",
                                                           "values": ["closed", "open"]}}}},
    )
    assert "error" not in res
    assert res["subjects"] == ["bud"]
    assert res["subjects_path"] == str(out)
    # Declared order is the id order (a value's id is its position): 0=closed, 1=open, and the
    # on-disk nested shape carries the same value order.
    reg = subject_registry.read_registry(tmp_path)
    assert reg.subjects[0].attribute("opening").values == ("closed", "open")  # type: ignore[union-attr]
    assert json.loads(out.read_text())["bud"]["attributes"]["opening"]["values"] == \
        ["closed", "open"]


def test_write_subject_registry_no_labels(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_subject_registry
    # An empty registry mapping is not authorable: the tool refuses rather than writing nothing.
    res = write_subject_registry(tmp_path, str(tmp_path), subjects={})
    assert "error" in res

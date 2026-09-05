"""Tier-A data/model derivations (channels/classes/anchors from the data in hand)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.pipelines.derivations import (
    derive_block_scale_px,
    derive_cross_tile_nms,
    derive_iou_match_threshold,
    derive_localization_kind,
    derive_localization_tolerance_frac,
    derive_sliver_frac,
    gt_aspect_ratios,
    num_classes_from_distribution,
    probe_channels,
)


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
    # tall boxes (h/w ~ 4) -> the derived ratio set must include a tall ratio the default (0.5,1,2) lacks
    boxes = [(10.0, 40.0)] * 20
    ratios = gt_aspect_ratios(boxes)
    assert max(ratios) >= 3.0


def test_derive_cross_tile_nms_dense_cluster_exceeds_sparse():
    # Dense boxes (20px, offset 4px -> neighbor IoU ~0.667) push the threshold up so genuinely-
    # overlapping dense objects aren't merged; sparse boxes (offset 16px -> IoU ~0.111) sit lower.
    dense = [[(0, 0, 20, 20), (4, 0, 20, 20), (8, 0, 20, 20), (12, 0, 20, 20)]]
    sparse = [[(0, 0, 20, 20), (16, 0, 20, 20), (32, 0, 20, 20)]]
    t_dense = derive_cross_tile_nms(dense)
    t_sparse = derive_cross_tile_nms(sparse)
    assert t_dense is not None and t_sparse is not None
    assert t_dense > t_sparse
    assert 0.2 <= t_sparse <= 0.8 and 0.2 <= t_dense <= 0.8
    assert t_dense == pytest.approx(0.6667 + 0.05, abs=1e-2)  # p99 of the neighbor-IoU tail + margin


def test_derive_cross_tile_nms_no_overlap_returns_none():
    # No genuine neighbor overlap anywhere -> underivable -> caller must fall back to an honest default.
    boxes = [[(0, 0, 20, 20), (100, 100, 20, 20)], [(0, 0, 20, 20)]]
    assert derive_cross_tile_nms(boxes) is None
    assert derive_cross_tile_nms([]) is None


def test_derive_cross_tile_nms_clamped_to_upper_bound():
    # Near-duplicate boxes (IoU ~0.90) would exceed the range; the result is clamped to the ceiling.
    boxes = [[(0, 0, 20, 20), (1, 0, 20, 20)]]
    assert derive_cross_tile_nms(boxes) == pytest.approx(0.8)


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
    # p10 nn-dist (10) * margin_frac (0.5) / char_size (20) = 0.25, no clamping.
    assert t_tight == pytest.approx(0.25)
    # Loose spacing's raw fraction (2.5) exceeds the clamp ceiling.
    assert t_loose == pytest.approx(0.75)


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
    assert f_tight == pytest.approx(0.9)  # clamped to the ceiling


def test_derive_sliver_frac_no_boxes_returns_none():
    assert derive_sliver_frac([]) is None
    assert derive_sliver_frac([0.0, 0.0]) is None


def test_derive_sliver_frac_too_few_samples_returns_none():
    # A single box's ratio to itself is trivially ~1.0 regardless of the class's real variation:
    # not a spread, just noise. Below min_samples must refuse rather than derive from it.
    assert derive_sliver_frac([25.6]) is None
    assert derive_sliver_frac([10.0, 20.0, 30.0, 40.0]) is None  # 4 < default min_samples=5


def test_derive_localization_kind_small_objects_are_center_match():
    # 20x20 boxes (char size 20): well under the ~45px default crossover; IoU would be unreliable
    # under realistic jitter, so center-match must govern.
    boxes = [[(0, 0, 20, 20), (100, 0, 20, 20)]]
    assert derive_localization_kind(boxes) == "center_match"


def test_derive_localization_kind_large_objects_are_iou_match():
    # 200x200 boxes (char size 200): well over the crossover; IoU is a meaningful criterion here.
    boxes = [[(0, 0, 200, 200), (500, 0, 200, 200)]]
    assert derive_localization_kind(boxes) == "iou_match"


def test_derive_localization_kind_no_boxes_returns_none():
    assert derive_localization_kind([]) is None
    assert derive_localization_kind([[], []]) is None
    assert derive_localization_kind([[(0, 0, 0, 0)]]) is None  # zero-area box, not a valid size


def test_derive_localization_kind_crossover_is_monotonic_in_size():
    # Larger characteristic size must never flip from iou_match back to center_match: the
    # achievable-IoU formula is monotonically increasing in size, so this must hold for any jitter.
    small = derive_localization_kind([[(0, 0, 10, 10)]])
    mid = derive_localization_kind([[(0, 0, 45, 45)]])
    large = derive_localization_kind([[(0, 0, 500, 500)]])
    assert small == "center_match"
    assert large == "iou_match"
    assert mid in ("center_match", "iou_match")  # near the crossover, either is defensible


def test_derive_iou_match_threshold_exact_value():
    # char size 60 -> achievable_iou = (60-15)/(60+15) = 0.6 -> threshold = 0.6 - margin(0.1) = 0.5.
    boxes = [[(0, 0, 60, 60)]]
    assert derive_iou_match_threshold(boxes) == pytest.approx(0.5)


def test_derive_iou_match_threshold_scales_with_object_size():
    # Larger characteristic size -> higher achievable IoU under the same jitter -> higher threshold.
    small = derive_iou_match_threshold([[(0, 0, 60, 60)]])
    large = derive_iou_match_threshold([[(0, 0, 300, 300)]])
    assert small is not None and large is not None
    assert large > small


def test_derive_iou_match_threshold_clamped_to_upper_bound():
    # A very large object's achievable IoU under jitter approaches 1.0 -> clamped at 0.7.
    assert derive_iou_match_threshold([[(0, 0, 1000, 1000)]]) == pytest.approx(0.7)


def test_derive_iou_match_threshold_no_boxes_returns_none():
    assert derive_iou_match_threshold([]) is None
    assert derive_iou_match_threshold([[], []]) is None
    assert derive_iou_match_threshold([[(0, 0, 0, 0)]]) is None


@pytest.mark.parametrize("fn", [
    derive_localization_tolerance_frac, derive_localization_kind,
    derive_iou_match_threshold, derive_cross_tile_nms,
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
    px, source = derive_block_scale_px(tile_size=50, gt_boxes_per_image=[boxes])
    assert px == 200
    assert "GT object-spacing" in source


def test_derive_block_scale_px_floors_at_tile_size_when_spacing_is_smaller():
    boxes = [(x, 0, 5, 5) for x in range(0, 100, 10)]  # 10px spacing
    px, source = derive_block_scale_px(tile_size=64, gt_boxes_per_image=[boxes])
    assert px == 64  # the tile_size floor wins over the smaller measured spacing
    assert "floored at tile_size" in source


def test_derive_block_scale_px_no_data_refuses_named():
    with pytest.raises(ValueError, match="no block scale is derivable"):
        derive_block_scale_px(tile_size=64, gt_boxes_per_image=[[]])


def test_derive_block_scale_px_insufficient_plant_registry_refuses_named():
    from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord

    one_plant = [PlantRecord("p0", "a0", 0, 0, 0, 45.0, -93.0)]
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    with pytest.raises(ValueError, match="plant grid pitch is underivable"):
        derive_block_scale_px(
            tile_size=50, gt_boxes_per_image=[boxes], plants=one_plant, raster_path=None)


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
        tile_size=16, gt_boxes_per_image=[boxes], plants=plants,
        raster_path=str(raster_path))
    assert "plant grid pitch" in source
    assert "EPSG:32615" in source
    assert px == pytest.approx(200, rel=0.05)  # ~100m / 0.5 m-per-px = ~200px


def test_derive_block_scale_px_converts_a_foot_unit_raster_through_its_crs(tmp_path):
    """A raster in US survey feet (EPSG 2264) converts the plant-pitch metres through the CRS's
    own unit conversion factor, not a naive metre-blind pixel-scale division."""
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
        tile_size=16, gt_boxes_per_image=[boxes], plants=plants,
        raster_path=str(raster_path))
    assert "plant grid pitch" in source
    assert "EPSG:2264" in source
    assert px == 328  # 99.96m / 0.3048006 ft-per-m factor; a metre-blind read would give 100


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
            tile_size=16, gt_boxes_per_image=[boxes], plants=plants,
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
        tile_size=50, gt_boxes_per_image=[boxes], plants=plants,
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

    def _boom(path):
        raise orthomosaic_mapping.GeoreferencingError("no geokeys")

    monkeypatch.setattr(orthomosaic_mapping, "read_geotransform", _boom)
    raster_path = tmp_path / "mosaic.tif"
    # a real georeferenced raster: without the stub above the plant path would win, so this
    # test exercises the GeoreferencingError fallback rather than an empty file's own decline
    write_geotiff(raster_path)
    boxes = [(x, 0, 20, 20) for x in range(0, 1000, 200)]
    px, source = derive_block_scale_px(
        tile_size=50, gt_boxes_per_image=[boxes], plants=plants,
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
            tile_size=50, gt_boxes_per_image=[boxes], plants=plants,
            raster_path=str(raster_path))


def test_write_class_map(tmp_path):
    import json

    from tcip_mcp import class_registry
    from tcip_mcp.tools.annotation_tools import write_class_map

    out = tmp_path / "classes.json"
    # The expert authors the nested registry: two ordered values of a categorical attribute.
    res = write_class_map(
        str(tmp_path),
        subjects={"bud": {"description": "a currant bud",
                             "attributes": {"opening": {"type": "categorical",
                                                           "values": ["closed", "open"]}}}},
        output_path=str(out),
    )
    assert "error" not in res
    assert res["subjects"] == ["bud"]
    assert res["classes_path"] == str(out)
    # Declared order is the id order (assign_class_ids is the one name->id derivation): 0=closed,
    # 1=open, and the on-disk nested shape carries the same value order.
    reg = class_registry.read_registry(out)
    assert class_registry.assign_class_ids(reg, "bud", "opening") == {"closed": 0, "open": 1}
    assert json.loads(out.read_text())["bud"]["attributes"]["opening"]["values"] == \
        ["closed", "open"]


def test_write_class_map_no_labels(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map
    # An empty registry mapping is not authorable: the tool refuses rather than writing nothing.
    res = write_class_map(str(tmp_path), subjects={}, output_path=str(tmp_path / "c.json"))
    assert "error" in res


def test_run_inference_dry_run_reports_operating_point(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")  # dry_run never loads it
    res = run_inference(str(ckpt), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                        dry_run=True, tile=True, tile_size=640)
    assert res["dry_run"] is True
    op = res["operating_point"]
    assert op["conf"] == 0.5  # DEFAULT_CONF (one shared source)
    assert op["cross_tile_nms"] == 0.3  # DEFAULT_NMS_IOU, tiled
    assert op["tiled"] is True and op["tile_size"] == 640


def test_run_inference_dry_run_unset_tile_is_pending_not_a_default(tmp_path):
    """Once tiled also derives from the checkpoint's own training geometry, an unset ``tile`` in a
    dry run (which never loads the checkpoint) can't be resolved to a concrete bool: it must report
    a genuine pending derivation, the same convention tile_size/overlap already use for the
    checkpoint-derived dimensions, never a fabricated True/False."""
    from tcip_mcp.tools.inference_tools import run_inference
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")  # dry_run never loads it
    res = run_inference(str(ckpt), images_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                        dry_run=True)
    assert res["dry_run"] is True
    op = res["operating_point"]
    assert op["tiled"] == "pending-checkpoint-derivation"
    assert op["tiled_source"] == "pending-checkpoint-derivation"
    assert op["cross_tile_nms"] == "pending-checkpoint-derivation"
    assert op["tile_size"] == "pending-checkpoint-derivation"
    assert op["overlap"] == "pending-checkpoint-derivation"


def test_write_class_map_defaults_into_the_dataset(tmp_path):
    """No output_path: the registry lands at the dataset's canonical classes.json."""
    from tcip_mcp.tools.annotation_tools import write_class_map

    res = write_class_map(
        str(tmp_path),
        subjects={"bud": {"description": "a currant bud",
                             "attributes": {"opening": {"type": "categorical",
                                                           "values": ["closed", "open"]}}}},
    )
    assert "error" not in res
    assert res["classes_path"] == str(tmp_path / "classes.json")
    assert (tmp_path / "classes.json").is_file()

"""Tier-A data/model derivations (channels/classes/anchors from the data in hand)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.pipelines.derivations import (
    derive_cross_tile_nms,
    gt_aspect_ratios,
    num_classes_from_distribution,
    probe_channels,
    resolve_spec_derivations,
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


def test_gt_aspect_ratios_covers_elongated():
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


def test_resolve_spec_derivations_fills_when_absent(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    spec = {"backbone": {"name": "resnet18"}, "heads": [{"name": "anchor_detection"}]}
    prov = resolve_spec_derivations(spec, sample_image=tmp_path / "a.png",
                                    class_distribution={0: 5, 1: 3})
    assert spec["backbone"]["in_chans"] == 3
    assert spec["heads"][0]["num_classes"] == 2
    assert set(prov) == {"in_chans", "num_classes"}
    assert prov["num_classes"].value == 2  # deterministic -> shippable


def test_resolve_spec_derivations_respects_explicit(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    spec = {"backbone": {"name": "resnet18", "in_chans": 4},
            "heads": [{"name": "anchor_detection", "num_classes": 7}]}
    prov = resolve_spec_derivations(spec, sample_image=tmp_path / "a.png",
                                    class_distribution={0: 5})
    assert spec["backbone"]["in_chans"] == 4  # explicit untouched
    assert spec["heads"][0]["num_classes"] == 7
    assert prov == {}  # nothing derived — the agent pinned both


def test_write_class_map(tmp_path):
    import json
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox
    from tcip_mcp.tools.annotation_tools import write_class_map
    ld = tmp_path / "labels"
    ld.mkdir()
    json_io.write_detect(ld / "a.json", [BBox(0, 0, 10, 10, 0), BBox(0, 0, 10, 10, 1)], 100, 100)
    json_io.write_detect(ld / "b.json", [BBox(0, 0, 10, 10, 0)], 100, 100)
    out = tmp_path / "classes.json"
    res = write_class_map(str(ld), class_names="dormant,elongated", output_path=str(out))
    assert res["num_classes"] == 2
    assert res["class_ids"] == [0, 1]
    assert res["class_map"]["1"]["name"] == "elongated"
    assert json.loads(out.read_text())["0"]["name"] == "dormant"


def test_write_class_map_no_labels(tmp_path):
    from tcip_mcp.tools.annotation_tools import write_class_map
    ld = tmp_path / "labels"
    ld.mkdir()
    res = write_class_map(str(ld), output_path=str(tmp_path / "c.json"))
    assert "error" in res


def test_run_inference_dry_run_reports_operating_point(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")  # dry_run never loads it
    res = run_inference(str(ckpt), images_dir=str(tmp_path), dry_run=True, tile=True, tile_size=640)
    assert res["dry_run"] is True
    op = res["operating_point"]
    assert op["conf"] == 0.5  # DEFAULT_CONF (one shared source)
    assert op["cross_tile_nms"] == 0.3  # DEFAULT_NMS_IOU, tiled
    assert op["tiled"] is True and op["tile_size"] == 640

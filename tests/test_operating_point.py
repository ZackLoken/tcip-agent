"""Center-match count-unbiased operating-point sweep, the in-model operating-point seam, the
detection cap, and the cross-tile merge threshold a tiled pass derives."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests._dense_op_fixtures import _box, dense_records  # noqa: E402
from tests._dense_op_fixtures import ann as _ann  # noqa: E402
from tests._dense_op_fixtures import toy_records as _records  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    derive_operating_point_curve,
)


def test_gt_class_avg_size_derived_from_data():
    assert gt_class_avg_size(_records()) == pytest.approx(20.0)


def test_count_unbiased_differs_from_f1_max():
    recs = _records()
    tol = 0.5 * gt_class_avg_size(recs)  # derived tolerance = half class avg size
    sweep = derive_operating_point_curve(recs, criterion={"kind": "center_match", "tolerance": tol})

    cu = pick_count_unbiased(sweep)
    f1m = pick_f1_max(sweep)
    assert cu == pytest.approx(0.6)
    assert f1m == pytest.approx(0.0)  # F1 peaks at low conf (hesitant true det lifts recall)
    assert cu != f1m  # the whole point: a count phenotype is not optimized by F1

    by_conf = {round(c["conf"], 6): c for c in sweep["curve"]}
    # at the count-unbiased point the net per-image count bias vanishes...
    assert by_conf[round(cu, 6)]["count_bias_mean"] == pytest.approx(0.0)
    # ...while the F1-max point over-counts (keeps A's spurious det for recall's sake).
    assert by_conf[round(f1m, 6)]["count_bias_mean"] == pytest.approx(0.5)


def test_center_match_respects_tolerance():
    # a correct detection just outside tolerance must not count as a hit
    recs = [{"width": 400, "height": 400, "gt": [_ann(100, 100)],
             "dt": [_ann(100 + 100, 100, score=0.9)]}]  # 100px off, tolerance ~10
    sweep = derive_operating_point_curve(recs, criterion={
        "kind": "center_match", "tolerance": 0.5 * gt_class_avg_size(recs)})
    at0 = sweep["curve"][0]  # conf=0.0 is always the first (lowest) grid point
    assert at0["conf"] == pytest.approx(0.0)
    assert at0["tp"] == 0 and at0["fp"] == 1 and at0["fn"] == 1  # miss + false positive


def test_sweep_curve_carries_dispersion_and_reference_size_fields():
    # Every curve entry also carries count_error_p90 / count_bias_std / n_images, computed from
    # the same per-image biases list, not a second pass over the data.
    recs = dense_records(n_images=4, objects_per_image=10,
                         miss_pattern=[0, 1, 0, 2], fp_pattern=[0, 0, 1, 0])
    sweep = derive_operating_point_curve(recs, criterion={
        "kind": "center_match", "tolerance": 0.5 * gt_class_avg_size(recs)})
    at09 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.9))
    # biases = fp - fn per image = [0, -1, 1, -2]
    assert at09["n_images"] == 4
    assert at09["count_bias_mean"] == pytest.approx(-0.5)
    assert at09["count_bias_std"] == pytest.approx(1.2909944, abs=1e-5)
    assert at09["count_error_p90"] == pytest.approx(1.7, abs=1e-6)


# --- the in-model seam ---

def _two_stage():
    from types import SimpleNamespace
    return SimpleNamespace(detector=SimpleNamespace(
        roi_heads=SimpleNamespace(score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)))


def _one_stage():
    from types import SimpleNamespace
    return SimpleNamespace(
        detector=SimpleNamespace(score_thresh=0.2, nms_thresh=0.6, detections_per_img=100))


def test_set_detector_operating_point_two_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    m = _two_stage()
    applied, attribute_path = set_detector_operating_point(
        m, score_thresh=0.4, detections_per_img=300)
    assert m.detector.roi_heads.score_thresh == 0.4
    assert m.detector.roi_heads.nms_thresh == 0.5  # the builder's own NMS, never set here
    assert m.detector.roi_heads.detections_per_img == 300
    assert applied == {"score_thresh": 0.4, "detections_per_img": 300}
    assert attribute_path == "detector.roi_heads"


def test_set_detector_operating_point_one_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    m = _one_stage()
    set_detector_operating_point(m, score_thresh=0.4)
    assert m.detector.score_thresh == 0.4 and m.detector.nms_thresh == 0.6


def test_the_detection_cap_is_its_formula_at_every_density_and_refuses_no_positive_cap():
    """``ceil(1.5 * p99 density * footprint)``, a sparse scene included, with no floor over it;
    a reference whose selected percentile is zero yields no positive cap and refuses naming
    ``max_dets``, an all-empty one and a sparse one holding a single object alike."""
    from tcip_mcp.pipelines.derivations import derive_max_dets

    frame = 100.0 * 100.0
    assert derive_max_dets([(80, frame)] * 20, frame) == 120
    assert derive_max_dets([(2, frame)] * 20, frame) == 3
    assert derive_max_dets([(10, frame)] * 3, frame * 10) == 150
    for sparse in ([0] * 5, [0] * 199 + [1]):
        with pytest.raises(ValueError, match="no positive cap; state max_dets"):
            derive_max_dets([(n, frame) for n in sparse], frame)


# --- the cross-tile merge threshold ---

def _overlap_records(idp="d"):
    """Records whose GT boxes overlap (20px, offset 8px), so the merge threshold is derivable."""
    boxes = [_box(100, 100), _box(108, 100), _box(116, 100)]  # neighbor IoU ~0.43
    return [{"width": 400, "height": 400, "image_id": f"{idp}_{i}",
             "gt": [{"category_id": 1, "bbox": bx, "iscrowd": 0} for bx in boxes],
             "dt": [{"category_id": 1, "bbox": bx, "score": 0.9} for bx in boxes]}
            for i in range(2)]


def _gt_boxes(records):
    return [[a["bbox"] for a in rec["gt"]] for rec in records]


def test_the_merge_threshold_derives_from_the_ground_truths_neighbor_overlap_tail():
    from tcip_mcp.pipelines.derivations import derive_cross_tile_nms

    value = derive_cross_tile_nms(_gt_boxes(_overlap_records()))
    # p99 of the GT neighbor-IoU tail + margin
    assert value == pytest.approx(0.4286 + 0.05, abs=1e-2)


def test_no_overlapping_ground_truth_derives_no_merge_threshold():
    from tcip_mcp.pipelines.derivations import derive_cross_tile_nms

    assert derive_cross_tile_nms([]) is None
    assert derive_cross_tile_nms(_gt_boxes(_records("c"))) is None


def _tiled_pass(tmp_path, records, **stated):
    """The tiled pass of a verified checkpoint at the sample conf and cap, made runnable from
    ``records``' ground truth."""
    from tcip_mcp.pipelines.execution import Reference, Stated, prepare
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_MAX_DETS, verified_checkpoint,
    )

    prep = prepare(verified_checkpoint(tmp_path),
                   Stated(tile=True, tile_size=64, conf=SAMPLE_CONF, max_dets=SAMPLE_MAX_DETS,
                          **stated))
    return prep.runnable(Reference(boxes_per_image=_gt_boxes(records), counted=None,
                                   footprint=None))


def test_a_tiled_pass_records_its_derived_merge_threshold_by_the_derivations_name(tmp_path):
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATION

    p = _tiled_pass(tmp_path, _overlap_records())

    assert p.execution.cross_tile_nms == pytest.approx(0.4786, abs=1e-2)
    assert p.execution.sources["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATION


def test_a_stated_merge_threshold_is_never_relabeled_derived(tmp_path):
    p = _tiled_pass(tmp_path, _overlap_records(), cross_tile_nms=0.55)

    assert p.execution.cross_tile_nms == pytest.approx(0.55)
    assert p.execution.sources["cross_tile_nms"] == "explicit"


def test_an_underivable_merge_threshold_refuses_naming_its_missing_basis(tmp_path):
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(ExecutionRefusedError, match="no two of its boxes in one image overlap"):
        _tiled_pass(tmp_path, _records("c"))


def test_classification_metrics_per_class_and_bias():
    from tcip_mcp.pipelines.training.evaluation import classification_metrics
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])    # 6 closed, 4 open
    pred = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # classifier predicts 6 as open
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["per_class"]["1"]["support"] == 4
    # over-predicting the open class inflates the open fraction: bias (6-4)/4 = +0.5
    assert m["count_bias"]["1"] == pytest.approx(0.5)
    assert "accuracy" in m and "f1" in m  # existing keys preserved (additive)

"""Tests of the operating point and the values a pass derives beside it.

Covered: the center-match count-unbiased operating-point sweep, the in-model operating-point
seam, the detection cap, and the cross-tile merge threshold a tiled pass derives.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch"
)  # evaluation.py imports torch at module load

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    derive_operating_point_curve,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
)
from tests._dense_op_fixtures import (  # noqa: E402
    ann as _ann,  # noqa: E402
    dense_records,
    toy_records as _records,  # noqa: E402
)
from tests._training_values import DISPERSION_QUANTILE  # noqa: E402


def test_gt_class_avg_size_derived_from_data():
    assert gt_class_avg_size(_records()) == pytest.approx(20.0)


def test_count_unbiased_differs_from_f1_max():
    recs = _records()
    tol = 0.5 * gt_class_avg_size(
        recs
    )  # derived tolerance = half class avg size
    sweep = derive_operating_point_curve(
        recs,
        criterion={"kind": "center_match", "tolerance": tol},
        dispersion_quantile=DISPERSION_QUANTILE,
    )

    cu = pick_count_unbiased(sweep)
    f1m = pick_f1_max(sweep)
    assert cu == pytest.approx(0.6)
    assert f1m == pytest.approx(
        0.0
    )  # F1 peaks at low conf (hesitant true det lifts recall)
    assert (
        cu != f1m
    )  # the whole point: a count phenotype is not optimized by F1

    by_conf = {round(c["conf"], 6): c for c in sweep["curve"]}
    # at the count-unbiased point the net per-image count bias vanishes...
    assert by_conf[round(cu, 6)]["count_bias_mean"] == pytest.approx(0.0)
    # ...while the F1-max point over-counts (keeps A's spurious det for recall's sake).
    assert by_conf[round(f1m, 6)]["count_bias_mean"] == pytest.approx(0.5)


def test_center_match_respects_tolerance():
    # a correct detection just outside tolerance must not count as a hit
    from tcip_mcp.pipelines.training.evaluation import prediction_record

    recs = [
        prediction_record(
            [_ann(100 + 100, 100, score=0.9)],
            [_ann(100, 100)],
            width=400,
            height=400,
            cap=None,
            count=1,
        )
    ]  # 100px off, tolerance ~10
    sweep = derive_operating_point_curve(
        recs,
        criterion={
            "kind": "center_match",
            "tolerance": 0.5 * gt_class_avg_size(recs),
        },
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    at0 = sweep["curve"][0]  # conf=0.0 is always the first (lowest) grid point
    assert at0["conf"] == pytest.approx(0.0)
    assert (
        at0["tp"] == 0 and at0["fp"] == 1 and at0["fn"] == 1
    )  # miss + false positive


def test_sweep_curve_carries_dispersion_and_reference_size_fields():
    # Every curve entry also carries count_error_at_quantile / count_bias_std / n_images, computed
    # from the same per-image biases list, not a second pass over the data.
    recs = dense_records(
        n_images=4,
        objects_per_image=10,
        miss_pattern=[0, 1, 0, 2],
        fp_pattern=[0, 0, 1, 0],
    )
    sweep = derive_operating_point_curve(
        recs,
        criterion={
            "kind": "center_match",
            "tolerance": 0.5 * gt_class_avg_size(recs),
        },
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    at09 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.9))
    # biases = fp - fn per image = [0, -1, 1, -2]
    assert at09["n_images"] == 4
    assert at09["count_bias_mean"] == pytest.approx(-0.5)
    assert at09["count_bias_std"] == pytest.approx(1.2909944, abs=1e-5)
    assert at09["count_error_at_quantile"] == pytest.approx(1.7, abs=1e-6)
    assert sweep["dispersion_quantile"] == DISPERSION_QUANTILE


def test_the_curve_evaluates_every_distinct_calibration_score():
    from tcip_mcp.pipelines.training.evaluation import prediction_record

    scores = [round(0.01 + 0.0097 * i, 6) for i in range(101)]
    recs = [
        prediction_record(
            [
                _ann(20 + 30 * (i % 50), 100, 1, score=s)
                for i, s in enumerate(scores)
            ],
            [_ann(20 + 30 * i, 100, 1) for i in range(50)],
            width=4000,
            height=400,
            cap=None,
            count=len(scores),
        )
    ]
    sweep = derive_operating_point_curve(
        recs,
        criterion={
            "kind": "center_match",
            "tolerance": 0.5 * gt_class_avg_size(recs),
        },
        dispersion_quantile=DISPERSION_QUANTILE,
    )
    assert [c["conf"] for c in sweep["curve"]] == pytest.approx([0.0, *scores])


# --- the in-model seam ---


def _two_stage():
    from types import SimpleNamespace

    return SimpleNamespace(
        detector=SimpleNamespace(
            roi_heads=SimpleNamespace(
                score_thresh=0.05, nms_thresh=0.5, detections_per_img=100
            )
        )
    )


def _one_stage():
    from types import SimpleNamespace

    return SimpleNamespace(
        detector=SimpleNamespace(
            score_thresh=0.2, nms_thresh=0.6, detections_per_img=100
        )
    )


def test_set_detector_operating_point_two_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    m = _two_stage()
    set_detector_operating_point(m, score_thresh=0.4, detections_per_img=300)
    assert m.detector.roi_heads.score_thresh == 0.4
    assert (
        m.detector.roi_heads.nms_thresh == 0.5
    )  # the builder's own NMS, never set here
    assert m.detector.roi_heads.detections_per_img == 300


def test_set_detector_operating_point_one_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point

    m = _one_stage()
    set_detector_operating_point(m, score_thresh=0.4, detections_per_img=300)
    assert m.detector.score_thresh == 0.4 and m.detector.nms_thresh == 0.6
    assert m.detector.detections_per_img == 300


def test_the_object_density_is_its_quantile_over_the_object_bearing_regions():
    """The object density of uniformly dense regions is their objects per pixel, empty ones skipped.

    Regions of 80 or 2 objects a frame, or 10 over a tenfold frame, derive that density within
    ``pytest.approx``'s default tolerance; a sparse reference whose one object-bearing region sits
    among empty ones derives that region's density, and a reference with no object-bearing region
    refuses naming them.
    """
    from tcip_mcp.pipelines.derivations import derive_object_density
    from tests._verified_checkpoint_fixtures import objects_over

    def counted(n: int, area: float) -> list:
        return [objects_over([[0.0, 0.0, 1.0, 1.0]] * n, area)]

    frame = 100.0 * 100.0
    assert derive_object_density(counted(80, frame) * 20) == pytest.approx(
        80 / frame
    )
    assert derive_object_density(counted(2, frame) * 20) == pytest.approx(
        2 / frame
    )
    assert derive_object_density(counted(10, frame * 10) * 3) == pytest.approx(
        1 / frame
    )
    sparse = [r for n in [0] * 199 + [1] for r in counted(n, frame)]
    assert derive_object_density(sparse) == pytest.approx(1 / frame)
    with pytest.raises(ValueError, match="object-bearing"):
        derive_object_density([r for _ in range(5) for r in counted(0, frame)])


# --- the cross-tile merge threshold ---


def _overlap_records(idp="d"):
    """Return records ``{idp}_0`` and ``{idp}_1``, each of three 20 px GT boxes 8 px apart.

    Each GT box is detected at 0.9.
    """
    from tcip_mcp.pipelines.training.evaluation import prediction_record

    centers = [100, 108, 116]  # neighbor IoU ~0.43
    return [
        prediction_record(
            [_ann(x, 100, 1, score=0.9) for x in centers],
            [_ann(x, 100, 1) for x in centers],
            width=400,
            height=400,
            cap=None,
            count=len(centers),
            image_id=f"{idp}_{i}",
        )
        for i in range(2)
    ]


def _regions(records):
    """Return each record's ground truth as the region of its objects over its frame."""
    from tests._verified_checkpoint_fixtures import objects_over

    return [
        objects_over(
            [
                [x, y, x + w, y + h]
                for x, y, w, h in (a["bbox"] for a in rec["gt"])
            ],
            rec["width"] * rec["height"],
        )
        for rec in records
    ]


def test_the_merge_threshold_derives_from_the_ground_truths_neighbor_overlap_tail():
    from tcip_mcp.pipelines.derivations import derive_cross_tile_nms

    value = derive_cross_tile_nms(_regions(_overlap_records()))
    # p99 of the GT neighbor-IoU tail + margin
    assert value == pytest.approx(0.4286 + 0.05, abs=1e-2)


def test_no_overlapping_ground_truth_derives_no_merge_threshold():
    from tcip_mcp.pipelines.derivations import derive_cross_tile_nms

    assert derive_cross_tile_nms([]) is None
    assert derive_cross_tile_nms(_regions(_records("c"))) is None


def _tiled_pass(tmp_path, records, **stated):
    """Return a verified checkpoint's tiled pass, made runnable from ``records``' ground truth.

    The pass runs at the sample conf and merges by NMS.
    """
    from tcip_mcp.pipelines.execution import Reference, Stated, prepare
    from tests._training_values import RATIONALE
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF,
        verified_checkpoint,
    )

    prep = prepare(
        verified_checkpoint(tmp_path),
        Stated(
            tile=True,
            tile_size=64,
            conf=SAMPLE_CONF,
            postprocess="nms",
            rationale=RATIONALE,
            **stated,
        ),
    )
    return prep.runnable(Reference(regions=_regions(records)))


def test_a_tiled_pass_records_its_derived_merge_threshold_by_the_derivations_name(
    tmp_path,
):
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATION

    p = _tiled_pass(tmp_path, _overlap_records())

    assert p.execution.cross_tile_nms == pytest.approx(0.4786, abs=1e-2)
    assert p.execution.sources["cross_tile_nms"] == CROSS_TILE_NMS_DERIVATION


def test_a_tiled_passs_derived_merge_threshold_is_the_method_its_label_names(
    tmp_path,
):
    """A tiled pass's derived merge threshold is the method its label names.

    Over a spread of neighbor overlaps, the threshold a tiled pass records is the labeled
    percentile of the tail plus the labeled margin.
    """
    import re

    import numpy as np

    from tcip_mcp.pipelines.training.evaluation import prediction_record
    from tests.test_derivations import _label_percentile

    offsets = range(1, 16)
    records = [
        prediction_record(
            [],
            [_ann(110, 110, 1), _ann(110 + d, 110, 1)],
            width=400,
            height=400,
            cap=None,
            count=0,
            image_id=f"o_{d}",
        )
        for d in offsets
    ]
    p = _tiled_pass(tmp_path, records)

    label = p.execution.sources["cross_tile_nms"]
    margin = re.search(r"plus ([\d.]+)$", label)
    assert margin, label
    tail = [(20 - d) / (20 + d) for d in offsets for _ in range(2)]
    assert p.execution.cross_tile_nms == pytest.approx(
        np.percentile(tail, _label_percentile(label)) + float(margin.group(1))
    )


def test_a_stated_merge_threshold_is_never_relabeled_derived(tmp_path):
    p = _tiled_pass(tmp_path, _overlap_records(), cross_tile_nms=0.55)

    assert p.execution.cross_tile_nms == pytest.approx(0.55)
    assert p.execution.sources["cross_tile_nms"] == "explicit"


def test_an_underivable_merge_threshold_refuses_naming_its_missing_basis(
    tmp_path,
):
    from tcip_mcp.pipelines.execution import ExecutionRefusedError

    with pytest.raises(
        ExecutionRefusedError,
        match="no two of its objects in one region overlap",
    ):
        _tiled_pass(tmp_path, _records("c"))


def test_classification_metrics_per_class_and_bias():
    from tcip_mcp.pipelines.training.evaluation import classification_metrics

    gt = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # 6 closed, 4 open
    pred = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    )  # classifier predicts 6 as open
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["per_class"]["1"]["support"] == 4
    # over-predicting the open class inflates the open fraction: bias (6-4)/4 = +0.5
    assert m["count_bias"]["1"] == pytest.approx(0.5)
    assert "accuracy" in m and "f1" in m  # existing keys preserved (additive)

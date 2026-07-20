"""Phase-0 CHARACTERIZATION GOLDENS — pin the CURRENT numeric behavior of the measurement
and provenance rails that later workstreams (W1/W5/W6/W7) will touch.

These are deliberately *exact*: they assert the numbers today's code produces on tiny
deterministic fixtures, so a later semantic change (a different conf pick, a re-defined
localization criterion, a consolidated default, a re-shaped stamp) fails LOUDLY instead of
sliding through silently. They are not aspirational — a golden turning red is the signal to
update it *deliberately* alongside the change that moved the number.

Rails pinned here (one section each):
  1. conf operating-point sweep + count-unbiased pick + resolve_operating_point
  2. phenology fraction curve + milestone dates (crossing_date / plant_milestones / per_plant_phenology)
  3. operating_point.json stamp shape + the validated flag path (calibrated vs raw)
  4. the CURRENTLY divergent NMS / max_dets defaults across the three modules (W6 target)
  5. IoU-matching eval metrics at iou_threshold=0.5 (W6-R3 criterion-change target)
  6. compute_phenology gate behavior (refuses without an elongation class; requires validated flags)
"""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import PredBBox  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology as PH  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    coco_detection_metrics,
    count_bias_at,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    sweep_operating_point,
)


# ── shared fixture helpers ────────────────────────────────────────────────

def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]  # xywh centered at (cx, cy)


def _ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _sweep_records(idp="c"):
    """Two images engineered so count-unbiased conf (0.6) != F1-max conf (0.0).

    Image A: 1 GT; correct det @0.9 + a spurious far det @0.6.
    Image B: 2 GT; correct det @0.9 + a hesitant-but-correct det @0.3.
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_ann(100, 100)],
         "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_ann(100, 100), _ann(200, 200)],
         "dt": [_ann(100, 100, score=0.9), _ann(200, 200, score=0.3)]}
    return [a, b]


# ══════════════════════════════════════════════════════════════════════════
# 1. conf operating-point sweep + count-unbiased pick + resolve_operating_point
# ══════════════════════════════════════════════════════════════════════════

def test_golden_sweep_curve_exact():
    recs = _sweep_records()
    assert gt_class_avg_size(recs) == pytest.approx(20.0)
    tol = 0.5 * gt_class_avg_size(recs)
    assert tol == pytest.approx(10.0)

    sweep = sweep_operating_point(recs, tolerance=tol)
    assert sweep["tolerance"] == pytest.approx(10.0)
    assert sweep["class_id"] is None

    # The FULL swept curve, pinned exactly (conf grid = {0.0} ∪ observed scores).
    expected = [
        {"conf": 0.0, "tp": 3, "fp": 1, "fn": 0, "count_bias_mean": 0.5, "abs_count_error_mean": 0.5},
        {"conf": 0.3, "tp": 3, "fp": 1, "fn": 0, "count_bias_mean": 0.5, "abs_count_error_mean": 0.5},
        {"conf": 0.6, "tp": 2, "fp": 1, "fn": 1, "count_bias_mean": 0.0, "abs_count_error_mean": 1.0},
        {"conf": 0.9, "tp": 2, "fp": 0, "fn": 1, "count_bias_mean": -0.5, "abs_count_error_mean": 0.5},
    ]
    curve = sweep["curve"]
    assert len(curve) == len(expected)
    for got, exp in zip(curve, expected):
        for k, v in exp.items():
            assert got[k] == pytest.approx(v), f"{k} at conf={exp['conf']}"

    # Derived precision/recall/f1 at the count-unbiased conf (0.6).
    at06 = curve[2]
    assert at06["precision"] == pytest.approx(2 / 3)
    assert at06["recall"] == pytest.approx(2 / 3)
    assert at06["f1"] == pytest.approx(2 / 3)


def test_golden_pick_count_unbiased_and_f1_max():
    recs = _sweep_records()
    sweep = sweep_operating_point(recs, tolerance=0.5 * gt_class_avg_size(recs))
    assert pick_count_unbiased(sweep) == pytest.approx(0.6)
    assert pick_f1_max(sweep) == pytest.approx(0.0)
    # count bias vanishes at the count-unbiased pick, over-counts (+0.5) at the F1-max pick
    assert count_bias_at(sweep, 0.6)["count_bias_mean"] == pytest.approx(0.0)
    assert count_bias_at(sweep, 0.0)["count_bias_mean"] == pytest.approx(0.5)


def test_golden_resolve_operating_point_validated_conf():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_sweep_records("c"),
                                holdout_records=_sweep_records("h"))
    conf = b.get("conf")
    assert conf._raw == pytest.approx(0.6)  # count-unbiased pick
    assert conf.derivation_class == "calibration"
    assert conf.derived_from == "count-unbiased center-match sweep"
    assert conf.validated_vs_gt == "validated_held_out"
    assert conf.dataset_scoped is True
    assert conf.dataset_hash == "h1"
    assert b.is_shippable is True
    assert b.get("max_dets")._raw == 100  # ~1.5x p99 GT/image, floored at 100 for this sparse fixture


# ══════════════════════════════════════════════════════════════════════════
# 2. phenology fraction curve + milestone dates
# ══════════════════════════════════════════════════════════════════════════

_PHENO_SERIES = [
    ("2026-02-10", 0.0),
    ("2026-02-20", 0.10),
    ("2026-03-02", 0.55),
    ("2026-03-12", 0.97),
]


def test_golden_crossing_dates_interpolated():
    assert PH.crossing_date(_PHENO_SERIES, 0.05) == "2026-02-15"  # midway 0.0→0.10 over 10 days
    assert PH.crossing_date(_PHENO_SERIES, 0.50) == "2026-03-01"
    assert PH.crossing_date(_PHENO_SERIES, 0.95) == "2026-03-12"
    assert PH.crossing_date(_PHENO_SERIES, 0.99) is None  # never reached


def test_golden_plant_milestones_shape_and_values():
    ms = PH.plant_milestones(_PHENO_SERIES)
    assert ms == {
        "catkin_05per_date": "2026-02-15",
        "catkin_50per_date": "2026-03-01",
        "catkin_95per_date": "2026-03-12",
        # provisional (breeders to confirm): elongation == the 95% majority crossing
        "catkin_elongation_date": "2026-03-12",
    }


def _write_preds(d: Path, stem: str, lines: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    boxes = [PredBBox(1.0, 1.0, 3.0, 3.0, int(float(ln.split()[0])),
                      confidence=float(ln.split()[1])) for ln in lines]
    json_io.write_detect(d / f"{stem}.json", boxes, 8, 8)


def test_golden_per_plant_phenology_series_and_milestones(tmp_path: Path):
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["0 0.9", "0 0.9", "0 0.9", "1 0.9"])  # 1/4 elongated -> 0.25
    _write_preds(d2, "P1_b", ["1 0.9", "1 0.9", "1 0.9", "0 0.9"])  # 3/4 elongated -> 0.75
    mapping = {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }
    res = PH.per_plant_phenology(
        mapping, {"2026-02-11": str(d1), "2026-03-09": str(d2)}, elongated_class_id=1)

    assert res["elongation_classified"] is True
    assert res["classes_seen"] == [0, 1]
    assert len(res["rows"]) == 1
    row = res["rows"][0]
    assert row["plant_id"] == "P1"
    assert row["accession"] == "acc-9"
    assert row["n_dates"] == 2
    assert row["n_observed_dates"] == 2
    assert row["series"] == [
        {"date": "2026-02-11", "n_total": 4, "n_elongated": 1, "ratio": 0.25},
        {"date": "2026-03-09", "n_total": 4, "n_elongated": 3, "ratio": 0.75},
    ]
    # 0.25 → 0.75 over 26 days: 5% at first date; 50% interpolated; 95% never reached.
    assert row["catkin_05per_date"] == "2026-02-11"
    assert row["catkin_50per_date"] == "2026-02-24"
    assert row["catkin_95per_date"] is None
    assert row["catkin_elongation_date"] is None  # == 95% crossing, which is None here


# ══════════════════════════════════════════════════════════════════════════
# 3. operating_point.json stamp shape + the validated flag path
# ══════════════════════════════════════════════════════════════════════════

_OP_PARAM_KEYS = {"conf", "cross_tile_nms", "tiled", "tile_size", "max_dets"}
_PARAM_PROVENANCE_KEYS = {
    "name", "value", "source", "derivation_class", "derived_from",
    "validated_vs_gt", "dataset_scoped", "dataset_hash", "has_sweep",
}


def _stamp(bundle, *, validated: bool, issues: list[str]) -> dict:
    """Replicate the exact operating_point.json stamp export_predictions writes."""
    return {"operating_point": bundle.to_provenance()["operating_point"],
            "validated": bool(validated), "shippable_issues": issues}


def test_golden_stamp_shape_calibrated_validated():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_sweep_records("c"),
                                holdout_records=_sweep_records("h"))
    stamp = _stamp(b, validated=b.is_shippable, issues=b.shippable_issues())
    assert set(stamp.keys()) == {"operating_point", "validated", "shippable_issues"}
    assert stamp["validated"] is True  # held-out calibration passed
    assert stamp["shippable_issues"] == []
    op = stamp["operating_point"]
    assert set(op.keys()) == _OP_PARAM_KEYS
    for name, prov in op.items():
        assert set(prov.keys()) == _PARAM_PROVENANCE_KEYS
    conf = op["conf"]
    assert conf["value"] == pytest.approx(0.6)
    assert conf["source"] == "derived"
    assert conf["derivation_class"] == "calibration"
    assert conf["validated_vs_gt"] == "validated_held_out"
    assert conf["has_sweep"] is True


def test_golden_stamp_shape_raw_uncalibrated_is_false():
    from tcip_mcp.pipelines.resolution import raw_operating_point

    b = raw_operating_point(conf=0.5, cross_tile_nms=0.3, tiled=True,
                            tile_size=640, max_dets=1000)
    # Raw inference always stamps validated=False (no per-dataset held-out calibration).
    assert b.is_shippable is False
    stamp = _stamp(b, validated=False, issues=[])
    assert set(stamp.keys()) == {"operating_point", "validated", "shippable_issues"}
    assert stamp["validated"] is False
    op = stamp["operating_point"]
    assert set(op.keys()) == _OP_PARAM_KEYS
    conf = op["conf"]
    assert conf["value"] == pytest.approx(0.5)
    assert conf["source"] == "default"
    assert conf["derivation_class"] == "calibration"
    assert conf["validated_vs_gt"] == "false"  # the firewall stamp on an uncalibrated conf
    assert conf["has_sweep"] is False


def test_golden_validated_flag_path_calibrated_no_holdout_is_false():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    # Calibrated but never held-out-measured -> validated=false, not shippable.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_sweep_records("c"))
    assert b.get("conf").validated_vs_gt == "false"
    assert b.is_shippable is False


# ══════════════════════════════════════════════════════════════════════════
# 4. consolidated inference operating-point defaults (W6-R1)
# ══════════════════════════════════════════════════════════════════════════

def test_golden_consolidated_operating_point_defaults():
    # W6-R1 old->new: operating_point.py previously carried a SECOND, divergent copy of the
    # inference operating-point knobs (_DEFAULT_CROSS_TILE_NMS=0.5, _DEFAULT_MAX_DETS=300,
    # _DEFAULT_CONF_PLACEHOLDER=0.5, _DEFAULT_TILE_SIZE=640) so the same model+images gave a
    # different count by entry door. Those private constants are deleted; every operating-point
    # fallback now resolves to resolution.py's single source of truth.
    from tcip_mcp.pipelines import operating_point as OP
    from tcip_mcp.pipelines import resolution as R
    from tcip_mcp.pipelines.inference import generic_predictor as GP
    from tcip_mcp.pipelines.training import evaluation as EV
    from tcip_mcp.tools import training_tools as TT

    # resolution.py — the shared inference operating-point defaults.
    assert R.DEFAULT_CONF == 0.5
    assert R.DEFAULT_NMS_IOU == 0.3
    assert R.DEFAULT_MAX_DETS == 1000
    assert R.DEFAULT_TILE_SIZE == 640
    assert R.DEFAULT_TILED is True

    # operating_point.py — the private _DEFAULT_* copies are gone; the module now imports the
    # shared constants (same objects), proving one source of truth.
    assert not hasattr(OP, "_DEFAULT_CROSS_TILE_NMS")
    assert not hasattr(OP, "_DEFAULT_MAX_DETS")
    assert not hasattr(OP, "_DEFAULT_CONF_PLACEHOLDER")
    assert not hasattr(OP, "_DEFAULT_TILE_SIZE")
    assert OP.DEFAULT_MAX_DETS is R.DEFAULT_MAX_DETS
    assert OP.DEFAULT_NMS_IOU is R.DEFAULT_NMS_IOU
    assert OP.DEFAULT_TILE_SIZE is R.DEFAULT_TILE_SIZE

    # The consolidated fallbacks flow through to a resolved bundle with no calibration/overrides:
    # cross_tile_nms == the shared NMS-IoU (was 0.5), max_dets == the shared cap (was 300).
    b = OP.resolve_operating_point("catkin", dataset_hash=None)
    assert b.get("cross_tile_nms")._raw == R.DEFAULT_NMS_IOU  # 0.3, was 0.5
    assert b.get("max_dets")._raw == R.DEFAULT_MAX_DETS        # 1000, was 300
    assert b.get("tile_size")._raw == R.DEFAULT_TILE_SIZE
    assert b.get("tiled")._raw is R.DEFAULT_TILED

    # generic_predictor tile/NMS fallbacks now reference the shared constants (was a hardcoded 224/0.3).
    gp_sig = inspect.signature(GP.GenericPredictor.predict_tiled)
    assert gp_sig.parameters["tile_size"].default == R.DEFAULT_TILE_SIZE
    assert gp_sig.parameters["global_nms_iou"].default == R.DEFAULT_NMS_IOU

    # training_tools.evaluate_model — the eval-surface max_dets default is a separate knob (not
    # the inference operating point) and is unchanged by R1.
    ev_sig = inspect.signature(TT.evaluate_model)
    assert ev_sig.parameters["max_dets"].default == 100
    assert ev_sig.parameters["iou_threshold"].default == 0.5
    assert ev_sig.parameters["global_nms_iou"].default == 0.3
    assert ev_sig.parameters["conf_threshold"].default == 0.5

    # evaluation.py surfaces — pinned so a metrics-default change is visible too.
    coco_sig = inspect.signature(EV.coco_detection_metrics)
    assert coco_sig.parameters["conf_threshold"].default == 0.25
    assert coco_sig.parameters["iou_threshold"].default == 0.5
    assert coco_sig.parameters["max_dets"].default == 100
    ff_sig = inspect.signature(EV.run_full_frame_evaluation)
    assert ff_sig.parameters["global_nms_iou"].default == 0.3
    assert ff_sig.parameters["max_dets"].default == 1000
    assert EV.DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.2}


# ══════════════════════════════════════════════════════════════════════════
# 5. IoU-matching eval metrics at iou_threshold=0.5 (W6-R3 criterion-change target)
# ══════════════════════════════════════════════════════════════════════════

def _iou_records():
    """Discriminating fixture: an exact match, a 2px-shifted match (IoU~0.68 -> hit@0.5, miss@0.75),
    a spurious FP, and a whole image of GT with no predictions (FN)."""
    def gt(x, y, w, h, cid=1):
        return {"category_id": cid, "bbox": [float(x), float(y), float(w), float(h)]}

    def dt(x, y, w, h, score, cid=1):
        return {"category_id": cid, "bbox": [float(x), float(y), float(w), float(h)], "score": score}

    return [
        {"width": 100, "height": 100,
         "gt": [gt(10, 10, 20, 20), gt(60, 60, 20, 20)],
         "dt": [dt(10, 10, 20, 20, 0.95), dt(62, 62, 20, 20, 0.85), dt(90, 5, 8, 8, 0.6)]},
        {"width": 100, "height": 100, "gt": [gt(40, 40, 20, 20)], "dt": []},
    ]


def test_golden_coco_metrics_at_iou_050():
    m = coco_detection_metrics(_iou_records(), iou_threshold=0.5,
                               conf_threshold=0.25, max_dets=100)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["n_gt"] == 3
    assert m["n_pred"] == 3
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["map50"] == pytest.approx(0.6633663366336634, abs=1e-9)
    assert m["map75"] == pytest.approx(0.33663366336633654, abs=1e-9)
    assert m["map"] == pytest.approx(0.46732673267326735, abs=1e-9)
    assert m["map_convention"] == "coco_ap100"
    counts = {int(c["image_id"]): (c["tp"], c["fp"], c["fn"]) for c in m["per_image_counts"]}
    assert counts == {1: (2, 1, 0), 2: (0, 0, 1)}


def test_golden_coco_matching_is_iou_threshold_sensitive():
    # The SAME predictions score differently at 0.75 — proof the criterion is IoU-thresholded
    # today (what W6-R3 replaces with a derived center-match tolerance for the count).
    m = coco_detection_metrics(_iou_records(), iou_threshold=0.75,
                               conf_threshold=0.25, max_dets=100)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)


# ══════════════════════════════════════════════════════════════════════════
# 6. compute_phenology gate behavior
# ══════════════════════════════════════════════════════════════════════════

def _write_op_sidecar(d: Path, *, validated: bool, conf: float = 0.4) -> None:
    """The operating_point.json stamp export_predictions writes beside a bucket's labels — the
    on-disk validity compute_phenology now reconciles against (W1-R3)."""
    ref = "validated_held_out" if validated else "false"
    (d / "operating_point.json").write_text(json.dumps({
        "validated": validated,
        "operating_point": {"conf": {"value": conf, "validated_vs_gt": ref}},
    }), encoding="utf-8")


def _pheno_setup(tmp_path: Path, *, elongated: bool, op_validated: bool | None = None):
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["0 0.9"])
    _write_preds(d2, "P1_b", ["1 0.9" if elongated else "0 0.9"])
    if op_validated is not None:
        _write_op_sidecar(d1, validated=op_validated)
        _write_op_sidecar(d2, validated=op_validated)
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps({
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }), encoding="utf-8")
    return mapping_path, d1, d2


def test_golden_compute_phenology_refuses_without_elongation_class(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=False)  # no class-1 anywhere
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        positive_class_id=1,
    )
    assert "error" in res
    assert res["elongation_classified"] is False
    assert res["classes_seen"] == [0]
    assert not out_csv.exists()


def test_golden_compute_phenology_requires_both_validated_flags(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=True)  # no operating_point.json sidecars
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    # Elongation class present, but neither validity flag supplied -> gate refuses.
    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        positive_class_id=1,
    )
    assert "error" in res and "requires BOTH" in res["error"]
    assert not out_csv.exists()


def test_golden_compute_phenology_asserted_op_validity_floored_by_missing_sidecar(tmp_path: Path):
    # W1-R3 old->new: previously a caller string operating_point_validated='validated_held_out' opened
    # the gate on its own (T5-3 hole). Now the count validity is read from each bucket's
    # operating_point.json and floored against the assertion — with NO sidecar the curve floors to
    # false and the gate refuses, even though the caller asserted validated.
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=True)  # no sidecars written
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        positive_class_id=1,
        classifier_validated="validated_held_out",
        operating_point_conf=0.4,
        operating_point_validated="validated_held_out",  # asserted, but unbacked on disk
    )
    assert "error" in res
    assert res["operating_point_validated"] == "false"
    assert res["operating_point_missing_sidecars"]  # both buckets flagged
    assert not out_csv.exists()


def test_golden_compute_phenology_delivers_when_both_validated(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    # W1-R3 old->new: delivery now also requires the count operating point to be validated ON DISK.
    # The buckets carry an operating_point.json stamped validated_held_out (as a calibrated
    # export_predictions writes), which the gate reconciles against the caller assertion.
    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=True, op_validated=True)
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        positive_class_id=1,
        classifier_validated="validated_held_out",
        operating_point_conf=0.4,
        operating_point_validated="validated_held_out",
    )
    assert "error" not in res
    assert res["elongation_classifier_validated"] == "validated_held_out"
    assert res["operating_point_validated"] == "validated_held_out"
    assert res["columns"] == PH.PHENOLOGY_CSV_COLUMNS
    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == PH.PHENOLOGY_CSV_COLUMNS
    assert rows[0]["operating_point_conf"] == "0.4"
    assert rows[0]["operating_point_validated"] == "validated_held_out"
    assert rows[0]["elongation_classifier_validated"] == "validated_held_out"

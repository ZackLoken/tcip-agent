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

import inspect
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology as PH  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    coco_detection_metrics,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    sweep_operating_point,
)
from tests._trait_fixtures import CATKIN  # noqa: E402
from tests._dense_op_fixtures import dense_records  # noqa: E402

# Round 10 (2026-07-29): no built-in traits — seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so resolve_operating_point("catkin", ...) /
# compute_phenology(trait="catkin", ...) keep resolving by default.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")

_N_IMAGES = 20
_OBJECTS_PER_IMAGE = 80


def _good_cal_holdout(*, shift: float = 5.0):
    """A dense, realistic (K2 rule 17) reference: a good detector with one low-conf spurious
    detection per image — the count-unbiased pick lands at the high, correct-match score (0.9)
    once that low-conf FP is filtered out, with zero bias/dispersion on the holdout."""
    miss = [0] * _N_IMAGES
    fp = [1] * _N_IMAGES
    cal = dense_records(n_images=_N_IMAGES, objects_per_image=_OBJECTS_PER_IMAGE, id_prefix="c",
                        miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    hold = dense_records(n_images=_N_IMAGES, objects_per_image=_OBJECTS_PER_IMAGE, id_prefix="h",
                         shift=shift, miss_pattern=miss, fp_pattern=fp, score=0.9, fp_score=0.05)
    return cal, hold


# ── shared fixture helpers ────────────────────────────────────────────────

def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]  # xywh centered at (cx, cy)


def _ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _sweep_records(idp="c", *, shift: float = 0.0):
    """Two images engineered so count-unbiased conf (0.6) != F1-max conf (0.0).

    Image A: 1 GT; correct det @0.9 + a spurious far det @0.6.
    Image B: 2 GT; correct det @0.9 + a hesitant-but-correct det @0.3.

    ``shift`` offsets every GT box's center by that many px (well inside the ~10px center-match
    tolerance derived from these boxes), leaving the detections in place. Used to give the holdout
    fixture (K1) genuinely different GT content from calibration's — a holdout differing from
    calibration only by ``image_id`` is byte-identical CONTENT and the content-overlap gate now
    (correctly) refuses it; see ``test_golden_duplicate_content_holdout_is_false`` below, which
    pins exactly that refusal on the OLD (shift=0) fixture pair.
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_ann(100 + shift, 100)],
         "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_ann(100 + shift, 100), _ann(200 + shift, 200)],
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
    by_conf = {round(c["conf"], 6): c for c in sweep["curve"]}
    assert by_conf[0.6]["count_bias_mean"] == pytest.approx(0.0)
    assert by_conf[0.0]["count_bias_mean"] == pytest.approx(0.5)


def test_golden_resolve_operating_point_validated_conf():
    # K2 (Fix D): resolve_operating_point now fails closed without an asserted staged_conf_floor,
    # and rule 17 requires a dense, realistic reference to exercise the holdout gate — the old
    # 2-image sparse fixture no longer suffices (its per-image variance now trips Fix C's
    # equivalence criterion; see test_golden_duplicate_content_holdout_is_false below for what a
    # sparse fixture STILL correctly refuses).
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = _good_cal_holdout()
    # tiled=False: this golden is about conf-calibration shippability, not tiling (K10 — tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf._raw == pytest.approx(0.9)  # count-unbiased pick: bias vanishes once the low-conf FP drops
    assert conf.requires_validation is True and conf.validation_kind == "annotations"
    assert conf.derived_from == "count-unbiased center-match sweep"
    assert conf.validated_against == "held_out_annotations"
    assert conf.dataset_scoped is True
    assert conf.dataset_hash == "h1"
    assert b.is_shippable is True
    assert b.get("max_dets")._raw == 120  # ~1.5x p99 GT/image (80/image here)
    assert conf.sweep["content_overlap_frac"] == pytest.approx(0.0)  # genuinely distinct holdout (K1)
    assert conf.sweep["failures"] == []


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
    # K4: the return is now a Crossing record (date + evidentiary bound), not a bare string.
    assert PH.crossing_date(_PHENO_SERIES, 0.05).date == "2026-02-15"  # midway 0.0→0.10, 10 days
    assert PH.crossing_date(_PHENO_SERIES, 0.50).date == "2026-03-01"
    assert PH.crossing_date(_PHENO_SERIES, 0.95).date == "2026-03-12"
    assert PH.crossing_date(_PHENO_SERIES, 0.95).bound == "interpolated"  # 0.97 observed, not 0.95 exactly
    assert PH.crossing_date(_PHENO_SERIES, 0.97).bound == "exact"
    # never reached within the observed window -> right-censored at the LAST observed date, not a
    # bare None (round 12, 2026-07-29): distinguishable from "no observations at all".
    c99 = PH.crossing_date(_PHENO_SERIES, 0.99)
    assert c99.date == "2026-03-12"
    assert c99.bound == "right_censored"
    assert PH.crossing_date([], 0.99) is None  # no observations at all -> still None


def test_golden_plant_milestones_shape_and_values():
    ms = PH.plant_milestones(_PHENO_SERIES, CATKIN)
    assert ms["catkin_05per_date"] == "2026-02-15"
    assert ms["catkin_50per_date"] == "2026-03-01"
    assert ms["catkin_95per_date"] == "2026-03-12"
    # provisional (breeders to confirm): elongation == the 95% majority crossing
    assert ms["catkin_elongation_date"] == "2026-03-12"


_ID_MAP = {"dormant": 0, "elongated": 1}


def _write_preds(d: Path, stem: str, subjects: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9) for s in subjects]
    json_io.write_annotations(d / f"{stem}.json", anns, 8, 8)


def _write_id_map_sidecar(d: Path, id_map: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "operating_point.json").write_text(json.dumps({"id_map": id_map}), encoding="utf-8")


def test_golden_per_plant_phenology_series_and_milestones(tmp_path: Path):
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["dormant", "dormant", "dormant", "elongated"])  # 1/4 -> 0.25
    _write_id_map_sidecar(d1, _ID_MAP)
    _write_preds(d2, "P1_b", ["elongated", "elongated", "elongated", "dormant"])  # 3/4 -> 0.75
    _write_id_map_sidecar(d2, _ID_MAP)
    mapping = {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }
    res = PH.per_plant_phenology(
        mapping, {"2026-02-11": str(d1), "2026-03-09": str(d2)},
        positive_class_name="elongated", spec=CATKIN)

    # K4/K5: both buckets are fully classified, so the fraction IS produced and delivered.
    assert res["elongation_classified"] is True
    assert len(res["rows"]) == 1
    row = res["rows"][0]
    assert row["plant_id"] == "P1"
    assert row["accession"] == "acc-9"
    assert row["n_dates"] == 2
    assert row["n_observed_dates"] == 2
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert [s["n_total"] for s in row["series"]] == [4, 4]
    assert [s["n_positive"] for s in row["series"]] == [1, 3]
    assert [s["ratio"] for s in row["series"]] == [0.25, 0.75]
    # 0.25 -> 0.75 crosses 50% at the midpoint between the two dates.
    assert row["catkin_50per_date"] == "2026-02-24"


# ══════════════════════════════════════════════════════════════════════════
# 3. operating_point.json stamp shape + the validated flag path
# ══════════════════════════════════════════════════════════════════════════

# raw_operating_point (no trait/dataset resolution) carries no localization_tolerance_frac; only
# resolve_operating_point (trait-aware) derives and stamps it.
_OP_PARAM_KEYS = {"conf", "cross_tile_nms", "tiled", "tile_size", "max_dets"}
_RESOLVED_OP_PARAM_KEYS = _OP_PARAM_KEYS | {"localization_tolerance_frac"}
_PARAM_PROVENANCE_KEYS = {
    "name", "value", "source", "derived_from",
    "requires_validation", "validation_kind", "validated_against",
    "dataset_scoped", "dataset_hash", "capture_scoped", "capture_id", "has_sweep",
}


def _stamp(bundle, *, validated: bool, issues: list[str]) -> dict:
    """Replicate the exact operating_point.json stamp export_predictions writes."""
    return {"operating_point": bundle.to_provenance()["operating_point"],
            "validated": bool(validated), "shippable_issues": issues}


def test_golden_stamp_shape_calibrated_validated():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = _good_cal_holdout()
    # tiled=False: this golden is about conf-calibration shippability, not tiling (K10 — tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                tiled=False, staged_conf_floor=0.01)
    stamp = _stamp(b, validated=b.is_shippable, issues=b.shippable_issues())
    assert set(stamp.keys()) == {"operating_point", "validated", "shippable_issues"}
    assert stamp["validated"] is True  # held-out calibration passed
    assert stamp["shippable_issues"] == []
    op = stamp["operating_point"]
    assert set(op.keys()) == _RESOLVED_OP_PARAM_KEYS
    for name, prov in op.items():
        assert set(prov.keys()) == _PARAM_PROVENANCE_KEYS
    conf = op["conf"]
    assert conf["value"] == pytest.approx(0.9)
    assert conf["source"] == "derived"
    assert conf["requires_validation"] is True
    assert conf["validation_kind"] == "annotations"
    assert conf["validated_against"] == "held_out_annotations"
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
    assert conf["requires_validation"] is True
    assert conf["validation_kind"] == "annotations"
    assert conf["validated_against"] == "false"  # the firewall stamp on an uncalibrated conf
    assert conf["has_sweep"] is False


def test_golden_validated_flag_path_calibrated_no_holdout_is_false():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    # Calibrated but never held-out-measured -> validated=false, not shippable.
    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_sweep_records("c"))
    assert b.get("conf").validated_against == "false"
    assert b.is_shippable is False


def test_golden_duplicate_content_holdout_is_false():
    """K1's delivered number, old->new: the SAME fixture pair the two goldens above used before
    this cluster (identical GT content, differing only by ``image_id`` prefix — the OLD
    ``_sweep_records("c")``/``_sweep_records("h")`` call with no ``shift``) used to stamp
    ``VALIDATED_HELD_OUT``/shippable=True. A byte-identical-content holdout can't function as an
    independent check, so K1 adds the content-overlap gate and this pair now stamps
    ``false``/shippable=False instead — pinned here exactly as before/after this cluster's change.
    """
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    b = resolve_operating_point("catkin", dataset_hash="h1",
                                calibration_records=_sweep_records("c"),
                                holdout_records=_sweep_records("h"))
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert b.is_shippable is False
    assert conf.sweep["content_overlap_frac"] == pytest.approx(1.0)


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

    # training_tools.evaluate_model — K10 finding 2: max_dets is no longer a plain 100 default
    # shared by both eval regimes via a rescuing ">100 else 1000" sentinel (which collided with
    # _max_dets_from_density's own floor of exactly 100). The signature default is now the honest
    # None sentinel; TRAP 5 (cluster-map.md) requires pinning what each regime RESOLVES it to for a
    # no-arg caller, not just the unspecified shape — see the two resolved-value assertions below.
    ev_sig = inspect.signature(TT.evaluate_model)
    assert ev_sig.parameters["max_dets"].default is None
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
    # K10 finding 1: tile_size/overlap are no longer pinned constants (640/0.2) — an honest None
    # sentinel resolved from the checkpoint's persisted geometry (or refused) by resolve_tile_geometry.
    assert ff_sig.parameters["tile_size"].default is None
    assert ff_sig.parameters["overlap"].default is None
    # max_dets itself keeps its own 1000 default at this layer (the delivery-grade default,
    # distinct from evaluate_model's None sentinel one layer up).
    assert ff_sig.parameters["max_dets"].default == 1000
    assert EV.DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.2}


def test_golden_k10_evaluate_model_resolves_max_dets_per_regime_when_unset():
    """TRAP 5 counterpart assertion (cluster-map.md): a signature-shape golden alone cannot see
    what a no-arg caller's max_dets actually RESOLVES to per regime — without this, the golden set
    would ratify "the default is unspecified" rather than pin the two real behaviors (1000 on the
    delivery-gating regime, 100 on the tile-level/diagnostic regime)."""
    from tcip_mcp.pipelines import resolution as R
    from tcip_mcp.pipelines.training import evaluation as EV
    from tcip_mcp.tools import training_tools as TT

    captured: dict = {}

    def _fake_gate(ckpt, images_dir, labels_dir, output_dir, **kw):
        captured["gate_max_dets"] = kw.get("max_dets")
        return {"eval_regime": "full-frame-tiled-inference"}

    def _fake_diagnostic(ckpt, loader, device, task, output_dir, **kw):
        captured["diagnostic_max_dets"] = kw.get("max_dets")
        return {"tiled": False, "eval_regime": "tile-level"}

    orig_gate = EV.run_full_frame_evaluation
    orig_diag = EV.run_test_evaluation
    try:
        EV.run_full_frame_evaluation = _fake_gate
        EV.run_test_evaluation = _fake_diagnostic
        import tempfile
        from pathlib import Path as _Path

        from PIL import Image
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox

        with tempfile.TemporaryDirectory() as td:
            tmp = _Path(td)
            images_dir, labels_dir = tmp / "images", tmp / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            Image.new("RGB", (64, 64)).save(images_dir / "a.png")
            json_io.write_annotations(str(labels_dir / "a.json"),
                                      [Annotation(subject="catkin", geometry=BBox(5, 5, 20, 20))],
                                      64, 64)
            ckpt = tmp / "m.pt"
            ckpt.write_bytes(b"x")

            TT.evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                              subject="catkin", use_tiled_inference=True)
            TT.evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                              subject="catkin")
    finally:
        EV.run_full_frame_evaluation = orig_gate
        EV.run_test_evaluation = orig_diag

    assert captured["gate_max_dets"] == R.DEFAULT_MAX_DETS == 1000
    assert captured["diagnostic_max_dets"] == 100


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

def _write_op_sidecar(d: Path, *, validated: bool, conf: float = 0.4, id_map: dict | None = None,
                      checkpoint_sha256: str | None = "deadbeef" * 8,
                      experiment_id: str | None = "exp-golden") -> None:
    """The operating_point.json stamp export_predictions writes beside a bucket's labels — the
    on-disk validity compute_phenology reconciles against (K3), including id_map (K4/K5) and
    producer identity (K12 finding 7: the real writer always stamps checkpoint_sha256/experiment_id
    at the top level — a fixture that omitted them blessed a shape the platform never produces)."""
    ref = "held_out_annotations" if validated else "false"
    d.mkdir(parents=True, exist_ok=True)
    (d / "operating_point.json").write_text(json.dumps({
        "validated": validated,
        "operating_point": {"conf": {"value": conf, "validated_against": ref}},
        "id_map": id_map,
        "checkpoint_sha256": checkpoint_sha256,
        "experiment_id": experiment_id,
    }), encoding="utf-8")


def _write_classifier_sidecar(d: Path, *, validated: bool, trait: str | None = "catkin") -> None:
    ref = "held_out_annotations" if validated else "false"
    d.mkdir(parents=True, exist_ok=True)
    (d / "classifier_operating_point.json").write_text(json.dumps({
        "validated": validated,
        "operating_point": {"classifier": {"value": "elongated", "validated_against": ref}},
        "trait": trait,
    }), encoding="utf-8")


def _pheno_setup(tmp_path: Path, *, elongated: bool, op_validated: bool | None = None):
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    id_map = {"dormant": 0, "elongated": 1} if elongated else {"catkin": 0}
    _write_preds(d1, "P1_a", ["catkin"] if not elongated else ["dormant"])
    _write_preds(d2, "P1_b", ["elongated"] if elongated else ["catkin"])
    if op_validated is not None:
        _write_op_sidecar(d1, validated=op_validated, id_map=id_map)
        _write_op_sidecar(d2, validated=op_validated, id_map=id_map)
    else:
        # count-operating-point sidecar still needs an id_map for the coverage rule even when its
        # own validity isn't the thing under test — a bucket with NO sidecar at all is the
        # "no operating_point.json" case, tested separately.
        pass
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps({
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }), encoding="utf-8")
    return mapping_path, d1, d2


def test_golden_compute_phenology_refuses_without_elongation_class(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=False, op_validated=True)  # bare detector
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert not out_csv.exists()


def test_golden_compute_phenology_requires_both_validated_flags(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=True)  # no operating_point.json sidecars
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert not out_csv.exists()


def test_golden_compute_phenology_asserted_op_validity_floored_by_missing_sidecar(tmp_path: Path):
    # K3: an asserted validity string is floored by the on-disk sidecar's real (false) state —
    # never trusted. The predictions ARE classified (a real id_map is on disk), but the sidecar's
    # own conf.validated_against is "false" — a caller asserting "held_out_annotations" cannot override it.
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=True, op_validated=False)
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        operating_point_conf=0.4,
        operating_point_validated="held_out_annotations",  # asserted, but unbacked on disk
    )
    assert "error" in res
    assert res["operating_point_validated"] == "false"
    assert not out_csv.exists()


def test_golden_compute_phenology_delivers_when_both_validated(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    # K4/K5: the elongated fraction IS now produced, so a fully-validated call (classifier + count
    # operating point both validated on disk) delivers a real bloom CSV.
    mapping_path, d1, d2 = _pheno_setup(tmp_path, elongated=True, op_validated=True)
    _write_classifier_sidecar(d1, validated=True)
    out_csv = tmp_path / "out" / "catkin_phenology.csv"
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
        operating_point_conf=0.4,
        operating_point_validated="held_out_annotations",
    )
    assert "error" not in res, res
    assert res["elongation_classified"] is True
    assert out_csv.exists()

    # K12 finding 7: a fully-validated delivery must carry real producer identity, not blank
    # producer columns — no test previously asserted compute_phenology actually populates them.
    import csv as _csv
    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert rows
    assert all(row["producer_model_sha256"] == "deadbeef" * 8 for row in rows)
    assert all(row["producer_experiment_id"] == "exp-golden" for row in rows)

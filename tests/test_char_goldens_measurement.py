"""Characterization goldens: pin the current numeric behavior of the measurement
and provenance rails that later work will touch.

These are deliberately *exact*: they assert the numbers today's code produces on tiny
deterministic fixtures, so a later semantic change (a different conf pick, a re-defined
localization criterion, a consolidated default, a re-shaped stamp) fails loudly instead of
sliding through silently. They are not aspirational: a golden turning red is the signal to
update it *deliberately* alongside the change that moved the number.

Rails pinned here (one section each):
  1. conf operating-point sweep + count-unbiased pick + resolve_operating_point
  2. phenology fraction curve + milestone dates (crossing_date / plant_milestones / per_plant_phenology)
  3. operating_point.json stamp shape + the validated flag path (calibrated vs raw)
  4. the currently divergent NMS / max_dets defaults across the three modules
  5. IoU-matching eval metrics at iou_threshold=0.5 (current criterion, to be replaced by a
     derived center-match tolerance)
  6. deliver_phenology_milestones gate behavior (refuses without an open class; requires validated flags)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

import tcip_store as ts  # noqa: E402
from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology as PH  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    coco_detection_metrics,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    derive_operating_point_curve,
)
from tests._binding_fixtures import (  # noqa: E402
    producer_checkpoint_sha256,
    record_producing_run,
    write_bound_sidecar,
)
from tests._trait_fixtures import BUD_OPENING  # noqa: E402
from tests._dense_op_fixtures import good_cal_holdout  # noqa: E402

# seed_bud_operationalization writes the spec plus the confirmed crossing record this root needs.
pytestmark = pytest.mark.usefixtures("seed_bud_operationalization")


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
    fixture genuinely different GT content from calibration's: a holdout differing from
    calibration only by ``image_id`` is byte-identical content and the content-overlap gate now
    (correctly) refuses it; see ``test_golden_duplicate_content_holdout_is_false`` below, which
    pins exactly that refusal on the shift=0 fixture pair.
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

    sweep = derive_operating_point_curve(recs, tolerance=tol)
    assert sweep["tolerance"] == pytest.approx(10.0)
    assert sweep["class_id"] is None

    # The full swept curve, pinned exactly (conf grid = {0.0} ∪ observed scores).
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
    sweep = derive_operating_point_curve(recs, tolerance=0.5 * gt_class_avg_size(recs))
    assert pick_count_unbiased(sweep) == pytest.approx(0.6)
    assert pick_f1_max(sweep) == pytest.approx(0.0)
    # count bias vanishes at the count-unbiased pick, over-counts (+0.5) at the F1-max pick
    by_conf = {round(c["conf"], 6): c for c in sweep["curve"]}
    assert by_conf[0.6]["count_bias_mean"] == pytest.approx(0.0)
    assert by_conf[0.0]["count_bias_mean"] == pytest.approx(0.5)


def test_golden_resolve_operating_point_validated_conf():
    # resolve_operating_point fails closed without an asserted staged_conf_floor,
    # and requires a dense, realistic reference to exercise the holdout gate: a
    # 2-image sparse fixture no longer suffices (its per-image variance trips the
    # equivalence criterion; see test_golden_duplicate_content_holdout_is_false below for what a
    # sparse fixture still correctly refuses).
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    # tiled=False: this golden is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf._raw == pytest.approx(0.9)  # count-unbiased pick: bias vanishes once the low-conf FP drops
    assert conf.requires_validation is True and conf.validation_kind == "annotations"
    assert conf.derived_from == "count-unbiased center-match curve"
    assert conf.validated_against == "held_out_annotations"
    assert conf.dataset_scoped is True
    assert conf.dataset_hash == "h1"
    assert b.is_shippable is True
    assert b.get("max_dets")._raw == 120  # ~1.5x p99 GT/image (80/image here)
    assert conf.gate_evidence["content_overlap_frac"] == pytest.approx(0.0)  # genuinely distinct holdout
    assert conf.gate_evidence["failures"] == []


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
    # The return is a Crossing record (date + evidentiary bound), not a bare string.
    assert PH.crossing_date(_PHENO_SERIES, 0.05).date == "2026-02-15"  # midway 0.0→0.10, 10 days
    assert PH.crossing_date(_PHENO_SERIES, 0.50).date == "2026-03-01"
    assert PH.crossing_date(_PHENO_SERIES, 0.95).date == "2026-03-12"
    assert PH.crossing_date(_PHENO_SERIES, 0.95).bound == "interpolated"  # 0.97 observed, not 0.95 exactly
    assert PH.crossing_date(_PHENO_SERIES, 0.97).bound == "exact"
    # never reached within the observed window -> right-censored at the last observed date, not a
    # bare None: distinguishable from "no observations at all".
    c99 = PH.crossing_date(_PHENO_SERIES, 0.99)
    assert c99.date == "2026-03-12"
    assert c99.bound == "right_censored"
    assert PH.crossing_date([], 0.99) is None  # no observations at all -> still None


def test_golden_plant_milestones_shape_and_values():
    ms = PH.plant_milestones(_PHENO_SERIES, BUD_OPENING)
    assert ms["bud_05per_date"] == "2026-02-15"
    assert ms["bud_50per_date"] == "2026-03-01"
    assert ms["bud_95per_date"] == "2026-03-12"
    # crossing-unconfirmed (breeders to confirm): the majority alias == the 95% crossing
    assert ms["bud_majority_date"] == "2026-03-12"


_ID_MAP = {"closed": 0, "open": 1}


def _write_preds(d: Path, stem: str, subjects: list[str], *, attribute: str | None = "opening",
                 object_subject: str = "bud") -> None:
    d.mkdir(parents=True, exist_ok=True)
    if attribute is None:
        anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9)
                for s in subjects]
    else:
        anns = [Annotation(subject=object_subject, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9,
                           attributes={attribute: s}) for s in subjects]
    json_io.write_annotations(d / f"{stem}.json", anns, 8, 8)


def _write_id_map_sidecar(d: Path, id_map: dict, *, subject: str = "bud",
                          attribute: str | None = "opening") -> None:
    from tcip_mcp.pipelines.resolution import write_sidecar

    write_sidecar(d, {"id_map": id_map, "subject": subject, "attribute": attribute},
                 "operating_point")


def test_golden_per_plant_phenology_series_and_milestones(tmp_path: Path):
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    _write_preds(d1, "P1_a", ["closed", "closed", "closed", "open"])  # 1/4 -> 0.25
    _write_id_map_sidecar(d1, _ID_MAP)
    _write_preds(d2, "P1_b", ["open", "open", "open", "closed"])  # 3/4 -> 0.75
    _write_id_map_sidecar(d2, _ID_MAP)
    mapping = {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }
    res = PH.per_plant_phenology(
        mapping, {"2026-02-11": str(d1), "2026-03-09": str(d2)},
        positive_class_name="open", spec=BUD_OPENING)

    # Both buckets are fully classified, so the fraction is produced and delivered.
    assert res["positive_class_assessed"] is True
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
    assert row["bud_50per_date"] == "2026-02-24"


# ══════════════════════════════════════════════════════════════════════════
# 3. operating_point.json stamp shape + the validated flag path
# ══════════════════════════════════════════════════════════════════════════

# raw_operating_point (no trait/dataset resolution) carries no localization_tolerance_frac; only
# resolve_operating_point (trait-aware) derives and stamps it.
_OP_PARAM_KEYS = {"conf", "cross_tile_nms", "tiled", "tile_size", "max_dets"}
_RESOLVED_OP_PARAM_KEYS = _OP_PARAM_KEYS | {"localization_tolerance_frac", "count_objective"}
_PARAM_PROVENANCE_KEYS = {
    "name", "value", "source", "derived_from",
    "requires_validation", "validation_kind", "validated_against",
    "dataset_scoped", "dataset_hash", "capture_scoped", "capture_id", "has_gate_evidence",
}


def _stamp(bundle, *, validated: bool, issues: list[str]) -> dict:
    """The three fields of a bucket's stamp this golden pins, not the whole stamp: what a resolved
    bundle contributes to it. The provenance and the pointer at the record behind a validated claim
    are `operating_point_stamp`'s own, checked where that constructor is."""
    return {"operating_point": bundle.to_provenance()["operating_point"],
            "validated": bool(validated), "shippable_issues": issues}


def test_golden_stamp_shape_calibrated_validated():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    # tiled=False: this golden is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1",
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
    assert conf["has_gate_evidence"] is True


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
    assert conf["has_gate_evidence"] is False


def test_golden_validated_flag_path_calibrated_no_holdout_is_false():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    # Calibrated but never held-out-measured -> validated=false, not shippable.
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=_sweep_records("c"))
    assert b.get("conf").validated_against == "false"
    assert b.is_shippable is False


def test_golden_content_shared_holdout_is_false():
    """A byte-identical-content holdout can't function as an
    independent check: the same fixture pair the two goldens above use (identical GT content,
    differing only by ``image_id`` prefix, ``_sweep_records("c")``/``_sweep_records("h")`` with no
    ``shift``) must stamp ``false``/shippable=False, gated by the content-overlap check.
    """
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=_sweep_records("c"),
                                holdout_records=_sweep_records("h"))
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert b.is_shippable is False
    assert conf.gate_evidence["content_overlap_frac"] == pytest.approx(1.0)
    assert "content_shared_with_calibration" in conf.gate_evidence["failures"]


# ══════════════════════════════════════════════════════════════════════════
# 4. consolidated inference operating-point defaults
# ══════════════════════════════════════════════════════════════════════════

def test_golden_consolidated_operating_point_defaults():
    # operating_point.py must not carry a second, divergent copy of the inference operating-point
    # knobs (a second copy would let the same model+images give a different count by entry door).
    from tcip_mcp.pipelines import operating_point as OP
    from tcip_mcp.pipelines import resolution as R
    from tcip_mcp.pipelines.inference import generic_predictor as GP
    from tcip_mcp.pipelines.training import eval_runners as runners
    from tcip_mcp.pipelines.training import evaluation as EV
    from tcip_mcp.tools import training_tools as TT

    # resolution.py: the shared inference operating-point defaults. tile_size/tiled carry no such
    # shared fallback constant at all (a caller derives/states them explicitly), nothing to pin here.
    assert R.DEFAULT_CONF == 0.5
    assert R.DEFAULT_NMS_IOU == 0.3
    assert R.DEFAULT_MAX_DETS == 1000
    assert not hasattr(R, "DEFAULT_TILE_SIZE")
    assert not hasattr(R, "DEFAULT_TILED")

    # operating_point.py: the private _DEFAULT_* copies are gone; the module now imports the
    # shared constants (same objects), proving one source of truth.
    assert not hasattr(OP, "_DEFAULT_CROSS_TILE_NMS")
    assert not hasattr(OP, "_DEFAULT_MAX_DETS")
    assert not hasattr(OP, "_DEFAULT_CONF_PLACEHOLDER")
    assert not hasattr(OP, "_DEFAULT_TILE_SIZE")
    assert not hasattr(OP, "DEFAULT_TILE_SIZE")
    assert OP.DEFAULT_MAX_DETS is R.DEFAULT_MAX_DETS
    assert OP.DEFAULT_NMS_IOU is R.DEFAULT_NMS_IOU

    # The consolidated fallbacks flow through a resolved bundle with no calibration/overrides.
    # tile_size has no fallback to flow through at all here (no explicit/derived basis): None.
    b = OP.resolve_operating_point("bud_opening", tiled=True, dataset_hash=None)
    assert b.get("cross_tile_nms")._raw == R.DEFAULT_NMS_IOU  # 0.3, was 0.5
    assert b.get("max_dets")._raw == R.DEFAULT_MAX_DETS        # 1000, was 300
    assert b.get("tile_size")._raw is None
    assert b.get("tiled")._raw is True

    # generic_predictor's own tiling primitive fabricates no tile_size default either (None,
    # caller must resolve a real basis first); NMS still shares the platform default.
    gp_sig = inspect.signature(GP.GenericPredictor.predict_tiled)
    assert gp_sig.parameters["tile_size"].default is None
    assert gp_sig.parameters["global_nms_iou"].default == R.DEFAULT_NMS_IOU

    # training_tools.evaluate_model: max_dets is no longer a plain 100 default
    # shared by both eval regimes via a rescuing ">100 else 1000" sentinel (which collided with
    # _max_dets_from_density's own floor of exactly 100). The signature default is now the honest
    # None sentinel; what each regime resolves it to for a
    # no-arg caller must be pinned too, not just the unspecified shape; see the two resolved-value
    # assertions below.
    ev_sig = inspect.signature(TT.evaluate_model)
    assert ev_sig.parameters["max_dets"].default is None
    assert ev_sig.parameters["iou_threshold"].default == 0.5
    # Honest None sentinels, resolved once through applied_operating_point ahead of the split.
    assert ev_sig.parameters["global_nms_iou"].default is None
    assert ev_sig.parameters["conf_threshold"].default is None

    # evaluation.py surfaces: pinned so a metrics-default change is visible too.
    coco_sig = inspect.signature(EV.coco_detection_metrics)
    assert coco_sig.parameters["conf_threshold"].default == 0.25
    assert coco_sig.parameters["iou_threshold"].default == 0.5
    assert coco_sig.parameters["max_dets"].default == 100
    ff_sig = inspect.signature(runners.run_full_frame_evaluation)
    # Also None sentinels, resolved inside the runner itself through applied_operating_point.
    assert ff_sig.parameters["conf_threshold"].default is None
    assert ff_sig.parameters["global_nms_iou"].default is None
    assert ff_sig.parameters["max_dets"].default is None
    # tile_size/overlap are no longer pinned constants (640/0.2): an honest None
    # sentinel resolved from the checkpoint's persisted geometry (or refused) by resolve_tile_geometry.
    assert ff_sig.parameters["tile_size"].default is None
    assert ff_sig.parameters["overlap"].default is None
    assert EV.DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.2}


def test_golden_evaluate_model_resolves_diagnostic_max_dets_when_unset(tmp_path, monkeypatch):
    """A signature-shape golden alone cannot see what a no-arg caller's max_dets actually resolves
    to on the tile-level/diagnostic regime (100, the COCOeval maxDets convention): evaluate_model
    still resolves this one itself, ahead of calling run_test_evaluation. The delivery-gating
    regime's own resolution (1000) now happens inside run_full_frame_evaluation itself, pinned on
    the runner's own record by test_gating_path_defaults_max_dets_to_1000_when_unset
    (test_delivery_grade_eval_regime.py)."""
    from tcip_mcp.pipelines.training import eval_runners as runners
    from tcip_mcp.tools import training_tools as TT
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    captured: dict = {}

    def _fake_diagnostic(ckpt, loader, device, task, output_dir, **kw):
        captured["diagnostic_max_dets"] = kw.get("max_dets")
        return {"tiled": False, "eval_regime": "tile-level"}

    orig_diag = runners.run_test_evaluation
    try:
        runners.run_test_evaluation = _fake_diagnostic

        from PIL import Image
        from tcip_annotation import json_io
        from tcip_annotation.state import Annotation, BBox

        tmp = tmp_path
        images_dir, labels_dir = tmp / "images", tmp / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        Image.new("RGB", (64, 64)).save(images_dir / "a.png")
        json_io.write_annotations(str(labels_dir / "a.json"),
                                  [Annotation(subject="bud", geometry=BBox(5, 5, 20, 20))],
                                  64, 64)
        monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp))
        ckpt = registered_checkpoint(tmp, project_root=tmp)

        TT.evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                          subject="bud")
    finally:
        runners.run_test_evaluation = orig_diag

    assert captured["diagnostic_max_dets"] == 100


def test_golden_evaluate_model_resolves_conf_threshold_per_regime_when_unset(tmp_path, monkeypatch):
    """A no-arg caller's conf_threshold resolves to the platform default on all three regimes,
    each constructed genuinely (a tiling dict for the tile-level run, nothing for the single
    pass, use_tiled_inference=True for the full frame), and the discriminating case: a caller
    stating the default value explicitly (0.5) still reaches the full-frame runner's record as an
    explicit stated value, never read back as an untouched default at the same number."""
    import tcip_mcp.pipelines.inference.predictor as predictor_mod
    import tcip_mcp.pipelines.model_build as model_build
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines import resolution as R
    from tcip_mcp.pipelines.training.eval_runners import evaluation_results_key
    from tcip_mcp.tools import training_tools as TT
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    def _dataset(root):
        images_dir, labels_dir = root / "images", root / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(120, 120, 120)).save(images_dir / "a.png")
        json_io.write_annotations(str(labels_dir / "a.json"),
                                  [Annotation(subject="bud", geometry=BBox(5, 5, 20, 20))], 64, 64)
        return images_dir, labels_dir

    from PIL import Image

    class _DummyModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            pass

    class _StubPredictor:
        train_tile_size = 64
        train_overlap = 0.0

        def predict_tiled(self, path, **kw):
            return {"width": 64, "height": 64, "boxes": [], "scores": [], "labels": []}

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))

    # Checkpoints are built (a real bespoke model, through the unpatched build_model) before the
    # model/predictor stubs below go in, so the fixture's own checkpoint save is never stubbed.
    def _prepare(root_name, name):
        root = tmp_path / root_name
        images_dir, labels_dir = _dataset(root)
        return images_dir, labels_dir, registered_checkpoint(
            root, project_root=tmp_path, name=name)

    tile_ds = _prepare("tile", "conf-tile-level")
    single_ds = _prepare("single", "conf-single-pass")
    ff_default_ds = _prepare("ff-default", "conf-full-frame-default")
    ff_stated_ds = _prepare("ff-stated", "conf-full-frame-stated")

    monkeypatch.setattr(model_build, "build_model", lambda ckpt: _DummyModel())
    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *a, **k: {"loss": 0.1, "precision": 0.4, "recall": 0.5, "f1": 0.44})
    monkeypatch.setattr(predictor_mod, "build_predictor", lambda *a, **kw: _StubPredictor())

    def _run(dataset, **kw):
        images_dir, labels_dir, ckpt = dataset
        r = TT.evaluate_model(str(ckpt), str(images_dir), str(labels_dir), task="detection",
                              subject="bud", **kw)
        assert "error" not in r, r
        return ts.read(evaluation_results_key(Path(ckpt).parent))

    tile_level = _run(tile_ds, tiling={"tile_size": 64, "overlap": 0.0})
    assert tile_level["conf_threshold"] == R.DEFAULT_CONF == 0.5

    single_pass = _run(single_ds)
    assert single_pass["conf_threshold"] == R.DEFAULT_CONF == 0.5

    full_frame_default = _run(ff_default_ds, use_tiled_inference=True)
    assert full_frame_default["conf_threshold"] == R.DEFAULT_CONF == 0.5
    assert full_frame_default["operating_point"]["conf"]["source"] == "default"

    full_frame_stated = _run(ff_stated_ds, use_tiled_inference=True, conf_threshold=0.5)
    assert full_frame_stated["conf_threshold"] == 0.5
    assert full_frame_stated["operating_point"]["conf"]["source"] == "explicit"


# ══════════════════════════════════════════════════════════════════════════
# 5. IoU-matching eval metrics at iou_threshold=0.5 (current criterion, to be replaced by a
#    derived center-match tolerance)
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
    # The same predictions score differently at 0.75, proof the criterion is IoU-thresholded
    # today (a future change replaces this with a derived center-match tolerance for the count).
    m = coco_detection_metrics(_iou_records(), iou_threshold=0.75,
                               conf_threshold=0.25, max_dets=100)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)


# ══════════════════════════════════════════════════════════════════════════
# 6. deliver_phenology_milestones gate behavior
# ══════════════════════════════════════════════════════════════════════════

def _write_stamp_bypassing_claim_rail(d: Path, stamp: dict, document: str) -> None:
    """Write a bucket's stamp through the storage seam, skipping the writer-side claim check."""
    from tcip_mcp.pipelines.resolution import sidecar_key

    d.mkdir(parents=True, exist_ok=True)
    key = sidecar_key(d, document)
    with ts.transaction(key) as txn:
        txn.write(key, stamp)


def _write_op_sidecar(d: Path, *, dataset_root: Path, validated: bool, conf: float = 0.4,
                      id_map: dict | None = None, subject: str = "bud",
                      attribute: str | None = None,
                      checkpoint_sha256: str | None = None,
                      experiment_id: str | None = "exp-golden") -> None:
    """The operating_point.json stamp run_inference writes beside a bucket's labels: the
    on-disk validity deliver_phenology_milestones reconciles against, including id_map and
    producer identity (the real writer always stamps checkpoint_sha256/experiment_id
    at the top level; a fixture that omitted them blessed a shape the platform never produces).

    A validated bucket also gets the producing run its stamp names filed, since a delivery repeats a
    producer identity only where an experiment outside the bucket corroborates the stamp's claim."""
    ref = "held_out_annotations" if validated else "false"
    d.mkdir(parents=True, exist_ok=True)
    if checkpoint_sha256 is None and validated and experiment_id:
        checkpoint_sha256 = record_producing_run(dataset_root, experiment_id)
    stamp = {
        "validated": validated,
        "trait": "bud_opening",
        "operating_point": {"conf": {"value": conf, "validated_against": ref}},
        "id_map": id_map,
        "checkpoint_sha256": checkpoint_sha256,
        "experiment_id": experiment_id,
        "subject": subject,
        "attribute": attribute,
    }
    if validated:
        write_bound_sidecar(d, stamp, dataset_root=dataset_root,
                            experiment_id=f"exp-record-{d.name}",
                            producing_experiment_id=experiment_id)
    else:
        _write_stamp_bypassing_claim_rail(d, stamp, "operating_point")


def _write_classifier_sidecar(d: Path, *, dataset_root: Path, validated: bool,
                              trait: str | None = "bud_opening") -> None:
    ref = "held_out_annotations" if validated else "false"
    d.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": validated,
        "operating_point": {"classifier": {"value": "open", "validated_against": ref}},
        "trait": trait,
    }
    if validated and trait:
        write_bound_sidecar(d, stamp, document="classifier_operating_point",
                            dataset_root=dataset_root, experiment_id=f"exp-classifier-{d.name}",
                            producing_experiment_id="exp-golden", trait=trait)
    else:
        _write_stamp_bypassing_claim_rail(d, stamp, "classifier_operating_point")


def _pheno_setup(tmp_path: Path, *, classified: bool, op_validated: bool | None = None):
    root = tmp_path / "ds"
    d1 = root / "predictions" / "run" / "2026-02-11"
    d2 = root / "predictions" / "run" / "2026-03-09"
    id_map = {"closed": 0, "open": 1} if classified else {"bud": 0}
    attribute = "opening" if classified else None
    _write_preds(d1, "P1_a", ["bud"] if not classified else ["closed"], attribute=attribute)
    _write_preds(d2, "P1_b", ["open"] if classified else ["bud"], attribute=attribute)
    if op_validated is not None:
        _write_op_sidecar(d1, dataset_root=root, validated=op_validated, id_map=id_map,
                          attribute=attribute)
        _write_op_sidecar(d2, dataset_root=root, validated=op_validated, id_map=id_map,
                          attribute=attribute)
    else:
        # count-operating-point sidecar still needs an id_map for the coverage rule even when its
        # own validity isn't the thing under test: a bucket with no sidecar at all is the
        # "no operating_point.json" case, tested separately.
        pass
    from tests._binding_fixtures import write_plant_mapping

    mapping_name = "valley"
    write_plant_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }, dataset_root=root)
    return mapping_name, d1, d2


def test_golden_deliver_phenology_milestones_refuses_without_opening_class(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    mapping_name, d1, d2 = _pheno_setup(tmp_path, classified=False, op_validated=True)  # bare detector
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert not out_csv.exists()


def test_golden_deliver_phenology_milestones_requires_both_validated_flags(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    mapping_name, d1, d2 = _pheno_setup(tmp_path, classified=True)  # no operating_point.json sidecars
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert not out_csv.exists()


def test_golden_deliver_phenology_milestones_refuses_on_a_present_but_unvalidated_stamp(tmp_path: Path):
    # An on-disk stamp refuses with no caller input at all: the sidecar's own conf.validated_against
    # is "false". The genuinely missing-sidecar case is the requires_both_validated_flags test above.
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    mapping_name, d1, d2 = _pheno_setup(tmp_path, classified=True, op_validated=False)
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
    )
    assert "error" in res
    assert res["operating_point_validated"] == "false"
    assert not out_csv.exists()


def test_golden_deliver_phenology_milestones_delivers_when_both_validated(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    # The positive-state fraction is now produced, so a fully-validated call (classifier + count
    # operating point both validated on disk) delivers a real phenology CSV.
    mapping_name, d1, d2 = _pheno_setup(tmp_path, classified=True, op_validated=True)
    _write_classifier_sidecar(d1, dataset_root=tmp_path / "ds", validated=True)
    out_csv = tmp_path / "out" / "bud_phenology.csv"
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )
    assert "error" not in res, res
    assert res["positive_class_assessed"] is True
    assert out_csv.exists()

    # A fully-validated delivery must carry real producer identity, not blank producer columns.
    import csv as _csv
    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert rows
    assert all(row["producer_model_sha256"] == producer_checkpoint_sha256("exp-golden") for row in rows)
    assert all(row["producing_experiment_id"] == "exp-golden" for row in rows)
    # And the record that answered for the claim, so a reader can reach the evidence from the CSV.
    assert all(row["validation_record"] for row in rows)

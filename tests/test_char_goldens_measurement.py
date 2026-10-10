"""Characterization goldens: pin the current numeric behavior of the measurement
and provenance rails that later work will touch.

These are deliberately *exact*: they assert the numbers today's code produces on tiny
deterministic fixtures, so a later semantic change (a different conf pick, a re-defined
localization criterion, a value no longer stated, a re-shaped stamp) fails loudly instead of
sliding through silently. They are not aspirational: a golden turning red is the signal to
update it *deliberately* alongside the change that moved the number.

Rails pinned here (one section each):
  1. conf operating-point sweep + count-unbiased pick + the count criterion over a reference
  2. phenology fraction curve + milestone dates (crossing_date / plant_milestones /
     per_plant_phenology)
  3. the execution record a pass runs under: the values it states, and the refusal of an
     unstated one
  4. no inference operating-point default anywhere, and the stated values each door hands on
  5. IoU-matching eval metrics at iou_threshold=0.5 (current criterion, to be replaced by a
     derived center-match tolerance)
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology as phenology_mod  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    detection_metrics,
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    derive_operating_point_curve,
)
from tests._trait_fixtures import BUD_OPENING  # noqa: E402
from tests._dense_op_fixtures import good_cal_holdout  # noqa: E402
from tests._dense_op_fixtures import toy_records as _sweep_records  # noqa: E402


# ── 1. conf operating-point sweep + count-unbiased pick + the count criterion ──

def test_golden_sweep_curve_exact():
    recs = _sweep_records()
    assert gt_class_avg_size(recs) == pytest.approx(20.0)
    tol = 0.5 * gt_class_avg_size(recs)
    assert tol == pytest.approx(10.0)

    sweep = derive_operating_point_curve(recs, criterion={"kind": "center_match", "tolerance": tol})
    assert sweep["criterion"]["tolerance"] == pytest.approx(10.0)
    assert sweep["class_id"] is None

    # The full swept curve, pinned exactly (conf grid = {0.0} ∪ observed scores).
    expected = [
        {"conf": 0.0, "tp": 3, "fp": 1, "fn": 0, "count_bias_mean": 0.5,
         "abs_count_error_mean": 0.5},
        {"conf": 0.3, "tp": 3, "fp": 1, "fn": 0, "count_bias_mean": 0.5,
         "abs_count_error_mean": 0.5},
        {"conf": 0.6, "tp": 2, "fp": 1, "fn": 1, "count_bias_mean": 0.0,
         "abs_count_error_mean": 1.0},
        {"conf": 0.9, "tp": 2, "fp": 0, "fn": 1, "count_bias_mean": -0.5,
         "abs_count_error_mean": 0.5},
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
    sweep = derive_operating_point_curve(recs, criterion={
        "kind": "center_match", "tolerance": 0.5 * gt_class_avg_size(recs)})
    assert pick_count_unbiased(sweep) == pytest.approx(0.6)
    assert pick_f1_max(sweep) == pytest.approx(0.0)
    # count bias vanishes at the count-unbiased pick, over-counts (+0.5) at the F1-max pick
    by_conf = {round(c["conf"], 6): c for c in sweep["curve"]}
    assert by_conf[0.6]["count_bias_mean"] == pytest.approx(0.0)
    assert by_conf[0.0]["count_bias_mean"] == pytest.approx(0.5)


def test_golden_count_criterion_over_a_dense_distinct_reference():
    """The count criterion needs a dense, realistic reference: a sparse 2-image fixture's
    per-image variance trips the equivalence test, a correct refusal."""
    from tcip_mcp.pipelines.derivations import derive_max_dets
    from tcip_mcp.pipelines.operating_point import count_criterion
    from tcip_mcp.pipelines.training.evaluation import gt_objects
    from tests import _trait_fixtures as fx

    cal, hold = good_cal_holdout()
    entry = fx.with_fields(BUD_OPENING, count_bias_tolerance_frac=0.1, count_error_tolerance=1.0)
    conf, evidence, failures = count_criterion(cal, hold, entry, staged_conf_floor=0.01,
                                               staged_conf_floor_attribute_path=None)
    # count-unbiased pick: bias vanishes once the low-conf FP drops
    assert conf == pytest.approx(0.9)
    assert evidence["conf_derived_from"] == "count-unbiased count curve"
    assert failures == []
    # ~1.5x p99 over frames of one size, published at that size
    frame = float(cal[0]["width"] * cal[0]["height"])
    assert derive_max_dets([(len(gt_objects(r)), float(r["width"] * r["height"]))
                            for r in cal + hold], frame) == 120


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
    # 0.05 lies midway 0.0 to 0.10, 10 days
    assert phenology_mod.crossing_date(_PHENO_SERIES, 0.05).date == "2026-02-15"
    assert phenology_mod.crossing_date(_PHENO_SERIES, 0.50).date == "2026-03-01"
    assert phenology_mod.crossing_date(_PHENO_SERIES, 0.95).date == "2026-03-12"
    # 0.97 observed, not 0.95 exactly
    assert phenology_mod.crossing_date(_PHENO_SERIES, 0.95).bound == "interpolated"
    assert phenology_mod.crossing_date(_PHENO_SERIES, 0.97).bound == "exact"
    # never reached within the observed window -> right-censored at the last observed date, not a
    # bare None: distinguishable from "no observations at all".
    c99 = phenology_mod.crossing_date(_PHENO_SERIES, 0.99)
    assert c99.date == "2026-03-12"
    assert c99.bound == "right_censored"
    assert phenology_mod.crossing_date([], 0.99) is None  # no observations at all -> still None


def test_golden_plant_milestones_shape_and_values():
    ms = phenology_mod.plant_milestones(_PHENO_SERIES, BUD_OPENING)
    assert ms["bud_05per_date"] == "2026-02-15"
    assert ms["bud_50per_date"] == "2026-03-01"
    assert ms["bud_95per_date"] == "2026-03-12"
    # crossing-unconfirmed (breeders to confirm): the majority alias == the 95% crossing
    assert ms["bud_majority_date"] == "2026-03-12"


def test_golden_per_plant_phenology_series_and_milestones(tmp_path: Path):
    from tcip_mcp import subject_registry as cr
    from tests._chain_fixtures import predicted, published

    opening = cr.Attribute("opening", "categorical", ("closed", "open"))
    registry = cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=(opening,)),))

    def bucket(date: str, stem: str, values: list[str]):
        image = tmp_path / "ds" / "images" / date / f"{stem}.png"
        return published(tmp_path, f"run/{date}", [predicted(image, values, (opening,))],
                         scope={"subject": "bud"}, registry=registry)

    buckets = {"2026-02-11": bucket("2026-02-11", "P1_a", ["closed", "closed", "closed", "open"]),
               "2026-03-09": bucket("2026-03-09", "P1_b", ["open", "open", "open", "closed"])}
    mapping = {
        "2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}],
    }
    res = phenology_mod.per_plant_phenology(
        mapping, buckets, BUD_OPENING, ["P1"],
        require_all_dates_complete=phenology_mod.REQUIRE_ALL_DATES_COMPLETE)

    # Both buckets carry the state's attribute on every detection, so the fraction is delivered.
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


# ── 3. the execution record a pass runs under: what it states, and an unstated value refused ──


def test_golden_execution_record_of_an_untiled_pass_runs_at_what_it_states(tmp_path):
    """Every value a pass runs under is recorded with where it came from: an untiled detector
    pass runs at the conf and cap it states, each sourced ``explicit``, and stating neither
    refuses."""
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, Stated, prepare
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_MAX_DETS, verified_checkpoint,
    )

    checkpoint = verified_checkpoint(tmp_path)
    record = prepare(checkpoint, Stated(tile=False, conf=SAMPLE_CONF,
                                        max_dets=SAMPLE_MAX_DETS)).runnable().execution.record()

    assert (record["conf"], record["max_dets"]) == (SAMPLE_CONF, SAMPLE_MAX_DETS)
    assert record["sources"] == {"conf": "explicit", "max_dets": "explicit"}
    assert record["tile_size"] is None and record["cross_tile_nms"] is None
    with pytest.raises(ExecutionRefusedError, match="conf"):
        prepare(checkpoint, Stated(tile=False)).runnable()


# ── 4. no inference operating-point default, and the stated values each door hands on ──

def test_golden_no_operating_point_default_survives_and_each_door_takes_stated_values(tmp_path):
    # No module carries an inference operating-point default: each value is stated, derived from
    # a reference the caller holds, or refused, and every door hands its stated values on whole.
    from tcip_mcp.pipelines import execution as execution_mod
    from tcip_mcp.pipelines import operating_point as operating_point_mod
    from tcip_mcp.pipelines.inference import generic_predictor as generic_predictor_mod
    from tcip_mcp.pipelines.training import eval_runners as runners
    from tcip_mcp.pipelines.training import evaluation as evaluation_mod
    from tcip_mcp.tools import training_tools as training_tools_mod

    # The operating point carries no shared fallback constant at all: a caller states each value
    # or derives it from a reference it holds.
    for name in ("DEFAULT_TILE_SIZE", "DEFAULT_TILED", "DEFAULT_NMS_IOU", "DEFAULT_CONF",
                 "DEFAULT_MAX_DETS"):
        assert not hasattr(execution_mod, name), name

    for name in ("DEFAULT_CONF", "DEFAULT_MAX_DETS", "DEFAULT_NMS_IOU", "DEFAULT_OVERLAP",
                 "_DEFAULT_CROSS_TILE_NMS", "_DEFAULT_MAX_DETS", "DEFAULT_TILE_SIZE"):
        assert not hasattr(operating_point_mod, name), name

    # generic_predictor's sliced primitive defaults nothing it runs under: the caller hands it
    # the whole execution record.
    gp_sig = inspect.signature(generic_predictor_mod.GenericPredictor.predict_sliced)
    for name in ("execution", "tile_batch_size", "require_masks"):
        assert gp_sig.parameters[name].default is inspect.Parameter.empty

    # evaluate_model states nothing when its caller states nothing; each regime then refuses what
    # a detector leaves unstated, as the two goldens below pin.
    ev_sig = inspect.signature(training_tools_mod.evaluate_model)
    assert ev_sig.parameters["stated"].default is None
    assert ev_sig.parameters["iou_threshold"].default == 0.5
    assert not {"conf_threshold", "cross_tile_nms", "max_dets"} & set(ev_sig.parameters)

    ff_sig = inspect.signature(runners.run_full_frame_evaluation)
    # The stated execution values arrive whole and resolve inside the runner's own prepared pass.
    assert ff_sig.parameters["stated"].default is inspect.Parameter.empty
    assert not {"conf_threshold", "cross_tile_nms", "max_dets", "tile_size", "overlap"} & set(
        ff_sig.parameters)
    assert evaluation_mod.DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.2}


def test_golden_evaluate_model_hands_the_diagnostic_its_stated_cap_and_refuses_none(
        tmp_path, monkeypatch):
    """A signature-shape golden alone cannot see what a caller's max_dets resolves to on the
    tile-level/diagnostic regime: evaluate_model resolves none of its own and hands the runner
    the pass the one execution resolver prepared, its cap the stated one; a call stating none
    refuses naming it."""
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.pipelines.training import eval_runners as runners
    from tcip_mcp.tools import training_tools as training_tools_mod
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_MAX_DETS, registered_checkpoint,
    )

    captured: dict = {}

    def _fake_diagnostic(pass_, loader, device, **kw):
        captured["diagnostic_max_dets"] = (pass_.execution.max_dets,
                                           pass_.execution.sources["max_dets"])
        return {"eval_regime": "tile-level"}

    orig_diag = runners.run_test_evaluation
    try:
        runners.run_test_evaluation = _fake_diagnostic

        from PIL import Image
        from tcip_annotation.state import Annotation, BBox

        from tests._producer_fixtures import label_image

        tmp = tmp_path
        images_dir = tmp / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(images_dir / "a.png")
        label_image(images_dir / "a.png", [Annotation(subject="bud", geometry=BBox(5, 5, 20, 20))],
                    64, 64)
        ckpt = registered_checkpoint(tmp)

        refused = training_tools_mod.evaluate_model(tmp, str(ckpt), str(images_dir),
                                                    stated=Stated(conf=SAMPLE_CONF))
        training_tools_mod.evaluate_model(
            tmp, str(ckpt), str(images_dir),
            stated=Stated(conf=SAMPLE_CONF, max_dets=SAMPLE_MAX_DETS))
    finally:
        runners.run_test_evaluation = orig_diag

    assert "max_dets" in refused.get("error", ""), refused
    assert captured["diagnostic_max_dets"] == (SAMPLE_MAX_DETS, "explicit")


def test_golden_evaluate_model_runs_each_regime_at_its_stated_conf_and_refuses_none(
        tmp_path, monkeypatch):
    """Each of the three regimes, constructed genuinely (a tiling block for the tile-level run,
    nothing for the single pass, use_tiled_inference=True for the full frame), runs at the conf
    the caller states, recorded ``explicit``, and refuses naming ``conf`` when none is stated."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines import execution as execution_mod
    from tcip_mcp.pipelines.schemas import TilingSpec
    from tcip_mcp.tools import training_tools as training_tools_mod
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_DETECTOR_PASS, registered_checkpoint,
    )

    from tests._producer_fixtures import label_image

    def _dataset(root):
        images_dir = root / "images" / UNDATED_BUCKET
        images_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(120, 120, 120)).save(images_dir / "a.png")
        label_image(images_dir / "a.png", [Annotation(subject="bud", geometry=BBox(5, 5, 20, 20))],
                    64, 64)
        return images_dir

    from PIL import Image

    class _DummyModel:
        def to(self, device):
            return self

    class _StubPredictor:
        task = "detection"
        train_tile_size = 64
        train_overlap = 0.0
        train_native_size = train_augmentation = None
        in_chans = 3
        dims = {"in_chans": 3, "num_classes": 1}
        model = _DummyModel()

        def governed(self, execution):
            return self.model

        def predict_sliced(self, path, **kw):
            return {"width": 64, "height": 64, "boxes": [], "scores": [], "labels": [],
                    "cap_hit": False}

    # The checkpoint is built through the unpatched build_model before the stubs below go in.
    checkpoint = registered_checkpoint(tmp_path)
    tile_ds, single_ds, ff_default_ds = (
        (_dataset(tmp_path / root_name), checkpoint)
        for root_name in ("tile", "single", "ff-default"))

    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *a, **k: {"loss": 0.1, "precision": 0.4, "recall": 0.5, "f1": 0.44})
    monkeypatch.setattr(predictor_mod, "GenericPredictor", lambda *a, **kw: _StubPredictor())

    unstated_conf = {k: v for k, v in SAMPLE_DETECTOR_PASS.items() if k != "conf"}
    tiling = TilingSpec.model_validate({"tile_size": 64, "overlap": 0.0, "sliver_frac": 0.5})
    regimes = [(tile_ds, {"tiling": tiling}),
               (single_ds, {}), (ff_default_ds, {"use_tiled_inference": True})]
    for (images_dir, ckpt), regime in regimes:
        refused = training_tools_mod.evaluate_model(
            tmp_path, str(ckpt), str(images_dir), **regime,
            stated=execution_mod.Stated(**unstated_conf))
        assert "conf" in refused.get("error", ""), (regime, refused)
        ran = training_tools_mod.evaluate_model(
            tmp_path, str(ckpt), str(images_dir), **regime,
            stated=execution_mod.Stated(**SAMPLE_DETECTOR_PASS))
        assert "error" not in ran, (regime, ran)
        assert (ran["execution"]["conf"], ran["execution"]["sources"]["conf"]) == (
            SAMPLE_CONF, "explicit"), regime


# ══════════════════════════════════════════════════════════════════════════
# 5. IoU-matching eval metrics at iou_threshold=0.5 (current criterion, to be replaced by a
#    derived center-match tolerance)
# ══════════════════════════════════════════════════════════════════════════

def _iou_records():
    """Discriminating fixture: an exact match, a 2px-shifted match (IoU~0.68 -> hit@0.5, miss@0.75),
    a spurious FP, and a whole image of GT with no predictions (FN)."""
    from tcip_mcp.pipelines.training.evaluation import gt_record

    def gt(x, y, w, h, cid=1):
        return gt_record([float(x), float(y), float(w), float(h)], cid, 0)

    def dt(x, y, w, h, score, cid=1):
        return {
            "category_id": cid, "bbox": [float(x), float(y), float(w), float(h)], "score": score
        }

    return [
        {"width": 100, "height": 100,
         "gt": [gt(10, 10, 20, 20), gt(60, 60, 20, 20)],
         "dt": [dt(10, 10, 20, 20, 0.95), dt(62, 62, 20, 20, 0.85), dt(90, 5, 8, 8, 0.6)]},
        {"width": 100, "height": 100, "gt": [gt(40, 40, 20, 20)], "dt": []},
    ]


def _metrics(iou_threshold: float) -> dict:
    return detection_metrics(_iou_records(), trait=None, conf_threshold=0.25,
                             iou_threshold=iou_threshold, by_mask=False)


def test_golden_coco_metrics_at_iou_050():
    m = _metrics(0.5)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["map50"] == pytest.approx(0.6633663366336634, abs=1e-6)
    assert m["map"] == pytest.approx(0.46732673267326735, abs=1e-6)


def test_golden_coco_matching_is_iou_threshold_sensitive():
    # The same predictions score differently at 0.75, proof the criterion is IoU-thresholded.
    m = _metrics(0.75)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 2)

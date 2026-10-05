"""Characterization goldens: pin the current numeric behavior of the measurement
and provenance rails that later work will touch.

These are deliberately *exact*: they assert the numbers today's code produces on tiny
deterministic fixtures, so a later semantic change (a different conf pick, a re-defined
localization criterion, a consolidated default, a re-shaped stamp) fails loudly instead of
sliding through silently. They are not aspirational: a golden turning red is the signal to
update it *deliberately* alongside the change that moved the number.

Rails pinned here (one section each):
  1. conf operating-point sweep + count-unbiased pick + the count criterion over a reference
  2. phenology fraction curve + milestone dates (crossing_date / plant_milestones / per_plant_phenology)
  3. the execution record a pass runs under, with no value stated
  4. the one set of inference operating-point defaults
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
from tcip_mcp.pipelines.postprocessing import phenology as PH  # noqa: E402
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
    from tcip_mcp.pipelines.derivations import derive_max_dets_from_counts
    from tcip_mcp.pipelines.operating_point import count_criterion
    from tcip_mcp.pipelines.training.evaluation import gt_objects
    from tests import _trait_fixtures as fx

    cal, hold = good_cal_holdout()
    entry = fx.with_fields(BUD_OPENING, count_bias_tolerance_frac=0.1, count_error_tolerance=1.0)
    conf, evidence, failures = count_criterion(cal, hold, entry, staged_conf_floor=0.01,
                                               staged_conf_floor_attribute_path=None)
    assert conf == pytest.approx(0.9)  # count-unbiased pick: bias vanishes once the low-conf FP drops
    assert evidence["conf_derived_from"] == "count-unbiased count curve"
    assert failures == []
    assert derive_max_dets_from_counts([len(gt_objects(r)) for r in cal + hold]) == 120  # ~1.5x p99


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
    res = PH.per_plant_phenology(mapping, buckets, BUD_OPENING, ["P1"],
                                 require_all_dates_complete=PH.REQUIRE_ALL_DATES_COMPLETE)

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


# ── 3. the execution record a pass runs under, with no value stated ──


def test_golden_execution_record_of_an_untiled_pass_with_nothing_stated(tmp_path):
    """Every value a pass runs under is recorded with where it came from; with nothing stated an
    untiled detector pass runs at the documented defaults, each sourced ``default``."""
    from tcip_mcp.pipelines.execution import DEFAULT_CONF, DEFAULT_MAX_DETS, Stated, prepare_pass
    from tests._verified_checkpoint_fixtures import verified_checkpoint

    record = prepare_pass(verified_checkpoint(tmp_path), Stated(tile=False)).execution.record()

    assert record["conf"] == DEFAULT_CONF == 0.5
    assert record["max_dets"] == DEFAULT_MAX_DETS == 1000
    assert record["sources"] == {"conf": "default", "max_dets": "default"}
    assert record["tile_size"] is None and record["cross_tile_nms"] is None


# ── 4. the one set of inference operating-point defaults ──

def test_golden_consolidated_operating_point_defaults(tmp_path):
    # No module but the execution record's own carries a copy of the inference operating-point
    # knobs: a second copy would let the same model and images give a different count by door.
    from tcip_mcp.pipelines import execution as R
    from tcip_mcp.pipelines import operating_point as OP
    from tcip_mcp.pipelines.inference import generic_predictor as GP
    from tcip_mcp.pipelines.training import eval_runners as runners
    from tcip_mcp.pipelines.training import evaluation as EV
    from tcip_mcp.tools import training_tools as TT

    # tile_size/tiled carry no shared fallback constant at all: a caller derives or states them.
    assert R.DEFAULT_CONF == 0.5
    assert R.DEFAULT_NMS_IOU == 0.3
    assert R.DEFAULT_MAX_DETS == 1000
    assert not hasattr(R, "DEFAULT_TILE_SIZE")
    assert not hasattr(R, "DEFAULT_TILED")

    for name in ("DEFAULT_CONF", "DEFAULT_MAX_DETS", "DEFAULT_NMS_IOU", "DEFAULT_OVERLAP",
                 "_DEFAULT_CROSS_TILE_NMS", "_DEFAULT_MAX_DETS", "DEFAULT_TILE_SIZE"):
        assert not hasattr(OP, name), name

    # generic_predictor's sliced primitive defaults nothing it runs under: the caller hands it
    # the whole execution record.
    gp_sig = inspect.signature(GP.GenericPredictor.predict_sliced)
    for name in ("execution", "tile_batch_size", "require_masks"):
        assert gp_sig.parameters[name].default is inspect.Parameter.empty

    # evaluate_model's stated values are a None sentinel each regime resolves for itself; what
    # each resolves to for a no-arg caller is pinned by the two goldens below.
    ev_sig = inspect.signature(TT.evaluate_model)
    assert ev_sig.parameters["stated"].default is None
    assert ev_sig.parameters["iou_threshold"].default == 0.5
    assert not {"conf_threshold", "cross_tile_nms", "max_dets"} & set(ev_sig.parameters)

    ff_sig = inspect.signature(runners.run_full_frame_evaluation)
    # The stated execution values arrive whole and resolve inside the runner's own prepared pass.
    assert ff_sig.parameters["stated"].default is inspect.Parameter.empty
    assert not {"conf_threshold", "cross_tile_nms", "max_dets", "tile_size", "overlap"} & set(
        ff_sig.parameters)
    assert EV.DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.2}


def test_golden_evaluate_model_resolves_diagnostic_max_dets_when_unset(tmp_path, monkeypatch):
    """A signature-shape golden alone cannot see what a no-arg caller's max_dets resolves to on
    the tile-level/diagnostic regime: evaluate_model resolves none of its own and hands the
    runner the pass the one execution resolver prepared, its cap the documented default."""
    from tcip_mcp.pipelines.execution import DEFAULT_MAX_DETS
    from tcip_mcp.pipelines.training import eval_runners as runners
    from tcip_mcp.tools import training_tools as TT
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

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
        ckpt = foreign_checkpoint(tmp)

        TT.evaluate_model(tmp, str(ckpt), str(images_dir))
    finally:
        runners.run_test_evaluation = orig_diag

    assert captured["diagnostic_max_dets"] == (DEFAULT_MAX_DETS, "default")


def test_golden_evaluate_model_resolves_conf_threshold_per_regime_when_unset(tmp_path, monkeypatch):
    """A no-arg caller's conf_threshold resolves to the platform default on all three regimes,
    each constructed genuinely (a tiling dict for the tile-level run, nothing for the single
    pass, use_tiled_inference=True for the full frame), and the discriminating case: a caller
    stating the default value explicitly (0.5) still reaches the full-frame runner's record as an
    explicit stated value, never read back as an untouched default at the same number."""
    import tcip_mcp.pipelines.inference.generic_predictor as predictor_mod
    import tcip_mcp.pipelines.training.evaluation as evaluation
    from tcip_mcp.pipelines import execution as R
    from tcip_mcp.tools import training_tools as TT
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

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

    # Checkpoints are built (a real bespoke model, through the unpatched build_model) before the
    # model/predictor stubs below go in, so the fixture's own checkpoint save is never stubbed.
    def _prepare(root_name, name):
        root = tmp_path / root_name
        return _dataset(root), foreign_checkpoint(tmp_path, name=name)

    tile_ds = _prepare("tile", "conf-tile-level")
    single_ds = _prepare("single", "conf-single-pass")
    ff_default_ds = _prepare("ff-default", "conf-full-frame-default")
    ff_stated_ds = _prepare("ff-stated", "conf-full-frame-stated")

    monkeypatch.setattr(evaluation, "evaluate",
                        lambda *a, **k: {"loss": 0.1, "precision": 0.4, "recall": 0.5, "f1": 0.44})
    monkeypatch.setattr(predictor_mod, "GenericPredictor", lambda *a, **kw: _StubPredictor())

    def _run(dataset, **kw):
        images_dir, ckpt = dataset
        r = TT.evaluate_model(tmp_path, str(ckpt), str(images_dir), **kw)
        assert "error" not in r, r
        return r

    tile_level = _run(tile_ds, tiling={"tile_size": 64, "overlap": 0.0, "sliver_frac": 0.5})
    assert tile_level["execution"]["conf"] == R.DEFAULT_CONF == 0.5

    single_pass = _run(single_ds)
    assert single_pass["execution"]["conf"] == R.DEFAULT_CONF == 0.5

    full_frame_default = _run(ff_default_ds, use_tiled_inference=True)
    assert full_frame_default["execution"]["conf"] == R.DEFAULT_CONF == 0.5
    assert full_frame_default["execution"]["sources"]["conf"] == "default"

    full_frame_stated = _run(ff_stated_ds, use_tiled_inference=True, stated=R.Stated(conf=0.5))
    assert full_frame_stated["execution"]["conf"] == 0.5
    assert full_frame_stated["execution"]["sources"]["conf"] == "explicit"


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
        return {"category_id": cid, "bbox": [float(x), float(y), float(w), float(h)], "score": score}

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

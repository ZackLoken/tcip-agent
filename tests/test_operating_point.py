"""Center-match count-unbiased operating-point sweep (count-trait calibration)."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load

from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    gt_class_avg_size,
    pick_count_unbiased,
    pick_f1_max,
    derive_operating_point_curve,
)

# No built-in traits: seed_bud_trait_spec (conftest.py) writes a real bud.yml into this
# test's pinned platform state root so resolve_operating_point("bud_opening", ...) keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

_DENSE_RECORDS_DEFAULTS = inspect.signature(dense_records).parameters
N_IMAGES = _DENSE_RECORDS_DEFAULTS["n_images"].default
OBJECTS_PER_IMAGE = _DENSE_RECORDS_DEFAULTS["objects_per_image"].default


def _box(cx: float, cy: float, s: float = 20.0) -> list[float]:
    return [cx - s / 2, cy - s / 2, s, s]  # xywh centered at (cx, cy)


def _ann(cx, cy, cid=0, score=None):
    a = {"category_id": cid, "bbox": _box(cx, cy)}
    if score is not None:
        a["score"] = score
    return a


def _records(idp="c", *, shift: float = 0.0):
    """Constructed so the count-unbiased conf (0.6) differs from the F1-max conf (0.9).

    Image A: 1 GT; a correct det @0.9 + a spurious far det @0.6.
    Image B: 2 GT; a correct det @0.9 + a hesitant-but-correct det @0.3.
    conf 0.6 -> A over-counts (+1, spurious kept), B under-counts (-1, hesitant dropped) => net bias 0.
    conf 0.9 -> A unbiased, B under-counts (-1) => net bias -0.5 but higher F1 (no spurious FP).

    ``shift`` offsets every GT box's center by that many px (well inside the ~10px center-match
    tolerance derived from these boxes) while leaving the detections in place, used to give a
    holdout fixture genuinely different GT content from calibration's (the content-overlap gate
    would otherwise flag a byte-identical-content holdout, differing only by ``image_id``, as a
    clone unable to function as an independent check).
    """
    a = {"width": 400, "height": 400, "image_id": f"{idp}_a",
         "gt": [_ann(100 + shift, 100)],
         "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}
    b = {"width": 400, "height": 400, "image_id": f"{idp}_b",
         "gt": [_ann(100 + shift, 100), _ann(200 + shift, 200)],
         "dt": [_ann(100, 100, score=0.9), _ann(200, 200, score=0.3)]}
    return [a, b]


def test_good_cal_holdout_hashes_are_disjoint_and_detection_counts_differ():
    from tcip_mcp.pipelines.operating_point import _record_content_hash

    cal, hold = good_cal_holdout()
    cal_hashes = {_record_content_hash(r) for r in cal}
    hold_hashes = {_record_content_hash(r) for r in hold}
    assert cal_hashes.isdisjoint(hold_hashes)
    cal_counts = {len(r["gt"]) for r in cal}
    hold_counts = {len(r["gt"]) for r in hold}
    assert cal_counts != hold_counts


def test_gt_class_avg_size_derived_from_data():
    assert gt_class_avg_size(_records()) == pytest.approx(20.0)


def test_count_unbiased_differs_from_f1_max():
    recs = _records()
    tol = 0.5 * gt_class_avg_size(recs)  # derived tolerance = half class avg size
    sweep = derive_operating_point_curve(recs, tolerance=tol)

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
    sweep = derive_operating_point_curve(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at0 = sweep["curve"][0]  # conf=0.0 is always the first (lowest) grid point
    assert at0["conf"] == pytest.approx(0.0)
    assert at0["tp"] == 0 and at0["fp"] == 1 and at0["fn"] == 1  # miss + false positive


def test_sweep_curve_carries_dispersion_and_reference_size_fields():
    # Every curve entry now also carries count_error_p90 / count_bias_std / n_images, computed
    # from the same per-image biases list, not a second pass over the data.
    recs = dense_records(n_images=4, objects_per_image=10,
                         miss_pattern=[0, 1, 0, 2], fp_pattern=[0, 0, 1, 0])
    sweep = derive_operating_point_curve(recs, tolerance=0.5 * gt_class_avg_size(recs))
    at09 = next(c for c in sweep["curve"] if c["conf"] == pytest.approx(0.9))
    # biases = fp - fn per image = [0, -1, 1, -2]
    assert at09["n_images"] == 4
    assert at09["count_bias_mean"] == pytest.approx(-0.5)
    assert at09["count_bias_std"] == pytest.approx(1.2909944, abs=1e-5)
    assert at09["count_error_p90"] == pytest.approx(1.7, abs=1e-6)


# --- resolve_operating_point + the in-model seam ---

def _two_stage():
    from types import SimpleNamespace
    return SimpleNamespace(detector=SimpleNamespace(
        roi_heads=SimpleNamespace(score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)))


def _one_stage():
    from types import SimpleNamespace
    return SimpleNamespace(detector=SimpleNamespace(score_thresh=0.2, nms_thresh=0.6, detections_per_img=100))


def test_set_detector_operating_point_two_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    m = _two_stage()
    applied, attribute_path = set_detector_operating_point(
        m, score_thresh=0.4, nms_thresh=0.3, detections_per_img=300)
    assert m.detector.roi_heads.score_thresh == 0.4
    assert m.detector.roi_heads.nms_thresh == 0.3
    assert m.detector.roi_heads.detections_per_img == 300
    assert applied == {"score_thresh": 0.4, "nms_thresh": 0.3, "detections_per_img": 300}
    assert attribute_path == "detector.roi_heads"


def test_set_detector_operating_point_one_stage():
    from tcip_mcp.pipelines.operating_point import set_detector_operating_point
    m = _one_stage()
    set_detector_operating_point(m, score_thresh=0.4, nms_thresh=0.35)
    assert m.detector.score_thresh == 0.4 and m.detector.nms_thresh == 0.35


def test_max_dets_from_density_scales_above_floor():
    # 1.5x the p99 GT-per-image count exceeds the 100 floor, so max_dets scales with
    # density instead of pinning to the floor (a dense scene must not be truncated).
    from tcip_mcp.pipelines.operating_point import _max_dets_from_density
    records = [{"gt": [_ann(0, 0)] * 80} for _ in range(20)]
    assert _max_dets_from_density(records) == 120  # ceil(1.5 * 80)


def test_max_dets_from_density_floors_sparse_scenes():
    from tcip_mcp.pipelines.operating_point import _max_dets_from_density
    records = [{"gt": [_ann(0, 0)] * 2} for _ in range(20)]
    assert _max_dets_from_density(records) == 100  # floor, not ceil(1.5 * 2)


def test_derive_max_dets_from_counts_is_the_shared_formula_records_delegate_to():
    # scripts/calibrate_operating_point.py derives its collection-pass cap from raw label counts
    # (known before any model pass), not from already-collected records: this is the same ~1.5x
    # p99 formula _max_dets_from_density applies over per-record GT counts, exposed directly so the
    # two callers share one implementation.
    from tcip_mcp.pipelines.operating_point import _max_dets_from_density, derive_max_dets_from_counts
    from tcip_mcp.pipelines.resolution import DEFAULT_MAX_DETS
    counts = [80] * 20
    assert derive_max_dets_from_counts(counts) == 120  # ceil(1.5 * 80)
    assert derive_max_dets_from_counts([2] * 20) == 100  # floor, not ceil(1.5 * 2)
    assert derive_max_dets_from_counts([]) == DEFAULT_MAX_DETS  # no counts to derive from
    # Same result either through the counts directly or through records carrying the same counts.
    records = [{"gt": [_ann(0, 0)] * 80} for _ in range(20)]
    assert derive_max_dets_from_counts(counts) == _max_dets_from_density(records)


def test_resolve_operating_point_validated_with_holdout():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    cal, hold = good_cal_holdout()
    # tiled=False: this test is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                tiled=False, staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.requires_validation is True and conf.validation_kind == "annotations"
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable
    assert conf.value == pytest.approx(0.9)  # count-unbiased pick: bias vanishes once the low-conf FP drops
    assert b.get("max_dets").value >= 100  # derived from GT density
    sweep = conf.gate_evidence
    assert sweep["failures"] == []
    assert sweep["content_overlap_frac"] == pytest.approx(0.0)  # genuinely distinct holdout content
    assert sweep["train_disjointness"] == {"checked": False, "unresolvable": False,
                                           "leaked_groups": [], "leaked_stems": [],
                                           "group_check": None}
    assert set(sweep["calibration_image_ids"]) == {f"c_{i}" for i in range(N_IMAGES)}
    assert set(sweep["holdout_image_ids"]) == {f"h_{i}" for i in range(N_IMAGES)}


def test_resolve_operating_point_overlapping_holdout_not_validated():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # same image ids in calibration and holdout -> not a real held-out split -> not validated
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=_records("c"), holdout_records=_records("c"))
    assert b.get("conf").validated_against == "false"
    assert not b.is_shippable


def test_resolve_operating_point_missing_image_ids_fails_closed():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # Records with no image_id: identity is unverifiable, so a held-out claim can't be proven, and
    # the same records used as cal+holdout must not be stamped validated (the firewall fails
    # closed) merely because empty id-sets make `disjoint` trivially True.
    recs = [{"width": 400, "height": 400, "gt": [_ann(100, 100)],
             "dt": [_ann(100, 100, score=0.9)]}]  # no image_id key
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=recs, holdout_records=recs)
    assert b.get("conf").validated_against == "false"
    assert not b.is_shippable


def test_resolve_operating_point_biased_holdout_is_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    cal, _ = good_cal_holdout()
    # a dense holdout with a real, consistent per-image miss (not just a sparse fixture's one-off
    # spread), count bias -3/image, well beyond tolerance regardless of dispersion/SE.
    biased_hold = dense_records(n_images=N_IMAGES, objects_per_image=OBJECTS_PER_IMAGE, id_prefix="h",
                                shift=5.0, miss_pattern=[3] * N_IMAGES, fp_pattern=[0] * N_IMAGES,
                                score=0.9)
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=cal, holdout_records=biased_hold,
                                staged_conf_floor=0.01)
    # measured on the disjoint split but failed (bias > tolerance) -> not validated, firewall holds
    assert b.get("conf").validated_against == "false"
    assert not b.is_shippable
    assert "count_bias_exceeds_tolerance" in b.get("conf").gate_evidence["failures"]


def test_resolve_operating_point_calibrated_but_no_holdout_is_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_records())
    assert b.get("conf").validated_against == "false"
    assert not b.is_shippable


# --- Content-overlap and train-disjointness gates ---

def test_resolve_operating_point_content_shared_holdout_is_false():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # Same GT content as calibration (only image_id differs, no shift) -> disjoint by image_id but
    # the holdout can't function as an independent check; the content-overlap gate must refuse it.
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=_records("c"), holdout_records=_records("h"))
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert not b.is_shippable
    assert conf.gate_evidence["content_overlap_frac"] == pytest.approx(1.0)
    assert "content_shared_with_calibration" in conf.gate_evidence["failures"]


def test_resolve_operating_point_train_disjointness_fires(tmp_path, monkeypatch):
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp1"), {"train": ["a_0_0", "a_0_1"], "group_by": "tile_prefix"})

    # Calibration/holdout share tile group "a" (stem "a_0_2") with the training split above.
    cal = [{"width": 400, "height": 400, "image_id": "a_0_2", "gt": [_ann(100, 100)],
            "dt": [_ann(100, 100, score=0.9), _ann(300, 300, score=0.6)]}]
    hold = [{"width": 400, "height": 400, "image_id": "a_0_3", "gt": [_ann(100, 100 + 5)],
             "dt": [_ann(100, 100, score=0.9)]}]
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, experiment_id="exp1")
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert conf.gate_evidence["train_disjointness"]["leaked_groups"] == ["a"]


def test_resolve_operating_point_train_disjointness_unresolvable_when_split_missing(tmp_path, monkeypatch):
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    # A known experiment_id whose split.json can't be read fails closed (unresolvable), unlike the
    # experiment_id=None case (a foreign/unregistered checkpoint).
    cal, hold = good_cal_holdout()
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold,
                                staged_conf_floor=0.01, experiment_id="does-not-exist")
    conf = b.get("conf")
    assert conf.validated_against == "false"
    assert conf.gate_evidence["train_disjointness"] == {"checked": False, "unresolvable": True,
                                                 "leaked_groups": [], "leaked_stems": [],
                                                 "group_check": None}
    assert "train_disjointness_unresolvable" in conf.gate_evidence["failures"]


def test_resolve_operating_point_train_disjointness_resolvable_no_leak_still_validates(tmp_path, monkeypatch):
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp2"), {"train": ["z_0_0", "z_0_1"], "group_by": "tile_prefix"})

    # Calibration/holdout use id prefixes "c"/"h", disjoint from training's "z" group.
    cal, hold = good_cal_holdout()
    # tiled=False: this test is about conf-calibration shippability, not tiling (tile_size
    # only gates a bundle when tiled).
    b = resolve_operating_point("bud_opening", dataset_hash="h1",
                                calibration_records=cal, holdout_records=hold, tiled=False,
                                staged_conf_floor=0.01, experiment_id="exp2")
    conf = b.get("conf")
    assert conf.validated_against == "held_out_annotations"
    assert b.is_shippable
    assert conf.gate_evidence["train_disjointness"] == {"checked": True, "unresolvable": False,
                                                 "leaked_groups": [], "leaked_stems": [],
                                                 "group_check": "performed"}


def test_resolve_operating_point_cal_rects_none_is_byte_identical(tmp_path, monkeypatch):
    """cal_rects/hold_rects default to None: an existing caller who never passes them (every
    caller before this phase) gets exactly today's lexical spatial_strip check, unchanged."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_rects_noop"), {
        "train": ["mosaic::strip_x_1"], "group_by": "spatial_strip",
    })
    cal, hold = good_cal_holdout()

    omitted = resolve_operating_point("bud_opening", tiled=False, dataset_hash="h1",
                                      calibration_records=cal, holdout_records=hold,
                                      staged_conf_floor=0.01, experiment_id="exp_rects_noop")
    explicit_none = resolve_operating_point("bud_opening", tiled=False, dataset_hash="h1",
                                            calibration_records=cal, holdout_records=hold,
                                            staged_conf_floor=0.01, experiment_id="exp_rects_noop",
                                            cal_rects=None, hold_rects=None)
    assert (omitted.get("conf").gate_evidence["train_disjointness"]
           == explicit_none.get("conf").gate_evidence["train_disjointness"]
           == {"checked": True, "unresolvable": False, "leaked_groups": [], "leaked_stems": [],
               "group_check": "spatial_strip"})


def test_resolve_operating_point_cal_rects_switches_to_geometric_check(tmp_path, monkeypatch):
    """Given cal_rects/hold_rects against a spatial_strip split, resolve_operating_point threads
    them into the geometric containment check instead of the lexical same-source one, catching a
    leak the lexical check alone would miss (a rect whose own source name isn't a training stem
    at all, but whose geometry spills into the persisted train region)."""
    import tcip_store

    from tcip_mcp.experiments import split_key
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    tcip_store.replace(split_key("exp_rects_geo"), {
        "train": ["mosaic::strip_x_1"], "group_by": "spatial_strip",
        "spatial": {
            "train_region": [[0, 0, 500, 1000]],
            "val_region": [[500, 0, 750, 1000]],
            "test_region": [[750, 0, 1000, 1000]],
        },
    })
    cal, hold = good_cal_holdout()
    cal_id = cal[0]["image_id"]

    leaked = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_rects_geo",
        cal_rects={cal_id: (400, 100, 600, 300)},  # straddles train/val: not fully contained
    )
    td = leaked.get("conf").gate_evidence["train_disjointness"]
    assert td["group_check"] == "spatial_strip_geometric"
    assert td["leaked_groups"] == [cal_id]


# --- Selection-disjointness: a checkpoint's own held-out (val) side, not its train side -----

def _persist_run_split(experiment_id, tmp_path, *, date, train, val,
                       group_by=None, manifest_dir=None):
    """A real ``split.json`` for one producing run, written through the platform's own
    ``persist_split_manifest``, never composed by hand: ``train``/``val`` are bare stems under
    ``date``, ``group_by`` an already-resolved policy (``"external"``/``"spatial_strip"``/a named
    strategy), and ``manifest_dir`` (when given) records the run as bound to that manifest."""
    from types import SimpleNamespace

    from tcip_mcp.experiments import create_experiment, experiment_exists
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    if not experiment_exists(experiment_id):
        create_experiment(experiment_id, {})
    split_cfg: dict = {}
    if group_by is not None:
        split_cfg["resolved_group_by"] = group_by
    if manifest_dir is not None:
        split_cfg["manifest_binding"] = {"manifest_dir": manifest_dir}
    data_cfg = {"labels_dir": str(tmp_path / "annotations" / date), "split": split_cfg}
    train_ds = SimpleNamespace(stems=list(train))
    val_ds = SimpleNamespace(stems=list(val))
    persist_split_manifest(experiment_id, train_ds, val_ds, data_cfg)


def test_selection_disjointness_leaked_whole_directory_calibration_of_a_bound_checkpoint(
    tmp_path, monkeypatch,
):
    """A checkpoint bound to a manifest, calibrated over the whole labelled directory on its own
    date (no split_manifest_dir stated for this particular calibration), still sweeps its own
    selection members and floors with the token: the check runs whenever the record carries a
    manifest_binding, whether or not this calibration itself names one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2-11-26"
    _persist_run_split(
        "exp_sel_leak_whole", tmp_path, date=date, train=["z"], val=["c_0"],
        group_by="tile_prefix", manifest_dir="some/manifest",
    )

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_sel_leak_whole", calibration_date=date,
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is True
    assert sd["leaked_groups"] == ["c_0"]
    assert "selection_disjointness_leaked" in b.get("conf").gate_evidence["failures"]
    assert b.get("conf").validated_against == "false"


def test_selection_disjointness_leaked_manifest_calibration_of_a_self_drawn_checkpoint(
    tmp_path, monkeypatch,
):
    """A checkpoint that drew its own split, calibrated under a stated manifest on its own date,
    is checked against its own val the same way: the universe a manifest names may be exactly
    the side this checkpoint was chosen on."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2-11-26"
    _persist_run_split(
        "exp_sel_leak_manifest", tmp_path, date=date, train=["z"], val=["c_0"],
        group_by="tile_prefix",
    )

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_sel_leak_manifest",
        split_manifest_dir="some/manifest", calibration_date=date,
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is True
    assert sd["leaked_groups"] == ["c_0"]
    assert "selection_disjointness_leaked" in b.get("conf").gate_evidence["failures"]


def test_selection_disjointness_not_applicable_across_dates(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _persist_run_split(
        "exp_sel_other_date", tmp_path, date="2-11-26", train=["z"], val=["c_0", "h_0"],
        manifest_dir="some/manifest",
    )

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_sel_other_date",
        split_manifest_dir="some/manifest", calibration_date="2-12-01",
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is False and sd["reason"]
    assert b.get("conf").validated_against == "held_out_annotations"


def test_selection_disjointness_not_applicable_on_a_spatial_record(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2-11-26"
    _persist_run_split(
        "exp_sel_spatial", tmp_path, date=date, train=["mosaic::strip_x_0"],
        val=["mosaic::strip_x_1"], group_by="spatial_strip", manifest_dir="some/manifest",
    )

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_sel_spatial",
        split_manifest_dir="some/manifest", calibration_date=date,
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is False and sd["reason"]
    assert b.get("conf").validated_against == "held_out_annotations"


def test_selection_disjointness_not_applicable_on_an_external_val_record(tmp_path, monkeypatch):
    """A run trained with an explicit val_images_dir and calibrated under a manifest validates,
    the selection check not-applicable: the val came from a directory the record's date says
    nothing about."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2-11-26"
    _persist_run_split(
        "exp_sel_external", tmp_path, date=date, train=["z"], val=["c_0", "h_0"],
        group_by="external", manifest_dir="some/manifest",
    )

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_sel_external",
        split_manifest_dir="some/manifest", calibration_date=date,
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is False and sd["reason"]
    assert b.get("conf").validated_against == "held_out_annotations"


def test_selection_disjointness_not_applicable_for_a_manifest_less_calibration(tmp_path, monkeypatch):
    """A checkpoint that drew its own split, calibrated with no manifest named, behaves exactly
    as before this family: the selection check is not-applicable and never blocks validation."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2-11-26"
    _persist_run_split(
        "exp_sel_no_manifest", tmp_path, date=date, train=["z"], val=["v_0"],
    )

    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id="exp_sel_no_manifest",
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is False and sd["reason"]
    assert b.get("conf").validated_against == "held_out_annotations"


def test_selection_disjointness_not_applicable_for_a_flat_run_with_no_calibration_date(
    tmp_path, monkeypatch,
):
    """calibration_date=None means the caller derived no date at all: never read as matching a
    flat run's own record (also date-empty, the manifest's own key, not None), which would
    wrongly run the check for real and could leak."""
    from types import SimpleNamespace

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    create_experiment("exp_sel_flat_no_date", {})
    data_cfg = {"labels_dir": str(tmp_path / "annotations"),
               "split": {"resolved_group_by": "tile_prefix",
                        "manifest_binding": {"manifest_dir": "some/manifest"}}}
    persist_split_manifest(
        "exp_sel_flat_no_date", SimpleNamespace(stems=["z"]), SimpleNamespace(stems=["c_0"]),
        data_cfg)

    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point

    cal_items = [{"image_id": "c_0", "is_true_positive": True, "is_pred_positive": True,
                 "bbox": [0.0, 0.0, 10.0, 10.0]}]
    hold_items = [{"image_id": "h_0", "is_true_positive": True, "is_pred_positive": True,
                  "bbox": [0.0, 0.0, 10.0, 10.0]}]
    result = resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal_items, holdout_items=hold_items,
        experiment_id="exp_sel_flat_no_date",
    )
    sd = result["gate_evidence"]["selection_disjointness"]
    assert sd["applicable"] is False and sd["reason"]
    assert "selection_disjointness_leaked" not in result["failures"]


def test_selection_disjointness_applicable_when_a_flat_calibration_matches_a_flat_run(
    tmp_path, monkeypatch,
):
    """A calibration that derives a date for a flat tree (the manifest's own empty key, not
    ``None``) matches a flat run's own record the same way a dated one matches a dated record,
    running the check for real."""
    from types import SimpleNamespace

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.pipelines.data.splits import manifest_date_key
    from tcip_mcp.pipelines.data.split_construction import persist_split_manifest

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    create_experiment("exp_sel_flat_match", {})
    data_cfg = {"labels_dir": str(tmp_path / "annotations"),
               "split": {"resolved_group_by": "tile_prefix",
                        "manifest_binding": {"manifest_dir": "some/manifest"}}}
    persist_split_manifest(
        "exp_sel_flat_match", SimpleNamespace(stems=["z"]), SimpleNamespace(stems=["c_0"]),
        data_cfg)

    from tcip_mcp.pipelines.operating_point import resolve_classifier_operating_point

    cal_items = [{"image_id": "c_0", "is_true_positive": True, "is_pred_positive": True,
                 "bbox": [0.0, 0.0, 10.0, 10.0]}]
    hold_items = [{"image_id": "h_0", "is_true_positive": True, "is_pred_positive": True,
                  "bbox": [0.0, 0.0, 10.0, 10.0]}]
    result = resolve_classifier_operating_point(
        "bud_opening", calibration_items=cal_items, holdout_items=hold_items,
        experiment_id="exp_sel_flat_match", calibration_date=manifest_date_key(None),
    )
    sd = result["gate_evidence"]["selection_disjointness"]
    assert sd["applicable"] is True and sd["checked"] is True
    assert sd["leaked_groups"] == ["c_0"]  # c_0 is on this run's own val


def test_selection_disjointness_unresolvable_for_experiment_id_none_under_a_stated_manifest():
    """A calibration that names a manifest but carries no experiment record to check the
    selection side against is unresolvable, not merely not-applicable: the shape a foreign
    checkpoint's train check allows through cannot be extended to a stated manifest, since the
    number the ruling forbids is exactly one whose provenance cannot be checked."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point

    cal, hold = good_cal_holdout()
    b = resolve_operating_point(
        "bud_opening", tiled=False, dataset_hash="h1", calibration_records=cal, holdout_records=hold,
        staged_conf_floor=0.01, experiment_id=None,
        split_manifest_dir="some/manifest", calibration_date="2-11-26",
    )
    sd = b.get("conf").gate_evidence["selection_disjointness"]
    assert sd["applicable"] is True and sd["unresolvable"] is True
    assert "selection_disjointness_unresolvable" in b.get("conf").gate_evidence["failures"]
    assert b.get("conf").validated_against == "false"


# --- tile_size gates the calibrated path too, not just raw_operating_point -----------------

def test_resolve_operating_point_fabricated_tile_size_floors_shippability_even_with_valid_conf():
    """The calibrated door (operating_point.py) shares resolve_tile_size_param with the raw door
    (resolution.py), not a second, divergent implementation, so a tiled run with no persisted
    training geometry and no explicit override is caught here too, not only on the uncalibrated
    path. A cleanly-validated conf must not paper over a fabricated tile scale."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import VALIDATED_FALSE

    cal, hold = good_cal_holdout()
    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=True, tile_size=640,
                                staged_conf_floor=0.01)
    conf = b.get("conf")
    assert conf.validated_against == "held_out_annotations"  # conf itself validates cleanly...
    tile = b.get("tile_size")
    assert tile.requires_validation is True and tile.validation_kind == "geometry"
    assert tile.validated_against == VALIDATED_FALSE          # ...but the fabricated scale doesn't
    assert b.is_shippable is False                            # so the bundle as a whole refuses
    assert any(i.startswith("tile_size:") for i in b.shippable_issues())


def test_resolve_operating_point_derived_tile_size_is_shippable():
    """The mirror case: a tile_size genuinely derived from the checkpoint's persisted training
    geometry has a real basis and must not be penalized alongside the fabricated-default case: the
    rail must admit this legitimate call, not only reject the fabricated one."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import VALIDATED_PERSISTED_GEOMETRY

    cal, hold = good_cal_holdout()
    b = resolve_operating_point("bud_opening", dataset_hash="h1", calibration_records=cal,
                                holdout_records=hold, tiled=True, tile_size=224,
                                tile_size_source="derived", staged_conf_floor=0.01)
    tile = b.get("tile_size")
    assert tile.validated_against == VALIDATED_PERSISTED_GEOMETRY
    assert tile.is_shippable is True
    assert b.is_shippable is True


def test_resolve_operating_point_no_gt_placeholder_unshippable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import UnvalidatedOperatingPointError
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="hX")
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = b.value("conf")  # firewall
    assert b.get("conf").unvalidated_value(acknowledge_unvalidated=True) == 0.5


def _overlap_records(idp="d"):
    """Calibration records whose GT boxes overlap (20px, offset 8px), so cross_tile_nms is derivable."""
    boxes = [_box(100, 100), _box(108, 100), _box(116, 100)]  # neighbor IoU ~0.43
    return [{"width": 400, "height": 400, "image_id": f"{idp}_{i}",
             "gt": [{"category_id": 1, "bbox": bx} for bx in boxes],
             "dt": [{"category_id": 1, "bbox": bx, "score": 0.9} for bx in boxes]}
            for i in range(2)]


def test_resolve_operating_point_derives_cross_tile_nms_from_gt():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", calibration_records=_overlap_records())
    p = b.get("cross_tile_nms")
    assert p.source == "derived"
    assert p.requires_validation is False  # a statistic from this dataset's own spread needs no validation
    assert "neighbor-IoU" in p.derived_from
    assert 0.2 <= p.value <= 0.8
    assert p.value == pytest.approx(0.4286 + 0.05, abs=1e-2)  # p99 of the GT neighbor-IoU tail + margin


def test_resolve_operating_point_explicit_cross_tile_nms_not_labeled_derived():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # An explicit override is honest even when overlapping GT was present to derive from.
    b = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                calibration_records=_overlap_records(), cross_tile_nms=0.55)
    p = b.get("cross_tile_nms")
    assert p.source == "explicit"
    assert p.value == pytest.approx(0.55)
    assert p.derived_from == "caller override"  # not a derivation costume on a caller-supplied number
    assert "neighbor-IoU" not in p.derived_from


def test_resolve_operating_point_cross_tile_nms_honest_default_when_underivable():
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    # No GT at all -> honest default, never a derivation label on an underived number.
    p_no_gt = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1").get("cross_tile_nms")
    assert p_no_gt.source == "default"
    assert "neighbor-IoU" not in p_no_gt.derived_from
    # Sparse, non-overlapping GT is likewise underivable -> still an honest default.
    p_sparse = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1",
                                       calibration_records=_records("c")).get("cross_tile_nms")
    assert p_sparse.source == "default"


def test_resolve_operating_point_tile_size_derived():
    """A bare truthy tile_size with no source claim must not be inferred as "derived"
    unconditionally (`if tile_size: derived(...)`): a caller-passed number with no real basis
    (``tile_size_source`` not "explicit"/"derived") is discarded entirely, never carried forward as
    if it meant something. The caller must say which it was via ``tile_size_source``; omitting it
    means there is no basis to trust, not "a default value of 640"."""
    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.resolution import UnvalidatedOperatingPointError

    b_no_claim = resolve_operating_point("bud_opening", tiled=True, dataset_hash="h1", tile_size=640)
    assert b_no_claim.get("tile_size").source == "default"
    # A "default"-sourced tile_size, when tiled, is a firewalled unvalidated dimension: the
    # caller's raw 640 is discarded, never fabricated into a trustworthy value.
    assert b_no_claim.get("tile_size")._raw is None
    assert b_no_claim.get("tile_size").is_shippable is False
    with pytest.raises(UnvalidatedOperatingPointError):
        _ = b_no_claim.get("tile_size").value

    b_derived = resolve_operating_point(
        "bud_opening", tiled=True, dataset_hash="h1", tile_size=640, tile_size_source="derived")
    assert b_derived.get("tile_size").source == "derived"
    assert b_derived.get("tile_size").value == 640


def test_classification_metrics_per_class_and_bias():
    from tcip_mcp.pipelines.training.evaluation import classification_metrics
    gt = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])    # 6 closed, 4 open
    pred = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # classifier predicts 6 as open
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["per_class"][1]["support"] == 4
    # over-predicting the open class inflates the open fraction: bias (6-4)/4 = +0.5
    assert m["count_bias"][1] == pytest.approx(0.5)
    assert "accuracy" in m and "f1" in m  # existing keys preserved (additive)

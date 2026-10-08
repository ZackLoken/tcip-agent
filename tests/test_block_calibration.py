"""A mosaic's own reserved calibration and test regions assessed directly, without a separate
held-out image set (``assessment.assess_reserved_regions``).

Covers: the completion-mark gate (refuses unmarked, admits marked), the sub-banding and
halo running real tiled inference over real bands under the one execution record the assessment
records, the geometric disjointness check, and the band geometry helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

from tcip_mcp.dataset_layout import UNDATED_BUCKET  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402
from tests._chain_fixtures import BESPOKE_DETECTION, SAVE_BUILT_WEIGHTS  # noqa: E402
from tests._mapping_fixtures import write_plant_csv  # noqa: E402
from tests._producer_fixtures import image_label_key, label_image  # noqa: E402

TILE = 32
OVERLAP = 0.2
WIDTH, HEIGHT = 3200, 200
BOX_STEP = 40


@pytest.fixture(autouse=True)
def _count_trait(tmp_path):
    """The count trait confirmed in this test's project, measuring the mosaic's subject."""
    fx.seed_confirmed_count(tmp_path, measured_subject="bud")


def _write_mosaic(path: Path, *, seed: int = 0, georeferenced: bool = False,
                  tiepoint_x: float = 500_000.0) -> None:
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    if not georeferenced:
        tifffile.imwrite(str(path), arr)
        return
    geokeys = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32615)  # UTM zone 15N
    tifffile.imwrite(
        str(path), arr, photometric="rgb",
        extratags=[
            (33550, "d", 3, (1.0, 1.0, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, tiepoint_x, 4_800_000.0, 0.0), False),
            (34735, "H", len(geokeys), geokeys, False),
        ],
    )


_BLOCK_MODEL_SOURCE = {"builder": BESPOKE_DETECTION,
                       "builder_kwargs": {"min_size": TILE, "max_size": TILE * 2},
                       "task": "detection"}


def _completed_over(project: Path, data_cfg: dict, experiment_id: str) -> dict:
    """The run ``experiment_id`` over ``data_cfg`` in ``project``, run through the child's own
    entry with a body saving the model as built: its checkpoint path and the spatial manifest its
    resolved record carries."""
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import worker_run

    run_dir = worker_run(project, {
        "model_source": _BLOCK_MODEL_SOURCE, "data": data_cfg, "device": "cpu",
        "training_source": SAVE_BUILT_WEIGHTS,
    }, experiment_id=experiment_id)
    observation = observe(run_dir)
    checkpoint = observation.checkpoint
    assert checkpoint is not None, observation.final
    return {"checkpoint_path": checkpoint["path"],
            "spatial_manifest":
                observation.record["resolved"]["data"]["split"]["spatial_manifest"]}


def _build_experiment(tmp_path: Path, *, calibration_ratio: float = 0.15,
                      experiment_id: str = "exp_block",
                      plant_csv_paths: list[str] | None = None) -> dict:
    """A real 4-way spatial-strip split over a real raster, resolved and recorded by a real run
    whose completed checkpoint is the one the reserved regions are assessed for.

    ``plant_csv_paths`` (when given) writes a georeferenced mosaic (the plant-pitch derivation
    needs a real geotransform to convert real-world plant spacing to pixels) and threads the
    paths into ``data.plant_csv_paths``, the config field the band scale reads to prefer
    plant pitch over ground-truth object spacing.
    """
    root = tmp_path / "ds"
    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    raster_path = images_dir / f"{stem}.tif"
    _write_mosaic(raster_path, georeferenced=bool(plant_csv_paths))

    boxes = [Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110))
             for x in range(10, WIDTH - 20, BOX_STEP)]
    label_image(raster_path, boxes, WIDTH, HEIGHT, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir), "scope": {"subject": "bud"},
        "auto_val": True, "tiling": {"enabled": True, "tile_size": TILE, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "holdout_ratio": 0.15, "seed": 1,
                  "calibration_ratio": calibration_ratio},
    }
    if plant_csv_paths:
        data_cfg["plant_csv_paths"] = plant_csv_paths
    return {
        "project": tmp_path, "root": root, "images_dir": images_dir,
        "label": image_label_key(raster_path), "stem": stem, "raster_path": raster_path,
        "experiment_id": experiment_id, **_completed_over(tmp_path, data_cfg, experiment_id),
    }


def _attest_regions_complete(root: Path, stem: str, regions: list[list[tuple[int, int, int, int]]],
                             *, subject: str = "bud") -> None:
    """Mark every rect of ``regions`` complete for ``subject`` in the mosaic's label document,
    through the editor's own save door, one mark per rect."""
    from tests._producer_fixtures import mark_complete

    for x0, y0, x1, y1 in (r for region in regions for r in region):
        mark_complete(root / "images" / UNDATED_BUCKET / f"{stem}.tif", subject,
                      project=root.parent, rect=(x0, y0, x1 - x0, y1 - y0))


def _attested(tmp_path: Path, **kwargs) -> dict:
    """:func:`_build_experiment` with both reserved regions attested complete."""
    exp = _build_experiment(tmp_path, **kwargs)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["holdout_region"]])
    return exp


def _assess(exp: dict, *, k_cal: int | None = None, k_test: int | None = None,
            **stated) -> dict:
    """``assess_reserved_regions`` of the count trait's per-image count for ``exp``'s checkpoint
    under the ``stated`` execution values, answered as its door answers it."""
    from tcip_mcp.assessment import assess_reserved_regions
    from tcip_mcp.pipelines.block_calibration import DEFAULT_K_CAL, DEFAULT_K_TEST
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.calibration_tools import _answer

    return _answer(assess_reserved_regions(
        exp["project"], checkpoint_path=exp["checkpoint_path"], trait=fx.COUNT_TRAIT,
        delivery_kind="per_image_count", device="cpu", stated=Stated(**stated),
        k_cal=DEFAULT_K_CAL if k_cal is None else k_cal,
        k_test=DEFAULT_K_TEST if k_test is None else k_test))


def _band_total(record: dict) -> int:
    return sum(n for side in record["criterion"]["count"]["band_gt_counts"].values()
               for n in side.values())


# ── the completeness gate and the pass the bands run under ─────────────────


def test_an_assessment_refuses_while_the_reserved_regions_are_unattested(tmp_path: Path):
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path)

    with pytest.raises(AssessmentRefusedError, match="not marked complete"):
        _assess(exp)


def test_a_mark_over_one_region_leaves_the_other_unattested(tmp_path: Path):
    """The check reads each mark's rect: a mark over the calibration region attests that region
    alone, so the assessment refuses naming the holdout region and only it."""
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path)
    _attest_regions_complete(exp["root"], exp["stem"],
                             [exp["spatial_manifest"]["calibration_region"]])

    with pytest.raises(
            AssessmentRefusedError, match=r"\['holdout_region'\] are not marked complete"):
        _assess(exp)


def test_completeness_is_checked_before_feasibility(tmp_path: Path):
    """A region both incomplete and, at this band count, infeasible names the completeness gap:
    the feasibility message never gets a chance to fire."""
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path)

    with pytest.raises(AssessmentRefusedError) as exc_info:
        _assess(exp, k_cal=40, k_test=40)
    assert "not marked complete" in str(exc_info.value)
    assert "leaves only" not in str(exc_info.value)


def test_a_stated_tile_edge_other_than_the_splits_refuses_naming_both(tmp_path: Path):
    """The reserved regions and the published mosaic are tiled at one edge: a stated edge the run
    did not resolve refuses before any band is read, naming both."""
    exp = _attested(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        _assess(exp, tile_size=TILE * 2)
    assert str(TILE * 2) in str(exc_info.value) and str(TILE) in str(exc_info.value)


def test_every_band_runs_under_the_execution_record_the_assessment_records(
    tmp_path: Path, monkeypatch,
):
    """The bands are predicted under the stated slicing and merge and at the merge threshold, cap
    and floor the recorded execution states; the record the assessment keeps is the record the
    bands ran."""
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    exp = _attested(tmp_path)
    real = GenericPredictor.predict_sliced
    passes: list[dict] = []

    def recorded(self, view, **kwargs):
        passes.append({"execution": kwargs["execution"]})
        return real(self, view, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_sliced", recorded)

    record = _assess(exp, overlap=0.25, postprocess="greedynmm")

    execution = record["execution"]
    assert execution["overlap"] == 0.25 and execution["postprocess"] == "greedynmm"
    assert len(passes) == 6
    assert {(p["execution"].overlap, p["execution"].postprocess,
             p["execution"].cross_tile_nms) for p in passes} == {
        (0.25, "greedynmm", execution["cross_tile_nms"])}
    assert {p["execution"].max_dets for p in passes} == {execution["max_dets"]}
    assert {p["execution"].conf for p in passes} == {
        record["criterion"]["count"]["staged_conf_floor"]}


def test_the_document_retained_is_the_document_measured(tmp_path: Path, monkeypatch):
    """A document emptied after the assessment read it and before it retains it changes neither:
    the bands are measured from the one read, and that read is what the reference keeps."""
    import tcip_store

    from tcip_mcp import assessment
    from tcip_mcp.assessment import assessment_dir, read_assessment

    exp = _attested(tmp_path)
    real_retained = assessment._retained

    def emptied_first(run_dir, samples, *args):
        label_image(exp["raster_path"], [], WIDTH, HEIGHT, keep_empty=True)
        return real_retained(run_dir, samples, *args)

    monkeypatch.setattr(assessment, "_retained", emptied_first)
    record = _assess(exp)

    (kept,) = read_assessment(exp["project"], record["assessment_id"]).reference.ground_truth
    copy = assessment_dir(exp["project"], record["assessment_id"]) / kept.copy
    retained = json_io.label_document(tcip_store.decode_value(copy.read_bytes()))
    assert _band_total(record) > 0
    assert len(retained.annotations) > 0


def test_an_attested_mosaic_is_assessed_and_recorded(tmp_path: Path):
    """Once every reserved cell is attested complete, the call that refused above records an
    assessment: one reference sample per band on each side, ground truth in both, no band inside
    a training region, and the mosaic's content identity as its scope."""
    from tcip_mcp.assessment import read_assessment

    exp = _attested(tmp_path)

    record = _assess(exp)

    samples = read_assessment(tmp_path, record["assessment_id"]).reference.samples
    assert record["reference"]["n_samples"] == len(samples)
    assert sorted(s.side for s in samples) == ["calibration"] * 3 + ["holdout"] * 3
    counts = record["criterion"]["count"]["band_gt_counts"]
    assert sum(counts["calibration"].values()) > 0 and sum(counts["holdout"].values()) > 0
    assert record["disjointness"] == {
        "holdout_shares_calibration": [], "training": {"groups": [], "source_digests": []},
        "selection": {"groups": [], "source_digests": []}}
    assert "reference_shares_training" not in record["failures"]
    assert record["reference"]["raster_identity"]["width"] == WIDTH
    assert record["producer"]["experiment_id"] == exp["experiment_id"]


def test_a_run_with_no_reserved_region_refuses_naming_the_remedy(tmp_path: Path):
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path, calibration_ratio=0.0, experiment_id="exp_no_reserve")

    with pytest.raises(AssessmentRefusedError, match="calibration_ratio"):
        _assess(exp)


@pytest.mark.parametrize("width", [WIDTH + 500, WIDTH - 500], ids=["larger", "smaller"])
def test_a_recorded_mosaic_size_other_than_the_file_refuses_by_name(tmp_path: Path, width: int):
    """The run's recorded mosaic dimensions are checked against the raster read now: a manifest
    recorded against a replaced or truncated file refuses rather than scoring a sub-area or
    addressing pixels past the edge."""
    from tcip_store import encode_record

    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _attested(tmp_path)
    run_dir = experiment_dir(exp["experiment_id"], project=tmp_path)
    run = read_record(run_dir / RUN_FILE)
    run["resolved"]["data"]["split"]["spatial_manifest"]["width"] = width
    (run_dir / RUN_FILE).write_bytes(encode_record(run))

    with pytest.raises(AssessmentRefusedError, match="now reads"):
        _assess(exp)


# ── what the bands count and the scale and cap they derive ─────────────────


def test_band_counts_and_spacing_count_objects_not_crowd_regions(tmp_path: Path):
    """The same mosaic assessed twice, its boxes plain and then every other one marked a crowd
    region: the bands count fewer objects and the objects' spacing widens, since a crowd region is
    no object to count or to space."""
    records = []
    for crowd_every_other in (False, True):
        project = tmp_path / ("crowd" if crowd_every_other else "plain")
        project.mkdir()
        fx.seed_confirmed_count(project, measured_subject="bud")
        exp = _build_experiment(project, experiment_id=f"exp_{project.name}")
        label = exp["label"]
        boxes = json_io.read_label_document(label).annotations
        json_io.write_label_document(label, [
            Annotation(subject=a.subject, geometry=a.geometry,
                       iscrowd=crowd_every_other and i % 2 == 1) for i, a in enumerate(boxes)],
            WIDTH, HEIGHT, keep_empty=True)
        manifest = exp["spatial_manifest"]
        _attest_regions_complete(
            exp["root"], exp["stem"], [manifest["calibration_region"], manifest["holdout_region"]])
        records.append(_assess(exp))

    plain, crowd = records
    assert crowd["criterion"]["count"]["block_scale_px"] > plain["criterion"]["count"][
        "block_scale_px"]
    assert 0 < _band_total(crowd) < _band_total(plain)


def test_the_band_scale_prefers_plant_pitch_when_the_run_names_plant_files(tmp_path: Path):
    # Two plants ~10m apart, independent of the mosaic's own pixel geometry: the plant path only
    # needs the raster's real geotransform to convert this real-world spacing to pixels.
    plant_csv = write_plant_csv(tmp_path / "plants.csv", [
        {"plot": "P1", "accession": "acc-A", "lat": 45.0, "lon": -93.0},
        {"plot": "P2", "accession": "acc-B", "lat": 45.0, "lon": -93.0001}])
    exp = _attested(tmp_path, plant_csv_paths=[str(plant_csv)])

    source = _assess(exp)["criterion"]["count"]["block_scale_source"]

    assert source.startswith("plant grid pitch"), source


def test_the_band_scale_falls_back_to_object_spacing_with_no_plant_files(tmp_path: Path):
    source = _assess(_attested(tmp_path))["criterion"]["count"]["block_scale_source"]

    assert source.startswith("GT object-spacing"), source


def test_a_saturated_band_cap_surfaces_as_cap_saturated_provenance(tmp_path: Path, monkeypatch):
    """A band whose raw detection count reaches the applied cap is recorded as saturated: the
    derived cap is forced down to one, so every band with more than one raw detection is
    truncated."""
    import tcip_mcp.pipelines.derivations as derivations_module

    exp = _attested(tmp_path)
    monkeypatch.setattr(derivations_module, "derive_max_dets_from_counts", lambda *a, **k: 1)

    record = _assess(exp)

    assert record["execution"]["max_dets"] == 1
    assert record["criterion"]["count"]["calibration_cap_saturated_frac"] > 0.0


def test_the_recorded_cap_is_derived_from_the_calibration_bands_alone(tmp_path: Path):
    """The recorded cap is fitted on the calibration bands' object counts only: ground truth made
    dense inside the test region alone, which a pooled derivation would follow, leaves it where
    the calibration side puts it."""
    from tcip_mcp.pipelines.derivations import derive_max_dets_from_counts

    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    existing = json_io.read_label_document(exp["label"]).annotations
    tx0, _ty0, tx1, _ty1 = manifest["holdout_region"][0]
    dense = [Annotation(subject="bud", geometry=BBox(x, y, x + 15, y + 30))
             for x in range(int(tx0) + 5, int(tx1) - 20, 2) for y in (40, 80, 120)]
    json_io.write_label_document(exp["label"], existing + dense, WIDTH, HEIGHT, keep_empty=True)
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["holdout_region"]])

    record = _assess(exp)

    counts = record["criterion"]["count"]["band_gt_counts"]
    pooled = derive_max_dets_from_counts(
        list(counts["calibration"].values()) + list(counts["holdout"].values()))
    calibration_only = derive_max_dets_from_counts(list(counts["calibration"].values()))
    assert pooled != calibration_only
    assert record["execution"]["max_dets"] == calibration_only


def test_an_unstated_merge_threshold_derives_from_the_calibration_bands(tmp_path: Path):
    """Each object gains an overlapping neighbor, so the ground-truth tail derives a merge
    threshold; the record states it as derived, never the documented default."""
    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    json_io.write_label_document(exp["label"], [
        Annotation(subject="bud", geometry=BBox(x + dx, 80 + dx, x + 15 + dx, 110 + dx))
        for x in range(10, WIDTH - 20, BOX_STEP) for dx in (0, 5)], WIDTH, HEIGHT,
        keep_empty=True)
    _attest_regions_complete(
        exp["root"], exp["stem"], [manifest["calibration_region"], manifest["holdout_region"]])

    record = _assess(exp)

    assert record["execution"]["sources"]["cross_tile_nms"] != "default"


# ── the raster door without an assessment ─────────────────────────────────


def test_a_raster_pass_stamps_an_explicit_conf_as_explicit_and_an_omitted_one_as_default(
    tmp_path: Path,
):
    """A stated conf equal to the documented default is recorded as stated, never laundered into
    the default, and an omitted one runs at the default, recorded so."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.execution import DEFAULT_CONF, Stated
    from tcip_mcp.tools.inference_tools import run_inference

    exp = _build_experiment(tmp_path, calibration_ratio=0.0, experiment_id="exp_conf")
    for name, stated in (("stated", {"conf": DEFAULT_CONF}), ("omitted", {})):
        result = run_inference(tmp_path, exp["checkpoint_path"], bucket=name,
                               raster_path=str(exp["raster_path"]),
                               stated=Stated(tile_size=TILE, overlap=0.2, **stated))
        assert "error" not in result, result
        execution = read_bucket(exp["root"], name).execution
        assert execution.conf == DEFAULT_CONF
        assert execution.sources["conf"] == ("explicit" if stated else "default")


# ── the band geometry ─────────────────────────────────────────────────────


def test_band_ground_truth_keeps_what_is_centered_in_the_band_at_full_extent():
    """A box centered inside the band is kept at full extent, translated to local coordinates; a
    box centered outside is dropped, even one that overlaps the band substantially."""
    import numpy as np

    from tcip_mcp.pipelines.block_calibration import select_gt_for_band

    band = (100, 100, 200, 200)
    boxes = np.array([
        [120.0, 120.0, 140.0, 140.0],   # center (130,130): inside, kept at full extent
        [190.0, 120.0, 260.0, 140.0],   # center (225,130): outside, past x1=200
        [50.0, 90.0, 110.0, 150.0],     # center (80,120): outside, before x0=100
    ])
    gt = {"boxes": boxes, "labels": np.array([1, 2, 3]),
          "iscrowd": np.array([True, False, False])}

    kept = select_gt_for_band(gt, band)

    assert kept["labels"].tolist() == [1]
    assert kept["iscrowd"].tolist() == [True]  # the flag travels with its row
    np.testing.assert_array_equal(kept["boxes"], np.array([[20.0, 20.0, 40.0, 40.0]]))

    empty = {"boxes": np.zeros((0, 4), dtype=np.float32), "labels": np.zeros((0,), dtype=np.int64),
             "iscrowd": np.zeros((0,), dtype=bool)}
    assert all(len(v) == 0 for v in select_gt_for_band(empty, (0, 0, 10, 10)).values())


def test_band_records_carry_each_ground_truth_rows_crowd_flag(tmp_path: Path):
    """A band's reference forms a crowd region's ground truth as a crowd region, read off the row
    it came from, never as one more object in the band, each box on the stored grid its document
    holds."""
    from types import SimpleNamespace

    import numpy as np

    from tcip_mcp.assessment import _band_records
    from tcip_mcp.pipelines.data.label_queries import json_det_targets, registry_scope
    from tcip_mcp.pipelines.data.selection import Sample

    class _OneDetection:
        def predict_sliced(self, view, **kwargs):
            return {"boxes": [[10.1, 10.1, 40.3, 30.3]], "scores": [0.9], "labels": [1],
                    "cap_hit": False}

    label = image_label_key(tmp_path / "images" / UNDATED_BUCKET / "mosaic.tif")
    json_io.write_label_document(label, [
        Annotation(subject="bur", geometry=BBox(20.3, 20.7, 40.1, 60.9)),
        Annotation(subject="bur", geometry=BBox(100.0, 100.0, 180.0, 180.0), iscrowd=True)],
        200, 200)
    target = json_det_targets(json_io.read_label_document(label).annotations,
                              registry_scope(tmp_path / "images", "bur"))
    gt = {"boxes": np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4),
          "labels": np.asarray(target["labels"], dtype=np.int64),
          "iscrowd": np.asarray(target["iscrowd"], dtype=bool)}
    stub = SimpleNamespace(predictor=_OneDetection(), tile_batch_size=8)
    band = Sample(member="a", source="mosaic.tif", ground_truth=label, group="a", side="",
                  rect=(0, 0, 200, 200))

    records = _band_records(SimpleNamespace(height=200, width=200, num_channels=3),
                            {"a": (0, 0, 200, 200)}, stub,
                            SimpleNamespace(tile_size=64, overlap=0.2), gt=gt,
                            digest_of={band.location: "band-digest"}, source="mosaic.tif")

    assert [g["iscrowd"] for g in records[0]["gt"]] == [0, 1]
    assert records[0]["gt"][0]["bbox"] == [20.3, 20.7, 19.8, 40.2]
    assert [d["bbox"] for d in records[0]["dt"]] == [[10.1, 10.1, 30.2, 20.2]]
    assert records[0]["image_id"] == "band-digest"


def test_band_rects_are_reported_in_full_mosaic_coordinates():
    """Sub-banding recurses the strip split over the region's own local extent, so every returned
    rect is translated back by the region's origin: a region not starting at (0, 0) is the only
    fixture that tells the two apart."""
    from tcip_mcp.pipelines.block_calibration import band_rects

    region = (500, 60, 1300, 260)
    rx0, ry0, rx1, ry1 = region
    bands = band_rects(region, 3, TILE, 0.2, 40, "cal")

    assert len(bands) == 3
    for name, (bx0, by0, bx1, by1) in bands.items():
        assert rx0 <= bx0 < bx1 <= rx1, name
        assert ry0 <= by0 < by1 <= ry1, name
    ordered = sorted(bands.values())
    for (_x0, _y0, x1, _y1), (nx0, _ny0, _nx1, _ny1) in zip(ordered, ordered[1:]):
        assert x1 <= nx0


def test_density_outlier_bands_are_flagged_against_their_own_siblings():
    """A band far sparser or far denser than its siblings is named; empty bands carry no density
    signal and are never flagged, since the feasibility gate speaks for them."""
    from tcip_mcp.pipelines.block_calibration import density_uniformity_flags

    skewed = {"cal_0": 10, "cal_1": 11, "cal_2": 1, "cal_3": 60, "cal_4": 0}
    assert density_uniformity_flags(skewed) == ["cal_2", "cal_3"]
    assert density_uniformity_flags({"cal_0": 10, "cal_1": 11, "cal_2": 9}) == []


def test_feasibility_counts_only_bands_that_carry_ground_truth():
    from tcip_mcp.assessment import AssessmentRefusedError
    from tcip_mcp.pipelines.block_calibration import check_feasibility

    with pytest.raises(AssessmentRefusedError, match="leaves only 1 band"):
        check_feasibility({"test_0": 0, "test_1": 7, "test_2": 0}, side="test")
    check_feasibility({"cal_0": 4, "cal_1": 0, "cal_2": 9}, side="cal")


# ── the class space and the breeder's own attestation ─────────────────────


def _build_attribute_scoped_experiment(
    tmp_path: Path, *, trained_values: tuple[str, ...], reordered_values: tuple[str, ...],
    labeled_value: str, experiment_id: str = "exp_block_attribute",
) -> dict:
    """An experiment whose subject declares a categorical attribute, with the dataset's registry
    reordered after the run resolved and recorded its own attributes."""
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry
    from tests._producer_fixtures import registry_over

    def _write_registry(values: tuple[str, ...]) -> None:
        registry_over(root, SubjectRegistry(subjects=(
            Subject(name="bud", attributes=(
                Attribute(name="stage", type="categorical", values=values),)),)))

    root = tmp_path / "ds_attribute"
    images_dir = root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    _write_mosaic(images_dir / f"{stem}.tif")
    _write_registry(trained_values)

    boxes = [Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110),
                        attributes={"stage": labeled_value})
             for x in range(10, WIDTH - 20, BOX_STEP)]
    label_image(images_dir / f"{stem}.tif", boxes, WIDTH, HEIGHT, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir), "scope": {"subject": "bud"}, "auto_val": True,
        "tiling": {"enabled": True, "tile_size": TILE, "overlap": 0.2},
        "split": {"val_ratio": 0.2, "holdout_ratio": 0.15, "seed": 1,
                  "calibration_ratio": 0.15},
    }
    completed = _completed_over(tmp_path, data_cfg, experiment_id)
    recorded_scope = ClassScope.of(run_resolution(experiment_id, project=tmp_path)["data"])
    assert recorded_scope.subject and recorded_scope.attributes

    _write_registry(reordered_values)
    manifest = completed["spatial_manifest"]
    _attest_regions_complete(root, stem, [manifest["calibration_region"],
                                          manifest["holdout_region"]])
    return {"project": tmp_path, "root": root, "stem": stem,
            "experiment_id": experiment_id, **completed, "recorded_scope": recorded_scope}


def test_ground_truth_decodes_through_the_checkpoints_own_recorded_attributes(tmp_path: Path):
    """The mosaic's ground truth is read through the attributes the training run recorded, never
    a live re-read of the dataset's registry: the registry reorders the values after the run, and
    the assessment still reads the reference in the order the run trained."""
    from tcip_mcp.subject_registry import read_registry

    exp = _build_attribute_scoped_experiment(
        tmp_path, trained_values=("closed", "open", "shed"),
        reordered_values=("open", "closed", "shed"), labeled_value="open")
    live = read_registry(exp["root"]).subjects[0].attributes
    assert exp["recorded_scope"].attributes[0].values == ("closed", "open", "shed")
    assert live[0].values != exp["recorded_scope"].attributes[0].values

    assert _band_total(_assess(exp)) > 0


def test_a_recorded_scope_needs_no_registry_on_disk(tmp_path: Path):
    exp = _build_attribute_scoped_experiment(
        tmp_path, trained_values=("closed", "open", "shed"),
        reordered_values=("closed", "open", "shed"), labeled_value="open",
        experiment_id="exp_block_recorded_no_registry")
    (exp["root"] / "subjects.json").unlink()

    assert _band_total(_assess(exp)) > 0


def test_regions_marked_through_the_editors_save_admit_the_assessment(tmp_path: Path, client):
    """The completeness gate reads the record the breeder's own gesture writes: each reserved
    region marked complete through the save the Annotate canvas posts to."""
    from tests._web_fixtures import open_new_project

    exp = _build_experiment(tmp_path)
    open_new_project(tmp_path)
    manifest = exp["spatial_manifest"]
    for region in (manifest["calibration_region"], manifest["holdout_region"]):
        x0, y0, x1, y1 = region[0]
        current = client.get("/api/annotate/labels", params={
            "image_path": str(exp["raster_path"])}).json()
        resp = client.post("/api/annotate/labels", json={
            "image_path": str(exp["raster_path"]),
            "annotations": current["annotations"], "base_mtime": current["base_mtime"],
            "user": "breeder", "complete": {"bud": True}, "rect": [x0, y0, x1 - x0, y1 - y0]})
        assert resp.status_code == 200, resp.text

    assert _band_total(_assess(exp)) > 0

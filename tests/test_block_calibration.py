"""A mosaic's own reserved calibration and test regions, assessed directly.

``assessment.assess_reserved_regions`` assesses them without a separate held-out image set.

Covers: the completion-mark gate (refuses unmarked, admits marked), the sub-banding and
halo running real tiled inference over real bands under the one execution record the assessment
records, the geometric disjointness check, and the band geometry helpers.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient

    from tcip_mcp.assessment import RetainedFile
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import Sample, Side
    from tcip_mcp.pipelines.execution import Execution, MaskCut, Pass
    from tcip_mcp.pipelines.inference.generic_predictor import (
        GenericPredictor,
        WindowedRasterReader,
    )
    from tcip_mcp.pipelines.raster_source import RasterSource

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.dataset_layout import (  # noqa: E402
    UNDATED_BUCKET,
    image_dir,
    image_root,
)
from tests import _trait_fixtures as fx  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
    SAVE_BUILT_WEIGHTS,
    training_config,
)
from tests._mapping_fixtures import write_plant_csv  # noqa: E402
from tests._producer_fixtures import image_label_key, label_image  # noqa: E402
from tests._training_values import (  # noqa: E402
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    RATIONALE,
    collection_pass,
    drawn_split,
    tiling_block,
)

TILE = 32
OVERLAP = 0.2
WIDTH, HEIGHT = 3200, 200
BOX_STEP = 40


@pytest.fixture(autouse=True)
def _count_trait(tmp_path: Path) -> None:
    """Confirm the count trait in this test's project, measuring the mosaic's subject."""
    fx.seed_confirmed_count(tmp_path, measured_subject="bud")


def _write_mosaic(
    path: Path,
    *,
    seed: int = 0,
    georeferenced: bool = False,
    tiepoint_x: float = 500_000.0,
) -> None:
    import numpy as np
    import tifffile

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    if not georeferenced:
        tifffile.imwrite(str(path), arr)
        return
    geokeys = (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32615)  # UTM zone 15N
    tifffile.imwrite(
        str(path),
        arr,
        photometric="rgb",
        extratags=[
            (33550, "d", 3, (1.0, 1.0, 0.0), False),
            (
                33922,
                "d",
                6,
                (0.0, 0.0, 0.0, tiepoint_x, 4_800_000.0, 0.0),
                False,
            ),
            (34735, "H", len(geokeys), geokeys, False),
        ],
    )


_BLOCK_MODEL_SOURCE = {
    "builder": BESPOKE_DETECTION,
    "source_files": [BESPOKE_MODELS],
    "task": "detection",
    "normalization": FRAME_STATISTICS,
}


def _completed_over(project: Path, data_cfg: dict, experiment_id: str) -> dict:
    """Run ``experiment_id`` over ``data_cfg`` in ``project`` through the child's own entry.

    The run's body saves the model as built. Returns its checkpoint path and the spatial
    manifest its resolved record carries.
    """
    from tcip_mcp.experiments import observe_run
    from tests._verified_checkpoint_fixtures import worker_run

    run_dir = worker_run(
        project,
        training_config(
            project,
            _BLOCK_MODEL_SOURCE,
            data_cfg,
            training_source=SAVE_BUILT_WEIGHTS,
        ),
        experiment_id=experiment_id,
    )
    observation = observe_run(run_dir)
    checkpoint = observation.checkpoint
    assert checkpoint is not None, observation.final
    return {
        "checkpoint_path": checkpoint.path,
        "spatial_manifest": observation.record.resolution.partition.spatial,
    }


def _build_experiment(
    tmp_path: Path,
    *,
    calibration_ratio: float = 0.15,
    experiment_id: str = "exp_block",
    plant_csv_paths: list[str] | None = None,
) -> dict:
    """Write and label a mosaic, run :func:`_completed_over` over it, and return what it left.

    The mosaic is ``_write_mosaic``'s, georeferenced when ``plant_csv_paths`` is given, labeled
    with a row of 15 by 30 ``bud`` boxes. The run's data section tiles at ``TILE`` with overlap
    0.2 and draws a tile-prefix split at seed 1 with validation ratio 0.2, holdout ratio 0.15 and
    ``calibration_ratio``, stating ``plant_csv_paths`` when given.
    """
    root = tmp_path / "ds"
    images_dir = image_dir(root, UNDATED_BUCKET)
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    raster_path = images_dir / f"{stem}.tif"
    _write_mosaic(raster_path, georeferenced=bool(plant_csv_paths))

    boxes = [
        Annotation(subject="bud", geometry=BBox(x, 80, x + 15, 110))
        for x in range(10, WIDTH - 20, BOX_STEP)
    ]
    label_image(raster_path, boxes, WIDTH, HEIGHT, keep_empty=True)

    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": tiling_block(tile_size=TILE, overlap=0.2),
        "split": drawn_split(
            val_ratio=0.2,
            holdout_ratio=0.15,
            seed=1,
            calibration_ratio=calibration_ratio,
            grouping="tile_prefix",
        ),
    }
    if plant_csv_paths:
        data_cfg["plant_csv_paths"] = plant_csv_paths
    return {
        "project": tmp_path,
        "root": root,
        "images_dir": images_dir,
        "label": image_label_key(raster_path),
        "stem": stem,
        "raster_path": raster_path,
        "experiment_id": experiment_id,
        **_completed_over(tmp_path, data_cfg, experiment_id),
    }


def _attest_regions_complete(
    root: Path,
    stem: str,
    regions: list[list[tuple[int, int, int, int]]],
    *,
    subject: str = "bud",
) -> None:
    """Mark every rect of ``regions`` complete for ``subject`` in the mosaic's label document.

    Each mark goes through ``_producer_fixtures.mark_complete``, one mark per rect.
    """
    from tests._producer_fixtures import mark_complete

    for x0, y0, x1, y1 in (r for region in regions for r in region):
        mark_complete(
            image_dir(root, UNDATED_BUCKET) / f"{stem}.tif",
            subject,
            project=root.parent,
            rect=(x0, y0, x1 - x0, y1 - y0),
        )


def _attested(
    tmp_path: Path, **kwargs: float | str | list[str] | None
) -> dict:
    """Run :func:`_build_experiment` and attest both reserved regions complete; return it."""
    exp = _build_experiment(tmp_path, **kwargs)
    manifest = exp["spatial_manifest"]
    _attest_regions_complete(
        exp["root"],
        exp["stem"],
        [manifest.calibration_region, manifest.holdout_region],
    )
    return exp


def _assess(
    exp: dict,
    *,
    k_cal: int = 3,
    k_test: int = 3,
    **stated: bool | int | float | str | MaskCut | None,
) -> dict:
    """Assess the count trait's per-image count over ``exp``'s reserved regions.

    ``assess_reserved_regions`` runs for ``exp``'s checkpoint in ``k_cal`` and ``k_test`` bands
    under the sample collection pass with the ``stated`` execution values over it (an NMS merge
    at the sample threshold unless they name one, ``None`` leaving the threshold to the
    calibration bands), at the sample count test settings, answered through
    ``calibration_tools._answer``.
    """
    from tcip_mcp.assessment import assess_reserved_regions
    from tcip_mcp.tools.calibration_tools import _answer
    from tests._verified_checkpoint_fixtures import SAMPLE_CROSS_TILE_NMS

    return _answer(
        assess_reserved_regions(
            exp["project"],
            checkpoint_path=exp["checkpoint_path"],
            trait=fx.COUNT_TRAIT,
            delivery_kind="per_image_count",
            device="cpu",
            stated=collection_pass(
                **{
                    "postprocess": "nms",
                    "cross_tile_nms": SAMPLE_CROSS_TILE_NMS,
                    **stated,
                }
            ),
            k_cal=k_cal,
            k_test=k_test,
            equivalence_z=EQUIVALENCE_Z,
            dispersion_quantile=DISPERSION_QUANTILE,
            rationale=RATIONALE,
            actor=None,
        )
    )


def _band_total(record: dict) -> int:
    return sum(
        n
        for side in record["criterion"]["count"]["band_gt_counts"].values()
        for n in side.values()
    )


# ── the completeness gate and the pass the bands run under ─────────────────


def test_an_assessment_refuses_while_the_reserved_regions_are_unattested(
    tmp_path: Path,
) -> None:
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path)

    with pytest.raises(AssessmentRefusedError, match="not marked complete"):
        _assess(exp)


def test_a_mark_over_one_region_leaves_the_other_unattested(
    tmp_path: Path,
) -> None:
    """A mark over the calibration region leaves the holdout region unattested and named.

    With the calibration region alone marked, the assessment refuses naming only the holdout
    region as not marked complete.
    """
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path)
    _attest_regions_complete(
        exp["root"],
        exp["stem"],
        [exp["spatial_manifest"].calibration_region],
    )

    with pytest.raises(
        AssessmentRefusedError,
        match=r"\['holdout_region'\] are not marked complete",
    ):
        _assess(exp)


def test_completeness_is_checked_before_feasibility(tmp_path: Path) -> None:
    """An incomplete and infeasible case reports the completeness gap, not feasibility.

    At this band count the regions are both unmarked and infeasible; the refusal names them as
    not marked complete and carries no feasibility message.
    """
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(tmp_path)

    with pytest.raises(AssessmentRefusedError) as exc_info:
        _assess(exp, k_cal=40, k_test=40)
    assert "not marked complete" in str(exc_info.value)
    assert "leaves only" not in str(exc_info.value)


def test_a_stated_tile_edge_other_than_the_splits_refuses_naming_both(
    tmp_path: Path,
) -> None:
    """A stated tile edge other than the split's refuses, naming both.

    Stating twice the run's tile edge raises ``ValueError`` naming the stated and the resolved
    edge.
    """
    exp = _attested(tmp_path)

    with pytest.raises(ValueError) as exc_info:
        _assess(exp, tile_size=TILE * 2)
    assert str(TILE * 2) in str(exc_info.value) and str(TILE) in str(
        exc_info.value
    )


def test_every_band_runs_under_the_execution_record_the_assessment_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bands' passes agree with the assessment's recorded execution on five fields.

    Across the six recorded passes, overlap, postprocessor and merge threshold equal the stated
    and recorded ones, density equals the recorded density, and conf equals the recorded floor.
    """
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    exp = _attested(tmp_path)
    real = GenericPredictor.predict_sliced
    passes: list[dict] = []

    def recorded(
        self: GenericPredictor,
        view: str | Path | BandGroupRef | WindowedRasterReader,
        **kwargs: Execution
        | int
        | str
        | dict
        | Callable[[int, int, dict], None]
        | None,
    ) -> dict:
        passes.append({"execution": kwargs["execution"]})
        return real(self, view, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_sliced", recorded)

    record = _assess(
        exp, overlap=0.25, postprocess="greedynmm", cross_tile_nms=0.5
    )

    execution = record["execution"]
    assert (
        execution["overlap"] == 0.25
        and execution["postprocess"] == "greedynmm"
    )
    assert execution["cross_tile_nms"] == 0.5
    assert len(passes) == 6
    assert {
        (
            p["execution"].overlap,
            p["execution"].postprocess,
            p["execution"].cross_tile_nms,
        )
        for p in passes
    } == {(0.25, "greedynmm", execution["cross_tile_nms"])}
    assert {p["execution"].density for p in passes} == {execution["density"]}
    assert {p["execution"].conf for p in passes} == {
        record["criterion"]["count"]["staged_conf_floor"]
    }


def test_the_document_retained_is_the_document_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source document emptied just before retention leaves both measurement and copy nonempty.

    The label document is emptied when the assessment reaches its retention step; the bands still
    count ground truth, and the retained copy still holds annotations.
    """
    import tcip_store
    from tcip_mcp import assessment
    from tcip_mcp.assessment import assessment_dir, read_assessment

    exp = _attested(tmp_path)
    real_retained = assessment._retained

    def emptied_first(
        run_dir: Path, samples: list[Sample], *args: dict
    ) -> tuple[list[RetainedFile], list[Sample]]:
        label_image(exp["raster_path"], [], WIDTH, HEIGHT, keep_empty=True)
        return real_retained(run_dir, samples, *args)

    monkeypatch.setattr(assessment, "_retained", emptied_first)
    record = _assess(exp)

    (kept,) = read_assessment(
        exp["project"], record["assessment_id"]
    ).reference.ground_truth
    copy = assessment_dir(exp["project"], record["assessment_id"]) / kept.copy
    retained = json_io.label_document(
        tcip_store.decode_value(copy.read_bytes())
    )
    assert _band_total(record) > 0
    assert len(retained.annotations) > 0


def test_an_attested_mosaic_is_assessed_and_recorded(tmp_path: Path) -> None:
    """An attested mosaic is assessed and recorded.

    With both reserved regions attested complete, the record holds three reference samples on
    each of the calibration and holdout sides, ground truth counted on both, no disjointness
    finding, the mosaic's raster width and the producing run.
    """
    from tcip_mcp.assessment import read_assessment

    exp = _attested(tmp_path)

    record = _assess(exp)

    samples = [
        s.sample
        for s in read_assessment(
            tmp_path, record["assessment_id"]
        ).reference.samples
    ]
    assert record["reference"]["n_samples"] == len(samples)
    assert (
        sorted(s.side for s in samples)
        == ["calibration"] * 3 + ["holdout"] * 3
    )
    counts = record["criterion"]["count"]["band_gt_counts"]
    assert (
        sum(counts["calibration"].values()) > 0
        and sum(counts["holdout"].values()) > 0
    )
    assert record["disjointness"] == {
        "holdout_shares_calibration": [],
        "training": {"groups": [], "source_digests": []},
        "selection": {"groups": [], "source_digests": []},
    }
    assert "reference_shares_training" not in record["failures"]
    assert record["reference"]["raster_identity"]["width"] == WIDTH
    assert record["producer"]["experiment_id"] == exp["experiment_id"]


def test_a_run_with_no_reserved_region_refuses_naming_the_remedy(
    tmp_path: Path,
) -> None:
    from tcip_mcp.assessment import AssessmentRefusedError

    exp = _build_experiment(
        tmp_path, calibration_ratio=0.0, experiment_id="exp_no_reserve"
    )

    with pytest.raises(AssessmentRefusedError, match="calibration_ratio"):
        _assess(exp)


@pytest.mark.parametrize(
    "width", [WIDTH + 500, WIDTH - 500], ids=["larger", "smaller"]
)
def test_a_recorded_mosaic_size_other_than_the_file_refuses_by_name(
    tmp_path: Path, width: int
) -> None:
    """A recorded mosaic size other than the file's refuses by name.

    The run's recorded mosaic dimensions are checked against the raster as read at assessment: a
    recorded width larger or smaller than the file's refuses.
    """
    from tcip_mcp.assessment import AssessmentRefusedError
    from tcip_mcp.experiments import RUN_FILE, experiment_dir, read_record
    from tcip_store import encode_record

    exp = _attested(tmp_path)
    run_dir = experiment_dir(exp["experiment_id"], project=tmp_path)
    run = read_record(run_dir / RUN_FILE)
    assert isinstance(run, dict)
    resolved = run["resolved"]
    assert isinstance(resolved, dict)
    partition = resolved["partition"]
    assert isinstance(partition, dict)
    spatial = partition["spatial"]
    assert isinstance(spatial, dict)
    spatial["width"] = width
    (run_dir / RUN_FILE).write_bytes(encode_record(run))

    with pytest.raises(AssessmentRefusedError, match="now reads"):
        _assess(exp)


# ── what the bands count and the scale and cap they derive ─────────────────


def test_band_counts_and_spacing_count_objects_not_crowd_regions(
    tmp_path: Path,
) -> None:
    """Band counts and object spacing count objects, never crowd regions.

    The same mosaic is assessed twice, its boxes plain and then every other one marked a crowd
    region: with the crowd regions, the bands count fewer objects and the objects' spacing
    widens.
    """
    records = []
    for crowd_every_other in (False, True):
        project = tmp_path / ("crowd" if crowd_every_other else "plain")
        project.mkdir()
        fx.seed_confirmed_count(project, measured_subject="bud")
        exp = _build_experiment(project, experiment_id=f"exp_{project.name}")
        label = exp["label"]
        boxes = json_io.read_label_document(label).annotations
        json_io.write_label_document(
            label,
            [
                Annotation(
                    subject=a.subject,
                    geometry=a.geometry,
                    iscrowd=crowd_every_other and i % 2 == 1,
                )
                for i, a in enumerate(boxes)
            ],
            WIDTH,
            HEIGHT,
            keep_empty=True,
        )
        manifest = exp["spatial_manifest"]
        _attest_regions_complete(
            exp["root"],
            exp["stem"],
            [manifest.calibration_region, manifest.holdout_region],
        )
        records.append(_assess(exp))

    plain, crowd = records
    assert (
        crowd["criterion"]["count"]["block_scale_px"]
        > plain["criterion"]["count"]["block_scale_px"]
    )
    assert 0 < _band_total(crowd) < _band_total(plain)


def test_the_band_scale_prefers_plant_pitch_when_the_run_names_plant_files(
    tmp_path: Path,
) -> None:
    # Two plants ~10m apart, independent of the mosaic's own pixel geometry: the plant path only
    # needs the raster's real geotransform to convert this real-world spacing to pixels.
    plant_csv = write_plant_csv(
        tmp_path / "plants.csv",
        [
            {"plot": "P1", "accession": "acc-A", "lat": 45.0, "lon": -93.0},
            {"plot": "P2", "accession": "acc-B", "lat": 45.0, "lon": -93.0001},
        ],
    )
    exp = _attested(tmp_path, plant_csv_paths=[str(plant_csv)])

    source = _assess(exp)["criterion"]["count"]["block_scale_source"]

    assert source.startswith("plant grid pitch"), source


def test_the_band_scale_falls_back_to_object_spacing_with_no_plant_files(
    tmp_path: Path,
) -> None:
    source = _assess(_attested(tmp_path))["criterion"]["count"][
        "block_scale_source"
    ]

    assert source.startswith("GT object-spacing"), source


def test_a_saturated_band_cap_surfaces_as_cap_saturated_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A band whose raw detection count reaches the applied cap is recorded as saturated.

    The derived density is forced low enough that the applied cap is reached, and the
    calibration side records a saturated fraction above zero.
    """
    import tcip_mcp.pipelines.derivations as derivations_module

    exp = _attested(tmp_path)
    monkeypatch.setattr(
        derivations_module, "derive_object_density", lambda *a, **k: 1e-12
    )

    record = _assess(exp)

    assert record["execution"]["density"] == 1e-12
    assert record["criterion"]["count"]["calibration_cap_saturated_frac"] > 0.0


def test_the_recorded_density_is_derived_from_the_calibration_bands_alone(
    tmp_path: Path,
) -> None:
    """The recorded density is derived from the calibration bands alone.

    Ground truth is made dense inside the test region alone, so a derivation pooled over both
    sides differs from the calibration-only one, and the recorded density equals the latter.
    """
    from tcip_mcp.pipelines.derivations import derive_object_density

    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    existing = json_io.read_label_document(exp["label"]).annotations
    tx0, _ty0, tx1, _ty1 = manifest.holdout_region[0]
    dense = [
        Annotation(subject="bud", geometry=BBox(x, y, x + 15, y + 30))
        for x in range(int(tx0) + 5, int(tx1) - 20, 2)
        for y in (40, 80, 120)
    ]
    json_io.write_label_document(
        exp["label"], existing + dense, WIDTH, HEIGHT, keep_empty=True
    )
    _attest_regions_complete(
        exp["root"],
        exp["stem"],
        [manifest.calibration_region, manifest.holdout_region],
    )

    record = _assess(exp)

    from tcip_mcp.assessment import read_assessment

    counts = record["criterion"]["count"]["band_gt_counts"]
    rects = {
        s.sample.member: s.sample.rect
        for s in read_assessment(
            exp["project"], record["assessment_id"]
        ).reference.samples
    }

    from tests._verified_checkpoint_fixtures import objects_over

    def counted(side: str) -> list:
        out = []
        for name, n in counts[side].items():
            rect = rects[name]
            assert rect is not None, name
            out.append(
                objects_over(
                    [[0.0, 0.0, 1.0, 1.0]] * n,
                    (rect[2] - rect[0]) * (rect[3] - rect[1]),
                )
            )
        return out

    pooled = derive_object_density(counted("calibration") + counted("holdout"))
    calibration_only = derive_object_density(counted("calibration"))
    assert pooled != calibration_only
    assert record["execution"]["density"] == pytest.approx(calibration_only)


def test_the_recorded_density_caps_the_whole_mosaic_at_its_own_pixels(
    tmp_path: Path,
) -> None:
    """The cap the recorded density gives the whole mosaic covers the mosaic's objects.

    The mosaic holds more objects than its densest calibration band, and the detection cap of the
    recorded density over the mosaic's own pixels is at least that object count.
    """
    from tcip_mcp.pipelines.derivations import detection_cap

    exp = _attested(tmp_path)
    record = _assess(exp)

    in_mosaic = len(json_io.read_label_document(exp["label"]).annotations)
    densest_band = max(
        record["criterion"]["count"]["band_gt_counts"]["calibration"].values()
    )
    assert in_mosaic > densest_band
    assert (
        detection_cap(record["execution"]["density"], WIDTH * HEIGHT)
        >= in_mosaic
    )


def test_an_unstated_merge_threshold_derives_from_the_calibration_bands(
    tmp_path: Path,
) -> None:
    """An unstated merge threshold is recorded with the cross-tile NMS derivation as its source.

    Each object in the ground truth has an overlapping neighbor.
    """
    from tcip_mcp.pipelines.derivations import CROSS_TILE_NMS_DERIVATION

    exp = _build_experiment(tmp_path)
    manifest = exp["spatial_manifest"]
    json_io.write_label_document(
        exp["label"],
        [
            Annotation(
                subject="bud",
                geometry=BBox(x + dx, 80 + dx, x + 15 + dx, 110 + dx),
            )
            for x in range(10, WIDTH - 20, BOX_STEP)
            for dx in (0, 5)
        ],
        WIDTH,
        HEIGHT,
        keep_empty=True,
    )
    _attest_regions_complete(
        exp["root"],
        exp["stem"],
        [manifest.calibration_region, manifest.holdout_region],
    )

    record = _assess(exp, cross_tile_nms=None)

    assert (
        record["execution"]["sources"]["cross_tile_nms"]
        == CROSS_TILE_NMS_DERIVATION
    )


def test_the_assessment_records_the_band_counts_and_test_settings_it_ran_under(
    tmp_path: Path,
) -> None:
    """The assessment records the band counts and test settings it ran under, none defaulted.

    The band counts, the equivalence multiplier and the dispersion quantile an assessment states
    are each recorded on it beside its reason, and the door takes none of them by default.
    """
    import inspect

    from tcip_mcp.assessment import assess_reserved_regions, read_assessment
    from tcip_mcp.tools import calibration_tools

    exp = _attested(tmp_path)
    answered = _assess(exp, k_cal=2, k_test=3)
    assert "error" not in answered, answered
    record = read_assessment(exp["project"], answered["assessment_id"])

    count = record.criterion["count"]
    assert isinstance(count, dict)
    assert (count["k_cal"], count["k_test"]) == (2, 3)
    assert count["equivalence_z"] == EQUIVALENCE_Z
    curve = count["calibration_curve"]
    assert isinstance(curve, dict)
    assert curve["dispersion_quantile"] == DISPERSION_QUANTILE
    assert record.rationale == RATIONALE
    for door in (
        assess_reserved_regions,
        calibration_tools.assess_reserved_regions,
    ):
        parameters = inspect.signature(door).parameters
        for name in (
            "k_cal",
            "k_test",
            "equivalence_z",
            "dispersion_quantile",
            "rationale",
        ):
            assert parameters[name].default is inspect.Parameter.empty, (
                door,
                name,
            )


# ── the raster door without an assessment ─────────────────────────────────


def test_a_raster_pass_stamps_its_stated_conf_and_refuses_an_omitted_one(
    tmp_path: Path,
) -> None:
    """A raster pass records the conf it states, and one stating none is refused naming it.

    A raster pass without an assessment records the conf it states with an explicit source, and
    one stating none refuses naming ``conf``.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF,
        SAMPLE_DETECTOR_PASS,
    )

    exp = _build_experiment(
        tmp_path, calibration_ratio=0.0, experiment_id="exp_conf"
    )
    unstated_conf = {
        k: v for k, v in SAMPLE_DETECTOR_PASS.items() if k != "conf"
    }
    for name, stated in (
        ("stated", SAMPLE_DETECTOR_PASS),
        ("omitted", unstated_conf),
    ):
        result = run_inference(
            tmp_path,
            exp["checkpoint_path"],
            bucket=name,
            raster_path=str(exp["raster_path"]),
            stated=Stated(tile_size=TILE, overlap=0.2, **stated),
            actor=None,
        )
        if name == "omitted":
            assert "conf" in result.get("error", ""), result
            continue
        assert "error" not in result, result
        execution = read_bucket(exp["root"], name).execution
        assert (execution.conf, execution.sources["conf"]) == (
            SAMPLE_CONF,
            "explicit",
        )


# ── the band geometry ─────────────────────────────────────────────────────


def test_band_ground_truth_keeps_what_is_centered_in_the_band_at_full_extent() -> (
    None
):
    """Band ground truth keeps each box centered in the band, at full extent.

    A box centered inside the band is kept at full extent, translated to local coordinates; a box
    centered outside is dropped, the two such boxes here each overlapping the band by 10 px.
    """
    import numpy as np

    from tcip_mcp.pipelines.block_calibration import select_gt_for_band

    band = (100, 100, 200, 200)
    boxes = np.array(
        [
            [
                120.0,
                120.0,
                140.0,
                140.0,
            ],  # center (130,130): inside, kept at full extent
            [
                190.0,
                120.0,
                260.0,
                140.0,
            ],  # center (225,130): outside, past x1=200
            [
                50.0,
                90.0,
                110.0,
                150.0,
            ],  # center (80,120): outside, before x0=100
        ]
    )
    gt = {
        "boxes": boxes,
        "labels": np.array([1, 2, 3]),
        "iscrowd": np.array([True, False, False]),
    }

    kept = select_gt_for_band(gt, band)

    assert kept["labels"].tolist() == [1]
    assert kept["iscrowd"].tolist() == [True]  # the flag travels with its row
    np.testing.assert_array_equal(
        kept["boxes"], np.array([[20.0, 20.0, 40.0, 40.0]])
    )

    empty = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "iscrowd": np.zeros((0,), dtype=bool),
    }
    assert all(
        len(v) == 0 for v in select_gt_for_band(empty, (0, 0, 10, 10)).values()
    )


def test_band_records_carry_each_ground_truth_rows_crowd_flag(
    tmp_path: Path,
) -> None:
    """Band records carry each ground-truth row's crowd flag.

    Of a plain box and a crowd region, the band's ground truth flags the second as a crowd
    region; the plain box and the detection keep the coordinates their sources state.
    """
    from types import SimpleNamespace

    import numpy as np

    from tcip_mcp.assessment import _band_records
    from tcip_mcp.pipelines.data.label_queries import (
        json_det_targets,
        registry_scope,
    )
    from tcip_mcp.pipelines.data.selection import Sample

    class _OneDetection:
        def predict_sliced(
            self,
            view: str | Path | BandGroupRef | WindowedRasterReader,
            **kwargs: Execution
            | int
            | str
            | dict
            | Callable[[int, int, dict], None]
            | None,
        ) -> dict:
            return {
                "width": 200,
                "height": 200,
                "boxes": [[10.1, 10.1, 40.3, 30.3]],
                "scores": [0.9],
                "labels": [1],
                "count": 1,
                "cap": 5,
            }

    label = image_label_key(image_dir(tmp_path, UNDATED_BUCKET) / "mosaic.tif")
    json_io.write_label_document(
        label,
        [
            Annotation(subject="bur", geometry=BBox(20.3, 20.7, 40.1, 60.9)),
            Annotation(
                subject="bur",
                geometry=BBox(100.0, 100.0, 180.0, 180.0),
                iscrowd=True,
            ),
        ],
        200,
        200,
    )
    target = json_det_targets(
        json_io.read_label_document(label).annotations,
        registry_scope(image_root(tmp_path), "bur"),
    )
    gt = {
        "boxes": np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4),
        "labels": np.asarray(target["labels"], dtype=np.int64),
        "iscrowd": np.asarray(target["iscrowd"], dtype=bool),
    }
    stub = SimpleNamespace(predictor=_OneDetection(), tile_batch_size=8)
    band = Sample(
        member="a",
        source="mosaic.tif",
        ground_truth=label,
        group="a",
        side="holdout",
        rect=(0, 0, 200, 200),
    )

    records = _band_records(
        SimpleNamespace(height=200, width=200, num_channels=3),
        "holdout",
        {"a": (0, 0, 200, 200)},
        stub,
        SimpleNamespace(tile_size=64, overlap=0.2),
        gt=gt,
        digest_of={band.location: "band-digest"},
        source="mosaic.tif",
    )

    assert [g["iscrowd"] for g in records[0]["gt"]] == [0, 1]
    assert records[0]["gt"][0]["bbox"] == [20.3, 20.7, 19.8, 40.2]
    assert [d["bbox"] for d in records[0]["dt"]] == [[10.1, 10.1, 30.2, 20.2]]
    assert records[0]["image_id"] == "band-digest"


def test_a_band_record_states_the_frame_its_forward_ran_on_and_the_band_it_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A band record states the frame its forward ran on and the band it scores.

    Through a real assessment, each band's record is the halo its tiled pass ran on: that
    frame's width, height and the cap the pass's density gives it, a count within it, the band
    inside it as the scored region, and only detections and ground truth centered in it.
    """
    import tcip_mcp.assessment as assessment
    from tcip_mcp.pipelines.derivations import detection_cap
    from tcip_mcp.pipelines.slicing import slice_lattice

    seen: list = []
    real = assessment._band_records

    def recording(
        reader: RasterSource,
        side: Side,
        bands: dict[str, tuple[int, int, int, int]],
        p: Pass,
        execution: Execution,
        **kwargs: dict | str,
    ) -> list[dict]:
        records = real(reader, side, bands, p, execution, **kwargs)
        seen.append(((reader.width, reader.height), bands, execution, records))
        return records

    monkeypatch.setattr(assessment, "_band_records", recording)
    assert "error" not in _assess(_attested(tmp_path), conf=0.0)
    assert seen

    for (width, height), bands, execution, records in seen:
        halo = (
            TILE - slice_lattice(TILE, 2 * TILE, TILE, execution.overlap)[1][0]
        )
        for (ix0, iy0, ix1, iy1), record in zip(
            (bands[name] for name in sorted(bands)), records, strict=True
        ):
            hx0, hy0 = max(0, ix0 - halo), max(0, iy0 - halo)
            frame = (
                min(width, ix1 + halo) - hx0,
                min(height, iy1 + halo) - hy0,
            )
            region = [ix0 - hx0, iy0 - hy0, ix1 - hx0, iy1 - hy0]
            assert (record["width"], record["height"]) == frame
            assert record["cap"] == detection_cap(
                execution.density, frame[0] * frame[1]
            )
            assert (
                record["count"] <= record["cap"] and record["region"] == region
            )
            for row in record["dt"] + record["gt"]:
                x, y, w, h = row["bbox"]
                assert (
                    region[0] <= x + w / 2 < region[2]
                    and region[1] <= y + h / 2 < region[3]
                )


def test_band_rects_are_reported_in_full_mosaic_coordinates() -> None:
    """Band rects are reported in full-mosaic coordinates.

    Over a region whose origin is (500, 60), the three bands lie inside the region and do not
    overlap along x.
    """
    from tcip_mcp.pipelines.block_calibration import band_rects

    region = (500, 60, 1300, 260)
    rx0, ry0, rx1, ry1 = region
    bands = band_rects(region, 3, TILE, 0.2, 40, "cal")

    assert len(bands) == 3
    for name, (bx0, by0, bx1, by1) in bands.items():
        assert rx0 <= bx0 < bx1 <= rx1, name
        assert ry0 <= by0 < by1 <= ry1, name
    ordered = sorted(bands.values())
    for (_x0, _y0, x1, _y1), (nx0, _ny0, _nx1, _ny1) in pairwise(ordered):
        assert x1 <= nx0


def test_density_outlier_bands_are_flagged_against_their_own_siblings() -> (
    None
):
    """Density-outlier bands are flagged against their own siblings, empty bands never.

    Among counts 10, 11, 1, 60 and 0, the bands at 1 and 60 are named and the band at 0 is not;
    counts 10, 11 and 9 flag none.
    """
    from tcip_mcp.pipelines.block_calibration import density_uniformity_flags

    skewed = {"cal_0": 10, "cal_1": 11, "cal_2": 1, "cal_3": 60, "cal_4": 0}
    assert density_uniformity_flags(skewed) == ["cal_2", "cal_3"]
    assert (
        density_uniformity_flags({"cal_0": 10, "cal_1": 11, "cal_2": 9}) == []
    )


def test_feasibility_counts_only_bands_that_carry_ground_truth() -> None:
    from tcip_mcp.assessment import AssessmentRefusedError
    from tcip_mcp.pipelines.block_calibration import check_feasibility

    with pytest.raises(AssessmentRefusedError, match="leaves only 1 band"):
        check_feasibility({"test_0": 0, "test_1": 7, "test_2": 0}, side="test")
    check_feasibility({"cal_0": 4, "cal_1": 0, "cal_2": 9}, side="cal")


# ── the class space and the breeder's own attestation ─────────────────────


def _build_attribute_scoped_experiment(
    tmp_path: Path,
    *,
    trained_values: tuple[str, ...],
    reordered_values: tuple[str, ...],
    labeled_value: str,
    experiment_id: str = "exp_block_attribute",
) -> dict:
    """Run an experiment whose subject declares a categorical attribute, then reorder it.

    The dataset's registry is reordered after the run resolved and recorded its own attributes,
    and both reserved regions are attested complete. Returns what the experiment left.
    """
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.subject_registry import Attribute
    from tests._producer_fixtures import registry_of, registry_over

    root = tmp_path / "ds_attribute"
    images_dir = image_dir(root, UNDATED_BUCKET)
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    _write_mosaic(images_dir / f"{stem}.tif")
    registry_over(
        root,
        registry_of("bud", Attribute("stage", "categorical", trained_values)),
    )

    boxes = [
        Annotation(
            subject="bud",
            geometry=BBox(x, 80, x + 15, 110),
            attributes={"stage": labeled_value},
        )
        for x in range(10, WIDTH - 20, BOX_STEP)
    ]
    label_image(
        images_dir / f"{stem}.tif", boxes, WIDTH, HEIGHT, keep_empty=True
    )

    data_cfg = {
        "images_dir": str(images_dir),
        "scope": {"subject": "bud"},
        "tiling": tiling_block(tile_size=TILE, overlap=0.2),
        "split": drawn_split(
            val_ratio=0.2,
            holdout_ratio=0.15,
            seed=1,
            calibration_ratio=0.15,
            grouping="tile_prefix",
        ),
    }
    completed = _completed_over(tmp_path, data_cfg, experiment_id)
    recorded_scope = run_resolution(
        experiment_id, project=tmp_path
    ).data.recorded_scope
    assert recorded_scope.subject and recorded_scope.attributes

    registry_over(
        root,
        registry_of(
            "bud", Attribute("stage", "categorical", reordered_values)
        ),
    )
    manifest = completed["spatial_manifest"]
    _attest_regions_complete(
        root,
        stem,
        [manifest.calibration_region, manifest.holdout_region],
    )
    return {
        "project": tmp_path,
        "root": root,
        "stem": stem,
        "experiment_id": experiment_id,
        **completed,
        "recorded_scope": recorded_scope,
    }


def test_ground_truth_decodes_through_the_checkpoints_own_recorded_attributes(
    tmp_path: Path,
) -> None:
    """An assessment still counts ground truth after the registry reorders the run's values.

    The run records its attribute values in one order and the live registry is reordered after
    it; the assessment's bands still count objects.
    """
    from tcip_mcp.subject_registry import read_registry

    exp = _build_attribute_scoped_experiment(
        tmp_path,
        trained_values=("closed", "open", "shed"),
        reordered_values=("open", "closed", "shed"),
        labeled_value="open",
    )
    live = read_registry(exp["root"]).subjects[0].attributes
    assert exp["recorded_scope"].attributes[0].values == (
        "closed",
        "open",
        "shed",
    )
    assert live[0].values != exp["recorded_scope"].attributes[0].values

    assert _band_total(_assess(exp)) > 0


def test_a_recorded_scope_needs_no_registry_on_disk(tmp_path: Path) -> None:
    exp = _build_attribute_scoped_experiment(
        tmp_path,
        trained_values=("closed", "open", "shed"),
        reordered_values=("closed", "open", "shed"),
        labeled_value="open",
        experiment_id="exp_block_recorded_no_registry",
    )
    (exp["root"] / "subjects.json").unlink()

    assert _band_total(_assess(exp)) > 0


def test_regions_marked_through_the_editors_save_admit_the_assessment(
    tmp_path: Path, client: TestClient
) -> None:
    """Regions marked complete through the editor's save admit the assessment.

    Both reserved regions are marked complete through posts to ``/api/annotate/labels``.
    """
    from tests._web_fixtures import open_new_project

    exp = _build_experiment(tmp_path)
    open_new_project(tmp_path)
    manifest = exp["spatial_manifest"]
    for region in (manifest.calibration_region, manifest.holdout_region):
        x0, y0, x1, y1 = region[0]
        current = client.get(
            "/api/annotate/labels",
            params={"image_path": str(exp["raster_path"])},
        ).json()
        resp = client.post(
            "/api/annotate/labels",
            json={
                "image_path": str(exp["raster_path"]),
                "annotations": current["annotations"],
                "base_mtime": current["base_mtime"],
                "user": "breeder",
                "complete": {"bud": True},
                "rect": [x0, y0, x1 - x0, y1 - y0],
            },
        )
        assert resp.status_code == 200, resp.text

    assert _band_total(_assess(exp)) > 0

"""The measurement chain's producers: ingest a synthetic capture, draw a reference selection,
train a tiny detector on it (one bright square per frame, found at conf 0.5), confirm the count
trait, assess the checkpoint and publish its predictions under the assessment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tcip_annotation.state import Annotation, BBox

from tcip_mcp.pipelines.execution import Stated
from tests._training_values import evaluation_block, schedule, sgd_optimizer

if TYPE_CHECKING:
    from tcip_web.state import OpenProject


def chain_pass() -> Stated:
    """The untiled pass the chain's detectors publish at without an assessment: the conf their
    one bright square is found at, and the sample cap."""
    from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS

    return Stated(tile=False, conf=0.5, max_dets=SAMPLE_MAX_DETS)


BESPOKE_MODELS = str(Path(__file__).with_name("bespoke_models.py"))
"""The file of ``tests.bespoke_models``, which a config naming one of its builders or loops
declares under ``model_source.source_files``; a run imports it as the top-level module
``bespoke_models`` from the suite's own directory, so no package of this repository joins a
run's plan."""
IMG = 64
SUBJECT = "bud"
DATE = "2-11-26"
STEMS = tuple(f"s{i:02d}" for i in range(40))
BESPOKE_DETECTION = "bespoke_models:build_bespoke_detection"
"""The torchvision bespoke detector's builder, its sizes left to each caller."""
BESPOKE_CLASSIFIER = "bespoke_models:build_bespoke_classifier"
BESPOKE_SEMANTIC_SEG = "bespoke_models:build_bespoke_semantic_seg"
BESPOKE_INSTANCE_SEG = "bespoke_models:build_bespoke_instance_seg"
BESPOKE_ORDINAL = "bespoke_models:build_bespoke_ordinal"
BESPOKE_REGRESSOR = "bespoke_models:build_bespoke_regressor"
GT_ANCHOR_DETECTOR = "bespoke_models:build_bespoke_detector"
"""The GroupNorm detector whose anchors come from the ground truth's own box shapes."""
BARE_SCORE_THRESH_DETECTOR = "bespoke_models:build_bare_score_thresh_detector"
BARE_NO_KNOB_DETECTOR = "bespoke_models:build_bare_no_knob_detector"
SAVE_BUILT_WEIGHTS = "bespoke_models:save_built_weights"
"""A ``training_source`` that takes no step and saves the weights its model was built with."""
TRAIN_BESPOKE = "bespoke_models:train_bespoke"
"""A ``training_source`` training through the ``ctx`` sinks."""
DETECTION_SOURCE = {"builder": BESPOKE_DETECTION, "source_files": [BESPOKE_MODELS],
                    "task": "detection"}
"""A ``model_source`` of :data:`BESPOKE_DETECTION` at its builder's own sizes."""
CLASSIFIER_SOURCE = {"builder": BESPOKE_CLASSIFIER, "source_files": [BESPOKE_MODELS],
                     "task": "classification"}
"""A ``model_source`` of :data:`BESPOKE_CLASSIFIER`."""
REGION_BUILDER = {"builder": "bespoke_models:build_bright_region_detector",
                  "builder_kwargs": {}, "source_files": [BESPOKE_MODELS], "task": "detection"}
BLOB_BUILDER = {"builder": "bespoke_models:build_bright_blob_detector",
                "builder_kwargs": {}, "source_files": [BESPOKE_MODELS], "task": "detection"}


def object_at(index: int) -> tuple[int, int, int]:
    """Where frame ``index``'s single object sits, and how big it is. Every frame differs in both,
    so no two frames are the same pixels."""
    x0 = 4 + (index % 5) * 8
    y0 = 4 + ((index // 5) % 5) * 8
    size = 14 + (index % 3) * 4
    return x0, y0, size


def synthetic_capture(root: Path, *, date: str = DATE) -> Path:
    """Ingest one capture date of dim frames through ``ingest_images``, each holding one bright
    square its label document names; the capture's image directory."""
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._producer_fixtures import BRIGHT, label_image, painted_frame

    raw = root.parent / f"raw-{date}"
    raw.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        x0, y0, size = object_at(index)
        shade = 28 + (index % 7)
        painted_frame(IMG, IMG, (shade, shade, shade),
                      [((x0, y0, x0 + size, y0 + size), BRIGHT)]).save(raw / f"{stem}.png")

    ingested = ingest_images(root, source=str(raw), date_from=date)
    assert "error" not in ingested, ingested
    assert ingested["copied"] == len(STEMS), ingested

    images_dir = root / "images" / date
    for index, stem in enumerate(STEMS):
        x0, y0, size = object_at(index)
        label_image(images_dir / f"{stem}.png",
                    [Annotation(subject=SUBJECT, geometry=BBox(x0, y0, x0 + size, y0 + size))],
                    IMG, IMG)
    return images_dir


def draw_reference_selection(project: Path, root: Path, out: Path):
    """Draw ``root``'s labels into train, val, calibration and holdout sides with
    ``draw_splits``; the selection as recorded."""
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(project, str(root), output_path=str(out), subject=SUBJECT, seed=2,
                         val_ratio=0.2, calibration_ratio=0.2, holdout_ratio=0.2)
    assert "error" not in result, result
    return read_selection(out, project=project)


def run_config(selection_dir: Path, model_source: dict = REGION_BUILDER,
               **overrides: Any) -> dict:
    """A :func:`training_config` of ``model_source`` with ``overrides`` that binds the drawn
    selection rather than drawing a partition of its own."""
    return training_config(model_source, {"split": {"selection_dir": str(selection_dir)}},
                           **overrides)


def training_config(model_source: dict, data: dict, **overrides: Any) -> dict:
    """A one-epoch CPU run of ``model_source`` over the ``data`` section, every other key the
    toy detectors train at (the optimizer, schedule and evaluation blocks the sample ones of
    ``tests._training_values``), with a copy of each of ``overrides`` in place of its key."""
    import copy

    return {
        "model_source": copy.deepcopy(model_source),
        "data": data,
        "batch_size": 2,
        "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "device": "cpu",
        "checkpoint_every_n_epochs": 1,
        "early_stopping": {"enabled": False},
        "evaluation": evaluation_block(),
        "optimizer": sgd_optimizer(),
        "scheduler": schedule("cosine"),
        "gradient_accumulation_steps": 1,
        **copy.deepcopy(overrides),
    }


def built_model(config: dict):
    """The model a run over ``config`` builds: its validated ``model_source``, imported from the
    layout its admission stages with this repository as its project
    (``model_build.staged_sources``), at the dims its data block records
    (``model_build.recorded_model_dims``)."""
    from tcip_mcp.pipelines.model_build import (
        build_from_model_source, recorded_model_dims, staged_sources,
    )
    from tcip_mcp.pipelines.schemas import train_config

    from tests import REPO_ROOT

    spec = train_config(config)
    return build_from_model_source(spec.model_source, staged_sources(spec, REPO_ROOT).layout,
                                   recorded_model_dims(spec))


def train_on(selection_dir: Path, project_root: Path, experiment_id: str) -> str:
    """Train the tiny detector over the selection's train side through the child's own entry,
    whose completion registers the checkpoint; the checkpoint's path."""
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import worker_run

    observation = observe(worker_run(project_root, run_config(selection_dir),
                                     experiment_id=experiment_id))
    checkpoint = observation.checkpoint
    assert checkpoint is not None, observation.final
    return checkpoint["path"]


def confirm_count_trait(project_root: Path, **fields: Any):
    """Propose the count trait stating its per-image count of :data:`SUBJECT`, with ``fields``
    over the fixture floors, and confirm it as the breeder would."""
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx

    revision = fx.propose(project_root, fx.with_operationalization(
        fx.with_fields(fx.COUNT_SPEC, **fields), traits.PER_IMAGE_COUNT,
        measured_subject=SUBJECT))
    assert not revision.confirmed, "a proposed revision must land unconfirmed"
    return fx.confirm(project_root, revision, user="chain-breeder")


def assess(project: Path, checkpoint_path: str, selection_dir: Path, *,
           device: str | None = None, **stated: Any) -> dict:
    """``assess_checkpoint`` of the count trait's per-image count over ``selection_dir`` under the
    ``stated`` execution values, the sample cap unless they name one."""
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tests import _trait_fixtures as fx
    from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS

    return assess_checkpoint(project, checkpoint_path=checkpoint_path, trait=fx.COUNT_TRAIT,
                             delivery_kind="per_image_count", selection_dir=str(selection_dir),
                             stated=Stated(**{"max_dets": SAMPLE_MAX_DETS, **stated}),
                             device=device)


@dataclass
class Chain:
    """What one run of the flow leaves behind, for a test to deliver from or disturb: ``bucket``
    is the name its predictions are published under ``root``."""

    root: Path
    images_dir: Path
    selection_dir: Path
    checkpoint_path: str
    assessment: dict
    bucket: str
    published: dict

    def read(self):
        """The chain's published bucket, as its record states it."""
        from tcip_mcp.buckets import read_bucket

        return read_bucket(self.root, self.bucket)


def unassessed_bucket(project: Path, *, experiment_id: str, bucket_name: str = "plain"):
    """A bucket published under no assessment: ingest, draw, train and publish under ``project``;
    the bucket."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    root = project / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    checkpoint_path = train_on(selection_dir, project, experiment_id)
    bucket = f"{bucket_name}/{DATE}"
    published = run_inference(project, checkpoint_path=checkpoint_path,
                              images_dir=str(images_dir), bucket=bucket, stated=chain_pass())
    assert "error" not in published, published
    return read_bucket(root, bucket)


def acknowledged(project: Path, deliver: Callable[[str | None], Any], *,
                 by: str = "user:breeder", reason: str = "fixture: shipped before any assessment"
                 ) -> Any:
    """Run ``deliver``, a delivery taking a recorded acknowledgment's id, as a breeder ships an
    unvalidated result: refused first, the refused result acknowledged
    (:func:`~tcip_mcp.delivery.record_acknowledgment`), then delivered under that acknowledgment;
    what the second call returns. A refusal no acknowledgment can answer propagates."""
    import pytest

    from tcip_mcp.delivery import DeliveryRefusedError, record_acknowledgment

    with pytest.raises(DeliveryRefusedError) as refused:
        deliver(None)
    if refused.value.result_sha256 is None:
        raise refused.value
    act = record_acknowledgment(project, acknowledged_by=by, reason=reason,
                                result_sha256=refused.value.result_sha256)
    return deliver(act.acknowledgment_id)


def deliver_acknowledged(project: Path, results: list[dict], out: Path, delivered_phenotype: str,
                         *, delivery_kind: str, buckets: list | None = None, **kwargs: Any):
    """``deliver_per_plant_aggregate`` of ``results``' plants over ``buckets`` (by default a bucket
    published under no assessment, :func:`unassessed_bucket`), shipped unvalidated under the
    breeder's acknowledgment (:func:`acknowledged`)."""
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate

    delivered = buckets if buckets is not None else [
        unassessed_bucket(project, experiment_id="exp-acknowledged")]
    return acknowledged(project, lambda ack: deliver_per_plant_aggregate(
        project, results, str(out), delivered_phenotype=delivered_phenotype,
        delivery_kind=delivery_kind, buckets=delivered,
        plants=[r["plant_id"] for r in results], door="test_delivery", acknowledgment_id=ack,
        actor=None, **kwargs))


def predicted(image: Path, values: list[str], attributes: tuple = ()) -> dict:
    """One predictor result for the image at ``image``: a detection of the subject per entry of
    ``values``, each carrying that value's id under the first of ``attributes`` (the scope's
    attribute records) and the first value of every other; no ``attributes`` row for none."""
    result = {"image": str(image), "width": IMG, "height": IMG,
              "boxes": [[4.0 * k, 0.0, 4.0 * k + 3.0, 3.0] for k in range(len(values))],
              "scores": [0.9] * len(values), "labels": [1] * len(values)}
    if attributes:
        result["attributes"] = [[attributes[0].values.index(v)] + [0] * (len(attributes) - 1)
                                for v in values]
    return result


def published(project: Path, name: str, results: list[dict], *, scope: dict,
              registry: Any = None, raster_path: Path | None = None) -> Any:
    """``results`` published as the bucket named ``name`` (:func:`~tcip_mcp.buckets.publish`)
    under their images' dataset root, under the untiled pass a registered checkpoint stating
    ``scope`` runs (its frames' dataset declaring ``registry``), over the raster ``raster_path``
    when one is named; the bucket."""
    from tcip_mcp.buckets import pass_documents, publish, source_root
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.split_construction import raster_identity
    from tcip_mcp.pipelines.execution import prepare
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF, SAMPLE_MAX_DETS, project_checkpoint,
    )

    from tcip_store import decode_value, encode_record

    checkpoint = load_registered_checkpoint(
        project_checkpoint(project, data={"num_channels": 3, "scope": scope}, registry=registry),
        project=project)
    identity = (decode_value(encode_record(raster_identity(resolve_image_path(raster_path))))
                if raster_path is not None else None)
    p = prepare(checkpoint,
                Stated(tile=False, conf=SAMPLE_CONF, max_dets=SAMPLE_MAX_DETS)).runnable()
    root = source_root([raster_path] if raster_path is not None else [r["image"] for r in results])
    return publish(project, root, name, pass_documents(p, results), producer=checkpoint.producer,
                   scope=p.scope, execution=p.execution,
                   raster_path=str(raster_path) if raster_path is not None else None,
                   raster_identity=identity, assessment_id=None, actor=None)


VALUES = ("open", "closed")
"""The attributed chain's attribute values, in the registry's declared order: a value's index is
the band its blobs are bright in, and the class the blob detector calls them."""
ATTRIBUTE = "opening"
SERIES_DATES = ("2026-02-11", "2026-02-25", "2026-03-10", "2026-03-24")
PLANTS = {"PLANT_A": (43.19700, -90.05800), "PLANT_B": (43.19700, -90.05780)}
"""Where each plant of the attributed chain's registry stands (WGS84 lat, lon), about 16 m
apart."""
REFERENCE_SITE = (43.20300, -90.05000)
"""Where the labeled reference frames of each date were taken, about 900 m from every plant, so
the mapping attributes none of them."""


def blob_boxes(values: list[str], index: int) -> list[tuple[str, BBox]]:
    """One box per entry of ``values`` on a grid of 16 px cells, beside its value; ``index``
    varies the sizes so no two frames share pixels."""
    out = []
    for k, value in enumerate(values):
        x0, y0 = 2 + (k % 4) * 16, 2 + (k // 4) * 16
        size = 8 + (index + k) % 4
        out.append((value, BBox(x0, y0, x0 + size, y0 + size)))
    return out


def blob_frame(values: list[str], index: int):
    """A dim frame holding each blob :func:`blob_boxes` places, bright in its value's band."""
    from tests._producer_fixtures import painted_frame

    shade = 20 + (index % 11)
    return painted_frame(IMG, IMG, (shade, shade, shade), [
        ((box.x1, box.y1, box.x2, box.y2),
         tuple(230 if band == VALUES.index(value) else 20 for band in range(3)))
        for value, box in blob_boxes(values, index)])


def write_attributed_registry(*roots: Path, attributes: tuple | None = None) -> None:
    """Declare :data:`SUBJECT` with ``attributes`` (by default :data:`ATTRIBUTE`, categorical over
    :data:`VALUES`) in each root's subject registry."""
    from tcip_mcp import subject_registry as cr
    from tests._producer_fixtures import registry_over

    registry = cr.SubjectRegistry(subjects=(cr.Subject(name=SUBJECT, attributes=attributes or (
        cr.Attribute(name=ATTRIBUTE, type="categorical", values=VALUES),)),))
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        registry_over(root, registry)


def confirm_crossing_trait(project_root: Path, **fields: Any):
    """Confirm ``bud_opening`` stating its crossing dates over :data:`SUBJECT`, every criterion
    field authored at the fixture floors unless ``fields`` states it."""
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx

    floors = {"count_bias_tolerance_frac": 0.1, "count_error_tolerance": 1.0,
              "classifier_agreement_floor": 0.6, **fields}
    spec = fx.with_fields(fx.BUD_OPENING, **floors)
    return fx.propose_and_confirm(project_root, fx.with_operationalization(
        spec, traits.STATE_CROSSING_DATES, measured_subject=SUBJECT,
        statement="the date each plant reached the open state the breeder scores",
        mechanism="the assessed classifier over one plant's objects",
        delivered_phenotypes=spec.delivers))


@dataclass
class Series:
    """What the attributed chain leaves behind: its dataset, the checkpoint, the assessment (when
    one was run), the plant mapping, the name of one published bucket per date and the project as
    the web backend opened it."""

    root: Path
    checkpoint_path: str
    assessment: dict | None
    mapping_name: str
    buckets: dict[str, str]
    project: OpenProject
    trait: str = "bud_opening"

    def body(self, **extra: Any) -> dict:
        """The request body a phenology door takes over this series."""
        return {"mapping_name": self.mapping_name, "trait": self.trait,
                "dataset_root": str(self.root), "buckets": list(self.buckets.values()),
                "plants": list(PLANTS), **extra}


def deliver_milestones(project: Path, body: dict, out_csv: Path) -> dict:
    """``deliver_phenology_milestones`` over a phenology door's request ``body`` (a
    :meth:`Series.body`), writing ``out_csv``."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones(
        project, trait=body["trait"], mapping_name=body["mapping_name"], plants=body["plants"],
        dataset_root=body["dataset_root"], buckets=body["buckets"],
        output_csv_path=str(out_csv))


def attributed_series(
    project: Path, *, fractions: tuple[float, ...] = (0.0, 0.25, 0.75, 1.0), detections: int = 4,
    images_per_plant: int = 1, assessed: bool = True, experiment_id: str = "exp-attributed",
    attributes: tuple | None = None, model_source: dict | None = None, stated: Any = None,
) -> Series:
    """Ingest a geotagged series of :data:`PLANTS` over ``len(fractions)`` of
    :data:`SERIES_DATES`, each date beside labeled reference frames taken at
    :data:`REFERENCE_SITE`; draw the labeled frames into a selection, train ``model_source``
    (by default the blob detector) on it, confirm the crossing trait, assess it when ``assessed``,
    publish one bucket per date under the ``stated`` execution values (under the assessment when
    there is one, :func:`chain_pass` when neither is given), build the plant mapping, and open
    ``project`` in the web backend.

    The registry declares ``attributes`` on :data:`SUBJECT` (:func:`write_attributed_registry`),
    one of them :data:`ATTRIBUTE` over :data:`VALUES`; every labeled object carries a value of
    each, :data:`ATTRIBUTE`'s the band its blob is bright in. On each date every plant's frames
    hold ``detections`` blobs of which ``fractions[i]`` are open."""
    from datetime import datetime, timedelta

    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image
    from tests._mapping_fixtures import register_plant_registry_for, write_plant_csv
    from tests._producer_fixtures import label_image
    from tests._verified_checkpoint_fixtures import worker_run
    from tests._web_fixtures import open_new_project

    from tcip_mcp import subject_registry as cr

    root = project / "ds"
    attributes = attributes or (cr.Attribute(name=ATTRIBUTE, type="categorical", values=VALUES),)
    write_attributed_registry(project, root, attributes=attributes)
    raw = project.parent / f"{project.name}-raw"
    dates = SERIES_DATES[: len(fractions)]
    per_date = 40 // len(dates)
    for d, (date, fraction) in enumerate(zip(dates, fractions)):
        capture = raw / date
        positive = int(round(fraction * detections))
        values = ["open"] * positive + ["closed"] * (detections - positive)
        when = datetime.fromisoformat(date).replace(hour=9)
        for p, (plant, (lat, lon)) in enumerate(PLANTS.items()):
            for i in range(images_per_plant):
                index = 1000 + 100 * d + 10 * p + i
                write_geo_image(capture / f"{plant}_{date}_{i}.jpg", lat, lon,
                                when + timedelta(minutes=index), blob_frame(values, index))
        labeled = {}
        for i in range(per_date):
            index = per_date * d + i
            labeled[f"ref_{date}_{i:02d}"] = (index, [VALUES[index % 2]] + (
                [VALUES[(index + 1) % 2]] if index % 3 else []))
        for stem, (index, frame_values) in labeled.items():
            write_geo_image(capture / f"{stem}.jpg", *REFERENCE_SITE,
                            when + timedelta(hours=2, minutes=index),
                            blob_frame(frame_values, index))
        ingested = ingest_images(root, source=str(capture), date_from=date)
        assert "error" not in ingested, ingested
        for stem, (index, frame_values) in labeled.items():
            label_image(root / "images" / date / f"{stem}.jpg", [
                Annotation(subject=SUBJECT, geometry=box, attributes={
                    a.name: value if a.name == ATTRIBUTE else a.values[(index + k) % len(a.values)]
                    for a in attributes})
                for k, (value, box) in enumerate(blob_boxes(frame_values, index))], IMG, IMG)
    registered = register_dataset(project, str(root), crop=sorted(registered_crops())[0])
    assert "error" not in registered, registered

    selection_dir = project / "selection"
    drawn = draw_splits(project, str(root), output_path=str(selection_dir), subject=SUBJECT,
                        seed=2, val_ratio=0.2, calibration_ratio=0.2,
                        holdout_ratio=0.2)
    assert "error" not in drawn, drawn
    config = run_config(selection_dir, model_source or BLOB_BUILDER, seed=2)
    from tcip_mcp.experiments import observe

    checkpoint = observe(worker_run(project, config, experiment_id=experiment_id)).checkpoint
    assert checkpoint is not None
    checkpoint_path = checkpoint["path"]
    confirm_crossing_trait(project)

    assessment = None
    if assessed:
        from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS

        assessment = assess_checkpoint(project, checkpoint_path=checkpoint_path,
                                       trait="bud_opening", delivery_kind="state_crossing_dates",
                                       selection_dir=str(selection_dir),
                                       stated=Stated(max_dets=SAMPLE_MAX_DETS))
        assert "error" not in assessment, assessment
        assert assessment["passed"] is True, assessment["failures"]

    if stated is None and assessment is None:
        stated = chain_pass()
    predictions: dict[str, str] = {}
    for date in dates:
        published = run_inference(
            project, checkpoint_path=checkpoint_path, images_dir=str(root / "images" / date),
            bucket=f"series/{date}", stated=stated,
            assessment_id=assessment["assessment_id"] if assessment else None)
        assert "error" not in published, published
        predictions[date] = f"series/{date}"

    plants_csv = write_plant_csv(raw / "plants.csv", [
        {"plot": plant, "accession": f"Acc{plant[-1]}", "lat": lat, "lon": lon}
        for plant, (lat, lon) in PLANTS.items()])
    registry = register_plant_registry_for(project, [plants_csv])
    mapped = build_plant_mapping(project, name="valley", images_root=str(root / "images"),
                                 plant_registry=registry, dates=list(dates))
    assert "error" not in mapped, mapped
    return Series(root, checkpoint_path, assessment, "valley", predictions,
                  open_new_project(project))


def run_the_chain(project: Path, *, experiment_id: str, bucket_name: str = "chain") -> Chain:
    """Ingest, train, confirm, assess and publish under ``project``; the assessment passes."""
    from tcip_mcp.tools.inference_tools import run_inference

    root = project / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    checkpoint_path = train_on(selection_dir, project, experiment_id)
    confirm_count_trait(project)

    assessment = assess(project, checkpoint_path, selection_dir)
    assert "error" not in assessment, assessment
    assert assessment["passed"] is True, assessment["failures"]

    bucket = f"{bucket_name}/{DATE}"
    published = run_inference(
        project, checkpoint_path=checkpoint_path, images_dir=str(images_dir),
        bucket=bucket, assessment_id=assessment["assessment_id"])
    assert "error" not in published, published
    return Chain(root, images_dir, selection_dir, checkpoint_path, assessment, bucket, published)

"""The measurement chain's producers, for any test that needs a real assessment or a real bucket:
ingest a synthetic capture, draw a reference selection, train a tiny detector on it, confirm the
count trait, assess the checkpoint and publish its predictions under the assessment.

Every record here is made by the platform's own function, never written by hand. The model is
tiny and its predictions are stable (one bright square per frame, found at conf 0.5), so the
chain runs in seconds and a test built on it measures the platform, not a fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox

IMG = 64
SUBJECT = "bud"
DATE = "2-11-26"
STEMS = tuple(f"s{i:02d}" for i in range(40))
BUILDER = "tests.bespoke_models:build_bright_region_detector"


def object_at(index: int) -> tuple[int, int, int]:
    """Where frame ``index``'s single object sits, and how big it is. Every frame differs in both,
    so no two frames are the same pixels."""
    x0 = 4 + (index % 5) * 8
    y0 = 4 + ((index // 5) % 5) * 8
    size = 14 + (index % 3) * 4
    return x0, y0, size


def synthetic_capture(root: Path, *, date: str = DATE) -> tuple[Path, Path]:
    """Ingest one capture date of dim frames through ``ingest_images``, each holding one bright
    square its label names; ``(images_dir, labels_dir)``."""
    from PIL import Image, ImageDraw

    from tcip_mcp.tools.ingest_tools import ingest_images

    raw = root.parent / f"raw-{date}"
    raw.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        x0, y0, size = object_at(index)
        shade = 28 + (index % 7)
        frame = Image.new("RGB", (IMG, IMG), color=(shade, shade, shade))
        ImageDraw.Draw(frame).rectangle([x0, y0, x0 + size - 1, y0 + size - 1],
                                        fill=(230, 230, 230))
        frame.save(raw / f"{stem}.png")

    ingested = ingest_images(root, source=str(raw), date_from=date)
    assert "error" not in ingested, ingested
    assert ingested["copied"] == len(STEMS), ingested

    images_dir, labels_dir = root / "images" / date, root / "annotations" / date
    labels_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        x0, y0, size = object_at(index)
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject=SUBJECT, geometry=BBox(x0, y0, x0 + size, y0 + size))], IMG, IMG)
    return images_dir, labels_dir


def draw_reference_selection(project: Path, root: Path, out: Path):
    """Draw ``root``'s labels into train, val, calibration and holdout sides with
    ``draw_splits``; the selection as recorded."""
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(project, str(root), output_path=str(out), subject=SUBJECT, seed=2,
                         train_ratio=0.4, val_ratio=0.2, calibration_ratio=0.2, holdout_ratio=0.2)
    assert "error" not in result, result
    return read_selection(out, project=project)


def run_config(selection_dir: Path) -> dict:
    """A run that binds the drawn selection rather than drawing a partition of its own."""
    return {
        "model_source": {"builder": BUILDER, "builder_kwargs": {}, "task": "detection"},
        "data": {"split": {"selection_dir": str(selection_dir)}},
        "batch_size": 2,
        "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False,
        "device": "cpu",
        "checkpoint_every_n_epochs": 1,
        "early_stopping": {"enabled": False},
        "optimizer": {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0},
        "scheduler": {"type": "cosine"},
        "gradient_accumulation_steps": 1,
    }


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
    ``stated`` execution values."""
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tests import _trait_fixtures as fx

    return assess_checkpoint(project, checkpoint_path=checkpoint_path, trait=fx.COUNT_TRAIT,
                             delivery_kind="per_image_count", selection_dir=str(selection_dir),
                             stated=Stated(**stated), device=device)


@dataclass
class Chain:
    """What one run of the flow leaves behind, for a test to deliver from or disturb."""

    root: Path
    images_dir: Path
    labels_dir: Path
    selection_dir: Path
    checkpoint_path: str
    assessment: dict
    bucket: Path
    published: dict


def unassessed_bucket(project: Path, *, experiment_id: str, bucket_name: str = "plain") -> Path:
    """A bucket published under no assessment: ingest, draw, train and publish under ``project``;
    the bucket's directory."""
    from tcip_mcp.tools.inference_tools import run_inference

    root = project / "ds"
    images_dir, _labels_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    checkpoint_path = train_on(selection_dir, project, experiment_id)
    bucket = root / "predictions" / bucket_name / DATE
    published = run_inference(project, checkpoint_path=checkpoint_path,
                              images_dir=str(images_dir), output_dir=str(bucket))
    assert "error" not in published, published
    return bucket


def acknowledged(project: Path, deliver: Callable[[str | None], Any], *,
                 by: str = "user:breeder", reason: str = "fixture: shipped before any assessment"
                 ) -> Any:
    """Run ``deliver``, a delivery taking a recorded acknowledgment's id, as a breeder ships an
    unvalidated result: refused first, the refused result acknowledged
    (:func:`~tcip_mcp.delivery.record_acknowledgment`), then delivered under that acknowledgment;
    what the second call returns. A refusal no acknowledgment can answer propagates."""
    import pytest

    from tcip_mcp.delivery import DeliveryRefused, record_acknowledgment

    with pytest.raises(DeliveryRefused) as refused:
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
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate

    delivered = buckets if buckets is not None else [
        read_bucket(unassessed_bucket(project, experiment_id="exp-acknowledged"))]
    return acknowledged(project, lambda ack: deliver_per_plant_aggregate(
        project, results, str(out), delivered_phenotype=delivered_phenotype,
        delivery_kind=delivery_kind, buckets=delivered,
        plants=[r["plant_id"] for r in results], door="test_delivery", acknowledgment_id=ack,
        actor=None, **kwargs))


def predicted(stem: str, values: list[str], attributes: tuple = ()) -> dict:
    """One predictor result for the image ``<stem>.png``: a detection of the subject per entry of
    ``values``, each carrying that value's id under the first of ``attributes`` (the scope's
    attribute records) and the first value of every other; no ``attributes`` row for none."""
    result = {"image": f"{stem}.png", "width": IMG, "height": IMG,
              "boxes": [[4.0 * k, 0.0, 4.0 * k + 3.0, 3.0] for k in range(len(values))],
              "scores": [0.9] * len(values), "labels": [1] * len(values)}
    if attributes:
        result["attributes"] = [[attributes[0].values.index(v)] + [0] * (len(attributes) - 1)
                                for v in values]
    return result


def published(project: Path, out: Path, results: list[dict], *, scope: dict,
              registry: Any = None, raster_path: Path | None = None) -> Any:
    """``results`` published as the bucket ``out`` (:func:`~tcip_mcp.buckets.publish`), under the
    untiled pass a registered checkpoint stating ``scope`` runs (its frames' dataset declaring
    ``registry``), over the raster ``raster_path`` when one is named; the bucket."""
    from tcip_mcp.buckets import pass_documents, publish
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.split_construction import raster_identity
    from tcip_mcp.pipelines.execution import Stated, prepare_pass
    from tests._verified_checkpoint_fixtures import project_checkpoint

    from tcip_store import decode_value, encode_record

    checkpoint = load_registered_checkpoint(
        project_checkpoint(project, data={"num_channels": 3, "scope": scope}, registry=registry),
        project=project)
    identity = (decode_value(encode_record(raster_identity(raster_path)))
                if raster_path is not None else None)
    p = prepare_pass(checkpoint, Stated(tile=False))
    return publish(project, out, pass_documents(p, results), producer=checkpoint.producer,
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
BLOB_BUILDER = {"builder": "tests.bespoke_models:build_bright_blob_detector",
                "builder_kwargs": {}, "task": "detection"}


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
    from PIL import Image, ImageDraw

    shade = 20 + (index % 11)
    frame = Image.new("RGB", (IMG, IMG), color=(shade, shade, shade))
    draw = ImageDraw.Draw(frame)
    for value, box in blob_boxes(values, index):
        color = tuple(230 if band == VALUES.index(value) else 20 for band in range(3))
        draw.rectangle([box.x1, box.y1, box.x2 - 1, box.y2 - 1], fill=color)
    return frame


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
    one was run), the plant mapping and one published bucket per date."""

    root: Path
    checkpoint_path: str
    assessment: dict | None
    mapping_name: str
    buckets: dict[str, str]
    trait: str = "bud_opening"

    def body(self, **extra: Any) -> dict:
        """The request body a phenology door takes over this series."""
        return {"mapping_name": self.mapping_name, "trait": self.trait,
                "buckets": list(self.buckets.values()), "plants": list(PLANTS), **extra}


def deliver_milestones(project: Path, body: dict, out_csv: Path) -> dict:
    """``deliver_phenology_milestones`` over a phenology door's request ``body`` (a
    :meth:`Series.body`), writing ``out_csv``."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones(
        project, trait=body["trait"], mapping_name=body["mapping_name"], plants=body["plants"],
        buckets=body["buckets"], output_csv_path=str(out_csv))


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
    there is one), build the plant mapping, and open ``project`` in the web backend.

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
    from tests._mapping_fixtures import register_plant_registry_for
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
                            when + timedelta(hours=2, minutes=index), blob_frame(frame_values, index))
        ingested = ingest_images(root, source=str(capture), date_from=date)
        assert "error" not in ingested, ingested
        labels_dir = root / "annotations" / date
        labels_dir.mkdir(parents=True, exist_ok=True)
        for stem, (index, frame_values) in labeled.items():
            json_io.write_annotations(str(labels_dir / f"{stem}.json"), [
                Annotation(subject=SUBJECT, geometry=box, attributes={
                    a.name: value if a.name == ATTRIBUTE else a.values[(index + k) % len(a.values)]
                    for a in attributes})
                for k, (value, box) in enumerate(blob_boxes(frame_values, index))], IMG, IMG)
    registered = register_dataset(project, str(root), crop=sorted(registered_crops())[0])
    assert "error" not in registered, registered

    selection_dir = project / "selection"
    drawn = draw_splits(project, str(root), output_path=str(selection_dir), subject=SUBJECT,
                        seed=2, train_ratio=0.4, val_ratio=0.2, calibration_ratio=0.2,
                        holdout_ratio=0.2)
    assert "error" not in drawn, drawn
    config = {**run_config(selection_dir), "model_source": dict(model_source or BLOB_BUILDER),
              "seed": 2}
    from tcip_mcp.experiments import observe

    checkpoint = observe(worker_run(project, config, experiment_id=experiment_id)).checkpoint
    assert checkpoint is not None
    checkpoint_path = checkpoint["path"]
    confirm_crossing_trait(project)

    assessment = None
    if assessed:
        assessment = assess_checkpoint(project, checkpoint_path=checkpoint_path,
                                       trait="bud_opening", delivery_kind="state_crossing_dates",
                                       selection_dir=str(selection_dir))
        assert "error" not in assessment, assessment
        assert assessment["passed"] is True, assessment["failures"]

    predictions: dict[str, str] = {}
    for date in dates:
        out = root / "predictions" / "series" / date
        published = run_inference(
            project, checkpoint_path=checkpoint_path, images_dir=str(root / "images" / date),
            output_dir=str(out), stated=stated,
            assessment_id=assessment["assessment_id"] if assessment else None)
        assert "error" not in published, published
        predictions[date] = str(out)

    plants_csv = raw / "plants.csv"
    plants_csv.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n" + "".join(
            f"{plant},Acc{plant[-1]},{lon},{lat}\n" for plant, (lat, lon) in PLANTS.items()),
        encoding="utf-8")
    registry = register_plant_registry_for(project, [plants_csv])
    mapped = build_plant_mapping(project, name="valley", images_root=str(root / "images"),
                                 plant_registry=registry, dates=list(dates))
    assert "error" not in mapped, mapped
    open_new_project(project)
    return Series(root, checkpoint_path, assessment, "valley", predictions)


def run_the_chain(project: Path, *, experiment_id: str, bucket_name: str = "chain") -> Chain:
    """Ingest, train, confirm, assess and publish under ``project``; the assessment passes."""
    from tcip_mcp.tools.inference_tools import run_inference

    root = project / "ds"
    images_dir, labels_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    checkpoint_path = train_on(selection_dir, project, experiment_id)
    confirm_count_trait(project)

    assessment = assess(project, checkpoint_path, selection_dir)
    assert "error" not in assessment, assessment
    assert assessment["passed"] is True, assessment["failures"]

    bucket = root / "predictions" / bucket_name / DATE
    published = run_inference(
        project, checkpoint_path=checkpoint_path, images_dir=str(images_dir),
        output_dir=str(bucket), assessment_id=assessment["assessment_id"])
    assert "error" not in published, published
    return Chain(root, images_dir, labels_dir, selection_dir, checkpoint_path, assessment,
                 bucket, published)

"""The measurement chain's producers.

They ingest a synthetic capture, draw a reference selection, train a tiny detector on it (one
bright square per frame, found at conf 0.5), confirm the count trait, assess the checkpoint and
publish its predictions under the assessment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_annotation.state import Annotation, BBox
from tcip_mcp.assessment import read_assessment
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.execution import Stated
from tests._training_values import (
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    MASK_CUT,
    RATIONALE,
    collection_pass,
    drawn_split,
    evaluation_block,
    schedule,
    sgd_optimizer,
    stage,
    stop_rule,
)

if TYPE_CHECKING:
    from PIL import Image
    from torch import nn

    from tcip_mcp.buckets import Bucket
    from tcip_mcp.pipelines.data.selection import Selection
    from tcip_mcp.pipelines.data.split_construction import Partition
    from tcip_mcp.pipelines.schemas import DetectorGeometry
    from tcip_mcp.subject_registry import SubjectRegistry
    from tcip_mcp.traits import TraitRevision
    from tcip_web.state import OpenProject


def chain_pass() -> Stated:
    """Return an untiled pass at conf 0.5 stating :data:`RATIONALE`."""
    return Stated(tile=False, conf=0.5, rationale=RATIONALE)


BESPOKE_MODELS = str(Path(__file__).with_name("bespoke_models.py"))
"""The path of ``tests/bespoke_models.py``, a ``model_source.source_files`` entry that imports as
the top-level module ``bespoke_models``."""
IMG = 64
SUBJECT = "bud"
DATE = "2026-02-11"
STEMS = tuple(f"s{i:02d}" for i in range(40))
BESPOKE_DETECTION = "bespoke_models:build_bespoke_detection"
"""The torchvision bespoke detector's builder, its further keywords passed to the detector."""
BESPOKE_CLASSIFIER = "bespoke_models:build_bespoke_classifier"
BESPOKE_SEMANTIC_SEG = "bespoke_models:build_bespoke_semantic_seg"
BESPOKE_INSTANCE_SEG = "bespoke_models:build_bespoke_instance_seg"
BESPOKE_ORDINAL = "bespoke_models:build_bespoke_ordinal"
BESPOKE_REGRESSOR = "bespoke_models:build_bespoke_regressor"
GT_ANCHOR_DETECTOR = "bespoke_models:build_bespoke_detector"
"""The GroupNorm detector's builder, taking its anchor geometry as keywords."""
BARE_SCORE_THRESH_DETECTOR = "bespoke_models:build_bare_score_thresh_detector"
BARE_NO_KNOB_DETECTOR = "bespoke_models:build_bare_no_knob_detector"
SAVE_BUILT_WEIGHTS = "bespoke_models:save_built_weights"
"""A ``training_source`` that takes no step and saves the weights its model was built with."""
TRAIN_BESPOKE = "bespoke_models:train_bespoke"
"""A ``training_source`` training through the ``ctx`` sinks."""
FRAME_STATISTICS = "the band statistics of the config's own frames"
"""A model source's ``normalization`` standing for the band statistics of the train side its
config's data section resolves to."""
DETECTION_SOURCE = {
    "builder": BESPOKE_DETECTION,
    "source_files": [BESPOKE_MODELS],
    "task": "detection",
    "normalization": FRAME_STATISTICS,
}
"""A ``model_source`` of :data:`BESPOKE_DETECTION` loading no weights."""
CLASSIFIER_SOURCE = {
    "builder": BESPOKE_CLASSIFIER,
    "source_files": [BESPOKE_MODELS],
    "task": "classification",
    "normalization": FRAME_STATISTICS,
}
"""A ``model_source`` of :data:`BESPOKE_CLASSIFIER`."""
REGION_BUILDER = {
    "builder": "bespoke_models:build_bright_region_detector",
    "builder_kwargs": {},
    "source_files": [BESPOKE_MODELS],
    "task": "detection",
    "normalization": FRAME_STATISTICS,
}
BLOB_BUILDER = {
    "builder": "bespoke_models:build_bright_blob_detector",
    "builder_kwargs": {},
    "source_files": [BESPOKE_MODELS],
    "task": "detection",
    "normalization": FRAME_STATISTICS,
}


def detector_geometry(
    frame: tuple[int, int] = (64, 64),
) -> DetectorGeometry:
    """Return the ``schemas.DetectorGeometry`` derived for ``frame``.

    ``derivations.derive_detector_geometry`` gives it over two boxes, one a square a quarter of
    the frame's short edge across and one an eighth of that edge wide by a quarter tall.
    """
    import numpy as np

    from tcip_mcp.pipelines.derivations import Region, derive_detector_geometry

    edge = min(frame)
    boxes = np.array(
        [[0, 0, edge / 4, edge / 4], [0, 0, edge / 8, edge / 4]],
        dtype=np.float32,
    )
    return derive_detector_geometry(
        frame,
        [Region(boxes, np.ones(2, dtype=np.int64), frame[0] * frame[1])],
    )


def stated_detector_geometry(frame: tuple[int, int] = (64, 64)) -> dict:
    """Return :func:`detector_geometry`'s values as a ``model_source.detector_geometry`` record.

    The record's source states :data:`RATIONALE`.
    """
    return {
        **detector_geometry(frame).handed(),
        "source": {"rationale": RATIONALE},
    }


def band_statistics(bands: int = 3) -> dict:
    """Return the ``image_mean`` and ``image_std`` of :func:`noise_frames` at ``bands``.

    They are the band statistics as their record hands them to a builder.
    """
    from tcip_mcp.pipelines.schemas import Normalization

    return Normalization.model_validate(
        frame_statistics(noise_frames(bands), bands)
    ).handed()


def detector_values(frame: tuple[int, int] = (64, 64), bands: int = 3) -> dict:
    """Return :func:`detector_geometry`'s keywords for ``frame`` and :func:`band_statistics`'.

    The band statistics are taken at ``bands``.
    """
    return {**detector_geometry(frame).handed(), **band_statistics(bands)}


def frame_statistics(images: list, bands: int) -> dict:
    """Return the ``normalization`` record of the band statistics of ``images``, in JSON form.

    ``derivations.band_normalization_stats`` takes them at ``bands`` over all of ``images``, at
    seed 0.
    """
    from tcip_mcp.pipelines.derivations import band_normalization_stats

    stats = band_normalization_stats(
        images, bands, max_images=len(images), seed=0
    )
    assert stats is not None, "no fixture frame could be read"
    return stats.model_dump(mode="json")


def weights_normalization() -> dict:
    """Return the normalization record of the timm tag ``resnet18.a1_in1k``, in JSON form.

    The record is the pretrained weights' own (``derivations.pretrained_normalization``),
    sourced by those weights.
    """
    from tcip_mcp.pipelines.derivations import pretrained_normalization
    from tcip_mcp.pipelines.schemas import PretrainedWeights

    weights = PretrainedWeights(library="timm", name="resnet18.a1_in1k")
    return pretrained_normalization(weights).model_dump()


def weights_statistics(model_source: dict) -> dict:
    """Return ``model_source`` normalizing by :func:`weights_normalization`."""
    return {**model_source, "normalization": weights_normalization()}


def own_statistics(config: dict, project: Path) -> dict:
    """Return ``config`` with a :data:`FRAME_STATISTICS` normalization made the frames' own.

    The normalization becomes the band statistics (:func:`frame_statistics`) of the train side
    of the partition ``config``'s data section resolves to under ``project``, at the band count
    that resolution records (``split_construction.auto_train_val``, through
    ``_producer_fixtures.train_val``). Every
    refusal of the resolution or the derivation propagates. A ``config`` stating any other
    normalization is returned as it is.
    """
    from tests._producer_fixtures import train_val

    source = config["model_source"]
    if source.get("normalization") != FRAME_STATISTICS:
        return config
    _train, _val, partition, resolved = train_val(
        project, source["task"], config["data"]
    )
    statistics = frame_statistics(
        train_sources(partition), resolved.num_channels
    )
    return {**config, "model_source": {**source, "normalization": statistics}}


def train_sources(partition: Partition) -> list[str]:
    """Return the sources of the train side of a resolved ``partition``, in its order."""
    return [str(s.source) for s in partition.samples if s.side == "train"]


_NOISE: dict = {}
"""The directory :func:`noise_frames` writes at each band count, held so it is removed at
exit."""


def noise_frames(bands: int = 3) -> list:
    """Return the paths of two TIFFs of ``(32, 32, bands)`` uint8 values drawn at seed ``bands``.

    The files are written by ``tifffile.imwrite`` once per ``bands`` for the session, to a
    temporary directory removed at exit.
    """
    import tempfile

    import numpy as np
    import tifffile

    if bands not in _NOISE:
        _NOISE[bands] = tempfile.TemporaryDirectory(prefix="tcip-noise-")
        rng = np.random.default_rng(bands)
        for index in range(2):
            tifffile.imwrite(
                Path(_NOISE[bands].name) / f"noise{index}.tif",
                rng.integers(0, 256, (32, 32, bands), dtype=np.uint8),
            )
    return sorted(str(p) for p in Path(_NOISE[bands].name).glob("*.tif"))


def object_at(index: int) -> tuple[int, int, int]:
    """Return the top-left corner and edge length of frame ``index``'s object, set by ``index``."""
    x0 = 4 + (index % 5) * 8
    y0 = 4 + ((index // 5) % 5) * 8
    size = 14 + (index % 3) * 4
    return x0, y0, size


def synthetic_capture(root: Path, *, date: str = DATE) -> Path:
    """Ingest one capture date of painted frames through ``ingest_images``.

    Each of :data:`STEMS` is painted with a :data:`~tests._producer_fixtures.BRIGHT` square at
    :func:`object_at` of its index, ingested, and labeled with that square. Returns the capture's
    image directory.
    """
    from datetime import datetime

    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._image_fixtures import capture_exif
    from tests._producer_fixtures import BRIGHT, label_image, painted_frame

    raw = root.parent / f"raw-{date}"
    raw.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(STEMS):
        x0, y0, size = object_at(index)
        shade = 28 + (index % 7)
        painted_frame(
            IMG,
            IMG,
            (shade, shade, shade),
            [((x0, y0, x0 + size, y0 + size), BRIGHT)],
        ).save(
            raw / f"{stem}.png",
            exif=capture_exif(datetime.fromisoformat(date).replace(hour=9)),
        )

    ingested = ingest_images(root, source=str(raw), actor=None)
    assert "error" not in ingested, ingested
    assert ingested["copied"] == len(STEMS), ingested

    images_dir = image_dir(root, date)
    for index, stem in enumerate(STEMS):
        x0, y0, size = object_at(index)
        label_image(
            images_dir / f"{stem}.png",
            [
                Annotation(
                    subject=SUBJECT,
                    geometry=BBox(x0, y0, x0 + size, y0 + size),
                )
            ],
            IMG,
            IMG,
        )
    return images_dir


def draw_reference_selection(
    project: Path, root: Path, out: Path
) -> Selection:
    """Draw ``root``'s labels into train, val, calibration and holdout sides with ``draw_splits``.

    Returns the selection as recorded.
    """
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.data_tools import draw_splits

    result = draw_splits(
        project,
        str(root),
        output_path=str(out),
        subject=SUBJECT,
        **drawn_split(
            seed=2,
            val_ratio=0.2,
            calibration_ratio=0.2,
            holdout_ratio=0.2,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in result, result
    return read_selection(out, project=project)


def run_config(
    project: Path,
    selection_dir: Path,
    model_source: dict = REGION_BUILDER,
    **overrides: object,
) -> dict:
    """Return a :func:`training_config` binding ``project``'s drawn selection at ``selection_dir``.

    The config names ``model_source`` and that selection, with ``overrides`` over its keys.
    """
    return training_config(
        project,
        model_source,
        {"split": {"selection_dir": str(selection_dir)}},
        **overrides,
    )


def training_config(
    project: Path, model_source: dict, data: dict, **overrides: object
) -> dict:
    """Return a run config of ``project`` over a copy of ``model_source`` and ``data``.

    The config states ``data`` over a ``tests._training_values.drawn_split`` block, seed 0, CPU,
    batch size 2, one :func:`~tests._training_values.stage` at 0, the sample stop rule, SGD
    optimizer and evaluation block (with the sample mask cut for an ``instance_seg`` task), and a
    cosine schedule at a one-epoch horizon, with a copy of each of ``overrides`` in place of its
    key. The result passes through :func:`own_statistics` under ``project``.
    """
    import copy

    config = {
        "model_source": copy.deepcopy(model_source),
        "data": {"split": drawn_split(), **data},
        "rationale": RATIONALE,
        "batch_size": 2,
        "seed": 0,
        "deterministic": False,
        "stages": [stage(0)],
        "mixed_precision": False,
        "device": "cpu",
        "checkpoint_every_n_epochs": 1,
        "early_stopping": stop_rule(),
        "evaluation": evaluation_block(
            **(
                {"mask_cut": asdict(MASK_CUT)}
                if model_source.get("task") == "instance_seg"
                else {}
            )
        ),
        "optimizer": sgd_optimizer(backbone_lr=1e-2),
        "scheduler": schedule("cosine", horizon_epochs=1),
        "gradient_accumulation_steps": 1,
        **copy.deepcopy(overrides),
    }
    return own_statistics(config, project)


def built_model(
    config: dict, detector: DetectorGeometry | None = None
) -> nn.Module:
    """Return the model a run over ``config`` builds.

    The model is its validated ``model_source``, imported from the layout its admission stages
    with this repository as its project (``model_build.staged_sources``), at the dims its data
    block records (``model_build.recorded_model_dims``) and, for a detector, at the geometry
    ``detector`` (a ``schemas.DetectorGeometry``).
    """
    from tcip_mcp.pipelines.model_build import (
        build_from_model_source,
        recorded_model_dims,
        staged_sources,
    )
    from tcip_mcp.pipelines.schemas import TrainGeometry, train_config
    from tests import REPO_ROOT

    spec = train_config(config)
    return build_from_model_source(
        spec.model_source,
        staged_sources(spec, REPO_ROOT).layout,
        recorded_model_dims(spec),
        TrainGeometry(
            native_size=None, object_density=None, detector=detector
        ),
    )


def train_on(
    selection_dir: Path, project_root: Path, experiment_id: str
) -> str:
    """Train the tiny detector over the selection's train side through the child's own entry.

    The entry's completion registers the checkpoint. Returns the checkpoint's path.
    """
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import worker_run

    observation = observe(
        worker_run(
            project_root,
            run_config(project_root, selection_dir),
            experiment_id=experiment_id,
        )
    )
    checkpoint = observation.checkpoint
    assert checkpoint is not None, observation.final
    return checkpoint.path


def confirm_count_trait(project_root: Path, **fields: object) -> TraitRevision:
    """Propose the count trait and confirm it as the user ``chain-breeder``.

    The trait states its per-image count of :data:`SUBJECT`, with ``fields`` over the fixture
    floors.
    """
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx

    revision = fx.propose(
        project_root,
        fx.with_operationalization(
            fx.with_fields(fx.COUNT_SPEC, **fields),
            traits.PER_IMAGE_COUNT,
            measured_subject=SUBJECT,
        ),
    )
    assert not revision.confirmed, "a proposed revision must land unconfirmed"
    return fx.confirm(project_root, revision, user="chain-breeder")


def assess(
    project: Path,
    checkpoint_path: str,
    selection_dir: Path,
    *,
    device: str | None = None,
    **stated: object,
) -> dict:
    """Assess the count trait's per-image count over ``selection_dir`` with ``assess_checkpoint``.

    The assessment runs under the sample collection pass with the ``stated`` execution values
    over it, at the sample count test settings.
    """
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tests import _trait_fixtures as fx

    return assess_checkpoint(
        project,
        checkpoint_path=checkpoint_path,
        trait=fx.COUNT_TRAIT,
        delivery_kind="per_image_count",
        selection_dir=str(selection_dir),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=collection_pass(**stated),
        device=device,
        actor=None,
    )


@dataclass
class Chain:
    """What one run of the flow leaves behind.

    ``bucket`` is the name its predictions are published under ``root``.
    """

    root: Path
    images_dir: Path
    selection_dir: Path
    checkpoint_path: str
    assessment: dict
    bucket: str
    published: dict

    def read(self) -> Bucket:
        """Return the chain's published bucket, as its record states it."""
        from tcip_mcp.buckets import read_bucket

        return read_bucket(self.root, self.bucket)


def unassessed_bucket(
    project: Path, *, experiment_id: str, bucket_name: str = "plain"
) -> Bucket:
    """Return a bucket published under no assessment.

    Ingests, draws, trains and publishes under ``project``.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    root = project / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    checkpoint_path = train_on(selection_dir, project, experiment_id)
    bucket = f"{bucket_name}/{DATE}"
    published = run_inference(
        project,
        checkpoint_path=checkpoint_path,
        images_dir=str(images_dir),
        bucket=bucket,
        stated=chain_pass(),
        actor=None,
    )
    assert "error" not in published, published
    return read_bucket(root, bucket)


def acknowledged[T](
    project: Path,
    deliver: Callable[[str | None], T],
    *,
    by: str = "user:breeder",
    reason: str = "fixture: shipped before any assessment",
) -> T:
    """Run ``deliver`` refused, then again under a recorded acknowledgment by ``by``.

    ``deliver`` is a delivery taking a recorded acknowledgment's id. It is refused first, the
    refused result acknowledged (:func:`~tcip_mcp.delivery.record_acknowledgment`), then
    delivered under that acknowledgment; returns what the second call returns. A refusal no
    acknowledgment can answer propagates.
    """
    import pytest

    from tcip_mcp.delivery import DeliveryRefusedError, record_acknowledgment

    with pytest.raises(DeliveryRefusedError) as refused:
        deliver(None)
    if refused.value.result_sha256 is None:
        raise refused.value
    act = record_acknowledgment(
        project,
        acknowledged_by=by,
        reason=reason,
        result_sha256=refused.value.result_sha256,
    )
    return deliver(act.acknowledgment_id)


def deliver_acknowledged(
    project: Path,
    records: list[dict],
    out: Path,
    delivered_phenotype: str,
    *,
    value_key: str,
    delivery_kind: str,
    buckets: list | None = None,
    **kwargs: str | dict | None,
) -> dict:
    """Deliver the per-image ``records``' ``value_key`` per plant, under an acknowledgment.

    ``deliver_per_plant_aggregate`` runs for the plants the records carry, over ``buckets`` (by
    default a bucket published under no assessment, :func:`unassessed_bucket`), shipped
    unvalidated under the breeder's acknowledgment (:func:`acknowledged`).
    """
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )

    delivered = (
        buckets
        if buckets is not None
        else [unassessed_bucket(project, experiment_id="exp-acknowledged")]
    )
    return acknowledged(
        project,
        lambda ack: deliver_per_plant_aggregate(
            project,
            records,
            str(out),
            value_key=value_key,
            delivered_phenotype=delivered_phenotype,
            delivery_kind=delivery_kind,
            buckets=delivered,
            plants=list(dict.fromkeys(r["plant_id"] for r in records)),
            door="test_delivery",
            acknowledgment_id=ack,
            actor=None,
            **kwargs,
        ),
    )


def predicted(image: Path, values: list[str], attributes: tuple = ()) -> dict:
    """Return one predictor result for the image at ``image``.

    The result holds one 3 px box labeled 1 and scored 0.9 per entry of ``values``, each
    carrying that value's index under the first of ``attributes`` (the scope's attribute
    records) and 0 under every other; no ``attributes`` row for none. Its count is the number
    of boxes and its cap one more.
    """
    result = {
        "image": str(image),
        "width": IMG,
        "height": IMG,
        "boxes": [
            [4.0 * k, 0.0, 4.0 * k + 3.0, 3.0] for k in range(len(values))
        ],
        "scores": [0.9] * len(values),
        "labels": [1] * len(values),
        "count": len(values),
        "cap": len(values) + 1,
    }
    if attributes:
        result["attributes"] = [
            [attributes[0].values.index(v)] + [0] * (len(attributes) - 1)
            for v in values
        ]
    return result


def published(
    project: Path,
    name: str,
    results: list[dict],
    *,
    scope: dict,
    registry: SubjectRegistry | None = None,
    raster_path: Path | None = None,
) -> Bucket:
    """Publish ``results`` as the bucket named ``name`` and return the bucket.

    :func:`~tcip_mcp.buckets.publish` writes it under their images' dataset root, under an
    untiled pass prepared, not run, for a registered checkpoint stating ``scope`` (its frames'
    dataset declaring ``registry``), over the raster ``raster_path`` when one is named.
    """
    from tcip_mcp.buckets import pass_documents, publish, source_root
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.split_construction import raster_identity
    from tcip_mcp.pipelines.execution import prepare
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_CONF,
        project_checkpoint,
    )

    checkpoint = load_registered_checkpoint(
        project_checkpoint(
            project,
            data={"num_channels": 3, "scope": scope},
            registry=registry,
        ),
        project=project,
    )
    identity = (
        raster_identity(resolve_image_path(raster_path))
        if raster_path is not None
        else None
    )
    p = prepare(
        checkpoint, Stated(tile=False, conf=SAMPLE_CONF, rationale=RATIONALE)
    ).runnable()
    root = source_root(
        [raster_path]
        if raster_path is not None
        else [r["image"] for r in results]
    )
    return publish(
        project,
        root,
        name,
        pass_documents(p, results),
        producer=checkpoint.producer,
        scope=p.scope,
        execution=p.execution,
        raster_path=str(raster_path) if raster_path is not None else None,
        raster_identity=identity,
        assessment_id=None,
        actor=None,
    )


VALUES = ("open", "closed")
"""The attributed chain's attribute values, in the registry's declared order: a value's index is
the band its blobs are bright in, and the class the blob detector calls them."""
ATTRIBUTE = "opening"
SERIES_DATES = ("2026-02-11", "2026-02-25", "2026-03-10", "2026-03-24")
PLANTS = {"PLANT_A": (43.19700, -90.05800), "PLANT_B": (43.19700, -90.05780)}
"""Where each plant of the attributed chain's registry stands (WGS84 lat, lon), about 16 m
apart."""
REFERENCE_SITE = (43.20300, -90.05000)
"""Where the labeled reference frames of each date were taken, about 900 m from every plant."""
SERIES_POSITION_ERROR_M = 3.0
"""The horizontal positioning error (meters) every photograph of the attributed series reports,
sample data well under the spacing of :data:`PLANTS`."""


def blob_boxes(values: list[str], index: int) -> list[tuple[str, BBox]]:
    """Return one square box per entry of ``values`` beside its value, sized by ``index``."""
    out = []
    for k, value in enumerate(values):
        x0, y0 = 2 + (k % 4) * 16, 2 + (k // 4) * 16
        size = 8 + (index + k) % 4
        out.append((value, BBox(x0, y0, x0 + size, y0 + size)))
    return out


def blob_frame(values: list[str], index: int) -> Image.Image:
    """Return a painted frame of a gray shade set by ``index``, with each :func:`blob_boxes` box.

    Each box is painted 230 in its value's band of :data:`VALUES` and 20 in the others.
    """
    from tests._producer_fixtures import painted_frame

    shade = 20 + (index % 11)
    return painted_frame(
        IMG,
        IMG,
        (shade, shade, shade),
        [
            (
                (box.x1, box.y1, box.x2, box.y2),
                tuple(
                    230 if band == VALUES.index(value) else 20
                    for band in range(3)
                ),
            )
            for value, box in blob_boxes(values, index)
        ],
    )


def confirm_crossing_trait(
    project_root: Path, **fields: object
) -> TraitRevision:
    """Confirm ``bud_opening`` stating its crossing dates over :data:`SUBJECT`.

    Over ``_trait_fixtures.BUD_OPENING``, ``count_bias_tolerance_frac`` is authored at 0.1,
    ``count_error_tolerance`` at 1.0 and ``classifier_agreement_floor`` at 0.6, with ``fields``
    over them.
    """
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx

    floors = {
        "count_bias_tolerance_frac": 0.1,
        "count_error_tolerance": 1.0,
        "classifier_agreement_floor": 0.6,
        **fields,
    }
    spec = fx.with_fields(fx.BUD_OPENING, **floors)
    return fx.propose_and_confirm(
        project_root,
        fx.with_operationalization(
            spec,
            traits.STATE_CROSSING_DATES,
            measured_subject=SUBJECT,
            statement="the date each plant reached the open state the breeder scores",
            mechanism="the assessed classifier over one plant's objects",
            delivered_phenotypes=spec.delivers,
        ),
    )


@dataclass
class Series:
    """What the attributed chain leaves behind.

    It holds the chain's dataset, the checkpoint, the assessment (when one was run), the plant
    mapping, the name of one published bucket per date and the project as the web backend opened
    it.
    """

    root: Path
    checkpoint_path: str
    assessment: dict | None
    mapping_name: str
    buckets: dict[str, str]
    project: OpenProject
    trait: str = "bud_opening"

    def body(self, **extra: object) -> dict:
        """Return the request body a phenology door takes over this series."""
        return {
            "mapping_name": self.mapping_name,
            "trait": self.trait,
            "dataset_root": str(self.root),
            "buckets": list(self.buckets.values()),
            "plants": list(PLANTS),
            **extra,
        }


def deliver_milestones(project: Path, body: dict, out_csv: Path) -> dict:
    """Run ``deliver_phenology_milestones`` over a phenology door's request ``body``.

    ``body`` is a :meth:`Series.body`; the delivery writes ``out_csv``.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones(
        project,
        trait=body["trait"],
        mapping_name=body["mapping_name"],
        plants=body["plants"],
        dataset_root=body["dataset_root"],
        buckets=body["buckets"],
        output_csv_path=str(out_csv),
        actor=None,
    )


def attributed_series(
    project: Path,
    *,
    fractions: tuple[float, ...] = (0.0, 0.25, 0.75, 1.0),
    detections: int = 4,
    images_per_plant: int = 1,
    assessed: bool = True,
    experiment_id: str = "exp-attributed",
    attributes: tuple | None = None,
    model_source: dict | None = None,
    stated: Stated | None = None,
) -> Series:
    """Ingest, train, assess and publish an attributed series, then open ``project``.

    It ingests a geotagged series of :data:`PLANTS` over ``len(fractions)`` of
    :data:`SERIES_DATES`, each date beside labeled reference frames taken at
    :data:`REFERENCE_SITE`; draws the labeled frames into a selection, trains ``model_source``
    (by default the blob detector) on it, confirms the crossing trait, assesses it when
    ``assessed``, publishes one bucket per date under the ``stated`` execution values (under the
    assessment when there is one, :func:`chain_pass` when neither is given), builds the plant
    mapping, and opens ``project`` in the web backend.

    The project's and the dataset's registries declare ``attributes`` on :data:`SUBJECT`, by
    default :data:`ATTRIBUTE` categorical over :data:`VALUES`.
    Series and reference frames are :func:`blob_frame` images over values built from
    ``fractions`` and ``detections``, and each reference frame is labeled with its
    :func:`blob_boxes`.
    """
    from datetime import datetime, timedelta

    from tcip_mcp import subject_registry as cr
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.inference_tools import run_inference
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image
    from tests._mapping_fixtures import (
        register_plant_registry_for,
        write_plant_csv,
    )
    from tests._producer_fixtures import (
        label_image,
        registry_of,
        registry_over,
    )
    from tests._verified_checkpoint_fixtures import worker_run
    from tests._web_fixtures import open_new_project

    root = project / "ds"
    attributes = attributes or (
        cr.Attribute(name=ATTRIBUTE, type="categorical", values=VALUES),
    )
    for declared in (project, root):
        registry_over(declared, registry_of(SUBJECT, *attributes))
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
                write_geo_image(
                    capture / f"{plant}_{date}_{i}.jpg",
                    lat,
                    lon,
                    when + timedelta(seconds=index),
                    blob_frame(values, index),
                    h_pos_err=SERIES_POSITION_ERROR_M,
                )
        labeled = {}
        for i in range(per_date):
            index = per_date * d + i
            held = detections if index % 2 == 0 else 2 if index % 3 else 1
            labeled[f"ref_{date}_{i:02d}"] = (
                index,
                [VALUES[(index + k) % 2] for k in range(held)],
            )
        for stem, (index, frame_values) in labeled.items():
            write_geo_image(
                capture / f"{stem}.jpg",
                *REFERENCE_SITE,
                when + timedelta(hours=2, minutes=index),
                blob_frame(frame_values, index),
                h_pos_err=SERIES_POSITION_ERROR_M,
            )
        ingested = ingest_images(root, source=str(capture), actor=None)
        assert "error" not in ingested, ingested
        for stem, (index, frame_values) in labeled.items():
            label_image(
                image_dir(root, date) / f"{stem}.jpg",
                [
                    Annotation(
                        subject=SUBJECT,
                        geometry=box,
                        attributes={
                            a.name: value
                            if a.name == ATTRIBUTE
                            else a.values[(index + k) % len(a.values)]
                            for a in attributes
                        },
                    )
                    for k, (value, box) in enumerate(
                        blob_boxes(frame_values, index)
                    )
                ],
                IMG,
                IMG,
            )
    registered = register_dataset(
        project, str(root), crop=sorted(registered_crops())[0], actor=None
    )
    assert "error" not in registered, registered

    selection_dir = project / "selection"
    drawn = draw_splits(
        project,
        str(root),
        output_path=str(selection_dir),
        subject=SUBJECT,
        **drawn_split(
            seed=2,
            val_ratio=0.2,
            calibration_ratio=0.2,
            holdout_ratio=0.2,
            grouping="tile_prefix",
        ),
        actor=None,
    )
    assert "error" not in drawn, drawn
    config = run_config(
        project, selection_dir, model_source or BLOB_BUILDER, seed=2
    )
    from tcip_mcp.experiments import observe

    checkpoint = observe(
        worker_run(project, config, experiment_id=experiment_id)
    ).checkpoint
    assert checkpoint is not None
    checkpoint_path = checkpoint.path
    confirm_crossing_trait(project)

    assessment = None
    if assessed:
        assessment = assess_checkpoint(
            project,
            checkpoint_path=checkpoint_path,
            trait="bud_opening",
            delivery_kind="state_crossing_dates",
            selection_dir=str(selection_dir),
            equivalence_z=EQUIVALENCE_Z,
            dispersion_quantile=DISPERSION_QUANTILE,
            rationale=RATIONALE,
            stated=collection_pass(),
            actor=None,
        )
        assert "error" not in assessment, assessment
        assert read_assessment(project, assessment["assessment_id"]).passed, (
            assessment["failures"]
        )

    if stated is None:
        stated = Stated() if assessment else chain_pass()
    predictions: dict[str, str] = {}
    for date in dates:
        published = run_inference(
            project,
            checkpoint_path=checkpoint_path,
            images_dir=str(image_dir(root, date)),
            bucket=f"series/{date}",
            stated=stated,
            assessment_id=assessment["assessment_id"] if assessment else None,
            actor=None,
        )
        assert "error" not in published, published
        predictions[date] = f"series/{date}"

    plants_csv = write_plant_csv(
        raw / "plants.csv",
        [
            {
                "plot": plant,
                "accession": f"Acc{plant[-1]}",
                "lat": lat,
                "lon": lon,
            }
            for plant, (lat, lon) in PLANTS.items()
        ],
    )
    registry = register_plant_registry_for(project, [plants_csv])
    mapped = build_plant_mapping(
        project,
        name="valley",
        dataset_root=str(root),
        plant_registry=registry,
        dates=list(dates),
        actor=None,
    )
    assert "error" not in mapped, mapped
    return Series(
        root,
        checkpoint_path,
        assessment,
        "valley",
        predictions,
        open_new_project(project),
    )


def trained_detector(
    project: Path, *, experiment_id: str
) -> tuple[Path, Path, Path, str]:
    """Ingest, draw, train and confirm the count trait under ``project``.

    Returns the dataset root, its capture's images, the reference selection and the checkpoint.
    """
    root = project / "ds"
    images_dir = synthetic_capture(root)
    selection_dir = project / "selection"
    draw_reference_selection(project, root, selection_dir)
    checkpoint_path = train_on(selection_dir, project, experiment_id)
    confirm_count_trait(project)
    return root, images_dir, selection_dir, checkpoint_path


def run_the_chain(
    project: Path, *, experiment_id: str, bucket_name: str = "chain"
) -> Chain:
    """Ingest, train, confirm, assess and publish under ``project``; the assessment must pass."""
    from tcip_mcp.tools.inference_tools import run_inference

    root, images_dir, selection_dir, checkpoint_path = trained_detector(
        project, experiment_id=experiment_id
    )
    assessment = assess(project, checkpoint_path, selection_dir)
    assert "error" not in assessment, assessment
    assert read_assessment(project, assessment["assessment_id"]).passed, (
        assessment["failures"]
    )

    bucket = f"{bucket_name}/{DATE}"
    published = run_inference(
        project,
        checkpoint_path=checkpoint_path,
        images_dir=str(images_dir),
        bucket=bucket,
        assessment_id=assessment["assessment_id"],
        stated=Stated(),
        actor=None,
    )
    assert "error" not in published, published
    return Chain(
        root,
        images_dir,
        selection_dir,
        checkpoint_path,
        assessment,
        bucket,
        published,
    )

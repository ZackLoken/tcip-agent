"""One scope over every attribute of a subject.

A detector carries one per-instance head per attribute its registry declares, gives each head a
loss over the instances that attribute assesses, publishes every attribute's value on every box,
merges tiles over the one subject, and is evaluated per attribute over matched pairs.

The registries, selections, runs and buckets are made by the platform's own producers: the
registry through ``registry_over``, the selection through ``draw_splits``, the run through the
child's own entry, the bucket through ``run_inference``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tcip_annotation.state import Annotation

pytest.importorskip("torch")

pytest.importorskip("torchvision")
import torch  # noqa: E402

from tcip_annotation.json_io import UNASSESSED  # noqa: E402
from tcip_mcp import subject_registry as cr  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    ATTRIBUTE,
    FRAME_STATISTICS,
    PLANTS,
    VALUES,
    attributed_series,
)
from tests._producer_fixtures import registry_of  # noqa: E402
from tests._training_values import (  # noqa: E402
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    IOU_THRESHOLD,
    RATIONALE,
)
from tests._verified_checkpoint_fixtures import BUILT_DETECTOR  # noqa: E402

UNFILTERED = Stated(tile=False, conf=0.0, rationale=RATIONALE)
"""An untiled pass keeping every box up to the checkpoint's density."""
SUBJECT = "object"
COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))
REGISTRY = registry_of(SUBJECT, COLOR, GRADE)
OPENING = cr.Attribute(ATTRIBUTE, "ordinal", VALUES)
"""The series' opening attribute declared ordinal, the one its crossing trait's state names."""


def _documents(root: Path, bucket: str) -> dict[str, list[Annotation]]:
    """Return every prediction document of the bucket named ``bucket`` under ``root`` by stem.

    Each document's annotations are as written.
    """
    from tcip_annotation.json_io import read_predictions
    from tcip_mcp.buckets import read_bucket

    return {
        stem: read_predictions(key).annotations
        for stem, key in read_bucket(root, bucket).stem_keys.items()
    }


def _frames(where: Path, values: dict[str, str]) -> str:
    """Write two detection frames of :data:`SUBJECT` under :data:`REGISTRY`.

    The frames are :func:`~tests._verified_checkpoint_fixtures.detection_images`', each object
    carrying ``values``. Returns the images directory.
    """
    from tests._verified_checkpoint_fixtures import detection_images

    return detection_images(
        where, {"subject": SUBJECT}, values=values, registry=REGISTRY
    )["images_dir"]


# --- two attributes, end to end ------------------------------------------


def test_two_attributes_train_two_heads_publish_both_values_and_deliver(
    tmp_path: Path,
) -> None:
    """Two attributes on one subject train two heads, publish both values and deliver.

    A registry declares an ordinal and a categorical attribute on one subject: the drawn
    selection's scope carries both, every box of every document the trained detector publishes
    carries a value of each, the crossing trait on the ordinal attribute is
    assessed over the selection and its milestones deliver.
    """
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.postprocessing.phenology import (
        deliver_phenology,
        measure_phenology,
    )
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tests._chain_fixtures import acknowledged

    series = attributed_series(
        tmp_path,
        fractions=(0.0, 1.0),
        assessed=False,
        attributes=(OPENING, COLOR),
        model_source=BUILT_DETECTOR,
        stated=UNFILTERED,
        experiment_id="exp-two-heads",
    )

    selection = read_selection(tmp_path / "selection", project=tmp_path)
    assert selection.scope.attributes == (OPENING, COLOR)
    boxes = [
        a
        for bucket in series.buckets.values()
        for written in _documents(series.root, bucket).values()
        for a in written
    ]
    assert boxes, {
        b: _documents(series.root, b) for b in series.buckets.values()
    }
    for a in boxes:
        assert set(a.attributes) == {ATTRIBUTE, COLOR.name}, a
        assert (
            a.attributes[ATTRIBUTE] in VALUES
            and a.attributes["color"] in COLOR.values
        )

    assessed = assess_checkpoint(
        tmp_path,
        checkpoint_path=series.checkpoint_path,
        trait=series.trait,
        delivery_kind="state_crossing_dates",
        selection_dir=str(tmp_path / "selection"),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=UNFILTERED,
        actor=None,
    )
    assert "error" not in assessed, assessed
    assert "failures" in assessed

    measurement = measure_phenology(
        tmp_path,
        trait=series.trait,
        mapping_name=series.mapping_name,
        dataset_root=str(series.root),
        buckets=list(series.buckets.values()),
        plants=list(PLANTS),
    )
    out = tmp_path / "out" / "milestones.csv"
    delivered = acknowledged(
        tmp_path,
        lambda ack: deliver_phenology(
            tmp_path,
            measurement,
            projection="milestones",
            output_path=out,
            acknowledgment_id=ack,
            door="test_delivery",
            actor=None,
        ),
    )
    assert delivered["n_rows"] == len(PLANTS), delivered
    assert out.exists()


def test_a_selection_and_the_bucket_a_run_over_it_publishes_carry_one_scope_record(
    tmp_path: Path,
) -> None:
    """A selection and the bucket a run over it publishes carry one scope record.

    Each bucket the run's checkpoint publishes states the drawn selection's scope, in its read
    record and in its stored one.
    """
    import tcip_store
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import bucket_key
    from tcip_mcp.pipelines.data.selection import read_selection

    series = attributed_series(
        tmp_path,
        fractions=(0.0, 1.0),
        assessed=False,
        attributes=(OPENING, COLOR),
    )

    drawn = read_selection(tmp_path / "selection", project=tmp_path).scope
    for bucket in series.buckets.values():
        published = read_bucket(series.root, bucket).scope
        assert published == drawn
        stored = tcip_store.read(bucket_key(series.root, bucket))
        assert isinstance(stored, dict)
        assert stored["scope"] == json.loads(json.dumps(asdict(drawn)))


# --- zero attributes --------------------------------------------------------


def test_a_scope_with_no_attribute_builds_the_plain_detector_and_writes_no_attribute_row(
    tmp_path: Path,
) -> None:
    """A scope with no attribute builds the plain detector and writes no attribute.

    A subject its registry declares no attribute for builds the torchvision detector
    ``build_detector`` returns, and its records and documents carry no attribute.
    """
    from torchvision.models.detection import FasterRCNN

    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._producer_fixtures import write_image
    from tests._verified_checkpoint_fixtures import (
        predicted_over,
        registered_checkpoint,
    )

    images_dir = image_dir(tmp_path, "frames")
    for i in range(2):
        write_image(images_dir / f"f{i}.png", (64, 64), (40 + 60 * i, 90, 60))
    checkpoint = registered_checkpoint(tmp_path)

    p, results = predicted_over(
        tmp_path, checkpoint, str(images_dir), tile=False, conf=0.0
    )
    assert type(p.predictor.model.detector) is FasterRCNN
    assert results and all("attributes" not in r for r in results), results

    published = run_inference(
        tmp_path,
        checkpoint,
        str(images_dir),
        bucket="plain",
        stated=UNFILTERED,
        actor=None,
    )
    assert "error" not in published, published
    assert len(_documents(tmp_path, "plain")) == 2
    for written in _documents(tmp_path, "plain").values():
        assert all(not a.attributes for a in written), written


def test_a_document_scope_whose_attributes_were_never_read_sizes_no_model(
    tmp_path: Path,
) -> None:
    """A scope whose attributes were never read sizes no model; one read by admission does.

    A run config naming a subject but recording no attributes was never admitted, so its
    model's head sizes are unknown: the dims refuse, naming the absent attributes. That subject
    read through the admission sizes one.
    """
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.model_build import recorded_model_dims
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config, weights_statistics

    unread = training_config(
        tmp_path,
        weights_statistics(BUILT_DETECTOR),
        {"num_channels": 3, "scope": {"subject": SUBJECT}},
    )
    with pytest.raises(ValueError, match="records no attributes"):
        recorded_model_dims(train_config(unread))

    images_dir = _frames(tmp_path / "ds", {"color": "red"})
    read = training_config(
        tmp_path,
        weights_statistics(BUILT_DETECTOR),
        {
            "num_channels": 3,
            "scope": asdict(registry_scope(images_dir, SUBJECT)),
        },
    )
    assert recorded_model_dims(train_config(read))["attributes"] == (
        COLOR,
        GRADE,
    )


# --- an unassessed instance -------------------------------------------------


def test_an_instance_unassessed_for_one_attribute_trains_that_head_on_nothing_and_the_other(
    tmp_path: Path,
) -> None:
    """``color``, assessed by no instance, gives its head no gradient while ``grade``'s gets one.

    Every object assesses ``grade`` and none assesses ``color``: both frames are admitted and the
    color column reads unassessed. After one backward pass the color loss is zero and its head's
    gradients empty, while the grade loss and its head's gradients are nonzero.
    """
    from tests._chain_fixtures import (
        built_model,
        detector_geometry,
        training_config,
    )
    from tests._producer_fixtures import run_over

    images_dir = _frames(tmp_path / "ds", {"grade": "high"})
    loader, data = run_over(
        "detection", images_dir, subject=SUBJECT, stated={"num_channels": 3}
    )
    assert len(loader) == 2
    image, target = loader[0]
    assert target["attributes"].tolist() == [[UNASSESSED, 2]]

    model = built_model(
        training_config(
            tmp_path, BUILT_DETECTOR, {**data, "images_dir": str(images_dir)}
        ),
        detector_geometry(),
    )
    model.train()
    losses = model([image], [target])
    color_head, grade_head = model.detector.attribute_heads
    (
        losses["attribute_loss_color"] + losses["attribute_loss_grade"]
    ).backward()

    assert float(losses["attribute_loss_color"].detach()) == 0.0
    assert float(losses["attribute_loss_grade"].detach()) > 0.0
    assert all(
        p.grad is None or not p.grad.any() for p in color_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.any() for p in grade_head.parameters()
    )


@pytest.mark.parametrize(
    "name", ["faster_rcnn", "fcos", "retinanet", "mask_rcnn"]
)
def test_each_builtin_detector_carries_one_head_per_attribute(
    name: str,
) -> None:
    """Each built-in detector carries one attribute head per attribute.

    Around each built-in detector, the attribute module returns one loss per attribute in train
    mode beside the detector's own losses, the unassessed attribute's at zero, writes
    ``attributes`` boxes by attributes in eval, and leaves the operating-point knobs with a
    holder.
    """
    from tcip_mcp.pipelines.operating_point import (
        detector_operating_point_holder,
    )
    from tests._chain_fixtures import detector_values
    from tests.bespoke_models import build_bespoke_detection

    torch.manual_seed(0)
    model = build_bespoke_detection(
        detector=name, attributes=(COLOR, GRADE), **detector_values()
    )
    image = torch.rand(3, 64, 64)
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]),
        "labels": torch.tensor([1]),
        "attributes": torch.tensor([[1, UNASSESSED]]),
    }
    if name == "mask_rcnn":
        from tests._producer_fixtures import painted_array

        target["masks"] = torch.as_tensor(
            painted_array(64, 64, [((8, 8, 40, 40), 1)])
        )[None]

    model.train()
    losses = model([image], [target])
    assert {"attribute_loss_color", "attribute_loss_grade"} <= set(losses)
    assert len(losses) > 2
    assert float(losses["attribute_loss_grade"].detach()) == 0.0

    model.eval()
    holder, path = detector_operating_point_holder(model)
    assert holder is not None, path
    with torch.no_grad():
        (out,) = model([image])
    assert out["attributes"].dtype == torch.int64
    assert tuple(out["attributes"].shape) == (len(out["boxes"]), 2)


def test_the_tiler_carries_exactly_the_per_box_keys_the_loader_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tiler carries ``attributes`` while ``PER_BOX_KEYS`` names it and omits it otherwise.

    While named, each tile carries one ``attributes`` row per box, holding the objects' values.
    """
    from tcip_mcp.pipelines.data import datasets
    from tests._producer_fixtures import dataset_over

    images_dir = _frames(tmp_path / "ds", {"color": "red", "grade": "mid"})
    from tests._training_values import tiling_block

    tiling = tiling_block(tile_size=32, overlap=0.0, sliver_frac=0.5)

    tiled = dataset_over(
        "detection", images_dir, subject=SUBJECT, tiling=tiling
    )
    rows = [tiled[i][1] for i in range(len(tiled))]
    assert rows and all(
        r["attributes"].tolist() == [[0, 1]] * len(r["boxes"]) for r in rows
    )

    monkeypatch.setattr(
        datasets,
        "PER_BOX_KEYS",
        tuple(k for k in datasets.PER_BOX_KEYS if k != "attributes"),
    )
    unnamed = dataset_over(
        "detection", images_dir, subject=SUBJECT, tiling=tiling
    )
    assert all("attributes" not in unnamed[i][1] for i in range(len(unnamed)))


# --- a tiled merge over attribute values ------------------------------------


WHOLE = cr.Attribute("whole", "categorical", ("cut", "whole"))
"""Whether a tile holds its blob whole: ``cut`` or ``whole``."""


@pytest.mark.parametrize("postprocess", ["nms", "nmm", "greedynmm"])
def test_two_tiles_calling_one_object_differently_merge_into_the_higher_scoring_call(
    tmp_path: Path,
    postprocess: str,
) -> None:
    """Two tiles calling one object differently merge into one box with the higher-scoring call.

    The second column's tile holds the object whole and the first column's tile cuts it at its
    edge; under ``nms``, ``nmm`` and ``greedynmm`` the merge keeps one box, carrying the whole
    call's value.
    """
    pytest.importorskip("sahi")
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._producer_fixtures import painted_frame
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(
        tmp_path,
        model_source={
            "builder": "bespoke_models:build_whole_blob_detector",
            "builder_kwargs": {},
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        data={"num_channels": 3, "scope": {"subject": SUBJECT}},
        registry=registry_of(SUBJECT, WHOLE),
    )
    source = tmp_path / "frame.png"
    painted_frame(
        200, 200, (0, 0, 0), [((100, 20, 150, 60), (255, 0, 0))]
    ).save(source)

    p = prepare(
        load_registered_checkpoint(ckpt, project=tmp_path),
        Stated(
            tile=True,
            tile_size=128,
            overlap=0.25,
            postprocess=postprocess,
            cross_tile_nms=0.3,
            conf=0.0,
            rationale=RATIONALE,
        ),
        device="cpu",
        tile_batch_size=2,
    ).runnable()
    result = p.predictor.predict_sliced(
        str(source),
        execution=p.execution,
        tile_batch_size=p.tile_batch_size,
    )

    assert result["boxes"] == [[100.0, 20.0, 150.0, 60.0]], result
    assert result["attributes"] == [[WHOLE.values.index("whole")]], result


# --- per-attribute agreement ------------------------------------------------


class _CallsEveryFrameOnce(torch.nn.Module):
    """A detector answering one box per frame at (8, 8, 24, 24), scored 0.9 and carrying ``ids``.

    ``ids`` are the box's attribute ids; in training its loss is zero.
    """

    def __init__(self, ids: list[int]) -> None:
        super().__init__()
        self.ids = ids
        self.score_thresh = 0.0
        self.detections_per_img = 100

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        if self.training:
            return {"loss": torch.zeros(())}
        return [
            {
                "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "attributes": torch.tensor([self.ids]),
            }
            for _ in images
        ]


def test_evaluate_reports_each_attributes_agreement_over_the_pairs_it_assesses(
    tmp_path: Path,
) -> None:
    """``evaluate`` reports each attribute's agreement over the pairs its reference assesses.

    Two frames whose one object each assesses ``color`` and only the first assesses ``grade``:
    color agrees over two matched pairs and grade over the one its reference assesses.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.pipelines.training.evaluation import evaluate
    from tests._producer_fixtures import label_image, run_over

    images_dir = _frames(tmp_path / "ds", {"color": "blue", "grade": "high"})
    label_image(
        Path(images_dir) / "img1.png",
        [
            Annotation(
                subject=SUBJECT,
                geometry=BBox(8, 8, 24, 24),
                attributes={"color": "blue"},
            )
        ],
        64,
        48,
    )
    dataset, data = run_over(
        "detection", images_dir, subject=SUBJECT, stated={"num_channels": 3}
    )
    images, targets = zip(*(dataset[i] for i in range(len(dataset))))
    dims = model_dims(ClassScope.of(data), {"num_channels": 3})

    metrics = evaluate(
        _CallsEveryFrameOnce([1, 2]),
        [(list(images), list(targets))],
        "cpu",
        "detection",
        dims=dims,
        conf_threshold=0.0,
        iou_threshold=IOU_THRESHOLD,
        score_weights=None,
        trait=None,
        density=1.0,
        mask_cut=None,
    )

    agreement = metrics["attribute_agreement"]
    assert set(agreement) == {"color", "grade"}
    assert agreement["color"]["pairs"] == 2
    assert agreement["grade"]["pairs"] == 1
    assert (
        agreement["color"]["accuracy"] == agreement["grade"]["accuracy"] == 1.0
    )

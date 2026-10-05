"""One scope over every attribute of a subject: a detector carries one per-instance head per
attribute its registry declares, trains each head on the instances that attribute assesses,
publishes every attribute's value on every box, merges tiles over the one subject, and is
evaluated per attribute over matched pairs.

Every record is made by the platform's own producer: the registry through ``registry_over``, the
selection through ``draw_splits``, the run through the child's own entry, the bucket through
``run_inference``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
import torch  # noqa: E402

from tcip_annotation.json_io import UNASSESSED  # noqa: E402
from tcip_mcp import subject_registry as cr  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402

from tests._chain_fixtures import ATTRIBUTE, PLANTS, VALUES, attributed_series  # noqa: E402
from tests._verified_checkpoint_fixtures import BUILT_DETECTOR  # noqa: E402

SUBJECT = "object"
COLOR = cr.Attribute("color", "categorical", ("red", "blue"))
GRADE = cr.Attribute("grade", "ordinal", ("low", "mid", "high"))
REGISTRY = cr.SubjectRegistry(subjects=(cr.Subject(name=SUBJECT, attributes=(COLOR, GRADE)),))
OPENING = cr.Attribute(ATTRIBUTE, "ordinal", VALUES)
"""The series' opening attribute declared ordinal, the one its crossing trait's state names."""
TWO_HEADS = {"builder": "tests.bespoke_models:build_bespoke_detection",
             "builder_kwargs": {"min_size": 64, "max_size": 64}, "task": "detection"}


def _documents(root: Path, bucket: str) -> dict[str, list[dict]]:
    """Every prediction document of the bucket named ``bucket`` under ``root`` by stem, its
    annotations as written."""
    import tcip_store

    from tcip_mcp.buckets import read_bucket

    return {key.parts[-1]: tcip_store.read(key)["annotations"]
            for key in read_bucket(root, bucket).document_keys}


def _frames(where: Path, values: dict[str, str]) -> str:
    """Two :func:`~tests._verified_checkpoint_fixtures.detection_images` frames of
    :data:`SUBJECT` under :data:`REGISTRY`, each object carrying ``values``; the images
    directory."""
    from tests._verified_checkpoint_fixtures import detection_images

    return detection_images(where, {"subject": SUBJECT}, values=values,
                            registry=REGISTRY)["images_dir"]


# --- two attributes, end to end ------------------------------------------


def test_two_attributes_train_two_heads_publish_both_values_and_deliver(tmp_path: Path):
    """A registry declaring an ordinal and a categorical attribute on one subject: the drawn
    selection's scope carries both, the tiny detector trains a head for each, every box of every
    published document carries a value of each, the crossing trait on the ordinal attribute is
    assessed over the selection and its milestones deliver."""
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.postprocessing.phenology import deliver_phenology, measure_phenology
    from tcip_mcp.tools.calibration_tools import assess_checkpoint
    from tests._chain_fixtures import acknowledged

    series = attributed_series(tmp_path, fractions=(0.0, 1.0), assessed=False,
                               attributes=(OPENING, COLOR), model_source=TWO_HEADS,
                               stated=Stated(tile=False, conf=0.0),
                               experiment_id="exp-two-heads")

    selection = read_selection(tmp_path / "selection", project=tmp_path)
    assert selection.scope.attributes == (OPENING, COLOR)
    boxes = [a for bucket in series.buckets.values()
             for written in _documents(series.root, bucket).values() for a in written]
    assert boxes, {b: _documents(series.root, b) for b in series.buckets.values()}
    for a in boxes:
        assert set(a["attributes"]) == {ATTRIBUTE, COLOR.name}, a
        assert a["attributes"][ATTRIBUTE] in VALUES and a["attributes"]["color"] in COLOR.values

    assessed = assess_checkpoint(tmp_path, checkpoint_path=series.checkpoint_path,
                                 trait=series.trait, delivery_kind="state_crossing_dates",
                                 selection_dir=str(tmp_path / "selection"),
                                 stated=Stated(tile=False, conf=0.0))
    assert "error" not in assessed, assessed
    assert "passed" in assessed

    measurement = measure_phenology(tmp_path, trait=series.trait,
                                    mapping_name=series.mapping_name,
                                    dataset_root=str(series.root),
                                    buckets=list(series.buckets.values()), plants=list(PLANTS),
                                    require_all_dates_complete=True)
    out = tmp_path / "out" / "milestones.csv"
    delivered = acknowledged(tmp_path, lambda ack: deliver_phenology(
        tmp_path, measurement, curves=False, output_path=out, acknowledgment_id=ack,
        door="test_delivery", actor=None))
    assert delivered["n_rows"] == len(PLANTS), delivered
    assert out.exists()


def test_a_selection_and_the_bucket_a_run_over_it_publishes_carry_one_scope_record(
    tmp_path: Path,
):
    """The draw writes the scope its samples were admitted under; the run over the selection
    records it, and the bucket the run's checkpoint publishes states it. Both written records,
    compared to each other."""
    import tcip_store

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import bucket_key
    from tcip_mcp.pipelines.data.selection import read_selection

    series = attributed_series(tmp_path, fractions=(0.0, 1.0), assessed=False,
                               attributes=(OPENING, COLOR))

    drawn = read_selection(tmp_path / "selection", project=tmp_path).scope
    for bucket in series.buckets.values():
        published = read_bucket(series.root, bucket).scope
        assert published == drawn
        assert tcip_store.read(bucket_key(series.root, bucket))["scope"] == json.loads(
            json.dumps(asdict(drawn)))


# --- zero attributes --------------------------------------------------------


def test_a_scope_with_no_attribute_builds_the_plain_detector_and_writes_no_attribute_row(
    tmp_path: Path,
):
    """A subject its registry declares no attribute for builds the torchvision detector
    ``build_detector`` returns, and its records and documents carry no attribute."""
    from PIL import Image
    from torchvision.models.detection import FasterRCNN

    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import predicted_over, registered_checkpoint

    images_dir = tmp_path / "images" / "frames"
    images_dir.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (64, 64), (40 + 60 * i, 90, 60)).save(images_dir / f"f{i}.png")
    checkpoint = registered_checkpoint(tmp_path)

    p, results = predicted_over(tmp_path, checkpoint, str(images_dir), tile=False, conf=0.0)
    assert type(p.predictor.model.detector) is FasterRCNN
    assert results and all("attributes" not in r for r in results), results

    published = run_inference(tmp_path, checkpoint, str(images_dir), bucket="plain",
                              stated=Stated(tile=False, conf=0.0))
    assert "error" not in published, published
    assert len(_documents(tmp_path, "plain")) == 2
    for written in _documents(tmp_path, "plain").values():
        assert all(not a.get("attributes") for a in written), written


def test_a_document_scope_whose_attributes_were_never_read_sizes_no_model(tmp_path: Path):
    """A run config naming a subject but recording no attributes was never admitted, so its
    model's head sizes are unknown: the dims refuse by the admission's own sentence rather than
    sizing a detector with no heads. The same subject read through the admission sizes one."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.model_build import recorded_model_dims

    unread = {"model_source": dict(BUILT_DETECTOR),
              "data": {"num_channels": 3, "scope": {"subject": SUBJECT}}}
    with pytest.raises(ValueError, match="records no attributes"):
        recorded_model_dims(unread)

    images_dir = _frames(tmp_path / "ds", {"color": "red"})
    read = {"model_source": dict(BUILT_DETECTOR),
            "data": {"num_channels": 3, "scope": asdict(registry_scope(images_dir, SUBJECT))}}
    assert recorded_model_dims(read)["attributes"] == (COLOR, GRADE)


# --- an unassessed instance -------------------------------------------------


def test_an_instance_unassessed_for_one_attribute_trains_that_head_on_nothing_and_the_other(
    tmp_path: Path,
):
    """Every object assesses ``grade`` and none assesses ``color``: both frames are admitted, the
    color column reads unassessed, the color head's loss is masked to nothing and the grade head
    trains."""
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tests._producer_fixtures import run_over

    images_dir = _frames(tmp_path / "ds", {"grade": "high"})
    loader, data = run_over("detection", images_dir, subject=SUBJECT, stated={"num_channels": 3})
    assert len(loader) == 2
    image, target = loader[0]
    assert target["attributes"].tolist() == [[UNASSESSED, 2]]

    config = {"model_source": dict(BUILT_DETECTOR), "data": data}
    model = build_model(config, recorded_model_dims(config))
    model.train()
    losses = model([image], [target])
    color_head, grade_head = model.detector.attribute_heads
    (losses["attribute_loss_color"] + losses["attribute_loss_grade"]).backward()

    assert float(losses["attribute_loss_color"].detach()) == 0.0
    assert float(losses["attribute_loss_grade"].detach()) > 0.0
    assert all(p.grad is None or not p.grad.any() for p in color_head.parameters())
    assert any(p.grad is not None and p.grad.any() for p in grade_head.parameters())


@pytest.mark.parametrize("name", ["faster_rcnn", "fcos", "retinanet", "mask_rcnn"])
def test_each_builtin_detector_carries_one_head_per_attribute(name: str):
    """Around each built-in detector, the attribute module trains one masked loss per attribute
    beside the detector's own losses, writes ``attributes`` boxes by attributes in eval, and leaves
    the operating-point knobs at exactly one holder."""
    from tcip_mcp.pipelines.operating_point import detector_operating_point_holder
    from tests.bespoke_models import build_bespoke_detection

    torch.manual_seed(0)
    model = build_bespoke_detection(detector=name, min_size=64, max_size=64,
                                    attributes=(COLOR, GRADE))
    image = torch.rand(3, 64, 64)
    target = {"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]), "labels": torch.tensor([1]),
              "attributes": torch.tensor([[1, UNASSESSED]])}
    if name == "mask_rcnn":
        target["masks"] = torch.zeros(1, 64, 64, dtype=torch.uint8)
        target["masks"][0, 8:40, 8:40] = 1

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
    tmp_path: Path, monkeypatch,
):
    """The tiler's index holds the per-box values ``PER_BOX_KEYS`` names and no list of its own:
    ``attributes`` rides in step with each tile's boxes while the key is named there, and is
    absent once it is not."""
    from tcip_mcp.pipelines.data import datasets
    from tests._producer_fixtures import dataset_over

    images_dir = _frames(tmp_path / "ds", {"color": "red", "grade": "mid"})
    tiling = {"enabled": True, "tile_size": 32, "overlap": 0.0, "sliver_frac": 0.5}

    tiled = dataset_over("detection", images_dir, subject=SUBJECT, tiling=tiling)
    rows = [tiled[i][1] for i in range(len(tiled))]
    assert rows and all(r["attributes"].tolist() == [[0, 1]] * len(r["boxes"]) for r in rows)

    monkeypatch.setattr(datasets, "PER_BOX_KEYS",
                        tuple(k for k in datasets.PER_BOX_KEYS if k != "attributes"))
    unnamed = dataset_over("detection", images_dir, subject=SUBJECT, tiling=tiling)
    assert all("attributes" not in unnamed[i][1] for i in range(len(unnamed)))


# --- a tiled merge over attribute values ------------------------------------


WHOLE = cr.Attribute("whole", "categorical", ("cut", "whole"))
"""Whether a tile holds its blob whole, as :class:`~tests.bespoke_models.WholeBlobDetector` calls
it."""


@pytest.mark.parametrize("postprocess", ["nms", "nmm", "greedynmm"])
def test_two_tiles_calling_one_object_differently_merge_into_the_higher_scoring_call(
    tmp_path: Path, postprocess: str,
):
    """An object the second column's tile holds whole and the first column's tile cuts at its
    edge: the whole call covers more of its tile and scores higher, and the merge, over the one
    subject, keeps one box carrying the whole call's value."""
    pytest.importorskip("sahi")
    from PIL import Image

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import prepare_pass
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.tools.model_tools import register_model
    from tests._producer_fixtures import registry_over

    registry_over(tmp_path / "ds", cr.SubjectRegistry(
        subjects=(cr.Subject(name=SUBJECT, attributes=(WHOLE,)),)))
    scope = registry_scope(tmp_path / "ds" / "images", SUBJECT)
    config = {"model_source": {"builder": "tests.bespoke_models:build_whole_blob_detector",
                               "builder_kwargs": {}, "task": "detection"},
              "data": {"num_channels": 3, "scope": asdict(scope)}}
    ckpt = tmp_path / "model_best.pt"
    model = build_model(config, recorded_model_dims(config))
    torch.save({"model_state_dict": model.state_dict(), "config": config}, str(ckpt))
    assert "error" not in register_model(name="whole", checkpoint_path=str(ckpt), config={},
                                         project=tmp_path)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[20:60, 100:150, 0] = 255
    source = tmp_path / "frame.png"
    Image.fromarray(frame).save(source)

    p = prepare_pass(load_registered_checkpoint(str(ckpt), project=tmp_path),
                     Stated(tile=True, tile_size=128, overlap=0.25, postprocess=postprocess,
                            cross_tile_nms=0.3, conf=0.0), device="cpu", tile_batch_size=2)
    result = p.predictor.predict_sliced(str(source), execution=p.execution,
                                        tile_batch_size=p.tile_batch_size, require_masks=False)

    assert result["boxes"] == [[100.0, 20.0, 150.0, 60.0]], result
    assert result["attributes"] == [[WHOLE.values.index("whole")]], result


# --- per-attribute agreement ------------------------------------------------


class _CallsEveryFrameOnce(torch.nn.Module):
    """One call per frame at the box :func:`~tests._verified_checkpoint_fixtures.
    detection_images` draws, carrying ``ids`` as its attribute ids."""

    def __init__(self, ids: list[int]) -> None:
        super().__init__()
        self.ids = ids

    def forward(self, images, targets=None):
        if self.training:
            return {"loss": torch.zeros(())}
        return [{"boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0]]), "scores": torch.tensor([0.9]),
                 "labels": torch.tensor([1]), "attributes": torch.tensor([self.ids])}
                for _ in images]


def test_evaluate_reports_each_attributes_agreement_over_the_pairs_it_assesses(tmp_path: Path):
    """Two frames whose one object each assesses ``color`` and only the first assesses ``grade``:
    color agrees over two matched pairs and grade over the one its reference assesses."""
    from tcip_annotation.state import Annotation, BBox

    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.pipelines.training.evaluation import evaluate
    from tests._producer_fixtures import label_image, run_over

    images_dir = _frames(tmp_path / "ds", {"color": "blue", "grade": "high"})
    label_image(
        Path(images_dir) / "frame1.png",
        [Annotation(subject=SUBJECT, geometry=BBox(8, 8, 24, 24), attributes={"color": "blue"})],
        64, 48)
    dataset, data = run_over("detection", images_dir, subject=SUBJECT, stated={"num_channels": 3})
    images, targets = zip(*(dataset[i] for i in range(len(dataset))))
    dims = model_dims(ClassScope.of(data), {"num_channels": 3})

    metrics = evaluate(_CallsEveryFrameOnce([1, 2]), [(list(images), list(targets))], "cpu",
                       "detection", dims=dims, conf_threshold=0.0)

    agreement = metrics["attribute_agreement"]
    assert set(agreement) == {"color", "grade"}
    assert agreement["color"]["pairs"] == 2
    assert agreement["grade"]["pairs"] == 1
    assert agreement["color"]["accuracy"] == agreement["grade"]["accuracy"] == 1.0

"""A crowd region reaches every reader of ground truth as a crowd region, never as one object.

Each fixture is written through the platform's own label writer and read back through the
reader under test: the loaders and their crop, the trainer's and the validation loss's hand-off
to the heads, the model contract's overfit probe, the completion mark's digest, the editor's
proposal pairing, and the assessment's cap, size, spacing and merge threshold. Where a reader
forms a number from objects, the same records with and without crowd regions must give the same
number.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key  # noqa: E402
from tests._producer_fixtures import label_image, labeled_frame  # noqa: E402
from tests.bespoke_models import BrightRegionDetector  # noqa: E402

SUBJECT = "bur"
IMG = 100
OBJECT = BBox(5.0, 5.0, 25.0, 25.0)
CROWD = BBox(55.0, 55.0, 95.0, 95.0)


class RecordingDetector(BrightRegionDetector):
    """A detector that records the crowd flags of every target a training step hands it."""

    handed: list[list[int]] = []

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            RecordingDetector.handed.append(
                [int(v) for t in targets for v in t.get("iscrowd", torch.zeros(len(t["boxes"])))])
        return super().forward(images, targets)


def build_recording_detector(*, in_chans: int = 3, num_classes: int = 1) -> RecordingDetector:
    return RecordingDetector(in_chans=in_chans)


def _read_back(root: Path, stem: str, anns: list) -> list:
    """``anns`` written as the label document of the image ``stem`` under ``root`` and read back
    through the platform's own reader."""
    key = label_key(root, UNDATED_BUCKET, stem)
    json_io.write_label_document(key, anns, IMG, IMG)
    return json_io.read_label_document(key).annotations


def _labeled(root: Path, stems=("c0", "c1"), crowd=True) -> Path:
    """Frames each holding one bright bur and, when ``crowd``, one region of unseparated burs;
    their images directory."""
    images = root / "images" / UNDATED_BUCKET
    for stem in stems:
        anns = [Annotation(subject=SUBJECT, geometry=OBJECT)]
        if crowd:
            anns.append(Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True))
        labeled_frame(images / f"{stem}.png", anns, IMG, IMG, (40, 40, 40))
    return images


def _detection_loader(root: Path, task: str = "detection"):
    from tests._producer_fixtures import dataset_over

    return dataset_over(task, _labeled(root), subject=SUBJECT)


def test_crowd_of_refuses_a_target_stating_no_flag_and_reads_the_producers_own():
    """Whether a row is a crowd region is its producer's to state: a target the platform's own
    producer built answers its flags, one stating none refuses rather than reading as no crowd."""
    from tcip_mcp.pipelines.data.datasets import crowd_of
    from tcip_mcp.pipelines.data.label_queries import json_det_targets
    from tcip_mcp.pipelines.data.selection import ClassScope

    target = json_det_targets([Annotation(subject=SUBJECT, geometry=OBJECT),
                               Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)],
                              ClassScope(subject=SUBJECT, attributes=()))
    assert list(crowd_of(target)) == [False, True]
    with pytest.raises(ValueError, match="states no 'iscrowd'"):
        crowd_of({"boxes": target["boxes"], "labels": target["labels"]})


def test_the_instance_loader_carries_each_polygons_crowd_flag(tmp_path: Path):
    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / "p0.png")
    label_image(images / "p0.png", [
        Annotation(subject=SUBJECT, geometry=Polygon([[(5.0, 5.0), (25.0, 5.0), (25.0, 25.0)]])),
        Annotation(subject=SUBJECT, geometry=Polygon([[(55.0, 55.0), (95.0, 55.0), (95.0, 95.0)]]),
                   iscrowd=True)], IMG, IMG)
    from tests._producer_fixtures import dataset_over

    _image, target = dataset_over("instance_seg", images, subject=SUBJECT)[0]

    assert target["iscrowd"].tolist() == [0, 1]
    assert len(target["masks"]) == len(target["boxes"]) == 2


def test_the_instance_loader_takes_each_polygons_class_from_the_one_target_decision(
        tmp_path: Path):
    """Another subject's polygon beside the run's own is no instance of the run: the loader's
    subject decision is ``attribute_ids``', the one every target reader makes."""
    from tests._producer_fixtures import dataset_over

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / "p0.png")
    label_image(images / "p0.png", [
        Annotation(subject=SUBJECT, geometry=Polygon([[(5.0, 5.0), (25.0, 5.0), (25.0, 25.0)]])),
        Annotation(subject="leaf", geometry=Polygon([[(55.0, 55.0), (95.0, 55.0), (95.0, 95.0)]])),
    ], IMG, IMG)

    _image, target = dataset_over("instance_seg", images, subject=SUBJECT)[0]

    assert target["labels"].tolist() == [1]
    assert target["boxes"].tolist() == [[5.0, 5.0, 25.0, 25.0]]


def test_a_crop_keeps_each_rows_crowd_flag_with_its_box(tmp_path: Path):
    from tcip_mcp.pipelines.data.augmentations import RandomResizedCrop
    from tcip_mcp.pipelines.data.datasets import target_tensors

    loader = _detection_loader(tmp_path)
    crop = RandomResizedCrop(size=(50, 50), min_scale=0.5, max_scale=0.5)

    def origin(seed: int) -> tuple[int, int]:
        # The crop's own draws: a scale, then the origin of its 50 x 50 window.
        random.seed(seed)
        random.uniform(0.5, 0.5)
        return random.randint(0, 50), random.randint(0, 50)

    # A window past the object (it ends at 25) and inside the crowd region (it starts at 55).
    seed = next(s for s in range(1000) if min(origin(s)) >= 30)
    target = target_tensors(loader.det_targets(loader.document(loader.stems[0])))
    assert target["iscrowd"].tolist() == [0, 1]
    random.seed(seed)
    _img, cropped = crop(Image.new("RGB", (IMG, IMG)), target)

    assert len(cropped["boxes"]) == 1
    assert cropped["iscrowd"].tolist() == [1] and cropped["labels"].tolist() == [1]


def test_the_trainer_and_the_validation_loss_hand_the_heads_objects_only(tmp_path: Path,
                                                                         monkeypatch):
    """Both hand-offs to a model's training forward, the training step and the validation loss,
    withhold every crowd row the loader keeps."""
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.pipelines.training.generic_trainer import (
        effective_data_geometry, run_loaders, train,
    )
    from tests._chain_fixtures import training_config
    from tests.tiny_trainer_fixtures import capture_model, trainer_run

    loader_ds = _detection_loader(tmp_path / "ds")
    assert loader_ds[0][1]["iscrowd"].tolist() == [0, 1]  # the loader keeps every row
    built: list = []
    capture_model(monkeypatch, built)
    data = effective_data_geometry("detection", DataSpec.model_validate(
        {"num_channels": 3, "scope": {"subject": SUBJECT, "attributes": []}}), loader_ds)
    config = training_config(
        {"builder": f"{Path(__file__).stem}:build_recording_detector", "source_files": [__file__],
         "task": "detection"}, data.record())
    run = trainer_run(config, tmp_path / "run", project=tmp_path, has_val_loader=True,
                      id="crowd-run")
    train_loader, val_loader = run_loaders(run, loader_ds, loader_ds)
    completed = train(run, train_loader, val_loader=val_loader)

    assert completed.status == "completed", completed.status
    (model,) = built
    handed = type(model).handed  # the run's own copy of the class records what it was handed
    assert len(handed) >= 2  # one training step and one validation-loss pass
    assert all(flags == [0, 0] for flags in handed), handed


def test_the_overfit_probe_drives_the_model_with_objects_only(tmp_path: Path):
    from tcip_mcp.pipelines.model_contract import overfit_check
    from tcip_mcp.pipelines.training.collation import task_collate

    loader_ds = _detection_loader(tmp_path)
    batch = task_collate("detection")([loader_ds[0], loader_ds[1]])
    RecordingDetector.handed.clear()

    overfit_check(RecordingDetector(), "detection", sample_batch=batch, steps=2)

    assert RecordingDetector.handed and all(f == [0, 0] for f in RecordingDetector.handed), (
        RecordingDetector.handed)


def test_the_completion_digest_changes_with_the_crowd_flag(tmp_path: Path):
    digests = []
    for crowd in (False, True):
        read = _read_back(tmp_path, "a", [Annotation(subject=SUBJECT, geometry=CROWD,
                                                     iscrowd=crowd)])
        digests.append(json_io.subject_digest(read, SUBJECT))
    assert digests[0] != digests[1]


def _records(tmp_path: Path, crowd: bool, n_crowd: int = 120) -> list[dict]:
    """Two images' evaluation records, each one bur and, when ``crowd``, many large crowd
    regions, written and read back through the label document."""
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    records = []
    for i, offset in enumerate((0.0, 30.0)):
        anns = [Annotation(subject=SUBJECT, geometry=BBox(5 + offset, 5, 15 + offset, 15))]
        if crowd:
            anns += [Annotation(subject=SUBJECT, geometry=BBox(40, 40, 99, 99), iscrowd=True)
                     for _ in range(n_crowd)]
        records.append(records_from_annotation(_read_back(tmp_path, f"r{i}_{crowd}", anns), [],
                                               width=IMG, height=IMG, cap=None))
    return records


def test_the_object_density_counts_objects_not_crowd_regions(tmp_path: Path):
    """The density derives from each frame's object count, which a crowd region beside the
    objects leaves unchanged."""
    from tcip_mcp.pipelines.derivations import derive_object_density
    from tests._producer_fixtures import dataset_over, seed_labeled_images

    regions = {}
    for crowd in (True, False):
        anns = [Annotation(subject=SUBJECT, geometry=OBJECT)] + (
            [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)] * 3 if crowd else [])
        images_dir = seed_labeled_images(tmp_path / f"ds_{crowd}" / "images" / UNDATED_BUCKET,
                                         anns, n=1, width=IMG, height=IMG)
        regions[crowd] = dataset_over("detection", str(images_dir), subject=SUBJECT,
                                      stated={"num_channels": 3}).regions
    assert [(len(r.boxes), r.area) for c in (True, False) for r in regions[c]] == [
        (1, float(IMG * IMG))] * 2
    assert derive_object_density(regions[True]) == derive_object_density(regions[False])


def test_the_object_size_and_spacing_ignore_crowd_regions(tmp_path: Path):
    from tests import _trait_fixtures as fx

    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size, resolve_match_criterion

    trait = fx.propose_and_confirm(tmp_path, fx.COUNT_SPEC).entry
    with_crowd, without = _records(tmp_path, crowd=True, n_crowd=3), _records(tmp_path, crowd=False)

    assert gt_class_avg_size(with_crowd) == gt_class_avg_size(without) == 10.0
    assert (resolve_match_criterion(trait, with_crowd)["tolerance"]
            == resolve_match_criterion(trait, without)["tolerance"])


def test_a_crowd_prediction_is_no_detection_at_the_scoring_side(tmp_path: Path):
    from tcip_mcp.pipelines.training.evaluation import detection_metrics, records_from_annotation

    preds = _read_back(tmp_path, "p", [
        Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, iscrowd=True)])
    record = records_from_annotation([], preds, width=IMG, height=IMG, cap=None)
    assert record["dt"] == []
    m = detection_metrics([record], trait=None, conf_threshold=0.25, iou_threshold=0.5,
                          by_mask=False)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 0)


def test_one_selector_answers_both_halves_of_the_crowd_split(tmp_path: Path):
    """The matcher's crowd regions are the selector's crowd half, the objects its other half."""
    from tcip_annotation.state import instances

    gt = _read_back(tmp_path, "g", [
        Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True),
        Annotation(subject=SUBJECT, geometry=OBJECT)])
    assert [a.geometry for a in instances(gt, crowd=True)] == [CROWD]
    assert [a.geometry for a in instances(gt)] == [OBJECT]


def test_the_editors_pairing_never_pairs_with_or_as_a_crowd_region(tmp_path: Path):
    """A predicted crowd region is no proposal to pair; a ground-truth crowd region is nothing to
    confirm; a proposal inside one pairs with nothing; every index still addresses the caller's
    own lists."""
    from tcip_annotation.matching import pair_proposals

    inside = BBox(60.0, 60.0, 80.0, 80.0)
    gt = _read_back(tmp_path, "g", [
        Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True),
        Annotation(subject=SUBJECT, geometry=OBJECT)])
    preds = _read_back(tmp_path, "p", [
        Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, iscrowd=True),
        Annotation(subject=SUBJECT, geometry=inside, score=0.9),
        Annotation(subject=SUBJECT, geometry=OBJECT, score=0.9)])

    assert pair_proposals(gt, preds, {"kind": "iou", "iou_threshold": 0.5}).pairs == [(1, 2)]
    assert pair_proposals([], preds[:1], {"kind": "iou", "iou_threshold": 0.5}).pairs == []


CENTER = {"kind": "center_match", "tolerance": 5.0}
"""A center match at a 5 px tolerance."""


def _center_records(tmp_path: Path) -> list[dict]:
    """Nine images holding a crowd region alone with a detection inside it, and one image whose
    one object nothing detected, each written and read back through the label document."""
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    name_id = {SUBJECT: 1}
    records = []
    for i in range(9):
        gt = _read_back(tmp_path, f"g{i}",
                        [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)])
        preds = _read_back(tmp_path, f"p{i}",
                           [Annotation(subject=SUBJECT, geometry=BBox(60.0, 60.0, 80.0, 80.0),
                                       score=0.9)])
        records.append(records_from_annotation(gt, preds, width=IMG, height=IMG, cap=None,
                                               name_id=name_id))
    missed = _read_back(tmp_path, "missed", [Annotation(subject=SUBJECT, geometry=OBJECT)])
    records.append(records_from_annotation(missed, [], width=IMG, height=IMG, cap=None,
                                           name_id=name_id))
    return records


def test_an_ignored_detection_is_no_present_image_observation(tmp_path: Path):
    from tcip_mcp.pipelines.training.evaluation import _count_stats_at_conf

    stats = _count_stats_at_conf(_center_records(tmp_path), criterion=CENTER, conf=0.5,
                                 class_id=None)
    assert (stats["tp"], stats["fp"], stats["fn"]) == (0, 0, 1)
    assert stats["n_present"] == 1
    assert stats["count_bias_mean_present"] == -1.0


def test_the_governing_center_count_is_the_count_statistics_own(tmp_path: Path):
    from tcip_mcp.pipelines.training.evaluation import _count_stats_at_conf, governing_counts

    records = _center_records(tmp_path)
    # A detection whose center sits 6.4 px from the object's: outside the tolerance, a miss.
    records.append(records[-1] | {"dt": [{"category_id": 1, "bbox": [12.0, 12.0, 15.0, 15.0],
                                         "score": 0.9}]})
    stats = _count_stats_at_conf(records, criterion=CENTER, conf=0.5, class_id=None)
    counts = governing_counts(records, CENTER, conf_threshold=0.5)
    assert {k: counts[k] for k in ("tp", "fp", "fn")} == {k: stats[k] for k in ("tp", "fp", "fn")}
    assert counts["recall"] == round(stats["recall"], 6)


def test_the_worst_predictions_triage_counts_objects_not_crowd_regions(tmp_path: Path):
    """Both sides of the triage's count are objects: a crowd region beside the one matching
    detection is no surplus, and a reference holding crowd regions alone is no missed image."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.vision_tools import get_worst_predictions
    from tests._chain_fixtures import published

    images = tmp_path / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    for stem in ("a", "b"):
        Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / f"{stem}.png")
    bucket = published(tmp_path, "preds", [
        {"image": str(images / "a.png"), "width": IMG, "height": IMG,
         "boxes": [[OBJECT.x1, OBJECT.y1, OBJECT.x2, OBJECT.y2]], "scores": [0.9],
         "labels": [1], "count": 1, "cap": 2}], scope={"subject": SUBJECT})
    # Crowd regions beside the detection, as an edit in place would leave them.
    json_io.write_label_document(bucket.document_key("a"), [
        Annotation(subject=SUBJECT, geometry=OBJECT, score=0.9),
        *[Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, iscrowd=True)] * 5], IMG, IMG)
    label_image(images / "a.png", [Annotation(subject=SUBJECT, geometry=OBJECT)], IMG, IMG)
    label_image(images / "b.png", [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)],
                IMG, IMG)

    result = get_worst_predictions(bucket)
    assert result["worst_images"] == [{"stem": "a", "error_score": 0.1}]
    assert result["not_predicted"] == ["b"]


def test_the_derived_spacing_and_cross_tile_nms_ignore_crowd_regions(tmp_path: Path):
    """An assessment derives its localization spacing and its cross-tile merge threshold from the
    objects of the calibration reference: stacked crowd regions beside them move neither."""
    from tests import _trait_fixtures as fx

    from tcip_mcp.pipelines.derivations import (
        LOCALIZATION_TOLERANCE_DERIVATION, derive_cross_tile_nms,
    )
    from tcip_mcp.pipelines.training.evaluation import (
        gt_objects, localization_frac, records_from_annotation,
    )

    def records(crowd: bool) -> list[dict]:
        out = []
        for i in range(6):
            anns = [Annotation(subject=SUBJECT, geometry=BBox(5 + i, 5, 25 + i, 25)),
                    Annotation(subject=SUBJECT, geometry=BBox(20 + i, 5, 40 + i, 25))]
            if crowd:
                anns += [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)] * 3
            out.append(records_from_annotation(_read_back(tmp_path, f"s{i}_{crowd}", anns), [],
                                               width=IMG, height=IMG, cap=None))
        return out

    boxes = {crowd: [[a["bbox"] for a in gt_objects(r)] for r in records(crowd)]
             for crowd in (True, False)}
    with_crowd, without = (localization_frac(fx.COUNT_SPEC, boxes[c]) for c in (True, False))
    assert without[1] == LOCALIZATION_TOLERANCE_DERIVATION, without
    assert with_crowd == without
    from tests._verified_checkpoint_fixtures import objects_over

    merges = [derive_cross_tile_nms([
        objects_over([[x, y, x + w, y + h] for x, y, w, h in image], IMG * IMG)
        for image in boxes[c]]) for c in (True, False)]
    assert merges[1] is not None
    assert merges[0] == merges[1]

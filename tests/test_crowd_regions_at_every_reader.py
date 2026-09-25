"""A crowd region reaches every reader of ground truth as a crowd region, never as one object.

Each fixture is written through the platform's own label writer and read back through the
reader under test: the loaders and their crop, the trainer's and the validation loss's hand-off
to the heads, the model contract's overfit probe, the completeness digest, review
materialization, and the operating point's cap, size and spacing. Where a reader forms a number
from objects, the same records with and without crowd regions must give the same number.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
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


def build_recording_detector(*, in_chans: int = 3) -> RecordingDetector:
    return RecordingDetector(in_chans=in_chans)


def _labeled(root: Path, stems=("c0", "c1"), crowd=True) -> tuple[Path, Path]:
    """Frames each holding one bur and, when ``crowd``, one region of unseparated burs."""
    images, labels = root / "images", root / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / f"{stem}.png")
        anns = [Annotation(subject=SUBJECT, geometry=OBJECT)]
        if crowd:
            anns.append(Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True))
        json_io.write_annotations(labels / f"{stem}.json", anns, IMG, IMG)
    return images, labels


def _detection_loader(root: Path, task: str = "detection"):
    from tests._producer_fixtures import dataset_over

    images, labels = _labeled(root)
    return dataset_over(task, images, labels, subject=SUBJECT)


def test_the_instance_loader_carries_each_polygons_crowd_flag(tmp_path: Path):
    images, labels = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / "p0.png")
    json_io.write_annotations(labels / "p0.json", [
        Annotation(subject=SUBJECT, geometry=Polygon([[(5.0, 5.0), (25.0, 5.0), (25.0, 25.0)]])),
        Annotation(subject=SUBJECT, geometry=Polygon([[(55.0, 55.0), (95.0, 55.0), (95.0, 95.0)]]),
                   iscrowd=True)], IMG, IMG)
    from tests._producer_fixtures import dataset_over

    _image, target = dataset_over("instance_seg", images, labels, subject=SUBJECT)[0]

    assert target["iscrowd"].tolist() == [0, 1]
    assert len(target["masks"]) == len(target["boxes"]) == 2


def test_the_instance_loader_takes_each_polygons_class_from_the_one_target_decision(
        tmp_path: Path):
    """Another subject's polygon beside the run's own is no instance of the run: the loader's
    class decision is ``target_class_id``'s, the one every target reader makes."""
    from tests._producer_fixtures import dataset_over

    images, labels = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (IMG, IMG), (40, 40, 40)).save(images / "p0.png")
    json_io.write_annotations(labels / "p0.json", [
        Annotation(subject=SUBJECT, geometry=Polygon([[(5.0, 5.0), (25.0, 5.0), (25.0, 25.0)]])),
        Annotation(subject="leaf", geometry=Polygon([[(55.0, 55.0), (95.0, 55.0), (95.0, 95.0)]])),
    ], IMG, IMG)

    _image, target = dataset_over("instance_seg", images, labels, subject=SUBJECT)[0]

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
    target = target_tensors(loader.det_targets(loader.stems[0]))
    assert target["iscrowd"].tolist() == [0, 1]
    random.seed(seed)
    _img, cropped = crop(Image.new("RGB", (IMG, IMG)), target)

    assert len(cropped["boxes"]) == 1
    assert cropped["iscrowd"].tolist() == [1] and cropped["labels"].tolist() == [1]


def _train_config(root: Path) -> dict:
    return {
        "model_source": {"builder": f"{__name__}:build_recording_detector", "builder_kwargs": {},
                         "task": "detection", "in_chans": 3},
        "data": {}, "batch_size": 2, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu", "checkpoint_every_n_epochs": 1,
        "early_stopping": {"enabled": False},
        "optimizer": {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0},
        "scheduler": {"type": "cosine"}, "gradient_accumulation_steps": 1,
    }


def test_the_trainer_and_the_validation_loss_hand_the_heads_objects_only(tmp_path: Path):
    """Both hand-offs to a model's training forward, the training step and the validation loss,
    withhold every crowd row the loader keeps."""
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.run_registry import create_run

    loader_ds = _detection_loader(tmp_path / "ds")
    assert loader_ds[0][1]["iscrowd"].tolist() == [0, 1]  # the loader keeps every row
    collate = task_collate("detection")
    RecordingDetector.handed.clear()
    run = create_run(_train_config(tmp_path), str(tmp_path / "run"), id="crowd-run")
    completed = train(run, DataLoader(loader_ds, batch_size=2, collate_fn=collate),
                      val_loader=DataLoader(loader_ds, batch_size=2, collate_fn=collate),
                      task="detection")

    assert completed.status == "completed", completed.status
    assert len(RecordingDetector.handed) >= 2  # one training step and one validation-loss pass
    assert all(flags == [0, 0] for flags in RecordingDetector.handed), RecordingDetector.handed


def test_the_overfit_probe_drives_the_model_with_objects_only(tmp_path: Path):
    from tcip_mcp.pipelines.model_contract import overfit_check
    from tcip_mcp.pipelines.training.collation import task_collate

    loader_ds = _detection_loader(tmp_path)
    batch = task_collate("detection")([loader_ds[0], loader_ds[1]])
    RecordingDetector.handed.clear()

    overfit_check(RecordingDetector(), "detection", sample_batch=batch, steps=2)

    assert RecordingDetector.handed and all(f == [0, 0] for f in RecordingDetector.handed), (
        RecordingDetector.handed)


def test_the_completeness_digest_changes_with_the_crowd_flag(tmp_path: Path):
    from tcip_mcp.pipelines.reference_grid import reference_cells
    from tcip_mcp.pipelines.region_completeness import cell_annotation_digest

    (cell,) = reference_cells(IMG, IMG, IMG, clamp=True)
    path = tmp_path / "a.json"
    digests = []
    for crowd in (False, True):
        json_io.write_annotations(path, [Annotation(subject=SUBJECT, geometry=CROWD,
                                                    iscrowd=crowd)], IMG, IMG)
        digests.append(cell_annotation_digest(json_io.read_annotations(path), SUBJECT, cell))
    assert digests[0] != digests[1]


def _records(tmp_path: Path, crowd: bool, n_crowd: int = 120) -> list[dict]:
    """Two images' evaluation records, each one bur and, when ``crowd``, many large crowd
    regions, written and read back through the label document."""
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    records = []
    for i, offset in enumerate((0.0, 30.0)):
        path = tmp_path / f"r{i}_{crowd}.json"
        anns = [Annotation(subject=SUBJECT, geometry=BBox(5 + offset, 5, 15 + offset, 15))]
        if crowd:
            anns += [Annotation(subject=SUBJECT, geometry=BBox(40, 40, 99, 99), iscrowd=True)
                     for _ in range(n_crowd)]
        json_io.write_annotations(path, anns, IMG, IMG)
        records.append(records_from_annotation(json_io.read_annotations(path), [],
                                               width=IMG, height=IMG)[1])
    return records


def test_the_density_cap_counts_objects_not_crowd_regions(tmp_path: Path):
    from tcip_mcp.pipelines.operating_point import _max_dets_from_density

    assert _max_dets_from_density(_records(tmp_path, crowd=True)) == _max_dets_from_density(
        _records(tmp_path, crowd=False)) == 100


def test_the_object_size_and_spacing_ignore_crowd_regions(tmp_path: Path):
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.pipelines.training.evaluation import gt_class_avg_size, resolve_match_criterion

    fx.write_spec(tmp_path, fx.COUNT_SPEC)
    with_crowd, without = _records(tmp_path, crowd=True, n_crowd=3), _records(tmp_path, crowd=False)

    assert gt_class_avg_size(with_crowd) == gt_class_avg_size(without) == 10.0
    assert (resolve_match_criterion(fx.COUNT_TRAIT, with_crowd)["tolerance"]
            == resolve_match_criterion(fx.COUNT_TRAIT, without)["tolerance"])


def _read_back(path: Path, anns: list) -> list:
    json_io.write_annotations(path, anns, IMG, IMG)
    return json_io.read_annotations(path)


def test_a_crowd_prediction_is_no_detection_at_the_scoring_side(tmp_path: Path):
    from tcip_mcp.pipelines.training.evaluation import coco_detection_metrics, records_from_annotation

    preds = _read_back(tmp_path / "p.json", [
        Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, iscrowd=True)])
    _, record = records_from_annotation([], preds, width=IMG, height=IMG)
    assert record["dt"] == []
    m = coco_detection_metrics([record])
    assert (m["tp"], m["fp"], m["fn"], m["n_pred"]) == (0, 0, 0, 0)


def test_one_selector_answers_both_halves_of_the_crowd_split(tmp_path: Path):
    """The matcher's crowd regions are the selector's crowd half, the objects its other half."""
    from tcip_annotation.state import instances

    gt = _read_back(tmp_path / "g.json", [
        Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True),
        Annotation(subject=SUBJECT, geometry=OBJECT)])
    assert [a.geometry for a in instances(gt, crowd=True)] == [CROWD]
    assert [a.geometry for a in instances(gt)] == [OBJECT]


def test_reviews_matching_gives_a_crowd_region_cocos_ignore_semantics(tmp_path: Path):
    """A predicted crowd region is no detection; a ground-truth crowd region is no miss; a
    prediction left unmatched inside one is neither true nor false; every index still addresses
    the caller's own lists."""
    from tcip_annotation.matching import compute_matches

    inside = BBox(60.0, 60.0, 80.0, 80.0)
    gt = _read_back(tmp_path / "g.json", [
        Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True),
        Annotation(subject=SUBJECT, geometry=OBJECT)])
    preds = _read_back(tmp_path / "p.json", [
        Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, iscrowd=True),
        Annotation(subject=SUBJECT, geometry=inside, score=0.9),
        Annotation(subject=SUBJECT, geometry=OBJECT, score=0.9)])

    matches = compute_matches(gt, preds)
    assert [(m["gt_idx"], m["pred_idx"]) for m in matches["tp"]] == [(1, 2)]
    assert matches["fp"] == [] and matches["fn"] == []
    assert compute_matches([], preds[:1]) == {"tp": [], "fp": [], "fn": []}


def test_the_classification_projections_keep_the_crowd_flag(tmp_path: Path):
    from tcip_annotation.matching import compute_classified_trait_matches

    gt = _read_back(tmp_path / "g.json", [
        Annotation(subject=SUBJECT, geometry=CROWD, attributes={"stage": "open"}, iscrowd=True)])
    preds = _read_back(tmp_path / "p.json", [
        Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, attributes={"stage": "open"},
                   iscrowd=True)])

    matches = compute_classified_trait_matches(
        gt, preds, subject=SUBJECT, attribute="stage", vocabulary={"open", "closed"})
    assert matches["tp"] == [] and matches["fp"] == [] and matches["fn"] == []


def _center_records(tmp_path: Path) -> list[dict]:
    """Nine images holding a crowd region alone with a detection inside it, and one image whose
    one object nothing detected, each written and read back through the label document."""
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    name_id = {SUBJECT: 1}
    records = []
    for i in range(9):
        gt = _read_back(tmp_path / f"g{i}.json",
                        [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)])
        preds = _read_back(tmp_path / f"p{i}.json",
                           [Annotation(subject=SUBJECT, geometry=BBox(60.0, 60.0, 80.0, 80.0),
                                       score=0.9)])
        records.append(records_from_annotation(gt, preds, width=IMG, height=IMG,
                                               name_id=name_id)[1])
    missed = _read_back(tmp_path / "missed.json", [Annotation(subject=SUBJECT, geometry=OBJECT)])
    records.append(records_from_annotation(missed, [], width=IMG, height=IMG, name_id=name_id)[1])
    return records


def test_an_ignored_detection_is_no_present_image_observation(tmp_path: Path):
    from tcip_mcp.pipelines.training.evaluation import _count_stats_at_conf

    stats = _count_stats_at_conf(_center_records(tmp_path), tolerance=5.0, conf=0.5, class_id=None)
    assert (stats["tp"], stats["fp"], stats["fn"]) == (0, 0, 1)
    assert stats["n_present"] == 1
    assert stats["count_bias_mean_present"] == -1.0


def test_the_governing_center_count_is_the_count_statistics_own(tmp_path: Path):
    from tcip_mcp.pipelines.training.evaluation import _count_stats_at_conf, governing_counts

    records = _center_records(tmp_path)
    # A detection whose center sits 6.4 px from the object's: outside the tolerance, a miss.
    records.append(records[-1] | {"dt": [{"category_id": 1, "bbox": [12.0, 12.0, 15.0, 15.0],
                                         "score": 0.9}]})
    stats = _count_stats_at_conf(records, tolerance=5.0, conf=0.5, class_id=None)
    counts = governing_counts(records, {"kind": "center_match", "tolerance": 5.0},
                              conf_threshold=0.5)
    assert {k: counts[k] for k in ("tp", "fp", "fn")} == {k: stats[k] for k in ("tp", "fp", "fn")}
    assert counts["recall"] == round(stats["recall"], 6)


def test_the_worst_predictions_triage_counts_objects_not_crowd_regions(tmp_path: Path):
    """Both sides of the triage's count are objects: a crowd region beside the one matching
    detection is no surplus, and a reference holding crowd regions alone is no missed image."""
    from tcip_mcp.tools.vision_tools import get_worst_predictions

    preds, labels = tmp_path / "preds", tmp_path / "labels"
    json_io.write_annotations(preds / "a.json", [
        Annotation(subject=SUBJECT, geometry=OBJECT, score=0.9),
        *[Annotation(subject=SUBJECT, geometry=CROWD, score=0.9, iscrowd=True)] * 5], IMG, IMG)
    json_io.write_annotations(labels / "a.json", [Annotation(subject=SUBJECT, geometry=OBJECT)],
                              IMG, IMG)
    json_io.write_annotations(labels / "b.json",
                              [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)], IMG, IMG)

    result = get_worst_predictions(str(preds), str(labels))
    assert result["worst_images"] == [{"stem": "a", "error_score": 0.1}]


def test_the_resolved_spacing_and_cross_tile_nms_ignore_crowd_regions(tmp_path: Path):
    """The operating point derives its localization spacing and its cross-tile NMS from the
    objects of the calibration reference: stacked crowd regions beside them move neither."""
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.pipelines.operating_point import resolve_operating_point
    from tcip_mcp.pipelines.training.evaluation import records_from_annotation

    fx.write_spec(tmp_path, fx.COUNT_SPEC)

    def records(crowd: bool) -> list[dict]:
        out = []
        for i in range(6):
            path = tmp_path / f"s{i}_{crowd}.json"
            anns = [Annotation(subject=SUBJECT, geometry=BBox(5 + i, 5, 25 + i, 25)),
                    Annotation(subject=SUBJECT, geometry=BBox(20 + i, 5, 40 + i, 25))]
            if crowd:
                anns += [Annotation(subject=SUBJECT, geometry=CROWD, iscrowd=True)] * 3
            json_io.write_annotations(path, anns, IMG, IMG)
            out.append(records_from_annotation(json_io.read_annotations(path), [],
                                               width=IMG, height=IMG)[1])
        return out

    resolved = [resolve_operating_point(fx.COUNT_TRAIT, dataset_hash="h", tiled=True,
                                        calibration_records=records(crowd))
                for crowd in (True, False)]
    for name in ("localization_tolerance_frac", "cross_tile_nms"):
        with_crowd, without = (bundle.get(name) for bundle in resolved)
        assert without.source == "derived", (name, without)
        assert with_crowd.value == without.value, name

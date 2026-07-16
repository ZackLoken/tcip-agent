"""W1 — evaluation metrics + composite selection objective.

Unit tests for the pycocotools metrics engine, the ported composite objective,
the in-house scalar metrics, and ``_selection_value``; plus light integration
tests that exercise the detection/classification ``_validate`` path end-to-end.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # evaluation.py imports torch at module load
pytest.importorskip("pycocotools")

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    DEFAULT_SCORE_WEIGHTS,
    build_coco_image_record,
    classification_metrics,
    coco_detection_metrics,
    compute_composite_objective,
    effective_iou_type,
    ordinal_metrics,
    regression_metrics,
    run_test_evaluation,
)
from tcip_mcp.pipelines.training.generic_trainer import _selection_value  # noqa: E402


# --------------------------------------------------------------------------
# Composite objective
# --------------------------------------------------------------------------

def test_composite_objective_matches_reference():
    assert compute_composite_objective(-1.0, 0.9, 0.9) == 1e6        # val_loss <= 0
    assert compute_composite_objective(2.0, 0.0, 0.0) == 1e6         # degenerate prune
    expected = 0.45 * 2.0 + 0.35 * 0.5 * 10 + 0.20 * 0.6 * 10        # 3.85
    assert compute_composite_objective(2.0, 0.5, 0.4) == pytest.approx(expected, abs=1e-9)
    # NaN coerces: val_loss -> inf branch, f1/map50 -> 0 -> degenerate sentinel.
    assert compute_composite_objective(float("nan"), float("nan"), float("nan")) == 1e6
    assert DEFAULT_SCORE_WEIGHTS == {"loss": 0.45, "f1": 0.35, "map50": 0.20}


# --------------------------------------------------------------------------
# pycocotools detection metrics
# --------------------------------------------------------------------------

def _rec(gt, dt, w=100, h=100):
    return build_coco_image_record(w, h, gt, dt)


def test_coco_map50_perfect():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    m = coco_detection_metrics([_rec(gt, dt)])
    assert m["map50"] == pytest.approx(1.0)
    assert m["map"] == pytest.approx(1.0)


def test_coco_operating_point_tp_fp_fn():
    gt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
    ]
    dt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},   # FP
    ]
    m = coco_detection_metrics([_rec(gt, dt)])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["map50"] > 0.0


def test_coco_conf_threshold_filters():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [
        {"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},   # TP
        {"category_id": 1, "bbox": [80, 80, 10, 10], "score": 0.7},   # FP, below 0.8
    ]
    m = coco_detection_metrics([_rec(gt, dt)], conf_threshold=0.8)
    assert m["tp"] == 1
    assert m["fp"] == 0


def test_coco_no_stdout(capsys):
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    coco_detection_metrics([_rec(gt, dt)])
    captured = capsys.readouterr()
    assert captured.out == ""  # pycocotools prints must be redirected (MCP stdio safety)


def test_coco_segm_path():
    poly = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0, "segmentation": [poly]}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9, "segmentation": [poly]}]
    m = coco_detection_metrics([_rec(gt, dt)], iou_type="segm")
    assert 0.0 <= m["map50"] <= 1.0
    assert m["map50"] == pytest.approx(1.0)


def test_coco_empty():
    gt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]
    dt = [{"category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    # (a) empty preds + non-empty GT -> guard the real loadRes([]) IndexError.
    m = coco_detection_metrics([_rec(gt, [])])
    assert m["fn"] == 1 and m["recall"] == 0.0 and m["map50"] == 0.0
    # (b) non-empty preds + empty GT -> guard the COCOeval stats == -1 sentinel.
    m = coco_detection_metrics([_rec([], dt)])
    assert m["map50"] == 0.0 and m["map50"] >= 0.0
    # (c) both empty.
    m = coco_detection_metrics([_rec([], [])])
    assert m["tp"] == 0 and m["fp"] == 0 and m["map50"] == 0.0


# --------------------------------------------------------------------------
# In-house scalar metrics
# --------------------------------------------------------------------------

def test_classification_metrics():
    pred = torch.tensor([0, 1, 1, 0])
    gt = torch.tensor([0, 1, 0, 0])
    m = classification_metrics(pred, gt, num_classes=2)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx((0.8 + 2 / 3) / 2, abs=1e-3)


def test_ordinal_metrics():
    m = ordinal_metrics(torch.tensor([0, 1, 2]), torch.tensor([0, 2, 2]))
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rank_acc"] == pytest.approx(2 / 3)


def test_regression_metrics():
    m = regression_metrics(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 4.0]))
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(1 / 3))


def test_selection_value_prefers_objective_for_detection():
    assert _selection_value("detection", {"val_loss": 0.1, "val_objective": 5.0}, 0.2) == 5.0
    assert _selection_value("classification", {"val_loss": 0.1}, 0.2) == 0.1
    assert _selection_value("detection", {"val_loss": 0.1}, 0.2) == 0.1  # no objective -> val_loss


# --------------------------------------------------------------------------
# Effective iou_type — evaluate() scoring and run_test_evaluation metadata
# --------------------------------------------------------------------------

def test_effective_iou_type_resolution():
    assert effective_iou_type("detection", None) == "bbox"
    assert effective_iou_type("instance_seg", None) == "segm"   # segm AP by default
    assert effective_iou_type("instance_seg", "bbox") == "bbox"  # explicit override wins
    assert effective_iou_type("detection", "segm") == "segm"
    assert effective_iou_type("classification", None) == ""


def test_run_test_evaluation_records_effective_iou_type(tmp_path, monkeypatch):
    """test_results.json must record the iou_type evaluate() actually scored with
    (instance_seg defaults to segm AP — recording 'bbox' would misreport mask AP)."""
    import tcip_mcp.pipelines.composer as composer
    import tcip_mcp.pipelines.training.evaluation as evaluation

    class _DummyModel:
        def load_state_dict(self, state_dict):
            pass

        def to(self, device):
            pass

    ckpt_path = tmp_path / "model_best.pt"
    torch.save({"model_spec": {}, "model_state_dict": {}}, str(ckpt_path))
    monkeypatch.setattr(composer, "compose_model", lambda spec: _DummyModel())
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **k: {"loss": 0.1, "map50": 0.5})

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "instance_seg", str(tmp_path / "seg"))
    assert r["iou_type"] == "segm"
    on_disk = json.loads((tmp_path / "seg" / "test_results.json").read_text())
    assert on_disk["iou_type"] == "segm"

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "detection", str(tmp_path / "det"))
    assert r["iou_type"] == "bbox"

    r = run_test_evaluation(str(ckpt_path), None, "cpu", "instance_seg", str(tmp_path / "ovr"),
                            iou_type="bbox")
    assert r["iou_type"] == "bbox"  # explicit override still recorded as-is


# --------------------------------------------------------------------------
# Light integration — _validate via train()
# --------------------------------------------------------------------------

torchvision = pytest.importorskip("torchvision")
from torch.utils.data import DataLoader  # noqa: E402

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.composer import compose_model  # noqa: E402
from tcip_mcp.pipelines.data.datasets import build_dataset  # noqa: E402
from tcip_mcp.pipelines.training.generic_trainer import create_run, task_collate, train  # noqa: E402

IMG = 64


def _save_png(path: Path) -> None:
    from torchvision.utils import save_image
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.rand(3, IMG, IMG) * 0.3, str(path))


def _cfg(spec) -> dict:
    return {
        "model_spec": spec, "device": "cpu",
        "stages": [{"freeze_to": -1, "epochs": 1}], "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False},
    }


def test_validate_detection_returns_metrics_and_objective(tmp_path):
    from tcip_annotation import json_io
    from tcip_annotation.state import BBox
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        json_io.write_detect(str(labels_dir / f"img{i}.json"),
                             [BBox(19.2, 19.2, 44.8, 44.8, 0)], IMG, IMG, keep_empty=True)
    ds = build_dataset("detection", images_dir=str(images_dir), labels_dir=str(labels_dir), num_classes=1)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("detection"))

    spec = {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 256},
        "heads": [{"name": "anchor_detection", "num_classes": 1, "min_size": IMG, "max_size": IMG * 2}],
    }
    compose_model(spec)
    run = create_run(_cfg(spec), str(tmp_path / "out"))
    run = train(run, loader, val_loader=loader, task="detection")  # no AttributeError on model.heads

    assert run.status == "completed", getattr(run, "error", run.status)
    last = run.metrics_history[-1]
    for k in ("val_loss", "val_precision", "val_recall", "val_f1", "val_map50", "val_map", "val_objective"):
        assert k in last, f"missing {k}"
    assert (tmp_path / "out" / "model_best.pt").is_file()
    assert run.best_metric == pytest.approx(last["val_objective"])


def test_validate_classification_metrics(tmp_path):
    images_dir = tmp_path / "images"
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    ds = build_dataset("classification", images_dir=str(images_dir), csv_path=str(csv_path), num_classes=2)
    loader = DataLoader(ds, batch_size=3, collate_fn=task_collate("classification"))

    spec = {
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "gap"},
        "heads": [{"name": "classification", "num_classes": 2}],
    }
    compose_model(spec)
    run = create_run(_cfg(spec), str(tmp_path / "out"))
    run = train(run, loader, val_loader=loader, task="classification")

    assert run.status == "completed", getattr(run, "error", run.status)
    last = run.metrics_history[-1]
    assert "val_accuracy" in last and "val_f1" in last
    assert run.best_metric == pytest.approx(last["val_loss"])  # selection falls back to val_loss


@pytest.fixture
def json_data_dir(tmp_path: Path) -> Path:
    """Minimal dataset with per-image JSON labels/predictions in the canonical layout.

    evaluate_dataset reads GT and predictions through the json_io per-image schema
    (pixel COCO xywh + native ``score``). Files keep a ``.txt`` name because the tool
    resolves them with ``fmt='yolo'`` — json_io parses the JSON *content*.
    """
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import BBox, PredBBox

    date = "2-11-26"
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = tmp_path / "annotations" / "default" / date / "detect"
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live" / date / "detect"
    preds_dir.mkdir(parents=True)

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{name}.jpg")
        json_io.write_detect(
            str(labels_dir / f"{name}.txt"),
            [BBox(288, 216, 352, 264, 0), BBox(176, 132, 208, 156, 0)],
            640, 480,
        )
        # 1 matching prediction (TP) + 1 elsewhere (FP), confidence in the JSON score.
        json_io.write_detect(
            str(preds_dir / f"{name}.txt"),
            [PredBBox(288, 216, 352, 264, 0, confidence=0.9),
             PredBBox(496, 372, 528, 396, 0, confidence=0.7)],
            640, 480,
        )
    return tmp_path


def test_evaluate_dataset_uses_pycocotools(json_data_dir):
    data_dir = json_data_dir
    from tcip_mcp.tools.annotation_tools import evaluate_dataset
    r = evaluate_dataset(str(data_dir))
    assert "map50" in r
    # fixture: each image has 2 GT, predictions = 1 TP + 1 FP -> tp=1,fp=1,fn=1 per image (x3 images).
    assert r["total_tp"] == 3 and r["total_fp"] == 3 and r["total_fn"] == 3
    assert r["precision"] == pytest.approx(0.5)
    assert all(p["tp"] == 1 and p["fp"] == 1 and p["fn"] == 1 for p in r["per_image"])

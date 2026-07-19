"""P3 — YoloPredictor + model-kind dispatch.

The label/max_dets conversion is tested purely (no model, no heavy deps); kind sniffing is
tested on a synthetic ultralytics-shaped checkpoint; and an end-to-end pass runs against the
real baseline weights when they're present (skipped in CI where they aren't).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _fake_pred(cid: int, score: float, box: tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    return SimpleNamespace(
        bbox=SimpleNamespace(minx=x1, miny=y1, maxx=x2, maxy=y2),
        score=SimpleNamespace(value=score),
        category=SimpleNamespace(id=cid),
    )


def test_build_result_shifts_labels_to_one_indexed():
    from tcip_mcp.pipelines.inference.yolo_predictor import build_result

    r = build_result([_fake_pred(0, 0.9, (1, 2, 3, 4)), _fake_pred(1, 0.5, (5, 6, 7, 8))],
                     "img.jpg", 100, 200)
    # ultralytics 0-indexed -> shared dict 1-indexed foreground (background=0)
    assert r["labels"] == [1, 2]
    assert r["boxes"][0] == [1.0, 2.0, 3.0, 4.0]
    assert r["width"] == 100 and r["height"] == 200 and r["count"] == 2
    assert "tiles" not in r


def test_build_result_caps_at_max_dets_by_score():
    from tcip_mcp.pipelines.inference.yolo_predictor import build_result

    preds = [_fake_pred(0, s, (0, 0, 1, 1)) for s in (0.2, 0.9, 0.5, 0.7)]
    r = build_result(preds, "img.jpg", 10, 10, max_dets=2, tiles=9)
    assert r["count"] == 2
    assert r["scores"] == [0.9, 0.7]  # highest-scoring survive the full-frame cap
    assert r["tiles"] == 9


def test_build_result_round_trips_through_json_export(tmp_path):
    import json

    from tcip_mcp.pipelines.inference.yolo_predictor import build_result
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    r = build_result([_fake_pred(0, 0.8, (40, 40, 60, 60))], "img.jpg", 100, 100)
    write_predictions_json(tmp_path / "img.json", r)
    obj = json.loads((tmp_path / "img.json").read_text())["objects"][0]
    assert obj["category_id"] == 0                 # 1-indexed dict label 1 -> 0-indexed class 0
    assert obj["score"] == 0.8
    assert obj["bbox"] == [40.0, 40.0, 20.0, 20.0]  # pixel COCO xywh from xyxy [40,40,60,60]


def test_detect_kind_sniffs_ultralytics_shaped_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from tcip_mcp.pipelines.inference.predictor import detect_kind, KIND_ULTRALYTICS

    p = tmp_path / "yolo.pt"
    torch.save({"model": torch.nn.Linear(2, 2), "train_args": {"imgsz": 640},
                "names": {0: "catkin"}}, p)
    assert detect_kind(str(p)) == KIND_ULTRALYTICS


def test_kind_from_ckpt_raises_on_ambiguous_checkpoint():
    # Integrity promise (predictor.py): an undeterminable kind raises rather than guessing a default
    # and running a wrong forward. A dict with neither tcip nor ultralytics markers is ambiguous.
    from tcip_mcp.pipelines.inference.predictor import _kind_from_ckpt

    with pytest.raises(ValueError):
        _kind_from_ckpt({"optimizer": {}, "epoch": 3}, "ambiguous.pt")


def test_kind_from_ckpt_sniffs_bespoke_checkpoint():
    from tcip_mcp.pipelines.inference.predictor import _kind_from_ckpt, KIND_TCIP_MODULE

    ckpt = {"model_source": {"builder": "x:y"}, "model_state_dict": {}}
    assert _kind_from_ckpt(ckpt, "bespoke.pt") == KIND_TCIP_MODULE


def test_kind_from_ckpt_bare_module_is_ultralytics():
    # ultralytics can pickle a bare nn.Module (non-dict) in some export paths.
    from tcip_mcp.pipelines.inference.predictor import _kind_from_ckpt, KIND_ULTRALYTICS

    class _NotADict:
        pass

    assert _kind_from_ckpt(_NotADict(), "bare.pt") == KIND_ULTRALYTICS


# ── real-model integration (runs on a machine with the baseline; skipped in CI) ──

def _baseline_weights() -> Path | None:
    env = os.environ.get("TCIP_VF_ROOT")
    vf = Path(env) if env else Path.home() / "tcip-projects" / "hazelnut_catkin-05-50-95-per-date_valley-farm"
    w = vf / "models" / "baseline" / "weights.pt"
    return w if w.is_file() else None


def test_prioritize_review_queue_rejects_non_composed_kind(tmp_path, monkeypatch):
    # Active-learning uncertainty scoring reads model logits, which a SAHI-wrapped YOLO doesn't
    # expose — so a non-composed kind must fail LOUD with a clear error, not crash on .model.
    # (This guard lived in the removed score_unlabeled; prioritize_review_queue is its superset.)
    from types import SimpleNamespace

    import tcip_mcp.pipelines.inference.predictor as predmod

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    # prioritize_review_queue imports build_predictor from the predictor module at call time.
    monkeypatch.setattr(predmod, "build_predictor",
                        lambda *a, **k: SimpleNamespace(kind="ultralytics"))
    from tcip_mcp.tools.feedback_tools import prioritize_review_queue

    r = prioritize_review_queue(checkpoint_path=str(ckpt), images_dir=str(images))
    assert "error" in r and "ultralytics" in r["error"]


@pytest.mark.skipif(_baseline_weights() is None, reason="baseline YOLO weights not present")
def test_yolo_predictor_sets_max_det_override():
    pytest.importorskip("ultralytics")
    pytest.importorskip("sahi")
    from tcip_mcp.pipelines.inference.predictor import build_predictor

    pred = build_predictor(str(_baseline_weights()), score_threshold=0.25, nms_iou=0.3, max_dets=1234)
    # The operating point's max_dets must govern the model, not ultralytics' default 300.
    assert int(pred.model.model.overrides.get("max_det")) == 1234
    assert float(pred.model.model.overrides.get("iou")) == 0.3


@pytest.mark.skipif(_baseline_weights() is None, reason="baseline YOLO weights not present")
def test_build_predictor_dispatches_and_runs_baseline_yolo():
    pytest.importorskip("ultralytics")
    pytest.importorskip("sahi")
    from tcip_mcp.pipelines.inference.predictor import build_predictor
    from tcip_mcp.pipelines.inference.yolo_predictor import YoloPredictor

    weights = _baseline_weights()
    vf = weights.parent.parent.parent
    imgs = sorted((vf / "images").rglob("*.JPG"))
    if not imgs:
        pytest.skip("no sample images alongside the baseline weights")

    pred = build_predictor(str(weights), score_threshold=0.25, max_dets=300)
    assert isinstance(pred, YoloPredictor)
    assert pred.in_chans == 3 and pred.class_map  # class map traveled with the model

    r = pred.predict_tiled(str(imgs[0]), tile_size=640, overlap=0.2, global_nms_iou=0.3)
    assert set(r) >= {"image", "width", "height", "boxes", "scores", "labels", "count", "tiles"}
    assert r["count"] == len(r["boxes"]) == len(r["labels"])
    assert all(lbl >= 1 for lbl in r["labels"])  # 1-indexed foreground

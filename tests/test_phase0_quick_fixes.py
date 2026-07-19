"""Phase 0.2 — quick-correctness bundle regression tests."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------
# export_predictions writes per-image COCO/JSON prediction files
# --------------------------------------------------------------------------

def test_export_predictions_writes_json(tmp_path, monkeypatch):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")  # only existence is checked
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)
    from tcip_mcp.tools.inference_tools import export_predictions

    out = tmp_path / "out"
    export_predictions(str(ckpt), str(images_dir), str(out))
    # export_predictions writes per-image COCO/JSON, not YOLO text lines.
    data = json.loads((out / "img.json").read_text())
    assert data["image"] == "img"
    assert (data["width"], data["height"]) == (100, 100)
    objs = data["objects"]
    assert len(objs) == 1                              # (was an empty file before the fix)
    assert objs[0]["category_id"] == 0                 # 1-indexed label 1 -> class max(1-1,0)=0
    assert objs[0]["score"] == pytest.approx(0.9)      # confidence
    # COCO xywh (pixel) from pixel-xyxy box [10,10,30,30].
    assert objs[0]["bbox"] == pytest.approx([10.0, 10.0, 20.0, 20.0])


# --------------------------------------------------------------------------
# export_predictions never overwrites a bucket that has review verdicts
# --------------------------------------------------------------------------

def _fake_predictor(monkeypatch):
    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)


def test_export_predictions_redirects_when_bucket_has_verdicts(tmp_path, monkeypatch):
    from pathlib import Path

    project_root = tmp_path / "proj"
    (project_root / ".tcip" / "state").mkdir(parents=True)
    # export_predictions resolves the review state via the pinned project root.
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(project_root))

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.png")

    out = tmp_path / "preds"
    out.mkdir()
    # A prediction file already sits in the bucket, and a human verdict is recorded against it.
    (out / "img.json").write_text(
        json.dumps({"image": "img", "width": 100, "height": 100, "objects": []}))
    from tcip_annotation.review_engine import ReviewContext, ReviewDetection, ReviewEngine
    from tcip_annotation.state import PredBBox

    engine = ReviewEngine(project_root / ".tcip" / "state")
    ctx = ReviewContext(img_name="img.png", img_width=100, img_height=100,
                        pred_boxes=[PredBBox(10.0, 10.0, 30.0, 30.0, 0, confidence=0.9)])
    det = ReviewDetection(det_type="fp", class_id=0, conf=0.9, iou=None, gt_type=None, gt_idx=None,
                          pred_type="box", pred_idx=0, bbox=(10.0, 10.0, 30.0, 30.0))
    engine.record_detection_action(det, ctx, action="accepted")

    _fake_predictor(monkeypatch)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")
    from tcip_mcp.tools.inference_tools import export_predictions

    # overwrite=True is refused with the verdict count and a suggested fresh bucket.
    res2 = export_predictions(str(ckpt), str(images_dir), str(out), overwrite=True)
    assert "error" in res2 and res2["verdict_count"] == 1
    assert Path(res2["suggested_bucket"]).name == "preds@r2"

    # Default: redirect to a fresh @r2 bucket; the reviewed bucket is left intact.
    res = export_predictions(str(ckpt), str(images_dir), str(out))
    assert res["bucket_redirected"] is True
    assert Path(res["output_dir"]).name == "preds@r2"
    assert (Path(res["output_dir"]) / "img.json").is_file()
    assert json.loads((out / "img.json").read_text())["objects"] == []  # untouched


# --------------------------------------------------------------------------
# DiversityScorer: no silent random-noise / no silent no-op
# --------------------------------------------------------------------------

def test_diversity_scorer_no_backbone_raises():
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer
    with pytest.raises(RuntimeError, match="backbone"):
        DiversityScorer().score(["whatever.png"], torch.nn.Linear(3, 3), torch.device("cpu"))


def test_diversity_scorer_no_labeled_warns_and_returns_uniform(tmp_path, caplog):
    from tcip_mcp.pipelines.active_learning.scorer import DiversityScorer

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 8, 3, padding=1)

    img = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (100, 100, 100)).save(img)
    with caplog.at_level("WARNING"):
        scored = DiversityScorer().score([str(img)], M(), torch.device("cpu"))
    assert scored == [(str(img), 1.0)]
    assert any("no labeled embeddings" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------
# validate_data_quality guards a missing directory
# --------------------------------------------------------------------------

def test_validate_data_quality_missing_dir():
    from tcip_mcp.tools.data_tools import validate_data_quality
    assert "error" in validate_data_quality(str("/nonexistent/path/xyz123"))

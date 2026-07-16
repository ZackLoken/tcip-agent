"""Phase 0.2 — quick-correctness bundle regression tests."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402


# --------------------------------------------------------------------------
# recommend_model_spec falls back to torchvision backbones without timm
# --------------------------------------------------------------------------

def test_recommend_falls_back_to_torchvision_without_timm(monkeypatch):
    import tcip_mcp.pipelines.composer as composer
    monkeypatch.setattr(composer, "HAS_TIMM", False)
    for ds in (300, 1000, 5000):
        bb = composer.recommend_model_spec("classification", ds, num_classes=2)["backbone"]["name"]
        assert bb.startswith("tv_"), f"dataset_size={ds} recommended timm-only backbone {bb!r}"


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

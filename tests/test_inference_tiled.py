"""W3 — tiled inference: GenericPredictor.predict_tiled + run_inference(tile=True)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import build_model  # noqa: E402

TILE = 64


def _detection_checkpoint(tmp_path: Path) -> str:
    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def _image(tmp_path: Path, size: int = 128) -> str:
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (size, size), (120, 120, 120)).save(p)
    return str(p)


def test_predict_tiled_shape_and_bounds(tmp_path):
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    ckpt = _detection_checkpoint(tmp_path)
    img = _image(tmp_path)
    pred = GenericPredictor(ckpt, device="cpu", score_threshold=0.0)
    r = pred.predict_tiled(img, tile_size=TILE, overlap=0.2)

    assert {"image", "width", "height", "boxes", "scores", "labels", "count"} <= set(r)
    assert isinstance(r["count"], int) and r["count"] == len(r["boxes"])
    assert r["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ grid
    for b in r["boxes"]:
        assert 0 <= b[0] <= r["width"] and 0 <= b[2] <= r["width"]
        assert 0 <= b[1] <= r["height"] and 0 <= b[3] <= r["height"]


def test_run_inference_tile_flag(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = _detection_checkpoint(tmp_path)
    img = _image(tmp_path)

    r = run_inference(ckpt, image_paths=[img], tile=True, tile_size=TILE, conf_threshold=0.0)
    assert r["tiled"] is True
    assert r["total_detections"] == sum(x["count"] for x in r["results"])
    assert len(r["results"]) == 1
    # the count carries a resolved-bundle operating point, unvalidated for raw inference
    assert r["operating_point"]["conf"]["validated_against"] == "false"

    r2 = run_inference(ckpt, image_paths=[img], tile=False, conf_threshold=0.0)
    assert r2["tiled"] is False
    assert len(r2["results"]) == 1  # non-tiled path still works

"""Phase 2.1c — the web inference job runs through the tcip pipeline GenericPredictor
(one detector code path), not a separate ultralytics+SAHI stack."""

import pytest


def test_result_to_yolo_lines_normalizes():
    from tcip_mcp.pipelines.postprocessing.export import result_to_yolo_lines
    lines = result_to_yolo_lines({
        "width": 100, "height": 100,
        "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1,
    })
    parts = lines[0].split()
    assert len(parts) == 6
    assert parts[0] == "0"                          # 1-indexed label 1 -> class 0
    assert float(parts[1]) == pytest.approx(0.9)    # confidence
    assert float(parts[2]) == pytest.approx(0.2)    # cx = ((10+30)/2)/100


def test_web_worker_uses_generic_predictor_and_writes_yolo(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from PIL import Image

    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (120, 120, 120)).save(images_dir / "img.jpg")
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"stub")

    captured = {}

    class FakePredictor:
        def __init__(self, checkpoint_path=None, device=None, score_threshold=0.5):
            captured["checkpoint"] = checkpoint_path

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            captured["tile"] = tile
            return [{"image": p, "width": 100, "height": 100,
                     "boxes": [[10.0, 10.0, 30.0, 30.0]], "scores": [0.9], "labels": [1], "count": 1}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="t", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), sahi=True, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2,
    )
    _worker(job)

    assert job.status == "completed"
    assert job.done == 1 and job.total == 1
    assert captured["checkpoint"] == str(ckpt)
    assert captured["tile"] is True                 # sahi=True -> pipeline tiling
    parts = (out_dir / "img.txt").read_text().strip().split()
    assert len(parts) == 6 and parts[0] == "0"      # YOLO "cls conf cx cy w h"
    assert float(parts[2]) == pytest.approx(0.2)

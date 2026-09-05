"""Tiled inference: GenericPredictor.predict_tiled + the verified pass at tile=True."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.model_build import build_model  # noqa: E402

TILE = 64


def _detection_checkpoint(tmp_path: Path) -> str:
    """Write a bespoke detection checkpoint and register it against ``tmp_path`` as the project
    root, so a caller can load it through ``load_registered_checkpoint`` or hand its bare path to
    an MCP tool that resolves the registry itself."""
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    model = build_model({"model_source": model_source})
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_source": model_source, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(name="test-model", checkpoint_path=str(ckpt), config={},
                            project_path=str(tmp_path))
    assert "error" not in result, result
    return str(ckpt)


def _image(tmp_path: Path, size: int = 128) -> str:
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (size, size), (120, 120, 120)).save(p)
    return str(p)


def test_predict_tiled_shape_and_bounds(tmp_path):
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    ckpt = _detection_checkpoint(tmp_path)
    img = _image(tmp_path)
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    pred = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)
    r = pred.predict_tiled(img, tile_size=TILE, overlap=0.2)

    assert {"image", "width", "height", "boxes", "scores", "labels", "count"} <= set(r)
    assert isinstance(r["count"], int) and r["count"] == len(r["boxes"])
    assert r["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ grid
    for b in r["boxes"]:
        assert 0 <= b[0] <= r["width"] and 0 <= b[2] <= r["width"]
        assert 0 <= b[1] <= r["height"] and 0 <= b[3] <= r["height"]


def test_predict_tiled_stamps_cap_hit_when_the_full_frame_cap_truncates(tmp_path):
    """``predict_tiled``'s post-merge full-frame cap already truncates a dense result
    (``self.max_dets``), but the truncation itself was invisible in the returned result: this
    stamps ``cap_hit`` (computed from the pre-truncation count) so a caller building its own
    records (block calibration's ``_band_records``) can surface cap saturation as provenance."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    ckpt = _detection_checkpoint(tmp_path)
    img = _image(tmp_path)
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    pred = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)
    uncapped = pred.predict_tiled(img, tile_size=TILE, overlap=0.2)
    assert uncapped["count"] > 1, "the bespoke model must produce more than one raw detection " \
        "for this test to force a real truncation, not merely assert an untested edge"

    pred.max_dets = uncapped["count"] - 1
    capped = pred.predict_tiled(img, tile_size=TILE, overlap=0.2)
    assert capped["cap_hit"] is True
    assert capped["count"] == uncapped["count"] - 1

    # Exactly at the cap: no slicing occurs, but cap_hit still reads True (matching
    # records_from_detector's own >= convention -- sitting at the ceiling is still uncertain).
    pred.max_dets = uncapped["count"]
    at_cap = pred.predict_tiled(img, tile_size=TILE, overlap=0.2)
    assert at_cap["cap_hit"] is True
    assert at_cap["count"] == uncapped["count"]

    pred.max_dets = uncapped["count"] + 1
    not_capped = pred.predict_tiled(img, tile_size=TILE, overlap=0.2)
    assert not_capped["cap_hit"] is False
    assert not_capped["count"] == uncapped["count"]


def test_predict_tiled_whole_decode_refuses_prior_or_progress_by_name(tmp_path):
    """``prior``/``progress`` only apply to the windowed-reader resume seam; a whole-decode source
    (a plain path or ``BandGroupRef``) has no resume seam to feed them into, and silently dropping
    them would let a caller believe a whole-decode pass resumed when it quietly started over."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    ckpt = _detection_checkpoint(tmp_path)
    img = _image(tmp_path)
    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    pred = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)
    empty_prior = {"tile_info": [], "boxes": [], "scores": [], "labels": []}

    with pytest.raises(ValueError, match="resume seam"):
        pred.predict_tiled(img, tile_size=TILE, overlap=0.2, prior=empty_prior)
    with pytest.raises(ValueError, match="resume seam"):
        pred.predict_tiled(img, tile_size=TILE, overlap=0.2, progress=lambda *a: None)


def test_run_inference_tile_flag(tmp_path, monkeypatch):
    from tests._verified_checkpoint_fixtures import run_inference_verified

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = _detection_checkpoint(tmp_path)
    img = _image(tmp_path)

    r = run_inference_verified(ckpt, image_paths=[img], tile=True, tile_size=TILE, conf_threshold=0.0)
    assert r["tiled"] is True
    assert r["total_detections"] == sum(x["count"] for x in r["results"])
    assert len(r["results"]) == 1
    # the count carries a resolved-bundle operating point, unvalidated for raw inference
    assert r["operating_point"]["conf"]["validated_against"] == "false"

    r2 = run_inference_verified(ckpt, image_paths=[img], tile=False, conf_threshold=0.0)
    assert r2["tiled"] is False
    assert len(r2["results"]) == 1  # non-tiled path still works


def test_predict_tiled_whole_decode_channel_mismatch_refuses(tmp_path):
    """The channel-count refusal promoted to the whole-decode path (:class:`predict_tiled`'s path/
    ``BandGroupRef`` source kind, not just the windowed-reader kind) must be built on
    ``derivations.probe_channels`` (the file's own real band count, independently probed), never on
    ``load_image``'s output: ``load_image(path, self.in_chans)`` is already told what channel count
    to coerce toward before it returns anything, so comparing against its own output would never
    actually catch a mismatch. A real 5-band ``.npy`` file against a 3-``in_chans`` predictor is the
    proof: this must raise before any tile is read, not silently route/coerce the file to 3 bands."""
    import numpy as np
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    arr = np.zeros((128, 128, 5), dtype=np.uint8)
    path = tmp_path / "five_band.npy"
    np.save(path, arr)

    p = GenericPredictor.__new__(GenericPredictor)
    p.task = "detection"
    p.score_threshold = 0.0
    p.max_dets = None
    p.in_chans = 3

    with pytest.raises(ValueError, match="channel"):
        p.predict_tiled(str(path), tile_size=TILE)


def test_predict_tiled_whole_decode_admits_a_photographic_rgba_file_at_in_chans_3(tmp_path):
    """The rail must admit valid work, not only reject invalid work: an ordinary RGBA PNG (any
    photo with an alpha channel, common) has no real 4-vs-3 mismatch, since ``load_image``'s own
    PIL conversion coerces it to RGB before the model ever sees it, the same as the untiled
    ``predict``/``predict_batch`` paths already do. ``probe_channels`` alone can't see that
    coercion (it reads the file's raw, uncoerced mode), so the promoted refusal must not fire here
    or every alpha-channel photo would abort a tiled run that untiled inference handles fine."""
    from PIL import Image

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor

    ckpt = _detection_checkpoint(tmp_path)
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (128, 128), (10, 20, 30, 255)).save(path)

    checkpoint = load_registered_checkpoint(ckpt, project_path=str(tmp_path))
    pred = GenericPredictor(checkpoint, device="cpu", score_threshold=0.0)
    result = pred.predict_tiled(str(path), tile_size=TILE)
    assert result["width"] == 128 and result["height"] == 128


def test_run_inference_prefers_the_checkpoints_own_recorded_id_map(tmp_path, monkeypatch):
    """When the checkpoint's own config carries a recorded id_map (stamped at train time by
    subprocess_worker.py), run_inference's decode/record map uses it, never re-derived from a
    live registry, and reachable with no images_dir/classes.json at all (proving it is not
    falling through to the registry-derivation branch)."""
    import torch as _torch

    from tests._verified_checkpoint_fixtures import run_inference_verified
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 3, "min_size": TILE, "max_size": TILE * 2},
                    "task": "detection"}
    from tcip_mcp.pipelines.model_build import build_model
    model = build_model({"model_source": model_source})
    recorded_id_map = {"closed": 0, "open": 1}
    ckpt_path = tmp_path / "model_best.pt"
    _torch.save({
        "model_source": model_source,
        "model_state_dict": model.state_dict(),
        "config": {"model_source": model_source,
                   "data": {"subject": "bud", "attribute": "opening",
                            "id_map": recorded_id_map}},
    }, str(ckpt_path))
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    result = register_model(name="test-model", checkpoint_path=str(ckpt_path), config={},
                            project_path=str(tmp_path))
    assert "error" not in result, result
    img = _image(tmp_path)

    r = run_inference_verified(str(ckpt_path), image_paths=[img], conf_threshold=0.0)
    assert r["id_map"] == recorded_id_map  # the recorded map, not a fresh registry re-derivation

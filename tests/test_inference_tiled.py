"""Tiled inference: GenericPredictor.predict_sliced under a tiled execution record, and the pass
run_inference prepares at tile=True."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

TILE = 64


def _detection_checkpoint(tmp_path: Path) -> str:
    """Write a bespoke detection checkpoint and register it against ``tmp_path`` as the project
    root, so a caller can load it through ``load_registered_checkpoint`` or hand its bare path to
    an MCP tool that resolves the registry itself."""
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    return registered_checkpoint(tmp_path, model_source={
        "builder": "tests.bespoke_models:build_bespoke_detection",
        "builder_kwargs": {"min_size": TILE, "max_size": TILE * 2}, "task": "detection"})


def _image(tmp_path: Path, size: int = 128) -> str:
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (size, size), (120, 120, 120)).save(p)
    return str(p)


def _tiled_pass(tmp_path: Path, ckpt: str):
    """The tiled pass ``ckpt`` runs at tile 64, overlap 0.2, NMS at 0.3 and conf 0."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare_pass

    return prepare_pass(load_registered_checkpoint(ckpt, project=tmp_path), Stated(
        tile=True, tile_size=TILE, overlap=0.2, postprocess="nms", cross_tile_nms=0.3, conf=0.0),
        device="cpu")


def _sliced(p, source, execution=None, **kwargs) -> dict:
    """``predict_sliced`` under the pass's own record, or ``execution`` where one is given."""
    return p.predictor.predict_sliced(source, execution=execution or p.execution,
                                      tile_batch_size=8, require_masks=True, **kwargs)


def test_predict_sliced_shape_and_bounds(tmp_path):
    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))

    r = _sliced(p, _image(tmp_path))

    assert {"image", "width", "height", "boxes", "scores", "labels", "count", "cap_hit"} <= set(r)
    assert isinstance(r["count"], int) and r["count"] == len(r["boxes"])
    assert r["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ lattice
    for b in r["boxes"]:
        assert 0 <= b[0] <= r["width"] and 0 <= b[2] <= r["width"]
        assert 0 <= b[1] <= r["height"] and 0 <= b[3] <= r["height"]


def test_predict_sliced_stamps_cap_hit_when_the_full_frame_cap_truncates(tmp_path):
    """The post-merge full-frame cap truncates a dense result and stamps ``cap_hit`` from the
    pre-truncation count; sitting exactly at the cap still reads as hit."""
    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    img = _image(tmp_path)
    uncapped = _sliced(p, img, p.execution.with_value("max_dets", None, "explicit"))
    assert uncapped["count"] > 1, "the bespoke model must produce more than one raw detection " \
        "for this test to force a real truncation, not merely assert an untested edge"

    outcomes = {}
    for cap in (uncapped["count"] - 1, uncapped["count"], uncapped["count"] + 1):
        r = _sliced(p, img, p.execution.with_value("max_dets", cap, "explicit"))
        outcomes[cap - uncapped["count"]] = (r["cap_hit"], r["count"])

    assert outcomes == {-1: (True, uncapped["count"] - 1), 0: (True, uncapped["count"]),
                        1: (False, uncapped["count"])}


def test_predict_sliced_whole_decode_refuses_prior_or_progress_by_name(tmp_path):
    """``prior``/``progress`` only apply to the windowed-reader resume seam; a whole-decode source
    has no resume seam to feed them into, and silently dropping them would let a caller believe a
    whole-decode pass resumed when it quietly started over."""
    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    img = _image(tmp_path)
    empty_prior = {"slices": [], "predictions": []}

    with pytest.raises(ValueError, match="resume seam"):
        _sliced(p, img, prior=empty_prior)
    with pytest.raises(ValueError, match="resume seam"):
        _sliced(p, img, progress=lambda *a: None)


def test_the_prepared_pass_tiles_when_asked_and_not_otherwise(tmp_path):
    from tests._verified_checkpoint_fixtures import predicted_over

    ckpt = _detection_checkpoint(tmp_path)
    images_dir = str(Path(_image(tmp_path)).parent)

    tiled, tiled_results = predicted_over(tmp_path, ckpt, images_dir, tile=True,
                                          tile_size=TILE, conf=0.0)
    whole, whole_results = predicted_over(tmp_path, ckpt, images_dir, tile=False, conf=0.0)

    assert tiled.execution.tile_size == TILE and len(tiled_results) == 1
    assert whole.execution.tile_size is None and len(whole_results) == 1


def test_predict_sliced_whole_decode_channel_mismatch_refuses(tmp_path):
    """The channel-count refusal on the whole-decode path is built on the file's own probed band
    count, never on ``load_image``'s coerced output: a real 5-band ``.npy`` file against a
    3-channel predictor raises before any slice is read."""
    import numpy as np

    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    path = tmp_path / "five_band.npy"
    np.save(path, np.zeros((128, 128, 5), dtype=np.uint8))

    with pytest.raises(ValueError, match="channel"):
        _sliced(p, str(path))


def test_predict_sliced_whole_decode_admits_a_photographic_rgba_file_at_in_chans_3(tmp_path):
    """An ordinary RGBA PNG has no real 4-versus-3 mismatch, since ``load_image``'s own PIL
    conversion coerces it to RGB before the model sees it, the same as the untiled path does."""
    from PIL import Image

    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (128, 128), (10, 20, 30, 255)).save(path)

    result = _sliced(p, str(path))

    assert result["width"] == 128 and result["height"] == 128


def test_the_pass_decodes_through_the_checkpoints_own_recorded_attributes(tmp_path):
    """A checkpoint whose config records its scope's attributes decodes through them, every
    detection carrying one id per attribute, never an order re-read from a live registry."""
    from tcip_mcp import subject_registry as cr
    from tests._verified_checkpoint_fixtures import predicted_over, registered_checkpoint

    attributes = (cr.Attribute("color", "categorical", ("red", "blue")),
                  cr.Attribute("grade", "ordinal", ("low", "mid", "high")))
    ckpt = registered_checkpoint(
        tmp_path,
        model_source={"builder": "tests.bespoke_models:build_bespoke_detection",
                      "builder_kwargs": {"min_size": TILE, "max_size": TILE * 2},
                      "task": "detection"},
        data={"num_channels": 3, "scope": {"subject": "bud"}},
        registry=cr.SubjectRegistry(subjects=(cr.Subject(name="bud", attributes=attributes),)))

    p, results = predicted_over(tmp_path, ckpt, str(Path(_image(tmp_path)).parent), conf=0.0)

    assert p.scope.attributes == attributes
    for result in results:
        assert len(result["attributes"]) == len(result["boxes"])
        assert all(len(row) == len(attributes) for row in result["attributes"])

"""Tiled inference through ``GenericPredictor.predict_sliced`` and ``run_inference``.

GenericPredictor.predict_sliced under a tiled execution record, and the pass run_inference
prepares at tile=True.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._chain_fixtures import (
    BESPOKE_DETECTION,
    BESPOKE_MODELS,
    FRAME_STATISTICS,
)
from tests._producer_fixtures import gray_frame

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

TILE = 64


def _detection_checkpoint(tmp_path: Path) -> str:
    """Write and register a bespoke detection checkpoint and return its path.

    The checkpoint is registered against ``tmp_path`` as the project root.
    """
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    return registered_checkpoint(
        tmp_path,
        model_source={
            "builder": BESPOKE_DETECTION,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
    )


def _tiled_pass(tmp_path: Path, ckpt: str):
    """Prepare and return a tiled pass of ``ckpt`` at tile 64 and the checkpoint's own density.

    The pass states overlap 0.2, cross-tile NMS at 0.3 and conf 0.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare

    return prepare(
        load_registered_checkpoint(ckpt, project=tmp_path),
        Stated(
            tile=True,
            tile_size=TILE,
            overlap=0.2,
            postprocess="nms",
            cross_tile_nms=0.3,
            conf=0.0,
            rationale="a sample reason the test states",
        ),
        device="cpu",
    ).runnable()


def _sliced(p, source, execution=None, **kwargs) -> dict:
    """Run ``predict_sliced`` under the pass's own record, or ``execution`` where one is given."""
    return p.predictor.predict_sliced(
        source,
        execution=execution or p.execution,
        tile_batch_size=8,
        **kwargs,
    )


def test_predict_sliced_shape_and_bounds(tmp_path):
    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))

    r = _sliced(p, gray_frame(tmp_path))

    assert {
        "image",
        "width",
        "height",
        "boxes",
        "scores",
        "labels",
        "count",
        "cap",
    } <= set(r)
    assert isinstance(r["count"], int) and r["count"] == len(r["boxes"])
    assert r["tiles"] >= 4  # 128px image at tile 64 -> a 2x2+ lattice
    for b in r["boxes"]:
        assert 0 <= b[0] <= r["width"] and 0 <= b[2] <= r["width"]
        assert 0 <= b[1] <= r["height"] and 0 <= b[3] <= r["height"]


def test_predict_sliced_keeps_at_most_the_full_frame_cap(tmp_path):
    """``predict_sliced`` keeps at most the full frame's cap.

    The post-merge cap, the density times the frame's pixels, truncates a dense result and is
    written on the record beside the count kept, which reaches it exactly when the frame held at
    least that many detections.
    """
    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    img = gray_frame(tmp_path)
    pixels = 128 * 128
    uncapped = _sliced(
        p, img, p.execution.with_value("density", 1.0, "explicit")
    )
    assert uncapped["count"] > 1, (
        "the bespoke model must produce more than one raw detection "
        "for this test to force a real truncation, not merely assert an untested edge"
    )

    outcomes = {}
    for cap in (
        uncapped["count"] - 1,
        uncapped["count"],
        uncapped["count"] + 1,
    ):
        r = _sliced(
            p,
            img,
            p.execution.with_value(
                "density", (cap - 0.5) / pixels, "explicit"
            ),
        )
        outcomes[cap - uncapped["count"]] = (r["cap"], r["count"])

    n = uncapped["count"]
    assert outcomes == {-1: (n - 1, n - 1), 0: (n, n), 1: (n + 1, n)}


def test_predict_sliced_whole_decode_refuses_prior_or_progress_by_name(
    tmp_path,
):
    """A whole-decode slice refuses ``prior`` or ``progress``, naming the resume seam."""
    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    img = gray_frame(tmp_path)
    empty_prior = {"slices": [], "predictions": []}

    with pytest.raises(ValueError, match="resume seam"):
        _sliced(p, img, prior=empty_prior)
    with pytest.raises(ValueError, match="resume seam"):
        _sliced(p, img, progress=lambda *a: None)


def test_the_prepared_pass_tiles_when_asked_and_not_otherwise(tmp_path):
    from tests._verified_checkpoint_fixtures import (
        SAMPLE_OVERLAP,
        predicted_over,
    )

    ckpt = _detection_checkpoint(tmp_path)
    images_dir = str(Path(gray_frame(tmp_path)).parent)

    tiled, tiled_results = predicted_over(
        tmp_path,
        ckpt,
        images_dir,
        tile=True,
        tile_size=TILE,
        overlap=SAMPLE_OVERLAP,
        conf=0.0,
    )
    whole, whole_results = predicted_over(
        tmp_path, ckpt, images_dir, tile=False, conf=0.0
    )

    assert tiled.execution.tile_size == TILE and len(tiled_results) == 1
    assert whole.execution.tile_size is None and len(whole_results) == 1


def test_predict_sliced_whole_decode_channel_mismatch_refuses(tmp_path):
    """A whole-decode slice refuses a file whose band count mismatches the predictor.

    A 5-band ``.npy`` file against a 3-channel predictor refuses, naming the channel count.
    """
    import numpy as np

    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    path = tmp_path / "five_band.npy"
    np.save(path, np.zeros((128, 128, 5), dtype=np.uint8))

    with pytest.raises(ValueError, match="channel"):
        _sliced(p, str(path))


def test_predict_sliced_whole_decode_admits_a_photographic_rgba_file_at_in_chans_3(
    tmp_path,
):
    """A whole-decode slice admits a photographic RGBA file at three input channels."""
    from PIL import Image

    p = _tiled_pass(tmp_path, _detection_checkpoint(tmp_path))
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (128, 128), (10, 20, 30, 255)).save(path)

    result = _sliced(p, str(path))

    assert result["width"] == 128 and result["height"] == 128


def test_the_pass_decodes_through_the_checkpoints_own_recorded_attributes(
    tmp_path,
):
    """The pass decodes through the checkpoint's own recorded attributes.

    A checkpoint whose config records its scope's attributes prepares a pass scoped to them,
    and each detection carries one id per attribute.
    """
    from tcip_mcp import subject_registry as cr
    from tests._verified_checkpoint_fixtures import (
        predicted_over,
        registered_checkpoint,
    )

    attributes = (
        cr.Attribute("color", "categorical", ("red", "blue")),
        cr.Attribute("grade", "ordinal", ("low", "mid", "high")),
    )
    ckpt = registered_checkpoint(
        tmp_path,
        model_source={
            "builder": BESPOKE_DETECTION,
            "source_files": [BESPOKE_MODELS],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        data={"num_channels": 3, "scope": {"subject": "bud"}},
        registry=cr.SubjectRegistry(
            subjects=(cr.Subject(name="bud", attributes=attributes),)
        ),
    )

    p, results = predicted_over(
        tmp_path, ckpt, str(Path(gray_frame(tmp_path)).parent), conf=0.0
    )

    assert p.scope.attributes == attributes
    for result in results:
        assert len(result["attributes"]) == len(result["boxes"])
        assert all(len(row) == len(attributes) for row in result["attributes"])

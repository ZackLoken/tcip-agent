"""Region-scoped `propose_annotations`: crop-and-offset, entirely on the tcip-mcp side.

`propose_annotations` gained an optional `grid_cells`/`tile_size`/`overlap` region: when given, the
tool crops the source image to the named cells' bounding rect, hands the engine only that crop, and
offsets the returned candidates back to the source image's full-frame coordinates. `auto_mask`
itself (`tcip_annotation.sam_wrapper`) is untouched; every hop here is on the tcip-mcp side of the
package boundary.

These tests drive the real crop/offset code with a fake SAM2 (the pattern
`tests/test_sam_multiring_proposals.py` uses), so `auto_mask`'s own re-orientation call is real,
not mocked away.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("torch")


def _install_fake_sam(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fake only the ``sam2`` package + checkpoint file, so ``auto_mask`` (contour extraction,
    EXIF re-orientation) is the real code under test.

    ``FakeAutoMaskGenerator.generate`` reads a mask straight out of whatever pixels it is handed
    (a bright-red patch on a dark background), so its result is genuinely driven by the crop the
    tool passes it, not a pre-baked coordinate.
    """
    class FakePredictor:
        def __init__(self, model):
            self.model = model

    class FakeAutoMaskGenerator:
        def __init__(self, **kwargs):
            pass

        def generate(self, img_rgb: np.ndarray) -> list[dict]:
            mask = (img_rgb[:, :, 0] > 200) & (img_rgb[:, :, 1] < 50) & (img_rgb[:, :, 2] < 50)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                return []
            bbox = [float(xs.min()), float(ys.min()),
                    float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            return [{"segmentation": mask, "area": int(mask.sum()), "bbox": bbox,
                     "stability_score": 0.95, "predicted_iou": 0.92}]

    build_mod = types.ModuleType("sam2.build_sam")
    build_mod.build_sam2 = lambda config, ckpt, device: object()
    predictor_mod = types.ModuleType("sam2.sam2_image_predictor")
    predictor_mod.SAM2ImagePredictor = FakePredictor
    autogen_mod = types.ModuleType("sam2.automatic_mask_generator")
    autogen_mod.SAM2AutomaticMaskGenerator = FakeAutoMaskGenerator
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_mod)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_mod)
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", autogen_mod)

    home = tmp_path / "sam_home"
    ckpt = home / ".cache" / "tcip" / "sam2" / "sam2.1_hiera_tiny.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake")
    monkeypatch.setattr(Path, "home", lambda: home)

    from tcip_annotation import sam_wrapper

    monkeypatch.setattr(sam_wrapper, "_predictor", None)
    monkeypatch.setattr(sam_wrapper, "_current_model_type", None)
    monkeypatch.setattr(sam_wrapper, "_current_image_path", None)


#: The upright (as-viewed) frame this fixture draws the patch against.
UPRIGHT_W, UPRIGHT_H = 240, 120
#: The patch's known full-frame location in the upright frame: x1, y1, x2, y2.
PATCH_BOX = (80, 10, 160, 50)
TILE_SIZE = 60  # cols at x=0/60/120/180 (4), rows at y=0/60 (2); PATCH_BOX spans cols B/C, row 1.


def _upright_frame() -> Image.Image:
    """The as-viewed canvas: a dark background with a solid red patch at ``PATCH_BOX``."""
    frame = Image.new("RGB", (UPRIGHT_W, UPRIGHT_H), color=(20, 20, 20))
    arr = np.asarray(frame).copy()
    x1, y1, x2, y2 = PATCH_BOX
    arr[y1:y2, x1:x2] = (255, 0, 0)
    return Image.fromarray(arr)


@pytest.fixture
def exif_rotated_source(tmp_path: Path) -> Path:
    """A JPEG whose stored (raw sensor) pixels are the upright frame rotated 90 degrees, tagged
    EXIF orientation 6 ("rotate 90 CW to correct"), so the upright, as-viewed frame only exists
    after EXIF correction is applied, exactly like a real camera's landscape/portrait JPEG.
    """
    upright = _upright_frame()
    raw = upright.rotate(90, expand=True)  # inverse of orientation 6's "rotate 270" correction

    path = tmp_path / "source_exif6.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation
    raw.save(path, exif=exif.tobytes())
    return path


def test_fixture_rotation_is_actually_reversed_by_exif_orientation(
    exif_rotated_source: Path,
) -> None:
    """Sanity check on the fixture itself: opening the raw file and applying
    ``auto_orient_image`` must reproduce the upright frame this test's expectations are stated in
    terms of, or the rest of this module is testing nothing real."""
    from tcip_annotation.utils import auto_orient_image

    with Image.open(exif_rotated_source) as im:
        corrected = np.asarray(auto_orient_image(im).convert("RGB"))
    assert corrected.shape == (UPRIGHT_H, UPRIGHT_W, 3)
    # JPEG's DCT ringing near a hard edge can swing tens of levels, so sample a background corner
    # far from any edge and the patch interior rather than compare the full array pixel-for-pixel.
    x1, y1, x2, y2 = PATCH_BOX
    assert all(abs(int(c) - 20) <= 15 for c in corrected[5, 5])  # far corner, upright's placement
    r, g, b = (int(c) for c in corrected[(y1 + y2) // 2, (x1 + x2) // 2])
    assert r > 200 and g < 50 and b < 50  # patch interior, still red


def test_region_crop_carries_no_exif_orientation_tag(
    exif_rotated_source: Path, tmp_path: Path,
) -> None:
    """The crop `propose_annotations` writes for the engine to read must carry no EXIF
    orientation tag of its own: it is taken from the already-oriented frame, and a second
    (wrong) rotation inside ``auto_mask`` would silently displace every coordinate."""
    from tcip_mcp.pipelines.raster_source import PhotographicSource
    from tcip_mcp.tools.vision_tools import _region_rect_from_cells, _write_region_crop
    from tcip_mcp.pipelines.reference_grid import reference_cells

    with PhotographicSource(str(exif_rotated_source), 3) as src:
        assert (src.width, src.height) == (UPRIGHT_W, UPRIGHT_H)
        cells = reference_cells(src.width, src.height, TILE_SIZE, 0.0, clamp=True)
        rect = _region_rect_from_cells(cells, ["B1", "C1"])
        pixels, _spec = src.read_region(rect)

    crop_path = _write_region_crop(pixels)
    try:
        with Image.open(crop_path) as im:
            exif = im.getexif()
            assert 0x0112 not in exif, (
                f"crop unexpectedly carries an EXIF orientation tag: {exif.get(0x0112)}"
            )
    finally:
        crop_path.unlink(missing_ok=True)


def test_region_scoped_proposal_lands_at_the_full_frame_coordinates(
    exif_rotated_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing test: an EXIF-rotated source, cropped to a region, proposed against, and
    offset back, must report the patch at its true upright full-frame location, not a crop-local
    or a wrongly-re-rotated one.
    """
    _install_fake_sam(monkeypatch, tmp_path)
    from tcip_mcp.tools.vision_tools import propose_annotations

    result = propose_annotations(
        image_path=str(exif_rotated_source),
        grid_cells=["B1", "C1"],
        tile_size=TILE_SIZE,
        engine_params={"model_type": "hiera_t"},
    )
    assert "error" not in result, result
    assert result["candidate_count"] == 1
    bbox = result["candidates"][0]["bbox"]
    assert bbox == [float(v) for v in PATCH_BOX]

    # No leftover temp crop file: cleanup runs even on the success path.
    tmp_dir = Path(tempfile_gettempdir())
    leftover = list(tmp_dir.glob("tcip_propose_crop_*"))
    assert leftover == [], f"temp region crop(s) not cleaned up: {leftover}"


def tempfile_gettempdir() -> str:
    import tempfile
    return tempfile.gettempdir()


def test_region_scoped_proposal_cleans_up_temp_crop_on_engine_failure(
    exif_rotated_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The temp crop must be deleted even when the engine raises, not only on the success path."""
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.vision_tools import propose_annotations

    class BoomProposer:
        def propose(self, image_path, **params):
            raise RuntimeError("engine exploded")

    monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: BoomProposer())

    before = set(Path(tempfile_gettempdir()).glob("tcip_propose_crop_*"))
    with pytest.raises(RuntimeError, match="engine exploded"):
        propose_annotations(
            image_path=str(exif_rotated_source),
            grid_cells=["B1", "C1"],
            tile_size=TILE_SIZE,
        )
    after = set(Path(tempfile_gettempdir()).glob("tcip_propose_crop_*"))
    assert after - before == set(), "temp region crop leaked after the engine raised"


class TestWholeFrameDefaultIsUnaffected:
    """``grid_cells=None`` (the default) must take the exact whole-frame path this tool has
    always taken: the offset step must never even run."""

    def test_offset_helper_is_never_called_without_grid_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tcip_mcp.pipelines import proposal
        from tcip_mcp.tools import vision_tools

        called = []
        monkeypatch.setattr(
            vision_tools, "_offset_candidates",
            lambda candidates, origin: called.append((candidates, origin)) or candidates,
        )

        class OneBoxProposer:
            def propose(self, image_path, **params):
                return [{
                    "candidate_id": 0, "bbox": [10.0, 10.0, 30.0, 30.0], "area": 400,
                    "score": 0.9, "engine": "sam", "engine_meta": {},
                    "rings": [[(10, 10), (30, 10), (30, 30), (10, 30)]],
                }]

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: OneBoxProposer())

        img_path = tmp_path / "whole_frame.jpg"
        Image.new("RGB", (64, 64), color=(50, 50, 50)).save(img_path)

        result = vision_tools.propose_annotations(image_path=str(img_path))
        assert "error" not in result, result
        assert called == []
        assert result["candidates"][0]["bbox"] == [10.0, 10.0, 30.0, 30.0]

    def test_state_envelope_has_no_region_key_without_grid_cells(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import json

        from tcip_mcp.pipelines import proposal
        from tcip_mcp.project_paths import resolve_state
        from tcip_mcp.tools import vision_tools

        class OneBoxProposer:
            def propose(self, image_path, **params):
                return [{
                    "candidate_id": 0, "bbox": [1.0, 1.0, 2.0, 2.0], "area": 1,
                    "score": 0.5, "engine": "sam", "engine_meta": {}, "rings": [[(1, 1), (2, 1), (2, 2)]],
                }]

        monkeypatch.setattr(proposal, "resolve_proposer", lambda engine: OneBoxProposer())

        img_path = tmp_path / "no_region.jpg"
        Image.new("RGB", (32, 32), color=(10, 10, 10)).save(img_path)

        result = vision_tools.propose_annotations(image_path=str(img_path))
        assert "error" not in result, result

        state_file = resolve_state(Path(".tcip") / "state" / "proposals_no_region.json")
        envelope = json.loads(state_file.read_text(encoding="utf-8"))
        assert set(envelope) == {"engine", "candidates"}

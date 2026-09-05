"""EXIF-orientation regression: the training loader must read the same upright frame the
labels are authored in (and that ``get_image_dimensions`` / the GUI / eval already use).

Every Valley_Farm JPEG is EXIF Orientation 6 (raw 5712×4284 stored, 4284×5712 upright).
Labels are normalized in the *upright* frame. If ``load_image`` ever returns the raw
sensor frame while ``get_image_dimensions`` returns the upright one, the loader
denormalizes upright coords against raw ``(w, h)`` and scatters every box, with in-loop
mAP blind to it (raw-vs-raw). These tests assert the frames are one, by construction and
spatially, through the real dataset classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
import torch  # noqa: E402


# --------------------------------------------------------------------------
# Fixture: an Orientation-6 JPEG whose upright frame carries a known red marker,
# plus a YOLO label normalized in that upright frame.
# --------------------------------------------------------------------------

UP_W, UP_H = 80, 120  # upright is portrait (taller than wide), like a rotated bud bush
MARKER = (20, 30, 55, 60)  # red rectangle in upright pixel coords (x1, y1, x2, y2)


def _make_orient6_dataset(tmp_path: Path) -> tuple[Path, Path, tuple[int, int, int, int]]:
    """Write ``images/m.jpg`` (Orientation 6) + ``labels/m.json`` (upright-frame pixel box).

    The on-disk pixels are the upright image rotated 90° so that auto-orient's rotate(270)
    restores the upright frame, mirroring a real orientation-6 capture.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    up = Image.new("RGB", (UP_W, UP_H), (0, 0, 0))
    ImageDraw.Draw(up).rectangle(list(MARKER), fill=(255, 0, 0))
    raw = up.rotate(90, expand=True)  # stored sensor frame (landscape)
    exif = raw.getexif()
    exif[274] = 6  # Orientation
    raw.save(images_dir / "m.jpg", format="JPEG", exif=exif, quality=95)

    x1, y1, x2, y2 = MARKER  # already pixel xyxy in the upright frame
    json_io.write_annotations(str(labels_dir / "m.json"),
                              [Annotation(subject="bud", geometry=BBox(x1, y1, x2, y2))],
                              UP_W, UP_H, keep_empty=True)
    return images_dir, labels_dir, MARKER


def _red_fraction(rchan: "np.ndarray | torch.Tensor", g: "np.ndarray | torch.Tensor") -> float:
    """Fraction of pixels that are strongly red (R high, G low): marker detector, JPEG-robust."""
    r = rchan.numpy() if isinstance(rchan, torch.Tensor) else rchan
    gg = g.numpy() if isinstance(g, torch.Tensor) else g
    if r.size == 0:
        return -1.0
    return float(((r > 0.55) & (gg < 0.45)).mean())


# --------------------------------------------------------------------------
# 1. The fixture itself is genuinely Orientation-6 (guards the test, not the code).
# --------------------------------------------------------------------------

def test_orientation6_fixture_is_real(tmp_path: Path) -> None:
    images_dir, _, _ = _make_orient6_dataset(tmp_path)
    reop = Image.open(images_dir / "m.jpg")
    assert reop.size == (UP_H, UP_W)  # stored landscape (raw), differs from upright
    assert (reop._getexif() or {}).get(274) == 6  # the orientation the code must honor


# --------------------------------------------------------------------------
# 2. The invariant: load_image returns the exact frame get_image_dimensions does.
# --------------------------------------------------------------------------

def test_load_image_frame_matches_get_image_dimensions(tmp_path: Path) -> None:
    """Both readers share one orientation-tag read, so this guards the axis-swap rule
    each applies from it, not whether the tag read itself agrees."""
    from tcip_annotation.utils import get_image_dimensions
    from tcip_mcp.pipelines.image_utils import load_image

    images_dir, _, _ = _make_orient6_dataset(tmp_path)
    p = str(images_dir / "m.jpg")
    assert load_image(p, 3).size == get_image_dimensions(p) == (UP_W, UP_H)  # upright, not (UP_H, UP_W)


# --------------------------------------------------------------------------
# 3. Spatial: a round-tripped YOLO box overlays the object in the real training loader.
# --------------------------------------------------------------------------

def test_detection_dataset_box_lands_on_object(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images_dir, labels_dir, _ = _make_orient6_dataset(tmp_path)
    ds = DetectionDataset(str(images_dir), str(labels_dir), subject="bud")
    img_t, target = ds[0]

    assert img_t.shape[1:] == (UP_H, UP_W)  # [C, H, W] upright, not the raw sensor frame
    box = target["boxes"][0]
    x1, y1, x2, y2 = (int(v) for v in box.tolist())
    # The denormalized box must bound the red marker in the frame the model actually sees.
    inside = _red_fraction(img_t[0, y1:y2, x1:x2], img_t[1, y1:y2, x1:x2])
    assert inside > 0.5, f"box ({x1},{y1},{x2},{y2}) does not cover the marker (red_frac={inside:.2f})"
    # And the box is where the upright marker is (±2px for JPEG/int rounding).
    assert abs(x1 - MARKER[0]) <= 2 and abs(y1 - MARKER[1]) <= 2
    assert abs(x2 - MARKER[2]) <= 2 and abs(y2 - MARKER[3]) <= 2


# --------------------------------------------------------------------------
# 4. The tiled (SAHI-style) path inherits the same single frame at both seams
#    (__init__ dims via get_image_dimensions, __getitem__ pixels via load_image).
# --------------------------------------------------------------------------

def test_tiled_detection_dataset_box_lands_on_object(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset

    images_dir, labels_dir, _ = _make_orient6_dataset(tmp_path)
    base = DetectionDataset(str(images_dir), str(labels_dir), subject="bud")
    tiled = TiledDetectionDataset(base, tile_size=64, overlap=0.25, skip_empty=True)
    assert tiled.num_samples > 0

    # The marker (20,30,55,60) lies fully inside the origin tile [0:64, 0:64] in the upright
    # frame. Assert on that specific tile so the test discriminates a raw revert at either
    # seam: __init__ dims (get_image_dimensions) or __getitem__ pixels (load_image).
    origin_idx = next(i for i, e in enumerate(tiled._index) if e["tile_x"] == 0 and e["tile_y"] == 0)
    tile_t, target = tiled[origin_idx]
    assert tile_t.shape[1:] == (64, 64)  # padded tile
    assert len(target["boxes"]) >= 1, "origin tile lost the fully-contained marker box"
    hit = False
    for box in target["boxes"]:
        x1, y1, x2, y2 = (max(0, int(v)) for v in box.tolist())
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        if _red_fraction(tile_t[0, y1:y2, x1:x2], tile_t[1, y1:y2, x1:x2]) > 0.5:
            hit = True
    assert hit, "origin tile's label box does not cover the marker: tile pixels/labels frame split"


# --------------------------------------------------------------------------
# 5. Train and eval/viz read one frame: the whole point (no split-brain).
# --------------------------------------------------------------------------

def test_train_and_eval_read_paths_share_one_frame(tmp_path: Path) -> None:
    from tcip_annotation.utils import get_image_dimensions
    from tcip_mcp.pipelines.data.datasets import DetectionDataset
    from tcip_mcp.pipelines.image_utils import load_image

    images_dir, labels_dir, _ = _make_orient6_dataset(tmp_path)
    p = str(images_dir / "m.jpg")

    train_frame = load_image(p, 3).size          # training loader read
    eval_frame = get_image_dimensions(p)          # eval / viz / GUI read
    ds_frame = DetectionDataset(str(images_dir), str(labels_dir), subject="bud")[0][0].shape[1:]

    assert train_frame == eval_frame == (UP_W, UP_H)
    assert tuple(ds_frame) == (UP_H, UP_W)  # (H, W) == upright

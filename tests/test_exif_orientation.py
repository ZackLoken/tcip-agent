"""The training loader reads the upright frame an EXIF-oriented image's labels are in.

A JPEG tagged EXIF Orientation 6 stores its sensor frame rotated from the upright one, and a
label document records its pixel boxes and its width and height in the upright frame. These
tests hold ``load_image``, ``image_dimensions``, ``SourceHeader.display_frame`` and the
detection datasets to the upright frame, and check that a loaded label box covers the marker
it was drawn on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir

pytest.importorskip("torch")
import torch  # noqa: E402

from tests._producer_fixtures import dataset_over  # noqa: E402

# --------------------------------------------------------------------------
# Fixture: an Orientation-6 JPEG whose upright frame carries a known red marker,
# plus a label document holding the marker's pixel box in that upright frame.
# --------------------------------------------------------------------------

UP_W, UP_H = (
    80,
    120,
)  # upright is portrait (taller than wide), like a rotated bud bush
MARKER = (
    20,
    30,
    55,
    60,
)  # red rectangle in upright pixel coords (x1, y1, x2, y2)


def _make_orient6_dataset(
    tmp_path: Path,
) -> tuple[Path, tuple[int, int, int, int]]:
    """Write an Orientation-6 ``m.jpg`` and its upright-frame label document.

    The image lands in the undated capture of ``tmp_path``, its on-disk pixels the
    upright image rotated 90°; the return is that directory and the marker's upright box.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image, painted_frame

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)

    x1, y1, x2, y2 = MARKER
    up = painted_frame(
        UP_W, UP_H, (0, 0, 0), [((x1, y1, x2 + 1, y2 + 1), (255, 0, 0))]
    )
    raw = up.rotate(90, expand=True)  # stored sensor frame (landscape)
    exif = raw.getexif()
    exif[274] = 6  # Orientation
    raw.save(images_dir / "m.jpg", format="JPEG", exif=exif, quality=95)

    label_image(
        images_dir / "m.jpg",
        [Annotation(subject="bud", geometry=BBox(x1, y1, x2, y2))],
        UP_W,
        UP_H,
    )
    return images_dir, MARKER


def _red_fraction(
    rchan: np.ndarray | torch.Tensor, g: np.ndarray | torch.Tensor
) -> float:
    """Return the fraction of pixels with R above 0.55 and G below 0.45, or -1 for no pixels."""
    r = rchan.numpy() if isinstance(rchan, torch.Tensor) else rchan
    gg = g.numpy() if isinstance(g, torch.Tensor) else g
    if r.size == 0:
        return -1.0
    return float(((r > 0.55) & (gg < 0.45)).mean())


# --------------------------------------------------------------------------
# 1. The fixture itself is genuinely Orientation-6 (guards the test, not the code).
# --------------------------------------------------------------------------


def test_orientation6_fixture_is_real(tmp_path: Path) -> None:
    images_dir, _ = _make_orient6_dataset(tmp_path)
    reop = Image.open(images_dir / "m.jpg")
    assert reop.size == (
        UP_H,
        UP_W,
    )  # stored landscape (raw), differs from upright
    assert (reop._getexif() or {}).get(
        274
    ) == 6  # the orientation the code must honor


# --------------------------------------------------------------------------
# 2. The invariant: load_image returns the exact frame image_dimensions does.
# --------------------------------------------------------------------------


def test_load_image_frame_matches_image_dimensions(tmp_path: Path) -> None:
    """``load_image`` serves the upright frame ``image_dimensions`` reports."""
    from tcip_mcp.pipelines.image_utils import image_dimensions, load_image

    images_dir, _ = _make_orient6_dataset(tmp_path)
    p = str(images_dir / "m.jpg")
    # upright, not (UP_H, UP_W)
    assert load_image(p, 3).size == image_dimensions(p, 3) == (UP_W, UP_H)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_the_header_frame_and_the_decoded_frame_agree_at_every_orientation(
    tmp_path: Path,
    orientation: int,
) -> None:
    """A photograph's header size and its decoded frame agree at every EXIF orientation."""
    from tcip_annotation.utils import oriented_size
    from tcip_mcp.pipelines.raster_source import PhotographicSource

    path = tmp_path / f"o{orientation}.jpg"
    image = Image.new("RGB", (UP_H, UP_W))
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, format="JPEG", exif=exif)
    with Image.open(path) as opened:
        header_size = oriented_size(opened)
    with PhotographicSource(Image.open(path), 3) as decoded:
        assert header_size == (decoded.width, decoded.height)


# --------------------------------------------------------------------------
# 3. Spatial: the label document's pixel box overlays the object in the real training loader.
# --------------------------------------------------------------------------


def test_detection_dataset_box_lands_on_object(tmp_path: Path) -> None:

    images_dir, _ = _make_orient6_dataset(tmp_path)
    ds = dataset_over("detection", str(images_dir), subject="bud")
    img_t, target = ds[0]

    assert img_t.shape[1:] == (
        UP_H,
        UP_W,
    )  # [C, H, W] upright, not the raw sensor frame
    box = target["boxes"][0]
    x1, y1, x2, y2 = (int(v) for v in box.tolist())
    # The loaded box must bound the red marker in the frame the model actually sees.
    inside = _red_fraction(img_t[0, y1:y2, x1:x2], img_t[1, y1:y2, x1:x2])
    assert inside > 0.5, (
        f"box ({x1},{y1},{x2},{y2}) does not cover the marker (red_frac={inside:.2f})"
    )
    # And the box is where the upright marker is (±2px for JPEG/int rounding).
    assert abs(x1 - MARKER[0]) <= 2 and abs(y1 - MARKER[1]) <= 2
    assert abs(x2 - MARKER[2]) <= 2 and abs(y2 - MARKER[3]) <= 2


# --------------------------------------------------------------------------
# 4. The tiled (SAHI-style) path inherits the same single frame at both seams
#    (__init__ dims via the source pool's acquisition, __getitem__ pixels via load_image).
# --------------------------------------------------------------------------


def test_tiled_detection_dataset_box_lands_on_object(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset

    images_dir, _ = _make_orient6_dataset(tmp_path)
    base = dataset_over("detection", str(images_dir), subject="bud")
    # sliver_frac stated: one marker box derives no size spread.
    tiled = TiledDetectionDataset(
        base, tile_size=64, overlap=0.25, skip_empty=True, sliver_frac=0.5
    )
    assert tiled.num_samples > 0

    # The marker (20,30,55,60) lies fully inside the origin tile [0:64, 0:64] in the upright
    # frame. Assert on that specific tile so the test discriminates a raw revert at either
    # seam: __init__ dims (the source pool's acquisition) or __getitem__ pixels (load_image).
    origin_idx = next(
        i for i, e in enumerate(tiled._index) if e["slice"][:2] == (0, 0)
    )
    tile_t, target = tiled[origin_idx]
    assert tile_t.shape[1:] == (64, 64)  # a full slice
    assert len(target["boxes"]) >= 1, (
        "origin tile lost the fully-contained marker box"
    )
    hit = False
    for box in target["boxes"]:
        x1, y1, x2, y2 = (max(0, int(v)) for v in box.tolist())
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        if (
            _red_fraction(tile_t[0, y1:y2, x1:x2], tile_t[1, y1:y2, x1:x2])
            > 0.5
        ):
            hit = True
    assert hit, (
        "origin tile's label box does not cover the marker: tile pixels/labels frame split"
    )


# --------------------------------------------------------------------------
# 5. Train and eval/viz read one frame: the whole point (no split-brain).
# --------------------------------------------------------------------------


def test_train_and_eval_read_paths_share_one_frame(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.image_utils import load_image
    from tcip_mcp.pipelines.raster_source import SourceHeader

    images_dir, _ = _make_orient6_dataset(tmp_path)
    p = str(images_dir / "m.jpg")

    train_frame = load_image(p, 3).size  # training loader read
    eval_frame = SourceHeader(p).display_frame  # eval / viz / GUI read
    ds_frame = dataset_over("detection", str(images_dir), subject="bud")[0][
        0
    ].shape[1:]

    assert train_frame == eval_frame == (UP_W, UP_H)
    assert tuple(ds_frame) == (UP_H, UP_W)  # (H, W) == upright

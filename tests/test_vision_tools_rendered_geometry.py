"""What the vision tools actually draw: per-ring mask coverage and the failure-case layers.

These artifacts are the agent's eyes, so the assertions here read pixels out of the rendered
image rather than checking that a file was written: a mask that renders one region of a
two-region instance, or a failure case whose green layer is the prediction file again, both
write a perfectly readable picture of the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: Plain gray so a drawn color's dominance over the background is unambiguous in either channel.
BACKGROUND = (100, 100, 100)

#: Two disjoint regions of one occlusion-split instance, deliberately different in size and
#: aspect so a render of only the first is not a scaled version of the pair.
GT_RING_LEFT = [(20.0, 20.0), (90.0, 20.0), (90.0, 70.0), (20.0, 70.0)]
GT_RING_RIGHT = [(200.0, 100.0), (300.0, 100.0), (300.0, 160.0), (200.0, 160.0)]
GT_INSIDE_LEFT = (55, 45)
GT_INSIDE_RIGHT = (250, 130)

PRED_RING_LOW = [(30.0, 110.0), (80.0, 110.0), (80.0, 165.0), (30.0, 165.0)]
PRED_RING_HIGH = [(210.0, 15.0), (305.0, 15.0), (305.0, 60.0), (210.0, 60.0)]
PRED_INSIDE_LOW = (55, 140)
PRED_INSIDE_HIGH = (255, 35)

#: A pixel inside neither instance's regions, in either fixture.
OUTSIDE_EVERY_RING = (150, 90)


def _red_over_gray(px: Image.Image, xy: tuple[int, int]) -> int:
    """How strongly the first palette color (red) covers the gray background at one pixel."""
    r, g, _b = px.getpixel(xy)
    return r - g


@pytest.fixture
def split_instance_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dataset whose one image carries a two-region ground-truth mask and a two-region
    predicted mask, each region far from the others."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, Polygon

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (320, 180), BACKGROUND).save(images / "split.png")

    labels = tmp_path / "annotations"
    labels.mkdir()
    json_io.write_annotations(
        labels / "split.json",
        [Annotation(subject="leaf", geometry=Polygon(rings=[GT_RING_LEFT, GT_RING_RIGHT]))],
        320, 180,
    )

    preds = tmp_path / "predictions" / "live"
    preds.mkdir(parents=True)
    json_io.write_annotations(
        preds / "split.json",
        [Annotation(subject="leaf", geometry=Polygon(rings=[PRED_RING_LOW, PRED_RING_HIGH]),
                    score=0.8)],
        320, 180,
    )
    return tmp_path


def test_every_region_of_a_split_annotation_is_drawn_on_the_mask_render(
    split_instance_dataset: Path,
) -> None:
    """A mask render covers every ring the stored annotation holds.

    An occlusion-split object is one instance in several regions; showing only its dominant
    fragment understates the extent a reviewer confirms and a model trains on.
    """
    from tcip_mcp.tools.vision_tools import visualize

    result = visualize("annotations", str(split_instance_dataset / "images" / "split.png"),
                       task="segment")
    assert "error" not in result, result
    assert result["count"] == 1                     # one instance, drawn as its several regions

    px = Image.open(result["image_path"]).convert("RGB")
    assert px.size == (320, 180)
    assert _red_over_gray(px, GT_INSIDE_LEFT) > 40
    assert _red_over_gray(px, GT_INSIDE_RIGHT) > 40
    assert _red_over_gray(px, OUTSIDE_EVERY_RING) < 15


def test_every_region_of_a_split_prediction_is_drawn_on_the_mask_render(
    split_instance_dataset: Path,
) -> None:
    """The prediction render covers every ring too: a split predicted mask must look split."""
    from tcip_mcp.tools.vision_tools import visualize

    result = visualize("predictions", str(split_instance_dataset / "images" / "split.png"),
                       task="segment")
    assert "error" not in result, result
    assert result["count"] == 1

    px = Image.open(result["image_path"]).convert("RGB")
    assert _red_over_gray(px, PRED_INSIDE_LOW) > 40
    assert _red_over_gray(px, PRED_INSIDE_HIGH) > 40
    assert _red_over_gray(px, OUTSIDE_EVERY_RING) < 15


#: Ground truth and prediction placed far apart, so a render that sourced both layers from one
#: file cannot look like a near miss.
GT_BOX = (20.0, 20.0, 80.0, 60.0)
PRED_BOX = (200.0, 140.0, 280.0, 190.0)


def _mask_center(arr: np.ndarray, channel: int) -> tuple[float, float]:
    """The center of the pixels where ``channel`` dominates both other channels."""
    others = [c for c in (0, 1, 2) if c != channel]
    ch = arr[:, :, channel].astype(int)
    mask = np.ones(ch.shape, dtype=bool)
    for o in others:
        mask &= ch - arr[:, :, o].astype(int) > 60
    assert mask.any(), f"nothing drawn in channel {channel}"
    ys, xs = np.nonzero(mask)
    return float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)


@pytest.fixture
def mislocalized_prediction_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One image whose single prediction sits nowhere near its single ground-truth box."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (300, 200), BACKGROUND).save(images / "miss.png")

    labels = tmp_path / "annotations"
    labels.mkdir()
    json_io.write_annotations(
        labels / "miss.json", [Annotation(subject="bud", geometry=BBox(*GT_BOX))], 300, 200,
    )

    preds = tmp_path / "predictions" / "live"
    preds.mkdir(parents=True)
    json_io.write_annotations(
        preds / "miss.json",
        [Annotation(subject="bud", geometry=BBox(*PRED_BOX), score=0.4)], 300, 200,
    )
    return tmp_path


def test_a_failure_case_draws_ground_truth_from_the_label_tree(
    mislocalized_prediction_dataset: Path,
) -> None:
    """The green layer of a failure case is the image's ground truth, not its predictions again.

    Both layers coming from one file renders a perfect overlap on exactly the images the ranking
    flagged as worst, so the agent reads the artifact as a model that matched ground truth.
    """
    from tcip_mcp.tools.vision_tools import render_failure_cases

    root = mislocalized_prediction_dataset
    result = render_failure_cases(
        predictions_dir=str(root / "predictions" / "live"),
        labels_dir=str(root / "annotations"),
        images_dir=str(root / "images"),
        top_k=3,
    )
    assert "error" not in result, result
    assert len(result["case_images"]) == 1

    arr = np.asarray(Image.open(result["case_images"][0]).convert("RGB"))
    gt_cx, gt_cy = _mask_center(arr, 1)       # green: ground truth
    pred_cx, pred_cy = _mask_center(arr, 0)   # red: predictions

    assert abs(gt_cx - (GT_BOX[0] + GT_BOX[2]) / 2) < 25
    assert abs(gt_cy - (GT_BOX[1] + GT_BOX[3]) / 2) < 25
    assert abs(pred_cx - (PRED_BOX[0] + PRED_BOX[2]) / 2) < 25
    assert abs(pred_cy - (PRED_BOX[1] + PRED_BOX[3]) / 2) < 25
    # The two layers are distinct geometry, never one file drawn twice.
    assert abs(gt_cx - pred_cx) > 100

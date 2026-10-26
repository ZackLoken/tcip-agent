"""What the vision tools draw, read off the rendered pixels.

The tests cover per-ring mask coverage and the failure-case layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._training_values import TRIAGE_WEIGHTS

#: Plain gray so a drawn color's dominance over the background is unambiguous in either channel.
BACKGROUND = (100, 100, 100)

#: Two disjoint regions of one occlusion-split instance, deliberately different in size and
#: aspect so a render of only the first is not a scaled version of the pair.
GT_RING_LEFT = [(20.0, 20.0), (90.0, 20.0), (90.0, 70.0), (20.0, 70.0)]
GT_RING_RIGHT = [
    (200.0, 100.0),
    (300.0, 100.0),
    (300.0, 160.0),
    (200.0, 160.0),
]
GT_INSIDE_LEFT = (55, 45)
GT_INSIDE_RIGHT = (250, 130)

PRED_RING_LOW = [(30.0, 110.0), (80.0, 110.0), (80.0, 165.0), (30.0, 165.0)]
PRED_RING_HIGH = [(210.0, 15.0), (305.0, 15.0), (305.0, 60.0), (210.0, 60.0)]
PRED_INSIDE_LOW = (55, 140)
PRED_INSIDE_HIGH = (255, 35)

#: A pixel inside neither instance's regions, in either fixture.
OUTSIDE_EVERY_RING = (150, 90)


def _red_over_gray(px: Image.Image, xy: tuple[int, int]) -> int:
    """Return how strongly the first palette color (red) covers the gray ground at one pixel."""
    r, g, _b = px.getpixel(xy)
    return r - g


@pytest.fixture
def split_instance_dataset(tmp_path: Path) -> Path:
    """Return a dataset of one 320x180 image labeled with one two-ring ``leaf`` polygon.

    The rings are ``GT_RING_LEFT`` and ``GT_RING_RIGHT``.
    """
    from tcip_annotation.state import Annotation, Polygon
    from tests._producer_fixtures import label_image, write_image

    images = image_dir(tmp_path, UNDATED_BUCKET)
    write_image(images / "split.png", (320, 180), BACKGROUND)
    label_image(
        images / "split.png",
        [
            Annotation(
                subject="leaf",
                geometry=Polygon(rings=[GT_RING_LEFT, GT_RING_RIGHT]),
            )
        ],
        320,
        180,
    )
    return tmp_path


def _flat(ring: list[tuple[float, float]]) -> list[float]:
    return [c for xy in ring for c in xy]


def test_every_region_of_a_split_annotation_is_drawn_on_the_mask_render(
    split_instance_dataset: Path,
) -> None:
    """A mask render covers both rings the stored annotation holds.

    An occlusion-split object is one instance in two regions: the render counts one instance
    and paints inside each region, not outside them.
    """
    from tcip_mcp.tools.vision_tools import AnnotationsRender, visualize

    result = visualize(
        split_instance_dataset,
        AnnotationsRender(
            source="annotations",
            path=str(
                image_dir(split_instance_dataset, UNDATED_BUCKET) / "split.png"
            ),
            task="segment",
        ),
    )
    assert "error" not in result, result
    assert result["count"] == 1  # one instance, drawn as its several regions

    px = Image.open(result["image_path"]).convert("RGB")
    assert px.size == (320, 180)
    assert _red_over_gray(px, GT_INSIDE_LEFT) > 40
    assert _red_over_gray(px, GT_INSIDE_RIGHT) > 40
    assert _red_over_gray(px, OUTSIDE_EVERY_RING) < 15


def test_every_region_of_a_split_prediction_is_drawn_on_the_mask_render(
    split_instance_dataset: Path,
) -> None:
    """A prediction render covers both rings of a split predicted mask.

    The render counts one instance and paints inside each region, not outside them.
    """
    from tcip_mcp.tools.vision_tools import PredictionsRender, visualize
    from tests._chain_fixtures import published

    root = split_instance_dataset
    image = image_dir(root, UNDATED_BUCKET) / "split.png"
    published(
        root,
        "split-masks",
        [
            {
                "image": str(image),
                "width": 320,
                "height": 180,
                "boxes": [[30.0, 15.0, 305.0, 165.0]],
                "scores": [0.8],
                "labels": [1],
                "cap": 2,
                "masks": [
                    {
                        "segmentation": [
                            _flat(PRED_RING_LOW),
                            _flat(PRED_RING_HIGH),
                        ]
                    }
                ],
            }
        ],
        scope={"subject": "leaf"},
    )

    result = visualize(
        root,
        PredictionsRender(
            source="predictions",
            path=str(image),
            task="segment",
            bucket="split-masks",
        ),
    )
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
    """Return the center of the pixels where ``channel`` dominates both other channels."""
    others = [c for c in (0, 1, 2) if c != channel]
    ch = arr[:, :, channel].astype(int)
    mask = np.ones(ch.shape, dtype=bool)
    for o in others:
        mask &= ch - arr[:, :, o].astype(int) > 60
    assert mask.any(), f"nothing drawn in channel {channel}"
    ys, xs = np.nonzero(mask)
    return float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)


@pytest.fixture
def mislocalized_prediction_dataset(tmp_path: Path) -> Path:
    """Create a 300x200 image labeled at ``GT_BOX``, with one prediction published at ``PRED_BOX``.

    The prediction is the bucket ``live``'s, scored 0.4; the return is the dataset root.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image, write_image

    images = image_dir(tmp_path, UNDATED_BUCKET)
    write_image(images / "miss.png", (300, 200), BACKGROUND)
    label_image(
        images / "miss.png",
        [Annotation(subject="bud", geometry=BBox(*GT_BOX))],
        300,
        200,
    )

    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(
        tmp_path,
        "live",
        [
            {
                "image": str(images / "miss.png"),
                "width": 300,
                "height": 200,
                "boxes": [list(PRED_BOX)],
                "scores": [0.4],
                "labels": [1],
                "cap": 2,
            }
        ],
        scope={"subject": "bud"},
    )
    return tmp_path


def test_a_failure_case_draws_ground_truth_from_the_label_document(
    mislocalized_prediction_dataset: Path,
) -> None:
    """The green layer of a failure case is the image's ground truth, not its predictions again.

    The green layer centers on the ground-truth box and the prediction layer on the prediction,
    more than 100 px apart.
    """
    from tcip_mcp.tools.vision_tools import render_failure_cases

    root = mislocalized_prediction_dataset
    result = render_failure_cases(
        root, str(root), "live", top_k=3, **TRIAGE_WEIGHTS
    )
    assert "error" not in result, result
    assert len(result["case_images"]) == 1

    arr = np.asarray(Image.open(result["case_images"][0]).convert("RGB"))
    gt_cx, gt_cy = _mask_center(arr, 1)  # green: ground truth
    pred_cx, pred_cy = _mask_center(arr, 0)  # red: predictions

    assert abs(gt_cx - (GT_BOX[0] + GT_BOX[2]) / 2) < 25
    assert abs(gt_cy - (GT_BOX[1] + GT_BOX[3]) / 2) < 25
    assert abs(pred_cx - (PRED_BOX[0] + PRED_BOX[2]) / 2) < 25
    assert abs(pred_cy - (PRED_BOX[1] + PRED_BOX[3]) / 2) < 25
    # The two layers are distinct geometry, never one document drawn twice.
    assert abs(gt_cx - pred_cx) > 100

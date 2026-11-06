"""The detection average precision agrees with pycocotools' COCOeval on a box set.

``detection_metrics`` reports ``map50`` and ``map`` from the platform's own accumulation over the
one shared matcher's matchings. The reference is the library that defines the number, scored over
the same boxes at the same IoU thresholds, so the platform's precision envelope, 101-point recall
interpolation and mean over classes and thresholds are held to the library's figure.

``GROUND_TRUTH`` and ``DETECTIONS`` are the one table both sides read. Class 2 has ground truth
on images 1 and 2 only, and a detection on image 3.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("torch")  # evaluation.py imports torch at module load
pytest.importorskip("pycocotools")

from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

from tcip_mcp.pipelines.schemas import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    COMPARABILITY_IOU,
)
from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    detection_metrics,
    dt_record,
    gt_record,
)
from tests._dense_op_fixtures import whole_frame_record  # noqa: E402

FRAME = 100

GROUND_TRUTH: list[tuple[int, int, list[float]]] = [
    (1, 1, [10, 10, 20, 20]),
    (1, 1, [50, 50, 20, 20]),
    (1, 2, [10, 60, 30, 30]),
    (2, 1, [20, 20, 40, 40]),
    (2, 2, [60, 10, 20, 20]),
    (2, 2, [10, 10, 20, 20]),
    (3, 1, [30, 30, 25, 25]),
]
"""One row per object: image id, class id, ``[x, y, w, h]``."""

DETECTIONS: list[tuple[int, int, list[float], float]] = [
    (1, 1, [10, 10, 20, 20], 0.95),
    (1, 1, [56, 50, 20, 20], 0.80),
    (1, 1, [80, 80, 10, 10], 0.60),
    (1, 2, [12, 60, 30, 30], 0.90),
    (2, 1, [28, 20, 40, 40], 0.85),
    (2, 1, [0, 70, 15, 15], 0.55),
    (2, 2, [60, 10, 20, 20], 0.70),
    (2, 2, [40, 40, 10, 10], 0.65),
    (3, 1, [32, 31, 25, 25], 0.75),
    (3, 2, [70, 70, 12, 12], 0.50),
]
"""One row per detection: image id, class id, ``[x, y, w, h]``, score.

No two detections of a class share a score, and no detection's IoU with its object equals one
of ``AP_IOU_THRESHOLDS``, so the two matchers cannot split on an inclusive threshold or a score
tie.
"""


def _platform_figures(image_ids: tuple[int, ...]) -> tuple[float, float]:
    """Return ``(map50, map)`` of the table's images through ``detection_metrics``."""
    records = [
        whole_frame_record(
            [
                dt_record(box, c, s)
                for i, c, box, s in DETECTIONS
                if i == image
            ],
            [gt_record(box, c, 0) for i, c, box in GROUND_TRUTH if i == image],
            width=FRAME,
            height=FRAME,
            cap=None,
            image_id=str(image),
        )
        for image in image_ids
    ]
    m = detection_metrics(
        records,
        trait=None,
        conf_threshold=0.25,
        iou_threshold=COMPARABILITY_IOU,
        by_mask=False,
    )
    return m["map50"], m["map"]


def _library_figures(image_ids: tuple[int, ...]) -> tuple[float, float]:
    """Return ``(map50, map)`` of the table's images through pycocotools' ``COCOeval``.

    Each threshold's figure is the mean of the precision array's entries over recall points and
    classes that are not -1, at the platform's own IoU thresholds, with one area range and one
    detection cap far above the table's.
    """
    classes = sorted({c for i, c, _ in GROUND_TRUTH if i in image_ids})
    truth = COCO()
    truth.dataset = {
        "images": [
            {"id": i, "width": FRAME, "height": FRAME} for i in image_ids
        ],
        "categories": [{"id": c} for c in classes],
        "annotations": [
            {
                "id": n,
                "image_id": i,
                "category_id": c,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0,
            }
            for n, (i, c, box) in enumerate(
                (g for g in GROUND_TRUTH if g[0] in image_ids), start=1
            )
        ],
    }
    results = [
        {"image_id": i, "category_id": c, "bbox": box, "score": s}
        for i, c, box, s in DETECTIONS
        if i in image_ids
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        truth.createIndex()
        evaluator = COCOeval(truth, truth.loadRes(results), iouType="bbox")
        evaluator.params.iouThrs = np.array(AP_IOU_THRESHOLDS)
        evaluator.params.areaRng = [[0.0, 1e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.params.maxDets = [100]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    per_threshold = [
        float(precision[t, :, :, 0, 0][precision[t, :, :, 0, 0] > -1].mean())
        for t in range(len(AP_IOU_THRESHOLDS))
    ]
    return (
        per_threshold[AP_IOU_THRESHOLDS.index(COMPARABILITY_IOU)],
        float(np.mean(per_threshold)),
    )


@pytest.mark.parametrize(
    "image_ids",
    [(1, 2), (1, 2, 3)],
    ids=["every_class_on_each_image", "class_absent_from_one_image"],
)
def test_map_matches_the_coco_evaluator(image_ids: tuple[int, ...]) -> None:
    """``map50`` and ``map`` equal COCOeval's figures within the platform's six-place rounding."""
    map50, map_ = _platform_figures(image_ids)
    ref50, ref = _library_figures(image_ids)
    assert 0.0 < ref < ref50 < 1.0
    assert map50 == pytest.approx(ref50, abs=1e-6)
    assert map_ == pytest.approx(ref, abs=1e-6)

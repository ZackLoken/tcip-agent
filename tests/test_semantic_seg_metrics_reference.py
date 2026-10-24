"""``semantic_seg_metrics`` agrees with scikit-learn on a seeded random label map.

Per-class IoU is ``jaccard_score`` and per-class Dice is ``f1_score``, each with ``labels`` over
the run's classes and ``average=None``; ``pixel_acc`` is ``accuracy_score``. Pixels whose ground
truth is the ignore index are dropped before either side scores. A class absent from both maps has
no defined ratio: the platform reports it ``None`` and leaves it out of ``mIoU`` and ``dice``,
where scikit-learn answers its ``zero_division`` value, so the present classes and the mean over
them are what is compared.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
metrics = pytest.importorskip("sklearn.metrics")

from tcip_mcp.pipelines.training.evaluation import (  # noqa: E402
    semantic_seg_metrics,
)

NUM_CLASSES = 5
"""Classes 0 to 3 are drawn into both maps; class 4 is in neither."""
IGNORE_INDEX = 255
DRAWN_CLASSES = 4


def test_semantic_seg_metrics_agree_with_scikit_learn() -> None:
    rng = np.random.default_rng(7)
    gt = rng.integers(0, DRAWN_CLASSES, size=(3, 40, 40))
    pred = np.where(
        rng.random(gt.shape) < 0.6,
        gt,
        rng.integers(0, DRAWN_CLASSES, size=gt.shape),
    )
    gt[rng.random(gt.shape) < 0.1] = IGNORE_INDEX

    m = semantic_seg_metrics(
        torch.from_numpy(pred),
        torch.from_numpy(gt),
        NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
    )

    kept = gt != IGNORE_INDEX
    y_true, y_pred = gt[kept], pred[kept]
    labels = list(range(NUM_CLASSES))
    iou = metrics.jaccard_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    dice = metrics.f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    present = [c for c in labels if c < DRAWN_CLASSES]

    absent = str(NUM_CLASSES - 1)
    assert (m["per_class_iou"][absent], m["per_class_dice"][absent]) == (
        None,
        None,
    )
    assert [m["per_class_iou"][str(c)] for c in present] == pytest.approx(
        iou[present].tolist()
    )
    assert [m["per_class_dice"][str(c)] for c in present] == pytest.approx(
        dice[present].tolist()
    )
    assert m["mIoU"] == pytest.approx(float(np.mean(iou[present])))
    assert m["dice"] == pytest.approx(float(np.mean(dice[present])))
    assert m["pixel_acc"] == pytest.approx(
        metrics.accuracy_score(y_true, y_pred)
    )

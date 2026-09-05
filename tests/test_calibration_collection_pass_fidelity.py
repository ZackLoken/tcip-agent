"""The reference a conf sweep is resolved from must be the model's own unfiltered output.

Two properties of ``calibrate_operating_point``'s record-collection pass are load-bearing for the
resolved operating point, and neither shows up in the bundle's own pass/fail booleans:

- the pass is staged at a conf floor far below the shipping default, so hesitant detections reach
  the sweep instead of being censored inside the model before it ever sees them;
- the detection side of each record carries the same class vocabulary the ground-truth side is
  lifted to, so every class-conditioned statistic compares like with like.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

# A deliberately non-square frame: a defect that swaps an axis produces a different number here.
IMG_W, IMG_H = 600, 400
GRID_COLS, GRID_ROWS = 10, 8
OBJECTS_PER_IMAGE = GRID_COLS * GRID_ROWS
SPACING = 40.0
HALF_BOX = 10.0
HESITANT_PER_IMAGE = 10
SPURIOUS_PER_IMAGE = 3
HIGH_SCORE, HESITANT_SCORE, SPURIOUS_SCORE = 0.9, 0.2, 0.05
N_STEMS = 40
STEM_PREFIX = "plot"


def _box(cx: float, cy: float) -> tuple[float, float, float, float]:
    return (cx - HALF_BOX, cy - HALF_BOX, cx + HALF_BOX, cy + HALF_BOX)


def _object_centers(i: int) -> list[tuple[float, float]]:
    """Grid centers for image ``i``, offset by ``i`` px so no two images share GT content."""
    dx = float(i)
    centers = []
    for k in range(OBJECTS_PER_IMAGE):
        row, col = divmod(k, GRID_COLS)
        centers.append((30.0 + col * SPACING + dx, 30.0 + row * SPACING))
    return centers


def _spurious_centers(i: int) -> list[tuple[float, float]]:
    """Centers well clear of the object grid, for detections that match no object."""
    dx = float(i)
    y = 30.0 + GRID_ROWS * SPACING
    return [(30.0 + j * SPACING + dx, y) for j in range(SPURIOUS_PER_IMAGE)]


def _stem_index(image_path: str) -> int:
    return int(Path(image_path).stem[len(STEM_PREFIX):])


def _hesitant_detector_dataset(root: Path) -> tuple[Path, Path]:
    """Images plus one label file each: ``OBJECTS_PER_IMAGE`` objects on a per-image-offset grid."""
    from PIL import Image

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_STEMS):
        stem = f"{STEM_PREFIX}{i:02d}"
        Image.new("RGB", (IMG_W, IMG_H), color=(110, 120, 130)).save(images_dir / f"{stem}.png")
        anns = [Annotation(subject="bud", geometry=BBox(*_box(cx, cy)))
                for cx, cy in _object_centers(i)]
        json_io.write_annotations(str(labels_dir / f"{stem}.json"), anns, IMG_W, IMG_H,
                                  keep_empty=True)
    return images_dir, labels_dir


class _HesitantDetectorStub:
    """A predictor whose output is filtered by the in-model score threshold the caller staged.

    Every object gets a matching detection, the last ``HESITANT_PER_IMAGE`` of them at a score a
    shipping default would discard, plus ``SPURIOUS_PER_IMAGE`` detections matching nothing at a
    lower score still. A detection below the currently-staged threshold never leaves the model, the
    way a real detector's own ``score_thresh`` works, so a censoring floor is observable downstream.
    """

    def __init__(self) -> None:
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None
        self.config = {"data": {"subject": "bud"}}
        self.staged_model_thresholds: list[float] = []
        self.staged_predictor_thresholds: list[float] = []
        self.returned_scores: list[float] = []

    def predict_batch(self, paths, **kw):
        floor = self.model.score_thresh
        self.staged_model_thresholds.append(floor)
        self.staged_predictor_thresholds.append(self.score_threshold)
        results = []
        for p in paths:
            i = _stem_index(p)
            candidates = []
            for k, (cx, cy) in enumerate(_object_centers(i)):
                score = (HESITANT_SCORE if k >= OBJECTS_PER_IMAGE - HESITANT_PER_IMAGE
                         else HIGH_SCORE)
                candidates.append((_box(cx, cy), score))
            for cx, cy in _spurious_centers(i):
                candidates.append((_box(cx, cy), SPURIOUS_SCORE))
            kept = [(b, s) for b, s in candidates if s >= floor]
            self.returned_scores.extend(s for _, s in kept)
            results.append({
                "image": p, "width": IMG_W, "height": IMG_H,
                "boxes": [b for b, _ in kept], "scores": [s for _, s in kept],
                "labels": [1] * len(kept), "count": len(kept),
            })
        return results


def test_calibration_collection_pass_stages_below_the_shipping_conf(tmp_path):
    """The sweep must see the low-score tail: staged at the floor, the count-unbiased conf lands in
    the hesitant band and the resolved conf is both uncensored and consistent with the floor the
    reference was generated at."""
    import tcip_mcp.pipelines.calibration as calibration
    from tcip_mcp.pipelines.resolution import DEFAULT_CONF, VALIDATED_HELD_OUT

    images_dir, labels_dir = _hesitant_detector_dataset(tmp_path / "ds")
    stub = _HesitantDetectorStub()

    bundle, _dh, n_excluded, _evidence = calibration.calibrate_operating_point(
        stub, "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=None, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        group_by="stem", seed=3, holdout_ratio=0.5,
    )

    assert n_excluded == 0
    assert stub.staged_model_thresholds, "the collection pass never ran"
    assert all(t == pytest.approx(0.01) for t in stub.staged_model_thresholds)
    assert stub.staged_model_thresholds[0] < DEFAULT_CONF
    # The in-model floor and the predictor's own conf are staged to one value, not two.
    assert stub.staged_predictor_thresholds == stub.staged_model_thresholds
    # The hesitant and spurious detections survived into the reference rather than being censored.
    assert min(stub.returned_scores) == pytest.approx(SPURIOUS_SCORE)

    conf = bundle.get("conf")
    sweep = conf.gate_evidence
    assert sweep["conf_censored"] is False
    assert sweep["conf_floor_mismatch"] is False
    # Keeping every detection at or above the hesitant score reproduces the object count exactly,
    # and keeping the spurious ones too overshoots it, so the unbiased pick sits between them.
    assert SPURIOUS_SCORE < conf._raw <= HESITANT_SCORE
    assert conf.validated_against == VALIDATED_HELD_OUT


ATTRIBUTE_ID_MAP = {"open": 0, "closed": 1, "shed": 2}
OPEN_ID = ATTRIBUTE_ID_MAP["open"] + 1  # +1 for the detector's background class
SHED_ID = ATTRIBUTE_ID_MAP["shed"] + 1
UNUSED_ID = ATTRIBUTE_ID_MAP["closed"] + 1  # no instance in this fixture: a sparse vocabulary
OPEN_PER_IMAGE = 4
SHED_PER_IMAGE = 2
TWO_CLASS_W, TWO_CLASS_H = 700, 300
TWO_CLASS_STEMS = 6


def _two_class_boxes(i: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    """This image's objects: several ``open`` on the left, fewer ``shed`` on the right.

    The two classes differ in both count and position, so a record whose detection ids are shifted
    off the ground truth's vocabulary produces different per-class numbers, never the same ones.
    """
    dx = float(i)
    boxes = [("open", _box(50.0 + k * 60.0 + dx, 60.0)) for k in range(OPEN_PER_IMAGE)]
    boxes += [("shed", _box(500.0 + k * 60.0 + dx, 200.0)) for k in range(SHED_PER_IMAGE)]
    return boxes


def _two_class_dataset(root: Path) -> tuple[Path, Path]:
    from PIL import Image

    images_dir, labels_dir = root / "images", root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i in range(TWO_CLASS_STEMS):
        stem = f"{STEM_PREFIX}{i:02d}"
        Image.new("RGB", (TWO_CLASS_W, TWO_CLASS_H), color=(90, 90, 90)).save(
            images_dir / f"{stem}.png")
        anns = [Annotation(subject="bud", geometry=BBox(*b), attributes={"state": value})
                for value, b in _two_class_boxes(i)]
        json_io.write_annotations(str(labels_dir / f"{stem}.json"), anns,
                                  TWO_CLASS_W, TWO_CLASS_H, keep_empty=True)
    return images_dir, labels_dir


class _TwoClassStub:
    """A predictor that finds every object and names it with the id the training map assigns."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(score_thresh=0.5, nms_thresh=0.5, detections_per_img=100)
        self.device = "cpu"
        self.score_threshold = 0.5
        self.train_tile_size = None
        self.train_overlap = None
        self.config = {"data": {"subject": "bud", "attribute": "state",
                                "id_map": dict(ATTRIBUTE_ID_MAP)}}

    def predict_batch(self, paths, **kw):
        results = []
        for p in paths:
            i = _stem_index(p)
            boxes, scores, labels = [], [], []
            for value, b in _two_class_boxes(i):
                boxes.append(b)
                scores.append(0.9 if value == "open" else 0.6)
                labels.append(OPEN_ID if value == "open" else SHED_ID)
            results.append({"image": p, "width": TWO_CLASS_W, "height": TWO_CLASS_H,
                            "boxes": boxes, "scores": scores, "labels": labels,
                            "count": len(boxes)})
        return results


def test_calibration_records_name_detections_in_the_ground_truths_class_vocabulary(
    tmp_path, monkeypatch,
):
    """Both sides of every calibration and holdout record speak one class vocabulary.

    The ground-truth side is lifted to the detector's 1-indexed ids through the run's own id map;
    the detection side must stay in that same space, or every class-conditioned statistic the sweep
    computes is measured against a vocabulary the detections never occupy.
    """
    import tcip_mcp.pipelines.calibration as calibration
    import tcip_mcp.pipelines.operating_point as operating_point

    images_dir, labels_dir = _two_class_dataset(tmp_path / "ds")
    captured: dict = {}
    real_resolve = operating_point.resolve_operating_point

    def _capturing_resolve(*args, **kwargs):
        captured["calibration_records"] = kwargs["calibration_records"]
        captured["holdout_records"] = kwargs["holdout_records"]
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(operating_point, "resolve_operating_point", _capturing_resolve)

    calibration.calibrate_operating_point(
        _TwoClassStub(), "bud_opening", str(labels_dir), str(images_dir),
        tile=False, tile_size=None, overlap=0.2, tile_batch_size=8,
        global_nms_iou=0.3, postprocess="nms", cross_tile_nms=None, max_dets=None,
        group_by="stem", seed=5, holdout_ratio=0.5,
    )

    cal_records = captured["calibration_records"]
    hold_records = captured["holdout_records"]
    assert cal_records and hold_records, "no records reached the sweep"
    for records in (cal_records, hold_records):
        for record in records:
            assert record["gt"] and record["dt"]
            gt_counts = Counter(g["category_id"] for g in record["gt"])
            dt_counts = Counter(d["category_id"] for d in record["dt"])
            # This detector finds every object and nothing else, so the two sides agree exactly,
            # per class, only while both are named in the same vocabulary.
            assert dt_counts == gt_counts
            assert gt_counts[OPEN_ID] == OPEN_PER_IMAGE
            assert gt_counts[SHED_ID] == SHED_PER_IMAGE

    dt_ids = {d["category_id"] for r in cal_records for d in r["dt"]}
    assert dt_ids == {OPEN_ID, SHED_ID}
    assert UNUSED_ID not in dt_ids  # the map's middle value has no instance to name

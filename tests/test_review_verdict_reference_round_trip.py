"""The review-confirmed reference is rebuilt from verdicts the review engine itself wrote.

Every other test of the calibration adapter hands it a hand-built review-state dict. That leaves
the two sides of one agreement free to drift: what ``record_detection_action`` persists and what
``review_to_records`` reads back. These drive the real engine and feed its own persisted state to
the real adapter, so the boxes and identities that reach the reference are the ones a review
session actually produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import (
    Annotation,
    BBox,
    ReviewContext,
    ReviewEngine,
    compute_matches,
)
from tcip_mcp.pipelines.feedback import review_to_records

IMG_NAME = "IMG_0501.JPG"
IMG_W, IMG_H = 1200, 500
PRODUCER = {"checkpoint_sha256": "e3b0c44298fc1c14", "experiment_id": "exp-bud-07"}
# The prediction bucket the session reviewed, as prediction_buckets.bucket_key_of spells one.
BUCKET = "predictions/exp-bud-07/2026-02-11"


def review_state(engine: ReviewEngine) -> dict:
    """The reviewed bucket's state in the shape the calibration adapter reads."""
    return {"image": engine.image_states(BUCKET)}


@pytest.fixture
def ctx() -> ReviewContext:
    """A non-square image with non-square boxes, GT and predictions in different orders."""
    return ReviewContext(
        img_name=IMG_NAME,
        img_width=IMG_W,
        img_height=IMG_H,
        gt=[
            Annotation(subject="bud", geometry=BBox(100, 100, 300, 200)),
            Annotation(subject="bud", geometry=BBox(700, 260, 900, 420)),
            Annotation(subject="bud", geometry=BBox(50, 400, 150, 460)),
        ],
        preds=[
            Annotation(subject="bud", geometry=BBox(704, 264, 904, 424), score=0.90),
            Annotation(subject="bud", geometry=BBox(1000, 20, 1080, 100), score=0.80),
            Annotation(subject="bud", geometry=BBox(104, 104, 304, 204), score=0.72),
        ],
    )


def _xywh(ann: Annotation) -> list[float]:
    b = ann.geometry
    return [b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1]


def _review_a_whole_image(engine: ReviewEngine, ctx: ReviewContext, *, class_id: int | None = 0) -> None:
    """Adjudicate every detection on the image and complete it, as a review session would."""
    matches = compute_matches(ctx.gt, ctx.preds, iou_threshold=0.5, conf_threshold=0.25)
    for det in engine.build_detection_list(ctx, matches):
        engine.record_detection_action(
            BUCKET, det, ctx,
            action="rejected" if det.det_type == "fp" else "accepted",
            producer_identity=PRODUCER,
            conf_threshold=0.25,
            class_id=class_id,
        )
    assert engine.check_image_review_complete(BUCKET, ctx, matches) is True


def test_reference_boxes_come_back_at_the_pixel_sizes_they_were_reviewed_at(
    tmp_path: Path, ctx: ReviewContext
) -> None:
    """A box that goes into a verdict at some pixel size comes back out of the reference at that
    same size. The image is wider than it is tall and no box is square, so a size normalized
    against the wrong dimension survives the round trip as a plausible but wrong object size."""
    engine = ReviewEngine(state_dir=tmp_path, current_user="alice")
    _review_a_whole_image(engine, ctx)

    records = review_to_records(
        review_state(engine),
        bucket_identities=[PRODUCER],
        image_dims={IMG_NAME: (IMG_W, IMG_H)},
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["image_id"] == "IMG_0501"
    assert (rec["width"], rec["height"]) == (IMG_W, IMG_H)

    # Every prediction that carried a score reaches dt, at its own pixel geometry.
    dt_boxes = sorted((d["bbox"] for d in rec["dt"]), key=lambda b: b[0])
    expected_dt = sorted((_xywh(p) for p in ctx.preds), key=lambda b: b[0])
    assert len(dt_boxes) == 3
    for got, want in zip(dt_boxes, expected_dt):
        assert got == pytest.approx(want, abs=0.01)
    assert sorted(d["score"] for d in rec["dt"]) == pytest.approx([0.72, 0.80, 0.90])

    # The affirmed boxes are the two matched GT boxes plus the confirmed miss; the rejected
    # prediction contributes none.
    gt_boxes = sorted((g["bbox"] for g in rec["gt"]), key=lambda b: b[0])
    expected_gt = sorted((_xywh(g) for g in ctx.gt), key=lambda b: b[0])
    assert len(gt_boxes) == 3
    for got, want in zip(gt_boxes, expected_gt):
        assert got == pytest.approx(want, abs=0.01)
    assert rec["adjudication_covered"] is True


def test_a_verdict_carries_the_identity_fields_the_reference_reads(
    tmp_path: Path, ctx: ReviewContext
) -> None:
    """A persisted verdict records who acted, on what class, against which producing model, and
    at which confidence floor. The reference scopes verdicts to a producing model by these
    fields, so a verdict missing one is silently invisible to the model it was recorded for."""
    engine = ReviewEngine(state_dir=tmp_path, current_user="alice")
    _review_a_whole_image(engine, ctx)

    entries = engine.image_states(BUCKET)[IMG_NAME]["detections"]
    assert len(entries) == 4
    for entry in entries:
        assert entry["det_status"] == "reviewed"
        assert entry["reviewed_by"] == "alice"
        assert entry["class_name"] == "bud"
        assert entry["class_id"] == 0
        assert entry["producer_identity"] == PRODUCER
        assert entry["conf_threshold"] == 0.25

    # A verdict recorded against one model must not build another model's reference.
    other = review_to_records(
        review_state(engine),
        bucket_identities=[{"checkpoint_sha256": "0000000000000000",
                            "experiment_id": "exp-other-01"}],
        image_dims={IMG_NAME: (IMG_W, IMG_H)},
    )
    assert other == []


def test_an_unresolvable_class_identity_refuses_the_whole_reference(
    tmp_path: Path, ctx: ReviewContext
) -> None:
    """When the caller could not resolve a class identity the engine records that honestly, and
    the reference refuses rather than assuming class 0."""
    engine = ReviewEngine(state_dir=tmp_path, current_user="alice")
    _review_a_whole_image(engine, ctx, class_id=None)

    entries = engine.image_states(BUCKET)[IMG_NAME]["detections"]
    assert [e["class_id"] for e in entries] == [None, None, None, None]

    with pytest.raises(ValueError, match="can't be tied to a class this prediction bucket"):
        review_to_records(
            review_state(engine),
            bucket_identities=[PRODUCER],
            image_dims={IMG_NAME: (IMG_W, IMG_H)},
        )

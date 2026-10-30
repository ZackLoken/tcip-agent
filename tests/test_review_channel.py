"""The agent to GUI proposal channel: focus_human_attention over a bucket + stage_proposals.

focus_human_attention with a bucket resolves a model's proposals on a frame and posts an
``annotate_focus`` event naming that bucket. stage_proposals publishes agent-proposed detections as
a prediction bucket, never ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store
from tcip_annotation import json_io
from tcip_annotation.state import RING_MIN_POINTS, BBox, Polygon
from tcip_mcp.buckets import Proposed, buckets_under, read_bucket
from tcip_mcp.dataset_layout import (
    LABEL_DOCUMENTS,
    image_dir,
    prediction_key,
)
from tcip_mcp.pipelines.proposal import Proposal
from tcip_mcp.tools.gui_tools import focus_human_attention
from tcip_mcp.tools.proposal_tools import stage_proposals
from tests._producer_fixtures import kept, stub_images, write_jpegs

if TYPE_CHECKING:
    from tcip_mcp.tools.proposal_tools import StagedAssignments, StagedShapes
    from tcip_mcp.workspace import BoundProject

SIZE = (640, 480)
"""The frame size a staged proposal's normalized coordinates are denormalized against."""


@pytest.fixture(autouse=True)
def _stub_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    # No GUI backend in tests: make delivery a deterministic soft miss (else a 2s socket timeout).
    import tcip_mcp.web_client as web_client

    monkeypatch.setattr(
        web_client,
        "post_panel_event",
        lambda *a, **k: web_client.PanelEventAnswer(
            url=None, reason="no backend answered"
        ),
    )


def _staged(
    root: Path, staged: StagedShapes, producer: str, date: str, stem: str
) -> tcip_store.Key:
    """Return the key of ``stem``'s document in the bucket the staging answer ``staged`` names.

    The bucket's record is asserted to name ``producer`` as what proposed it and ``date`` as its
    capture, and to hold a document for ``stem``.
    """
    bucket = read_bucket(root, staged.bucket)
    assert (bucket.producer, bucket.date) == (
        Proposed(proposed_by=producer),
        date,
    )
    key = bucket.document_key(stem)
    assert key is not None, bucket
    return key


def _publish(root: Path, date: str, preds: dict[Path, list[float]]) -> str:
    """Publish the model bucket ``baseline/<date>`` under ``root`` and return its name.

    The bucket holds one document per image of ``preds``, a scored ``bud`` box per score (none
    for an empty list).
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return published(
        root.parent,
        f"baseline/{date}",
        [
            {
                "image": str(image),
                "width": 100,
                "height": 100,
                "boxes": [[10.0, 10.0, 20.0, 20.0]] * len(scores),
                "scores": scores,
                "labels": [1] * len(scores),
                "cap": len(scores) + 1,
            }
            for image, scores in preds.items()
        ],
        scope={"subject": "bud"},
    ).name


def _no_ground_truth(root: Path) -> bool:
    """Return whether no label document was written under ``root``."""
    return tcip_store.keys(LABEL_DOCUMENTS, str(root)) == []


def _damage(root: Path, bucket: str, stem: str) -> None:
    """Damage ``stem``'s published document in ``bucket`` so its stored bytes do not decode."""
    from tests._record_damage_fixtures import damage_record

    damage_record(prediction_key(root, bucket, stem), b"{not json")


def test_focus_over_a_bucket_lands_on_first_frame_with_proposals(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    images = stub_images(
        image_dir(root, date), [f"IMG_{i:04d}.JPG" for i in range(5)]
    )
    bucket = _publish(root, date, {images[2]: [0.9], images[3]: [0.8]})

    res = focus_human_attention(bound, str(root), "bud", date, bucket=bucket)
    assert "error" not in res
    assert (
        res["image_index"] == 2
    )  # first frame with proposals from this model
    assert res["image"] == "IMG_0002.JPG"
    assert res["n_holding_subject"] == 2
    assert res["bucket"] == bucket
    assert res["reason"] is not None


def test_focus_over_a_bucket_skips_an_empty_prediction_document(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    images = stub_images(
        image_dir(root, date), [f"IMG_{i:04d}.JPG" for i in range(3)]
    )
    # IMG_0000's document holds no detections, so it is no target.
    bucket = _publish(root, date, {images[0]: [], images[2]: [0.9]})

    res = focus_human_attention(bound, str(root), "bud", date, bucket=bucket)
    assert res["image_index"] == 2
    assert res["n_holding_subject"] == 1


def test_focus_over_a_bucket_navigates_past_an_unreadable_prediction_on_another_frame(
    tmp_path: Path, bound: BoundProject
) -> None:
    """A focus over a bucket navigates past a corrupt prediction document on another frame.

    The corrupt document does not close the call: the frame it lands on is readable, and the
    answer names the unreadable one.
    """
    root = tmp_path / "proj"
    date = "2026-02-11"
    images = stub_images(
        image_dir(root, date), [f"IMG_{i:04d}.JPG" for i in range(3)]
    )
    bucket = _publish(root, date, {images[0]: [0.7], images[2]: [0.9]})
    _damage(root, bucket, "IMG_0000")

    res = focus_human_attention(bound, str(root), "bud", date, bucket=bucket)

    assert "error" not in res
    assert res["image"] == "IMG_0002.JPG"
    assert res["unreadable"] == ["IMG_0000.JPG"]


def test_focus_over_a_bucket_refuses_an_explicitly_named_unreadable_frame(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    images = stub_images(
        image_dir(root, date), [f"IMG_{i:04d}.JPG" for i in range(3)]
    )
    bucket = _publish(root, date, {images[0]: [0.7]})
    _damage(root, bucket, "IMG_0000")

    res = focus_human_attention(
        bound, str(root), "bud", date, bucket=bucket, image_index=0
    )

    assert "error" in res
    assert "IMG_0000" in res["error"]


def test_focus_over_a_bucket_carries_the_explicit_index_and_proposal(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    images = stub_images(
        image_dir(root, date), [f"IMG_{i:04d}.JPG" for i in range(4)]
    )
    bucket = _publish(root, date, {images[0]: [0.9]})

    res = focus_human_attention(
        bound, str(root), "bud", date, bucket=bucket, image_index=3, proposal=2
    )
    assert res["image_index"] == 3
    assert res["proposal"] == {"bucket": bucket, "index": 2}


def test_focus_refuses_a_name_that_is_no_bucket(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    stub_images(image_dir(root, "2026-02-11"), ["IMG_0000.JPG"])
    res = focus_human_attention(
        bound, str(root), "bud", "2026-02-11", bucket="baseline/2026-02-11"
    )
    assert "error" in res


def test_stage_proposals_writes_prediction_format_not_gt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    # Two distinct subject NAMES (the name-based replacement for two numeric class ids).
    boxes = [
        {
            "subject": "bud",
            "conf": 0.8,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
        },
        {
            "subject": "leaf",
            "conf": 0.6,
            "cx": 0.25,
            "cy": 0.25,
            "w": 0.05,
            "h": 0.05,
        },
    ]
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="claude",
        boxes=boxes,
    )
    assert (res.n_detect, res.n_segment) == (2, 0)

    out = _staged(root, res, "claude", date, "IMG_0001")
    preds = json_io.read_label_document(out).annotations
    # Normalized cx/cy/w/h denormalized against a 640x480 image, carrying per-object confidence.
    assert len(preds) == 2
    b0 = preds[0]
    assert b0.subject == "bud"
    assert b0.score == pytest.approx(0.8)
    assert isinstance(b0.geometry, BBox)
    assert (
        b0.geometry.x1,
        b0.geometry.y1,
        b0.geometry.x2,
        b0.geometry.y2,
    ) == pytest.approx((288.0, 216.0, 352.0, 264.0))
    # Every staged object stamps the producer (model_name) as created_by + a created_at.
    for obj in preds:
        assert obj.created_by == "claude"
        assert obj.created_at
    # It must not have written any ground truth.
    assert _no_ground_truth(root)


def test_stage_proposals_rejects_unnormalized_coords(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    # pixel coords (>1) must be caught, not written off-canvas.
    boxes = [
        {
            "subject": "bud",
            "conf": 0.9,
            "cx": 320.0,
            "cy": 240.0,
            "w": 40.0,
            "h": 40.0,
        }
    ]
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="agent_proposals",
        boxes=boxes,
    )
    assert "error" in res and "normal" in res["error"].lower()


@pytest.mark.parametrize(
    "bad_model", ["../annotations/bud", "..", "a/b", "D:/evil", "a\\b", " ", ""]
)
def test_stage_proposals_rejects_a_model_name_that_is_no_single_segment(
    tmp_path: Path, bound: BoundProject, bad_model: str
) -> None:
    """The tool's schema refuses a model_name that is not one path segment.

    Each malformed model_name (a traversal, a nested or drive path, a backslash path, a blank, an
    empty string) is refused by the argument schema an MCP client calls through, and nothing
    reaches ground truth.
    """
    from tests._server_calls import called_through_the_server

    root = tmp_path / "proj"
    (image,) = write_jpegs(image_dir(root, "2026-02-11"), ["IMG_0001"], SIZE)
    good = [{"subject": "bud", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    refused = called_through_the_server(
        "stage_proposals",
        {"image_path": str(image), "model_name": bad_model, "boxes": good},
        bound=bound,
    )
    assert isinstance(refused, str) and "should match pattern" in refused, refused
    assert _no_ground_truth(root)


def test_stage_proposals_admits_a_model_name_of_one_segment_through_the_server(
    tmp_path: Path, bound: BoundProject
) -> None:
    """A model_name of one path segment passes the same schema and stages its bucket."""
    from tests._server_calls import called_through_the_server

    (image,) = write_jpegs(
        image_dir(tmp_path / "proj", "2026-02-11"), ["IMG_0001"], SIZE
    )
    good = [{"subject": "bud", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    staged = called_through_the_server(
        "stage_proposals",
        {"image_path": str(image), "model_name": "agent_proposals", "boxes": good},
        bound=bound,
    )
    assert isinstance(staged, dict) and "bucket" in staged, staged


def test_focus_rejects_path_traversal(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    stub_images(image_dir(root, date), ["IMG_0000.JPG"])
    # The focus is read-only, but a traversal date still refuses: it becomes a path segment under
    # images/, the guard stage_proposals applies.
    res = focus_human_attention(bound, str(root), "bud", "../evil")
    assert "should match pattern" in res["error"]


def test_stage_proposals_rejects_box_missing_subject(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    boxes = [
        {"conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    ]  # no subject name
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="agent_proposals",
        boxes=boxes,
    )
    assert "error" in res  # returns cleanly, doesn't crash the audited tool


@pytest.mark.parametrize(
    "subject", [None, "", 7], ids=["absent", "empty", "not_a_string"]
)
@pytest.mark.parametrize("shape", ["boxes", "polygons"])
def test_stage_proposals_refuses_a_shape_naming_no_subject_by_index(
    tmp_path: Path, shape: str, subject: str | int | None
) -> None:
    """A staged shape naming no subject is refused by its index, and nothing is staged.

    A box or polygon naming no subject is refused with an error opening with its kind and index
    1 and naming ``subject``, and no document is staged.
    """
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    good = (
        {
            "subject": "bur",
            "conf": 0.9,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
        }
        if shape == "boxes"
        else {
            "subject": "bur",
            "conf": 0.9,
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]],
        }
    )
    bad = {k: v for k, v in good.items() if k != "subject"}
    if subject is not None:
        bad["subject"] = subject
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="agent_proposals",
        **{shape: [good, bad]},
    )
    assert res["error"].startswith(
        f"{'box' if shape == 'boxes' else 'polygon'} 1: "
    )
    assert "subject" in res["error"]
    assert buckets_under(root) == []


def test_stage_proposals_writes_polygon_prediction(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0132"], SIZE)
    image_path = str(image)
    # A SAM-style mask staged as a prediction, with the mask-quality score carried as conf.
    polygons = [
        {
            "subject": "leaf",
            "conf": 0.91,
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]],
        },
    ]
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=polygons,
    )
    assert (res.n_detect, res.n_segment) == (0, 1)

    out = _staged(root, res, "sam", date, "IMG_0132")
    polys = json_io.read_label_document(out).annotations
    # Polygon denormalized to pixel space, mask-quality score carried as confidence.
    assert len(polys) == 1
    p0 = polys[0]
    assert p0.subject == "leaf"
    assert p0.score == pytest.approx(0.91)
    assert isinstance(p0.geometry, Polygon)
    # stage_proposals' input contract is one contour per proposal, so it stages exactly one ring.
    assert len(p0.geometry.rings) == 1
    assert len(p0.geometry.rings[0]) == 4
    assert p0.geometry.rings[0][0] == pytest.approx((64.0, 48.0))
    # Each staged polygon stamps the producer (model_name) as created_by + a created_at.
    assert p0.created_by == "sam"
    assert p0.created_at
    # It must not touch GT; the polygon is stored as its rings alone, its box derived by readers.
    assert _no_ground_truth(root)
    data = tcip_store.read(out)
    assert isinstance(data, dict) and isinstance(data["annotations"], list)
    (stored,) = data["annotations"]
    assert isinstance(stored, dict)
    assert "segmentation" in stored and "bbox" not in stored


def test_stage_proposals_stages_boxes_and_polygons_together(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    boxes = [
        {
            "subject": "bud",
            "conf": 0.7,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
        }
    ]
    polygons = [
        {
            "subject": "leaf",
            "conf": 0.8,
            "points": [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]],
        }
    ]
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="claude",
        boxes=boxes,
        polygons=polygons,
    )
    assert (res.n_detect, res.n_segment) == (1, 1)
    # Boxes and polygons alike land in the one per-image prediction document.
    objs = json_io.read_label_document(
        _staged(root, res, "claude", date, "IMG_0001")
    ).annotations
    assert len(objs) == 2
    # Both the box and the polygon stamp the producer (model_name) as created_by + a created_at.
    for obj in objs:
        assert obj.created_by == "claude"
        assert obj.created_at
    # One box and one polygon in the single document: no detect/segment split.
    kinds = sorted(
        "segmentation" if isinstance(o.geometry, Polygon) else "bbox"
        for o in objs
    )
    assert kinds == ["bbox", "segmentation"]
    assert _no_ground_truth(root)


def test_stage_proposals_rejects_bad_polygon(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    # A two-point ring is not a polygon.
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "points": [[0.1, 0.1], [0.2, 0.2]],
            }
        ],
    )
    assert "error" in res and f"{RING_MIN_POINTS} or more" in res["error"]
    # Un-normalized (pixel) points must be caught.
    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "points": [[100, 100], [200, 100], [150, 200]],
            }
        ],
    )
    assert "error" in res and "normal" in res["error"].lower()
    # A rejected shape must leave no partial stage.
    assert buckets_under(root) == []


def test_stage_proposals_requires_a_shape(tmp_path: Path) -> None:
    """A call naming neither input regime is refused naming both regimes."""
    res = stage_proposals(
        tmp_path,
        str(image_dir(tmp_path / "proj", "2026-02-11") / "IMG_0001.jpg"),
        actor=None,
    )
    assert "error" in res and "boxes/polygons" in res["error"]


class _FakeSingleBoxEngine:
    """A proposal engine whose ``propose`` answers one fixed candidate with one ring."""

    def propose(self, image_path: str, **params: object) -> Proposal:
        return Proposal(
            [
                {
                    "candidate_id": 1,
                    "bbox": [10.0, 10.0, 40.0, 30.0],
                    "area": 1200.0,
                    "rings": [
                        [
                            (10.0, 10.0),
                            (50.0, 10.0),
                            (50.0, 40.0),
                            (10.0, 40.0),
                        ]
                    ],
                    "score": 0.9,
                }
            ],
            None,
        )


def _staged_assignments_image(
    tmp_path: Path,
) -> tuple[str, StagedAssignments]:
    """Return a real image with one candidate staged, and the assignments keeping it as a leaf.

    The candidate is staged by ``propose_annotations``, whose ``staging_id`` the assignments name.
    """
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.tools.proposal_tools import propose_annotations

    register_proposal_engine("fake_single_box", _FakeSingleBoxEngine())
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0205"], SIZE)
    image_path = str(image)
    proposed = propose_annotations(
        tmp_path, image_path, actor=None, engine="fake_single_box"
    )
    return image_path, kept(proposed, (1, "leaf"))


def test_stage_proposals_refuses_assignments_combined_with_boxes_or_polygons(
    tmp_path: Path,
) -> None:
    image_path, assignments = _staged_assignments_image(tmp_path)
    a_box = [
        {
            "subject": "leaf",
            "conf": 0.9,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
        }
    ]
    a_polygon = [
        {
            "subject": "leaf",
            "conf": 0.9,
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]],
        }
    ]

    res_boxes = stage_proposals(
        tmp_path, image_path, assignments=assignments, boxes=a_box, actor=None
    )
    assert (
        "error" in res_boxes
        and "assignments cannot be combined" in res_boxes["error"]
    )

    res_polygons = stage_proposals(
        tmp_path,
        image_path,
        assignments=assignments,
        polygons=a_polygon,
        actor=None,
    )
    assert (
        "error" in res_polygons
        and "assignments cannot be combined" in res_polygons["error"]
    )

    # The admit case: assignments alone, the regime the combined calls above tried to pair.
    admitted = stage_proposals(
        tmp_path, image_path, assignments=assignments, actor=None
    )
    assert "error" not in admitted and admitted.proposal_count == 1


def test_stage_proposals_treats_an_empty_boxes_list_beside_assignments_as_no_shapes_given(
    tmp_path: Path,
) -> None:
    """An empty ``boxes`` list beside assignments counts as no shapes given.

    Beside an empty ``boxes`` list, the assignments regime stages its one proposal.
    """
    image_path, assignments = _staged_assignments_image(tmp_path)

    admitted = stage_proposals(
        tmp_path, image_path, assignments=assignments, boxes=[], actor=None
    )
    assert "error" not in admitted and admitted.proposal_count == 1


def test_stage_proposals_refuses_model_name_beside_assignments(
    tmp_path: Path,
) -> None:
    image_path, assignments = _staged_assignments_image(tmp_path)

    res = stage_proposals(
        tmp_path,
        image_path,
        assignments=assignments,
        actor=None,
        model_name="sam",
    )
    assert (
        "error" in res
        and "model_name is refused alongside assignments" in res["error"]
    )

    # The admit case: the identical assignments call with no model_name.
    admitted = stage_proposals(
        tmp_path, image_path, assignments=assignments, actor=None
    )
    assert "error" not in admitted and admitted.proposal_count == 1


def test_stage_proposals_refuses_boxes_or_polygons_without_model_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0206"], SIZE)
    image_path = str(image)
    a_box = [
        {
            "subject": "leaf",
            "conf": 0.9,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
        }
    ]
    a_polygon = [
        {
            "subject": "leaf",
            "conf": 0.9,
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]],
        }
    ]

    res_boxes = stage_proposals(tmp_path, image_path, boxes=a_box, actor=None)
    assert (
        "error" in res_boxes and "model_name is required" in res_boxes["error"]
    )

    res_polygons = stage_proposals(
        tmp_path, image_path, polygons=a_polygon, actor=None
    )
    assert (
        "error" in res_polygons
        and "model_name is required" in res_polygons["error"]
    )

    # The admit case: the identical boxes call with model_name stated.
    admitted = stage_proposals(
        tmp_path, image_path, actor=None, model_name="sam", boxes=a_box
    )
    assert "error" not in admitted and (
        admitted.n_detect,
        admitted.n_segment,
    ) == (
        1,
        0,
    )


# ── Rings: the pixel-frame counterpart of points ─────────────────────────────


class _FakeMultiRingEngine:
    """An engine whose ``propose`` answers one fixed candidate carrying the two rings of ``_RINGS``.

    The candidate's ``bbox`` is computed from those rings' coordinates.
    """

    _RINGS = [
        [(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)],
        [(100.0, 100.0), (140.0, 100.0), (120.0, 140.0)],
    ]

    def propose(self, image_path: str, **params: object) -> Proposal:
        xs = [x for ring in self._RINGS for x, _ in ring]
        ys = [y for ring in self._RINGS for _, y in ring]
        return Proposal(
            [
                {
                    "candidate_id": 1,
                    "bbox": [
                        min(xs),
                        min(ys),
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    ],
                    "area": 1900.0,
                    "rings": self._RINGS,
                    "score": 0.9,
                }
            ],
            None,
        )


def test_stage_proposals_admits_a_two_ring_pixel_proposal_with_pair_vertices(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0200"], SIZE)
    image_path = str(image)
    rings = [
        [[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]],
        [[100.0, 100.0], [140.0, 100.0], [120.0, 140.0]],
    ]

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[{"subject": "leaf", "conf": 0.9, "rings": rings}],
    )

    assert "error" not in res and (res.n_detect, res.n_segment) == (0, 1)
    polys = json_io.read_label_document(
        _staged(root, res, "sam", date, "IMG_0200")
    ).annotations
    assert len(polys) == 1
    assert isinstance(polys[0].geometry, Polygon)
    assert [len(r) for r in polys[0].geometry.rings] == [4, 3]
    assert polys[0].geometry.rings[0][0] == pytest.approx((10.0, 10.0))


def test_a_two_ring_proposal_round_trips_through_propose_and_accept(
    tmp_path: Path,
) -> None:
    """A two-ring proposal reads back with both rings after propose and accept.

    The proposer seam's return is staged as one candidate, accepted with a class, and read back
    with its two rings of three and four vertices.
    """
    from tcip_mcp.pipelines.proposal import register_proposal_engine
    from tcip_mcp.tools.annotation_tools import read_annotations
    from tcip_mcp.tools.proposal_tools import (
        propose_annotations,
        stage_proposals,
    )

    register_proposal_engine("fake_multi_ring", _FakeMultiRingEngine())

    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0201_b"], SIZE)
    accept_image_path = str(image)
    proposed = propose_annotations(
        tmp_path, accept_image_path, actor=None, engine="fake_multi_ring"
    )
    assert proposed.staging_id is not None and len(proposed.candidates) == 1

    accepted = stage_proposals(
        tmp_path,
        accept_image_path,
        assignments=kept(proposed, (1, "leaf")),
        actor=None,
    )
    assert "error" not in accepted

    read_back = read_annotations(tmp_path, accept_image_path, accepted.bucket)
    accepted_preds = read_back["predictions"]["annotations"]
    assert sorted(
        len(ring) for pred in accepted_preds for ring in pred["rings"]
    ) == [3, 4]


def test_stage_proposals_refuses_both_points_and_rings(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0202"], SIZE)
    image_path = str(image)

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]],
                "rings": [[[10.0, 10.0], [50.0, 10.0], [50.0, 40.0]]],
            }
        ],
    )

    assert (
        "error" in res and "exactly one of 'points' or 'rings'" in res["error"]
    )
    assert buckets_under(root) == []


def test_stage_proposals_refuses_neither_points_nor_rings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0203"], SIZE)
    image_path = str(image)

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[{"subject": "leaf", "conf": 0.9}],
    )

    assert (
        "error" in res and "exactly one of 'points' or 'rings'" in res["error"]
    )


def test_stage_proposals_refuses_a_short_ring_under_rings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0204"], SIZE)
    image_path = str(image)

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "rings": [[[10.0, 10.0], [50.0, 10.0]]],
            }
        ],
    )

    assert "error" in res and f"{RING_MIN_POINTS} or more" in res["error"]
    assert buckets_under(root) == []


def test_stage_proposals_refuses_a_pixel_vertex_outside_the_image_bounds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0205"], SIZE)
    image_path = str(image)

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "rings": [[[10.0, 10.0], [50.0, 10.0], [5000.0, 40.0]]],
            }
        ],
    )

    assert "error" in res and "pixel bounds" in res["error"]
    assert buckets_under(root) == []


def test_stage_proposals_refuses_an_out_of_range_coordinate_under_rings(
    tmp_path: Path,
) -> None:
    """A pixel vertex outside the image is refused at the default image size.

    A ring under ``rings`` with vertices at negative pixel coordinates is refused at the
    module's own default image size, and nothing is staged.
    """
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0206"], SIZE)
    image_path = str(image)

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "rings": [[[-50.0, -50.0], [50.0, -50.0], [50.0, 40.0]]],
            }
        ],
    )

    assert "error" in res and "pixel bounds" in res["error"]
    assert buckets_under(root) == []


def test_stage_proposals_refuses_a_normalized_ring_handed_under_rings(
    tmp_path: Path,
) -> None:
    """A ring of normalized coordinates handed under ``rings`` is refused.

    A ring of normalized [0,1] coordinates handed under the pixel-frame key is refused at the
    module's own default image size, naming a ring that spans under a pixel, and nothing is
    staged.
    """
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0207"], SIZE)
    image_path = str(image)

    res = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="sam",
        polygons=[
            {
                "subject": "leaf",
                "conf": 0.9,
                "rings": [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]]],
            }
        ],
    )

    assert "error" in res and "span under a pixel" in res["error"]
    assert buckets_under(root) == []


# ── Each staging is a publication of its own ───────────────────────────────

_BOX = [
    {"subject": "bud", "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
]


def test_a_second_stage_of_the_same_stem_publishes_a_second_bucket(
    tmp_path: Path,
) -> None:
    """A second stage of one stem under one producer publishes its own bucket.

    Both buckets read back, each holding its own staging, and the first document keeps its
    version.
    """
    root = tmp_path / "proj"
    date = "2026-02-11"
    (image,) = write_jpegs(image_dir(root, date), ["IMG_0001"], SIZE)
    image_path = str(image)
    first = stage_proposals(
        tmp_path, image_path, actor=None, model_name="claude", boxes=_BOX
    )
    document = _staged(root, first, "claude", date, "IMG_0001")
    written = tcip_store.read_versioned(document).version

    second = stage_proposals(
        tmp_path,
        image_path,
        actor=None,
        model_name="claude",
        boxes=[{**_BOX[0], "conf": 0.3}],
    )

    assert "error" not in second, second
    assert second.bucket != first.bucket
    (restaged,) = json_io.read_label_document(
        _staged(root, second, "claude", date, "IMG_0001")
    ).annotations
    assert restaged.score == pytest.approx(0.3)
    assert tcip_store.read_versioned(document).version == written


def test_stage_proposals_admits_two_stems_staged_in_turn(
    tmp_path: Path,
) -> None:
    """Two stems staged in turn under one model name both land, each in its own bucket."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    first_image, second_image = (
        str(image)
        for image in write_jpegs(
            image_dir(root, date), ["IMG_0001", "IMG_0002"], SIZE
        )
    )

    first = stage_proposals(
        tmp_path,
        first_image,
        actor=None,
        model_name="claude",
        boxes=_BOX,
    )
    second = stage_proposals(
        tmp_path,
        second_image,
        actor=None,
        model_name="claude",
        boxes=_BOX,
    )

    assert "error" not in first and "error" not in second
    _staged(root, first, "claude", date, "IMG_0001")
    _staged(root, second, "claude", date, "IMG_0002")

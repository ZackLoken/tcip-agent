"""The agent to GUI proposal channel: focus_human_attention over a bucket + stage_proposals.

focus_human_attention with a bucket resolves a model's proposals on a frame and posts an
``annotate_focus`` event naming that bucket (a soft miss with no GUI, but the resolution must be
right). stage_proposals writes agent-proposed detections to the predictions tree (never GT) for
canvas sign-off.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import BBox, Polygon
from tcip_mcp.dataset_layout import image_dir, prediction_root
from tcip_mcp.tools.gui_tools import focus_human_attention
from tcip_mcp.tools.proposal_tools import stage_proposals


@pytest.fixture(autouse=True)
def _stub_gui(monkeypatch):
    # No GUI backend in tests: make delivery a deterministic soft miss (else a 2s socket timeout).
    import tcip_mcp.web_client as web_client
    monkeypatch.setattr(web_client, "post_panel_event",
                        lambda *a, **k: {"delivered": False, "status": "no_subscribers"})


def _project_root(tmp_path: Path) -> Path:
    """A project directory that is genuinely not the dataset directory.

    ``focus_human_attention`` resolves images and predictions from the dataset root and carries the
    project root through to the GUI event untouched, so passing one directory for both would let a
    resolution off the wrong root pass unnoticed.
    """
    root = tmp_path / "workspace" / "proj"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _images(root: Path, date: str, names: list[str]) -> None:
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (idir / name).write_bytes(b"x")


def _bucket(root: Path, producer: str, date: str) -> Path:
    """The directory ``producer``'s documents for ``date`` sit in under ``root``'s predictions."""
    return prediction_root(root) / producer / date


def _staged(root: Path, producer: str, date: str, stem: str) -> Path:
    """The document ``stage_proposals`` publishes for ``stem`` under ``producer``: its own
    bucket, one per staged image."""
    return _bucket(root, producer, date) / stem / f"{stem}.json"


def _publish(root: Path, date: str, preds: dict[str, list[float]]) -> Path:
    """The model bucket ``baseline`` for ``date`` under ``root``: one document per image name of
    ``preds``, a scored ``bud`` box per score (none for an empty list); its directory."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return published(root.parent, _bucket(root, "baseline", date), [
        {"image": str(Path(image_dir(root, date)) / name), "width": 100, "height": 100,
         "boxes": [[10.0, 10.0, 20.0, 20.0]] * len(scores), "scores": scores,
         "labels": [1] * len(scores)} for name, scores in preds.items()],
        scope={"subject": "bud"}).path


def _image(root: Path, date: str, stem: str, size: tuple[int, int] = (640, 480)) -> None:
    # stage_proposals now denormalizes to pixel space, so it needs a real, readable image.
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(90, 110, 70)).save(idir / f"{stem}.jpg")


def _img_path(root: Path, date: str, stem: str) -> str:
    """The path ``_image`` wrote ``stem`` at, for the explicit regime's own ``image_path``
    argument, which carries the dataset root, date and stem in one value."""
    return str(Path(image_dir(root, date)) / f"{stem}.jpg")


def test_focus_over_a_bucket_lands_on_first_frame_with_proposals(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(5)]
    _images(root, date, imgs)
    bucket = _publish(root, date, {"IMG_0002.JPG": [0.9], "IMG_0003.JPG": [0.8]})

    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud", date,
                                predictions_dir=str(bucket))
    assert "error" not in res
    assert res["image_index"] == 2  # first frame with proposals from this model
    assert res["image"] == "IMG_0002.JPG"
    assert res["n_holding_subject"] == 2
    assert res["predictions_dir"] == str(bucket)
    assert isinstance(res["delivered"], bool)


def test_focus_over_a_bucket_skips_an_empty_prediction_document(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, [f"IMG_{i:04d}.JPG" for i in range(3)])
    # IMG_0000's document holds no detections, so it is no target.
    bucket = _publish(root, date, {"IMG_0000.JPG": [], "IMG_0002.JPG": [0.9]})

    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud", date,
                                predictions_dir=str(bucket))
    assert res["image_index"] == 2
    assert res["n_holding_subject"] == 1


def test_focus_over_a_bucket_navigates_past_an_unreadable_prediction_on_another_frame(
        tmp_path: Path) -> None:
    """A corrupt prediction document elsewhere on the date does not close the call: the frame it
    lands on is readable, and the unreadable one is named instead of raising."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, [f"IMG_{i:04d}.JPG" for i in range(3)])
    bucket = _publish(root, date, {"IMG_0000.JPG": [0.7], "IMG_0002.JPG": [0.9]})
    bad = bucket / "IMG_0000.json"
    bad.write_bytes(b"{not json")

    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud", date,
                                predictions_dir=str(bucket))

    assert "error" not in res
    assert res["image"] == "IMG_0002.JPG"
    assert res["unreadable"] == ["IMG_0000.JPG"]


def test_focus_over_a_bucket_refuses_an_explicitly_named_unreadable_frame(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, [f"IMG_{i:04d}.JPG" for i in range(3)])
    bad = _publish(root, date, {"IMG_0000.JPG": [0.7]}) / "IMG_0000.json"
    bad.write_bytes(b"{not json")

    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud", date,
                                predictions_dir=str(bad.parent), image_index=0)

    assert "error" in res
    assert str(bad) in res["error"]


def test_focus_over_a_bucket_carries_the_explicit_index_and_proposal(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, [f"IMG_{i:04d}.JPG" for i in range(4)])
    bucket = _publish(root, date, {"IMG_0000.JPG": [0.9]})

    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud", date,
                                predictions_dir=str(bucket), image_index=3, proposal=2)
    assert res["image_index"] == 3
    assert res["proposal"] == 2


def test_focus_refuses_a_directory_that_is_no_bucket(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _images(root, "2026-02-11", ["IMG_0000.JPG"])
    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud",
                                "2026-02-11", predictions_dir=str(_bucket(root, "baseline", "2026-02-11")))
    assert "error" in res


def test_stage_proposals_writes_prediction_format_not_gt(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001", size=(640, 480))
    # Two distinct subject NAMES (the name-based replacement for two numeric class ids).
    boxes = [
        {"subject": "bud", "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        {"subject": "leaf", "conf": 0.6, "cx": 0.25, "cy": 0.25, "w": 0.05, "h": 0.05},
    ]
    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0001"), model_name="claude", boxes=boxes)
    assert res["staged"] == 2

    out = _staged(root, "claude", date, "IMG_0001")
    assert out.is_file()
    assert res["path"] == str(out)
    preds = json_io.read_annotations(out)
    # Normalized cx/cy/w/h denormalized against a 640x480 image, carrying per-object confidence.
    assert len(preds) == 2
    b0 = preds[0]
    assert b0.subject == "bud"
    assert b0.score == pytest.approx(0.8)
    assert isinstance(b0.geometry, BBox)
    assert (b0.geometry.x1, b0.geometry.y1, b0.geometry.x2, b0.geometry.y2) == pytest.approx(
        (288.0, 216.0, 352.0, 264.0)
    )
    # Every staged object stamps the producer (model_name) as created_by + a created_at.
    data = json.loads(out.read_text())
    assert len(data["annotations"]) == 2
    for obj in data["annotations"]:
        assert obj["created_by"] == "claude"
        assert obj["created_at"]
    # It must not have written into annotations/ (GT).
    assert not (root / "annotations").exists()


def test_stage_proposals_rejects_unnormalized_coords(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001")
    # pixel coords (>1) must be caught, not written off-canvas.
    boxes = [{"subject": "bud", "conf": 0.9, "cx": 320.0, "cy": 240.0, "w": 40.0, "h": 40.0}]
    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0001"), model_name="agent_proposals", boxes=boxes)
    assert "error" in res and "normal" in res["error"].lower()


def test_stage_proposals_rejects_path_traversal_into_gt(tmp_path: Path) -> None:
    """A malformed model_name must never escape predictions/ into the GT tree; date and stem are
    no longer caller-supplied strings to traverse with (image_path resolves them structurally
    through the same parser propose_annotations uses)."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001")
    image_path = _img_path(root, date, "IMG_0001")
    good = [{"subject": "bud", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    # Include a backslash segment (a Windows separator: "predictions\..\annotations" would
    # escape) and a whitespace/empty segment, both of which is_valid_name rejects.
    for bad_model in ("../annotations/bud", "..", "a/b", "D:/evil", "a\\b", " ", ""):
        res = stage_proposals(tmp_path, image_path, model_name=bad_model, boxes=good)
        assert "error" in res
    assert not (root / "annotations").exists()  # nothing leaked into ground truth


def test_focus_rejects_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, ["IMG_0000.JPG"])
    # The focus is read-only, but a traversal date still refuses: it becomes a path segment under
    # images/, the guard stage_proposals applies.
    res = focus_human_attention(_project_root(tmp_path), tmp_path.parent, str(root), "bud", "../evil")
    assert "date must be a single safe path segment" in res["error"]


def test_stage_proposals_rejects_box_missing_subject(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001")
    boxes = [{"conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]  # no subject name
    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0001"), model_name="agent_proposals", boxes=boxes)
    assert "error" in res  # returns cleanly, doesn't crash the audited tool


@pytest.mark.parametrize("subject", [None, "", 7], ids=["absent", "empty", "not_a_string"])
@pytest.mark.parametrize("shape", ["boxes", "polygons"])
def test_stage_proposals_refuses_a_shape_naming_no_subject_by_index(
        tmp_path: Path, shape: str, subject) -> None:
    """A staged shape is an annotation, made where every annotation is: one naming no subject is
    refused by construction, the refusal naming the shape, and nothing is staged."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001")
    good = ({"subject": "bur", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
            if shape == "boxes" else
            {"subject": "bur", "conf": 0.9, "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]]})
    bad = {k: v for k, v in good.items() if k != "subject"}
    if subject is not None:
        bad["subject"] = subject
    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0001"), model_name="agent_proposals",
                          **{shape: [good, bad]})
    assert res["error"].startswith(f"{'box' if shape == 'boxes' else 'polygon'} 1: ")
    assert "subject" in res["error"]
    assert not _staged(root, "agent_proposals", date, "IMG_0001").exists()


def test_stage_proposals_writes_polygon_prediction(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0132", size=(640, 480))
    # A SAM-style mask staged as a prediction, with the mask-quality score carried as conf.
    polygons = [
        {"subject": "leaf", "conf": 0.91, "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]},
    ]
    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0132"), model_name="sam", polygons=polygons)
    assert res["staged"] == 1 and res["n_segment"] == 1 and res["n_detect"] == 0

    out = _staged(root, "sam", date, "IMG_0132")
    assert out.is_file()
    assert res["path"] == str(out)
    polys = json_io.read_annotations(out)
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
    data = json.loads(out.read_text())
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["created_by"] == "sam"
    assert data["annotations"][0]["created_at"]
    # It must not touch GT; the polygon is stored as its rings alone, its box derived by readers.
    assert not (root / "annotations").exists()
    assert "segmentation" in data["annotations"][0] and "bbox" not in data["annotations"][0]


def test_stage_proposals_stages_boxes_and_polygons_together(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001", size=(640, 480))
    boxes = [{"subject": "bud", "conf": 0.7, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    polygons = [{"subject": "leaf", "conf": 0.8, "points": [[0.1, 0.1], [0.2, 0.1], [0.15, 0.2]]}]
    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0001"), model_name="claude",
                          boxes=boxes, polygons=polygons)
    assert res["n_detect"] == 1 and res["n_segment"] == 1 and res["staged"] == 2
    # Boxes and polygons alike land in the one per-image prediction file now.
    out = _staged(root, "claude", date, "IMG_0001")
    assert out.is_file()
    assert res["path"] == str(out)
    objs = json.loads(out.read_text())["annotations"]
    assert len(objs) == 2
    # Both the box and the polygon stamp the producer (model_name) as created_by + a created_at.
    for obj in objs:
        assert obj["created_by"] == "claude"
        assert obj["created_at"]
    # One box and one polygon in the single file: no detect/segment split.
    kinds = sorted("segmentation" if "segmentation" in o else "bbox" for o in objs)
    assert kinds == ["bbox", "segmentation"]
    assert not (root / "annotations").exists()


def test_stage_proposals_rejects_bad_polygon(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001")
    image_path = _img_path(root, date, "IMG_0001")
    # Fewer than 3 points is not a polygon.
    res = stage_proposals(
        tmp_path, image_path, model_name="sam",
        polygons=[{"subject": "leaf", "conf": 0.9, "points": [[0.1, 0.1], [0.2, 0.2]]}],
    )
    assert "error" in res and "three or more points" in res["error"]
    # Un-normalized (pixel) points must be caught.
    res = stage_proposals(
        tmp_path, image_path, model_name="sam",
        polygons=[{"subject": "leaf", "conf": 0.9, "points": [[100, 100], [200, 100], [150, 200]]}],
    )
    assert "error" in res and "normal" in res["error"].lower()
    # A rejected shape must leave no partial stage.
    assert not (root / "predictions").exists()


def test_stage_proposals_requires_a_shape(tmp_path: Path) -> None:
    """Neither input regime named: refused before ever touching the image, naming both regimes
    rather than guessing which one was meant."""
    res = stage_proposals(tmp_path, str(tmp_path / "proj" / "images" / "2026-02-11" / "IMG_0001.jpg"))
    assert "error" in res and "boxes/polygons" in res["error"]


class _FakeSingleBoxEngine:
    """One candidate, whole-image proposal only, for the assignments-regime refusal tests below."""

    def propose(self, image_path, **params):
        return [{"candidate_id": 1, "bbox": [10.0, 10.0, 40.0, 30.0], "area": 1200.0,
                 "rings": [[(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)]],
                 "score": 0.9}]


def _staged_assignments_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A real image with a candidate already staged by ``propose_annotations``, the platform's
    own producer for the assignments regime's admitting call."""
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.proposal_tools import propose_annotations

    monkeypatch.setitem(proposal._ENGINES, "fake_single_box", _FakeSingleBoxEngine())
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0205", size=(640, 480))
    image_path = _img_path(root, date, "IMG_0205")
    proposed = propose_annotations(tmp_path, image_path, engine="fake_single_box")
    assert proposed["staged"] is True
    return image_path


def test_stage_proposals_refuses_assignments_combined_with_boxes_or_polygons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _staged_assignments_image(tmp_path, monkeypatch)
    assignments = [{"candidate_id": 1, "subject": "leaf"}]
    a_box = [{"subject": "leaf", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    a_polygon = [{"subject": "leaf", "conf": 0.9,
                  "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]]}]

    res_boxes = stage_proposals(tmp_path, image_path, assignments=assignments, boxes=a_box)
    assert "error" in res_boxes and "assignments cannot be combined" in res_boxes["error"]

    res_polygons = stage_proposals(tmp_path, image_path, assignments=assignments, polygons=a_polygon)
    assert "error" in res_polygons and "assignments cannot be combined" in res_polygons["error"]

    # The admit case: assignments alone, the regime the combined calls above tried to pair.
    admitted = stage_proposals(tmp_path, image_path, assignments=assignments)
    assert "error" not in admitted and admitted["proposal_count"] == 1


def test_stage_proposals_treats_an_empty_boxes_list_beside_assignments_as_no_shapes_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``boxes``/``polygons`` list carries no shape at all, so it is not "combined with
    boxes/polygons" any more than omitting the argument would be: the assignments regime runs."""
    image_path = _staged_assignments_image(tmp_path, monkeypatch)
    assignments = [{"candidate_id": 1, "subject": "leaf"}]

    admitted = stage_proposals(tmp_path, image_path, assignments=assignments, boxes=[])
    assert "error" not in admitted and admitted["proposal_count"] == 1


def test_stage_proposals_refuses_model_name_beside_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = _staged_assignments_image(tmp_path, monkeypatch)
    assignments = [{"candidate_id": 1, "subject": "leaf"}]

    res = stage_proposals(tmp_path, image_path, assignments=assignments, model_name="sam")
    assert "error" in res and "model_name is refused alongside assignments" in res["error"]

    # The admit case: the identical assignments call with no model_name.
    admitted = stage_proposals(tmp_path, image_path, assignments=assignments)
    assert "error" not in admitted and admitted["proposal_count"] == 1


def test_stage_proposals_refuses_boxes_or_polygons_without_model_name(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0206", size=(640, 480))
    image_path = _img_path(root, date, "IMG_0206")
    a_box = [{"subject": "leaf", "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    a_polygon = [{"subject": "leaf", "conf": 0.9,
                  "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]]}]

    res_boxes = stage_proposals(tmp_path, image_path, boxes=a_box)
    assert "error" in res_boxes and "model_name is required" in res_boxes["error"]

    res_polygons = stage_proposals(tmp_path, image_path, polygons=a_polygon)
    assert "error" in res_polygons and "model_name is required" in res_polygons["error"]

    # The admit case: the identical boxes call with model_name stated.
    admitted = stage_proposals(tmp_path, image_path, model_name="sam", boxes=a_box)
    assert "error" not in admitted and admitted["staged"] == 1


# ── Rings: the pixel-frame counterpart of points ─────────────────────────────


class _FakeMultiRingEngine:
    """An engine whose one object always splits into the same two disjoint pixel rings, through
    the whole-image proposal seam (``propose_annotations``/``stage_proposals``)."""

    _RINGS = [[(10.0, 10.0), (50.0, 10.0), (50.0, 40.0), (10.0, 40.0)],
              [(100.0, 100.0), (140.0, 100.0), (120.0, 140.0)]]

    def propose(self, image_path, **params):
        xs = [x for ring in self._RINGS for x, _ in ring]
        ys = [y for ring in self._RINGS for _, y in ring]
        return [{"candidate_id": 1, "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": 1900.0, "rings": self._RINGS, "score": 0.9}]


def test_stage_proposals_admits_a_two_ring_pixel_proposal_with_pair_vertices(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0200", size=(640, 480))
    rings = [[[10.0, 10.0], [50.0, 10.0], [50.0, 40.0], [10.0, 40.0]],
             [[100.0, 100.0], [140.0, 100.0], [120.0, 140.0]]]

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0200"), model_name="sam",
                          polygons=[{"subject": "leaf", "conf": 0.9, "rings": rings}])

    assert res["staged"] == 1 and "error" not in res
    out = _staged(root, "sam", date, "IMG_0200")
    polys = json_io.read_annotations(out)
    assert len(polys) == 1
    assert isinstance(polys[0].geometry, Polygon)
    assert [len(r) for r in polys[0].geometry.rings] == [4, 3]
    assert polys[0].geometry.rings[0][0] == pytest.approx((10.0, 10.0))


def test_a_two_ring_proposal_round_trips_through_propose_and_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admit case is the proposer seam's actual return, not a hand-built shape: staged as
    candidates, accepted with a class, and read back with both rings intact."""
    from tcip_mcp.pipelines import proposal
    from tcip_mcp.tools.annotation_tools import read_annotations
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    monkeypatch.setitem(proposal._ENGINES, "fake_multi_ring", _FakeMultiRingEngine())

    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0201_b", size=(640, 480))
    accept_image_path = str(Path(image_dir(root, date)) / "IMG_0201_b.jpg")
    proposed = propose_annotations(tmp_path, accept_image_path, engine="fake_multi_ring")
    assert proposed["staged"] is True and proposed["candidate_count"] == 1

    accepted = stage_proposals(tmp_path, accept_image_path, assignments=[{"candidate_id": 1, "subject": "leaf"}])
    assert "error" not in accepted

    read_back = read_annotations(accept_image_path, str(Path(accepted["path"]).parent))
    accepted_preds = read_back["predictions"]["annotations"]
    assert sorted(len(ring) for pred in accepted_preds for ring in pred["rings"]) == [3, 4]


def test_stage_proposals_refuses_both_points_and_rings(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0202", size=(640, 480))

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0202"), model_name="sam", polygons=[{
        "subject": "leaf", "conf": 0.9,
        "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]],
        "rings": [[[10.0, 10.0], [50.0, 10.0], [50.0, 40.0]]],
    }])

    assert "error" in res and "exactly one of 'points' or 'rings'" in res["error"]
    assert not (root / "predictions").exists()


def test_stage_proposals_refuses_neither_points_nor_rings(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0203", size=(640, 480))

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0203"), model_name="sam",
                          polygons=[{"subject": "leaf", "conf": 0.9}])

    assert "error" in res and "exactly one of 'points' or 'rings'" in res["error"]


def test_stage_proposals_refuses_a_short_ring_under_rings(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0204", size=(640, 480))

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0204"), model_name="sam", polygons=[{
        "subject": "leaf", "conf": 0.9,
        "rings": [[[10.0, 10.0], [50.0, 10.0]]],
    }])

    assert "error" in res and "three or more points" in res["error"]
    assert not (root / "predictions").exists()


def test_stage_proposals_refuses_a_pixel_vertex_outside_the_image_bounds(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0205", size=(640, 480))

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0205"), model_name="sam", polygons=[{
        "subject": "leaf", "conf": 0.9,
        "rings": [[[10.0, 10.0], [50.0, 10.0], [5000.0, 40.0]]],
    }])

    assert "error" in res and "pixel bounds" in res["error"]
    assert not (root / "predictions").exists()


def test_stage_proposals_refuses_an_out_of_range_coordinate_under_rings(tmp_path: Path) -> None:
    """The pixel-frame counterpart of the normalized-range check: a pixel vertex outside the image
    keeps its refusal at the module's own default image size, not only a specially tiny one."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0206")

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0206"), model_name="sam", polygons=[{
        "subject": "leaf", "conf": 0.9,
        "rings": [[[-50.0, -50.0], [50.0, -50.0], [50.0, 40.0]]],
    }])

    assert "error" in res and "pixel bounds" in res["error"]
    assert not (root / "predictions").exists()


def test_stage_proposals_refuses_a_normalized_ring_handed_under_rings(tmp_path: Path) -> None:
    """A ring of normalized [0,1] coordinates handed under the pixel-frame key must refuse at the
    module's own default image size: it never spans a real pixel, unlike a mask contour."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0207")

    res = stage_proposals(tmp_path, _img_path(root, date, "IMG_0207"), model_name="sam", polygons=[{
        "subject": "leaf", "conf": 0.9,
        "rings": [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4]]],
    }])

    assert "error" in res and "span under a pixel" in res["error"]
    assert not (root / "predictions").exists()


# ── Staged documents are written once ──────────────────────────────────────

_BOX = [{"subject": "bud", "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]


def test_stage_proposals_refuses_a_second_stage_of_the_same_stem(tmp_path: Path) -> None:
    """A staged document a reviewer may already have judged is never rewritten: the second stage
    answers an error naming the document, and the first document's bytes are unchanged."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001", size=(640, 480))
    image_path = _img_path(root, date, "IMG_0001")
    first = stage_proposals(tmp_path, image_path, model_name="claude", boxes=_BOX)
    written = Path(first["path"]).read_bytes()

    res = stage_proposals(tmp_path, image_path, model_name="claude",
                          boxes=[{**_BOX[0], "conf": 0.3}])

    assert "already exists" in res["error"]
    assert str(Path(first["path"]).parent) in res["error"]
    assert Path(first["path"]).read_bytes() == written


def test_stage_proposals_admits_two_stems_staged_in_turn(tmp_path: Path) -> None:
    """Each staged image is its own bucket, so two stems staged in turn under one model name
    both land."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "IMG_0001", size=(640, 480))
    _image(root, date, "IMG_0002", size=(640, 480))

    first = stage_proposals(tmp_path, _img_path(root, date, "IMG_0001"), model_name="claude", boxes=_BOX)
    second = stage_proposals(tmp_path, _img_path(root, date, "IMG_0002"), model_name="claude", boxes=_BOX)

    assert "error" not in first and "error" not in second
    assert first["path"] == str(_staged(root, "claude", date, "IMG_0001"))
    assert second["path"] == str(_staged(root, "claude", date, "IMG_0002"))


def test_stage_proposals_refuses_a_reserved_stem_with_an_error_dict(tmp_path: Path) -> None:
    """An image whose stem names a bucket's own record can never become a per-image prediction
    document, and the audited door says so in its answer rather than raising."""
    root = tmp_path / "proj"
    date = "2026-02-11"
    _image(root, date, "bucket", size=(640, 480))
    res = stage_proposals(tmp_path, _img_path(root, date, "bucket"), model_name="claude", boxes=_BOX)
    assert "a prediction bucket's own record" in res["error"]
    assert not any((_bucket(root, "claude", date) / "bucket").glob("*.json"))



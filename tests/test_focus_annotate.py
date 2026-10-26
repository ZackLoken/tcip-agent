"""Tests for the focus_human_attention MCP tool (drives the live Annotate tab).

The tool resolves the first image holding the subject and the mode its labels imply, then posts an
``annotate_focus`` event. With no GUI running, delivery is a soft miss (``delivered=False``),
and the resolution (index + mode) is what these tests check.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Point, Polygon
from tcip_mcp.dataset_layout import image_dir, label_key
from tcip_mcp.tools.gui_tools import focus_human_attention

if TYPE_CHECKING:
    from tcip_mcp.workspace import BoundProject


def _scene(root: Path, date: str, images: list[str]) -> None:
    idir = image_dir(root, date)
    idir.mkdir(parents=True, exist_ok=True)
    for name in images:
        (idir / name).write_bytes(
            b"x"
        )  # the tool lists by suffix, never opens


def _label(
    root: Path, subject: str, date: str, task: str, stem: str, count: int
) -> None:
    """Write the per-image label document, holding ``count`` shapes of ``subject``.

    ``count`` 0 writes a present empty document, else ``count`` shapes of ``subject`` with the
    geometry ``task`` names: ``segment`` a polygon, ``point`` a point, any other task a box.
    """
    geoms = {
        "segment": Polygon([[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]]),
        "point": Point(10.0, 10.0),
    }
    anns = []
    for _ in range(count):
        geom = geoms.get(task, BBox(10.0, 10.0, 20.0, 20.0))
        anns.append(Annotation(subject=subject, geometry=geom))
    json_io.write_label_document(
        label_key(root, date, stem), anns, 100, 100, keep_empty=True
    )


def _unreadable(root: Path, date: str, stem: str) -> None:
    """Write an empty ``bush`` label document for ``stem``; overwrite it with ``{not json``."""
    from tests._record_damage_fixtures import damage_record

    _label(root, "bush", date, "segment", stem, 0)
    damage_record(label_key(root, date, stem), b"{not json")


def test_focus_annotate_lands_on_first_annotated_polygon_frame(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(5)]  # 0000..0004
    _scene(root, date, imgs)
    # bush polygons (segment) on the 3rd and 4th image only.
    _label(root, "bush", date, "segment", "IMG_0002", 1)
    _label(root, "bush", date, "segment", "IMG_0003", 1)

    res = focus_human_attention(bound, str(root), "bush", date)

    assert "error" not in res
    assert res["image_index"] == 2  # first non-empty label
    assert res["image"] == "IMG_0002.JPG"
    assert res["mode"] == "polygon"  # inferred from segment labels
    assert (
        res["subject"] == "bush"
    )  # the name-based rendering identity the tool surfaces
    assert res["n_images"] == 5
    assert res["n_holding_subject"] == 2
    # Delivery depends on whether a GUI backend happens to be listening; the resolution
    # above is what the tool guarantees.
    assert isinstance(res["delivered"], bool)


def test_a_named_bucket_sets_the_mode_from_its_own_document_not_the_label(
    tmp_path: Path,
    bound: BoundProject,
) -> None:
    """A named bucket sets the focus mode from its own document, not the label.

    The frame's ground truth is a box and the bucket's proposal for it a polygon: with no bucket
    named the focus resolves to box mode, and with the bucket named to polygon mode.
    """
    import pytest

    pytest.importorskip("torch")
    from tests._chain_fixtures import published
    from tests._producer_fixtures import write_image

    root, date = tmp_path, "2026-03-02"
    frame = write_image(
        image_dir(root, date) / "IMG_0000.jpg", (100, 100), (90, 110, 70)
    )
    _label(root, "bud", date, "detect", "IMG_0000", 1)
    published(
        root,
        "preds",
        [
            {
                "image": str(frame),
                "width": 100,
                "height": 100,
                "boxes": [[10.0, 10.0, 20.0, 20.0]],
                "scores": [0.9],
                "labels": [1],
                "cap": 2,
                "masks": [
                    {"segmentation": [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0]]}
                ],
            }
        ],
        scope={"subject": "bud"},
    )

    assert (
        focus_human_attention(bound, str(root), "bud", date)["mode"] == "box"
    )
    assert (
        focus_human_attention(bound, str(root), "bud", date, bucket="preds")[
            "mode"
        ]
        == "polygon"
    )


def test_focus_annotate_scopes_annotated_to_the_requested_subject(
    tmp_path: Path, bound: BoundProject
) -> None:
    # IMG_0001 is labeled only for 'leaf' and IMG_0002 only for 'bud': focusing on 'leaf' lands on
    # IMG_0001 in polygon mode and counts only leaf's frame, so 'annotated' is scoped by subject.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bud", date, "detect", "IMG_0002", 1)
    _label(root, "leaf", date, "segment", "IMG_0001", 1)

    res = focus_human_attention(bound, str(root), "leaf", date)
    assert res["image_index"] == 1  # leaf's frame, not bud's
    assert (
        res["mode"] == "polygon"
    )  # from leaf's polygon geometry on that frame
    assert res["subject"] == "leaf"
    assert res["n_holding_subject"] == 1  # only leaf's frame counts for leaf


def test_focus_annotate_mode_follows_the_explicit_index_not_the_first_frame(
    tmp_path: Path,
    bound: BoundProject,
) -> None:
    # IMG_0002 has polygons, IMG_0007 has only boxes. Asking for index 7 with mode=None must
    # infer 'box' from frame 7, not 'polygon' from the first annotated frame.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(8)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0002", 1)
    _label(root, "bush", date, "detect", "IMG_0007", 1)

    res = focus_human_attention(bound, str(root), "bush", date, image_index=7)
    assert res["image_index"] == 7
    assert (
        res["mode"] == "box"
    )  # from frame 7's detect label, not frame 2's segment
    assert res["subject"] == "bush"


def test_focus_annotate_index_matches_frontend_listing_ignoring_non_files(
    tmp_path: Path,
    bound: BoundProject,
) -> None:
    # A directory named like an image must not shift the index (the frontend's image_list uses
    # is_file()); the tool must match that so it doesn't land one frame off.
    root = tmp_path / "proj"
    date = "2026-03-02"
    idir = image_dir(root, date)
    idir.mkdir(parents=True)
    (idir / "IMG_0000.JPG").write_bytes(b"x")
    (
        idir / "IMG_0001.png"
    ).mkdir()  # a directory with an image suffix, must be ignored
    (idir / "IMG_0002.JPG").write_bytes(b"x")
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention(bound, str(root), "bush", date)
    assert res["n_images"] == 2  # the directory is not counted
    assert res["image"] == "IMG_0002.JPG"
    assert res["image_index"] == 1  # index into [IMG_0000, IMG_0002]


def test_focus_annotate_infers_box_mode_from_detect_labels(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bud", date, "detect", "IMG_0001", 1)

    res = focus_human_attention(bound, str(root), "bud", date)
    assert res["image_index"] == 1
    assert res["mode"] == "box"


def test_focus_annotate_sends_a_point_only_frame_in_point_mode(
    tmp_path: Path, bound: BoundProject
) -> None:
    """A frame carrying point annotations of the subject resolves to point mode."""
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bud", date, "point", "IMG_0001", 2)

    res = focus_human_attention(bound, str(root), "bud", date)
    assert "error" not in res
    assert res["image_index"] == 1
    assert res["mode"] == "point"
    assert res["n_holding_subject"] == 1


def test_focus_annotate_accepts_an_explicit_point_mode(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention(bound, str(root), "bud", date, mode="point")
    assert "error" not in res
    assert res["mode"] == "point"  # honored over the frame's own box geometry


def test_focus_annotate_still_rejects_a_mode_the_gui_has_no_tool_for(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention(bound, str(root), "bud", date, mode="lasso")
    assert "error" in res and "lasso" in res["error"]


def test_focus_annotate_rejects_a_mode_the_gui_has_no_tool_for_naming_the_real_vocabulary(
    tmp_path: Path,
    bound: BoundProject,
) -> None:
    from tcip_mcp.web_client import ANNOTATE_MODES

    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention(bound, str(root), "bud", date, mode="lasso")
    assert "error" in res
    for name in ANNOTATE_MODES:
        assert name in res["error"]


def test_focus_annotate_empty_label_is_not_a_focus_target(
    tmp_path: Path, bound: BoundProject
) -> None:
    # An empty label file holds no subject (nothing to show); skip it.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(
        root, "bush", date, "segment", "IMG_0000", 0
    )  # empty negative ({"annotations": []})
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention(bound, str(root), "bush", date)
    assert res["image_index"] == 2  # skipped the empty-negative frame 0
    assert res["n_holding_subject"] == 1


def test_focus_annotate_explicit_mode_and_index_override(
    tmp_path: Path, bound: BoundProject
) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(4)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0003", 1)

    res = focus_human_attention(
        bound, str(root), "bush", date, mode="box", image_index=1
    )
    assert res["image_index"] == 1  # explicit override
    assert res["mode"] == "box"  # explicit override


def test_focus_annotate_no_images(tmp_path: Path, bound: BoundProject) -> None:
    root = tmp_path / "proj"
    image_dir(root, "2026-03-02").mkdir(parents=True)
    res = focus_human_attention(bound, str(root), "bush", "2026-03-02")
    assert "error" in res


def test_focus_annotate_navigates_past_an_unreadable_label_on_another_frame(
    tmp_path: Path,
    bound: BoundProject,
) -> None:
    """The focus tool navigates past an unreadable label on another frame, naming it.

    With an unreadable label on the first frame, the tool lands on the frame holding the
    subject and names the first frame in the result's ``unreadable`` list.
    """
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _unreadable(root, date, "IMG_0000")
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention(bound, str(root), "bush", date)
    assert "error" not in res
    assert res["image"] == "IMG_0002.JPG"
    assert res["unreadable"] == ["IMG_0000.JPG"]


def test_focus_annotate_refuses_when_the_landed_frame_itself_is_unreadable(
    tmp_path: Path,
    bound: BoundProject,
) -> None:
    """An unreadable label on the requested frame answers an error naming the document."""
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _unreadable(root, date, "IMG_0000")
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention(bound, str(root), "bush", date, image_index=0)
    assert "error" in res
    assert "IMG_0000" in res["error"]

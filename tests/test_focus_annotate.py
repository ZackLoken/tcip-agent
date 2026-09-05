"""Tests for the focus_human_attention(tab='annotate') MCP tool (drives the live Annotate tab).

The tool resolves the first *annotated* image and the mode its labels imply, then posts an
``annotate_focus`` event. With no GUI running, delivery is a soft miss (``delivered=False``),
but the resolution (index + mode) must still be correct: that's what these pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Point, Polygon
from tcip_mcp.dataset_layout import annotation_dir, image_dir
from tcip_mcp.tools.gui_tools import focus_human_attention

from tests.test_canvas_liveview import _mint_binding


@pytest.fixture(autouse=True)
def _matching_canvas_binding(tmp_path: Path) -> None:
    """Every call here drives ``tmp_path / "proj"``; mint the GUI binding it now requires.

    None of these tests are about the live-GUI binding rail, only about frame resolution, so
    they mint a real matching binding through the store seam rather than exercising the refusal
    ``focus_human_attention`` now carries.
    """
    _mint_binding(tmp_path / "proj")


def _scene(root: Path, date: str, images: list[str]) -> None:
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    for name in images:
        (idir / name).write_bytes(b"x")  # the tool lists by suffix, never opens


def _label(root: Path, subject: str, date: str, task: str, stem: str, count: int) -> None:
    # Write the one per-image JSON label (all subjects, name-based). count==0 writes a present
    # {"annotations": []} (confirmed negative); count>0 writes `count` shapes of `subject`, with the
    # geometry `task` names ('segment' -> polygon, 'point' -> point, else box), so focus_human_attention infers the
    # mode from the frame's own geometry.
    d = Path(annotation_dir(root, date))
    geoms = {"segment": Polygon([[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]]),
             "point": Point(10.0, 10.0)}
    anns = []
    for _ in range(count):
        geom = geoms.get(task, BBox(10.0, 10.0, 20.0, 20.0))
        anns.append(Annotation(subject=subject, geometry=geom))
    json_io.write_annotations(str(d / f"{stem}.json"), anns, 100, 100, keep_empty=True)


def test_focus_annotate_lands_on_first_annotated_polygon_frame(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(5)]  # 0000..0004
    _scene(root, date, imgs)
    # bush polygons (segment) on the 3rd and 4th image only.
    _label(root, "bush", date, "segment", "IMG_0002", 1)
    _label(root, "bush", date, "segment", "IMG_0003", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date)

    assert "error" not in res
    assert res["image_index"] == 2  # first non-empty label
    assert res["image"] == "IMG_0002.JPG"
    assert res["mode"] == "polygon"  # inferred from segment labels
    assert res["subject"] == "bush"  # the name-based rendering identity the tool surfaces
    assert res["n_images"] == 5
    assert res["n_annotated"] == 2
    # Delivery depends on whether a GUI backend happens to be listening; the resolution
    # above is what the tool guarantees.
    assert isinstance(res["delivered"], bool)


def test_focus_annotate_scopes_annotated_to_the_requested_subject(tmp_path: Path) -> None:
    # Name-based schema: one file per image holds every subject. IMG_0001 is labeled only for 'leaf',
    # IMG_0002 only for 'bud'. Focusing on 'leaf' must land on IMG_0001 (its subject's frame) in
    # polygon mode and count only leaf's frame: the tool scopes 'annotated' to the requested subject
    # (the name-based replacement for the old per-frame active-class resolution).
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bud", date, "detect", "IMG_0002", 1)
    _label(root, "leaf", date, "segment", "IMG_0001", 1)

    res = focus_human_attention("annotate", str(root), str(root), "leaf", date)
    assert res["image_index"] == 1  # leaf's frame, not bud's
    assert res["mode"] == "polygon"  # from leaf's polygon geometry on that frame
    assert res["subject"] == "leaf"
    assert res["n_annotated"] == 1  # only leaf's frame counts for leaf


def test_focus_annotate_mode_follows_the_explicit_index_not_the_first_frame(tmp_path: Path) -> None:
    # IMG_0002 has polygons, IMG_0007 has only boxes. Asking for index 7 with mode=None must
    # infer 'box' from frame 7, not 'polygon' from the first annotated frame.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(8)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0002", 1)
    _label(root, "bush", date, "detect", "IMG_0007", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date, image_index=7)
    assert res["image_index"] == 7
    assert res["mode"] == "box"  # from frame 7's detect label, not frame 2's segment
    assert res["subject"] == "bush"


def test_focus_annotate_index_matches_frontend_listing_ignoring_non_files(tmp_path: Path) -> None:
    # A directory named like an image must not shift the index (the frontend's image_list uses
    # is_file()); the tool must match that so it doesn't land one frame off.
    root = tmp_path / "proj"
    date = "2026-03-02"
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True)
    (idir / "IMG_0000.JPG").write_bytes(b"x")
    (idir / "IMG_0001.png").mkdir()  # a directory with an image suffix, must be ignored
    (idir / "IMG_0002.JPG").write_bytes(b"x")
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date)
    assert res["n_images"] == 2  # the directory is not counted
    assert res["image"] == "IMG_0002.JPG"
    assert res["image_index"] == 1  # index into [IMG_0000, IMG_0002]


def test_focus_annotate_infers_box_mode_from_detect_labels(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bud", date, "detect", "IMG_0001", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bud", date)
    assert res["image_index"] == 1
    assert res["mode"] == "box"


def test_focus_annotate_sends_a_point_only_frame_in_point_mode(tmp_path: Path) -> None:
    """A frame whose only geometry for the subject is a point is edited in point mode.

    ``_subject_task`` already answers "point"; delivering that frame in box mode hands the human a
    tool that cannot touch what is on the canvas (the GUI's Mode union carries "point").
    """
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bud", date, "point", "IMG_0001", 2)

    res = focus_human_attention("annotate", str(root), str(root), "bud", date)
    assert "error" not in res
    assert res["image_index"] == 1
    assert res["mode"] == "point"
    assert res["n_annotated"] == 1


def test_focus_annotate_accepts_an_explicit_point_mode(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bud", date, mode="point")
    assert "error" not in res
    assert res["mode"] == "point"  # honored over the frame's own box geometry


def test_focus_annotate_still_rejects_a_mode_the_gui_has_no_tool_for(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bud", date, mode="lasso")
    assert "error" in res and "lasso" in res["error"]


def test_focus_annotate_rejects_a_mode_the_gui_has_no_tool_for_naming_the_real_vocabulary(
    tmp_path: Path,
) -> None:
    from tcip_mcp.web_client import ANNOTATE_MODES

    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bud", date, mode="lasso")
    assert "error" in res
    for name in ANNOTATE_MODES:
        assert name in res["error"]


def test_focus_annotate_accepts_the_map_mode(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    _scene(root, date, ["IMG_0000.JPG"])
    _label(root, "bud", date, "detect", "IMG_0000", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bud", date, mode="map")
    assert "error" not in res
    assert res["mode"] == "map"


def test_focus_annotate_empty_label_is_not_a_focus_target(tmp_path: Path) -> None:
    # An empty label file is a confirmed negative (nothing to show); skip it.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0000", 0)  # empty negative ({"annotations": []})
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date)
    assert res["image_index"] == 2  # skipped the empty-negative frame 0
    assert res["n_annotated"] == 1


def test_focus_annotate_explicit_mode_and_index_override(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(4)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0003", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date, mode="box", image_index=1)
    assert res["image_index"] == 1  # explicit override
    assert res["mode"] == "box"  # explicit override


def test_focus_annotate_no_images(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (Path(image_dir(root, "2026-03-02"))).mkdir(parents=True)
    res = focus_human_attention("annotate", str(root), str(root), "bush", "2026-03-02")
    assert "error" in res


def test_focus_annotate_navigates_past_an_unreadable_label_on_another_frame(tmp_path: Path) -> None:
    """A present, unreadable label on a frame the tool is not landing on does not close the
    navigation surface for the date; it is named in the result's ``unreadable`` list instead."""
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    d = Path(annotation_dir(root, date))
    d.mkdir(parents=True, exist_ok=True)
    bad = d / "IMG_0000.json"
    bad.write_bytes(b"{not json")
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date)
    assert "error" not in res
    assert res["image"] == "IMG_0002.JPG"
    assert res["unreadable"] == ["IMG_0000.JPG"]


def test_focus_annotate_refuses_when_the_landed_frame_itself_is_unreadable(tmp_path: Path) -> None:
    """A present, unreadable label naming the requested frame is an error naming the file, never
    a raise through the tool boundary."""
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    d = Path(annotation_dir(root, date))
    d.mkdir(parents=True, exist_ok=True)
    bad = d / "IMG_0000.json"
    bad.write_bytes(b"{not json")
    _label(root, "bush", date, "segment", "IMG_0002", 1)

    res = focus_human_attention("annotate", str(root), str(root), "bush", date, image_index=0)
    assert "error" in res
    assert str(bad) in res["error"]

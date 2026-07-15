"""Tests for the focus_annotate MCP tool (drives the live Annotate tab).

The tool resolves the first *annotated* image and the mode its labels imply, then posts an
``annotate_focus`` event. With no GUI running, delivery is a soft miss (``delivered=False``),
but the resolution (index + mode) must still be correct — that's what these pin.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import BBox, Polygon
from tcip_mcp.dataset_layout import annotation_dir, image_dir
from tcip_mcp.tools.annotation_tools import focus_annotate


def _scene(root: Path, date: str, images: list[str]) -> None:
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    for name in images:
        (idir / name).write_bytes(b"x")  # the tool lists by suffix, never opens


def _label(root: Path, trait: str, date: str, task: str, stem: str, classes: list[int]) -> None:
    # Write a per-image JSON label. An empty `classes` writes a present {"objects": []}
    # (confirmed negative); a non-empty list writes one shape per class id.
    d = Path(annotation_dir(root, trait, date, task))
    path = str(d / f"{stem}.json")
    if task == "segment":
        polys = [Polygon([(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)], c) for c in classes]
        json_io.write_segment(path, polys, 100, 100, keep_empty=True)
    else:
        boxes = [BBox(10.0, 10.0, 20.0, 20.0, c) for c in classes]
        json_io.write_detect(path, boxes, 100, 100, keep_empty=True)


def test_focus_annotate_lands_on_first_annotated_polygon_frame(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(5)]  # 0000..0004
    _scene(root, date, imgs)
    # bush polygons (segment) on the 3rd and 4th image only.
    _label(root, "bush", date, "segment", "IMG_0002", [0])
    _label(root, "bush", date, "segment", "IMG_0003", [0])

    res = focus_annotate(str(root), str(root), "bush", date)

    assert "error" not in res
    assert res["image_index"] == 2  # first non-empty label
    assert res["image"] == "IMG_0002.JPG"
    assert res["mode"] == "polygon"  # inferred from segment labels
    assert res["active_class"] == 0  # first class id on the focused frame
    assert res["n_images"] == 5
    assert res["n_annotated"] == 2
    # Delivery depends on whether a GUI backend happens to be listening; the resolution
    # above is what the tool guarantees.
    assert isinstance(res["delivered"], bool)


def test_focus_annotate_resolves_active_class_from_the_frame(tmp_path: Path) -> None:
    # Labels on the focused frame are class 1 — the canvas only renders the active class, so
    # the tool must surface active_class=1 (else the frame shows blank even in the right mode).
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0001", [1])

    res = focus_annotate(str(root), str(root), "bush", date)
    assert res["image_index"] == 1
    assert res["active_class"] == 1


def test_focus_annotate_mode_follows_the_explicit_index_not_the_first_frame(tmp_path: Path) -> None:
    # IMG_0002 has polygons, IMG_0007 has only boxes. Asking for index 7 with mode=None must
    # infer 'box' from frame 7 — NOT 'polygon' from the first annotated frame.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(8)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0002", [0])
    _label(root, "bush", date, "detect", "IMG_0007", [3])

    res = focus_annotate(str(root), str(root), "bush", date, image_index=7)
    assert res["image_index"] == 7
    assert res["mode"] == "box"  # from frame 7's detect label, not frame 2's segment
    assert res["active_class"] == 3


def test_focus_annotate_index_matches_frontend_listing_ignoring_non_files(tmp_path: Path) -> None:
    # A directory named like an image must NOT shift the index (the frontend's image_list uses
    # is_file()); the tool must match that so it doesn't land one frame off.
    root = tmp_path / "proj"
    date = "2026-03-02"
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True)
    (idir / "IMG_0000.JPG").write_bytes(b"x")
    (idir / "IMG_0001.png").mkdir()  # a directory with an image suffix — must be ignored
    (idir / "IMG_0002.JPG").write_bytes(b"x")
    _label(root, "bush", date, "segment", "IMG_0002", [0])

    res = focus_annotate(str(root), str(root), "bush", date)
    assert res["n_images"] == 2  # the directory is not counted
    assert res["image"] == "IMG_0002.JPG"
    assert res["image_index"] == 1  # index into [IMG_0000, IMG_0002]


def test_focus_annotate_infers_box_mode_from_detect_labels(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "catkin", date, "detect", "IMG_0001", [0])

    res = focus_annotate(str(root), str(root), "catkin", date)
    assert res["image_index"] == 1
    assert res["mode"] == "box"


def test_focus_annotate_empty_label_is_not_a_focus_target(tmp_path: Path) -> None:
    # An empty label file is a confirmed negative (nothing to show) — skip it.
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(3)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0000", [])  # empty negative ({"objects": []})
    _label(root, "bush", date, "segment", "IMG_0002", [0])

    res = focus_annotate(str(root), str(root), "bush", date)
    assert res["image_index"] == 2  # skipped the empty-negative frame 0
    assert res["n_annotated"] == 1


def test_focus_annotate_explicit_mode_and_index_override(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-03-02"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(4)]
    _scene(root, date, imgs)
    _label(root, "bush", date, "segment", "IMG_0003", [0])

    res = focus_annotate(str(root), str(root), "bush", date, mode="box", image_index=1)
    assert res["image_index"] == 1  # explicit override
    assert res["mode"] == "box"  # explicit override


def test_focus_annotate_no_images(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (Path(image_dir(root, "2026-03-02"))).mkdir(parents=True)
    res = focus_annotate(str(root), str(root), "bush", "2026-03-02")
    assert "error" in res

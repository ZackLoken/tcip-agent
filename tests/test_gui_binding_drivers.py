"""The live-GUI binding rail, shared: ``focus_human_attention`` and ``push_panel_event`` now
refuse under the same ``canvas_open_binding`` mismatch ``capture_live_canvas`` already refused
under, through the one predicate ``web_client.gui_binding_matches`` all three read.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import annotation_dir, image_dir
from tcip_mcp.tools.gui_tools import focus_human_attention, push_panel_event

from tests.test_canvas_liveview import _mint_binding


def _scene(root: Path, date: str) -> None:
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    (idir / "IMG_0001.JPG").write_bytes(b"x")
    adir = Path(annotation_dir(root, date))
    adir.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        str(adir / "IMG_0001.json"),
        [Annotation(subject="catkin", geometry=BBox(10.0, 10.0, 20.0, 20.0))],
        100, 100,
    )


def test_focus_human_attention_refuses_under_a_foreign_binding(
    tmp_path: Path, monkeypatch, tmp_path_factory,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-02"
    _scene(tmp_path, date)
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name="hazelnut_catkin_valley")

    res = focus_human_attention("annotate", str(tmp_path), str(tmp_path), "catkin", date)

    assert "error" in res
    assert res["bound_project"] == "hazelnut_catkin_valley"
    assert res["compared_root"] == str(tmp_path)


def test_push_panel_event_refuses_under_a_foreign_binding(
    tmp_path: Path, monkeypatch, tmp_path_factory,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name="hazelnut_catkin_valley")

    res = push_panel_event(panel="training", event_type="metrics_update", data={"epoch": 1})

    assert "error" in res
    assert res["delivered"] is False
    assert res["bound_project"] == "hazelnut_catkin_valley"
    assert res["panel"] == "training"


def test_focus_human_attention_delivers_under_a_matching_binding(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    date = "2026-03-02"
    _scene(tmp_path, date)
    _mint_binding(tmp_path)

    res = focus_human_attention("annotate", str(tmp_path), str(tmp_path), "catkin", date)

    assert "error" not in res, res
    assert res["image"] == "IMG_0001.JPG"


def test_push_panel_event_delivers_under_a_matching_binding(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _mint_binding(tmp_path)

    res = push_panel_event(panel="training", event_type="metrics_update", data={"epoch": 1})

    assert "bound_project" not in res  # not refused by the binding rail
    assert res["panel"] == "training"

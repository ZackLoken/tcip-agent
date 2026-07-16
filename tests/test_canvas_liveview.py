"""Live canvas view: the GUI's canvas-state push + the agent's visualize_canvas render.

Covers the two-file scheme (meta heartbeats never touch the geometry blob; geometry is valid
only when its (image_path, tab) identity matches the meta), the display-resolved shape renderer
(crop-to-viewport math, two-pass draw, malformed-shape tolerance), and the MCP tool end
(missing state, identity-stale shapes, ages, tag/creator counts).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tcip_web.app import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _payload(root: Path, image_path: str, shapes=None, **over) -> dict:
    body = {
        "schema_version": 1,
        "project_root": str(root),
        "tab": "annotate",
        "image_path": image_path,
        "image": Path(image_path).name,
        "img_width": 200,
        "img_height": 100,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100, "scale": 1.0},
        "mode": "polygon",
        "user": "zack",
        "classes": [{"id": 0, "name": "catkin", "color": "#FF0000"}],
        "counts": {"boxes": 0, "polygons": 1, "drawing_points": 0},
        "shapes": shapes,
    }
    body.update(over)
    return body


def _meta(root: Path) -> dict:
    return json.loads((root / ".tcip" / "state" / "canvas_live.json").read_text())


def _shapes_doc(root: Path) -> dict:
    return json.loads((root / ".tcip" / "state" / "canvas_shapes.json").read_text())


SHAPES = [
    {"kind": "polygon", "points": [[10, 10], [60, 10], [60, 60]], "color": "#00FF00",
     "fill": True, "tag": "gt", "created_by": "user:zack"},
    {"kind": "box", "xyxy": [80, 20, 140, 70], "color": "#FF0000", "tag": "gt",
     "created_by": "derived:user:zack"},
    {"kind": "polyline", "points": [[5, 90], [30, 85], [55, 92]], "color": "#FFE7B1",
     "dashed": True, "tag": "in_progress", "label": "drawing"},
]


# ── route: two-file scheme ──────────────────────────────────────────────────

def test_full_push_writes_geometry_and_meta(client, tmp_path):
    r = client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg", shapes=SHAPES))
    assert r.status_code == 200 and r.json()["shapes_written"] is True
    assert len(_shapes_doc(tmp_path)["shapes"]) == 3
    assert _shapes_doc(tmp_path)["image_path"] == "C:/img/a.jpg"
    assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"


def test_heartbeat_updates_meta_without_touching_geometry(client, tmp_path):
    client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg", shapes=SHAPES))
    before = (tmp_path / ".tcip" / "state" / "canvas_shapes.json").stat().st_mtime_ns
    hb = _payload(tmp_path, "C:/img/a.jpg", shapes=None,
                  viewport={"x": 40, "y": 10, "w": 80, "h": 50, "scale": 2.0})
    r = client.post("/api/canvas/state", json=hb)
    assert r.json()["shapes_written"] is False
    assert _meta(tmp_path)["viewport"]["x"] == 40                       # meta moved
    after = (tmp_path / ".tcip" / "state" / "canvas_shapes.json").stat().st_mtime_ns
    assert after == before                                              # geometry blob untouched


def test_push_state_rejects_project_root_outside_image_roots(client, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    r = client.post("/api/canvas/state", json=_payload(outside, "C:/img/a.jpg"))
    assert r.status_code == 403
    assert not (outside / ".tcip" / "state" / "canvas_live.json").exists()


def test_push_state_allows_project_root_inside_image_roots(client, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path))
    r = client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg"))
    assert r.status_code == 200


def test_push_state_does_not_fsync(client, tmp_path, monkeypatch):
    """canvas_live/canvas_shapes are ephemeral — a push must not depend on fsync."""
    import os as _os

    def _boom(*_a, **_kw):
        raise AssertionError("fsync should not be called for canvas state")

    monkeypatch.setattr(_os, "fsync", _boom)
    r = client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg", shapes=SHAPES))
    assert r.status_code == 200
    assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"


def test_heartbeat_for_new_image_invalidates_geometry_by_identity(client, tmp_path):
    client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg", shapes=SHAPES))
    client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/b.jpg", shapes=None))
    # The geometry file still holds a.jpg's shapes, but its identity no longer matches the meta —
    # the reader must treat it as stale (a.jpg's polygons never render under b.jpg).
    assert _shapes_doc(tmp_path)["image_path"] == "C:/img/a.jpg"
    assert _meta(tmp_path)["image_path"] == "C:/img/b.jpg"


# ── renderer: crop math + two-pass draw + tolerance ─────────────────────────

def _make_image(tmp_path: Path) -> str:
    p = tmp_path / "img.jpg"
    Image.new("RGB", (200, 100), (90, 110, 90)).save(p)
    return str(p)


def test_render_crops_to_viewport(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    out = render_canvas_state(img, SHAPES, viewport={"x": 50, "y": 0, "w": 100, "h": 100},
                              output_path=str(tmp_path / "out.jpg"))
    assert Image.open(out).size == (100, 100)   # exactly the visible region


def test_render_full_frame_and_downscale(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    out = render_canvas_state(img, SHAPES, viewport={"x": 50, "y": 0, "w": 100, "h": 100},
                              crop_to_viewport=False, max_edge=100,
                              output_path=str(tmp_path / "full.jpg"))
    assert Image.open(out).size == (100, 50)    # full frame, downscaled to max_edge


def test_render_two_pass_fill_does_not_erase_outlines(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [
        {"kind": "box", "xyxy": [20, 20, 80, 80], "color": "#00FF00"},          # green outline
        {"kind": "box", "xyxy": [10, 10, 90, 90], "color": "#FF0000", "fill": True},  # later red fill over it
    ]
    out = render_canvas_state(img, shapes, output_path=str(tmp_path / "overlap.png"))
    px = Image.open(out).convert("RGB").getpixel((50, 20))  # a point on the green outline
    assert px[1] > px[0]  # outline survives the later overlapping fill (green-dominant)


def test_render_tolerates_malformed_shapes(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    bad = [{"kind": "box"}, {"kind": "polygon", "points": [[1, 1]]}, "junk",
           {"kind": "box", "xyxy": [10, 10, 40, 40], "color": "not-a-color"}]
    out = render_canvas_state(img, bad, output_path=str(tmp_path / "bad.jpg"))
    assert Path(out).is_file()                  # renders what it can, never raises


# ── MCP tool ────────────────────────────────────────────────────────────────

def _write_state(tmp_path: Path, img: str, shapes=SHAPES, *, shapes_image: str | None = None,
                 tab: str = "annotate", shapes_tab: str | None = None) -> None:
    import time
    sd = tmp_path / ".tcip" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "canvas_live.json").write_text(json.dumps({
        "received_at": time.time(), "project_root": str(tmp_path), "tab": tab,
        "image": Path(img).name, "image_path": img,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100}, "user": "zack", "mode": "polygon",
        "classes": [{"id": 0, "name": "catkin", "color": "#FF0000"}],
    }))
    if shapes is not None:
        (sd / "canvas_shapes.json").write_text(json.dumps({
            "image_path": shapes_image or img, "tab": shapes_tab or tab,
            "shapes": shapes, "received_at": time.time(),
        }))


def test_visualize_canvas_missing_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.tools.vision_tools import visualize_canvas
    res = visualize_canvas(refresh=False)
    assert "error" in res


def test_visualize_canvas_renders_pushed_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)

    from tcip_mcp.tools.vision_tools import visualize_canvas
    res = visualize_canvas(refresh=False)
    assert "error" not in res
    assert Path(res["image_path"]).is_file()
    assert res["classes"][0]["name"] == "catkin"
    assert res["shape_counts_by_tag"] == {"gt": 2, "in_progress": 1}
    assert res["shape_counts_by_creator"] == {"user:zack": 1, "derived:user:zack": 1}
    assert res["state_age_seconds"] >= 0
    assert res["shapes_missing"] is False
    assert res["project_root"] == str(tmp_path)


def test_visualize_canvas_identity_stale_shapes_do_not_render(tmp_path, monkeypatch):
    """Geometry left over from a previous image must not render under the current one."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img, shapes_image="C:/img/other.jpg")  # stale identity

    from tcip_mcp.tools.vision_tools import visualize_canvas
    res = visualize_canvas(refresh=False)
    assert "error" not in res
    assert res["shapes_missing"] is True
    assert res["shape_counts_by_tag"] == {}
    assert Path(res["image_path"]).is_file()    # still renders the image + viewport

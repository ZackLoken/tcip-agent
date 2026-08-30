"""Live canvas view: the GUI's canvas-state push + the agent's capture_live_canvas render.

Covers the two-file scheme (meta heartbeats never touch the geometry blob; geometry is valid
only when its (image_path, tab) identity matches the meta), the display-resolved shape renderer
(crop-to-viewport math, two-pass draw, malformed-shape tolerance), and the MCP tool end
(missing state, identity-stale shapes, ages, tag/creator counts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import tcip_store  # noqa: E402
from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key  # noqa: E402
from tcip_web.app import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _payload(root: Path, image_path: str, shapes=None, **over) -> dict:
    body = {
        "project_root": str(root),
        "tab": "annotate",
        "image_path": image_path,
        "image": Path(image_path).name,
        "img_width": 200,
        "img_height": 100,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100, "scale": 1.0},
        "mode": "polygon",
        "user": "breeder",
        "classes": [{"id": 0, "name": "catkin", "color": "#FF0000"}],
        "counts": {"boxes": 0, "polygons": 1, "drawing_points": 0},
        "shapes": shapes,
    }
    body.update(over)
    return body


def _meta(root: Path) -> dict:
    return tcip_store.read(canvas_meta_key(str(root)))


def _shapes_doc(root: Path) -> dict:
    return tcip_store.read(canvas_geometry_key(str(root)))


SHAPES = [
    {"kind": "polygon", "points": [[10, 10], [60, 10], [60, 60]], "color": "#00FF00",
     "fill": True, "tag": "gt", "created_by": "user:breeder"},
    {"kind": "box", "xyxy": [80, 20, 140, 70], "color": "#FF0000", "tag": "gt",
     "created_by": "derived:user:breeder"},
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
    before = tcip_store.read_versioned(canvas_geometry_key(str(tmp_path))).version
    hb = _payload(tmp_path, "C:/img/a.jpg", shapes=None,
                  viewport={"x": 40, "y": 10, "w": 80, "h": 50, "scale": 2.0})
    r = client.post("/api/canvas/state", json=hb)
    assert r.json()["shapes_written"] is False
    assert _meta(tmp_path)["viewport"]["x"] == 40                       # meta moved
    after = tcip_store.read_versioned(canvas_geometry_key(str(tmp_path))).version
    assert after == before                                              # geometry blob untouched


def test_push_state_rejects_project_root_outside_image_roots(client, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    r = client.post("/api/canvas/state", json=_payload(outside, "C:/img/a.jpg"))
    assert r.status_code == 403
    assert not (outside / ".tcip" / "state" / "canvas_live.json").exists()


def test_push_state_allows_project_root_inside_image_roots(client, tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path))
    r = client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg"))
    assert r.status_code == 200


def test_push_state_refuses_a_project_root_that_is_not_the_pinned_platform_root(
    client, tmp_path_factory, monkeypatch,
):
    """A root inside the allowed set but not this process's pinned platform root (the
    autouse fixture pins every test to its own ``tmp_path``): the push must refuse rather than
    write a file ``capture_live_canvas`` would never look at."""
    other = tmp_path_factory.mktemp("other")
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(other))
    r = client.post("/api/canvas/state", json=_payload(other, "C:/img/a.jpg"))
    assert r.status_code == 403
    assert "pinned platform root" in r.json()["detail"]
    assert not (other / ".tcip" / "state" / "canvas_live.json").exists()


def test_push_and_capture_live_canvas_land_on_the_same_file(client, tmp_path):
    """The writer (this route) and the reader (capture_live_canvas) anchor to one file for the
    same session: a push through the real route under the pinned root is read back through the
    real MCP tool, not a store write standing in for either end."""
    img = _make_image(tmp_path)
    r = client.post("/api/canvas/state", json=_payload(tmp_path, img, shapes=SHAPES))
    assert r.status_code == 200

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert res["project_root"] == str(tmp_path)
    assert res["shape_counts_by_tag"] == {"gt": 2, "in_progress": 1}


def test_push_state_does_not_fsync(client, tmp_path, monkeypatch):
    """canvas_live/canvas_shapes are ephemeral: a push must not depend on fsync.

    Bound to the file backend on purpose: canvas records declare durable=False, and this
    backend's own write path calls os.fsync only when a record's descriptor declares itself
    durable, so the patch proves the push's per-record durability choice. A fresh sqlite root
    also fsyncs once, unconditionally, to create its database file before any record is
    written; that is bootstrap infrastructure a root incurs regardless of what it stores, not
    something this push depends on, and it would trip the same patch for an unrelated reason.
    """
    import os as _os

    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())

    def _boom(*_a, **_kw):
        raise AssertionError("fsync should not be called for canvas state")

    monkeypatch.setattr(_os, "fsync", _boom)
    r = client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg", shapes=SHAPES))
    assert r.status_code == 200
    assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"


def test_a_push_waits_for_a_holder_of_the_records_lock_and_then_lands(tmp_path):
    """A push takes the meta record's lock, so it cannot overwrite what a holder is editing.

    The push runs on its own thread while another thread holds the key, and is observed still
    waiting; once the holder lets go the push completes and its document is what the record
    holds. Both halves matter: waiting alone would be a push that never lands.
    """
    import threading

    import tcip_store as ts

    from tcip_web.routes.canvas import CanvasStatePayload, canvas_meta_key, push_canvas_state

    key = canvas_meta_key(str(tmp_path))
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with ts.transaction(key):
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert holding.wait(10)

    payload = CanvasStatePayload(**_payload(tmp_path, "C:/img/a.jpg"))
    pushing = threading.Thread(target=lambda: push_canvas_state(payload))
    pushing.start()
    pushing.join(0.5)
    waiting = pushing.is_alive()

    release.set()
    holder.join(10)
    pushing.join(10)
    assert waiting, "the push wrote while another writer held the record's lock"
    assert not pushing.is_alive()
    assert ts.read(key)["image_path"] == "C:/img/a.jpg"


def test_heartbeat_for_new_image_invalidates_geometry_by_identity(client, tmp_path):
    client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/a.jpg", shapes=SHAPES))
    client.post("/api/canvas/state", json=_payload(tmp_path, "C:/img/b.jpg", shapes=None))
    # The geometry file still holds a.jpg's shapes, but its identity no longer matches the meta:
    # the reader must treat it as stale (a.jpg's polygons never render under b.jpg).
    assert _shapes_doc(tmp_path)["image_path"] == "C:/img/a.jpg"
    assert _meta(tmp_path)["image_path"] == "C:/img/b.jpg"


# ── renderer: origin/scale placement + two-pass draw + tolerance ────────────

def _make_image(tmp_path: Path) -> str:
    p = tmp_path / "img.jpg"
    Image.new("RGB", (200, 100), (90, 110, 90)).save(p)
    return str(p)


def _pixels(image_path: str, region=None, scale: float = 1.0):
    """The pixels a caller reads for the canvas render: a region of the image at ``scale``.

    Stands in for the raster read ``capture_live_canvas`` performs, so a renderer test exercises
    placement and nothing else.
    """
    import numpy as np

    with Image.open(image_path) as im:
        frame = im.convert("RGB")
        if region is not None:
            frame = frame.crop(region)
        if scale != 1.0:
            frame = frame.resize((max(1, round(frame.width * scale)),
                                  max(1, round(frame.height * scale))), Image.LANCZOS)
        return np.asarray(frame)


def test_render_places_a_shape_at_its_offset_inside_a_cropped_region(tmp_path):
    """A shape's native coordinate lands where the crop origin puts it, not where it sat in the
    full frame: the placement the viewport read replaced the renderer's own crop with."""
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [{"kind": "point", "points": [[100, 50]], "color": "#FF0000"}]
    out = render_canvas_state(_pixels(img, region=(50, 0, 150, 100)), shapes,
                              origin=(50, 0), scale=1.0,
                              output_path=str(tmp_path / "crop.png"))
    px = Image.open(out).convert("RGB")
    assert px.size == (100, 100)                        # exactly the region handed in

    def r_at(xy):
        return px.getpixel(xy)[0] - px.getpixel(xy)[1]

    assert r_at((50, 50)) > 40      # native x=100 minus origin x=50
    assert r_at((95, 50)) < 10      # nothing near where the unshifted coordinate would have hit


def test_render_scales_a_shape_with_the_pixels_it_is_drawn_on(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [{"kind": "point", "points": [[100, 50]], "color": "#FF0000"}]
    out = render_canvas_state(_pixels(img, scale=0.5), shapes, origin=(0, 0), scale=0.5,
                              output_path=str(tmp_path / "half.png"))
    px = Image.open(out).convert("RGB")
    assert px.size == (100, 50)

    def r_at(xy):
        return px.getpixel(xy)[0] - px.getpixel(xy)[1]

    assert r_at((50, 25)) > 40      # native (100, 50) at half resolution
    assert r_at((90, 25)) < 10


def test_render_two_pass_fill_does_not_erase_outlines(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [
        {"kind": "box", "xyxy": [20, 20, 80, 80], "color": "#00FF00"},          # green outline
        {"kind": "box", "xyxy": [10, 10, 90, 90], "color": "#FF0000", "fill": True},  # later red fill over it
    ]
    out = render_canvas_state(_pixels(img), shapes, origin=(0, 0), scale=1.0,
                              output_path=str(tmp_path / "overlap.png"))
    px = Image.open(out).convert("RGB").getpixel((50, 20))  # a point on the green outline
    assert px[1] > px[0]  # outline survives the later overlapping fill (green-dominant)


def test_render_draws_a_point_shape_and_never_widens_it_to_a_box(tmp_path):
    """A pushed point must reach the agent's view as a mark: not be dropped, not become a box.

    The GUI can now author point annotations, so a shape kind the renderer skips would show the
    agent a canvas with fewer annotations than the annotator sees. Widening it into a box is the
    other failure: that invents an extent the annotation does not claim (see state.Point/bbox_of).
    """
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [{"kind": "point", "points": [[100, 50]], "color": "#FF0000", "label": "tip"}]
    out = render_canvas_state(_pixels(img), shapes, origin=(0, 0), scale=1.0,
                              output_path=str(tmp_path / "point.png"))
    px = Image.open(out).convert("RGB")

    def r_at(xy):  # red-over-green dominance at one pixel
        return px.getpixel(xy)[0] - px.getpixel(xy)[1]

    assert r_at((100, 50)) > 40          # the core sits on the coordinate
    assert r_at((100, 42)) > 40          # a radial tick above it (the mark's reticle)
    assert r_at((140, 50)) < 10          # nothing 40px away: no box, no fill, no outline


def test_render_tolerates_malformed_shapes(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    bad = [{"kind": "box"}, {"kind": "polygon", "points": [[1, 1]]}, "junk",
           {"kind": "box", "xyxy": [10, 10, 40, 40], "color": "not-a-color"}]
    out = render_canvas_state(_pixels(img), bad, origin=(0, 0), scale=1.0,
                              output_path=str(tmp_path / "bad.jpg"))
    assert Path(out).is_file()                  # renders what it can, never raises


# ── MCP tool ────────────────────────────────────────────────────────────────

def _write_state(tmp_path: Path, img: str, shapes=SHAPES, *, shapes_image: str | None = None,
                 tab: str = "annotate", shapes_tab: str | None = None) -> None:
    import time
    root = str(tmp_path)
    if shapes is not None:
        tcip_store.replace(canvas_geometry_key(root), {
            "image_path": shapes_image or img, "tab": shapes_tab or tab,
            "shapes": shapes, "received_at": time.time(),
        })
    tcip_store.replace(canvas_meta_key(root), {
        "received_at": time.time(), "project_root": root, "tab": tab,
        "image": Path(img).name, "image_path": img,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100}, "user": "breeder", "mode": "polygon",
        "classes": [{"id": 0, "name": "catkin", "color": "#FF0000"}],
    })


def test_capture_live_canvas_missing_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res


def test_capture_live_canvas_renders_pushed_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert Path(res["image_path"]).is_file()
    assert res["classes"][0]["name"] == "catkin"
    assert res["shape_counts_by_tag"] == {"gt": 2, "in_progress": 1}
    assert res["shape_counts_by_creator"] == {"user:breeder": 1, "derived:user:breeder": 1}
    assert res["state_age_seconds"] >= 0
    assert res["shapes_missing"] is False
    assert res["project_root"] == str(tmp_path)


def test_capture_live_canvas_renders_exactly_the_viewport_region(tmp_path, monkeypatch):
    """The tool reads the visible rectangle and renders that, so the artifact is the region the
    human sees rather than the whole frame."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)
    live = tcip_store.read(canvas_meta_key(str(tmp_path)))
    live["viewport"] = {"x": 50, "y": 0, "w": 100, "h": 100}
    tcip_store.replace(canvas_meta_key(str(tmp_path)), live)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert res["cropped_to_viewport"] is True
    assert Image.open(res["image_path"]).size == (100, 100)


def test_capture_live_canvas_full_frame_downscales_to_max_edge(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False, crop_to_viewport=False, max_edge=100)
    assert res["cropped_to_viewport"] is False
    assert Image.open(res["image_path"]).size == (100, 50)


def test_capture_live_canvas_reads_a_multiband_raster_without_writing_a_preview(
    tmp_path, monkeypatch,
):
    """A raster PIL has no true-color mode for is composited in memory for the render: the capture
    path writes no throwaway preview file beside the artifact."""
    import numpy as np
    import tifffile

    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(3)
    src = images / "capture.tif"
    tifffile.imwrite(str(src), rng.integers(0, 4096, size=(100, 200, 6)).astype(np.uint16))
    _write_state(tmp_path, str(src))

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert Image.open(res["image_path"]).mode == "RGB"
    assert not (tmp_path / ".tcip" / "artifacts" / "viz" / "_band_previews").exists()


def test_capture_live_canvas_identity_stale_shapes_do_not_render(tmp_path, monkeypatch):
    """Geometry left over from a previous image must not render under the current one."""
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img, shapes_image="C:/img/other.jpg")  # stale identity

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert res["shapes_missing"] is True
    assert res["shape_counts_by_tag"] == {}
    assert Path(res["image_path"]).is_file()    # still renders the image + viewport

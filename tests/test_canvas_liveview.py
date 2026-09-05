"""Live canvas view: the GUI's canvas-state push + the agent's capture_live_canvas render.

Covers the two-file scheme (meta heartbeats never touch the geometry blob; geometry is valid
only when its (image_path, tab) identity matches the meta), the display-resolved shape renderer
(crop-to-viewport math, two-pass draw, malformed-shape tolerance), the canvas_open_binding
write-authority fence on the push route, and the MCP tool end (missing binding, a binding naming
another project, identity-stale shapes, ages, tag/creator counts).
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


def _select(client: TestClient, root: Path, **over) -> dict:
    """Mint a real canvas_open_binding for ``root`` through the real select route."""
    body = {"project_root": str(root), "dataset_root": str(root)}
    body.update(over)
    r = client.post("/api/dataset/select", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _write_binding_raw(workspace: Path, *, generation: int, root: Path,
                       project_name: str | None = None) -> None:
    """Write a canvas_open_binding record as a plain file, bypassing the store seam entirely.

    Used only by guard proofs run against a baseline that predates this store's registration: a
    seam call (even a lazy import inside a helper) would raise ``ImportError`` there, which
    ``prove_test_fails_before.py`` treats as unreached rather than as evidence. The path is still
    computed through the generic ``RootedFileLocator`` primitive (the same one the registered
    store's own locator wraps), not hand-spelled, so the layout is stated once even though this
    fixture cannot go through the registration itself; that primitive predates this store by a
    wide margin, so it carries none of the baseline dependency the registration would.
    """
    import json

    from tcip_store.file_backend import RootedFileLocator

    locator = RootedFileLocator(prefix=(".tcip", "state"), suffix=".json")
    doc = workspace / locator.relative_path(str(workspace), ("canvas_open_binding",))
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(json.dumps({
        "generation": generation, "root": str(root), "project_name": project_name,
        "issued_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")


def _mint_binding(root: Path, *, generation: int = 1, project_name: str | None = None) -> None:
    """Write a canvas_open_binding record directly through the seam, standing in for the one
    production writer, ``_write_canvas_binding`` (``routes/dataset.py``), for an MCP-only test
    with no HTTP round trip to mint one through.

    Not a call to that writer itself: it derives ``project_name`` from
    ``workspace.workspace_project_name(root)``, which names a project only for a root that is
    exactly one workspace project's own directory, so a foreign-binding test that needs an
    arbitrary ``project_name`` on a root outside the workspace (the shape a mismatch test wants)
    has no way to reach that name through the real writer.
    """
    from tcip_mcp.web_client import canvas_open_binding_key

    key = canvas_open_binding_key()
    stored = tcip_store.read_versioned(key, default=None)
    tcip_store.replace(key, {
        "generation": generation,
        "root": str(root),
        "project_name": project_name,
        "issued_at": "2026-01-01T00:00:00+00:00",
    }, expect=stored.version)


def _payload(image_path: str, generation: int, shapes=None, **over) -> dict:
    body = {
        "binding_generation": generation,
        "tab": "annotate",
        "image_path": image_path,
        "image": Path(image_path).name,
        "img_width": 200,
        "img_height": 100,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100, "scale": 1.0},
        "mode": "polygon",
        "user": "breeder",
        "classes": [{"id": 0, "name": "bud", "color": "#FF0000"}],
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


# ── route: the write destination is the binding's own root ──────────────────

def test_full_push_writes_geometry_and_meta(client, tmp_path):
    sel = _select(client, tmp_path)
    r = client.post("/api/canvas/state", json=_payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES))
    assert r.status_code == 200 and r.json()["shapes_written"] is True
    assert len(_shapes_doc(tmp_path)["shapes"]) == 3
    assert _shapes_doc(tmp_path)["image_path"] == "C:/img/a.jpg"
    assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"


def test_cut_armed_flag_rides_the_meta_push(client, tmp_path):
    """A client fact like dirty or mode: the completed-cut and refusal cases both leave the flag
    set but clear the mirrored pending segment, so the meta document has to carry the flag itself
    for the mirror to read as armed rather than disarmed."""
    sel = _select(client, tmp_path)
    r = client.post(
        "/api/canvas/state",
        json=_payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES, cut_armed=True),
    )
    assert r.status_code == 200
    assert _meta(tmp_path)["cut_armed"] is True


def test_heartbeat_updates_meta_without_touching_geometry(client, tmp_path):
    sel = _select(client, tmp_path)
    client.post("/api/canvas/state", json=_payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES))
    before = tcip_store.read_versioned(canvas_geometry_key(str(tmp_path))).version
    hb = _payload("C:/img/a.jpg", sel["generation"], shapes=None,
                  viewport={"x": 40, "y": 10, "w": 80, "h": 50, "scale": 2.0})
    r = client.post("/api/canvas/state", json=hb)
    assert r.json()["shapes_written"] is False
    assert _meta(tmp_path)["viewport"]["x"] == 40                       # meta moved
    after = tcip_store.read_versioned(canvas_geometry_key(str(tmp_path))).version
    assert after == before                                              # geometry blob untouched


def test_push_lands_under_the_bindings_root_never_a_client_named_one(client, tmp_path):
    """Payload-authority proof: the write destination is the canvas_open_binding record's own
    root, never anything the client names; a body that still names project_root refuses outright
    (extra='forbid') rather than being silently ignored."""
    other = tmp_path.parent / "elsewhere"
    other.mkdir()
    sel = _select(client, tmp_path)

    body = _payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES)
    r = client.post("/api/canvas/state", json=body)
    assert r.status_code == 200
    assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"
    assert not (other / ".tcip" / "state" / "canvas_live.json").exists()

    r2 = client.post("/api/canvas/state", json={**body, "project_root": str(other)})
    assert r2.status_code == 422


def test_push_answers_409_on_a_stale_generation_and_nothing_lands_under_the_old_root(client, tmp_path):
    """The binding round trip through real producers: a stale generation after a second select of
    a different root answers 409 with the current generation, and nothing stale-checked lands
    under the old root."""
    first = _select(client, tmp_path)
    second_root = tmp_path.parent / "proj2"
    second_root.mkdir()
    second = _select(client, second_root)
    assert second["generation"] == first["generation"] + 1

    stale = _payload("C:/img/a.jpg", first["generation"], shapes=SHAPES)
    r = client.post("/api/canvas/state", json=stale)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["generation"] == second["generation"]
    assert not (tmp_path / ".tcip" / "state" / "canvas_live.json").exists()

    # A push carrying the current generation still lands correctly (no ordering claim beyond
    # this: a push accepted before a generation bump legitimately finishes under the old root).
    fresh = _payload("C:/img/b.jpg", second["generation"], shapes=SHAPES)
    r2 = client.post("/api/canvas/state", json=fresh)
    assert r2.status_code == 200
    assert _meta(second_root)["image_path"] == "C:/img/b.jpg"


def test_push_answers_409_on_no_binding_at_all(client, tmp_path):
    """The refusal's other half: a push before any select has ever run names a missing record,
    not a generation mismatch."""
    r = client.post("/api/canvas/state", json=_payload("C:/img/a.jpg", 1, shapes=SHAPES))
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["generation"] is None
    assert detail["project_name"] is None
    assert not (tmp_path / ".tcip" / "state" / "canvas_live.json").exists()


def test_push_lands_under_a_non_workspace_root_bound_with_a_null_project_name(
    client, tmp_path_factory, monkeypatch,
):
    """A registered dataset root or TCIP_IMAGE_ROOTS entry binds by root all the same, with no
    workspace name: the binding-by-root-with-null-name case."""
    outside = tmp_path_factory.mktemp("outside")
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(outside))

    sel = _select(client, outside)
    assert sel["generation"] >= 1

    r = client.post("/api/canvas/state", json=_payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES))
    assert r.status_code == 200
    assert _meta(outside)["image_path"] == "C:/img/a.jpg"


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
    sel = _select(client, tmp_path)  # the select's own binding write may fsync; only the push must not

    def _boom(*_a, **_kw):
        raise AssertionError("fsync should not be called for canvas state")

    monkeypatch.setattr(_os, "fsync", _boom)
    r = client.post("/api/canvas/state", json=_payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES))
    assert r.status_code == 200
    assert _meta(tmp_path)["image_path"] == "C:/img/a.jpg"


def test_a_push_waits_for_a_holder_of_the_records_lock_and_then_lands(client, tmp_path):
    """A push takes the meta record's lock, so it cannot overwrite what a holder is editing.

    The push runs on its own thread while another thread holds the key, and is observed still
    waiting; once the holder lets go the push completes and its document is what the record
    holds. Both halves matter: waiting alone would be a push that never lands.
    """
    import threading

    import tcip_store as ts

    from tcip_web.routes.canvas import CanvasStatePayload, canvas_meta_key, push_canvas_state

    sel = _select(client, tmp_path)
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

    payload = CanvasStatePayload(**_payload("C:/img/a.jpg", sel["generation"]))
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
    sel = _select(client, tmp_path)
    client.post("/api/canvas/state", json=_payload("C:/img/a.jpg", sel["generation"], shapes=SHAPES))
    client.post("/api/canvas/state", json=_payload("C:/img/b.jpg", sel["generation"], shapes=None))
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
                 tab: str = "annotate", shapes_tab: str | None = None,
                 cut_armed: bool | None = None) -> None:
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
        "cut_armed": cut_armed,
        "classes": [{"id": 0, "name": "bud", "color": "#FF0000"}],
    })


def test_capture_live_canvas_no_binding_names_the_consulted_workspace_root(tmp_path, monkeypatch):
    """No binding record exists: the message names the workspace root it consulted and that
    opening a project in the GUI creates one. Nothing else is written (no canvas state either),
    so this also covers the plain absence case the old 'no live canvas state' message used to
    answer for a different reason."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    from tcip_mcp.workspace import workspace_root

    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert "no current canvas binding" in res["error"].lower()
    assert str(workspace_root(create=False)) in res["error"]


def test_capture_live_canvas_binding_present_but_no_state_pushed_yet(tmp_path, monkeypatch):
    """The binding already names this same pinned root, so the message must not suggest
    activate_project toward a project it has already confirmed is the open one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    _mint_binding(tmp_path)
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert "no live canvas state" in res["error"].lower()
    assert "activate_project" not in res["error"]


def test_capture_live_canvas_renders_pushed_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)
    _mint_binding(tmp_path)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert Path(res["image_path"]).is_file()
    assert res["classes"][0]["name"] == "bud"
    assert res["shape_counts_by_tag"] == {"gt": 2, "in_progress": 1}
    assert res["shape_counts_by_creator"] == {"user:breeder": 1, "derived:user:breeder": 1}
    assert res["state_age_seconds"] >= 0
    assert res["shapes_missing"] is False
    assert res["project_root"] == str(tmp_path)
    assert "divergence" not in res


def test_capture_live_canvas_names_the_armed_cut(tmp_path, monkeypatch):
    """The mirror's armed state must be readable beside the mode it names, not just implied by
    the pending-segment polyline (which a completed cut or a refusal both clear while the flag
    stays set)."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img, cut_armed=True)
    _mint_binding(tmp_path)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert res["cut_armed"] is True


def test_capture_live_canvas_renders_exactly_the_viewport_region(tmp_path, monkeypatch):
    """The tool reads the visible rectangle and renders that, so the artifact is the region the
    human sees rather than the whole frame."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)
    _mint_binding(tmp_path)
    live = tcip_store.read(canvas_meta_key(str(tmp_path)))
    live["viewport"] = {"x": 50, "y": 0, "w": 100, "h": 100}
    tcip_store.replace(canvas_meta_key(str(tmp_path)), live)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert res["cropped_to_viewport"] is True
    assert Image.open(res["image_path"]).size == (100, 100)


def test_capture_live_canvas_full_frame_downscales_to_max_edge(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)
    _mint_binding(tmp_path)

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

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(3)
    src = images / "capture.tif"
    tifffile.imwrite(str(src), rng.integers(0, 4096, size=(100, 200, 6)).astype(np.uint16))
    _write_state(tmp_path, str(src))
    _mint_binding(tmp_path)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert Image.open(res["image_path"]).mode == "RGB"
    assert not (tmp_path / ".tcip" / "artifacts" / "viz" / "_band_previews").exists()


def test_capture_live_canvas_identity_stale_shapes_do_not_render(tmp_path, monkeypatch):
    """Geometry left over from a previous image must not render under the current one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img, shapes_image="C:/img/other.jpg")  # stale identity
    _mint_binding(tmp_path)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" not in res
    assert res["shapes_missing"] is True
    assert res["shape_counts_by_tag"] == {}
    assert Path(res["image_path"]).is_file()    # still renders the image + viewport


def test_capture_live_canvas_hit_case_divergence_is_not_silently_rendered(tmp_path, monkeypatch):
    """The reader pinned to A must not render A's own stale documents as live once the binding
    names B: GUARDS at the parent, where no binding concept existed and A's stale documents
    rendered unconditionally.

    The binding is written as a plain file at the exact path its store's locator places it at
    (bypassing the store seam), so this stays constructible against a baseline that predates the
    canvas_open_binding registration entirely: no symbol this proof needs is new at the parent.
    """
    from tcip_store.file_backend import FileBackend

    tcip_store.bind(FileBackend())
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)  # A's own (this reader's) stale documents: a "hit"

    workspace = tmp_path.parent
    _write_binding_raw(workspace, generation=7, root=workspace / "elsewhere")

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert "divergence" in res


def test_capture_live_canvas_miss_case_also_answers_divergence(tmp_path, monkeypatch, tmp_path_factory):
    """The divergence is the default answer on a miss (this reader has nothing of its own) just
    as much as on a hit."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name="currant_bud_valley")

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert res["divergence"]["bound_project"] == "currant_bud_valley"
    assert res["divergence"]["bound_root"] == str(other)


def test_capture_live_canvas_render_last_known_renders_labelled_not_live(
    tmp_path, monkeypatch, tmp_path_factory,
):
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name="currant_bud_valley")

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    refused = capture_live_canvas(refresh=False)
    assert "error" in refused

    rendered = capture_live_canvas(refresh=False, render_last_known=True)
    assert "error" not in rendered
    assert Path(rendered["image_path"]).is_file()
    assert rendered["divergence"]["bound_project"] == "currant_bud_valley"
    assert "not live" in rendered["summary"]


def test_capture_live_canvas_render_last_known_with_nothing_pushed_names_the_divergence(
    tmp_path, monkeypatch, tmp_path_factory,
):
    """render_last_known=True with nothing ever pushed under this pinned root: the message must
    not drop the divergence already in hand, nor tell the caller to activate_project toward a
    nameless root activate_project cannot converge."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    other = tmp_path_factory.mktemp("elsewhere")
    _mint_binding(other, generation=3, project_name=None)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False, render_last_known=True)
    assert "error" in res
    assert "activate_project" not in res["error"]
    assert res["divergence"]["bound_root"] == str(other)
    assert "reselect this project in the GUI" in res["divergence"]["converge"]


def test_capture_live_canvas_unreadable_binding_store_is_reported_distinctly(tmp_path, monkeypatch):
    """An unreadable binding store must not read as absent (which would misname the fix as
    'open a project') nor escape the audited tool as a raw exception."""
    import tcip_store as ts

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    real_read = ts.read

    def _boom(key, **kwargs):
        if key.store == "canvas_open_binding":
            raise ts.DecodeError("corrupt binding record")
        return real_read(key, **kwargs)

    monkeypatch.setattr(ts, "read", _boom)
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert "could not read the canvas-open binding" in res["error"].lower()


def test_capture_live_canvas_binding_oserror_is_reported_distinctly(tmp_path, monkeypatch):
    """A permission or path-shape fault on the binding record is an OSError, not a StoreError:
    the softener widens to catch it too, rather than letting it escape the audited tool raw."""
    import tcip_store as ts

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    real_read = ts.read

    def _boom(key, **kwargs):
        if key.store == "canvas_open_binding":
            raise PermissionError("simulated permission fault")
        return real_read(key, **kwargs)

    monkeypatch.setattr(ts, "read", _boom)
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert "could not read the canvas-open binding" in res["error"].lower()


def test_capture_live_canvas_generation_fence_retries_once_then_answers_divergence(
    tmp_path, monkeypatch,
):
    """A project switch mid-call (the binding moves between the two reads the fence takes) must
    not produce a false live result: it retries once, and if the mismatch persists, answers with
    the divergence rather than the render it just produced.

    Minted through the seam (``_mint_binding``), not a raw file: unlike the hit-case divergence
    guard above, this test has no baseline-constructibility need to bypass the store, so it runs
    on whichever backend the leg actually bound rather than being pinned to the file backend.
    """
    import tcip_store as ts

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    img = _make_image(tmp_path)
    _write_state(tmp_path, img)
    _mint_binding(tmp_path, generation=1)

    real_read = ts.read
    reads = {"n": 0}
    elsewhere = str(tmp_path.parent / "elsewhere")

    def _patched(key, **kwargs):
        if key.store == "canvas_open_binding":
            reads["n"] += 1
            if reads["n"] % 2 == 0:  # every "after the render" read: the binding has moved
                return {"generation": 2, "root": elsewhere, "project_name": None,
                        "issued_at": "2026-01-01T00:00:00+00:00"}
        return real_read(key, **kwargs)

    monkeypatch.setattr(ts, "read", _patched)
    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(refresh=False)
    assert "error" in res
    assert "divergence" in res
    assert reads["n"] == 4  # exactly one retry: two reads per attempt, two attempts

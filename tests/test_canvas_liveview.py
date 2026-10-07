"""Live canvas view: the GUI's canvas-state push + the agent's capture_live_canvas render.

Covers the two-file scheme (meta heartbeats never touch the geometry blob; geometry is valid
only when its (image_path, tab) identity matches the meta), the display-resolved shape renderer
(crop-to-viewport math, two-pass draw, malformed-shape tolerance), the push route answering a push
built for any project but the backend's open one as a mismatch, and the MCP tool end (no state
pushed, identity-stale shapes, ages, tag/creator counts).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")

import tcip_store  # noqa: E402
from tcip_mcp.project_record import read_record  # noqa: E402
from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key  # noqa: E402


def _payload(project: Path, image_path: str, shapes=None, **over) -> dict:
    body = {
        "project_id": read_record(project)["id"],
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


def _meta(root: Path) -> dict | None:
    return tcip_store.read(canvas_meta_key(str(root)), default=None)


def _shapes_doc(root: Path) -> dict | None:
    return tcip_store.read(canvas_geometry_key(str(root)), default=None)


A_IMG, B_IMG = "C:/img/a.jpg", "C:/img/b.jpg"
"""Images outside every project; a push stores each as its resolved absolute path."""
A_STORED, B_STORED = str(Path(A_IMG).resolve()), str(Path(B_IMG).resolve())

SHAPES = [
    {"kind": "polygon", "points": [[10, 10], [60, 10], [60, 60]], "color": "#00FF00",
     "fill": True, "tag": "gt", "created_by": "user:breeder"},
    {"kind": "box", "xyxy": [80, 20, 140, 70], "color": "#FF0000", "tag": "gt",
     "created_by": "derived:user:breeder"},
    {"kind": "polyline", "points": [[5, 90], [30, 85], [55, 92]], "color": "#FFE7B1",
     "dash": [4, 4], "tag": "in_progress", "label": "drawing"},
]


def _other_project(tmp_path: Path) -> Path:
    from tcip_mcp.tools.project_tools import initialize_project

    other = tmp_path.parent / "other_project"
    assert "error" not in initialize_project(str(other), "Other project", "south block")
    return other


# ── route: a push lands only under the project it was built for, while that one is open ─────

def test_full_push_writes_geometry_and_meta(client, opened_project):
    image = str(opened_project / "images" / "a.jpg")
    r = client.post("/api/canvas/state", json=_payload(opened_project, image, shapes=SHAPES))
    assert r.status_code == 200 and r.json()["shapes_written"] is True
    assert len(_shapes_doc(opened_project)["shapes"]) == 3
    assert _shapes_doc(opened_project)["image_path"] == "images/a.jpg"
    assert _meta(opened_project)["image_path"] == "images/a.jpg"


def test_a_push_naming_another_project_is_answered_as_a_mismatch_and_writes_nothing(
    client, opened_project, tmp_path,
):
    """The backend compares the project a push names with the one it has open: a push built for
    another project answers 409 naming the open one, and nothing lands under either."""
    other = _other_project(tmp_path)

    r = client.post("/api/canvas/state",
                    json=_payload(other, A_IMG, shapes=SHAPES))

    assert r.status_code == 409
    assert r.json()["detail"]["open_project_id"] == read_record(opened_project)["id"]
    assert _meta(opened_project) is None
    assert _meta(other) is None


def test_a_push_naming_the_open_project_is_admitted(client, opened_project, tmp_path):
    """The admitting half: the same push, built for the project the backend has open, lands."""
    _other_project(tmp_path)

    r = client.post("/api/canvas/state",
                    json=_payload(opened_project, A_IMG, shapes=SHAPES))

    assert r.status_code == 200, r.text
    assert _meta(opened_project)["image_path"] == A_STORED


def test_a_push_while_no_project_is_open_is_answered_as_a_mismatch(client, project):
    r = client.post("/api/canvas/state", json=_payload(project, A_IMG, shapes=SHAPES))

    assert r.status_code == 409
    assert r.json()["detail"]["open_project_id"] is None
    assert _meta(project) is None


def test_a_push_naming_no_one_is_refused_before_anything_is_written(client, opened_project):
    """A push may state no person, but one stating a person must name one: a blank name is
    refused before the geometry or the meta document is written."""
    r = client.post("/api/canvas/state",
                    json={**_payload(opened_project, A_IMG, shapes=SHAPES), "user": "  "})

    assert r.status_code == 400
    assert _shapes_doc(opened_project) is None and _meta(opened_project) is None


def test_a_push_naming_a_project_root_refuses_as_an_unknown_field(client, opened_project):
    """The write destination is the open project, never anything the client names: a body that
    names a root refuses outright (extra='forbid') rather than being silently ignored."""
    body = _payload(opened_project, A_IMG, shapes=SHAPES)
    r = client.post("/api/canvas/state", json={**body, "project_root": str(opened_project)})
    assert r.status_code == 422


def test_cut_armed_flag_rides_the_meta_push(client, opened_project):
    """A client fact like dirty or mode: the completed-cut and refusal cases both leave the flag
    set but clear the mirrored pending segment, so the meta document has to carry the flag itself
    for the mirror to read as armed rather than disarmed."""
    r = client.post(
        "/api/canvas/state",
        json=_payload(opened_project, A_IMG, shapes=SHAPES, cut_armed=True),
    )
    assert r.status_code == 200
    assert _meta(opened_project)["cut_armed"] is True


def test_heartbeat_updates_meta_without_touching_geometry(client, opened_project):
    client.post("/api/canvas/state",
                json=_payload(opened_project, A_IMG, shapes=SHAPES))
    before = tcip_store.read_versioned(canvas_geometry_key(str(opened_project))).version
    hb = _payload(opened_project, A_IMG, shapes=None,
                  viewport={"x": 40, "y": 10, "w": 80, "h": 50, "scale": 2.0})
    r = client.post("/api/canvas/state", json=hb)
    assert r.json()["shapes_written"] is False
    assert _meta(opened_project)["viewport"]["x"] == 40                    # meta moved
    after = tcip_store.read_versioned(canvas_geometry_key(str(opened_project))).version
    assert after == before                                                 # geometry untouched


def test_a_push_waits_for_a_holder_of_the_records_lock_and_then_lands(opened_project):
    """A push takes the meta record's lock, so it cannot overwrite what a holder is editing.

    The push runs on its own thread while another thread holds the key, and is observed still
    waiting; once the holder lets go the push completes and its document is what the record
    holds. Both halves matter: waiting alone would be a push that never lands.
    """
    import threading

    import tcip_store as ts

    from tcip_web.routes.canvas import CanvasStatePayload, push_canvas_state
    from tests._audit_fixtures import held_by_another_writer

    key = canvas_meta_key(str(opened_project))
    payload = CanvasStatePayload(**_payload(opened_project, A_IMG))
    pushing = threading.Thread(target=lambda: push_canvas_state(payload))
    with held_by_another_writer(key):
        pushing.start()
        pushing.join(0.5)
        waiting = pushing.is_alive()
    pushing.join(10)
    assert waiting, "the push wrote while another writer held the record's lock"
    assert not pushing.is_alive()
    assert ts.read(key)["image_path"] == A_STORED


def test_heartbeat_for_new_image_invalidates_geometry_by_identity(client, opened_project):
    client.post("/api/canvas/state",
                json=_payload(opened_project, A_IMG, shapes=SHAPES))
    client.post("/api/canvas/state",
                json=_payload(opened_project, B_IMG, shapes=None))
    # The geometry file still holds a.jpg's shapes, but its identity no longer matches the meta:
    # the reader must treat it as stale (a.jpg's polygons never render under b.jpg).
    assert _shapes_doc(opened_project)["image_path"] == A_STORED
    assert _meta(opened_project)["image_path"] == B_STORED


# ── renderer: region placement + two-pass draw + tolerance ────────────

_WHOLE = (0, 0, 200, 100)
"""The whole frame of the image :func:`_make_image` writes, as a native region."""


def _make_image(tmp_path: Path) -> str:
    from tests._producer_fixtures import write_image

    return str(write_image(tmp_path / "img.jpg", (200, 100), (90, 110, 90)))


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


def _red_over_green(rendered: Image.Image, xy: tuple[int, int]) -> int:
    """How far red exceeds green at pixel ``xy`` of ``rendered``."""
    red, green, _ = rendered.getpixel(xy)
    return red - green


def test_render_places_a_shape_at_its_offset_inside_a_cropped_region(tmp_path):
    """A shape's native coordinate lands where the crop origin puts it, not where it sat in the
    full frame: the placement the viewport read replaced the renderer's own crop with."""
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [{"kind": "point", "points": [[100, 50]], "color": "#FF0000"}]
    out = render_canvas_state(_pixels(img, region=(50, 0, 150, 100)), shapes,
                              region=(50, 0, 150, 100),
                              output_path=str(tmp_path / "crop.png"))
    px = Image.open(out).convert("RGB")
    assert px.size == (100, 100)                        # exactly the region handed in
    assert _red_over_green(px, (50, 50)) > 40      # native x=100 minus origin x=50
    assert _red_over_green(px, (95, 50)) < 10      # nothing where the unshifted one would hit


def test_render_scales_a_shape_with_the_pixels_it_is_drawn_on(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [{"kind": "point", "points": [[100, 50]], "color": "#FF0000"}]
    out = render_canvas_state(_pixels(img, scale=0.5), shapes, region=_WHOLE,
                              output_path=str(tmp_path / "half.png"))
    px = Image.open(out).convert("RGB")
    assert px.size == (100, 50)
    assert _red_over_green(px, (50, 25)) > 40      # native (100, 50) at half resolution
    assert _red_over_green(px, (90, 25)) < 10


def test_render_two_pass_fill_does_not_erase_outlines(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shapes = [
        {"kind": "box", "xyxy": [20, 20, 80, 80], "color": "#00FF00"},          # green outline
        # later red fill over it
        {"kind": "box", "xyxy": [10, 10, 90, 90], "color": "#FF0000", "fill": True},
    ]
    out = render_canvas_state(_pixels(img), shapes, region=_WHOLE,
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
    out = render_canvas_state(_pixels(img), shapes, region=_WHOLE,
                              output_path=str(tmp_path / "point.png"))
    px = Image.open(out).convert("RGB")
    assert _red_over_green(px, (100, 50)) > 40     # the core sits on the coordinate
    assert _red_over_green(px, (100, 42)) > 40     # a radial tick above it (the mark's reticle)
    assert _red_over_green(px, (140, 50)) < 10     # nothing 40px away: no box, fill or outline


def test_render_draws_the_focused_shape_with_a_halo_under_its_own_stroke(tmp_path):
    """The GUI's focus is a halo, never a recolor: a focused shape reaches the agent's view with a
    wider white stroke beneath its own color, and an unfocused one with nothing around it."""
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)

    def rendered(focused: bool) -> Image.Image:
        halo = {"color": "#FFFFFF", "opacity": 0.55, "width_factor": 3}
        shapes = [{"kind": "box", "xyxy": [40, 20, 160, 80], "color": "#FF0000",
                   **({"halo": halo} if focused else {})}]
        out = render_canvas_state(_pixels(img), shapes, region=_WHOLE,
                                  output_path=str(tmp_path / f"focus-{focused}.png"))
        return Image.open(out).convert("RGB")

    plain, haloed = rendered(False), rendered(True)
    # One pixel outside the stroke the halo brightens every channel; the plain render does not.
    beside = (100, 19)
    assert min(haloed.getpixel(beside)) > min(plain.getpixel(beside)) + 40
    # The shape's own color still sits on the stroke itself.
    assert _red_over_green(haloed, (100, 20)) > 40


_MIRROR = json.loads((Path(__file__).parent / "fixtures" / "canvas_mirror_shapes.json")
                     .read_text(encoding="utf-8"))
"""The shapes the frontend's ``buildAnnotateShapes`` produced for a tool-authored polygon, a
tool-authored point and the first vertex of a polygon in progress; ``canvasSync.test.ts`` asserts
the producer still emits exactly this file."""
(TOOL_POLYGON,) = _MIRROR["tool_polygon"]
(TOOL_POINT,) = _MIRROR["tool_point"]
(FIRST_VERTEX,) = _MIRROR["first_vertex"]


def _wide_render(tmp_path: Path, shape: dict, *, dashed: bool) -> Image.Image:
    """``shape`` rendered on a 1400-pixel frame, where the render's stroke is two pixels wide."""
    from tcip_annotation.viz import render_canvas_state
    from tests._producer_fixtures import write_image

    img = str(write_image(tmp_path / "wide.jpg", (1400, 200), (90, 110, 90)))
    shape = {k: v for k, v in shape.items() if dashed or k != "dash"}
    out = render_canvas_state(_pixels(img), [shape], region=(0, 0, 1400, 200),
                              output_path=str(tmp_path / f"wide-{dashed}.png"))
    return Image.open(out).convert("RGB")


def _ink(rendered: Image.Image, xy: tuple[int, int]) -> int:
    """How far pixel ``xy`` differs from the frame's untouched corner."""
    return max(abs(a - b) for a, b in zip(rendered.getpixel(xy), rendered.getpixel((2, 197))))


def _red_in_rows(rendered: Image.Image, x: int, rows) -> int:
    """The most ink the pixels of column ``x`` in ``rows`` carry."""
    return max(_ink(rendered, (x, y)) for y in rows)


def test_render_lays_a_dash_down_in_stroke_widths_at_the_renders_own_stroke(tmp_path):
    """The two-pixel stroke of a 1400-pixel frame makes [1, 3] two pixels on, six off: a render
    that left the pattern unscaled would be one on, three off and light the gap's pixels."""
    dashed = _wide_render(tmp_path, TOOL_POLYGON, dashed=True)
    solid = _wide_render(tmp_path, TOOL_POLYGON, dashed=False)
    rows = range(49, 52)
    assert _red_in_rows(dashed, 301, rows) > 40      # inside a dash
    assert _red_in_rows(dashed, 304, rows) < 10      # inside the gap after it
    assert _red_in_rows(dashed, 309, rows) > 40      # the next dash
    assert _red_in_rows(solid, 304, rows) > 40       # the gap's pixel on a solid outline


def test_render_dashes_a_points_ticks_as_it_does_any_outline(tmp_path):
    """A tool's point reaches the agent dotted, as the browser draws it: the tick above the core
    has its one dash near the core and a gap beyond, where a solid tick would still be drawn."""
    dashed = _wide_render(tmp_path, TOOL_POINT, dashed=True)
    solid = _wide_render(tmp_path, TOOL_POINT, dashed=False)
    assert _red_in_rows(dashed, 700, range(90, 91)) > 40
    assert _red_in_rows(dashed, 700, range(86, 87)) < 10
    assert _red_in_rows(solid, 700, range(86, 87)) > 40


def test_render_admits_the_one_vertex_polyline_a_drawing_in_progress_pushes(tmp_path):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    out = render_canvas_state(_pixels(img), [FIRST_VERTEX], region=_WHOLE,
                              output_path=str(tmp_path / "one-vertex.png"))
    drawn = Image.open(out).convert("RGB")
    assert max(abs(a - b) for a, b in zip(drawn.getpixel((60, 40)), drawn.getpixel((10, 90)))) > 40


def test_render_dots_a_polylines_vertices_unless_it_says_it_is_a_tail(tmp_path):
    """A draft's laid stroke dots each vertex; its tail to the cursor, which says so, dots none."""
    def stroke(**over) -> dict:
        return {"kind": "polyline", "points": [[300, 100], [500, 100]], "color": "#00CED1",
                "tag": "in_progress", **over}

    laid = _wide_render(tmp_path, stroke(), dashed=True)
    tail = _wide_render(tmp_path, stroke(vertices=False), dashed=True)
    assert _ink(laid, (500, 96)) > 40
    assert _ink(tail, (500, 96)) < 10


@pytest.mark.parametrize("dash", [[0, 4], [4], "xx", [4, -1], [], 0, False, "", ["1", "3"], "13",
                                  [True, False], [1, 3, 5], [float("nan"), 3], [1, float("inf")]],
                         ids=["zero_on", "one_length", "text", "negative_off", "empty", "zero",
                              "false", "empty_text", "numeric_strings", "digit_string",
                              "booleans", "three_lengths", "nan_on", "infinite_off"])
def test_render_refuses_a_dash_that_is_not_a_pattern_naming_the_shape(tmp_path, dash):
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    shape = {"kind": "box", "xyxy": [10, 10, 40, 40], "color": "#FF0000", "dash": dash}
    with pytest.raises(ValueError, match="canvas shape 0 carries a dash"):
        render_canvas_state(_pixels(img), [shape], region=_WHOLE,
                            output_path=str(tmp_path / "bad-dash.jpg"))


@pytest.mark.parametrize("bad", [
    {"kind": "box", "color": "#FF0000"},
    {"kind": "polygon", "points": [[1, 1]], "color": "#FF0000"},
    "junk", {"kind": "box", "xyxy": [10, 10, 40, 40], "color": "not-a-color"},
    {"kind": "box", "xyxy": [10, 10, 40, 40]}],
    ids=["box_without_corners", "one_point_polygon", "not_a_shape", "unparsable_color", "no_color"])
def test_render_refuses_a_malformed_shape_naming_it(tmp_path, bad):
    """A shape the canvas push could not have drawn refuses naming its index, never renders a
    canvas the human is not seeing; the well-formed shapes beside it admit."""
    from tcip_annotation.viz import render_canvas_state
    img = _make_image(tmp_path)
    good = {"kind": "box", "xyxy": [10, 10, 40, 40], "color": "#0f0"}
    assert Path(render_canvas_state(_pixels(img), [good], region=_WHOLE,
                                    output_path=str(tmp_path / "good.jpg"))).is_file()
    with pytest.raises(ValueError, match="canvas shape 1 "):
        render_canvas_state(_pixels(img), [good, bad], region=_WHOLE,
                            output_path=str(tmp_path / "bad.jpg"))


# ── MCP tool ────────────────────────────────────────────────────────────────

def _write_state(project: Path, img: str, shapes=SHAPES, *, shapes_image: str | None = None,
                 tab: str = "annotate", shapes_tab: str | None = None,
                 cut_armed: bool | None = None) -> None:
    """The two documents the push route writes, in the shape it writes them: the image path
    spelled against the project the way the route stores it."""
    from tcip_mcp.audit import now_iso
    from tcip_mcp.registry_paths import stored_path

    root = str(project)
    stored = stored_path(img, project)
    now = now_iso()
    if shapes is not None:
        tcip_store.replace(canvas_geometry_key(root), {
            "image_path": stored_path(shapes_image, project) if shapes_image else stored,
            "tab": shapes_tab or tab, "shapes": shapes, "received_at": now,
        })
    tcip_store.replace(canvas_meta_key(root), {
        "received_at": now, "tab": tab,
        "image": Path(img).name, "image_path": stored,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100}, "user": "breeder", "mode": "polygon",
        "cut_armed": cut_armed,
        "classes": [{"id": 0, "name": "bud", "color": "#FF0000"}],
    })


def test_capture_live_canvas_with_no_state_pushed_names_what_pushes_one(project):
    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(project, project.parent, refresh=False)
    assert "error" in res
    assert "no canvas state has been pushed" in res["error"].lower()


def test_capture_live_canvas_renders_pushed_state(project):
    img = _make_image(project)
    _write_state(project, img)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(project, project.parent, refresh=False)
    assert "error" not in res
    assert Path(res["image_path"]).is_file()
    assert res["source_image"] == img
    assert res["classes"][0]["name"] == "bud"
    assert res["shape_counts_by_tag"] == {"gt": 2, "in_progress": 1}
    assert res["shape_counts_by_creator"] == {"user:breeder": 1, "derived:user:breeder": 1}
    assert res["state_age_seconds"] >= 0
    assert res["shapes_missing"] is False


def test_capture_live_canvas_names_the_armed_cut(project):
    """The mirror's armed state must be readable beside the mode it names, not just implied by
    the pending-segment polyline (which a completed cut or a refusal both clear while the flag
    stays set)."""
    img = _make_image(project)
    _write_state(project, img, cut_armed=True)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(project, project.parent, refresh=False)
    assert res["cut_armed"] is True


def test_capture_live_canvas_renders_exactly_the_viewport_region(project):
    """The tool reads the visible rectangle and renders that, so the artifact is the region the
    human sees rather than the whole frame."""
    img = _make_image(project)
    _write_state(project, img)
    live = tcip_store.read(canvas_meta_key(str(project)))
    live["viewport"] = {"x": 50, "y": 0, "w": 100, "h": 100}
    tcip_store.replace(canvas_meta_key(str(project)), live)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(project, project.parent, refresh=False)
    assert res["cropped_to_viewport"] is True
    assert Image.open(res["image_path"]).size == (100, 100)


def test_capture_live_canvas_full_frame_downscales_to_max_edge(project):
    img = _make_image(project)
    _write_state(project, img)

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(
        project, project.parent, refresh=False, crop_to_viewport=False, max_edge=100
    )
    assert res["cropped_to_viewport"] is False
    assert Image.open(res["image_path"]).size == (100, 50)


def test_capture_live_canvas_reads_a_multiband_raster_without_writing_a_preview(project):
    """A raster PIL has no true-color mode for is composited in memory for the render: the capture
    path writes no throwaway preview file beside the artifact."""
    import numpy as np
    import tifffile

    images = project / "images"
    images.mkdir()
    rng = np.random.default_rng(3)
    src = images / "capture.tif"
    tifffile.imwrite(str(src), rng.integers(0, 4096, size=(100, 200, 6)).astype(np.uint16))
    _write_state(project, str(src))

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(project, project.parent, refresh=False)
    assert "error" not in res
    assert Image.open(res["image_path"]).mode == "RGB"
    assert not (project / ".tcip" / "artifacts" / "viz" / "_band_previews").exists()


def test_capture_live_canvas_identity_stale_shapes_do_not_render(project):
    """Geometry left over from a previous image must not render under the current one."""
    img = _make_image(project)
    _write_state(project, img, shapes_image="C:/img/other.jpg")  # stale identity

    from tcip_mcp.tools.vision_tools import capture_live_canvas
    res = capture_live_canvas(project, project.parent, refresh=False)
    assert "error" not in res
    assert res["shapes_missing"] is True
    assert res["shape_counts_by_tag"] == {}
    assert Path(res["image_path"]).is_file()    # still renders the image + viewport

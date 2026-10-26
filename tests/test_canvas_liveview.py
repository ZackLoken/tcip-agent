"""Live canvas view: the GUI's canvas-state push + the agent's capture_live_canvas render.

Covers the two-file scheme (meta heartbeats never touch the geometry blob; geometry is valid
only when its (image_path, tab) identity matches the meta), the display-resolved shape renderer
(crop-to-viewport math, two-pass draw), the push route answering a push built for any project
but the backend's open one as a mismatch, and the MCP tool end (no state pushed, identity-stale
shapes, ages, tag counts).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image
from pydantic import TypeAdapter, ValidationError

pytest.importorskip("fastapi")

import tcip_store  # noqa: E402
from tcip_annotation.viz import CanvasShape  # noqa: E402
from tcip_mcp.dataset_layout import image_root  # noqa: E402
from tcip_mcp.web_client import (  # noqa: E402
    canvas_geometry_key,
    canvas_meta_key,
    read_canvas_geometry,
    read_canvas_meta,
)
from tests import REFUSED_NAMES  # noqa: E402

_SHAPES = TypeAdapter(list[CanvasShape])
"""The pushed shapes as the canvas records validate them."""

if TYPE_CHECKING:
    import numpy as np
    from fastapi.testclient import TestClient

    from tcip_mcp.web_client import CanvasGeometry, CanvasMeta
    from tcip_mcp.workspace import BoundProject
    from tcip_web.state import OpenProject


def _payload(
    project_id: str,
    image_path: str,
    shapes: list[dict] | None = None,
    **over: bool | dict,
) -> dict:
    body = {
        "project_id": project_id,
        "tab": "annotate",
        "image_path": image_path,
        "image": Path(image_path).name,
        "img_width": 200,
        "img_height": 100,
        "viewport": {"x": 0, "y": 0, "w": 200, "h": 100, "scale": 1.0},
        "mode": "polygon",
        "active_subject": None,
        "cut_armed": False,
        "dirty": False,
        "user": "breeder",
        "classes": [{"id": 0, "name": "bud", "color": "#FF0000"}],
        "counts": {"boxes": 0, "polygons": 1, "drawing_points": 0},
        "shapes": shapes,
    }
    body.update(over)
    return body


def _meta(root: Path) -> CanvasMeta | None:
    return read_canvas_meta(str(root))


def _shapes_doc(root: Path) -> CanvasGeometry | None:
    return read_canvas_geometry(str(root))


_OUTSIDE = Path(tempfile.gettempdir()).resolve() / "tcip-canvas-liveview"
A_IMG, B_IMG = str(_OUTSIDE / "a.jpg"), str(_OUTSIDE / "b.jpg")
"""Images outside every project, absolute on every platform; a push stores each as its resolved
absolute path."""
A_STORED, B_STORED = str(Path(A_IMG).resolve()), str(Path(B_IMG).resolve())

SHAPES = [
    {
        "kind": "polygon",
        "points": [[10, 10], [60, 10], [60, 60]],
        "color": "#00FF00",
        "fill": True,
        "tag": "gt",
    },
    {
        "kind": "box",
        "xyxy": [80, 20, 140, 70],
        "color": "#FF0000",
        "tag": "gt",
    },
    {
        "kind": "polyline",
        "points": [[5, 90], [30, 85], [55, 92]],
        "color": "#FFE7B1",
        "dash": [4, 4],
        "tag": "in_progress",
        "label": "drawing",
    },
]


def _other_project(tmp_path: Path) -> OpenProject:
    """Return a second workspace project beside ``tmp_path``, built by ``named_project``."""
    from tests._web_fixtures import named_project

    return named_project(tmp_path.parent / "other_project", "Other project")


# ── route: a push lands only under the project it was built for, while that one is open ─────


def test_full_push_writes_geometry_and_meta(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    image = str(image_root(opened_project) / "a.jpg")
    r = client.post(
        "/api/canvas/state", json=_payload(opened.id, image, shapes=SHAPES)
    )
    assert r.status_code == 200 and r.json()["shapes_written"] is True
    assert len(_shapes_doc(opened_project).shapes) == 3
    assert _shapes_doc(opened_project).image_path == "images/a.jpg"
    assert _meta(opened_project).image_path == "images/a.jpg"


def test_a_push_naming_another_project_is_answered_as_a_mismatch_and_writes_nothing(
    client: TestClient,
    opened: OpenProject,
    opened_project: Path,
    tmp_path: Path,
) -> None:
    """A push built for another project answers 409 naming the open one.

    The canvas meta record stays absent under both projects.
    """
    other = _other_project(tmp_path)

    r = client.post(
        "/api/canvas/state", json=_payload(other.id, A_IMG, shapes=SHAPES)
    )

    assert r.status_code == 409
    assert r.json()["detail"]["open_project_id"] == opened.id
    assert _meta(opened_project) is None
    assert _meta(other.root) is None


def test_a_push_naming_the_open_project_is_admitted(
    client: TestClient,
    opened: OpenProject,
    opened_project: Path,
    tmp_path: Path,
) -> None:
    """A push built for the project the backend has open lands."""
    _other_project(tmp_path)

    r = client.post(
        "/api/canvas/state", json=_payload(opened.id, A_IMG, shapes=SHAPES)
    )

    assert r.status_code == 200, r.text
    assert _meta(opened_project).image_path == A_STORED


def test_a_push_while_no_project_is_open_is_answered_as_a_mismatch(
    client: TestClient, made: OpenProject
) -> None:
    r = client.post(
        "/api/canvas/state", json=_payload(made.id, A_IMG, shapes=SHAPES)
    )

    assert r.status_code == 409
    assert "no project is open" in r.json()["detail"]
    assert _meta(made.root) is None


def test_a_push_naming_no_one_is_refused_before_anything_is_written(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    """A push whose name names no one is refused, and no canvas document is written.

    A push must name its person; neither the geometry nor the meta document is written for one
    that does not.
    """
    for name in REFUSED_NAMES:
        r = client.post(
            "/api/canvas/state",
            json={**_payload(opened.id, A_IMG, shapes=SHAPES), "user": name},
        )
        assert r.status_code == 400, (name, r.text)
    assert (
        _shapes_doc(opened_project) is None and _meta(opened_project) is None
    )


def test_a_push_stating_no_person_is_refused_before_anything_is_written(
    client: TestClient,
    opened: OpenProject,
    opened_project: Path,
) -> None:
    body = _payload(opened.id, A_IMG, shapes=SHAPES)
    del body["user"]

    r = client.post("/api/canvas/state", json=body)

    assert r.status_code == 422
    assert (
        _shapes_doc(opened_project) is None and _meta(opened_project) is None
    )


def test_a_push_naming_a_person_is_stored_under_that_person(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    r = client.post(
        "/api/canvas/state", json=_payload(opened.id, A_IMG, shapes=SHAPES)
    )

    assert r.status_code == 200
    assert _meta(opened_project).user == "user:breeder"


def test_a_push_naming_a_project_root_refuses_as_an_unknown_field(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    """A push body that names a project root is refused as an unknown field, answering 422."""
    body = _payload(opened.id, A_IMG, shapes=SHAPES)
    r = client.post(
        "/api/canvas/state", json={**body, "project_root": str(opened_project)}
    )
    assert r.status_code == 422


def test_cut_armed_flag_rides_the_meta_push(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    """The cut-armed flag a push carries lands in the meta document."""
    r = client.post(
        "/api/canvas/state",
        json=_payload(opened.id, A_IMG, shapes=SHAPES, cut_armed=True),
    )
    assert r.status_code == 200
    assert _meta(opened_project).cut_armed is True


def test_heartbeat_updates_meta_without_touching_geometry(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    client.post(
        "/api/canvas/state", json=_payload(opened.id, A_IMG, shapes=SHAPES)
    )
    before = tcip_store.read_versioned(
        canvas_geometry_key(str(opened_project))
    ).version
    hb = _payload(
        opened.id,
        A_IMG,
        shapes=None,
        viewport={"x": 40, "y": 10, "w": 80, "h": 50, "scale": 2.0},
    )
    r = client.post("/api/canvas/state", json=hb)
    assert r.json()["shapes_written"] is False
    meta = _meta(opened_project)
    assert meta is not None and meta.viewport is not None
    assert meta.viewport.x == 40  # meta moved
    after = tcip_store.read_versioned(
        canvas_geometry_key(str(opened_project))
    ).version
    assert after == before  # geometry untouched


def test_a_push_is_held_off_by_a_holder_of_the_records_lock_and_then_lands(
    opened: OpenProject, opened_project: Path
) -> None:
    """A push takes the meta record's lock, so it cannot overwrite what a holder is editing.

    Under a store that never waits for a lock, a push while another thread holds the key is
    refused busy with nothing written; once the holder lets go the same push lands.
    """
    from tcip_store import StoreBusyError
    from tcip_store.sqlite_backend import SqliteBackend
    from tcip_web.routes.canvas import CanvasStatePayload, push_canvas_state
    from tests._audit_fixtures import held_by_another_writer

    key = canvas_meta_key(str(opened_project))
    payload = CanvasStatePayload(**_payload(opened.id, A_IMG))
    backend = tcip_store.bind(SqliteBackend(lock_timeout_s=0.0))
    try:
        with held_by_another_writer(key), pytest.raises(StoreBusyError):
            push_canvas_state(payload)
        assert _meta(opened_project) is None
        push_canvas_state(payload)
        assert _meta(opened_project).image_path == A_STORED
    finally:
        backend.close()


def test_heartbeat_for_new_image_invalidates_geometry_by_identity(
    client: TestClient, opened: OpenProject, opened_project: Path
) -> None:
    client.post(
        "/api/canvas/state", json=_payload(opened.id, A_IMG, shapes=SHAPES)
    )
    client.post(
        "/api/canvas/state", json=_payload(opened.id, B_IMG, shapes=None)
    )
    # The geometry file still holds a.jpg's shapes, but its identity no longer matches the meta:
    # the reader must treat it as stale (a.jpg's polygons never render under b.jpg).
    assert _shapes_doc(opened_project).image_path == A_STORED
    assert _meta(opened_project).image_path == B_STORED


# ── renderer: region placement + two-pass draw + tolerance ────────────

_WHOLE = (0, 0, 200, 100)
"""The whole frame of the image :func:`_make_image` writes, as a native region."""


def _make_image(tmp_path: Path) -> str:
    from tests._producer_fixtures import write_image

    return str(write_image(tmp_path / "img.jpg", (200, 100), (90, 110, 90)))


def _pixels(
    image_path: str,
    region: tuple[int, int, int, int] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the image's RGB pixels, cropped to ``region`` when given and resized by ``scale``."""
    import numpy as np

    with Image.open(image_path) as im:
        frame = im.convert("RGB")
        if region is not None:
            frame = frame.crop(region)
        if scale != 1.0:
            frame = frame.resize(
                (
                    max(1, round(frame.width * scale)),
                    max(1, round(frame.height * scale)),
                ),
                Image.LANCZOS,
            )
        return np.asarray(frame)


def _red_over_green(rendered: Image.Image, xy: tuple[int, int]) -> int:
    """Return how far red exceeds green at pixel ``xy`` of ``rendered``."""
    red, green, _ = rendered.getpixel(xy)
    return red - green


def test_render_places_a_shape_at_its_offset_inside_a_cropped_region(
    tmp_path: Path,
) -> None:
    """A shape's native coordinate lands where the crop origin puts it inside a cropped region.

    It does not land where it sat in the full frame.
    """
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)
    shapes = _SHAPES.validate_python(
        [
            {
                "kind": "point",
                "points": [[100, 50]],
                "color": "#FF0000",
                "tag": "gt",
            }
        ]
    )
    out = render_canvas_state(
        _pixels(img, region=(50, 0, 150, 100)),
        shapes,
        region=(50, 0, 150, 100),
        output_path=str(tmp_path / "crop.png"),
    )
    px = Image.open(out).convert("RGB")
    assert px.size == (100, 100)  # exactly the region handed in
    assert _red_over_green(px, (50, 50)) > 40  # native x=100 minus origin x=50
    assert (
        _red_over_green(px, (95, 50)) < 10
    )  # nothing where the unshifted one would hit


def test_render_scales_a_shape_with_the_pixels_it_is_drawn_on(
    tmp_path: Path,
) -> None:
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)
    shapes = _SHAPES.validate_python(
        [
            {
                "kind": "point",
                "points": [[100, 50]],
                "color": "#FF0000",
                "tag": "gt",
            }
        ]
    )
    out = render_canvas_state(
        _pixels(img, scale=0.5),
        shapes,
        region=_WHOLE,
        output_path=str(tmp_path / "half.png"),
    )
    px = Image.open(out).convert("RGB")
    assert px.size == (100, 50)
    assert (
        _red_over_green(px, (50, 25)) > 40
    )  # native (100, 50) at half resolution
    assert _red_over_green(px, (90, 25)) < 10


def test_render_two_pass_fill_does_not_erase_outlines(tmp_path: Path) -> None:
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)
    shapes = _SHAPES.validate_python(
        [
            {
                "kind": "box",
                "xyxy": [20, 20, 80, 80],
                "color": "#00FF00",
                "tag": "gt",
            },  # green outline
            # later red fill over it
            {
                "kind": "box",
                "xyxy": [10, 10, 90, 90],
                "color": "#FF0000",
                "fill": True,
                "tag": "gt",
            },
        ]
    )
    out = render_canvas_state(
        _pixels(img),
        shapes,
        region=_WHOLE,
        output_path=str(tmp_path / "overlap.png"),
    )
    px = (
        Image.open(out).convert("RGB").getpixel((50, 20))
    )  # a point on the green outline
    assert (
        px[1] > px[0]
    )  # outline survives the later overlapping fill (green-dominant)


def test_render_draws_a_point_shape_and_never_widens_it_to_a_box(
    tmp_path: Path,
) -> None:
    """A pushed point reaches the agent's view as a mark with a tick, and no box around it.

    The core sits on the coordinate and a radial tick lies above it; nothing is drawn 40 px
    away.
    """
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)
    shapes = _SHAPES.validate_python(
        [
            {
                "kind": "point",
                "points": [[100, 50]],
                "color": "#FF0000",
                "label": "tip",
                "tag": "gt",
            }
        ]
    )
    out = render_canvas_state(
        _pixels(img),
        shapes,
        region=_WHOLE,
        output_path=str(tmp_path / "point.png"),
    )
    px = Image.open(out).convert("RGB")
    assert (
        _red_over_green(px, (100, 50)) > 40
    )  # the core sits on the coordinate
    assert (
        _red_over_green(px, (100, 42)) > 40
    )  # a radial tick above it (the mark's reticle)
    assert (
        _red_over_green(px, (140, 50)) < 10
    )  # nothing 40px away: no box, fill or outline


def test_render_draws_the_focused_shape_with_a_halo_under_its_own_stroke(
    tmp_path: Path,
) -> None:
    """A focused shape renders with a halo beneath its stroke, which keeps the shape's own color.

    A focused shape reaches the agent's view with a wider white stroke beneath its own color, and
    an unfocused one with nothing around it.
    """
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)

    def rendered(focused: bool) -> Image.Image:
        halo = {"color": "#FFFFFF", "opacity": 0.55, "width_factor": 3}
        shapes = _SHAPES.validate_python(
            [
                {
                    "kind": "box",
                    "xyxy": [40, 20, 160, 80],
                    "color": "#FF0000",
                    "tag": "gt",
                    **({"halo": halo} if focused else {}),
                }
            ]
        )
        out = render_canvas_state(
            _pixels(img),
            shapes,
            region=_WHOLE,
            output_path=str(tmp_path / f"focus-{focused}.png"),
        )
        return Image.open(out).convert("RGB")

    plain, haloed = rendered(False), rendered(True)
    # One pixel outside the stroke the halo brightens every channel; the plain render does not.
    beside = (100, 19)
    assert min(haloed.getpixel(beside)) > min(plain.getpixel(beside)) + 40
    # The shape's own color still sits on the stroke itself.
    assert _red_over_green(haloed, (100, 20)) > 40


def test_render_draws_the_casing_under_every_outline_dashed_as_the_outline_is(
    tmp_path: Path,
) -> None:
    """A shape carrying a casing renders that darker stroke beside its own, inside the dashes.

    One pixel outside a solid stroke is darker with the casing than without; inside a dotted
    outline's gap the casing leaves the pixel alone, as the outline does.
    """
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)
    casing = {"color": "#000000", "opacity": 0.6, "width_factor": 3}

    def rendered(cased: bool, dashed: bool) -> Image.Image:
        shapes = _SHAPES.validate_python(
            [
                {
                    "kind": "box",
                    "xyxy": [40, 20, 160, 80],
                    "color": "#FF0000",
                    "tag": "gt",
                    **({"casing": casing} if cased else {}),
                    **({"dash": [2, 2]} if dashed else {}),
                }
            ]
        )
        out = render_canvas_state(
            _pixels(img),
            shapes,
            region=_WHOLE,
            output_path=str(tmp_path / f"casing-{cased}-{dashed}.png"),
        )
        return Image.open(out).convert("RGB")

    plain, cased = rendered(False, False), rendered(True, False)
    beside = (100, 19)
    assert max(cased.getpixel(beside)) < max(plain.getpixel(beside)) - 40
    assert _red_over_green(cased, (100, 20)) > 40
    dotted = rendered(True, True)
    # Each stroke width is one pixel here, so from the corner at 40 the top edge is two on, two
    # off: 45 sits in a dash, 43 in the gap before it.
    assert max(dotted.getpixel((45, 19))) < max(plain.getpixel((45, 19))) - 40
    assert (
        abs(max(dotted.getpixel((43, 19))) - max(plain.getpixel((43, 19))))
        < 10
    )


_MIRROR = json.loads(
    (
        Path(__file__).parent / "fixtures" / "canvas_mirror_shapes.json"
    ).read_text(encoding="utf-8")
)
"""The shapes the frontend's ``buildAnnotateShapes`` produced for a tool-authored polygon, a
tool-authored point and the first vertex of a polygon in progress."""
(TOOL_POLYGON,) = _MIRROR["tool_polygon"]
(TOOL_POINT,) = _MIRROR["tool_point"]
(FIRST_VERTEX,) = _MIRROR["first_vertex"]


def _wide_render(tmp_path: Path, shape: dict, *, dashed: bool) -> Image.Image:
    """Render ``shape`` on a 1400 by 200 px frame, its ``dash`` dropped unless ``dashed``."""
    from tcip_annotation.viz import render_canvas_state
    from tests._producer_fixtures import write_image

    img = str(write_image(tmp_path / "wide.jpg", (1400, 200), (90, 110, 90)))
    shape = {k: v for k, v in shape.items() if dashed or k != "dash"}
    out = render_canvas_state(
        _pixels(img),
        _SHAPES.validate_python([shape]),
        region=(0, 0, 1400, 200),
        output_path=str(tmp_path / f"wide-{dashed}.png"),
    )
    return Image.open(out).convert("RGB")


def _ink(rendered: Image.Image, xy: tuple[int, int]) -> int:
    """Return the largest channel difference between pixel ``xy`` and pixel (2, 197)."""
    return max(
        abs(a - b)
        for a, b in zip(rendered.getpixel(xy), rendered.getpixel((2, 197)))
    )


def _red_in_rows(rendered: Image.Image, x: int, rows: range) -> int:
    """Return the largest :func:`_ink` over the pixels of column ``x`` in ``rows``."""
    return max(_ink(rendered, (x, y)) for y in rows)


def test_render_lays_a_dash_down_in_stroke_widths_at_the_renders_own_stroke(
    tmp_path: Path,
) -> None:
    """A dash pattern is laid down in stroke widths at the render's own stroke.

    The two-pixel stroke of a 1400-pixel frame makes [2, 2] four pixels on, four off: the gap
    after a dash is dark where a solid outline lights it.
    """
    dashed = _wide_render(tmp_path, TOOL_POLYGON, dashed=True)
    solid = _wide_render(tmp_path, TOOL_POLYGON, dashed=False)
    rows = range(49, 52)
    assert _red_in_rows(dashed, 301, rows) > 40  # inside a dash
    assert _red_in_rows(dashed, 306, rows) < 10  # inside the gap after it
    assert _red_in_rows(dashed, 309, rows) > 40  # the next dash
    assert (
        _red_in_rows(solid, 306, rows) > 40
    )  # the gap's pixel on a solid outline


def test_render_dashes_a_points_ticks_as_it_does_any_outline(
    tmp_path: Path,
) -> None:
    """A tool's point renders dotted.

    The tick above the core has its one dash near the core and a gap beyond, which the solid
    render draws.
    """
    dashed = _wide_render(tmp_path, TOOL_POINT, dashed=True)
    solid = _wide_render(tmp_path, TOOL_POINT, dashed=False)
    assert _red_in_rows(dashed, 700, range(90, 91)) > 40
    assert _red_in_rows(dashed, 700, range(84, 85)) < 10
    assert _red_in_rows(solid, 700, range(84, 85)) > 40


def test_render_admits_the_one_vertex_polyline_a_drawing_in_progress_pushes(
    tmp_path: Path,
) -> None:
    from tcip_annotation.viz import render_canvas_state

    img = _make_image(tmp_path)
    out = render_canvas_state(
        _pixels(img),
        _SHAPES.validate_python([FIRST_VERTEX]),
        region=_WHOLE,
        output_path=str(tmp_path / "one-vertex.png"),
    )
    drawn = Image.open(out).convert("RGB")
    assert (
        max(
            abs(a - b)
            for a, b in zip(drawn.getpixel((60, 40)), drawn.getpixel((10, 90)))
        )
        > 40
    )


def test_render_dots_a_polylines_vertices_unless_it_says_it_is_a_tail(
    tmp_path: Path,
) -> None:
    """A draft's laid stroke inks beside its second vertex; with ``vertices=False`` it does not."""

    def stroke(**over: bool) -> dict:
        return {
            "kind": "polyline",
            "points": [[300, 100], [500, 100]],
            "color": "#00CED1",
            "tag": "in_progress",
            **over,
        }

    laid = _wide_render(tmp_path, stroke(), dashed=True)
    tail = _wide_render(tmp_path, stroke(vertices=False), dashed=True)
    assert _ink(laid, (500, 96)) > 40
    assert _ink(tail, (500, 96)) < 10


@pytest.mark.parametrize(
    "dash",
    [
        [0, 4],
        [4],
        "xx",
        [4, -1],
        [],
        0,
        False,
        "",
        ["1", "3"],
        "13",
        [True, False],
        [1, 3, 5],
        [float("nan"), 3],
        [1, float("inf")],
    ],
    ids=[
        "zero_on",
        "one_length",
        "text",
        "negative_off",
        "empty",
        "zero",
        "false",
        "empty_text",
        "numeric_strings",
        "digit_string",
        "booleans",
        "three_lengths",
        "nan_on",
        "infinite_off",
    ],
)
def test_a_shape_whose_dash_is_not_a_pattern_is_refused_naming_it(
    dash: list[int] | list[float] | list[str] | list[bool] | str | int,
) -> None:
    shape = {
        "kind": "box",
        "xyxy": [10, 10, 40, 40],
        "color": "#FF0000",
        "tag": "gt",
        "dash": dash,
    }
    with pytest.raises(ValidationError) as refused:
        _SHAPES.validate_python([shape])

    assert {e["loc"][:3] for e in refused.value.errors()} == {
        (0, "box", "dash")
    }


@pytest.mark.parametrize(
    ("bad", "field"),
    [
        ({"kind": "box", "color": "#FF0000", "tag": "gt"}, "xyxy"),
        (
            {
                "kind": "polygon",
                "points": [[1, 1]],
                "color": "#FF0000",
                "tag": "gt",
            },
            "points",
        ),
        (
            {
                "kind": "box",
                "xyxy": [10, 10, 40, 40],
                "color": "not-a-color",
                "tag": "gt",
            },
            "color",
        ),
        ({"kind": "box", "xyxy": [10, 10, 40, 40], "tag": "gt"}, "color"),
    ],
    ids=[
        "box_without_corners",
        "one_point_polygon",
        "unparsable_color",
        "no_color",
    ],
)
def test_a_malformed_shape_is_refused_naming_its_field(
    bad: dict, field: str
) -> None:
    """A malformed canvas shape is refused naming its index and field; a well-formed one reads."""
    good = {
        "kind": "box",
        "xyxy": [10, 10, 40, 40],
        "color": "#0f0",
        "tag": "gt",
    }
    assert len(_SHAPES.validate_python([good])) == 1

    with pytest.raises(ValidationError) as refused:
        _SHAPES.validate_python([good, bad])

    assert {(e["loc"][0], e["loc"][2]) for e in refused.value.errors()} == {
        (1, field)
    }


# ── MCP tool ────────────────────────────────────────────────────────────────


def _write_state(
    project: Path,
    img: str,
    shapes: list[dict] | None = SHAPES,
    *,
    shapes_image: str | None = None,
    tab: str = "annotate",
    shapes_tab: str | None = None,
    cut_armed: bool = False,
) -> None:
    """Write the canvas geometry document, unless ``shapes`` is ``None``, and the meta document.

    Each image path is spelled against the project through ``stored_path``; the meta carries a
    200 by 100 viewport at the origin.
    """
    from tcip_mcp.audit import now_iso
    from tcip_mcp.registry_paths import stored_path
    from tcip_mcp.web_client import CanvasGeometry, CanvasMeta, Viewport

    root = str(project)
    stored = stored_path(img, project)
    now = now_iso()
    if shapes is not None:
        geometry = CanvasGeometry(
            image_path=stored_path(shapes_image, project)
            if shapes_image
            else stored,
            tab=shapes_tab or tab,
            shapes=_SHAPES.validate_python(shapes),
            received_at=now,
        )
        tcip_store.replace(
            canvas_geometry_key(root), geometry.model_dump(mode="json")
        )
    meta = CanvasMeta(
        received_at=now,
        tab=tab,
        image=Path(img).name,
        image_path=stored,
        img_width=200,
        img_height=100,
        viewport=Viewport(x=0, y=0, w=200, h=100, scale=1.0),
        user="breeder",
        mode="polygon",
        active_subject=None,
        cut_armed=cut_armed,
        dirty=False,
        classes=[{"id": 0, "name": "bud", "color": "#FF0000"}],
        counts={},
    )
    tcip_store.replace(canvas_meta_key(root), meta.model_dump(mode="json"))


def test_capture_live_canvas_with_no_state_pushed_names_what_pushes_one(
    project: Path, bound: BoundProject
) -> None:
    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(bound, refresh=False)
    assert "error" in res
    assert "no canvas state has been pushed" in res["error"].lower()


def test_capture_live_canvas_renders_pushed_state(
    project: Path, bound: BoundProject
) -> None:
    img = _make_image(project)
    _write_state(project, img)

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(bound, refresh=False)
    assert "error" not in res
    assert Path(res["image_path"]).is_file()
    assert res["source_image"] == img
    assert res["classes"][0]["name"] == "bud"
    assert res["shape_counts_by_tag"] == {"gt": 2, "in_progress": 1}
    assert res["state_age_seconds"] >= 0
    assert res["shapes_missing"] is False


def test_capture_live_canvas_names_the_armed_cut(
    project: Path, bound: BoundProject
) -> None:
    """The live canvas capture reports the armed cut its meta document records."""
    img = _make_image(project)
    _write_state(project, img, cut_armed=True)

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(bound, refresh=False)
    assert res["cut_armed"] is True


def test_capture_live_canvas_renders_exactly_the_viewport_region(
    project: Path, bound: BoundProject
) -> None:
    """The live canvas capture reports itself cropped to the viewport, at the viewport's size.

    The artifact of a 100 px viewport is 100 by 100.
    """
    img = _make_image(project)
    _write_state(project, img)
    from tcip_mcp.web_client import Viewport

    live = read_canvas_meta(str(project))
    assert live is not None
    moved = live.model_copy(
        update={"viewport": Viewport(x=50, y=0, w=100, h=100, scale=1.0)}
    )
    tcip_store.replace(
        canvas_meta_key(str(project)), moved.model_dump(mode="json")
    )

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(bound, refresh=False)
    assert res["cropped_to_viewport"] is True
    assert Image.open(res["image_path"]).size == (100, 100)


def test_capture_live_canvas_full_frame_downscales_to_max_edge(
    project: Path, bound: BoundProject
) -> None:
    img = _make_image(project)
    _write_state(project, img)

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(
        bound, refresh=False, crop_to_viewport=False, max_edge=100
    )
    assert res["cropped_to_viewport"] is False
    assert Image.open(res["image_path"]).size == (100, 50)


def test_capture_live_canvas_reads_a_multiband_raster_without_writing_a_preview(
    project: Path, bound: BoundProject
) -> None:
    """A six-band raster captures as an RGB artifact, with no band-preview directory written."""
    import numpy as np
    import tifffile

    images = image_root(project)
    images.mkdir()
    rng = np.random.default_rng(3)
    src = images / "capture.tif"
    tifffile.imwrite(
        str(src), rng.integers(0, 4096, size=(100, 200, 6)).astype(np.uint16)
    )
    _write_state(project, str(src))

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(bound, refresh=False)
    assert "error" not in res
    assert Image.open(res["image_path"]).mode == "RGB"
    assert not (
        project / ".tcip" / "artifacts" / "viz" / "_band_previews"
    ).exists()


def test_capture_live_canvas_identity_stale_shapes_do_not_render(
    project: Path, bound: BoundProject
) -> None:
    """Geometry left over from a previous image is reported missing under the current one.

    The capture renders the image, with no shape counted.
    """
    img = _make_image(project)
    _write_state(
        project, img, shapes_image="C:/img/other.jpg"
    )  # stale identity

    from tcip_mcp.tools.vision_tools import capture_live_canvas

    res = capture_live_canvas(bound, refresh=False)
    assert "error" not in res
    assert res["shapes_missing"] is True
    assert res["shape_counts_by_tag"] == {}
    assert Path(
        res["image_path"]
    ).is_file()  # still renders the image + viewport

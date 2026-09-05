"""The live-canvas capture: where the human's shapes land, and whether the state is fresh.

`capture_live_canvas` is the agent's only view of work that exists nowhere on disk yet, so two
things have to hold. The shapes have to be drawn at the resolution the pixels underneath them
were served at, and a capture that no GUI answered has to say so rather than present a stale
push as the canvas the human is looking at now.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image

#: Neutral gray, so a drawn color's dominance is the same size in either channel.
BACKGROUND = (100, 100, 100)
#: The source frame; served at half resolution below, which is what places the shapes.
FRAME_W, FRAME_H = 400, 200
SERVED_MAX_EDGE = 200

#: A point at its native coordinate, which is itself inside the served frame, and where a
#: half-resolution render must put it.
POINT_NATIVE = (160, 20)
POINT_AT_SERVED_SCALE = (80, 10)

#: A filled box and one interior pixel of each of its two possible placements.
BOX_NATIVE = [20, 40, 100, 90]
BOX_INSIDE_AT_SERVED_SCALE = (30, 32)
BOX_INSIDE_AT_NATIVE_SCALE = (60, 70)

SHAPES = [
    {"kind": "point", "points": [list(POINT_NATIVE)], "color": "#FF0000", "tag": "gt"},
    {"kind": "box", "xyxy": BOX_NATIVE, "color": "#00FF00", "fill": True, "tag": "gt"},
]


def _canvas_image(tmp_path: Path) -> str:
    path = tmp_path / "canvas.png"
    Image.new("RGB", (FRAME_W, FRAME_H), BACKGROUND).save(path)
    return str(path)


def _mint_binding(tmp_path: Path) -> None:
    """Bind ``tmp_path`` as the GUI's open root: the reader refuses to render with none."""
    import tcip_store
    from tcip_mcp.web_client import canvas_open_binding_key

    key = canvas_open_binding_key()
    stored = tcip_store.read_versioned(key, default=None)
    tcip_store.replace(key, {
        "generation": 1, "root": str(tmp_path), "project_name": None,
        "issued_at": "2026-01-01T00:00:00+00:00",
    }, expect=stored.version)


def _push_state(tmp_path: Path, image: str, *, received_at: float,
                shapes: list[dict] | None = None) -> None:
    """Write the two documents the GUI pushes: the meta heartbeat and the geometry blob."""
    import tcip_store
    from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key

    root = str(tmp_path)
    tcip_store.replace(canvas_geometry_key(root), {
        "image_path": image, "tab": "annotate", "received_at": received_at,
        "shapes": SHAPES if shapes is None else shapes,
    })
    tcip_store.replace(canvas_meta_key(root), {
        "received_at": received_at, "project_root": root, "tab": "annotate",
        "image": Path(image).name, "image_path": image,
        "viewport": {"x": 0, "y": 0, "w": FRAME_W, "h": FRAME_H},
        "user": "breeder", "mode": "polygon",
        "classes": [{"id": 0, "name": "bud", "color": "#FF0000"}],
    })


def _dominance(px: Image.Image, xy: tuple[int, int], channel: int) -> int:
    """How far ``channel`` rises above the other two at one pixel of the rendered artifact."""
    values = px.getpixel(xy)
    others = [v for i, v in enumerate(values) if i != channel]
    return values[channel] - max(others)


def test_canvas_shapes_are_drawn_at_the_resolution_the_pixels_were_served_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame read down to ``max_edge`` carries its shapes down with it.

    Shape coordinates arrive in the raster's native grid, so drawing them on reduced pixels
    without that same reduction shows the agent every polygon, box and point enlarged and
    displaced while the tool reports it as the human's live canvas.
    """
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp.tools.vision_tools import capture_live_canvas

    image = _canvas_image(tmp_path)
    _push_state(tmp_path, image, received_at=time.time())
    _mint_binding(tmp_path)

    result = capture_live_canvas(refresh=False, crop_to_viewport=False,
                                 max_edge=SERVED_MAX_EDGE)
    assert "error" not in result, result
    assert result["shapes_missing"] is False

    px = Image.open(result["image_path"]).convert("RGB")
    assert px.size == (FRAME_W // 2, FRAME_H // 2)

    assert _dominance(px, POINT_AT_SERVED_SCALE, 0) > 40
    assert _dominance(px, POINT_NATIVE, 0) < 15
    assert _dominance(px, BOX_INSIDE_AT_SERVED_SCALE, 1) > 20
    assert _dominance(px, BOX_INSIDE_AT_NATIVE_SCALE, 1) < 10


def test_a_capture_no_gui_answered_reports_the_state_as_last_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ping that reaches the hub but produces no new push leaves the capture unrefreshed.

    The one thing this tool knows that no file on disk does is whether the canvas it drew is
    what the human is looking at right now. Reading an unchanged heartbeat as a fresh answer
    presents a stale canvas, and its stale shape counts, as the live one.
    """
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp import web_client
    from tcip_mcp.tools.vision_tools import capture_live_canvas

    image = _canvas_image(tmp_path)
    _push_state(tmp_path, image, received_at=time.time() - 600)
    _mint_binding(tmp_path)

    pings: list[tuple[str, str]] = []

    def silent_hub(panel: str, event_type: str, data: dict, **kwargs: object) -> dict:
        pings.append((panel, event_type))
        return {"status": "ok", "delivered": True}

    monkeypatch.setattr(web_client, "post_panel_event", silent_hub)

    result = capture_live_canvas(refresh=True, crop_to_viewport=False)
    assert "error" not in result, result
    assert pings == [("app", "canvas_state_request")]
    assert result["refresh_ping_delivered"] is True
    assert result["refreshed"] is False
    assert result["state_age_seconds"] > 5
    assert "last known" in result["summary"]


def test_a_capture_the_gui_answered_reports_the_state_as_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GUI that pushes fresh state in response to the ping is reported as live: the refresh
    round trip has to admit the answered case, not only flag the unanswered one."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    from tcip_mcp import web_client
    from tcip_mcp.tools.vision_tools import capture_live_canvas

    image = _canvas_image(tmp_path)
    _push_state(tmp_path, image, received_at=time.time() - 600)
    _mint_binding(tmp_path)

    def answering_hub(panel: str, event_type: str, data: dict, **kwargs: object) -> dict:
        _push_state(tmp_path, image, received_at=time.time())
        return {"status": "ok", "delivered": True}

    monkeypatch.setattr(web_client, "post_panel_event", answering_hub)

    result = capture_live_canvas(refresh=True, crop_to_viewport=False)
    assert "error" not in result, result
    assert result["refreshed"] is True
    assert "last known" not in result["summary"]
    assert result["shape_counts_by_tag"] == {"gt": 2}

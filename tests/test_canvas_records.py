"""Tests for the canvas meta and geometry records read through their models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import tcip_store
from tcip_mcp.web_client import canvas_geometry_key, canvas_meta_key

if TYPE_CHECKING:
    from tcip_web.state import OpenProject

SHAPES = [
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
    },
]


def _body(project_id: str, image: Path, **fields: object) -> dict:
    """Return a canvas push for ``project_id`` showing ``image``, every field the GUI states."""
    return {
        "project_id": project_id,
        "tab": "annotate",
        "image_path": str(image),
        "image": image.name,
        "img_width": 200,
        "img_height": 100,
        "viewport": None,
        "mode": "box",
        "active_subject": None,
        "cut_armed": False,
        "dirty": False,
        "user": "breeder",
        "classes": [],
        "counts": {},
        "shapes": None,
        **fields,
    }


def _push(
    client: TestClient, project_id: str, image: Path, **fields: object
) -> dict:
    """Post one canvas push (:func:`_body`) and return the posted body."""
    body = _body(project_id, image, **fields)
    resp = client.post("/api/canvas/state", json=body)
    assert resp.status_code == 200, resp.text
    return body


def test_the_records_a_full_push_writes_read_back_through_their_models_as_pushed(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_mcp.web_client import (
        CanvasView,
        read_canvas_geometry,
        read_canvas_meta,
    )

    root = str(opened.root)
    pushed = _push(
        client,
        opened.id,
        opened.root / "images" / "a.jpg",
        img_width=200,
        img_height=100,
        viewport={"x": 0, "y": 0, "w": 200, "h": 100, "scale": 1.5},
        mode="polygon",
        active_subject="leaf",
        cut_armed=True,
        dirty=False,
        classes=[{"id": 0, "name": "leaf", "color": "#00FF00"}],
        counts={"boxes": 1, "polygons": 0},
        shapes=SHAPES,
    )

    meta = read_canvas_meta(root)
    geometry = read_canvas_geometry(root)

    assert meta is not None and geometry is not None
    assert meta.model_dump(mode="json") == tcip_store.read(
        canvas_meta_key(root)
    )
    assert geometry.model_dump(mode="json") == tcip_store.read(
        canvas_geometry_key(root)
    )
    restated = set(CanvasView.model_fields) - {"image_path", "user"}
    assert {
        name: value
        for name, value in meta.model_dump(mode="json").items()
        if name in restated
    } == {name: pushed[name] for name in restated}
    assert (meta.image_path, meta.user) == ("images/a.jpg", "user:breeder")
    assert (
        geometry.image_path,
        geometry.tab,
        geometry.model_dump(mode="json", exclude_defaults=True)["shapes"],
    ) == (meta.image_path, meta.tab, SHAPES)
    assert geometry.received_at == meta.received_at


def test_a_push_omitting_its_counts_is_refused_and_writes_nothing(
    client: TestClient, opened: OpenProject
) -> None:
    """A push that leaves out a field the GUI states answers 422 naming it, and stores nothing."""
    from tcip_mcp.web_client import read_canvas_meta

    body = _body(opened.id, opened.root / "images" / "b.jpg")
    del body["counts"]

    resp = client.post("/api/canvas/state", json=body)

    assert resp.status_code == 422
    assert [e["loc"][-1] for e in resp.json()["detail"]] == ["counts"]
    assert read_canvas_meta(str(opened.root)) is None


def test_a_heartbeat_is_stored_whole_and_leaves_the_geometry_absent(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_mcp.web_client import read_canvas_geometry, read_canvas_meta

    root = str(opened.root)
    _push(client, opened.id, opened.root / "images" / "b.jpg")

    meta = read_canvas_meta(root)

    assert meta is not None
    assert meta.image_path == "images/b.jpg"
    assert read_canvas_geometry(root) is None


def test_a_stored_meta_record_missing_its_arrival_instant_is_refused_naming_it(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_mcp.web_client import read_canvas_meta

    root = str(opened.root)
    _push(client, opened.id, opened.root / "images" / "a.jpg")
    raw = tcip_store.read(canvas_meta_key(root))
    assert isinstance(raw, dict)
    del raw["received_at"]
    tcip_store.replace(canvas_meta_key(root), raw)

    with pytest.raises(ValidationError) as refused:
        read_canvas_meta(root)

    (error,) = refused.value.errors()
    assert (error["loc"], error["type"]) == (("received_at",), "missing")


def test_a_stored_meta_record_missing_its_dirty_flag_is_refused_naming_it(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_mcp.web_client import read_canvas_meta

    root = str(opened.root)
    _push(client, opened.id, opened.root / "images" / "a.jpg")
    raw = tcip_store.read(canvas_meta_key(root))
    assert isinstance(raw, dict)
    del raw["dirty"]
    tcip_store.replace(canvas_meta_key(root), raw)

    with pytest.raises(ValidationError) as refused:
        read_canvas_meta(root)

    (error,) = refused.value.errors()
    assert (error["loc"], error["type"]) == (("dirty",), "missing")


def test_a_stored_geometry_record_missing_its_shapes_is_refused_naming_them(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_mcp.web_client import read_canvas_geometry

    root = str(opened.root)
    _push(client, opened.id, opened.root / "images" / "a.jpg", shapes=SHAPES)
    raw = tcip_store.read(canvas_geometry_key(root))
    assert isinstance(raw, dict)
    del raw["shapes"]
    tcip_store.replace(canvas_geometry_key(root), raw)

    with pytest.raises(ValidationError) as refused:
        read_canvas_geometry(root)

    (error,) = refused.value.errors()
    assert (error["loc"], error["type"]) == (("shapes",), "missing")


def test_a_stored_geometry_record_whose_shape_states_no_kind_is_refused_naming_it(
    client: TestClient, opened: OpenProject
) -> None:
    from tcip_mcp.web_client import read_canvas_geometry

    root = str(opened.root)
    _push(client, opened.id, opened.root / "images" / "a.jpg", shapes=SHAPES)
    raw = tcip_store.read(canvas_geometry_key(root))
    assert isinstance(raw, dict) and isinstance(raw["shapes"], list)
    first = raw["shapes"][0]
    assert isinstance(first, dict)
    del first["kind"]
    tcip_store.replace(canvas_geometry_key(root), raw)

    with pytest.raises(ValidationError) as refused:
        read_canvas_geometry(root)

    (error,) = refused.value.errors()
    assert error["loc"] == ("shapes", 0)
    assert error["type"] == "union_tag_not_found"

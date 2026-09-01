"""Path-traversal validation: allowed-root image guard + route-level run_id/path guards.

(safe_join itself is covered by test_tcip_web_routes.py::TestSafeJoin: under-root, parent-traversal,
absolute, forward-slashes; so it is not re-tested here.)
"""

import pytest


def test_assert_path_allowed_admits_the_workspace_and_refuses_a_sibling(
    tmp_path, tmp_path_factory: pytest.TempPathFactory
):
    from tcip_web.paths import assert_path_allowed
    target = tmp_path / "data" / "img.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    assert assert_path_allowed(str(target)) == target.resolve()  # under the workspace -> allowed

    outside = tmp_path_factory.mktemp("outside") / "img.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        assert_path_allowed(str(outside))                        # a sibling workspace -> refused


def test_assert_project_root_allowed_matches_assert_path_allowed(
    tmp_path, tmp_path_factory: pytest.TempPathFactory
):
    from tcip_web.paths import assert_path_allowed, assert_project_root_allowed
    target = tmp_path / "proj"
    target.mkdir()

    assert assert_project_root_allowed(str(target)) == target.resolve()

    outside = tmp_path_factory.mktemp("outside")
    with pytest.raises(ValueError):
        assert_project_root_allowed(str(outside))
    with pytest.raises(ValueError):
        assert_path_allowed(str(outside))  # same policy as the generic guard


def test_training_stream_closes_on_run_id_traversal(tmp_path):
    """A run_id carrying a path separator (BadKey) closes the stream rather than resolving to a
    path component; the WS surface is where this parameter is served now. A backslash, not a
    ``..`` segment, since a URL client normalizes dot segments before the request is even sent.
    The endpoint's own code checks project_root before accepting the socket and run_id only
    after, so the socket connects cleanly here and the disconnect surfaces on the first read."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from tcip_web.app import app

    client = TestClient(app, base_url="http://127.0.0.1")
    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(
            f"ws://127.0.0.1/api/training/runs/a\\b/stream?project_root={tmp_path}",
        ) as ws:
            ws.receive_json()
    assert ei.value.code == 1008
    assert "is not a single name" in ei.value.reason


def test_training_stream_refuses_a_project_root_outside_allowed_roots(
    tmp_path_factory: pytest.TempPathFactory
):
    """The stream must confine project_root the same way the identical parameter is confined
    on meta.py's report routes. The endpoint's own code checks project_root before accepting the
    socket, so nothing is replayed before the refusal."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from tcip_web.app import app

    client = TestClient(app, base_url="http://127.0.0.1")
    outside = tmp_path_factory.mktemp("outside")
    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(
            f"ws://127.0.0.1/api/training/runs/run-1/stream?project_root={outside}",
        ):
            pass
    assert ei.value.code == 1008
    assert "outside the allowed roots" in ei.value.reason


def test_annotate_labels_route_blocks_outside_allowed_root(
    tmp_path, tmp_path_factory: pytest.TempPathFactory
):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from PIL import Image

    from tcip_web.routes.annotate import load_labels

    img = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8)).save(img)
    outside = tmp_path_factory.mktemp("outside") / "secret.jpg"
    Image.new("RGB", (8, 8)).save(outside)

    assert load_labels(str(img))["img_width"] == 8       # inside the workspace -> served
    with pytest.raises(HTTPException) as ei:
        load_labels(str(outside))                         # a sibling workspace -> 403
    assert ei.value.status_code == 403


def test_images_route_blocks_outside_allowed_root(
    tmp_path, tmp_path_factory: pytest.TempPathFactory
):
    """images.py's own ``_checked`` guard, exercised on a route that still calls it (the annotate
    route above guards a different module's copy)."""
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from PIL import Image

    from tcip_web.routes.images import get_bands

    img = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8)).save(img)
    outside = tmp_path_factory.mktemp("outside") / "secret.jpg"
    Image.new("RGB", (8, 8)).save(outside)

    assert get_bands(path=str(img))["band_count"] == 3    # inside the workspace -> served
    with pytest.raises(HTTPException) as ei:
        get_bands(path=str(outside))                       # a sibling workspace -> 403
    assert ei.value.status_code == 403

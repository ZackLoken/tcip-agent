"""Tests of path-traversal validation.

Covered: the allowed-root image guard and the route-level run id and path guards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


def test_assert_path_allowed_admits_the_workspace_and_refuses_a_sibling(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    from tcip_web.paths import assert_path_allowed

    target = tmp_path / "data" / "img.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    assert (
        assert_path_allowed(str(target)) == target.resolve()
    )  # under the workspace -> allowed

    outside = tmp_path_factory.mktemp("outside") / "img.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        assert_path_allowed(str(outside))  # a sibling workspace -> refused


def test_training_stream_closes_on_run_id_traversal(
    opened_project: Path, client: TestClient
) -> None:
    """A run_id carrying a path separator closes the training stream.

    For a run_id holding a backslash, the stream closes with code 1008 naming the run_id as not a
    single directory name.
    """
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as ei:
        with client.websocket_connect(
            "ws://127.0.0.1/api/training/runs/a\\b/stream"
        ) as ws:
            ws.receive_json()
    assert ei.value.code == 1008
    assert "is not a single directory name" in ei.value.reason


def test_annotate_labels_route_blocks_outside_allowed_root(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
    from tcip_web.routes.annotate import load_labels
    from tests._producer_fixtures import write_image

    img = write_image(image_dir(tmp_path, UNDATED_BUCKET) / "ok.jpg", (8, 8))
    outside = write_image(
        tmp_path_factory.mktemp("outside") / "secret.jpg", (8, 8)
    )

    assert (
        load_labels(str(img))["img_width"] == 8
    )  # inside the workspace -> served
    with pytest.raises(HTTPException) as ei:
        load_labels(str(outside))  # a sibling workspace -> 403
    assert ei.value.status_code == 403


def test_images_route_blocks_outside_allowed_root(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The images route serves a path inside the workspace and refuses one outside with 403.

    The guard exercised is images.py's own ``_checked``, through ``get_bands``.
    """
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.images import get_bands
    from tests._producer_fixtures import write_image

    img = write_image(tmp_path / "ok.jpg", (8, 8))
    outside = write_image(
        tmp_path_factory.mktemp("outside") / "secret.jpg", (8, 8)
    )

    assert (
        get_bands(path=str(img))["band_count"] == 3
    )  # inside the workspace -> served
    with pytest.raises(HTTPException) as ei:
        get_bands(path=str(outside))  # a sibling workspace -> 403
    assert ei.value.status_code == 403

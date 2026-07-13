"""Phase 4.4 — path-traversal validation: allowed-root image guard + route-level run_id/path guards.

(safe_join itself is covered by test_tcip_web_routes.py::TestSafeJoin — under-root, parent-traversal,
absolute, forward-slashes — so it is not re-tested here.)
"""

import pytest


def test_assert_path_allowed_permissive_then_restricted(tmp_path, monkeypatch):
    from tcip_web.paths import assert_path_allowed
    target = tmp_path / "data" / "img.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    assert assert_path_allowed(str(target)) == target.resolve()        # unset -> permissive

    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path / "other"))
    with pytest.raises(ValueError):
        assert_path_allowed(str(target))                                # outside the allow-list

    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path / "data"))
    assert assert_path_allowed(str(target)) == target.resolve()        # inside -> allowed


def test_training_metrics_rejects_run_id_traversal(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import get_run_metrics
    with pytest.raises(HTTPException) as ei:
        get_run_metrics(project_root=str(tmp_path), run_id="../../../../etc")
    assert ei.value.status_code == 400


def test_images_route_blocks_outside_allowed_root(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from PIL import Image

    from tcip_web.routes.images import get_dimensions

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    img = allowed / "ok.jpg"
    Image.new("RGB", (8, 8)).save(img)
    outside = tmp_path / "secret.jpg"
    Image.new("RGB", (8, 8)).save(outside)

    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    assert get_dimensions(str(img))["width"] == 8       # inside the root -> served
    with pytest.raises(HTTPException) as ei:
        get_dimensions(str(outside))                     # outside the root -> 403
    assert ei.value.status_code == 403

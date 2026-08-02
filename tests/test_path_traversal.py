"""Path-traversal validation: allowed-root image guard + route-level run_id/path guards.

(safe_join itself is covered by test_tcip_web_routes.py::TestSafeJoin: under-root, parent-traversal,
absolute, forward-slashes; so it is not re-tested here.)
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


def test_assert_project_root_allowed_matches_assert_path_allowed(tmp_path, monkeypatch):
    from tcip_web.paths import assert_path_allowed, assert_project_root_allowed
    target = tmp_path / "proj"
    target.mkdir()

    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    assert assert_project_root_allowed(str(target)) == target.resolve()

    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(tmp_path / "other"))
    with pytest.raises(ValueError):
        assert_project_root_allowed(str(target))
    with pytest.raises(ValueError):
        assert_path_allowed(str(target))  # same policy as the generic guard


def test_training_metrics_rejects_run_id_traversal(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import get_run_metrics
    with pytest.raises(HTTPException) as ei:
        get_run_metrics(project_root=str(tmp_path), run_id="../../../../etc")
    assert ei.value.status_code == 400


def test_training_metrics_confines_project_root_to_allowed_roots(tmp_path, monkeypatch):
    # get_run_metrics must confine project_root the same way the identical parameter is confined
    # on meta.py's report routes.
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import get_run_metrics

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(HTTPException) as ei:
        get_run_metrics(project_root=str(outside), run_id="run-1")
    assert ei.value.status_code == 403


def test_training_launch_confines_output_dir_to_allowed_roots(tmp_path, monkeypatch):
    # launch_training_route must guard output_dir the same way the sibling tuning.py launch route
    # does, not pass it straight to launch_training unguarded.
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import LaunchPayload, launch_training_route

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(HTTPException) as ei:
        launch_training_route(LaunchPayload(config={}, output_dir=str(outside)))
    assert ei.value.status_code == 403


def test_training_launch_output_dir_guard_is_a_no_op_when_unrestricted(tmp_path, monkeypatch):
    # The rail must admit valid work, not only reject invalid work: with TCIP_IMAGE_ROOTS unset
    # (the default), an unconfined output_dir must reach launch_training as before, not 403:
    # launch_training itself then reports an invalid (model_source-less) config as a normal
    # {"error": ...} result, not an exception, which is exactly the point: the guard lets the
    # request past it to reach that existing behavior unchanged.
    from tcip_web.routes.training import LaunchPayload, launch_training_route

    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    result = launch_training_route(LaunchPayload(config={}, output_dir=str(tmp_path)))
    assert result["error"] == "Invalid config"


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

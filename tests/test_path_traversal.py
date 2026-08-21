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


def test_training_metrics_rejects_run_id_traversal(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import get_run_metrics
    with pytest.raises(HTTPException) as ei:
        get_run_metrics(project_root=str(tmp_path), run_id="../../../../etc")
    assert ei.value.status_code == 400


def test_training_metrics_confines_project_root_to_allowed_roots(
    tmp_path_factory: pytest.TempPathFactory
):
    # get_run_metrics must confine project_root the same way the identical parameter is confined
    # on meta.py's report routes.
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import get_run_metrics

    outside = tmp_path_factory.mktemp("outside")
    with pytest.raises(HTTPException) as ei:
        get_run_metrics(project_root=str(outside), run_id="run-1")
    assert ei.value.status_code == 403


def test_training_launch_confines_output_dir_to_allowed_roots(
    tmp_path_factory: pytest.TempPathFactory
):
    # launch_training_route must guard output_dir the same way the sibling tuning.py launch route
    # does, not pass it straight to launch_training unguarded.
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from tcip_web.routes.training import LaunchPayload, launch_training_route

    outside = tmp_path_factory.mktemp("outside")
    with pytest.raises(HTTPException) as ei:
        launch_training_route(LaunchPayload(config={}, output_dir=str(outside)))
    assert ei.value.status_code == 403


def test_training_launch_output_dir_guard_is_a_no_op_when_unrestricted(tmp_path):
    """The rail must admit valid work: an output_dir under the workspace clears the guard and
    reaches launch_training as before, which then reports an invalid (model_source-less) config
    as a normal {"error": ...} result, not an exception raised by the guard itself."""
    from tcip_web.routes.training import LaunchPayload, launch_training_route

    result = launch_training_route(LaunchPayload(config={}, output_dir=str(tmp_path)))
    assert result["error"] == "Invalid config"


def test_images_route_blocks_outside_allowed_root(tmp_path, tmp_path_factory: pytest.TempPathFactory):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from PIL import Image

    from tcip_web.routes.images import get_dimensions

    img = tmp_path / "ok.jpg"
    Image.new("RGB", (8, 8)).save(img)
    outside = tmp_path_factory.mktemp("outside") / "secret.jpg"
    Image.new("RGB", (8, 8)).save(outside)

    assert get_dimensions(str(img))["width"] == 8       # inside the workspace -> served
    with pytest.raises(HTTPException) as ei:
        get_dimensions(str(outside))                     # a sibling workspace -> 403
    assert ei.value.status_code == 403

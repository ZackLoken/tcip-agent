"""tcip preflight-config: the demoted door's own command-line entry point.

Structural validation always runs, for the project --project names.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests._cli_fixtures import run_tcip


def _fixture_config(root: Path) -> Path:
    """A valid config: a builder the operator's own process imports (never called, since no
    --smoke is passed here) over two labeled frames of the subject its scope names
    (``_verified_checkpoint_fixtures.detection_images``)."""
    from tests._verified_checkpoint_fixtures import SCOPED_DATA, detection_images

    config = {
        "model_source": {"builder": "tcip_mcp.pipelines.model_build:build_model",
                         "task": "detection"},
        "data": {**detection_images(root / "data", SCOPED_DATA["scope"]), **SCOPED_DATA},
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_refuses_a_run_naming_no_project(tmp_path):
    config_path = _fixture_config(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip("preflight-config", ["--config", str(config_path)], cwd=cwd)

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_validates_a_fixture_config_for_the_named_project(project, tmp_path):
    config_path = _fixture_config(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = run_tcip("preflight-config", ["--config", str(config_path), "--project", str(project)], cwd=cwd)

    assert result.returncode == 0, result.stdout
    body = json.loads(result.stdout)
    assert body["valid"] is True, body["issues"]
    assert body["issues"] == []

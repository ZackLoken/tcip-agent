"""tcip preflight-config: the demoted door's own command-line entry point.

Structural validation always runs, root pinning through require_platform_root
(test_script_root_pinning.py covers the shared mechanism directly), the same pinning
tcip calibrate-operating-point uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _fixture_config(tmp_path: Path) -> Path:
    """A structurally valid config: a real, importable builder (never called, since no --smoke
    is passed here) and real, empty images/labels directories."""
    imgs = tmp_path / "images"
    lbls = tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    config = {
        "model_source": {"builder": "tcip_mcp.pipelines.model_build:build_model",
                         "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _run(args: list[str], cwd: Path, platform_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if platform_root is None:
        env.pop("TCIP_STATE_ROOT", None)
    else:
        env["TCIP_STATE_ROOT"] = platform_root
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "preflight-config", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def test_refuses_a_run_with_no_platform_root_pinned(tmp_path):
    config_path = _fixture_config(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(["--config", str(config_path)], cwd=cwd, platform_root=None)

    assert result.returncode != 0, result.stdout
    assert "TCIP_STATE_ROOT" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_validates_a_fixture_config_over_a_pinned_platform_root(tmp_path):
    config_path = _fixture_config(tmp_path)
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()
    platform_root = tmp_path / "platform"
    platform_root.mkdir()

    result = _run(["--config", str(config_path)], cwd=cwd, platform_root=str(platform_root))

    assert result.returncode == 0, result.stdout
    body = json.loads(result.stdout)
    assert body["valid"] is True, body["issues"]
    assert body["issues"] == []

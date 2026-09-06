"""tcip triage-predictions: the demoted door's own command-line entry point.

--project (or $TCIP_STATE_ROOT) is required unconditionally, unlike tcip score-predictions:
the checkpoint verification this door always runs reads the registry under it, so an
unpinned root is always the wrong root to search, never only for one optional feature.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _run(args: list[str], cwd: Path, platform_root: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if platform_root is None:
        env.pop("TCIP_STATE_ROOT", None)
    else:
        env["TCIP_STATE_ROOT"] = platform_root
    return subprocess.run(
        [sys.executable, "-m", "tcip_web.cli", "triage-predictions", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=60,
    )


def test_refuses_from_an_unpinned_cwd_and_plants_no_store(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = _run(
        ["--checkpoint", str(tmp_path / "x.pt"), "--images-dir", str(images)],
        cwd=cwd, platform_root=None,
    )

    assert result.returncode != 0, result.stdout
    assert "TCIP_STATE_ROOT" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_triages_over_a_fixture_root_with_the_checkpoint_registered(tmp_path, monkeypatch, capsys):
    import tcip_mcp.pipelines.inference.predictor as predmod
    from tcip_mcp.cli.triage_predictions import main
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "scores": [0.9]}]
    monkeypatch.setattr(
        predmod, "build_predictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources: predictions))

    rc = main(["--checkpoint", str(ckpt), "--images-dir", str(images), "--project", str(tmp_path)])

    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert "error" not in body
    assert body["total_images"] == 1

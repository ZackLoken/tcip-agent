"""tcip triage-predictions: the demoted door's own command-line entry point.

--project is required unconditionally, unlike tcip score-predictions: the checkpoint verification
this door always runs reads the registry of that project, never only for one optional feature.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from tests._cli_fixtures import run_tcip


def test_refuses_a_run_naming_no_project_and_plants_no_store(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip("triage-predictions",
                      ["--checkpoint", str(tmp_path / "x.pt"), "--images-dir", str(images)],
                      cwd=cwd)

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_triages_with_the_checkpoint_registered_in_the_named_project(
        project, monkeypatch, capsys):
    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.cli.triage_predictions import main
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(project)
    images = project / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "scores": [0.9]}]
    monkeypatch.setattr(
        predmod, "GenericPredictor",
        lambda *a, **k: SimpleNamespace(predict_batch=lambda sources, **kw: predictions))

    rc = main(["--checkpoint", str(ckpt), "--images-dir", str(images), "--project", str(project)])

    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert "error" not in body
    assert body["total_images"] == 1

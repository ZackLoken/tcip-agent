"""tcip triage-predictions: its own command-line entry point.

A run naming no --project exits nonzero naming it and plants no store; a run naming one
triages with a checkpoint registered in that project.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from tests._cli_fixtures import run_tcip

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_refuses_a_run_naming_no_project_and_plants_no_store(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "triage-predictions",
        ["--checkpoint", str(tmp_path / "x.pt"), "--images-dir", str(images)],
        cwd=cwd,
    )

    assert result.returncode != 0, result.stdout
    assert "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_triages_with_the_checkpoint_registered_in_the_named_project(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_mcp.cli.triage_predictions import main
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(project)
    images = project / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")

    predictions = [{"image": "a.jpg", "scores": [0.9]}]
    monkeypatch.setattr(
        predmod,
        "GenericPredictor",
        lambda *a, **k: SimpleNamespace(
            predict_batch=lambda sources, **kw: predictions
        ),
    )

    rc = main(
        [
            "--checkpoint",
            str(ckpt),
            "--images-dir",
            str(images),
            "--project",
            str(project),
            "--low",
            "0.3",
            "--high",
            "0.8",
            "--rationale",
            "the band under test",
        ]
    )

    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert "error" not in body
    assert body["total_images"] == 1


def test_the_band_and_the_ranking_settings_are_required_on_every_command(
    tmp_path: Path,
) -> None:
    """Three console commands each refuse a call leaving their stated settings out, naming them.

    The settings are triage's band and its reason, the failure-case ranking's top-k and
    weights, and the reference grid's overlap.
    """
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()
    for command, args, required in (
        (
            "triage-predictions",
            ["--checkpoint", "x.pt", "--images-dir", ".", "--project", "."],
            ("--low", "--high", "--rationale"),
        ),
        (
            "render-failure-cases",
            ["--dataset-root", ".", "--bucket", "b", "--project", "."],
            (
                "--top-k",
                "--shortfall-weight",
                "--surplus-weight",
                "--confidence-weight",
            ),
        ),
        (
            "overlay-reference-grid",
            ["--image", "a.png", "--project", "."],
            ("--overlap",),
        ),
    ):
        result = run_tcip(command, args, cwd=cwd)
        assert result.returncode != 0, (command, result.stdout)
        refusal = result.stderr.rpartition(
            "the following arguments are required:"
        )[2]
        for flag in required:
            assert flag in refusal, (command, flag, result.stderr)

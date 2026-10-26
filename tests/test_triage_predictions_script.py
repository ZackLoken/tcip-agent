"""tcip triage-predictions: the tool's command, through the console's one adapter.

A run naming no --project is refused naming it and plants no store; a run naming one triages
with a checkpoint registered in that project.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

from tcip_mcp.dataset_layout import image_root
from tests._cli_fixtures import run_tcip

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_refuses_a_run_naming_no_project_and_plants_no_store(
    tmp_path: Path,
) -> None:
    images = image_root(tmp_path)
    images.mkdir()
    cwd = tmp_path / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "triage-predictions",
        [
            *("--checkpoint-path", str(tmp_path / "x.pt")),
            *("--images-dir", str(images), "--low", "0.3", "--high", "0.8"),
            *("--rationale", "the band under test", "--user", "tester"),
        ],
        cwd=cwd,
    )

    assert result.returncode == 2, result.stdout
    assert "no project" in result.stderr and "--project" in result.stderr
    assert not (cwd / ".tcip").exists()


def test_triages_with_the_checkpoint_registered_in_the_named_project(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tcip_mcp.pipelines.inference.generic_predictor as predmod
    from tcip_web.cli import main
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    ckpt = registered_checkpoint(project)
    images = image_root(project)
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
            "triage-predictions",
            "--checkpoint-path",
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
            "--user",
            "tester",
        ]
    )

    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert "error" not in body
    assert body["total_images"] == 1


def test_the_band_and_the_ranking_settings_are_required_on_every_command(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three console commands each refuse a call leaving their stated settings out, naming them.

    The settings are triage's band and its reason, the failure-case ranking's top-k and
    weights, and the reference grid's overlap; the server's argument validation names each.
    """
    from tcip_web.cli import main

    named = ["--project", str(project), "--user", "tester"]
    for command, args, required in (
        (
            "triage-predictions",
            ["--checkpoint-path", "x.pt", "--images-dir", ".", *named],
            ("low", "high", "rationale"),
        ),
        (
            "render-failure-cases",
            ["--dataset-root", ".", "--bucket", "b", *named],
            (
                "top_k",
                "shortfall_weight",
                "surplus_weight",
                "confidence_weight",
            ),
        ),
        (
            "overlay-reference-grid",
            ["--image-path", "a.png", *named],
            ("overlap",),
        ),
    ):
        assert main([command, *args]) == 2, command
        refusal = capsys.readouterr().err
        for field in required:
            assert f"\n{field}\n  Field required" in refusal, (
                command,
                refusal,
            )

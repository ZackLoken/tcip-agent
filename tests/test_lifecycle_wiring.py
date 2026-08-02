"""Training↔experiment↔registry lifecycle wiring (registration from an
experiment links final metrics + a back-reference + lineage)."""

import json


def test_register_model_from_experiment_links_metrics_and_lineage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # .tcip/experiments + .tcip/models live under cwd
    from tcip_mcp.experiments import (
        create_experiment,
        log_metrics,
        register_model_from_experiment,
    )
    from tcip_mcp.model_registry import ModelRegistry

    create_experiment("exp1", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    log_metrics("exp1", 1, {"map50": 0.5})
    log_metrics("exp1", 2, {"map50": 0.81})

    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights")

    result = register_model_from_experiment("exp1", str(ckpt))
    assert result["registered"] == "exp1"
    assert result["metrics"]["map50"] == 0.81  # final-epoch metrics, not fabricated

    # The registry entry carries the experiment back-reference + final metrics.
    m = ModelRegistry(".").get_model("exp1")
    assert m is not None
    assert "experiment:exp1" in m["tags"]
    assert m["metrics"]["map50"] == 0.81

    # The experiment's lineage records the model weights.
    lineage = json.loads((tmp_path / ".tcip" / "experiments" / "exp1" / "lineage.json").read_text())
    assert lineage["model_weights"] == str(ckpt)


def test_register_model_from_experiment_unknown_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import register_model_from_experiment
    assert "error" in register_model_from_experiment("does_not_exist", "x.pt")

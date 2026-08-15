"""Registry metric default (``val_map50``, not the never-present ``mAP``) is not silently applied:
``select_best_model`` requires an explicit ``metric``, since ``val_map50`` is a labeled
comparability metric, not necessarily what governs a trait's phenotype. Also covers the "no model
has metric X" vs "no models" distinction, lower-is-better ranking, and registered metrics sourced
from the checkpoint's own epoch rather than the last training epoch."""


def test_select_best_model_requires_explicit_metric(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import select_best_model

    project = str(tmp_path)

    # Empty registry → the "no models" message, even with no metric passed.
    assert select_best_model(project)["error"] == "No models registered"

    reg = ModelRegistry(project)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("a", str(ckpt), {}, metrics={"val_map50": 0.70}, tags=[])
    reg.register_model("b", str(ckpt), {}, metrics={"val_map50": 0.90}, tags=[])

    # A populated registry with no metric → required-metric error, not a silent pick.
    res = select_best_model(project)
    assert "metric is required" in res["error"]
    assert res["available_metrics"] == [{"metric": "val_map50", "role": "comparability_only"}]
    assert res["n_models"] == 2

    # An explicit, legitimate metric still succeeds: a rail must admit valid work.
    res = select_best_model(project, metric="val_map50")
    assert res["name"] == "b"
    assert res["ranking_basis"] == "val_map50"

    # A metric no model carries → a distinct error that lists what's actually available.
    res = select_best_model(project, metric="val_map99")
    assert "No registered model has metric" in res["error"]
    assert res["available_metrics"] == [{"metric": "val_map50", "role": "comparability_only"}]
    assert res["n_models"] == 2


def test_register_model_refuses_nonexistent_checkpoint(tmp_path):
    """register_model must refuse a phantom deliverable, not silently store a null-checksum
    entry: the shared chokepoint both register_model_from_experiment and
    model_tools.register_model's explicit mode route through."""
    import pytest as _pytest

    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    with _pytest.raises(FileNotFoundError):
        reg.register_model("ghost", str(tmp_path / "nonexistent.pt"), {})
    assert reg.get_model("ghost") is None


def test_best_model_lower_is_better_for_loss(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("hi", str(ckpt), {}, metrics={"val_loss": 0.9}, tags=[])
    reg.register_model("lo", str(ckpt), {}, metrics={"val_loss": 0.2}, tags=[])

    # loss/error keys rank ascending → the lower val_loss is "best".
    assert reg.best_model("val_loss")["name"] == "lo"
    # A metric no model has → None (cleanly distinguishable from "no models").
    assert reg.best_model("nonexistent") is None


def test_register_model_sources_metrics_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import torch

    from tcip_mcp.experiments import (
        create_experiment,
        log_metrics,
        register_model_from_experiment,
    )

    create_experiment("exp", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    log_metrics("exp", 1, {"val_map50": 0.60})
    log_metrics("exp", 2, {"val_map50": 0.40})  # last epoch is worse (overfit)

    # model_best.pt carries the best epoch's metrics (epoch 1), not the last row (epoch 2).
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_state_dict": {}, "epoch": 1, "metrics": {"val_map50": 0.60}}, ckpt)

    result = register_model_from_experiment("exp", str(ckpt))
    assert result["metrics"]["val_map50"] == 0.60  # from checkpoint, not 0.40 (last jsonl row)
    assert result["metrics"]["epoch"] == 1


def test_a_registry_payload_that_json_cannot_hold_is_refused_at_register_model(tmp_path):
    """Config and metrics arrive from a caller, an agent's own dict or a checkpoint's stamp,
    so the field that will not encode is named before anything reaches the registry.

    A stringified measurement is the failure this closes: it reads as a recorded number to
    every later reader, with nothing marking it as a repr of something else.
    """
    from pathlib import Path

    import pytest

    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    with pytest.raises(TypeError) as config_refused:
        reg.register_model("a", str(ckpt), {"weights": Path("model_best.pt")})
    assert "config.weights" in str(config_refused.value)

    with pytest.raises(ValueError) as metrics_refused:
        reg.register_model("a", str(ckpt), {}, metrics={"val_map50": float("inf")})
    assert "metrics.val_map50" in str(metrics_refused.value)

    assert reg.list_models() == []


def test_an_ordinary_registry_payload_is_still_registered(tmp_path):
    """The refusal above must not cost a real deliverable its registry entry."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    entry = reg.register_model("a", str(ckpt), {"epochs": 3}, metrics={"val_map50": 0.70})

    assert entry["name"] == "a"
    assert [m["name"] for m in reg.list_models()] == ["a"]
    assert reg.best_model("val_map50")["metrics"]["val_map50"] == 0.70

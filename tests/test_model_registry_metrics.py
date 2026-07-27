"""L9 — registry metric default (``val_map50``, not the never-present ``mAP``), a clear
"no model has metric X" vs "no models" distinction, lower-is-better ranking, and registered
metrics sourced from the checkpoint's own epoch rather than the last training epoch.

K9: ``select_best_model`` no longer defaults to ``val_map50`` (a labeled comparability metric,
not necessarily what governs a trait's phenotype) — an explicit ``metric`` is required."""


def test_select_best_model_requires_explicit_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # registry lives under .tcip/models in cwd
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import select_best_model

    # Empty registry → the "no models" message, even with no metric passed.
    assert select_best_model(".")["error"] == "No models registered"

    reg = ModelRegistry(".")
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("a", str(ckpt), {}, metrics={"val_map50": 0.70}, tags=[])
    reg.register_model("b", str(ckpt), {}, metrics={"val_map50": 0.90}, tags=[])

    # A populated registry with no metric → required-metric error, not a silent pick.
    res = select_best_model(".")
    assert "metric is required" in res["error"]
    assert res["available_metrics"] == [{"metric": "val_map50", "role": "comparability_only"}]
    assert res["n_models"] == 2

    # An explicit, legitimate metric still succeeds — a rail must admit valid work.
    res = select_best_model(".", metric="val_map50")
    assert res["name"] == "b"
    assert res["ranking_basis"] == "val_map50"

    # A metric no model carries → a DISTINCT error that lists what's actually available.
    res = select_best_model(".", metric="val_map99")
    assert "No registered model has metric" in res["error"]
    assert res["available_metrics"] == [{"metric": "val_map50", "role": "comparability_only"}]
    assert res["n_models"] == 2


def test_best_model_lower_is_better_for_loss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(".")
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
    log_metrics("exp", 2, {"val_map50": 0.40})  # last epoch is WORSE (overfit)

    # model_best.pt carries the BEST epoch's metrics (epoch 1), not the last row (epoch 2).
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_state_dict": {}, "epoch": 1, "metrics": {"val_map50": 0.60}}, ckpt)

    result = register_model_from_experiment("exp", str(ckpt))
    assert result["metrics"]["val_map50"] == 0.60  # from checkpoint, not 0.40 (last jsonl row)
    assert result["metrics"]["epoch"] == 1

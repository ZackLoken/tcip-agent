"""Registry metric default (``val_map50``, not the never-present ``mAP``) is not silently applied:
``rank_registered_models`` lists rather than ranking when ``metric`` is left empty, since
``val_map50`` is a labeled comparability metric, not necessarily what governs a trait's
phenotype; a stated metric that no model carries is a distinct error from an empty registry.
Also covers lower-is-better ranking and registered metrics sourced from the checkpoint's own
epoch rather than the last training epoch."""


def test_rank_registered_models_lists_rather_than_ranking_on_an_empty_metric(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = str(tmp_path)

    # Empty registry, no metric → an empty listing, not a refusal.
    empty = rank_registered_models(project)
    assert "error" not in empty
    assert empty == {"models": [], "count": 0, "available_metrics": []}

    reg = ModelRegistry(project)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("a", str(ckpt), {}, metrics={"val_map50": 0.70}, tags=[],
                       metrics_source="trainer")
    reg.register_model("b", str(ckpt), {}, metrics={"val_map50": 0.90}, tags=[],
                       metrics_source="trainer")

    # A populated registry with no metric → the listing, not a required-metric refusal.
    res = rank_registered_models(project)
    assert "error" not in res
    assert res["count"] == 2
    assert {m["name"] for m in res["models"]} == {"a", "b"}
    assert res["available_metrics"] == [
        {"metric": "val_map50", "role": "comparability_only", "direction": "higher",
         "sources": ["trainer"]}
    ]

    # An explicit, legitimate metric still succeeds: a rail must admit valid work.
    res = rank_registered_models(project, metric="val_map50")
    assert res["name"] == "b"
    assert res["ranking_basis"] == "val_map50"
    assert res["higher_is_better"] is True
    assert res["direction_source"] == "declared"
    assert res["excluded_unverified"] == []

    # A declared metric no model carries → a distinct error that lists what's actually available.
    res = rank_registered_models(project, metric="val_loss")
    assert "No registered model has metric" in res["error"]
    assert res["available_metrics"][0]["metric"] == "val_map50"
    assert res["n_models"] == 2

    # A metric with no declared ranking direction → refused before ever looking for it.
    res = rank_registered_models(project, metric="val_map99")
    assert "no declared ranking direction" in res["error"]
    assert res["available_metrics"][0]["metric"] == "val_map50"


def test_rank_registered_models_excludes_unverified_entries_by_default(tmp_path):
    """A caller-asserted metric is not silently trusted: it is ranked only when the caller
    explicitly says to consider unverified numbers."""
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = str(tmp_path)
    reg = ModelRegistry(project)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("asserted", str(ckpt), {}, metrics={"val_map50": 0.99}, tags=[],
                       metrics_source="caller")

    res = rank_registered_models(project, metric="val_map50")
    assert "unverified" in res["error"]
    assert res["excluded_unverified"] == [{"name": "asserted", "metrics_source": "caller"}]

    res = rank_registered_models(project, metric="val_map50", include_unverified=True)
    assert res["name"] == "asserted"
    assert res["unverified_included"] is True
    assert res["excluded_unverified"] == []


def test_rank_registered_models_refusals_name_no_argument_a_breeder_would_not_pass(tmp_path):
    """The two ranking refusals a breeder can reach through the GUI (an undeclared direction,
    every carrier unverified) read as plain sentences: neither names this tool's own
    parameters, since a breeder using the rank control never calls it directly."""
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = str(tmp_path)
    reg = ModelRegistry(project)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("asserted", str(ckpt), {}, metrics={"val_map50": 0.99}, tags=[],
                       metrics_source="caller")

    no_direction = rank_registered_models(project, metric="val_map99")["error"]
    all_unverified = rank_registered_models(project, metric="val_map50")["error"]

    assert no_direction == (
        "'val_map99' has no declared ranking direction (evaluation.HIGHER_IS_BETTER_BY_METRIC "
        "names no entry for it). State a direction to rank by it anyway, or pick one of the "
        "available metrics."
    )
    assert all_unverified == (
        "every registered model carrying 'val_map50' is unverified (metrics_source is not "
        "'trainer'); include unverified models to rank them, or register a verified run."
    )

    for text in (no_direction, all_unverified):
        assert "higher_is_better" not in text
        assert "include_unverified" not in text
        assert "available_metrics" not in text
        assert "rank_registered_models" not in text


def test_register_model_refuses_nonexistent_checkpoint(tmp_path):
    """register_model must refuse a phantom deliverable, not silently store a null-checksum
    entry: the shared chokepoint both register_model_from_experiment and
    model_tools.register_model's explicit mode route through."""
    import pytest as _pytest

    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    with _pytest.raises(FileNotFoundError):
        reg.register_model("ghost", str(tmp_path / "nonexistent.pt"), {}, metrics_source=None)
    assert reg.get_model("ghost") is None


def test_register_model_refuses_a_metrics_source_pairing_mismatch(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    import pytest as _pytest
    with _pytest.raises(ValueError, match="metrics_source"):
        reg.register_model("a", str(ckpt), {}, metrics={"val_map50": 0.5}, metrics_source=None)
    with _pytest.raises(ValueError, match="metrics_source"):
        reg.register_model("a", str(ckpt), {}, metrics=None, metrics_source="caller")
    assert reg.list_models() == []


def test_best_model_lower_is_better_for_loss(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("hi", str(ckpt), {}, metrics={"val_loss": 0.9}, tags=[],
                       metrics_source="trainer")
    reg.register_model("lo", str(ckpt), {}, metrics={"val_loss": 0.2}, tags=[],
                       metrics_source="trainer")

    # higher_is_better=False ranks ascending → the lower val_loss is "best".
    assert reg.best_model("val_loss", higher_is_better=False)["name"] == "lo"
    # A metric no model has → None (cleanly distinguishable from "no models").
    assert reg.best_model("nonexistent", higher_is_better=False) is None


def test_best_model_refuses_an_entry_with_no_metrics_source_key(tmp_path):
    """An entry predating the field is malformed, not just unverified: best_model refuses by
    name and says no operator door adds the missing field to an existing entry, rather than
    silently skipping it like a legitimate metrics-less entry."""
    import pytest as _pytest

    import tcip_store as ts

    from tcip_mcp.model_registry import ModelRegistry, registry_index_key

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("ok", str(ckpt), {}, metrics={"val_loss": 0.5}, tags=[],
                       metrics_source="trainer")

    key = registry_index_key(str(tmp_path))
    with ts.transaction(key) as txn:
        document = txn.read(key)
        del document["entries"][0]["metrics_source"]
        txn.write(key, document)

    with _pytest.raises(ValueError, match="no operator door adds either missing field"):
        ModelRegistry(str(tmp_path)).best_model("val_loss", higher_is_better=False)


def test_best_model_treats_a_present_null_source_as_an_honest_empty_pairing(tmp_path):
    """A stored ``metrics_source: null`` (an entry with no metrics) is not malformed: it is
    simply excluded from ranking, like any other entry that does not carry the metric."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    reg.register_model("empty", str(ckpt), {}, metrics=None, tags=[], metrics_source=None)
    reg.register_model("real", str(ckpt), {}, metrics={"val_loss": 0.5}, tags=[],
                       metrics_source="trainer")

    assert reg.best_model("val_loss", higher_is_better=False)["name"] == "real"


def test_register_model_sources_metrics_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import torch

    from tcip_mcp.experiments import (
        complete_run,
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
    assert "error" not in complete_run("exp", str(ckpt))

    result = register_model_from_experiment("exp", str(ckpt))
    assert result["metrics"]["val_map50"] == 0.60  # from checkpoint, not 0.40 (last jsonl row)
    assert result["metrics"]["epoch"] == 1

    from tcip_mcp.model_registry import ModelRegistry

    m = ModelRegistry(str(tmp_path)).get_model("exp")
    assert m["metrics_source"] == "trainer"  # config carries no training_source


def test_register_model_from_experiment_twice_on_a_completed_record_is_idempotent(tmp_path, monkeypatch):
    """Documents pre-existing behaviour, not new to this row: _register_entry replaces by name,
    and the eviction rail admits a replace whose experiment_id is this write's own, so a second
    register_model_from_experiment call on an already-completed record, with the same
    checkpoint path, succeeds again rather than refusing, the remedy _finalize_run's own
    docstring names for a registration that fails after complete_run succeeded. complete_run is
    used here only as the producer of a completed record to register against, not itself
    under test."""
    monkeypatch.chdir(tmp_path)
    import torch

    from tcip_mcp.experiments import (
        complete_run, create_experiment, register_model_from_experiment, update_status,
    )
    from tcip_mcp.model_registry import ModelRegistry

    create_experiment("exp", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    update_status("exp", "running")
    ckpt = tmp_path / "model_best.pt"
    torch.save({"model_state_dict": {}, "metrics": {"val_map50": 0.60}}, ckpt)
    complete_run("exp", str(ckpt))

    first = register_model_from_experiment("exp", str(ckpt))
    assert "error" not in first
    second = register_model_from_experiment("exp", str(ckpt))
    assert "error" not in second

    entries = [e for e in ModelRegistry(str(tmp_path)).list_models() if e["name"] == "exp"]
    assert len(entries) == 1
    assert entries[0]["checkpoint_path"] == str(ckpt)


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
        reg.register_model("a", str(ckpt), {"weights": Path("model_best.pt")},
                           metrics_source=None)
    assert "config.weights" in str(config_refused.value)

    with pytest.raises(ValueError) as metrics_refused:
        reg.register_model("a", str(ckpt), {}, metrics={"val_map50": float("inf")},
                           metrics_source="caller")
    assert "metrics.val_map50" in str(metrics_refused.value)

    assert reg.list_models() == []


def test_an_ordinary_registry_payload_is_still_registered(tmp_path):
    """The refusal above must not cost a real deliverable its registry entry."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    entry = reg.register_model("a", str(ckpt), {"epochs": 3}, metrics={"val_map50": 0.70},
                               metrics_source="trainer")

    assert entry["name"] == "a"
    assert [m["name"] for m in reg.list_models()] == ["a"]
    assert reg.best_model("val_map50", higher_is_better=True)["metrics"]["val_map50"] == 0.70


def test_register_model_tool_refuses_metrics_beside_experiment_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import pytest as _pytest

    from tcip_mcp.experiments import create_experiment
    from tcip_mcp.tools.model_tools import register_model

    create_experiment("exp", {"model_source": {"builder": "x:y"}}, data_source="imgs")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"weights")

    with _pytest.raises(ValueError, match="experiment_id"):
        register_model(experiment_id="exp", checkpoint_path=str(ckpt),
                       metrics={"val_map50": 0.5})


def test_register_model_tool_explicit_mode_sets_the_source_from_whether_metrics_were_passed(
    tmp_path,
):
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import register_model

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    register_model(name="a", checkpoint_path=str(ckpt), project_path=str(tmp_path),
                   metrics={"val_map50": 0.5})
    assert ModelRegistry(str(tmp_path)).get_model("a")["metrics_source"] == "caller"

    register_model(name="b", checkpoint_path=str(ckpt), project_path=str(tmp_path))
    assert ModelRegistry(str(tmp_path)).get_model("b")["metrics_source"] is None

"""Registry metric default (``val_map50``, not the never-present ``mAP``) is not silently applied:
``rank_registered_models`` lists rather than ranking when ``metric`` is left empty, since
``val_map50`` is a labeled comparability metric, not necessarily what governs a trait's
phenotype; a stated metric that no model carries is a distinct error from an empty registry.
Also covers lower-is-better ranking, the metrics source a registration derives, and a completed
run's checkpoint metrics sourced from the checkpoint's own epoch rather than the last one."""

import pytest


def _checkpoint(tmp_path, name: str) -> str:
    """A checkpoint file of its own under ``tmp_path`` for ``name``; its path."""
    pytest.importorskip("torch")
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    return str(produced_checkpoint(tmp_path / f"{name}.pt", name))


def _registered(tmp_path, *entries: tuple[str, dict | None]):
    """A registry holding one foreign entry per ``(name, metrics)``, each over a checkpoint file
    of its own."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    for name, metrics in entries:
        reg.register_model(name, _checkpoint(tmp_path, name), metrics=metrics, tags=[])
    return reg


def _named(reg, name: str) -> dict:
    """The one entry ``reg`` lists under ``name``."""
    (entry,) = [m for m in reg.list_models() if m["name"] == name]
    return entry


def test_rank_registered_models_lists_rather_than_ranking_on_an_empty_metric(tmp_path):
    from tcip_mcp.tools.model_tools import rank_registered_models

    project = tmp_path

    # Empty registry, no metric: an empty listing, not a refusal.
    assert rank_registered_models(project) == {"models": [], "count": 0, "available_metrics": []}

    _registered(tmp_path, ("a", {"val_map50": 0.70}), ("b", {"val_map50": 0.90}))

    # A populated registry with no metric: the listing, not a required-metric refusal.
    res = rank_registered_models(project)
    assert res["count"] == 2
    assert {m["name"] for m in res["models"]} == {"a", "b"}
    assert res["available_metrics"] == [
        {"metric": "val_map50", "role": "comparability_only", "direction": "higher",
         "sources": ["caller"]}
    ]

    # An explicit, legitimate metric still succeeds: a rail must admit valid work.
    res = rank_registered_models(project, metric="val_map50", include_unverified=True)
    assert res["name"] == "b"
    assert res["ranking_basis"] == "val_map50"
    assert (res["higher_is_better"], res["direction_source"]) == (True, "declared")

    # A declared metric no model carries: a distinct error listing what's actually available.
    res = rank_registered_models(project, metric="val_loss", include_unverified=True)
    assert "No registered model has metric" in res["error"]
    assert res["available_metrics"][0]["metric"] == "val_map50"
    assert res["n_models"] == 2

    # A metric with no declared ranking direction: refused before ever looking for it.
    res = rank_registered_models(project, metric="val_map99")
    assert "no declared ranking direction" in res["error"]


def test_rank_registered_models_excludes_unverified_entries_by_default(tmp_path):
    """A caller-asserted metric is not silently trusted: it is ranked only when the caller
    explicitly says to consider unverified numbers."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _registered(tmp_path, ("asserted", {"val_map50": 0.99}))

    res = rank_registered_models(tmp_path, metric="val_map50")
    assert "unverified" in res["error"]
    assert res["excluded_unverified"] == [{"name": "asserted", "metrics_source": "caller"}]

    res = rank_registered_models(tmp_path, metric="val_map50", include_unverified=True)
    assert res["name"] == "asserted"
    assert res["unverified_included"] is True
    assert res["excluded_unverified"] == []


def test_rank_registered_models_refusals_name_no_argument_a_breeder_would_not_pass(tmp_path):
    """The two ranking refusals a breeder can reach through the GUI (an undeclared direction,
    every carrier unverified) read as plain sentences: neither names this tool's own
    parameters, since a breeder using the rank control never calls it directly."""
    from tcip_mcp.tools.model_tools import rank_registered_models

    _registered(tmp_path, ("asserted", {"val_map50": 0.99}))

    no_direction = rank_registered_models(tmp_path, metric="val_map99")["error"]
    all_unverified = rank_registered_models(tmp_path, metric="val_map50")["error"]

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
        for parameter in ("higher_is_better", "include_unverified", "available_metrics",
                          "rank_registered_models"):
            assert parameter not in text


def test_register_model_refuses_a_nonexistent_checkpoint(tmp_path):
    """A phantom deliverable is refused, never stored as a null-checksum entry."""
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        reg.register_model("ghost", str(tmp_path / "nonexistent.pt"))
    assert reg.list_models() == []


def test_register_model_refuses_bytes_the_verified_reader_refuses(tmp_path):
    """A registration names a checkpoint every later reader can load: bytes the verified reader
    refuses are refused at the door and nothing is stored, and a readable checkpoint registers."""
    from tcip_mcp.model_registry import ModelRegistry, UnregisteredCheckpointError

    reg = ModelRegistry(str(tmp_path))
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")

    with pytest.raises(UnregisteredCheckpointError):
        reg.register_model("garbage", str(garbage))
    assert reg.list_models() == []

    reg.register_model("real", _checkpoint(tmp_path, "real"))
    assert [m["name"] for m in reg.list_models()] == ["real"]


def test_a_foreign_registrations_source_is_read_off_whether_it_carries_metrics(tmp_path):
    """The source is derived, never stated: ``caller`` for asserted metrics, ``None`` for none."""
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.model_tools import register_model

    register_model(name="a", checkpoint_path=_checkpoint(tmp_path, "a"),
                   project=tmp_path, metrics={"val_map50": 0.5})
    register_model(name="b", checkpoint_path=_checkpoint(tmp_path, "b"),
                   project=tmp_path)

    reg = ModelRegistry(str(tmp_path))
    assert _named(reg, "a")["metrics_source"] == "caller"
    assert _named(reg, "b")["metrics_source"] is None


def test_best_model_lower_is_better_for_loss(tmp_path):
    from tcip_mcp.model_registry import best_model

    models = _registered(tmp_path, ("hi", {"val_loss": 0.9}), ("lo", {"val_loss": 0.2})
                         ).list_models()

    assert best_model(models, "val_loss", higher_is_better=False,
                      include_unverified=True)["name"] == "lo"
    # A metric no model has: None, cleanly distinguishable from "no models".
    assert best_model(models, "nonexistent", higher_is_better=False,
                      include_unverified=True) is None


def test_best_model_excludes_an_entry_with_no_metrics_from_ranking(tmp_path):
    """An entry with no metrics (``metrics_source: null``) is not malformed: it is simply
    excluded from ranking, like any other entry that does not carry the metric."""
    from tcip_mcp.model_registry import best_model

    models = _registered(tmp_path, ("empty", None), ("real", {"val_loss": 0.5})).list_models()

    assert best_model(models, "val_loss", higher_is_better=False,
                      include_unverified=True)["name"] == "real"


def test_a_completed_runs_metrics_come_from_its_best_checkpoint_not_its_last_epoch(
    tmp_path, monkeypatch,
):
    """A run the default trainer completes names its best checkpoint, whose metrics are that
    epoch's own, sourced ``trainer``; the registry lists it under the run's id."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.experiments import METRICS_FILE, partition_rows, read_rows
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import worker_run
    from tests.tiny_trainer_fixtures import regressor_config, write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path, intensities=[0.1, 0.3, 0.5, 0.7], values=[0.2, 0.6, 1.0, 1.4])
    run_dir = worker_run(tmp_path, regressor_config(3, data={
        "images_dir": str(images_dir), "labels_dir": str(csv_path), "auto_val": False}),
        experiment_id="exp-best")

    entry = _named(ModelRegistry(str(tmp_path)), "exp-best")
    assert entry["metrics_source"] == "trainer"
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    (row,) = [r for r in partition_rows(read_rows(run_dir / METRICS_FILE)[0])[0]
              if r["epoch"] == best["epoch"]]
    assert entry["metrics"]["selection"] == row["selection"]


def test_a_registry_payload_that_json_cannot_hold_is_refused_at_register_model(tmp_path):
    """Metrics arrive from a caller, so the field that will not encode is named before anything
    reaches the registry: a stringified measurement would read as a recorded number to every
    later reader."""
    import tcip_store as ts
    from tcip_mcp.model_registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    ckpt = _checkpoint(tmp_path, "m")

    with pytest.raises(ts.StoreError) as metrics_refused:
        reg.register_model("a", str(ckpt), metrics={"val_map50": float("inf")})
    assert "metrics.val_map50" in str(metrics_refused.value)

    assert reg.list_models() == []


def test_an_ordinary_registry_payload_is_still_registered(tmp_path):
    """The refusal above must not cost a real deliverable its registry entry."""
    from tcip_mcp.model_registry import ModelRegistry, best_model

    reg = ModelRegistry(str(tmp_path))
    entry = reg.register_model("a", _checkpoint(tmp_path, "a"), metrics={"val_map50": 0.70})

    assert entry["name"] == "a"
    assert [m["name"] for m in reg.list_models()] == ["a"]
    assert best_model(reg.list_models(), "val_map50", higher_is_better=True,
                      include_unverified=True)["metrics"]["val_map50"] == 0.70

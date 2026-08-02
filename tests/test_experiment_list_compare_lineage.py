"""Experiment-tool coverage (list / compare / lineage)."""


def test_experiment_list_compare_lineage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        compare_experiments,
        create_experiment,
        get_experiment_lineage,
        list_experiments,
        log_metrics,
        update_lineage,
    )

    create_experiment("e1", {"model_source": {"builder": "my_models:tv_resnet50_det"}}, data_source="imgs")
    log_metrics("e1", 1, {"map50": 0.6})
    update_lineage("e1", model_weights="w.pt")
    create_experiment("e2", {"model_source": {"builder": "my_models:fcos_det"}})

    assert {e["experiment_id"] for e in list_experiments()} == {"e1", "e2"}

    cmp = compare_experiments(["e1", "e2", "missing"])
    assert cmp["count"] == 3
    e1 = next(c for c in cmp["experiments"] if c["experiment_id"] == "e1")
    assert e1["model"] == "my_models:tv_resnet50_det" and e1["final_metrics"]["map50"] == 0.6
    assert any("error" in c for c in cmp["experiments"])   # the missing experiment is reported

    lin = get_experiment_lineage("e1")
    assert lin["lineage"]["model_weights"] == "w.pt"
    assert "error" in get_experiment_lineage("nope")

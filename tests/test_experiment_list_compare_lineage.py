"""Experiment-tool coverage (list / compare / lineage)."""

import pytest


@pytest.mark.parametrize("backend_name", ["file", "sqlite"])
def test_a_created_experiment_is_listed_whichever_backend_holds_its_record(
    tmp_path, monkeypatch, backend_name
):
    """What names an experiment is a status record, not a directory some backend happens to make.

    The listing and the run resolver have to answer over the same set: an experiment the resolver
    finds and the listing omits is a run the breeder cannot see in the GUI's experiment list while
    the tools resolve it fine.
    """
    import tcip_store as ts
    from tcip_store.binding import BACKEND_ENV, bind_default

    monkeypatch.setenv(BACKEND_ENV, backend_name)
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))
    backend = bind_default()
    try:
        from tcip_mcp.experiments import (
            create_experiment,
            list_experiments,
            resolve_experiment_for_run,
        )

        create_experiment("e1", {"model_source": {"builder": "my_models:fcos_det"}})

        assert resolve_experiment_for_run("e1") == "e1"
        listed = list_experiments()
        assert [e["experiment_id"] for e in listed] == ["e1"]
        assert listed[0]["state"] == "created"
    finally:
        ts.unbind()
        backend.close()


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

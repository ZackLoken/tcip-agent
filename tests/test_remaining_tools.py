"""Phase 4.6 — component-registry listing + experiment-tool coverage.

(``list_available_models`` was removed as a strict subset of ``list_components``; this pins the
superset the agent now uses instead.)
"""

import pytest


def test_list_components_returns_registry_lists():
    pytest.importorskip("torch")  # populating the registries imports the component modules
    from tcip_mcp.tools.pipeline_tools import list_components
    res = list_components()
    # the superset that replaced list_available_models: same four keys plus detectors/optimizers
    assert {"backbones", "necks", "heads", "losses"} <= set(res.keys())
    assert isinstance(res["backbones"], list) and res["backbones"]   # non-empty, no crash


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

    create_experiment("e1", {"model_spec": {"backbone": {"name": "tv_resnet50"}}}, data_source="imgs")
    log_metrics("e1", 1, {"map50": 0.6})
    update_lineage("e1", model_weights="w.pt")
    create_experiment("e2", {"model": {"backbone": {"name": "fcos_bb"}}})

    assert {e["experiment_id"] for e in list_experiments()} == {"e1", "e2"}

    cmp = compare_experiments(["e1", "e2", "missing"])
    assert cmp["count"] == 3
    e1 = next(c for c in cmp["experiments"] if c["experiment_id"] == "e1")
    assert e1["backbone"] == "tv_resnet50" and e1["final_metrics"]["map50"] == 0.6
    assert any("error" in c for c in cmp["experiments"])   # the missing experiment is reported

    lin = get_experiment_lineage("e1")
    assert lin["lineage"]["model_weights"] == "w.pt"
    assert "error" in get_experiment_lineage("nope")

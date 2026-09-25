"""Each record here is written by the platform's own producer and read back by its reader, which
reads every key the producer writes as stated: a key the producer stopped writing fails the read by
name (``KeyError``) rather than being tolerated with a default."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def test_a_registered_entry_ranks_through_best_model(tmp_path: Path) -> None:
    from tcip_mcp.model_registry import ModelRegistry

    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    registry = ModelRegistry(str(tmp_path))
    registry.register_model("m", str(ckpt), {}, metrics={"val_map50": 0.7},
                            metrics_source="trainer")

    reader = ModelRegistry(str(tmp_path))
    best = reader.best_model("val_map50", higher_is_better=True)
    assert best is not None and best["name"] == "m"
    # The producer filter reads each entry's own recorded producer: this one names none.
    assert reader.best_model("val_map50", higher_is_better=True, experiment_ids=["exp-a"]) is None


def test_a_checkpoint_stating_no_task_refuses_naming_where_to_state_it(tmp_path: Path) -> None:
    import pytest

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    stated = registered_checkpoint(tmp_path, project_root=tmp_path, name="stated",
                                   filename="stated.pt")
    assert load_registered_checkpoint(stated, project_path=str(tmp_path)).task == "detection"

    unstated = registered_checkpoint(
        tmp_path, project_root=tmp_path, name="unstated", filename="unstated.pt",
        model_source={"builder": "tests.bespoke_models:build_bespoke_detection",
                      "builder_kwargs": {"num_classes": 1, "in_chans": 3, "min_size": 64,
                                         "max_size": 128}})
    with pytest.raises(ValueError, match="model_source.task"):
        load_registered_checkpoint(unstated, project_path=str(tmp_path)).task


def test_an_authored_trait_spec_restates_through_its_carried_forward_fields(tmp_path: Path) -> None:
    import tcip_store as ts

    from tcip_mcp import traits

    def _author() -> dict:
        return traits.author_trait_spec(
            str(tmp_path), "leaf", delivers=("leaf_length",),
            rationale="the breeder described the measurement in their own terms")

    _author()
    scope = traits.trait_spec_statements_scope(tmp_path)
    key = traits.trait_spec_statement_key(scope, "leaf")
    ts.delete(key, expect=ts.read_versioned(key).version)

    restated = _author()

    assert restated["trait"] == "leaf"
    assert traits.get_trait_for("leaf", str(tmp_path)).delivers == ("leaf_length",)


def test_an_authored_trait_spec_record_lacking_a_field_fails_at_the_decoder(tmp_path: Path) -> None:
    import pytest
    import tcip_store as ts

    from tcip_mcp import traits

    traits.author_trait_spec(
        str(tmp_path), "fruit", delivers=("leaf_length",),
        rationale="the breeder described the measurement in their own terms")
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "fruit")
    stored = ts.read_versioned(key)
    ts.replace(key, {k: v for k, v in stored.value.items() if k != "positive_value"},
               expect=stored.version)

    with pytest.raises(KeyError, match="positive_value"):
        traits.get_trait_for("fruit", str(tmp_path))


def test_a_sweep_manifest_lacking_its_status_fails_in_sweep_state(
    tmp_path: Path, real_hpo_base_config: dict, monkeypatch,
) -> None:
    """The manifest the sweep launch itself writes reads back as its status; the same manifest
    with ``status`` removed fails in ``sweep_state`` on both of its branches."""
    import pytest
    import tcip_store as ts

    import tcip_mcp.tools.training_tools as tt

    def fake_search(**kw):
        return {"best_params": {"lr": 0.1}, "best_value": 0.25, "n_trials": 1,
                "study_name": kw["study_name"]}

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    result = tt.run_hyperparameter_search(
        base_config=real_hpo_base_config, n_trials=1, output_dir=str(tmp_path), search_seed=0)
    manifest = ts.read(tt.sweep_manifest_key(result["study_name"], str(tmp_path)))
    assert tt.sweep_state(manifest, stale_seconds=600.0) == "completed"

    del manifest["status"]
    for driver_live in (False, True):
        with pytest.raises(KeyError, match="status"):
            tt.sweep_state(manifest, stale_seconds=600.0, driver_live=driver_live)


def test_an_image_status_entry_lacking_its_time_fails_at_the_read(tmp_path: Path) -> None:
    import pytest
    import tcip_store as ts

    from tcip_mcp.dataset_layout import (
        image_status_key, read_image_status_store, record_image_statuses, status_confirmations,
        status_tokens,
    )

    record_image_statuses(tmp_path, "bud/2026-03-01", {"IMG_1.JPG": "negative"},
                          recorded_by="user:breeder")
    assert status_tokens(read_image_status_store(tmp_path)) == {
        "bud/2026-03-01": {"IMG_1.JPG": "negative"}}

    stored = ts.read_versioned(image_status_key(tmp_path))
    del stored.value["bud/2026-03-01"]["IMG_1.JPG"]["recorded_at"]
    ts.replace(image_status_key(tmp_path), stored.value, expect=stored.version)

    with pytest.raises(KeyError, match="recorded_at"):
        status_confirmations(read_image_status_store(tmp_path))


def test_a_rehydrated_job_status_interrupts_a_live_one_and_refuses_an_unknown_one() -> None:
    import pytest

    from tcip_web.jobstore import rehydrated_status

    assert rehydrated_status({"status": "pending"}) == "interrupted"
    assert rehydrated_status({"status": "canceled"}) == "canceled"
    with pytest.raises(ValueError, match="cancelled"):
        rehydrated_status({"status": "cancelled"})


def test_a_persisted_job_summary_rehydrates_under_its_own_root(tmp_path: Path) -> None:
    from tcip_web.routes import inference

    job = inference.InferenceJob(
        job_id="guard-job", checkpoint_path="c", images_dir="i", output_dir="o",
        status="completed", done=2, total=2,
    )
    inference._register(job)
    inference._registry.jobs.clear()
    try:
        inference.rehydrate_for_current_root()
        restored = inference._registry.jobs["guard-job"]
        assert restored.platform_root == job.platform_root
        assert restored.status == "completed"
    finally:
        inference._registry.jobs.clear()


def test_an_attested_region_reads_back_through_the_completeness_route(tmp_path: Path) -> None:
    img_dir = tmp_path / "ds" / "images" / "2026-03-01"
    img_dir.mkdir(parents=True)
    path = str(img_dir / "plot.tif")
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)
    from tcip_web.app import app

    client = TestClient(app, base_url="http://127.0.0.1")

    grid_resp = client.get("/api/coverage/grid", params={"path": path, "tile_size": 64})
    assert grid_resp.status_code == 200, grid_resp.text
    grid = {k: v for k, v in grid_resp.json()["grid"].items() if k not in ("cells", "derivation")}
    posted = client.post("/api/coverage/completeness", json={
        "image_path": path, "subject": "bud", "grid": grid, "cell": "A1", "complete": True,
        "user": "breeder", "view_scale": None})
    assert posted.status_code == 200, posted.text

    got = client.get("/api/coverage/completeness", params={"path": path})

    assert got.status_code == 200, got.text
    record = got.json()["by_subject"]["bud"]
    assert record["cells_complete"] == ["A1"]
    assert record["stale_cells"] == []

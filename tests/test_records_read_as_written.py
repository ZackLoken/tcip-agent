"""Each record here is written by the platform's own producer and read back by its reader, which
reads every key the producer writes as stated: a key the producer stopped writing fails the read by
name (``KeyError``) rather than being tolerated with a default."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def test_a_registered_entry_ranks_through_best_model(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("torch")
    from tcip_mcp.model_registry import ModelRegistry, best_model
    from tests._verified_checkpoint_fixtures import checkpoint_file

    ckpt = checkpoint_file(tmp_path / "m.pt", "weights")
    registry = ModelRegistry(str(tmp_path))
    registry.register_model("m", str(ckpt), {}, metrics={"val_map50": 0.7})

    models = ModelRegistry(str(tmp_path)).list_models()
    best = best_model(models, "val_map50", higher_is_better=True, include_unverified=True)
    assert best is not None and best["name"] == "m"
    # Each entry's own recorded producer reads back: this foreign one names none.
    assert [m["experiment_id"] for m in models] == [None]


def test_a_checkpoint_stating_no_task_refuses_naming_where_to_state_it(tmp_path: Path) -> None:
    import pytest

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    stated = registered_checkpoint(tmp_path)
    assert load_registered_checkpoint(stated, project=tmp_path).task == "detection"

    # Written past the producer, which builds no model for a config naming no task.
    torch = pytest.importorskip("torch")
    from tcip_mcp.tools.model_tools import register_model
    from tests._verified_checkpoint_fixtures import SCOPED_DATA

    unstated = tmp_path / "unstated.pt"
    torch.save({"kind": "tcip_module", "model_state_dict": {},
                "config": {"model_source": {"builder": "tests.bespoke_models:build_bespoke_detection"},
                           "data": dict(SCOPED_DATA)}}, str(unstated))
    assert "error" not in register_model(name="unstated", checkpoint_path=str(unstated),
                                         config={}, project=tmp_path)
    with pytest.raises(ValueError, match="model_source.task"):
        load_registered_checkpoint(str(unstated), project=tmp_path).task


def test_a_proposed_trait_reads_back_as_the_entry_it_proposed(tmp_path: Path) -> None:
    from tcip_mcp import traits

    from tests import _trait_fixtures as fx

    proposed = fx.entry("leaf", ("leaf_length",))
    fx.propose(tmp_path, proposed)

    assert traits.read_trait("leaf", tmp_path).latest.entry == proposed


def test_a_trait_record_lacking_an_entry_field_fails_at_the_schema(tmp_path: Path) -> None:
    import pytest
    import tcip_store as ts
    from pydantic import ValidationError

    from tcip_mcp import traits

    from tests import _trait_fixtures as fx

    fx.propose(tmp_path, fx.entry("fruit", ("leaf_length",)))
    key = traits.trait_key(tmp_path, "fruit")
    stored = ts.read_versioned(key)
    del stored.value["revisions"][0]["entry"]["positive_value"]
    ts.replace(key, stored.value, expect=stored.version)

    with pytest.raises(ValidationError, match="positive_value"):
        traits.read_trait("fruit", tmp_path)


def test_a_sweep_final_status_lacking_its_state_fails_at_the_read(
    tmp_path: Path, real_hpo_base_config: dict, monkeypatch,
) -> None:
    """The final status the sweep launch itself writes reads back as its state; the same record
    with ``state`` removed fails the read by name."""
    import pytest

    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import FINAL_STATUS_FILE, read_record
    from tcip_store import RECORD_JSON

    def fake_search(**kw):
        return str(Path(kw["storage_path"]) / kw["study_name"])

    monkeypatch.setattr("tcip_mcp.pipelines.training.hpo.tune_search", fake_search)
    result = tt.run_hyperparameter_search(
        tmp_path, base_config=real_hpo_base_config, n_trials=1, search_seed=0)
    assert tt.monitor_training(tmp_path, sweep_id=result["study_name"])["status"] == "completed"

    final = tt.sweep_dir(result["study_name"], project=tmp_path) / FINAL_STATUS_FILE
    damaged = read_record(final)
    del damaged["state"]
    final.write_bytes(RECORD_JSON.encode(damaged))
    with pytest.raises(KeyError, match="state"):
        tt.monitor_training(tmp_path, sweep_id=result["study_name"])


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

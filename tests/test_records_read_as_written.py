"""Records written by the platform's own producers read back through their readers as stated.

Each record here is written by the platform's own producer and read back by its reader, and a
record missing a key its reader reads fails the read by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir

if TYPE_CHECKING:
    import pytest


def test_a_registered_entry_ranks_through_best_model(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("torch")
    from tcip_mcp.model_registry import ModelRegistry, best_model
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    ckpt = produced_checkpoint(tmp_path / "m.pt", "weights")
    registry = ModelRegistry(str(tmp_path))
    registry.register_model(
        "m", str(ckpt), metrics={"val_map50": 0.7}, actor=None
    )

    models = ModelRegistry(str(tmp_path)).list_models()
    best, _ = best_model(
        models, "val_map50", higher_is_better=True, include_unverified=True
    )
    assert best is not None and best.entry.name == "m"
    # Each entry's own recorded producer reads back: this foreign one names none.
    assert [m.entry.experiment_id for m in models] == [None]


def test_a_checkpoint_stating_no_task_refuses_naming_where_to_state_it(
    tmp_path: Path,
) -> None:
    import pytest

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    stated = registered_checkpoint(tmp_path)
    assert (
        load_registered_checkpoint(stated, project=tmp_path).task
        == "detection"
    )

    # Written past the producer, which builds no model for a config naming no task.
    torch = pytest.importorskip("torch")
    from tcip_mcp.pipelines.model_build import CONFIG_KEY
    from tcip_mcp.tools.model_tools import register_model

    unstated = tmp_path / "unstated.pt"
    payload = torch.load(stated, weights_only=True)
    del payload[CONFIG_KEY]["model_source"]["task"]
    torch.save(payload, str(unstated))
    register_model(
        name="unstated",
        checkpoint_path=str(unstated),
        project=tmp_path,
        actor=None,
    )
    with pytest.raises(ValueError, match="model_source.task"):
        load_registered_checkpoint(str(unstated), project=tmp_path).task


def test_a_proposed_trait_reads_back_as_the_entry_it_proposed(
    tmp_path: Path,
) -> None:
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx

    proposed = fx.entry("leaf", ("leaf_length",))
    fx.propose(tmp_path, proposed)

    assert traits.read_trait("leaf", tmp_path).latest.entry == proposed


def test_a_trait_record_lacking_an_entry_field_fails_at_the_schema(
    tmp_path: Path,
) -> None:
    import pytest
    from pydantic import ValidationError

    import tcip_store as ts
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx

    fx.propose(tmp_path, fx.entry("fruit", ("leaf_length",)))
    key = traits.trait_key(tmp_path, "fruit")
    stored = ts.read_versioned(key)
    del stored.value["revisions"][0]["entry"]["positive_state"]
    ts.replace(key, stored.value, expect=stored.version)

    with pytest.raises(ValidationError, match="positive_state"):
        traits.read_trait("fruit", tmp_path)


def test_a_sweep_final_status_lacking_its_state_fails_at_the_read(
    tmp_path: Path,
    real_hpo_base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweep's final status reads back its state, and fails by name with ``state`` removed.

    The final status the sweep launch itself writes reads back as its state; the same record
    with ``state`` removed fails the read by name.
    """
    import pytest
    from pydantic import ValidationError

    import tcip_mcp.tools.training_tools as tt
    from tcip_mcp.experiments import FINAL_STATUS_FILE, read_record
    from tcip_store import encode_record
    from tests._training_values import fifo_search, sweep_space

    def fake_search(**kw: object) -> None:
        return None

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.hpo.tune_search", fake_search
    )
    result = tt.run_hyperparameter_search(
        tmp_path,
        base_config=real_hpo_base_config,
        param_space=sweep_space(),
        n_trials=1,
        **fifo_search(),
        search_seed=0,
        actor=None,
    )
    sweep_id = result["sweep"]["sweep_id"]
    assert (
        tt.monitor_training(tmp_path, sweep_id)["sweep"]["state"]
        == "completed"
    )

    final = (
        tt.experiments.experiment_dir(sweep_id, project=tmp_path)
        / FINAL_STATUS_FILE
    )
    damaged = read_record(final)
    del damaged["state"]
    final.write_bytes(encode_record(damaged))
    with pytest.raises(ValidationError, match="state"):
        tt.monitor_training(tmp_path, sweep_id)


def test_a_completion_mark_lacking_its_time_fails_at_the_read(
    tmp_path: Path,
) -> None:
    import pytest

    import tcip_store
    from tcip_annotation.json_io import (
        UnreadableLabelDocumentError,
        read_label_document,
    )
    from tests._producer_fixtures import image_label_key, mark_complete

    image = image_dir(tmp_path, UNDATED_BUCKET) / "IMG_1.png"
    image.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(image)
    mark_complete(image, "bud", project=tmp_path, by="user:breeder")
    label = image_label_key(image)
    assert read_label_document(label).state("bud") == "negative"

    stored = tcip_store.read(label)
    assert isinstance(stored, dict) and isinstance(stored["complete"], dict)
    marks = stored["complete"]["bud"]
    assert isinstance(marks, list) and isinstance(marks[0], dict)
    del marks[0]["at"]
    tcip_store.replace(label, stored)

    with pytest.raises(
        UnreadableLabelDocumentError, match=r"KeyError\('at'\)"
    ):
        read_label_document(label)

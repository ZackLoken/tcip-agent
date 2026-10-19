"""The registry's producer binding.

A run's own final status names the digest it produced, and nothing else can name a producer for
weights the run did not complete with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_two_runs_each_produce_only_their_own_weights(tmp_path: Path) -> None:
    """Each of two runs' checkpoints loads naming the run that produced it.

    Each checkpoint loads naming the run whose final status names its digest, never the other.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import finished_run

    first = observe(
        finished_run(tmp_path, experiment_id="exp-rail11-base")
    ).checkpoint
    second = observe(
        finished_run(tmp_path, experiment_id="exp-rail11-next")
    ).checkpoint

    assert first is not None and second is not None
    assert first.sha256 != second.sha256
    assert load_registered_checkpoint(
        first.path, project=tmp_path
    ).experiment_id == ("exp-rail11-base")
    assert load_registered_checkpoint(
        second.path, project=tmp_path
    ).experiment_id == ("exp-rail11-next")


def test_a_tag_naming_a_run_never_makes_that_run_a_producer(
    tmp_path: Path,
) -> None:
    """A foreign checkpoint tagged with a run's id loads naming no producing run.

    The checkpoint, one a run completed under another root, is registered with an
    ``experiment:<id>`` tag naming a run of this project that did not complete with it, and it
    loads with no producing run.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.tools.model_tools import register_model
    from tests._verified_checkpoint_fixtures import (
        detection_config,
        opened_run,
        produced_checkpoint,
    )

    opened_run(
        tmp_path,
        detection_config(tmp_path / "data", project=tmp_path),
        experiment_id="exp-rail1",
    )
    forged = produced_checkpoint(
        tmp_path / "forged.pt", "completed under another root"
    )
    registered = register_model(
        name="exp-rail1-forged",
        checkpoint_path=str(forged),
        project=tmp_path,
        tags=["experiment:exp-rail1"],
    )
    assert "error" not in registered, registered

    assert (
        load_registered_checkpoint(forged, project=tmp_path).experiment_id
        is None
    )


def test_a_replacements_registry_event_names_the_superseded_entry(
    tmp_path: Path,
) -> None:
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    reg = ModelRegistry(str(tmp_path))
    ckpt = produced_checkpoint(tmp_path / "a.pt", "first content")
    reg.register_model("m", str(ckpt), tags=["first"])
    reg.register_model("m-renamed", str(ckpt))

    events = ts.read_log(audit_log_key(tmp_path)).records
    replace = [
        e for e in events if "superseded_name" in e.get("arguments", {})
    ]
    assert len(replace) == 1 and replace[0]["tool"] == "model_registered"
    arguments = replace[0]["arguments"]
    assert isinstance(arguments, dict)
    assert arguments["superseded_name"] == "m"
    assert arguments["superseded_tags"] == ["first"]


def test_caller_tag_still_round_trips_and_filters(tmp_path: Path) -> None:
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    reg = ModelRegistry(str(tmp_path))
    ckpt = produced_checkpoint(tmp_path / "m.pt", "weights")
    reg.register_model("m", str(ckpt), tags=["current"])

    assert [m.entry.name for m in reg.list_models(tag="current")] == ["m"]
    assert reg.list_models(tag="nonexistent") == []


def test_a_checkpoint_carrying_no_weights_refuses_at_registration_naming_the_field(
    tmp_path: Path,
) -> None:
    """A checkpoint payload carrying no weights is refused at registration naming the field.

    A checkpoint is admitted for the weights it loads: a payload carrying none refuses at
    registration, naming the field, and registers nothing; one carrying weights registers.
    """
    import pytest

    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    weightless = tmp_path / "weightless.pt"
    torch.save({CONFIG_KEY: {}}, weightless)
    registry = ModelRegistry(str(tmp_path))

    with pytest.raises(ValueError, match=STATE_DICT_KEY):
        registry.register_model("weightless", str(weightless))

    assert registry.list_models() == []
    registry.register_model(
        "weighted", str(produced_checkpoint(tmp_path / "w.pt", "weights"))
    )
    assert [m.entry.name for m in registry.list_models()] == ["weighted"]


def test_an_indexed_checkpoint_carrying_no_weights_refuses_at_load_naming_the_field(
    tmp_path: Path,
) -> None:
    """An indexed weightless checkpoint is refused at load naming the field.

    The load boundary states the weights rail itself: an index entry naming a weightless
    payload's digest, written past registration, refuses at ``load_registered_checkpoint``
    naming the field, while a registered checkpoint carrying weights loads.
    """
    import hashlib

    import pytest

    torch = pytest.importorskip("torch")
    import tcip_store as ts
    from tcip_mcp.model_registry import (
        ModelRegistry,
        load_registered_checkpoint,
        registry_index_key,
    )
    from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    weighted = produced_checkpoint(tmp_path / "w.pt", "weights")
    ModelRegistry(str(tmp_path)).register_model("weighted", str(weighted))
    weightless = tmp_path / "weightless.pt"
    torch.save({CONFIG_KEY: {}}, weightless)
    key = registry_index_key(tmp_path)
    index = ts.read_versioned(key)
    entry = {
        **index.value["entries"][0],
        "name": "weightless",
        "checkpoint_path": "weightless.pt",
        "sha256": hashlib.sha256(weightless.read_bytes()).hexdigest(),
    }
    ts.replace(
        key,
        {"entries": [*index.value["entries"], entry]},
        expect=index.version,
    )

    assert load_registered_checkpoint(weighted, project=tmp_path).sha256
    with pytest.raises(ValueError, match=STATE_DICT_KEY):
        load_registered_checkpoint(weightless, project=tmp_path)

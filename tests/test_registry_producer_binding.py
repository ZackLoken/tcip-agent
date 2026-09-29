"""The registry's producer binding: a run's own final status names the digest it produced, and
nothing else can name a producer for weights the run did not complete with.

tests/test_checkpoint_digest_rails.py holds the load boundary's own rails (two runs naming one
digest, a foreign registration naming a run's name or bytes).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _platform(tmp_path, monkeypatch):
    """Pin the platform state root at this test's tmp dir, where its runs land."""
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))


def test_corroborated_producer_reports_unknown_when_a_digest_disagrees_with_the_completion():
    """A stamp naming a real run with a digest that is not the one the run's final status names
    is reported producer-unknown."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.model_registry import _sha256_of_bytes
    from tcip_mcp.pipelines.resolution import corroborated_producer
    from tests._verified_checkpoint_fixtures import finished_run

    recorded = observe(finished_run(None, experiment_id="exp-rail6")).checkpoint["sha256"]

    forged_digest = _sha256_of_bytes(b"a different checkpoint entirely, not what this run wrote")
    assert forged_digest != recorded
    assert corroborated_producer(forged_digest, "exp-rail6") == (None, None)
    assert corroborated_producer(recorded, "exp-rail6") == (recorded, "exp-rail6")


def test_two_runs_each_bind_only_their_own_weights():
    """A run resumed into a new run directory completes with its own weights; each run's
    binding names its own digest and never the other's."""
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.resolution import corroborated_producer
    from tests._verified_checkpoint_fixtures import finished_run

    first = observe(finished_run(None, experiment_id="exp-rail11-base")).checkpoint["sha256"]
    second = observe(finished_run(None, experiment_id="exp-rail11-next")).checkpoint["sha256"]

    assert first != second
    assert corroborated_producer(second, "exp-rail11-next") == (second, "exp-rail11-next")
    assert corroborated_producer(first, "exp-rail11-base") == (first, "exp-rail11-base")
    assert corroborated_producer(first, "exp-rail11-next") == (None, None)


def test_a_tag_naming_a_run_never_makes_that_run_a_producer(tmp_path):
    """A foreign checkpoint registered with an ``experiment:<id>`` tag, for a run that never
    completed with it, never makes ``corroborated_producer`` name that run: the tag is caller
    metadata no producer resolver reads."""
    from tcip_mcp.model_registry import _sha256_of_bytes
    from tcip_mcp.pipelines.resolution import corroborated_producer
    from tcip_mcp.tools.model_tools import register_model
    from tests._verified_checkpoint_fixtures import checkpoint_file, detection_config, opened_run

    opened_run(None, detection_config(tmp_path / "data"), experiment_id="exp-rail1")
    forged = checkpoint_file(tmp_path / "forged.pt", "a checkpoint no run completed with")
    registered = register_model(name="exp-rail1-forged", checkpoint_path=str(forged), config={},
                                project_path=str(tmp_path), tags=["experiment:exp-rail1"])
    assert "error" not in registered, registered

    assert corroborated_producer(_sha256_of_bytes(forged.read_bytes()), "exp-rail1") == (
        None, None)


def test_a_replacements_registry_event_names_the_superseded_entry(tmp_path):
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import checkpoint_file

    reg = ModelRegistry(str(tmp_path))
    ckpt = checkpoint_file(tmp_path / "a.pt", "first content")
    reg.register_model("m", str(ckpt), {}, tags=["first"])
    reg.register_model("m-renamed", str(ckpt), {})

    events = ts.read_log(audit_log_key()).records
    replace = [e for e in events if "superseded_name" in e.get("arguments", {})]
    assert len(replace) == 1 and replace[0]["tool"] == "model_registered"
    assert replace[0]["arguments"]["superseded_name"] == "m"
    assert replace[0]["arguments"]["superseded_tags"] == ["first"]


def test_caller_tag_still_round_trips_and_filters(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import checkpoint_file

    reg = ModelRegistry(str(tmp_path))
    ckpt = checkpoint_file(tmp_path / "m.pt", "weights")
    reg.register_model("m", str(ckpt), {}, tags=["current"])

    assert [m["name"] for m in reg.list_models(tag="current")] == ["m"]
    assert reg.list_models(tag="nonexistent") == []

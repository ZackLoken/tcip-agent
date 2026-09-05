"""The registry's producer binding: a run's own completion records the digest it produced, and
nothing else can name a producer for weights the run did not record.

Rails from docs/audit/remediation/milestone-s/registry-producer-binding-design.md, numbered as
that note's section 4 numbers them. Rail 8's full shape lives in tests/test_training_envelope.py
(the round trip needs the audited envelope); rail 13's admits live in
tests/test_provenance_spine.py and tests/test_registry_entry_shape_agreement.py; rail 4 of the
checkpoint-digest family's own rails (a different, same-numbered rail set) is rewritten in
tests/test_checkpoint_digest_rails.py.
"""

from __future__ import annotations

import os

import pytest


# Rail 3: update_lineage(experiment_id, model_weights_sha256=...) and
# update_lineage(experiment_id, model_weights=...) raise naming complete_run.

def test_update_lineage_refuses_the_two_completion_fields(tmp_path, monkeypatch):
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail3"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    with pytest.raises(ValueError, match="complete_run"):
        update_lineage(exp_id, model_weights_sha256="deadbeef")
    with pytest.raises(ValueError, match="complete_run"):
        update_lineage(exp_id, model_weights="/some/path.pt")

    lineage = ts_read_lineage(tmp_path, exp_id)
    assert lineage["model_weights"] is None
    assert lineage["model_weights_sha256"] is None


def test_update_lineage_refuses_a_completion_field_even_beside_a_legitimate_one(tmp_path, monkeypatch):
    """The whole call refuses before any field lands: a legitimate companion in the same call
    does not land either, unlike the identity fields' pop-and-audit treatment."""
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail3b"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    with pytest.raises(ValueError, match="complete_run"):
        update_lineage(exp_id, predictions="/preds/should-not-land", model_weights="/w.pt")

    lineage = ts_read_lineage(tmp_path, exp_id)
    assert lineage["predictions"] is None


def ts_read_lineage(root, experiment_id: str) -> dict:
    from tcip_mcp.experiments import lineage_key, read_member

    return read_member(lineage_key(experiment_id, root=root), {})


# Rail 6 (coverage): a checkpoint whose payload stamps a real experiment's id but whose bytes are
# not the ones the run's completion recorded is reported producer-unknown by corroborated_producer.

def test_corroborated_producer_reports_unknown_when_stamped_bytes_disagree_with_completion(
    tmp_path, monkeypatch,
):
    from tcip_mcp.experiments import complete_run, create_experiment, update_status
    from tcip_mcp.model_registry import _sha256_of_bytes
    from tcip_mcp.pipelines.resolution import corroborated_producer

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail6"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    weights = tmp_path / "model_best.pt"
    weights.write_bytes(b"the real weights this run produced")
    completed = complete_run(exp_id, str(weights))
    assert "error" not in completed, completed

    forged_digest = _sha256_of_bytes(b"a different checkpoint entirely, not what this run wrote")
    assert forged_digest != completed["model_weights_sha256"]
    assert corroborated_producer(forged_digest, exp_id) == (None, None)


# Rail 9 (admits, the part built so far): update_lineage still records predictions, data_source
# and review_session.

def test_update_lineage_still_records_predictions_data_source_and_review_session(tmp_path, monkeypatch):
    from tcip_mcp.experiments import create_experiment, update_lineage, update_status

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail9"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    res = update_lineage(exp_id, predictions="/preds/rail9", data_source="/data/rail9",
                         review_session="session-1")
    assert res["lineage"]["predictions"] == "/preds/rail9"
    assert res["lineage"]["data_source"] == "/data/rail9"
    assert res["lineage"]["review_session"] == "session-1"


# Rail 11 (admits): a resumed run (the relaunch producing a fresh id) records and binds its own
# weights, and its delivery names the new run.

def test_resumed_run_records_and_binds_its_own_weights(tmp_path, monkeypatch):
    from tcip_mcp.experiments import complete_run, create_experiment, log_metrics, update_status
    from tcip_mcp.pipelines.resolution import corroborated_producer
    from tcip_mcp.tools.training_tools import _ensure_experiment

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    base = "exp-rail11-base"
    create_experiment(base, {"model_source": {"builder": "x:y"}})
    update_status(base, "running")
    log_metrics(base, 1, {"val_map50": 0.5})
    first_weights = tmp_path / "out1" / "model_best.pt"
    first_weights.parent.mkdir(parents=True, exist_ok=True)
    first_weights.write_bytes(b"the first run's own weights")
    first = complete_run(base, str(first_weights))
    assert "error" not in first, first

    resumed_id = _ensure_experiment(
        base, {"model_source": {"builder": "x:y"}}, None, str(first_weights), "run-2",
        output_dir=str(tmp_path / "out2"))
    assert resumed_id != base  # a non-pristine base mints a fresh id, per _ensure_experiment

    second_weights = tmp_path / "out2" / "model_best.pt"
    second_weights.parent.mkdir(parents=True, exist_ok=True)
    second_weights.write_bytes(b"the resumed run's own, different weights")
    second = complete_run(resumed_id, str(second_weights))
    assert "error" not in second, second
    assert second["model_weights_sha256"] != first["model_weights_sha256"]

    assert corroborated_producer(second["model_weights_sha256"], resumed_id) == (
        second["model_weights_sha256"], resumed_id)
    # The base run's own binding stands, unaffected by the resumed run's completion.
    assert corroborated_producer(first["model_weights_sha256"], base) == (
        first["model_weights_sha256"], base)


# Rail 1: a forged tag never feeds the run's own recorded checkpoint.

def test_forged_tag_never_feeds_the_experiment_s_own_recorded_checkpoint(tmp_path, monkeypatch):
    """An entry registered through explicit mode with an ``experiment:<id>`` tag, for a run
    that never registered itself, never makes ``corroborated_producer`` name that run: the tag
    is caller metadata read by no producer resolver any more. Registered before the run's own
    name (coverage: replaced by name once the run registers) and under another name (coverage:
    never scanned at all)."""
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.resolution import corroborated_producer, experiment_recorded_checkpoint
    from tcip_mcp.tools.model_tools import register_model

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail1"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")

    before = tmp_path / "before.pt"
    before.write_bytes(b"a forged checkpoint registered under the run's own name")
    reg_before = register_model(name=exp_id, checkpoint_path=str(before), config={},
                                project_path=str(tmp_path), tags=[f"experiment:{exp_id}"])
    assert "error" not in reg_before, reg_before

    after = tmp_path / "after.pt"
    after.write_bytes(b"a forged checkpoint registered under a different name")
    reg_after = register_model(name=f"{exp_id}-other", checkpoint_path=str(after), config={},
                               project_path=str(tmp_path), tags=[f"experiment:{exp_id}"])
    assert "error" not in reg_after, reg_after

    from tcip_mcp.model_registry import _sha256_of_bytes

    assert experiment_recorded_checkpoint(exp_id) is None

    before_digest = _sha256_of_bytes(before.read_bytes())
    assert corroborated_producer(before_digest, exp_id) == (None, None)

    after_digest = _sha256_of_bytes(after.read_bytes())
    assert corroborated_producer(after_digest, exp_id) == (None, None)


# Rail 2: bytes the run's completion did not record refuse, writing nothing.

def test_register_model_from_experiment_refuses_bytes_the_run_did_not_record(tmp_path, monkeypatch):
    """A completed run's recorded path overwritten in place, and a running run given any file,
    both refuse rather than bind or overwrite the lineage, each appending an audit line naming
    the caller's digest, the recorded digest and the recorded path."""
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.experiments import (
        complete_run, create_experiment, lineage_key, read_member, register_model_from_experiment,
        update_status,
    )
    from tcip_mcp.model_registry import ModelRegistry, _sha256_of_bytes

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail2"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the bytes this run actually produced")
    completed = complete_run(exp_id, str(ckpt))
    assert "error" not in completed, completed

    overwritten = b"a different payload written over the same path after completion"
    ckpt.write_bytes(overwritten)
    refused = register_model_from_experiment(exp_id, str(ckpt))
    assert "error" in refused
    assert ModelRegistry(str(tmp_path)).list_models() == []
    assert read_member(lineage_key(exp_id), {})["model_weights_sha256"] == completed["model_weights_sha256"]

    exp_id2 = "exp-rail2-running"
    create_experiment(exp_id2, {"model_source": {"builder": "x:y"}})
    update_status(exp_id2, "running")
    other = tmp_path / "other.pt"
    other.write_bytes(b"any file at all")
    refused2 = register_model_from_experiment(exp_id2, str(other))
    assert "error" in refused2
    assert ModelRegistry(str(tmp_path)).list_models() == []

    events = ts.read_log(audit_log_key(tmp_path)).records
    refusals = [e["arguments"] for e in events if e.get("tool") == "experiment_mutation_refused"
                and e["arguments"].get("op") == "register_model_from_experiment"]
    assert len(refusals) == 2

    mismatch = next(a for a in refusals if a["experiment_id"] == exp_id)
    assert mismatch["caller_sha256"] == _sha256_of_bytes(overwritten)
    assert mismatch["recorded_sha256"] == completed["model_weights_sha256"]
    assert mismatch["recorded_path"] == str(ckpt)

    not_completed = next(a for a in refusals if a["experiment_id"] == exp_id2)
    assert not_completed["caller_sha256"] == _sha256_of_bytes(b"any file at all")
    assert not_completed["recorded_sha256"] is None
    assert not_completed["recorded_path"] is None


# Rail 4: a project_path other than the experiment's root refuses by name.

def test_register_model_from_experiment_refuses_a_project_path_not_the_experiment_s_root(
    tmp_path, monkeypatch,
):
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment, update_status

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail4"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the run's own weights")
    assert "error" not in complete_run(exp_id, str(ckpt))

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    refused = register_model_from_experiment(exp_id, str(ckpt), project_path=str(elsewhere))
    assert "error" in refused


# Rail 9 (the other admits half): an explicit project_path naming the experiment's own root in
# another spelling still binds.

def test_register_model_from_experiment_admits_a_differently_spelled_own_root(tmp_path, monkeypatch):
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail9b"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the run's own weights")
    assert "error" not in complete_run(exp_id, str(ckpt))

    same_dir_other_spelling = str(tmp_path) + os.sep
    admitted = register_model_from_experiment(exp_id, str(ckpt), project_path=same_dir_other_spelling)
    assert "error" not in admitted, admitted

    # A relative-looking (forward-slash) spelling of the same root resolves to an absolute
    # registry root, never a BadKey out of the store.
    forward_slash_spelling = str(tmp_path).replace(os.sep, "/")
    admitted_slash = register_model_from_experiment(
        exp_id, str(ckpt), project_path=forward_slash_spelling, name=f"{exp_id}-slash",
    )
    assert "error" not in admitted_slash, admitted_slash
    assert ModelRegistry(str(tmp_path)).get_model(f"{exp_id}-slash") is not None


# Rail 5: a name a run bound refuses eviction by anything but that run.

def test_explicit_mode_refuses_to_replace_a_run_bound_entry(tmp_path, monkeypatch):
    """An explicit-mode registration under the name of an entry a run bound raises rather than
    evicting it (a plain ``ValueError`` at the baseline this guards against; the specific class
    is checked separately, once the fail-before proof no longer needs to import a symbol the
    baseline predates)."""
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail5"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the run's own weights")
    assert "error" not in complete_run(exp_id, str(ckpt))
    bound = register_model_from_experiment(exp_id, str(ckpt), name="bound-name")
    assert "error" not in bound, bound

    other_ckpt = tmp_path / "hand_built.pt"
    other_ckpt.write_bytes(b"a hand-built checkpoint with no run behind it")
    with pytest.raises(ValueError) as exc_info:
        ModelRegistry(str(tmp_path)).register_model(
            "bound-name", str(other_ckpt), {}, metrics_source=None)
    assert type(exc_info.value).__name__ == "EntryOwnedByRun"
    assert exp_id in str(exc_info.value)
    entry = ModelRegistry(str(tmp_path)).get_model("bound-name")
    assert entry["sha256"] == bound["sha256"]  # the original binding stands


def test_a_second_run_s_binding_under_a_bound_name_refuses_naming_the_run(tmp_path, monkeypatch):
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail5c"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the run's own weights")
    assert "error" not in complete_run(exp_id, str(ckpt))
    bound = register_model_from_experiment(exp_id, str(ckpt), name="bound-name")
    assert "error" not in bound, bound

    exp_id2 = "exp-rail5-second"
    create_experiment(exp_id2, {"model_source": {"builder": "x:y"}})
    update_status(exp_id2, "running")
    ckpt2 = tmp_path / "second_run.pt"
    ckpt2.write_bytes(b"a second run's own, different weights")
    assert "error" not in complete_run(exp_id2, str(ckpt2))
    refused = register_model_from_experiment(exp_id2, str(ckpt2), name="bound-name")
    assert "error" in refused
    assert exp_id in refused["error"]

    entry = ModelRegistry(str(tmp_path)).get_model("bound-name")
    assert entry["experiment_id"] == exp_id
    assert entry["sha256"] == bound["sha256"]


# Rail 7: a registry entry with no experiment_id key refuses the load by name.

def test_missing_experiment_id_key_refuses_the_load(tmp_path, monkeypatch):
    """The payload is a real, torch-loadable checkpoint, so a baseline without this rail loads
    it cleanly (no raise at all) rather than raising for an unrelated reason (an unloadable
    payload) that would leave this proof unable to tell the two apart."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.model_registry import UnregisteredCheckpoint, load_registered_checkpoint, registry_index_key
    import tcip_store as ts

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = tmp_path / "pre_field.pt"
    torch.save({"model_state_dict": {}}, ckpt)
    from tcip_mcp.model_registry import _sha256_of_bytes

    digest = _sha256_of_bytes(ckpt.read_bytes())
    key = registry_index_key(str(tmp_path))
    entry = {
        "name": "pre-field-entry", "checkpoint_path": str(ckpt), "kind": None, "sha256": digest,
        "file_size_bytes": ckpt.stat().st_size, "registered_at": "2026-01-01T00:00:00+00:00",
        "config": {}, "metrics": {}, "metrics_source": None, "tags": [],
    }  # experiment_id key deliberately absent
    with ts.transaction(key) as txn:
        txn.write(key, {"entries": [entry]})

    with pytest.raises(UnregisteredCheckpoint, match="experiment_id"):
        load_registered_checkpoint(ckpt, project_path=str(tmp_path))


# Rail 10: a repeat registration, and a byte-identical copy, both admit.

def test_register_model_from_experiment_twice_and_from_a_copy_both_admit(tmp_path, monkeypatch):
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail10"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the run's own weights")
    assert "error" not in complete_run(exp_id, str(ckpt))

    first = register_model_from_experiment(exp_id, str(ckpt))
    assert "error" not in first, first
    second = register_model_from_experiment(exp_id, str(ckpt))
    assert "error" not in second, second
    assert len(ModelRegistry(str(tmp_path)).list_models()) == 1

    copy = tmp_path / "copy.pt"
    copy.write_bytes(ckpt.read_bytes())
    from_copy = register_model_from_experiment(exp_id, str(copy), name=f"{exp_id}-copy")
    assert "error" not in from_copy, from_copy
    entry = ModelRegistry(str(tmp_path)).get_model(f"{exp_id}-copy")
    assert entry["sha256"] == first["sha256"]


# Rail 14: the model_registry_replace event names the superseded entry's experiment_id.

def test_model_registry_replace_event_carries_superseded_experiment_id(tmp_path, monkeypatch):
    import tcip_store as ts
    from tcip_mcp.audit import audit_log_key
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    reg = ModelRegistry(str(tmp_path))
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"first content")
    reg.register_model("m", str(ckpt_a), {}, metrics_source=None)
    ckpt_b = tmp_path / "b.pt"
    ckpt_b.write_bytes(b"second, different content")
    reg.register_model("m", str(ckpt_b), {}, metrics_source=None)

    events = ts.read_log(audit_log_key()).records
    replace = [e for e in events if e.get("tool") == "model_registry_replace"]
    assert len(replace) == 1
    assert replace[0]["arguments"]["superseded_experiment_id"] is None


# Rail 17: a caller tag still round-trips and filters.

def test_caller_tag_still_round_trips_and_filters(tmp_path, monkeypatch):
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    reg = ModelRegistry(str(tmp_path))
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"weights")
    reg.register_model("m", str(ckpt), {}, tags=["current"], metrics_source=None)

    assert [m["name"] for m in reg.list_models(tag="current")] == ["m"]
    assert reg.list_models(tag="nonexistent") == []


# Rail 18: weights registered again in experiment mode under a new name bind the new entry.

def test_a_bound_run_s_weights_registered_under_a_new_name_binds_the_new_entry(tmp_path, monkeypatch):
    """The old name stays the run's; the new name binds to the same run."""
    from tcip_mcp.experiments import complete_run, create_experiment, register_model_from_experiment, update_status
    from tcip_mcp.model_registry import ModelRegistry

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    exp_id = "exp-rail18"
    create_experiment(exp_id, {"model_source": {"builder": "x:y"}})
    update_status(exp_id, "running")
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"the run's own weights")
    assert "error" not in complete_run(exp_id, str(ckpt))

    first = register_model_from_experiment(exp_id, str(ckpt), name="first-name")
    assert "error" not in first, first
    second = register_model_from_experiment(exp_id, str(ckpt), name="second-name")
    assert "error" not in second, second

    reg = ModelRegistry(str(tmp_path))
    first_entry = reg.get_model("first-name")
    second_entry = reg.get_model("second-name")
    assert first_entry["experiment_id"] == exp_id
    assert second_entry["experiment_id"] == exp_id
    assert first_entry["sha256"] == second_entry["sha256"] == first["sha256"]

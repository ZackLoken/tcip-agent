"""Who launched a run, stamped on the record: ``stamp_run_identity``'s ``launched_by`` and
``launch_training``'s resolution of it (a declared launcher, an MCP agent's identity, or a bare
process), read back through ``reconstruct_from_status``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.fixture(autouse=True)
def _forget_agent_identity_between_tests():
    from tcip_mcp import agent_identity

    agent_identity.end()
    yield
    agent_identity.end()


def test_stamp_run_identity_requires_launched_by(tmp_path, monkeypatch):
    """The keyword is required: a caller with nothing to declare must still say so explicitly,
    naming the fact available (``process``) rather than leaving the field to a default."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, stamp_run_identity

    create_experiment("exp-launcher-required", {"model_source": {"builder": "x:y"}})

    with pytest.raises(TypeError):
        stamp_run_identity("exp-launcher-required", "out")  # type: ignore[call-arg]

    stamp_run_identity("exp-launcher-required", "out", launched_by={"launcher": "process"})


def test_stamp_run_identity_moves_the_record_to_running_in_one_write(tmp_path, monkeypatch):
    """The stamp is the launch's own transition: output_dir, launched_by, state, heartbeat and
    started all land in the one compare-and-set write, never a separate update_status call."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, read_member, stamp_run_identity, status_key

    create_experiment("exp-one-write", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp-one-write", "out", launched_by={"launcher": "process"})

    status = read_member(status_key("exp-one-write"), {})
    assert status["state"] == "running"
    assert status["output_dir"] == "out"
    assert status["launched_by"] == {"launcher": "process"}
    assert status["heartbeat"]
    assert status["started"]


def test_stamp_run_identity_refuses_a_record_already_stamped(tmp_path, monkeypatch):
    """The precondition: state must be "created" and no output_dir already stamped. A second
    stamp attempt against the same record (the race two concurrent launches could lose) raises
    the typed exception the launch branches on, rather than silently overwriting the winner's
    identity."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        StampPreconditionFailed, create_experiment, stamp_run_identity,
    )

    create_experiment("exp-already-stamped", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp-already-stamped", "out-1", launched_by={"launcher": "process"})

    with pytest.raises(StampPreconditionFailed):
        stamp_run_identity("exp-already-stamped", "out-2", launched_by={"launcher": "process"})


def test_stamp_run_identity_refuses_a_record_not_in_the_created_state(tmp_path, monkeypatch):
    """A record already moved past "created" by some other path (here, update_status) fails the
    same precondition, whether or not it carries an output_dir."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        StampPreconditionFailed, create_experiment, stamp_run_identity, update_status,
    )

    create_experiment("exp-not-created", {"model_source": {"builder": "x:y"}})
    update_status("exp-not-created", "canceled")

    with pytest.raises(StampPreconditionFailed):
        stamp_run_identity("exp-not-created", "out", launched_by={"launcher": "process"})


def test_reconstruct_run_status_on_a_malformed_id_answers_none(tmp_path, monkeypatch):
    """An id no record could ever carry (a path separator here) folds to None, the same answer
    an absent record gets, never a raised BadKey reaching the caller."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import reconstruct_run_status

    assert reconstruct_run_status("not/a/single/name", stale_seconds=600.0) is None


def _fake_popen(monkeypatch: pytest.MonkeyPatch, captured: list[list[str]]) -> None:
    import subprocess

    class _FakeProc:
        pid = 424242

    def _popen(argv, **kwargs):
        captured.append(argv)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})


def _detection_cfg(images_dir, labels_dir, experiment_id: str) -> dict:
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "subject": "bud"},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
                     "mixed_precision": False, "device": "cpu",
        "experiment_id": experiment_id,
    }


def _seed_one_image(images_dir, labels_dir) -> None:
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir.mkdir()
    labels_dir.mkdir()
    Image.new("RGB", (32, 32)).save(images_dir / "img0.png")
    json_io.write_annotations(str(labels_dir / "img0.json"),
                              [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)


def test_a_bare_launch_writes_launcher_process(tmp_path, monkeypatch):
    """A launch through neither a declared launcher nor an MCP handshake stamps ``process``: the
    fact available, never a guess about who."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-bare-launch"), str(tmp_path / "out"))
    assert "error" not in result, result

    status = read_member(status_key(result["experiment_id"]))
    assert status["launched_by"] == {"launcher": "process"}


def test_a_launch_inside_an_mcp_handshake_writes_launcher_agent_with_identity_fields(
    tmp_path, monkeypatch,
):
    """A launch made while an MCP handshake is in force stamps the connected agent's own
    identity, the same fields its audit line already carries."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TCIP_TERMINAL_SESSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_EFFORT", raising=False)
    from tcip_mcp import agent_identity
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    identity = agent_identity.begin("claude-code", "2.1.238")
    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-agent-launch"), str(tmp_path / "out"))
    assert "error" not in result, result

    status = read_member(status_key(result["experiment_id"]))
    assert status["launched_by"] == {
        "launcher": "agent", "agent_client_name": "claude-code", "agent_client_version": "2.1.238",
        "agent_session": identity.session,
    }


def test_declare_launcher_stamps_the_declared_name(tmp_path, monkeypatch):
    """The mechanism the web route uses: a launch made inside ``declare_launcher(name)`` stamps
    that name whatever the connected identity, since a declaration on the calling thread takes
    precedence over the handshake fallback. An identity is begun on this thread first, so the
    precedence branch of ``_resolve_launched_by`` is the one exercised, not the untested case of
    no identity at all."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp import agent_identity
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    agent_identity.begin("claude-code", "2.1.238")
    try:
        with training_tools.declare_launcher("gui"):
            result = training_tools.launch_training(
                _detection_cfg(images_dir, labels_dir, "exp-gui-launch"), str(tmp_path / "out"))
    finally:
        agent_identity.end()
    assert "error" not in result, result

    status = read_member(status_key(result["experiment_id"]))
    assert status["launched_by"] == {"launcher": "gui"}


def test_every_ensure_experiment_branch_stamps_the_same_launched_by(tmp_path, monkeypatch):
    """The fresh-creation, pristine-reuse and fresh-id-conflict branches of _ensure_experiment
    each stamp the launch's own resolved declaration, not only the fresh-creation branch."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    with training_tools.declare_launcher("gui"):
        # Fresh creation.
        fresh = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-branch-fresh"), str(tmp_path / "out1"))
        assert "error" not in fresh, fresh

        # Pristine pre-created reuse.
        create_experiment("exp-branch-pristine", {"a": 1})
        pristine = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-branch-pristine"), str(tmp_path / "out2"))
        assert "error" not in pristine, pristine
        assert pristine["experiment_id"] == "exp-branch-pristine"

        # Fresh-id conflict: relaunching a config that already has a run mints <id>_<run_id>.
        forked = training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-branch-pristine"), str(tmp_path / "out3"))
        assert "error" not in forked, forked
        assert forked["experiment_id"] != "exp-branch-pristine"

    for result in (fresh, pristine, forked):
        status = read_member(status_key(result["experiment_id"]))
        assert status["launched_by"] == {"launcher": "gui"}


def test_all_training_runs_reads_launched_by_from_the_record_not_the_live_row(tmp_path, monkeypatch):
    """A live row's launched_by is read fresh from the experiment's own status record, never
    carried on the in-memory TrainRun: a run whose stamp failed (or was never made) renders the
    same way any other unstamped record does, launcher not recorded, rather than a caller's
    in-memory intent that the record itself never held."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import create_experiment, update_status
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import _all_training_runs

    # create_experiment/update_status alone, never stamp_run_identity: the shape a failed or
    # dropped stamp leaves behind.
    create_experiment("exp-stamp-write-failed", {"model_source": {"builder": "x:y"}})
    update_status("exp-stamp-write-failed", "running")
    run = create_run({"model_source": {"builder": "x:y"}}, str(tmp_path / "out"),
                     id="exp-stamp-write-failed")

    rows = _all_training_runs(read_progress=False)
    row = next(r for r in rows if r["experiment_id"] == run.id)
    assert row["launched_by"] is None


def test_a_pid_bearing_live_row_takes_launched_by_from_its_disk_overlay(tmp_path, monkeypatch):
    """A pid-bearing row's launched_by is the disk overlay's own reconstructed value, not a
    second independent read of the record: the record this id actually resolves to is stamped
    process (what a second, independent read would answer), while the overlay row is
    reconstruct_from_status's own shape built on a different record stamped gui, and the merged
    row renders the overlay's value rather than the record's own."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        create_experiment, read_member, reconstruct_from_status, stamp_run_identity, status_key,
    )
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools import training_tools

    run = create_run({"model_source": {"builder": "x:y"}}, str(tmp_path / "out"),
                     id="exp-overlay-wins")
    run.pid = 424242

    create_experiment("exp-overlay-wins", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp-overlay-wins", str(tmp_path / "out"),
                        launched_by={"launcher": "process"})

    create_experiment("exp-overlay-source", {"model_source": {"builder": "x:y"}})
    stamp_run_identity("exp-overlay-source", str(tmp_path / "out"),
                        launched_by={"launcher": "gui"})
    other_status = read_member(status_key("exp-overlay-source"))
    overlay_row = reconstruct_from_status("exp-overlay-wins", other_status, stale_seconds=600.0,
                                          read_progress=False)
    overlay_row["external"] = True
    monkeypatch.setattr(training_tools, "_launched_training_runs", lambda **kwargs: [overlay_row])

    rows = training_tools._all_training_runs(read_progress=False)
    row = next(r for r in rows if r["experiment_id"] == run.id)
    assert row["launched_by"] == {"launcher": "gui"}


def test_stamp_run_identity_with_config_refuses_once_metrics_are_logged(tmp_path, monkeypatch):
    """The config-carrying precondition adds metrics_logged to state and output_dir: a record
    still formally "created" but already carrying a metrics row is not pristine, and a refused
    stamp leaves config.json exactly as create_experiment wrote it."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import (
        StampPreconditionFailed, config_key, create_experiment, log_metrics, read_member,
        stamp_run_identity,
    )

    create_experiment("exp-metrics-before-stamp", {"a": 1})
    log_metrics("exp-metrics-before-stamp", 1, {"loss": 0.5})

    with pytest.raises(StampPreconditionFailed):
        stamp_run_identity("exp-metrics-before-stamp", "out", launched_by={"launcher": "process"},
                           config={"a": 2})

    assert read_member(config_key("exp-metrics-before-stamp")) == {"a": 1}


def test_two_launches_racing_a_pristine_record_never_split_the_winners_config_from_its_stamp(
    tmp_path, monkeypatch,
):
    """The pristine-reuse branch stamps a run's config and its identity in one store
    transaction, so two launches that both pass the pristine check can never leave the record
    running under one launch's stamp with the other's config snapshot. Timed wrapper waits
    around ``overwrite_config_if_pristine`` and ``stamp_run_identity`` force the interleaving
    that would split them; the overwrite wrapper is never reached, since the stamp itself
    carries the config, so its own wait times out rather than deadlocking the other thread.
    Both launches must return without error, one id is the pre-created record's own and the
    other a fresh fork naming it as parent, and for each returned id the config snapshot and the
    launch config the run actually trains from carry the identical seed."""
    monkeypatch.chdir(tmp_path)
    import threading

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import config_key, create_experiment, lineage_key, read_member
    from tcip_mcp.tools import training_tools
    from tcip_mcp.tools.training_tools import launch_config_key

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    create_experiment("exp-concurrent-pristine", {"a": 1})
    order = ["a_overwrite", "b_overwrite", "a_stamp", "b_stamp"]
    events = {name: threading.Event() for name in order}

    def _wait_turn(name):
        idx = order.index(name)
        if idx > 0:
            events[order[idx - 1]].wait(timeout=5)

    real_overwrite = experiments_mod.overwrite_config_if_pristine
    real_stamp = experiments_mod.stamp_run_identity

    def fake_overwrite(experiment_id, config, *, root=None):
        name = f"{threading.current_thread().name}_overwrite"
        _wait_turn(name)
        try:
            return real_overwrite(experiment_id, config, root=root)
        finally:
            events[name].set()

    def fake_stamp(experiment_id, output_dir, *, launched_by, config=None):
        name = f"{threading.current_thread().name}_stamp"
        _wait_turn(name)
        try:
            if config is not None:
                return real_stamp(experiment_id, output_dir, launched_by=launched_by,
                                  config=config)
            return real_stamp(experiment_id, output_dir, launched_by=launched_by)
        finally:
            events[name].set()

    monkeypatch.setattr(experiments_mod, "overwrite_config_if_pristine", fake_overwrite,
                        raising=False)
    monkeypatch.setattr(experiments_mod, "stamp_run_identity", fake_stamp, raising=False)

    results = {}

    def _launch(name, out_dir):
        threading.current_thread().name = name
        cfg = _detection_cfg(images_dir, labels_dir, "exp-concurrent-pristine")
        cfg["marker"] = name
        results[name] = training_tools.launch_training(cfg, str(out_dir))

    t_a = threading.Thread(target=_launch, args=("a", tmp_path / "outA"))
    t_b = threading.Thread(target=_launch, args=("b", tmp_path / "outB"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=15)
    t_b.join(timeout=15)

    result_a, result_b = results["a"], results["b"]
    assert "error" not in result_a, result_a
    assert "error" not in result_b, result_b

    ids = {result_a["experiment_id"], result_b["experiment_id"]}
    assert "exp-concurrent-pristine" in ids
    forked_id = next(i for i in ids if i != "exp-concurrent-pristine")
    assert forked_id.startswith("exp-concurrent-pristine_")

    lineage = read_member(lineage_key(forked_id))
    assert lineage["parent_experiment"] == "exp-concurrent-pristine"

    for result in (result_a, result_b):
        eid = result["experiment_id"]
        recorded_seed = read_member(config_key(eid))["seed"]
        launched_seed = read_member(launch_config_key(result["output_dir"]))["seed"]
        assert recorded_seed == launched_seed


def test_launch_refuses_a_dataset_identity_above_the_readers_ceiling(tmp_path, monkeypatch):
    """launch_training reads dataset_identity outside any wrapper: a version-refused document
    refuses the launch by name, before any record exists and before Popen is ever reached. The
    document is written through the store's own put_blob, the platform's own producer for a
    schema_version this reader does not accept (a document register_dataset itself never writes,
    since it always writes the current version). The admitting half (a legal id and a readable
    identity) is test_a_bare_launch_writes_launcher_process above."""
    monkeypatch.chdir(tmp_path)
    import tcip_store as ts

    from tcip_mcp.dataset_layout import dataset_identity_key
    from tcip_mcp.experiments import experiment_ids_with_status
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    key = dataset_identity_key(tmp_path)
    document = {"crop": "test-crop", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-identity-refused"), str(tmp_path / "out"))

    assert result["error"].startswith(
        "launch_training: dataset identity is unreadable at this reader's ceiling, refusing to "
        "train against it untracked:")
    assert captured == []
    assert "exp-identity-refused" not in experiment_ids_with_status()


def test_launch_refuses_an_experiment_id_that_is_not_a_legal_directory_name(tmp_path, monkeypatch):
    """The id becomes the run's own artifact directory: a path separator refuses by name,
    before any process starts."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import experiment_ids_with_status
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "not/legal"), str(tmp_path / "out"))

    assert result["error"] == (
        "launch_training: experiment_id 'not/legal' is not a legal directory "
        "name (a path separator, a drive, an empty name or '.'/'..'), and the id "
        "becomes the run's own artifact directory.")
    assert captured == []
    assert experiment_ids_with_status() == []


def test_launch_refuses_when_the_record_write_raises(tmp_path, monkeypatch):
    """Any exception out of _ensure_experiment's record writes refuses the launch by name,
    before create_run, the launch config or the subprocess, and no status record survives it."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import experiment_ids_with_status
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    captured: list[list[str]] = []
    _fake_popen(monkeypatch, captured)

    def _raise(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(experiments_mod, "create_experiment", _raise)

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-write-raises"), str(tmp_path / "out"))

    assert result["error"] == (
        "launch_training: could not record this run's experiment: store unavailable")
    assert captured == []
    assert experiment_ids_with_status() == []


def test_fresh_launch_records_the_seed_it_trains_with(tmp_path, monkeypatch):
    """No seed in the caller's config: the seed drawn before the record is written and the seed
    the run actually trains from (its own launch config) are the identical value."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import config_key, read_member
    from tcip_mcp.tools import training_tools
    from tcip_mcp.tools.training_tools import launch_config_key

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-fresh-seed"), str(tmp_path / "out"))
    assert "error" not in result, result

    recorded = read_member(config_key(result["experiment_id"]))
    launched = read_member(launch_config_key(result["output_dir"]))
    assert recorded["seed"] == launched["seed"]


def test_pristine_reuse_launch_records_the_seed_it_trains_with(tmp_path, monkeypatch):
    """A pre-created record's own snapshot carries no seed (written before the launch that
    reuses it draws one): the refreshed snapshot and the launch config carry the identical
    drawn value."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.experiments import config_key, create_experiment, read_member
    from tcip_mcp.tools import training_tools
    from tcip_mcp.tools.training_tools import launch_config_key

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    create_experiment("exp-pristine-seed", {"a": 1})
    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-pristine-seed"), str(tmp_path / "out"))
    assert "error" not in result, result
    assert result["experiment_id"] == "exp-pristine-seed"

    recorded = read_member(config_key("exp-pristine-seed"))
    launched = read_member(launch_config_key(result["output_dir"]))
    assert recorded["seed"] == launched["seed"]


def test_launch_refuses_when_the_forks_own_fresh_id_is_already_taken(tmp_path, monkeypatch):
    """mint_experiment_id is pinned so the fork branch's own fresh id is predictable, and that
    id is pre-created before the relaunch runs: the fork's create_experiment call finds it
    already exists, and the launch refuses by name rather than silently training under a record
    this launch never created. exp-fork-taken carries recorded history, so a relaunch against it
    can only reach the fork branch."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import create_experiment, log_metrics, update_status
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    create_experiment("exp-fork-taken", {"a": 1})
    update_status("exp-fork-taken", "running")
    log_metrics("exp-fork-taken", 1, {"loss": 0.5})

    monkeypatch.setattr(experiments_mod, "mint_experiment_id", lambda: "pinned-fork-id")
    create_experiment("exp-fork-taken_pinned-fork-id", {"already": "here"})

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-fork-taken"), str(tmp_path / "out"))

    assert "error" in result
    assert "exp-fork-taken_pinned-fork-id" in result["error"]


def test_all_training_runs_answers_launched_by_none_for_a_malformed_live_id(tmp_path, monkeypatch):
    """create_run reached directly (never through launch_training's own directory-name refusal)
    can register an id no record could ever carry; _all_training_runs answers it the way
    reconstruct_run_status and cancel_run already fold BadKey, rather than raising it out of the
    whole merge."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.training_tools import _all_training_runs

    run = create_run({"model_source": {"builder": "x:y"}}, str(tmp_path / "out"),
                     id="not/a/single/name")

    rows = _all_training_runs(read_progress=False)
    row = next(r for r in rows if r["experiment_id"] == run.id)
    assert row["launched_by"] is None


def test_spawn_failure_after_the_stamp_leaves_a_running_record_that_forks_on_relaunch(
    tmp_path, monkeypatch,
):
    """No cooperative grace period covers the spawn itself: a Popen failure after the stamp
    propagates out of launch_training uncaught (nothing wraps the spawn), the record it stamped
    is left running with no process, reads interrupted once its heartbeat window has passed, and
    a relaunch against the same id forks rather than reusing a record with no recorded history
    of its own."""
    monkeypatch.chdir(tmp_path)
    import subprocess

    from tcip_mcp.experiments import reconstruct_run_status
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)

    def _raise_popen(*args, **kwargs):
        raise OSError("no such executable")

    monkeypatch.setattr(subprocess, "Popen", _raise_popen)

    with pytest.raises(OSError):
        training_tools.launch_training(
            _detection_cfg(images_dir, labels_dir, "exp-spawn-fails"), str(tmp_path / "out"))

    disk = reconstruct_run_status("exp-spawn-fails", stale_seconds=0.0)
    assert disk["status"] == "interrupted"

    _fake_popen(monkeypatch, [])
    relaunch = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-spawn-fails"), str(tmp_path / "out2"))
    assert "error" not in relaunch, relaunch
    assert relaunch["experiment_id"].startswith("exp-spawn-fails_")


def test_a_store_refused_stamp_leaves_the_record_pristine_and_the_launch_refuses(
    tmp_path, monkeypatch,
):
    """A store failure during the stamp (not a StampPreconditionFailed, a real write refusal)
    propagates out of _ensure_experiment as the exception it is, caught only by
    launch_training's own except Exception; the record it tried to stamp is left exactly as
    create_experiment wrote it, since the stamp's own transaction never applied a write."""
    monkeypatch.chdir(tmp_path)
    from tcip_store import StoreError

    from tcip_mcp import experiments as experiments_mod
    from tcip_mcp.experiments import read_member, status_key
    from tcip_mcp.tools import training_tools

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    _seed_one_image(images_dir, labels_dir)
    _fake_popen(monkeypatch, [])

    def _raise(*args, **kwargs):
        raise StoreError("disk full")

    monkeypatch.setattr(experiments_mod, "stamp_run_identity", _raise)

    result = training_tools.launch_training(
        _detection_cfg(images_dir, labels_dir, "exp-stamp-store-error"), str(tmp_path / "out"))

    assert result["error"] == (
        "launch_training: could not record this run's experiment: disk full")

    status = read_member(status_key("exp-stamp-store-error"))
    assert status["state"] == "created"
    assert "output_dir" not in status

"""generic_trainer unit tests: minted-id uniqueness, the run's stated seeding, terminal
status on setup failure, and atomic checkpoint writes.
"""

import threading

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.training import generic_trainer as gt
from tcip_mcp.pipelines.training.generic_trainer import train

# mint_experiment_id: id uniqueness (same-second and cross-thread)


def test_mint_experiment_id_unique_within_one_second():
    from tcip_mcp.experiments import mint_experiment_id

    ids = {mint_experiment_id() for _ in range(50)}
    assert (
        len(ids) == 50
    )  # the uuid suffix, not a counter, keeps same-second calls from colliding


def test_mint_experiment_id_unique_across_threads():
    from tcip_mcp.experiments import mint_experiment_id

    n = 16
    barrier = threading.Barrier(n)
    results: list[str] = []
    lock = threading.Lock()

    def make():
        barrier.wait()  # maximize same-instant contention
        minted = mint_experiment_id()
        with lock:
            results.append(minted)

    threads = [threading.Thread(target=make) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == n


# the run's own seeding: the seed and determinism its config states


def _context(tmp_path, **stated):
    """The context of a one-epoch regression run over two frames and no validation side, with
    ``stated``'s config keys, opened by the launcher's own producer.
    """
    from tests.tiny_trainer_fixtures import (
        frames_data,
        regressor_config,
        run_context,
    )

    data = frames_data(tmp_path / "ds", [0.1, 0.5], [0.2, 1.0])
    return run_context(tmp_path, regressor_config(data=data, **stated))


def test_train_applies_the_runs_seed(tmp_path, monkeypatch):
    """The trainer seeds every stream at the seed and determinism the run's config states."""
    captured = {}

    def fake_set_seed(seed, *, deterministic):
        captured["seed"] = seed
        captured["deterministic"] = deterministic

    monkeypatch.setattr(gt, "set_seed", fake_set_seed)
    ctx = _context(tmp_path, seed=123, deterministic=True)
    run = train(ctx.run, ctx.train_loader)

    assert run.status == "completed", run.status_error
    assert captured == {"seed": 123, "deterministic": True}


# ====================================================================
# train: setup failures must reach a terminal status, never strand "running"
# ====================================================================


def test_train_with_unwritable_output_dir_marks_run_failed(tmp_path):
    import shutil

    ctx = _context(tmp_path)
    # The run's directory replaced by a file, so out_dir.mkdir() raises.
    shutil.rmtree(ctx.run_dir)
    ctx.run_dir.write_text("I am a file, not a directory")
    run = train(ctx.run, ctx.train_loader)

    assert run.status == "failed"  # not stuck at "running"
    assert run.status_error
    assert run.end_time >= run.start_time > 0


# ── checkpoints: durable, torn-read-free writes ──────────────────────


def _run_artifacts(directory):
    """What a run left behind, without the storage layer's own lock bookkeeping."""
    return sorted(
        p.name for p in directory.iterdir() if not p.name.endswith(".lock")
    )


def test_a_checkpoint_loads_back_exactly_what_was_saved(tmp_path):
    """A checkpoint is only worth writing if ``torch.load`` returns the payload unchanged."""
    path = gt.checkpoint_path(tmp_path, "model_best")

    landed = gt.write_checkpoint(
        {"epoch": 3, "weights": torch.zeros(2, 2)}, path
    )

    loaded = torch.load(landed, weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["weights"], torch.zeros(2, 2))
    assert landed == tmp_path / "model_best.pt"
    assert _run_artifacts(tmp_path) == ["model_best.pt"]


def test_a_failed_checkpoint_write_preserves_the_previous_one(
    tmp_path, monkeypatch
):
    path = gt.checkpoint_path(tmp_path, "model_best")
    gt.write_checkpoint({"epoch": 1}, path)

    def broken_save(*args, **kwargs):
        raise RuntimeError("simulated crash mid-serialization")

    monkeypatch.setattr(gt.torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        gt.write_checkpoint({"epoch": 2}, path)
    monkeypatch.undo()

    # The previous checkpoint is intact and loadable; the staged file was cleaned up.
    assert (
        torch.load(tmp_path / "model_best.pt", weights_only=False)["epoch"]
        == 1
    )
    assert _run_artifacts(tmp_path) == ["model_best.pt"]


# capture_rng_state / restore_rng_state: put the four streams back where they were


def test_capture_and_restore_rng_state_roundtrip():
    import random

    import numpy as np

    gt.set_seed(0, deterministic=False)
    state = gt.capture_rng_state(None)
    expected_next = (random.random(), np.random.rand(), torch.rand(1))

    # Advance every stream further, simulating a diagnostic that draws from them.
    random.random(), np.random.rand(), torch.rand(1)

    gt.restore_rng_state(state, None)
    got_next = (random.random(), np.random.rand(), torch.rand(1))

    assert got_next[0] == expected_next[0]
    assert got_next[1] == expected_next[1]
    assert torch.equal(got_next[2], expected_next[2])

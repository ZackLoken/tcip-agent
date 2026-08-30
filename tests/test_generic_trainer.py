"""generic_trainer unit tests: run-id uniqueness, seed defaulting, terminal
status on setup failure, and atomic checkpoint writes."""

import threading

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.training import generic_trainer as gt
from tcip_mcp.pipelines.training import run_registry as rr
from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.run_registry import create_run


# ====================================================================
# create_run: run-id uniqueness (same-second and cross-thread)
# ====================================================================

def test_create_run_ids_unique_within_one_second():
    ids = {create_run({"model_source": {}}, "out").run_id for _ in range(50)}
    assert len(ids) == 50  # len(_RUNS)-suffixed ids would collide here


def test_create_run_ids_unique_across_threads():
    n = 16
    barrier = threading.Barrier(n)
    results: list[str] = []
    lock = threading.Lock()

    def make():
        barrier.wait()  # maximize same-instant contention
        run = create_run({"model_source": {}}, "out")
        with lock:
            results.append(run.run_id)

    threads = [threading.Thread(target=make) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == n
    # Every run must be retrievable under its own id (no silent overwrite).
    for run_id in results:
        assert rr.get_run(run_id) is not None
        assert rr.get_run(run_id).run_id == run_id


# ====================================================================
# create_run: reproducibility, every run gets a recorded seed
# ====================================================================

def test_create_run_draws_and_records_seed_when_unset():
    config = {"model_source": {}}
    run = create_run(config, "out")
    seed = run.config.get("seed")
    assert isinstance(seed, int) and 0 <= seed < 2**31
    # The caller's dict is mutated in place: launch_training snapshots this same
    # dict into the experiment record *after* create_run, so the effective seed
    # must land in it, not in a detached copy.
    assert config["seed"] == seed


def test_create_run_keeps_explicit_top_level_seed():
    run = create_run({"model_source": {}, "seed": 123}, "out")
    assert run.config["seed"] == 123


def test_create_run_keeps_training_section_seed():
    config = {"model_source": {}, "training": {"seed": 7}}
    run = create_run(config, "out")
    assert "seed" not in run.config  # no competing top-level override drawn
    assert run.config["training"]["seed"] == 7


def test_create_run_drawn_seeds_are_independent():
    seeds = {create_run({"model_source": {}}, "out").config["seed"] for _ in range(8)}
    assert len(seeds) == 8  # OS entropy per run, not one fixed default


def test_train_applies_the_drawn_seed(tmp_path, monkeypatch):
    captured = {}

    def fake_set_seed(seed, deterministic=False):
        captured["seed"] = seed
        captured["deterministic"] = deterministic

    monkeypatch.setattr(gt, "set_seed", fake_set_seed)
    run = create_run({"model_source": {}}, str(tmp_path / "out"))
    train(run, train_loader=None, task="classification")  # fails at build, after seeding

    assert captured["seed"] == run.config["seed"]
    assert captured["deterministic"] is False


# ====================================================================
# train: setup failures must reach a terminal status, never strand "running"
# ====================================================================

def test_train_with_unwritable_output_dir_marks_run_failed(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("I am a file, not a directory")

    # output_dir nests under an existing *file*, so out_dir.mkdir() raises.
    run = create_run({"model_source": {}}, str(blocker / "out"))
    run = train(run, train_loader=None, task="classification")

    assert run.status == "failed"  # not stuck at "running"
    assert run.error
    assert run.end_time >= run.start_time > 0


# ── checkpoints: durable, torn-read-free writes ──────────────────────


def _run_artifacts(directory):
    """What a run left behind, without the storage layer's own lock bookkeeping."""
    return sorted(p.name for p in directory.iterdir() if not p.name.endswith(".lock"))


def test_a_checkpoint_loads_back_exactly_what_was_saved(tmp_path):
    """A checkpoint is only worth writing if ``torch.load`` returns the payload unchanged."""
    key = gt.checkpoint_key(tmp_path, "model_best")

    landed = gt.write_checkpoint({"epoch": 3, "weights": torch.zeros(2, 2)}, key)

    loaded = torch.load(landed, weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["weights"], torch.zeros(2, 2))
    assert landed == tmp_path / "model_best.pt"
    assert _run_artifacts(tmp_path) == ["model_best.pt"]


def test_a_failed_checkpoint_write_preserves_the_previous_one(tmp_path, monkeypatch):
    key = gt.checkpoint_key(tmp_path, "model_best")
    gt.write_checkpoint({"epoch": 1}, key)

    def broken_save(*args, **kwargs):
        raise RuntimeError("simulated crash mid-serialization")

    monkeypatch.setattr(gt.torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        gt.write_checkpoint({"epoch": 2}, key)
    monkeypatch.undo()

    # The previous checkpoint is intact and loadable; the staged file was cleaned up.
    assert torch.load(tmp_path / "model_best.pt", weights_only=False)["epoch"] == 1
    assert _run_artifacts(tmp_path) == ["model_best.pt"]


# capture_rng_state / restore_rng_state: put the four streams back where they were

def test_capture_and_restore_rng_state_roundtrip():
    import random

    import numpy as np

    gt.set_seed(0)
    state = gt.capture_rng_state()
    expected_next = (random.random(), np.random.rand(), torch.rand(1))

    # Advance every stream further, simulating a diagnostic that draws from them.
    random.random(), np.random.rand(), torch.rand(1)

    gt.restore_rng_state(state)
    got_next = (random.random(), np.random.rand(), torch.rand(1))

    assert got_next[0] == expected_next[0]
    assert got_next[1] == expected_next[1]
    assert torch.equal(got_next[2], expected_next[2])

"""generic_trainer unit tests — run-id uniqueness, seed defaulting, terminal
status on setup failure, and atomic checkpoint writes."""

import threading

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.training import generic_trainer as gt
from tcip_mcp.pipelines.training.generic_trainer import (
    _atomic_torch_save,
    create_run,
    train,
)


# ====================================================================
# create_run — run-id uniqueness (same-second and cross-thread)
# ====================================================================

def test_create_run_ids_unique_within_one_second():
    ids = {create_run({"model_spec": {}}, "out").run_id for _ in range(50)}
    assert len(ids) == 50  # len(_RUNS)-suffixed ids would collide here


def test_create_run_ids_unique_across_threads():
    n = 16
    barrier = threading.Barrier(n)
    results: list[str] = []
    lock = threading.Lock()

    def make():
        barrier.wait()  # maximize same-instant contention
        run = create_run({"model_spec": {}}, "out")
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
        assert gt.get_run(run_id) is not None
        assert gt.get_run(run_id).run_id == run_id


# ====================================================================
# create_run — reproducibility: every run gets a recorded seed
# ====================================================================

def test_create_run_draws_and_records_seed_when_unset():
    config = {"model_spec": {}}
    run = create_run(config, "out")
    seed = run.config.get("seed")
    assert isinstance(seed, int) and 0 <= seed < 2**31
    # The caller's dict is mutated in place: launch_training snapshots this same
    # dict into the experiment record *after* create_run, so the effective seed
    # must land in it, not in a detached copy.
    assert config["seed"] == seed


def test_create_run_keeps_explicit_top_level_seed():
    run = create_run({"model_spec": {}, "seed": 123}, "out")
    assert run.config["seed"] == 123


def test_create_run_keeps_training_section_seed():
    config = {"model_spec": {}, "training": {"seed": 7}}
    run = create_run(config, "out")
    assert "seed" not in run.config  # no competing top-level override drawn
    assert run.config["training"]["seed"] == 7


def test_create_run_drawn_seeds_are_independent():
    seeds = {create_run({"model_spec": {}}, "out").config["seed"] for _ in range(8)}
    assert len(seeds) == 8  # OS entropy per run, not one fixed default


def test_train_applies_the_drawn_seed(tmp_path, monkeypatch):
    captured = {}

    def fake_set_seed(seed, deterministic=False):
        captured["seed"] = seed
        captured["deterministic"] = deterministic

    monkeypatch.setattr(gt, "set_seed", fake_set_seed)
    run = create_run({"model_spec": {}}, str(tmp_path / "out"))
    train(run, train_loader=None, task="classification")  # fails at compose, after seeding

    assert captured["seed"] == run.config["seed"]
    assert captured["deterministic"] is False


# ====================================================================
# train — setup failures must reach a terminal status, never strand "running"
# ====================================================================

def test_train_with_unwritable_output_dir_marks_run_failed(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("I am a file, not a directory")

    # output_dir nests under an existing *file*, so out_dir.mkdir() raises.
    run = create_run({"model_spec": {}}, str(blocker / "out"))
    run = train(run, train_loader=None, task="classification")

    assert run.status == "failed"  # not stuck at "running"
    assert run.error
    assert run.end_time >= run.start_time > 0


# ====================================================================
# _atomic_torch_save — durable, torn-read-free checkpoint writes
# ====================================================================

def test_atomic_torch_save_roundtrips_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "model_best.pt"
    _atomic_torch_save({"epoch": 3, "weights": torch.zeros(2, 2)}, path)

    loaded = torch.load(path, weights_only=False)
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["weights"], torch.zeros(2, 2))
    assert [p.name for p in tmp_path.iterdir()] == ["model_best.pt"]


def test_atomic_torch_save_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "model_best.pt"
    _atomic_torch_save({"epoch": 1}, path)

    def broken_save(*args, **kwargs):
        raise RuntimeError("simulated crash mid-serialization")

    monkeypatch.setattr(gt.torch, "save", broken_save)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _atomic_torch_save({"epoch": 2}, path)
    monkeypatch.undo()

    # The previous checkpoint is intact and loadable; the temp file was cleaned up.
    assert torch.load(path, weights_only=False)["epoch"] == 1
    assert [p.name for p in tmp_path.iterdir()] == ["model_best.pt"]

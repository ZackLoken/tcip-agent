"""Global seeding + resume-from-checkpoint."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from tcip_mcp.pipelines.training.generic_trainer import set_seed  # noqa: E402


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(123)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_set_seed_sets_cudnn_flags():
    det0, bench0 = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    try:
        set_seed(1, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = det0
        torch.backends.cudnn.benchmark = bench0


# --------------------------------------------------------------------------
# Integration (needs torchvision)
# --------------------------------------------------------------------------

pytest.importorskip("torchvision")

import tcip_mcp.pipelines.components.backbones  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.collation import task_collate
from tcip_mcp.pipelines.training.run_registry import create_run  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tests._producer_fixtures import dataset_over  # noqa: E402


def _classification_data(tmp_path: Path, n: int = 6):
    from PIL import Image
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        Image.new("RGB", (32, 32), (40 * (i % 5), 50, 60)).save(images_dir / f"img{i}.png")
        rows.append((f"img{i}", i % 2))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("stem", "label"))
        w.writerows(rows)
    return str(images_dir), str(csv_path)


def _model_source():
    return {"builder": "tests.bespoke_models:build_bespoke_classifier",
            "builder_kwargs": {"num_classes": 2}, "task": "classification"}


def _cfg(stages, **extra):
    cfg = {
        "model_source": _model_source(), "device": "cpu", "stages": stages,
        "mixed_precision": False,
        "optimizer": {"name": "adamw", "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0},
        "early_stopping": {"enabled": False}, "checkpoint_every_n_epochs": 1,
    }
    cfg.update(extra)
    return cfg


def test_seeded_train_reproducible(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)

    def run_once(out):
        ds = dataset_over("classification", images_dir, csv_path)
        loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
        run = create_run(_cfg([{"freeze_to": -1, "epochs": 1}], seed=7), str(out), id="auto-run-51")
        run = train(run, loader, task="classification")
        return run.metrics_history[0]["train_loss"]

    assert run_once(tmp_path / "a") == pytest.approx(run_once(tmp_path / "b"))


def test_resume_continues_epochs(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-52"), loader, task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"), id="auto-run-53")
    run2 = train(run2, loader, task="classification", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_epoch == 2          # continued global epoch count
    assert len(run2.metrics_history) == 1   # only the one remaining epoch
    assert (tmp_path / "out2" / "model_final.pt").is_file()


def test_resume_skips_completed_stage_and_restores_optimizer(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}, {"freeze_to": 0, "epochs": 1}])

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-54"), loader, task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"  # end of stage 0
    assert ckpt.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"), id="auto-run-55")
    run2 = train(run2, loader, task="classification", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_stage == 1          # stage 0 skipped, stage 1 ran
    assert run2.current_epoch == 2          # restored optimizer + ran stage 1's epoch


def test_resume_restores_rng_state_not_just_reseeds(tmp_path):
    """Resuming must restore the RNG stream position, not just reseed from scratch: a resumed
    epoch's loss must match the straight-through run's epoch at the same point, even if the
    global RNG is deliberately corrupted between save and resume. ``shuffle=True`` with no
    explicit ``generator=`` is deliberate, since PyTorch's ``RandomSampler`` draws a fresh
    per-epoch seed from the global torch RNG on every ``__iter__`` call when no generator is
    given, so the resumed epoch's batch order is genuinely sensitive to whatever
    ``torch.set_rng_state`` last set it to; a sequential (unshuffled) loader would make this
    test pass identically whether or not RNG restoration actually ran.
    """
    images_dir, csv_path = _classification_data(tmp_path)

    def build_loader():
        ds = dataset_over("classification", images_dir, csv_path)
        return DataLoader(ds, batch_size=2, shuffle=True, collate_fn=task_collate("classification"))

    cfg = _cfg([{"freeze_to": -1, "epochs": 2}], seed=11)

    # Straight-through baseline: both epochs in one uninterrupted run.
    straight = train(create_run(cfg, str(tmp_path / "straight"), id="auto-run-56"), build_loader(), task="classification")
    baseline_epoch2_loss = straight.metrics_history[1]["train_loss"]

    # Split run: epoch 1 checkpointed, global RNG deliberately corrupted, then resumed for epoch 2.
    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-57"), build_loader(), task="classification")
    ckpt = tmp_path / "out" / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()
    assert "torch_rng_state" in torch.load(ckpt, weights_only=False)

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    resumed = train(create_run(cfg, str(tmp_path / "out2"), id="auto-run-58"), build_loader(),
                    task="classification", resume_from=str(ckpt))
    resumed_epoch2_loss = resumed.metrics_history[0]["train_loss"]  # the one epoch this run ran

    assert resumed_epoch2_loss == pytest.approx(baseline_epoch2_loss)


def test_resume_from_a_checkpoint_missing_a_resume_key_refuses_naming_it(tmp_path):
    """A checkpoint lacking a key the resume reads fails the run naming that key, rather than
    resuming from a guessed default."""
    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-59"), loader, task="classification")
    ckpt_path = tmp_path / "out" / "checkpoint_epoch_1.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    del ckpt["torch_rng_state"]
    torch.save(ckpt, ckpt_path)

    run2 = train(create_run(cfg, str(tmp_path / "out2"), id="auto-run-60"), loader,
                 task="classification", resume_from=str(ckpt_path))
    assert run2.status == "failed"
    assert "torch_rng_state" in run2.error


def test_resume_from_a_checkpoint_missing_its_scheduler_state_refuses_naming_it(tmp_path):
    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-61"), loader, task="classification")
    ckpt_path = tmp_path / "out" / "checkpoint_epoch_1.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt.pop("scheduler_state_dict", None)
    torch.save(ckpt, ckpt_path)

    run2 = train(create_run(cfg, str(tmp_path / "out2"), id="auto-run-62"), loader,
                 task="classification", resume_from=str(ckpt_path))
    assert run2.status == "failed"
    assert "scheduler_state_dict" in run2.error


def test_a_resume_whose_optimizer_restore_raises_fails_the_run(tmp_path, monkeypatch):
    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}])

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-63"), loader, task="classification")

    def _refuse(self, state_dict):
        raise ValueError("optimizer state does not fit")

    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", _refuse)
    run2 = train(create_run(cfg, str(tmp_path / "out2"), id="auto-run-64"), loader,
                 task="classification",
                 resume_from=str(tmp_path / "out" / "checkpoint_epoch_1.pt"))
    assert run2.status == "failed"
    assert "optimizer state does not fit" in run2.error


def _draws() -> tuple:
    """One draw from each of the four generator streams a checkpoint carries."""
    import random

    import numpy as np

    return (random.random(), float(np.random.rand()), float(torch.rand(1)),
            float(torch.rand(1, device="cuda")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_resume_restores_the_rng_streams(tmp_path, monkeypatch):
    """The streams a CUDA resume continues from are the ones the checkpoint captured: the draws
    taken just after the trainer's restore equal the draws the captured states yield when set
    through the four RNG APIs directly, independent of the restore under test."""
    import tcip_mcp.pipelines.training.generic_trainer as trainer

    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 2}], device="cuda")

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-65"), loader, task="classification")
    ckpt_path = tmp_path / "out" / "checkpoint_epoch_1.pt"
    real_restore = trainer.restore_rng_state
    after_restore: list = []

    def _observing_restore(state):
        real_restore(state)
        after_restore.append(_draws())
        real_restore(state)  # the run continues from the restored streams, not the drawn ones

    monkeypatch.setattr(trainer, "restore_rng_state", _observing_restore)
    run2 = train(create_run(cfg, str(tmp_path / "out2"), id="auto-run-66"), loader,
                 task="classification", resume_from=str(ckpt_path))
    assert run2.status == "completed", run2.error
    assert run2.current_epoch == 2

    import random

    import numpy as np

    captured = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    random.setstate(captured["python_rng_state"])
    np.random.set_state(captured["numpy_rng_state"])
    torch.set_rng_state(captured["torch_rng_state"])
    torch.cuda.set_rng_state_all(captured["cuda_rng_state"])
    assert after_restore == [_draws()]


def test_seeded_loader_kwargs_reproducible_shuffle():
    """The platform's own DataLoader construction sites are seeded: two loaders built
    from the same seed shuffle identically; an unseeded run stays honestly unseeded
    (a worker init fn, but no generator)."""
    from tcip_mcp.pipelines.training.generic_trainer import seeded_loader_kwargs

    kw1 = seeded_loader_kwargs(42)
    kw2 = seeded_loader_kwargs(42)
    perm1 = torch.randperm(10, generator=kw1["generator"])
    perm2 = torch.randperm(10, generator=kw2["generator"])
    assert torch.equal(perm1, perm2)

    assert "generator" not in seeded_loader_kwargs(None)


def test_loader_worker_init_fn_is_picklable():
    """Windows spawn pickles worker_init_fn to every DataLoader worker, so a closure here
    breaks every num_workers > 0 run on this platform. Seeded and unseeded runs alike must
    hand out a picklable init fn."""
    import pickle

    from tcip_mcp.pipelines.training.generic_trainer import seeded_loader_kwargs

    for seed in (42, None):
        fn = seeded_loader_kwargs(seed)["worker_init_fn"]
        assert pickle.loads(pickle.dumps(fn)) is not None


def test_loader_worker_init_configures_gdal_cache_and_seeds(monkeypatch):
    """Every spawned worker starts on GDAL's stock cache default, so the init fn must apply
    the platform budget in the worker; per-worker numpy/random seeding applies only when the
    run is seeded."""
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.training.generic_trainer import seeded_loader_kwargs

    calls: list[str] = []
    monkeypatch.setattr(raster_source, "configure_gdal_cache",
                        lambda share=1.0: calls.append("cache"))

    seeded_loader_kwargs(7)["worker_init_fn"](worker_id=1)
    first = (random.random(), float(np.random.rand()))
    seeded_loader_kwargs(7)["worker_init_fn"](worker_id=1)
    again = (random.random(), float(np.random.rand()))
    assert first == again

    seeded_loader_kwargs(None)["worker_init_fn"](worker_id=0)
    assert calls == ["cache", "cache", "cache"]


def test_resume_from_non_resumable_checkpoint_fails_loudly(tmp_path):
    # Resuming a checkpoint without optimizer state (e.g. model_best.pt) must fail
    # loudly, not silently restart from scratch.
    images_dir, csv_path = _classification_data(tmp_path)
    ds = dataset_over("classification", images_dir, csv_path)
    loader = DataLoader(ds, batch_size=2, collate_fn=task_collate("classification"))
    cfg = _cfg([{"freeze_to": -1, "epochs": 1}])

    train(create_run(cfg, str(tmp_path / "out"), id="auto-run-61"), loader, task="classification")
    best = tmp_path / "out" / "model_best.pt"   # has model_state_dict but no optimizer_state_dict
    assert best.is_file()

    run2 = create_run(cfg, str(tmp_path / "out2"), id="auto-run-62")
    run2 = train(run2, loader, task="classification", resume_from=str(best))
    assert run2.status == "failed"
    assert "resume" in (run2.error or "").lower()
    assert not (tmp_path / "out2" / "model_final.pt").is_file()  # did not silently train

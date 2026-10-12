"""Global seeding + resume-from-checkpoint."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tests._chain_fixtures import CLASSIFIER_SOURCE

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from tcip_mcp.pipelines.training.generic_trainer import set_seed  # noqa: E402


def test_set_seed_reproducible():
    set_seed(123, deterministic=False)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_seed(123, deterministic=False)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_set_seed_sets_cudnn_flags():
    det0, bench0 = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
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
import tcip_mcp.pipelines.components.heads  # noqa: F401,E402
import tcip_mcp.pipelines.components.losses  # noqa: F401,E402
import tcip_mcp.pipelines.components.necks  # noqa: F401,E402
from tcip_mcp.pipelines.training.generic_trainer import run_loaders, train  # noqa: E402
from tests._producer_fixtures import dataset_over  # noqa: E402
from tests._training_values import (  # noqa: E402
    adamw_optimizer,
    drawn_split,
    evaluation_block,
    schedule,
    stage,
)


def _classification_data(tmp_path: Path, n: int = 6):
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [(i % 5) / 5 for i in range(n)],
        [i % 2 for i in range(n)],
        height=32,
        width=32,
    )
    return str(images_dir), str(csv_path)


def _model_source():
    return dict(CLASSIFIER_SOURCE)


def _cfg(stages, horizon: int = 1, **extra):
    """A classifier config of ``stages``, each ending at its cosine schedule's ``horizon``."""
    from tests._chain_fixtures import training_config

    return training_config(
        _model_source(),
        {},
        stages=stages,
        optimizer=adamw_optimizer(),
        scheduler=schedule("cosine", horizon_epochs=horizon),
        evaluation=evaluation_block(selection_metric="loss"),
        **extra,
    )


def _run_dir(project: Path, name: str) -> Path:
    """The directory of the run of ``project`` named ``name``."""
    from tcip_mcp.experiments import experiment_dir

    return experiment_dir(name, project=project)


def _trained(
    data: tuple[str, str], project: Path, cfg: dict, name: str, **kwargs
):
    """The run named ``name`` of ``project`` over ``cfg`` and ``data``'s frames and table, opened
    by the launcher's own producer with no validation side and trained through the training
    loader the platform builds for it (``generic_trainer.run_loaders``); ``kwargs`` are
    ``train``'s.
    """
    from tcip_mcp.experiments import observe
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tests._verified_checkpoint_fixtures import opened_run

    images_dir, labels_dir = data
    run = observed_run(
        observe(
            opened_run(
                project,
                {
                    **cfg,
                    "data": {
                        "images_dir": images_dir,
                        "labels_dir": labels_dir,
                        "split": drawn_split(val_ratio=0.0),
                    },
                },
                experiment_id=name,
            )
        )
    )
    ds = dataset_over("classification", *data)
    return train(run, run_loaders(run, ds, None)[0], **kwargs)


def test_seeded_train_reproducible(tmp_path):
    data = _classification_data(tmp_path)

    def run_once(name):
        run = _trained(data, tmp_path, _cfg([stage(-1)], seed=7), name)
        return run.metrics_history[0]["train_loss"]

    assert run_once("a") == pytest.approx(run_once("b"))


def test_resume_continues_epochs(tmp_path):
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)], 2)

    _trained(data, tmp_path, cfg, "out")
    ckpt = _run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()

    run2 = _trained(data, tmp_path, cfg, "out2", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_epoch == 2  # continued global epoch count
    assert len(run2.metrics_history) == 1  # only the one remaining epoch
    assert (_run_dir(tmp_path, "out2") / "model_final.pt").is_file()


def test_resume_skips_completed_stage_and_restores_optimizer(tmp_path):
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1), stage(0)])

    _trained(data, tmp_path, cfg, "out")
    ckpt = (
        _run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"
    )  # end of stage 0
    assert ckpt.is_file()

    run2 = _trained(data, tmp_path, cfg, "out2", resume_from=str(ckpt))
    assert run2.status == "completed"
    assert run2.current_stage == 1  # stage 0 skipped, stage 1 ran
    assert run2.current_epoch == 2  # restored optimizer + ran stage 1's epoch


def test_resume_restores_rng_state_not_just_reseeds(tmp_path):
    """Resuming must restore the RNG stream positions, not just reseed from scratch: a resumed
    epoch's loss must match the straight-through run's epoch at the same point, even if the
    global RNG is deliberately corrupted between save and resume. The run's loader shuffles
    from a generator of its own seeded from the run's config, whose position the checkpoint
    captures beside the global streams.
    """
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)], 2, seed=11)

    # Straight-through baseline: both epochs in one uninterrupted run.
    straight = _trained(data, tmp_path, cfg, "straight")
    baseline_epoch2_loss = straight.metrics_history[1]["train_loss"]

    # Split run: epoch 1 checkpointed, global RNG deliberately corrupted, then resumed for epoch 2.
    _trained(data, tmp_path, cfg, "out")
    ckpt = _run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"
    assert ckpt.is_file()
    assert "torch_rng_state" in torch.load(ckpt, weights_only=False)

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    resumed = _trained(data, tmp_path, cfg, "out2", resume_from=str(ckpt))
    resumed_epoch2_loss = resumed.metrics_history[0][
        "train_loss"
    ]  # the one epoch this run ran

    assert resumed_epoch2_loss == pytest.approx(baseline_epoch2_loss)


def test_resume_from_a_checkpoint_missing_a_resume_key_refuses_naming_it(
    tmp_path,
):
    """A checkpoint lacking a key the resume reads fails the run naming that key, rather than
    resuming from a guessed default.
    """
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)], 2)

    _trained(data, tmp_path, cfg, "out")
    ckpt_path = _run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    del ckpt["torch_rng_state"]
    torch.save(ckpt, ckpt_path)

    run2 = _trained(data, tmp_path, cfg, "out2", resume_from=str(ckpt_path))
    assert run2.status == "failed"
    assert "torch_rng_state" in run2.status_error


def test_resume_from_a_checkpoint_missing_its_scheduler_state_refuses_naming_it(
    tmp_path,
):
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)], 2)

    _trained(data, tmp_path, cfg, "out")
    ckpt_path = _run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt.pop("scheduler_state_dict", None)
    torch.save(ckpt, ckpt_path)

    run2 = _trained(data, tmp_path, cfg, "out2", resume_from=str(ckpt_path))
    assert run2.status == "failed"
    assert "scheduler_state_dict" in run2.status_error


def test_a_resume_whose_optimizer_restore_raises_fails_the_run(
    tmp_path, monkeypatch
):
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)], 2)

    _trained(data, tmp_path, cfg, "out")

    import tcip_mcp.pipelines.training.generic_trainer as trainer

    def _refuse(model, optimizer, state, *, group_settings):
        raise ValueError("optimizer state does not fit")

    monkeypatch.setattr(trainer, "restore_training_state", _refuse)
    run2 = _trained(
        data,
        tmp_path,
        cfg,
        "out2",
        resume_from=str(_run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"),
    )
    assert run2.status == "failed"
    assert "optimizer state does not fit" in run2.status_error


def _draws() -> tuple:
    """One draw from each of the four generator streams a checkpoint carries."""
    import random

    import numpy as np

    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1)),
        float(torch.rand(1, device="cuda")),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
def test_a_cuda_resume_restores_the_rng_streams(tmp_path, monkeypatch):
    """The streams a CUDA resume continues from are the ones the checkpoint captured: the draws
    taken just after the trainer's restore equal the draws the captured states yield when set
    through the four RNG APIs directly, independent of the restore under test.
    """
    import tcip_mcp.pipelines.training.generic_trainer as trainer

    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)], 2, device="cuda")

    _trained(data, tmp_path, cfg, "out")
    ckpt_path = _run_dir(tmp_path, "out") / "checkpoint_epoch_1.pt"
    real_restore = trainer.restore_rng_state
    after_restore: list = []

    def _observing_restore(state, loader_generator):
        real_restore(state, loader_generator)
        after_restore.append(_draws())
        # The run continues from the restored streams, not the drawn ones.
        real_restore(state, loader_generator)

    monkeypatch.setattr(trainer, "restore_rng_state", _observing_restore)
    run2 = _trained(data, tmp_path, cfg, "out2", resume_from=str(ckpt_path))
    assert run2.status == "completed", run2.status_error
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
    from the same seed shuffle identically.
    """
    from tcip_mcp.pipelines.training.generic_trainer import (
        seeded_loader_kwargs,
    )

    kw1 = seeded_loader_kwargs(42)
    kw2 = seeded_loader_kwargs(42)
    perm1 = torch.randperm(10, generator=kw1["generator"])
    perm2 = torch.randperm(10, generator=kw2["generator"])
    assert torch.equal(perm1, perm2)


def test_loader_worker_init_fn_is_picklable():
    """Windows spawn pickles worker_init_fn to every DataLoader worker, so a closure here
    breaks every num_workers > 0 run on this platform.
    """
    import pickle

    from tcip_mcp.pipelines.training.generic_trainer import (
        seeded_loader_kwargs,
    )

    fn = seeded_loader_kwargs(42)["worker_init_fn"]
    assert pickle.loads(pickle.dumps(fn)) is not None


def test_loader_worker_init_configures_gdal_cache_and_seeds(monkeypatch):
    """Every spawned worker starts on GDAL's stock cache default, so the init fn must apply
    the platform budget in the worker and seed numpy and random from the seed torch gave that
    worker (``torch.initial_seed`` inside it, set here as torch sets it).
    """
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.training.generic_trainer import (
        seeded_loader_kwargs,
    )

    calls: list[str] = []
    monkeypatch.setattr(
        raster_source,
        "configure_gdal_cache",
        lambda share=1.0: calls.append("cache"),
    )

    with torch.random.fork_rng():
        torch.manual_seed(2**32 + 21)
        seeded_loader_kwargs(7)["worker_init_fn"](worker_id=1)
    drawn = (random.random(), float(np.random.rand()))
    assert drawn == (
        random.Random(21).random(),
        float(np.random.RandomState(21).rand()),
    )
    assert calls == ["cache"]


def test_resume_from_non_resumable_checkpoint_fails_loudly(tmp_path):
    # Resuming a checkpoint without optimizer state (e.g. model_best.pt) must fail
    # loudly, not silently restart from scratch.
    data = _classification_data(tmp_path)
    cfg = _cfg([stage(-1)])

    _trained(data, tmp_path, cfg, "out")
    best = (
        _run_dir(tmp_path, "out") / "model_best.pt"
    )  # weights, no optimizer_state_dict
    assert best.is_file()

    run2 = _trained(data, tmp_path, cfg, "out2", resume_from=str(best))
    assert run2.status == "failed"
    assert "resume" in (run2.status_error or "").lower()
    assert not (
        _run_dir(tmp_path, "out2") / "model_final.pt"
    ).is_file()  # did not silently train

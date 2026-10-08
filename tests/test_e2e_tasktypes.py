"""Minimal end-to-end smoke tests for each task type via bespoke ``model_source`` builders.

The existing ``test_full_pipeline.py`` covers classification (synthetic) and
detection (gated on real sample data). This file locks the
build_model -> build_dataset -> train contract for the task types that were
otherwise untested, on tiny synthetic data, CPU, one epoch. The assertion is
deliberately weak: the pipeline runs to completion and produces finite losses
and a checkpoint. That is enough to catch the regressions this repo is prone to
(e.g. a dataset/head/collate shape mismatch).

Each smoke drives one of the sibling bespoke builders in ``tests.bespoke_models``.

Kept tiny (a handful of 64x64 images, 1 epoch) to stay well under the CI
per-test timeout.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import csv
import math
from functools import partial
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from torch.utils.data import DataLoader

from tcip_mcp.pipelines.training.generic_trainer import train
from tcip_mcp.pipelines.training.collation import task_collate  # noqa: E402
from tests.tiny_trainer_fixtures import trainer_run  # noqa: E402
from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    BESPOKE_INSTANCE_SEG, BESPOKE_ORDINAL, BESPOKE_REGRESSOR, BESPOKE_SEMANTIC_SEG,
)
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import dataset_over, label_image, run_over  # noqa: E402

IMG = 64


# --------------------------------------------------------------------------
# Synthetic-data helpers
# --------------------------------------------------------------------------

_save_png = partial(write_noise_image, size=IMG)


def _model_source(builder: str, task: str, **kwargs) -> dict:
    return {"builder": builder, "builder_kwargs": kwargs, "task": task}


def _train_config(model_source: dict, data: dict) -> dict:
    from tests._chain_fixtures import ADAMW, training_config

    # No val_loader at any call site below: loss is the only metric coherent to select on
    # without one, detection/instance_seg's own default (objective) needs a validation pass.
    return training_config(model_source, data, optimizer=ADAMW,
                           evaluation={"selection_metric": "loss"})


def _run(model_source: dict, data: dict, tmp_path: Path, run_id: str):
    """A run of :func:`_train_config` writing into ``tmp_path / "out"``, with no val loader."""
    return trainer_run(_train_config(model_source, data), tmp_path / "out",
                       has_val_loader=False, id=run_id, project=tmp_path)


def _assert_trained(run, output_dir: Path) -> None:
    assert run.status == "completed", run.status_error
    assert run.current_epoch == 1
    assert run.metrics_history
    train_loss = run.metrics_history[-1]["train_loss"]
    assert math.isfinite(train_loss), f"non-finite train_loss: {train_loss}"
    assert (output_dir / "model_best.pt").is_file()


# --------------------------------------------------------------------------
# Box-style tasks (detection, instance_seg): torchvision detector path
# --------------------------------------------------------------------------

def test_detection_e2e(tmp_path: Path):
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # one centered box covering the middle of the image
        label_image(images_dir / f"img{i}.png",
                    [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))], IMG, IMG)

    dataset, data = run_over("detection", str(images_dir), subject="bud")
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("detection"))

    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    run = _run(BUILT_DETECTOR, data, tmp_path, "auto-run-15")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")


def test_instance_seg_e2e(tmp_path: Path):
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    square = Polygon([[(19.2, 19.2), (44.8, 19.2), (44.8, 44.8), (19.2, 44.8)]])
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # a square polygon (>= 3 vertices)
        label_image(images_dir / f"img{i}.png", [Annotation(subject="bud", geometry=square)],
                    IMG, IMG)

    dataset, data = run_over("instance_seg", str(images_dir), subject="bud")
    # Guard the polygon -> mask rasterization path (datasets.py). With the mask_rcnn
    # detector these masks now reach the Mask R-CNN mask loss during training.
    assert dataset[0][1]["masks"].shape[0] > 0
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("instance_seg"))

    model_source = _model_source(BESPOKE_INSTANCE_SEG, "instance_seg", min_size=IMG,
                                 max_size=IMG * 2)
    run = _run(model_source, data, tmp_path, "auto-run-16")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")


# --------------------------------------------------------------------------
# Dense / scalar tasks (semantic_seg, ordinal, regression): stack-collate path
# --------------------------------------------------------------------------

def test_semantic_seg_e2e(tmp_path: Path):
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    from tests._producer_fixtures import painted_frame

    block = (IMG // 4, IMG // 4, IMG // 2, IMG // 2)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        painted_frame(IMG, IMG, 0, [(block, 1)], mode="L").save(masks_dir / f"img{i}.png")

    dataset, data = run_over("semantic_seg", str(images_dir), str(masks_dir))
    loader = DataLoader(dataset, batch_size=2, collate_fn=task_collate("semantic_seg"))

    model_source = _model_source(BESPOKE_SEMANTIC_SEG, "semantic_seg")
    run = _run(model_source, data, tmp_path, "auto-run-17")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")


def _write_csv(path: Path, rows: list[tuple[str, object]], header: tuple[str, str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_ordinal_e2e(tmp_path: Path):
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png", bright=(i % 3 == 0))
        rows.append((f"img{i}", i % 3))  # ranks 0..2
    csv_path = tmp_path / "ranks.csv"
    _write_csv(csv_path, rows, ("stem", "rank"))

    dataset, data = run_over("ordinal", str(images_dir), str(csv_path))
    loader = DataLoader(dataset, batch_size=3, collate_fn=task_collate("ordinal"))

    model_source = _model_source(BESPOKE_ORDINAL, "ordinal")
    run = _run(model_source, data, tmp_path, "auto-run-18")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")


def test_ordinal_derives_num_ranks_from_data(tmp_path: Path):
    """A rank scale of 7 ranks must derive its count from the CSV, never truncate to a narrower
    bound. crops.yml's kernel_pellicle trait is a real 1-7 scale (ranks 0-6, 7 ranks) once
    0-indexed."""
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for rank in range(7):
        _save_png(images_dir / f"img{rank}.png", bright=(rank >= 4))
        rows.append((f"img{rank}", rank))
    csv_path = tmp_path / "ranks.csv"
    _write_csv(csv_path, rows, ("stem", "rank"))

    dataset, data = run_over("ordinal", str(images_dir), str(csv_path))
    assert data["num_ranks"] == 7 and data.get("num_classes") is None

    loader = DataLoader(dataset, batch_size=7, collate_fn=task_collate("ordinal"))
    model_source = _model_source(BESPOKE_ORDINAL, "ordinal")
    run = _run(model_source, data, tmp_path, "auto-run-19")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")


def test_ordinal_num_ranks_stated_raises(tmp_path: Path):
    """A caller-configured num_ranks raises: the rank count is the CSV's own, so a stated one
    (here too small, which would train the excess ranks as silent duplicates of the top rank)
    is refused rather than compared."""
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for rank in range(7):
        _save_png(images_dir / f"img{rank}.png", bright=(rank >= 4))
        rows.append((f"img{rank}", rank))
    csv_path = tmp_path / "ranks.csv"
    _write_csv(csv_path, rows, ("stem", "rank"))

    with pytest.raises(ValueError, match="num_ranks"):
        dataset_over("ordinal", str(images_dir), str(csv_path), stated={"num_ranks": 5})


def test_classification_derives_num_classes_from_data(tmp_path: Path):
    """A label range of 4 classes must derive its count from the CSV, never truncate to a
    narrower bound."""
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for label in range(4):
        _save_png(images_dir / f"img{label}.png", bright=(label >= 2))
        rows.append((f"img{label}", label))
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, rows, ("stem", "label"))

    _dataset, data = run_over("classification", str(images_dir), str(csv_path))
    assert data["num_classes"] == 4 and data.get("num_ranks") is None


def test_classification_num_classes_stated_raises(tmp_path: Path):
    """A caller-configured num_classes raises: the class count is the CSV's own."""
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for label in range(4):
        _save_png(images_dir / f"img{label}.png", bright=(label >= 2))
        rows.append((f"img{label}", label))
    csv_path = tmp_path / "labels.csv"
    _write_csv(csv_path, rows, ("stem", "label"))

    with pytest.raises(ValueError, match="num_classes"):
        dataset_over("classification", str(images_dir), str(csv_path), stated={"num_classes": 2})


def test_regression_e2e(tmp_path: Path):
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png", bright=(i % 2 == 0))
        rows.append((f"img{i}", float(i) / 6.0))
    csv_path = tmp_path / "values.csv"
    _write_csv(csv_path, rows, ("stem", "value"))

    dataset, data = run_over("regression", str(images_dir), str(csv_path))
    loader = DataLoader(dataset, batch_size=3, collate_fn=task_collate("regression"))

    model_source = _model_source(BESPOKE_REGRESSOR, "regression")
    run = _run(model_source, data, tmp_path, "auto-run-20")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")


def test_ordinal_evaluate_model_e2e(tmp_path: Path, monkeypatch):
    """evaluate_model must actually run for ordinal, not just build_dataset/train directly: it
    threads a CSV path into its own dataset build, since OrdinalDataset requires csv_path and
    ds_kwargs must not stay images_dir-only."""
    from tcip_mcp.tools.model_tools import register_model
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png", bright=(i % 3 == 0))
        rows.append((f"img{i}", i % 3))
    csv_path = tmp_path / "ranks.csv"
    _write_csv(csv_path, rows, ("stem", "rank"))

    dataset, data = run_over("ordinal", str(images_dir), str(csv_path))
    loader = DataLoader(dataset, batch_size=3, collate_fn=task_collate("ordinal"))
    model_source = _model_source(BESPOKE_ORDINAL, "ordinal")
    run = _run(model_source, data, tmp_path, "auto-run-21")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")

    ckpt_path = str(tmp_path / "out" / "model_best.pt")
    reg = register_model(name="ordinal-model", checkpoint_path=ckpt_path, config={},
                         project=tmp_path)
    assert "error" not in reg, reg

    result = evaluate_model(tmp_path, ckpt_path, str(images_dir), str(csv_path))
    assert "error" not in result, result
    assert "mae" in result
    assert "quadratic_weighted_kappa" in result


def test_regression_evaluate_model_e2e(tmp_path: Path, monkeypatch):
    """evaluate_model must actually run for regression, same fix as the ordinal case above."""
    from tcip_mcp.tools.model_tools import register_model
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir = tmp_path / "images" / UNDATED_BUCKET
    rows = []
    for i in range(6):
        _save_png(images_dir / f"img{i}.png", bright=(i % 2 == 0))
        rows.append((f"img{i}", float(i) / 6.0))
    csv_path = tmp_path / "values.csv"
    _write_csv(csv_path, rows, ("stem", "value"))

    dataset, data = run_over("regression", str(images_dir), str(csv_path))
    loader = DataLoader(dataset, batch_size=3, collate_fn=task_collate("regression"))
    model_source = _model_source(BESPOKE_REGRESSOR, "regression")
    run = _run(model_source, data, tmp_path, "auto-run-22")
    run = train(run, loader, val_loader=None)
    _assert_trained(run, tmp_path / "out")

    ckpt_path = str(tmp_path / "out" / "model_best.pt")
    reg = register_model(name="regression-model", checkpoint_path=ckpt_path, config={},
                         project=tmp_path)
    assert "error" not in reg, reg

    result = evaluate_model(tmp_path, ckpt_path, str(images_dir), str(csv_path))
    assert "error" not in result, result
    assert "mae" in result
    assert "r_squared" in result

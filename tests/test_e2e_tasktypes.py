"""Minimal end-to-end smoke tests for each task type via ``model_source`` builders.

Each training smoke runs one builder through build_model, build_dataset and train over a handful
of 64 by 64 synthetic images on CPU, and asserts the run completes one epoch with a finite loss
and a checkpoint; the derivation cases check the rank and class counts a CSV gives.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.pipelines.execution import Stated

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.experiments import RunObservation

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_annotation.state import Annotation, BBox, Polygon  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    BESPOKE_INSTANCE_SEG,
    BESPOKE_ORDINAL,
    BESPOKE_REGRESSOR,
    BESPOKE_SEMANTIC_SEG,
)
from tests._image_fixtures import write_noise_image  # noqa: E402
from tests._producer_fixtures import dataset_over, label_image, run_over  # noqa: E402
from tests._training_values import (  # noqa: E402
    IOU_THRESHOLD,
    adamw_optimizer,
    drawn_split,
)
from tests.tiny_trainer_fixtures import write_regression_dataset  # noqa: E402

IMG = 64


# --------------------------------------------------------------------------
# Synthetic-data helpers
# --------------------------------------------------------------------------

_save_png = partial(write_noise_image, size=IMG)


def _model_source(builder: str, task: str, **kwargs: JsonValue) -> dict:
    from tests._chain_fixtures import BESPOKE_MODELS, FRAME_STATISTICS

    return {
        "builder": builder,
        "builder_kwargs": kwargs,
        "source_files": [BESPOKE_MODELS],
        "task": task,
        "normalization": FRAME_STATISTICS,
    }


def _train_config(
    project: Path, model_source: dict, data: dict, batch_size: int
) -> dict:
    from tests._chain_fixtures import training_config

    # No val_loader at any call site below: loss is the only metric coherent to select on
    # without one.
    config = training_config(
        project,
        model_source,
        data,
        optimizer=adamw_optimizer(),
        batch_size=batch_size,
    )
    config["evaluation"]["selection_metric"] = "loss"
    return config


def _trained(
    model_source: dict, data: dict, tmp_path: Path, *, batch_size: int = 2
) -> RunObservation:
    """Train a run of ``tmp_path`` over :func:`_train_config` and return its observation.

    The run trains at ``batch_size``, opened by the launcher's own producer over the places
    ``data`` names with no validation side and trained to its end by the child's own entry.
    """
    from tcip_mcp.experiments import observe
    from tests._verified_checkpoint_fixtures import worker_run

    return observe(
        worker_run(
            tmp_path,
            _train_config(
                tmp_path,
                model_source,
                {**data, "split": drawn_split(val_ratio=0.0)},
                batch_size,
            ),
        )
    )


def _assert_trained(observation: RunObservation) -> None:
    from tcip_mcp.experiments import (
        EPOCH_KEY,
        METRICS_FILE,
        epoch_rows,
        read_rows,
    )

    assert observation.state == "completed", observation.final
    rows = epoch_rows(read_rows(observation.directory / METRICS_FILE)[0])
    assert [row[EPOCH_KEY] for row in rows] == [1]
    train_loss = rows[-1]["train_loss"]
    assert math.isfinite(train_loss), f"non-finite train_loss: {train_loss}"
    assert (observation.directory / "model_best.pt").is_file()


# --------------------------------------------------------------------------
# Box-style tasks (detection, instance_seg): torchvision detector path
# --------------------------------------------------------------------------


def test_detection_e2e(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # one centered box covering the middle of the image
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=BBox(19.2, 19.2, 44.8, 44.8))],
            IMG,
            IMG,
        )

    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    _assert_trained(
        _trained(
            BUILT_DETECTOR,
            {"images_dir": str(images_dir), "scope": {"subject": "bud"}},
            tmp_path,
        )
    )


def test_instance_seg_e2e(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    square = Polygon(
        [[(19.2, 19.2), (44.8, 19.2), (44.8, 44.8), (19.2, 44.8)]]
    )
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        # a square polygon (>= 3 vertices)
        label_image(
            images_dir / f"img{i}.png",
            [Annotation(subject="bud", geometry=square)],
            IMG,
            IMG,
        )

    dataset, _data = run_over("instance_seg", str(images_dir), subject="bud")
    # Guard the polygon -> mask rasterization path (datasets.py). With the mask_rcnn
    # detector these masks now reach the Mask R-CNN mask loss during training.
    assert dataset[0][1]["masks"].shape[0] > 0

    _assert_trained(
        _trained(
            _model_source(BESPOKE_INSTANCE_SEG, "instance_seg"),
            {"images_dir": str(images_dir), "scope": {"subject": "bud"}},
            tmp_path,
        )
    )


# --------------------------------------------------------------------------
# Dense / scalar tasks (semantic_seg, ordinal, regression): stack-collate path
# --------------------------------------------------------------------------


def test_semantic_seg_e2e(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / UNDATED_BUCKET
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    from tests._producer_fixtures import painted_frame

    block = (IMG // 4, IMG // 4, IMG // 2, IMG // 2)
    for i in range(4):
        _save_png(images_dir / f"img{i}.png")
        painted_frame(IMG, IMG, 0, [(block, 1)], mode="L").save(
            masks_dir / f"img{i}.png"
        )

    model_source = _model_source(BESPOKE_SEMANTIC_SEG, "semantic_seg")
    _assert_trained(
        _trained(
            model_source,
            {"images_dir": str(images_dir), "labels_dir": str(masks_dir)},
            tmp_path,
        )
    )


def test_ordinal_e2e(tmp_path: Path) -> None:
    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [0.8 if i % 3 == 0 else 0.2 for i in range(6)],
        [i % 3 for i in range(6)],
        height=IMG,
        width=IMG,
    )

    model_source = _model_source(BESPOKE_ORDINAL, "ordinal")
    _assert_trained(
        _trained(
            model_source,
            {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
            tmp_path,
            batch_size=3,
        )
    )


def test_ordinal_derives_num_ranks_from_data(tmp_path: Path) -> None:
    """An ordinal run derives its rank count from the data and trains.

    A rank scale of 7 ranks derives a rank count of 7 from the CSV and no class count, and the
    run trains to completion.
    """
    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [0.8 if rank >= 4 else 0.2 for rank in range(7)],
        list(range(7)),
        height=IMG,
        width=IMG,
    )

    _dataset, data = run_over("ordinal", str(images_dir), str(csv_path))
    assert data["num_ranks"] == 7 and data.get("num_classes") is None

    model_source = _model_source(BESPOKE_ORDINAL, "ordinal")
    _assert_trained(
        _trained(
            model_source,
            {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
            tmp_path,
            batch_size=7,
        )
    )


def test_ordinal_num_ranks_stated_raises(tmp_path: Path) -> None:
    """A caller-stated ``num_ranks`` is refused naming it.

    Over a CSV of seven ranks, a stated count of five is refused.
    """
    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [0.8 if rank >= 4 else 0.2 for rank in range(7)],
        list(range(7)),
        height=IMG,
        width=IMG,
    )

    with pytest.raises(ValueError, match="num_ranks"):
        dataset_over(
            "ordinal", str(images_dir), str(csv_path), stated={"num_ranks": 5}
        )


def test_classification_derives_num_classes_from_data(tmp_path: Path) -> None:
    """A label range of 4 classes derives a class count of 4 from the CSV and no rank count."""
    images_dir, csv_path = write_regression_dataset(
        tmp_path, [0.2, 0.2, 0.8, 0.8], [0, 1, 2, 3], height=IMG, width=IMG
    )

    _dataset, data = run_over("classification", str(images_dir), str(csv_path))
    assert data["num_classes"] == 4 and data.get("num_ranks") is None


def test_classification_num_classes_stated_raises(tmp_path: Path) -> None:
    """A caller-configured num_classes raises: the class count is the CSV's own."""
    images_dir, csv_path = write_regression_dataset(
        tmp_path, [0.2, 0.2, 0.8, 0.8], [0, 1, 2, 3], height=IMG, width=IMG
    )

    with pytest.raises(ValueError, match="num_classes"):
        dataset_over(
            "classification",
            str(images_dir),
            str(csv_path),
            stated={"num_classes": 2},
        )


def test_regression_e2e(tmp_path: Path) -> None:
    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [0.8 if i % 2 == 0 else 0.2 for i in range(6)],
        [float(i) / 6.0 for i in range(6)],
        height=IMG,
        width=IMG,
    )

    model_source = _model_source(BESPOKE_REGRESSOR, "regression")
    _assert_trained(
        _trained(
            model_source,
            {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
            tmp_path,
            batch_size=3,
        )
    )


def test_ordinal_evaluate_model_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_model runs end to end for ordinal over a CSV of ranks and reports its metrics."""
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [0.8 if i % 3 == 0 else 0.2 for i in range(6)],
        [i % 3 for i in range(6)],
        height=IMG,
        width=IMG,
    )

    model_source = _model_source(BESPOKE_ORDINAL, "ordinal")
    observation = _trained(
        model_source,
        {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
        tmp_path,
        batch_size=3,
    )
    _assert_trained(observation)

    result = evaluate_model(
        tmp_path,
        observation.directory.name,
        str(images_dir),
        str(csv_path),
        stated=Stated(),
        iou_threshold=IOU_THRESHOLD,
    )
    assert "error" not in result, result
    assert "mae" in result
    assert "quadratic_weighted_kappa" in result


def test_regression_evaluate_model_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """evaluate_model runs end to end for regression over a CSV of values."""
    from tcip_mcp.tools.training_tools import evaluate_model

    images_dir, csv_path = write_regression_dataset(
        tmp_path,
        [0.8 if i % 2 == 0 else 0.2 for i in range(6)],
        [float(i) / 6.0 for i in range(6)],
        height=IMG,
        width=IMG,
    )

    model_source = _model_source(BESPOKE_REGRESSOR, "regression")
    observation = _trained(
        model_source,
        {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
        tmp_path,
        batch_size=3,
    )
    _assert_trained(observation)

    result = evaluate_model(
        tmp_path,
        observation.directory.name,
        str(images_dir),
        str(csv_path),
        stated=Stated(),
        iou_threshold=IOU_THRESHOLD,
    )
    assert "error" not in result, result
    assert "mae" in result
    assert "r_squared" in result

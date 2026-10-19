"""The bespoke ``model_source`` pipeline as one connected system: build → train → infer.

Classification runs over synthetic data and detection over real sample data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests._chain_fixtures import BESPOKE_CLASSIFIER, FRAME_STATISTICS

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines.model_build import CONFIG_KEY, STATE_DICT_KEY  # noqa: E402

torchvision = pytest.importorskip("torchvision")

from tests import REPO_ROOT, bespoke_models  # noqa: E402
from tests._producer_fixtures import run_over  # noqa: E402
from tests._training_values import (  # noqa: E402
    RATIONALE,
    adamw_optimizer,
    drawn_split,
    evaluation_block,
    schedule,
    stage,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_classification_data(tmp_path: Path) -> tuple[str, str]:
    """Write a two-class classification dataset and return its images and table paths.

    Six dark frames of class 0 and six bright ones of class 1, plus their ground-truth table.
    """
    from tests.tiny_trainer_fixtures import write_regression_dataset

    images_dir, csv_path = write_regression_dataset(
        tmp_path / "cls",
        [0.05 * i for i in range(6)] + [0.7 + 0.05 * i for i in range(6)],
        [0] * 6 + [1] * 6,
        height=64,
        width=64,
    )
    return str(images_dir), str(csv_path)


# ---------------------------------------------------------------------------
# Test: full classification pipeline
# ---------------------------------------------------------------------------


class TestFullClassificationPipeline:
    """Tests of the bespoke classification pipeline from builder through checkpoint to predict."""

    def test_build_train_infer(
        self, tiny_classification_data: tuple[str, str], tmp_path: Path
    ) -> None:
        # --- Step 1: A bespoke classification model_source ---
        model_source = {
            "builder": BESPOKE_CLASSIFIER,
            "source_files": [bespoke_models.__file__],
            "task": "classification",
            "normalization": FRAME_STATISTICS,
        }

        # --- Step 2: Build the model and verify it runs ---
        from tests._chain_fixtures import band_statistics

        model = bespoke_models.build_bespoke_classifier(
            num_classes=2, **band_statistics()
        )
        dummy = torch.randn(2, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        assert isinstance(out, dict)
        # Should have head0 predictions
        pred_keys = [k for k in out if k.startswith("head0")]
        assert len(pred_keys) > 0

        # --- Step 3: Build dataset ---
        images_dir, csv_path = tiny_classification_data
        dataset, data = run_over("classification", images_dir, csv_path)
        assert data["num_classes"] == 2
        assert dataset.num_samples == 12

        # --- Step 4: A run the launcher's producer opens, trained 2 epochs by the child ---
        from tcip_mcp.experiments import (
            METRICS_FILE,
            epoch_rows,
            observe,
            read_rows,
        )
        from tests._chain_fixtures import training_config
        from tests._verified_checkpoint_fixtures import worker_run

        config = training_config(
            tmp_path,
            model_source,
            {
                "images_dir": images_dir,
                "labels_dir": csv_path,
                "split": drawn_split(val_ratio=0.34, grouping="tile_prefix"),
            },
            stages=[stage(-1)],
            batch_size=4,
            optimizer=adamw_optimizer(),
            scheduler=schedule("cosine", horizon_epochs=2),
            evaluation=evaluation_block(selection_metric="loss"),
        )
        observation = observe(worker_run(tmp_path, config))

        assert observation.state == "completed", observation.final
        out = observation.directory
        assert (out / "model_best.pt").is_file()
        assert (out / "model_final.pt").is_file()
        # Every epoch's row reached the log through the run's own sink, val_loss included.
        rows = epoch_rows(read_rows(out / METRICS_FILE)[0])
        assert [row["epoch"] for row in rows] == [1, 2]
        assert "train_loss" in rows[0] and "val_loss" in rows[-1]

        # Verify checkpoint has required keys
        ckpt = torch.load(
            out / "model_best.pt", map_location="cpu", weights_only=False
        )
        assert STATE_DICT_KEY in ckpt
        assert (
            "model_source" in ckpt[CONFIG_KEY] and "model_source" not in ckpt
        )

        # --- Step 5: Load the checkpoint its completion registered, and run inference ---
        from tcip_mcp.model_registry import load_registered_checkpoint
        from tcip_mcp.pipelines.execution import Stated, prepare

        assert observation.checkpoint is not None
        checkpoint = load_registered_checkpoint(
            observation.checkpoint.path, project=tmp_path
        )
        p = prepare(
            checkpoint, Stated(tile=False, rationale=RATIONALE), device="cpu"
        ).runnable()

        # Pick some test images
        test_images = sorted(Path(images_dir).rglob("*.png"))[:4]
        results = p.predict([str(path) for path in test_images])

        assert len(results) == 4
        for r in results:
            assert "image" in r


# ---------------------------------------------------------------------------
# Test: detection pipeline with real bud data
# ---------------------------------------------------------------------------

SAMPLE_PROJECT = Path(
    os.environ.get("TCIP_SAMPLE_PROJECT", str(REPO_ROOT / "data"))
)
"""A real dataset to run the detection pipeline against: TCIP_SAMPLE_PROJECT names a project
root holding subjects.json, images/<date>/ and their label documents; defaults to an in-repo
<repo>/data sample. The tests below skip when neither is present."""


def _sample_date() -> str | None:
    """Return a capture date under SAMPLE_PROJECT holding images and bud annotations, or None."""
    if not (SAMPLE_PROJECT / "subjects.json").is_file():
        return None
    from tcip_annotation import json_io
    from tcip_mcp.dataset_layout import capture_label_keys, list_dates

    for date in list_dates(SAMPLE_PROJECT):
        for key in capture_label_keys(SAMPLE_PROJECT, date):
            if any(
                a.subject == "bud" and a.geometry is not None
                for a in json_io.read_label_document(key).annotations
            ):
                return date
    return None


@pytest.mark.skipif(
    _sample_date() is None,
    reason="No nested-schema sample project (set TCIP_SAMPLE_PROJECT to a converted dataset)",
)
class TestDetectionPipelineRealData:
    """Tests of the detection pipeline, build to train to infer, over real bud images."""

    def test_build_train_infer(self, tmp_path: Path) -> None:
        from tests._chain_fixtures import detector_values
        from tests._verified_checkpoint_fixtures import built_detector

        # --- Step 1: A bespoke detection model_source at the real images' larger input sizes ---
        model_source = built_detector()
        model = bespoke_models.build_bespoke_detection(
            num_classes=1, **detector_values((512, 320))
        )
        assert isinstance(model, bespoke_models.BespokeDetection)

        # --- Step 2: Build dataset from the nested-schema labels (name-based, one file per
        # image) ---
        date = _sample_date()
        assert date is not None
        images_dir = SAMPLE_PROJECT / "images" / date
        dataset, data = run_over("detection", str(images_dir), subject="bud")
        # num_classes is the one subject the scope isolates (bud here), and num_samples comes
        # from the bud-annotated images on this date.
        assert dataset.num_classes == 1
        assert dataset.num_samples > 0

        # Verify samples load with the right structure. Sample 0 may be a confirmed negative (zero
        # boxes is a valid training sample), so check the shape here and that bud boxes exist
        # somewhere in the set rather than assuming the first sample is annotated.
        img, target = dataset[0]
        assert img.ndim == 3 and img.shape[0] == 3  # [C, H, W]
        assert (
            target["boxes"].ndim == 2 and target["boxes"].shape[1] == 4
        )  # [N, 4], N may be 0
        assert target["labels"].ndim == 1
        assert any(
            len(dataset[i][1]["labels"]) > 0
            for i in range(min(dataset.num_samples, 8))
        )

        # --- Step 3: A run the launcher's producer opens over the date, trained 1 epoch ---
        from tcip_mcp.experiments import observe
        from tests._chain_fixtures import training_config
        from tests._verified_checkpoint_fixtures import worker_run

        config = training_config(
            tmp_path,
            model_source,
            {
                **data,
                "images_dir": str(images_dir),
                "split": drawn_split(val_ratio=0.0),
            },
            stages=[stage(0)],
            evaluation=evaluation_block(selection_metric="loss"),
        )
        observation = observe(worker_run(tmp_path, config))

        assert observation.state == "completed", observation.final
        out = observation.directory
        assert (out / "model_best.pt").is_file()

        # Verify checkpoint format
        ckpt = torch.load(
            out / "model_best.pt", map_location="cpu", weights_only=False
        )
        assert STATE_DICT_KEY in ckpt
        assert (
            "model_source" in ckpt[CONFIG_KEY] and "model_source" not in ckpt
        )

        # --- Step 4: Load the checkpoint its completion registered, and run inference ---
        from tcip_mcp.model_registry import load_registered_checkpoint
        from tcip_mcp.pipelines.execution import Stated, prepare

        assert observation.checkpoint is not None
        checkpoint = load_registered_checkpoint(
            observation.checkpoint.path, project=tmp_path
        )
        detector = prepare(
            checkpoint,
            Stated(tile=False, conf=0.01, rationale=RATIONALE),
            device="cpu",
        ).runnable()

        img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        test_images = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in img_exts
        )[:3]
        assert test_images, "no images on the sample date"
        results = detector.predict([str(p) for p in test_images])

        assert len(results) == len(test_images)
        for r in results:
            assert "image" in r
            assert "boxes" in r
            assert "scores" in r
            assert "count" in r

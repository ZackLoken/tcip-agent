"""Characterization tests pinning current orchestrator behavior for refactor safety.

Complements tests/test_orchestrator.py (risk-zone regressions) by pinning the
remaining branches of pipelines/orchestrator.py exactly as they behave today:
validate_pipeline issue strings, phase-type inference, per-runner error and
context-inheritance branches, the aggregation/export success paths, and
checkpoint edge cases. No production code is changed by this file.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest
from tcip_annotation import json_io
from tcip_annotation.state import PredBBox

from tcip_mcp.pipelines.orchestrator import (
    _PHASE_RUNNERS,
    PhaseResult,
    PipelineOrchestrator,
    _infer_phase_type,
    register_phase_runner,
    validate_pipeline,
)

MODEL_SPEC = {
    "backbone": "resnet18",
    "heads": [{"name": "detection", "task": "detection", "num_classes": 2}],
}


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture
def det_data(tmp_path: Path) -> dict:
    """Tiny YOLO detection dataset (4 images, 1 box each)."""
    from PIL import Image

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    stems = [f"img_{i}" for i in range(4)]
    for stem in stems:
        Image.new("RGB", (32, 32), color=(100, 100, 100)).save(images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return {"images_dir": str(images_dir), "labels_dir": str(labels_dir), "stems": stems}


class FakeRun:
    def __init__(self, run_id: str, output_dir: str, status: str = "completed",
                 error: str = "") -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.status = status
        self.metrics_history = [{"train_loss": 0.25}]
        self.error = error


@pytest.fixture
def stub_trainer(monkeypatch, tmp_path: Path) -> dict:
    """Stub create_run/train so training phases run without real training.

    ``captured["run_status"]`` / ``captured["run_error"]`` control the outcome
    the fake run reports. Keeps .tcip writes inside tmp_path."""
    import tcip_mcp.pipelines.training.generic_trainer as gt

    captured: dict = {"run_status": "completed", "run_error": ""}

    def fake_create_run(config, output_dir, origin="training"):
        captured["config"] = config
        run = FakeRun("run_char_test", output_dir,
                      status=captured["run_status"], error=captured["run_error"])
        captured["run"] = run
        return run

    def fake_train(run, train_loader, val_loader=None, task="detection",
                   epoch_callback=None, resume_from=""):
        captured["train_len"] = len(train_loader.dataset)
        out = Path(run.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_best.pt").write_bytes(b"fake checkpoint")
        if epoch_callback is not None:
            epoch_callback(0, {"train_loss": 0.25})
        return run

    monkeypatch.setattr(gt, "create_run", fake_create_run)
    monkeypatch.setattr(gt, "train", fake_train)
    monkeypatch.chdir(tmp_path)
    return captured


@pytest.fixture
def fake_predictor(monkeypatch) -> dict:
    """Replace GenericPredictor with a stub returning a fixed 2-box result."""
    import tcip_mcp.pipelines.inference.generic_predictor as gp

    captured: dict = {"predicted": []}

    class FakePredictor:
        def __init__(self, checkpoint_path=None, **kwargs):
            captured["checkpoint"] = checkpoint_path

        def predict(self, image_path):
            captured["predicted"].append(image_path)
            return {
                "image": image_path,
                "width": 100, "height": 50,
                "boxes": [[10.0, 10.0, 30.0, 20.0], [50.0, 25.0, 90.0, 45.0]],
                "labels": [1, 2],
                "scores": [0.9, 0.75],
                "count": 2,
            }

    monkeypatch.setattr(gp, "GenericPredictor", FakePredictor)
    return captured


def _stub_runner(status: str = "completed", artifacts: dict | None = None):
    def runner(phase, context, work_dir):
        return PhaseResult(phase_name=phase["name"], status=status,
                           artifacts=dict(artifacts or {}))
    return runner


# ====================================================================
# validate_pipeline: exact issue strings
# ====================================================================

class TestValidatePipeline:
    def test_missing_name(self):
        issues = validate_pipeline({"phases": [{"name": "a", "task": "detection"}]})
        assert issues == ["Pipeline spec missing 'name'"]

    def test_no_phases_short_circuits(self):
        # Both issues for {} — but the empty-phases check returns immediately,
        # so no per-phase issues are ever appended.
        assert validate_pipeline({}) == [
            "Pipeline spec missing 'name'",
            "Pipeline has no phases",
        ]

    def test_phase_missing_name_and_task(self):
        issues = validate_pipeline({"name": "p", "phases": [{}]})
        assert issues == [
            "Phase 0 (?): missing 'name'",
            "Phase 0 (?): missing 'task'",
        ]

    def test_dangling_input_reference(self):
        issues = validate_pipeline({
            "name": "p",
            "phases": [{"name": "a", "task": "detection", "input": "nowhere"}],
        })
        assert issues == ["Phase 0 (a): input 'nowhere' not produced by any prior phase"]

    def test_input_must_come_from_prior_phase_not_later(self):
        # Outputs are collected in order — a forward reference is invalid.
        issues = validate_pipeline({
            "name": "p",
            "phases": [
                {"name": "a", "task": "detection", "input": "b_out"},
                {"name": "b", "task": "detection", "output": "b_out"},
            ],
        })
        assert issues == ["Phase 0 (a): input 'b_out' not produced by any prior phase"]

    def test_duplicate_output_name(self):
        issues = validate_pipeline({
            "name": "p",
            "phases": [
                {"name": "a", "task": "detection", "output": "dets"},
                {"name": "b", "task": "detection", "output": "dets"},
            ],
        })
        assert issues == ["Phase 1 (b): duplicate output name 'dets'"]

    def test_valid_multi_phase_spec_has_no_issues(self):
        assert validate_pipeline({
            "name": "p",
            "phases": [
                {"name": "a", "task": "detection", "output": "dets"},
                {"name": "b", "task": "aggregation", "input": "dets"},
            ],
        }) == []

    def test_dict_model_spec_is_validated_and_deduplicated(self):
        issues = validate_pipeline({
            "name": "p",
            "phases": [{
                "name": "train",
                "task": "detection",
                "model_spec": {"backbone": {"name": "not_a_backbone"}},
            }],
        })
        assert any("Unknown backbone: not_a_backbone" in i for i in issues)
        assert any("heads" in i for i in issues)
        # All model_spec issues carry the phase prefix and are deduplicated.
        assert all(i.startswith("Phase 0 (train): ") for i in issues)
        assert len(issues) == len(set(issues))

    def test_string_model_spec_skips_composer_validation(self):
        # Only dict model_specs go through validate_model_spec.
        assert validate_pipeline({
            "name": "p",
            "phases": [{"name": "t", "task": "detection", "model_spec": "some_registered_name"}],
        }) == []


# ====================================================================
# _infer_phase_type
# ====================================================================

class TestInferPhaseType:
    def test_explicit_type_always_wins(self):
        assert _infer_phase_type(
            {"type": "custom_x", "task": "aggregation", "model_spec": {}}) == "custom_x"

    def test_task_aggregation_and_export(self):
        assert _infer_phase_type({"task": "aggregation"}) == "aggregation"
        assert _infer_phase_type({"task": "export"}) == "export"

    def test_model_spec_means_training(self):
        assert _infer_phase_type({"task": "detection", "model_spec": {}}) == "training"

    def test_checkpoint_without_model_spec_means_inference(self):
        assert _infer_phase_type({"task": "detection", "checkpoint": "x.pt"}) == "inference"

    def test_bare_phase_defaults_to_training(self):
        assert _infer_phase_type({}) == "training"


# ====================================================================
# register_phase_runner
# ====================================================================

class TestRegisterPhaseRunner:
    def test_registered_runner_is_used_by_run_phase(self, tmp_path):
        register_phase_runner("char_test_custom", _stub_runner(artifacts={"k": "v"}))
        try:
            result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
                {"name": "x", "task": "anything", "type": "char_test_custom"})
        finally:
            del _PHASE_RUNNERS["char_test_custom"]
        assert result.status == "completed"
        assert result.artifacts == {"k": "v"}


# ====================================================================
# Training phase: error and context-inheritance branches
# ====================================================================

class TestTrainingPhaseBranches:
    def test_missing_model_spec_fails_before_timing(self, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "t", "task": "detection"})
        assert result.status == "failed"
        assert result.error == "Training phase needs 'model_spec'"
        # Early return happens before the elapsed clock is stamped.
        assert result.elapsed_seconds == 0.0

    def test_dataset_dirs_inherited_from_input_context(self, det_data, stub_trainer, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase(
            {"name": "t", "task": "detection", "model_spec": MODEL_SPEC,
             "input": "crops", "dataset": {}},
            context={"crops": {"images_dir": det_data["images_dir"],
                               "labels_dir": det_data["labels_dir"]}},
        )
        assert result.status == "completed", result.error
        assert stub_trainer["config"]["data"]["images_dir"] == det_data["images_dir"]
        assert stub_trainer["train_len"] == 4

    def test_explicit_dataset_dirs_win_over_context(self, det_data, stub_trainer, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase(
            {"name": "t", "task": "detection", "model_spec": MODEL_SPEC,
             "input": "crops",
             "dataset": {"images_dir": det_data["images_dir"],
                         "labels_dir": det_data["labels_dir"]}},
            context={"crops": {"images_dir": str(tmp_path / "does_not_exist"),
                               "labels_dir": str(tmp_path / "does_not_exist")}},
        )
        assert result.status == "completed", result.error
        assert stub_trainer["config"]["data"]["images_dir"] == det_data["images_dir"]

    def test_exception_in_phase_body_fails_and_logs(self, det_data, stub_trainer, tmp_path,
                                                    monkeypatch, caplog):
        import tcip_mcp.pipelines.training.generic_trainer as gt

        def boom(config, output_dir, origin="training"):
            raise RuntimeError("trainer setup exploded")

        monkeypatch.setattr(gt, "create_run", boom)
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        with caplog.at_level(logging.ERROR, logger="tcip_mcp.pipelines.orchestrator"):
            result = orch.run_phase({
                "name": "t", "task": "detection", "model_spec": MODEL_SPEC,
                "dataset": {"images_dir": det_data["images_dir"],
                            "labels_dir": det_data["labels_dir"]},
            })
        assert result.status == "failed"
        assert result.error == "trainer setup exploded"
        assert "Training phase 't' failed" in caplog.text
        assert result.elapsed_seconds > 0.0

    def test_non_completed_run_fails_phase_with_run_error(self, det_data, stub_trainer, tmp_path):
        stub_trainer["run_status"] = "failed"
        stub_trainer["run_error"] = "training diverged"
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "t", "task": "detection", "model_spec": MODEL_SPEC,
            "dataset": {"images_dir": det_data["images_dir"],
                        "labels_dir": det_data["labels_dir"]},
        })
        assert result.status == "failed"
        assert result.error == "training diverged"
        # Artifacts are still populated (checkpoint path + run_id) even on failure.
        assert result.artifacts["run_id"] == "run_char_test"
        # Experiment state mirrors the run status; no model is registered.
        status = json.loads(
            (tmp_path / ".tcip" / "experiments" / "run_char_test" / "status.json").read_text())
        assert status["state"] == "failed"
        from tcip_mcp.model_registry import ModelRegistry
        assert not any(e["name"] == "run_char_test" for e in ModelRegistry(".").list_models())

    def test_metric_log_failure_does_not_fail_phase(self, det_data, stub_trainer, tmp_path,
                                                    monkeypatch):
        from tcip_mcp import experiments

        def boom(*args, **kwargs):
            raise OSError("metrics disk full")

        monkeypatch.setattr(experiments, "log_metrics", boom)
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "t", "task": "detection", "model_spec": MODEL_SPEC,
            "dataset": {"images_dir": det_data["images_dir"],
                        "labels_dir": det_data["labels_dir"]},
        })
        assert result.status == "completed", result.error

    def test_completion_wiring_failure_does_not_fail_phase(self, det_data, stub_trainer,
                                                           tmp_path, monkeypatch):
        from tcip_mcp import experiments

        def boom(*args, **kwargs):
            raise OSError("registry locked")

        monkeypatch.setattr(experiments, "register_model_from_experiment", boom)
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "t", "task": "detection", "model_spec": MODEL_SPEC,
            "dataset": {"images_dir": det_data["images_dir"],
                        "labels_dir": det_data["labels_dir"]},
        })
        assert result.status == "completed", result.error


# ====================================================================
# Inference phase: checkpoint/images_dir resolution and filtering
# ====================================================================

class TestInferencePhaseBranches:
    def test_checkpoint_and_images_dir_from_input_context(self, fake_predictor, tmp_path):
        from PIL import Image

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        Image.new("RGB", (100, 50)).save(images_dir / "plot.jpg")

        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase(
            {"name": "infer", "task": "detection", "type": "inference", "input": "train_out"},
            context={"train_out": {"checkpoint": "ctx.pt", "images_dir": str(images_dir)}},
        )
        assert result.status == "completed", result.error
        assert fake_predictor["checkpoint"] == "ctx.pt"
        assert result.artifacts["images_dir"] == str(images_dir)
        assert result.artifacts["count"] == 1
        assert result.metrics == {"num_images": 1}

    def test_missing_checkpoint_fails(self, fake_predictor, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "infer", "task": "detection", "type": "inference"})
        assert result.status == "failed"
        assert result.error == "Inference phase needs a checkpoint (from prior training or explicit)"

    def test_missing_images_dir_fails(self, fake_predictor, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "infer", "task": "detection", "checkpoint": "x.pt"})
        assert result.status == "failed"
        assert result.error == "Inference phase needs images_dir"

    def test_only_image_suffixes_are_predicted(self, fake_predictor, tmp_path):
        from PIL import Image

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        Image.new("RGB", (100, 50)).save(images_dir / "a.jpg")
        Image.new("RGB", (100, 50)).save(images_dir / "b.png")
        (images_dir / "notes.txt").write_text("not an image")
        Image.new("RGB", (100, 50)).save(images_dir / "c.bmp")  # unsupported suffix

        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({"name": "infer", "task": "detection",
                                 "checkpoint": "x.pt", "images_dir": str(images_dir)})
        assert result.status == "completed", result.error
        assert result.artifacts["count"] == 2
        written = sorted(p.name for p in Path(result.artifacts["predictions_dir"]).glob("*.json"))
        assert written == ["a.json", "b.json"]


# ====================================================================
# Cropping phase: line filtering, clamping, and missing-image skip
# ====================================================================

class TestCroppingPhaseBranches:
    def test_missing_input_reference_fails(self, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "crop", "task": "detection", "type": "cropping"})
        assert result.status == "failed"
        assert result.error == "Cropping phase needs an input reference to a detection/seg phase"

    def test_prediction_without_matching_image_is_skipped(self, tmp_path):
        preds_dir = tmp_path / "preds"
        images_dir = tmp_path / "images"
        preds_dir.mkdir()
        images_dir.mkdir()
        json_io.write_detect(preds_dir / "ghost.json",
                             [PredBBox(10, 10, 30, 30, 0, confidence=0.9)], 100, 100)

        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "crop", "task": "detection", "type": "cropping", "input": "dets"},
            context={"dets": {"predictions_dir": str(preds_dir),
                              "images_dir": str(images_dir)}},
        )
        assert result.status == "completed", result.error
        assert result.artifacts["count"] == 0

    def test_object_filtering_clamping_and_crop_naming(self, tmp_path):
        from PIL import Image

        preds_dir = tmp_path / "preds"
        images_dir = tmp_path / "images"
        preds_dir.mkdir()
        images_dir.mkdir()
        Image.new("RGB", (100, 50), color=(100, 100, 100)).save(images_dir / "plot.jpg")
        # Pixel-xyxy PredBBoxes (COCO/JSON is pixel-space); the crop index is the object
        # index in the file, so filtered objects leave gaps in the crop names.
        json_io.write_detect(preds_dir / "plot.json", [
            PredBBox(49.5, 12.5, 50.5, 37.5, 0, confidence=0.9),   # 1px wide → skipped (< 2px)
            PredBBox(49.5, 12.5, 50.5, 37.5, 0, confidence=0.9),   # 1px wide → skipped (< 2px)
            PredBBox(80.0, 15.0, 110.0, 35.0, 0, confidence=0.9),  # overflows right edge → clamped to 20x20
            PredBBox(40.0, 20.0, 60.0, 30.0, 0, confidence=0.9),   # normal → 20x10
        ], 100, 50)

        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "crop", "task": "detection", "type": "cropping", "input": "dets"},
            context={"dets": {"predictions_dir": str(preds_dir),
                              "images_dir": str(images_dir)}},
        )
        assert result.status == "completed", result.error
        assert result.artifacts["count"] == 2
        crops = {p.name: Image.open(p).size
                 for p in Path(result.artifacts["images_dir"]).glob("*.jpg")}
        # Crop index is the source object index, so skipped objects leave gaps.
        assert crops == {"plot_crop2.jpg": (20, 20), "plot_crop3.jpg": (20, 10)}


# ====================================================================
# Aggregation phase: success paths
# ====================================================================

class TestAggregationPhaseBranches:
    def test_counts_from_predictions_dir_grouped_per_plant(self, tmp_path):
        preds_dir = tmp_path / "preds"
        preds_dir.mkdir()
        # Stems group by _extract_plant_id (strips last two underscore tokens).
        def _box() -> PredBBox:
            return PredBBox(10, 10, 30, 30, 0, confidence=0.9)
        json_io.write_detect(preds_dir / "plant1_a_1.json", [_box(), _box()], 100, 100)
        json_io.write_detect(preds_dir / "plant1_a_2.json", [_box(), _box(), _box(), _box()], 100, 100)
        # Present {"objects": []} confirmed negative → count 0, still counted as an image.
        json_io.write_detect(preds_dir / "plant2_b_1.json", [], 100, 100, keep_empty=True)

        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "agg", "task": "aggregation", "input": "dets",
             "trait_name": "catkin_count", "crop": "hazelnut"},
            context={"dets": {"predictions_dir": str(preds_dir)}},
        )
        assert result.status == "completed", result.error
        assert result.artifacts["strategy"] == "count"
        assert result.artifacts["n_plants"] == 2
        assert result.metrics == {"strategy": "count", "n_plants": 2, "n_images": 3}

        with open(result.artifacts["csv_path"], newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["plant_id"] for r in rows] == ["plant1", "plant2"]
        # Default 'count' strategy takes the median count per plant.
        assert [float(r["value"]) for r in rows] == [3.0, 0.0]
        assert rows[0]["crop"] == "hazelnut"
        assert rows[0]["trait_name"] == "catkin_count"
        assert [int(r["n_images"]) for r in rows] == [2, 1]

    def test_precomputed_results_override_predictions_dir(self, tmp_path):
        preds_dir = tmp_path / "preds"
        preds_dir.mkdir()
        json_io.write_detect(preds_dir / "ignored_a_1.json",
                             [PredBBox(10, 10, 30, 30, 0, confidence=0.9)], 100, 100)

        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "agg", "task": "aggregation", "input": "dets"},
            context={"dets": {
                "predictions_dir": str(preds_dir),
                "results": [{"image": "p9_x_1", "count": 7}],
            }},
        )
        assert result.status == "completed", result.error
        assert result.metrics["n_images"] == 1
        with open(result.artifacts["csv_path"], newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["plant_id"] for r in rows] == ["p9"]
        assert float(rows[0]["value"]) == 7.0

    def test_no_input_produces_empty_csv_and_completes(self, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "agg", "task": "aggregation"})
        assert result.status == "completed", result.error
        assert result.artifacts["n_plants"] == 0
        assert result.metrics["n_images"] == 0
        with open(result.artifacts["csv_path"], newline="") as f:
            reader = csv.reader(f)
            assert next(reader) == ["plant_id", "crop", "trait_name", "value",
                                    "confidence", "n_images", "pipeline_version"]
            assert list(reader) == []


# ====================================================================
# Export phase: input resolution branches
# ====================================================================

class TestExportPhaseBranches:
    def test_missing_input_reference_fails(self, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "exp", "task": "export"})
        assert result.status == "failed"
        assert result.error == "Export phase needs an input reference"

    def test_incompatible_input_artifacts_fail(self, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "exp", "task": "export", "input": "prev"},
            context={"prev": {"images_dir": str(tmp_path)}},
        )
        assert result.status == "failed"
        assert result.error == "Export phase: no compatible input artifacts"

    def test_csv_path_input_is_copied(self, tmp_path):
        src_csv = tmp_path / "agg.csv"
        src_csv.write_text("plant_id,value\np1,3\n")
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "exp", "task": "export", "input": "prev"},
            context={"prev": {"csv_path": str(src_csv)}},
        )
        assert result.status == "completed", result.error
        out = Path(result.artifacts["csv_path"])
        assert out.name == "results.csv"
        assert out.read_text() == src_csv.read_text()

    def test_missing_score_reads_as_zero_but_object_still_counted(self, tmp_path):
        preds_dir = tmp_path / "preds"
        preds_dir.mkdir()
        # A per-image COCO/JSON prediction file. An object with a null/absent score is
        # tolerated (read as 0.0) and still counts as a detection — so the count reflects
        # all three objects while the average is dragged down by the two zero scores.
        (preds_dir / "img.json").write_text(json.dumps({
            "image": "img", "width": 100, "height": 50,
            "objects": [
                {"category_id": 0, "bbox": [40, 20, 20, 10], "score": 0.5},   # good score
                {"category_id": 0, "bbox": [40, 20, 20, 10], "score": None},  # null → 0.0
                {"category_id": 0, "bbox": [40, 20, 20, 10]},                 # absent → 0.0
            ],
        }))
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "exp", "task": "export", "input": "prev"},
            context={"prev": {"predictions_dir": str(preds_dir)}},
        )
        assert result.status == "completed", result.error
        with open(result.artifacts["csv_path"], newline="") as f:
            rows = list(csv.DictReader(f))
        # avg_confidence = (0.5 + 0.0 + 0.0) / 3 = 0.1667.
        assert rows == [{"image": "img", "detection_count": "3", "avg_confidence": "0.1667"}]


# ====================================================================
# run_pipeline: unknown runner + checkpoint without the resume point
# ====================================================================

class TestRunPipelineBranches:
    def test_unknown_phase_type_fails_pipeline_with_known_types(self, tmp_path):
        pr = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_pipeline({
            "name": "unknown_type",
            "phases": [{"name": "a", "task": "custom", "type": "no_such_runner"}],
        })
        assert pr.status == "failed"
        assert pr.phases[0].status == "failed"
        assert "No runner for phase type 'no_such_runner'" in pr.phases[0].error
        assert str(sorted(_PHASE_RUNNERS.keys())) in pr.phases[0].error

    def test_checkpoint_without_resume_point_is_not_resumable(self, tmp_path, monkeypatch):
        # A run that fails at its first phase writes a checkpoint with zero
        # completed phases — resuming from that phase must fail loudly.
        monkeypatch.setitem(_PHASE_RUNNERS, "char_boom",
                            _stub_runner(status="failed"))
        spec = {
            "name": "early_fail",
            "phases": [{"name": "a", "task": "custom", "type": "char_boom"}],
        }
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        first = orch.run_pipeline(spec)
        assert first.status == "failed"
        assert list((tmp_path / "runs").glob("early_fail_*/checkpoint.json"))

        second = orch.run_pipeline(spec, resume_from="a")
        assert second.status == "failed"
        assert second.phases[0].phase_name == "resume"
        assert "Resume point 'a' not found" in second.phases[0].error

    def test_result_json_pins_full_shape(self, tmp_path, monkeypatch):
        monkeypatch.setitem(_PHASE_RUNNERS, "char_ok",
                            _stub_runner(artifacts={"k": "v"}))
        pr = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_pipeline({
            "name": "shape",
            "phases": [{"name": "a", "task": "custom", "type": "char_ok", "output": "o"}],
        })
        assert pr.status == "completed"
        result_files = list((tmp_path / "runs").glob("shape_*/pipeline_result.json"))
        assert len(result_files) == 1
        data = json.loads(result_files[0].read_text())
        assert data["pipeline_name"] == "shape"
        assert data["status"] == "completed"
        assert data["elapsed_seconds"] >= 0.0
        assert data["phases"] == [{
            "name": "a", "status": "completed", "artifacts": {"k": "v"},
            "metrics": {}, "error": "", "elapsed_seconds": 0.0,
        }]

"""Characterization + regression tests for the pipeline orchestrator.

Covers the risk-zone behaviors of pipelines/orchestrator.py: train/val stem
splits in the training phase (regression: split keys must not reach dataset
constructors), experiment-tracking wiring, artifact passing between phases,
checkpoint/resume (including resume-point-not-found), transient-vs-permanent
retry, failure logging, and the single-phase path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from tcip_mcp.pipelines.orchestrator import (
    _PHASE_RUNNERS,
    PhaseResult,
    PipelineOrchestrator,
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
    def __init__(self, run_id: str, output_dir: str) -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.status = "completed"
        self.metrics_history = [{"train_loss": 0.25}]
        self.error = ""


@pytest.fixture
def stub_trainer(monkeypatch, tmp_path: Path) -> dict:
    """Stub create_run/train so training phases exercise dataset building and
    tracking wiring without real training. Keeps .tcip writes inside tmp_path."""
    import tcip_mcp.pipelines.training.generic_trainer as gt

    captured: dict = {}

    def fake_create_run(config, output_dir, origin="training"):
        captured["config"] = config
        run = FakeRun("run_orch_test", output_dir)
        captured["run"] = run
        return run

    def fake_train(run, train_loader, val_loader=None, task="detection",
                   epoch_callback=None, resume_from=""):
        captured["train_len"] = len(train_loader.dataset)
        captured["val_len"] = len(val_loader.dataset) if val_loader is not None else None
        captured["task"] = task
        out = Path(run.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model_best.pt").write_bytes(b"fake checkpoint")
        if epoch_callback is not None:
            epoch_callback(0, {"train_loss": 0.25})
        return run

    monkeypatch.setattr(gt, "create_run", fake_create_run)
    monkeypatch.setattr(gt, "train", fake_train)
    # Experiment tracking + model registry write to cwd's .tcip — sandbox them.
    monkeypatch.chdir(tmp_path)
    return captured


def _stub_runner(record: list | None = None, status: str = "completed",
                 artifacts: dict | None = None, error: str = ""):
    """Build a phase runner that records its invocations."""
    def runner(phase, context, work_dir):
        if record is not None:
            record.append({"name": phase["name"], "context": dict(context)})
        return PhaseResult(
            phase_name=phase["name"], status=status,
            artifacts=dict(artifacts or {}), error=error,
        )
    return runner


# ====================================================================
# Training phase: train_stems/val_stems split (regression)
# ====================================================================

class TestTrainingPhaseSplitStems:
    def test_split_keys_do_not_reach_dataset_ctor(self, det_data, stub_trainer, tmp_path):
        """train_stems/val_stems must be stripped before spreading into build_dataset —
        previously raised TypeError: unexpected keyword argument 'train_stems'."""
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "train_det",
            "task": "detection",
            "model_spec": MODEL_SPEC,
            "dataset": {
                "images_dir": det_data["images_dir"],
                "labels_dir": det_data["labels_dir"],
                "train_stems": det_data["stems"][:3],
                "val_stems": det_data["stems"][3:],
            },
        })
        assert result.status == "completed", result.error
        assert stub_trainer["train_len"] == 3
        assert stub_trainer["val_len"] == 1

    def test_no_split_trains_on_all_stems_without_val(self, det_data, stub_trainer, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "train_det",
            "task": "detection",
            "model_spec": MODEL_SPEC,
            "dataset": {
                "images_dir": det_data["images_dir"],
                "labels_dir": det_data["labels_dir"],
            },
        })
        assert result.status == "completed", result.error
        assert stub_trainer["train_len"] == 4
        assert stub_trainer["val_len"] is None


# ====================================================================
# Training phase: experiment tracking wiring
# ====================================================================

class TestTrainingPhaseTracking:
    def test_run_config_carries_seed_data_and_batch_size(self, det_data, stub_trainer, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "train_det",
            "task": "detection",
            "model_spec": MODEL_SPEC,
            "seed": 1234,
            "deterministic": True,
            "batch_size": 2,
            "dataset": {
                "images_dir": det_data["images_dir"],
                "labels_dir": det_data["labels_dir"],
            },
        })
        assert result.status == "completed", result.error
        cfg = stub_trainer["config"]
        assert cfg["seed"] == 1234
        assert cfg["deterministic"] is True
        assert cfg["batch_size"] == 2
        assert cfg["data"]["images_dir"] == det_data["images_dir"]

    def test_seed_read_from_training_section(self, det_data, stub_trainer, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "train_det",
            "task": "detection",
            "model_spec": MODEL_SPEC,
            "training": {"seed": 7},
            "dataset": {
                "images_dir": det_data["images_dir"],
                "labels_dir": det_data["labels_dir"],
            },
        })
        assert result.status == "completed", result.error
        assert stub_trainer["config"]["seed"] == 7

    def test_experiment_created_metrics_logged_model_registered(self, det_data, stub_trainer, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "train_det",
            "task": "detection",
            "model_spec": MODEL_SPEC,
            "dataset": {
                "images_dir": det_data["images_dir"],
                "labels_dir": det_data["labels_dir"],
            },
        })
        assert result.status == "completed", result.error
        assert result.artifacts["run_id"] == "run_orch_test"

        exp_dir = tmp_path / ".tcip" / "experiments" / "run_orch_test"
        status = json.loads((exp_dir / "status.json").read_text())
        assert status["state"] == "completed"

        metrics_lines = (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert json.loads(metrics_lines[0])["train_loss"] == 0.25

        lineage = json.loads((exp_dir / "lineage.json").read_text())
        assert lineage["data_source"] == det_data["images_dir"]
        assert lineage["model_weights"].endswith("model_best.pt")

        from tcip_mcp.model_registry import ModelRegistry
        entries = ModelRegistry(".").list_models()
        assert any(e["name"] == "run_orch_test" for e in entries)

    def test_tracking_failure_does_not_fail_phase(self, det_data, stub_trainer, tmp_path, monkeypatch):
        from tcip_mcp import experiments

        def boom(*args, **kwargs):
            raise OSError("no .tcip here")

        monkeypatch.setattr(experiments, "create_experiment", boom)
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase({
            "name": "train_det",
            "task": "detection",
            "model_spec": MODEL_SPEC,
            "dataset": {
                "images_dir": det_data["images_dir"],
                "labels_dir": det_data["labels_dir"],
            },
        })
        assert result.status == "completed", result.error


# ====================================================================
# Inference phase: canonical prediction format (regression)
# ====================================================================

class TestInferencePredictionFormat:
    """The inference phase must write the platform's single canonical
    prediction format — ``cls conf cx cy w h`` (result_to_yolo_lines) — and the
    cropping/export readers must consume it. Regression for the old private
    writer that put confidence LAST, which parse_detect_predictions silently
    mis-read (conf←cx, box←cy,w,h,conf) with no parse error."""

    PRED = {
        "width": 100,
        "height": 50,
        "boxes": [[10.0, 10.0, 30.0, 20.0], [50.0, 25.0, 90.0, 45.0]],
        "labels": [1, 2],
        "scores": [0.9, 0.75],
        "count": 2,
    }

    @pytest.fixture
    def inference_phase(self, monkeypatch, tmp_path) -> dict:
        """Fake predictor + one 100x50 image; returns the phase dict to run."""
        from PIL import Image
        import tcip_mcp.pipelines.inference.generic_predictor as gp

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        Image.new("RGB", (100, 50), color=(100, 100, 100)).save(images_dir / "plot.jpg")

        pred = dict(self.PRED)

        class FakePredictor:
            def __init__(self, ckpt):
                pass

            def predict(self, image_path):
                return {"image": image_path, **pred}

        monkeypatch.setattr(gp, "GenericPredictor", FakePredictor)
        return {
            "name": "infer",
            "task": "detection",
            "checkpoint": str(tmp_path / "fake.pt"),
            "images_dir": str(images_dir),
        }

    def test_predictions_round_trip_through_parse_detect_predictions(
            self, inference_phase, tmp_path):
        from tcip_annotation.label_io import parse_detect_predictions

        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        result = orch.run_phase(inference_phase)
        assert result.status == "completed", result.error

        txt_path = Path(result.artifacts["predictions_dir"]) / "plot.txt"
        pred_boxes, class_ids = parse_detect_predictions(str(txt_path), 100, 50)
        assert len(pred_boxes) == 2
        assert class_ids == {0, 1}
        for pb, box, label, score in zip(
                pred_boxes, self.PRED["boxes"], self.PRED["labels"], self.PRED["scores"]):
            assert pb.class_id == label - 1  # torchvision 1-indexed → YOLO 0-indexed
            assert pb.confidence == pytest.approx(score, abs=1e-4)
            assert (pb.x1, pb.y1, pb.x2, pb.y2) == pytest.approx(tuple(box), abs=0.01)

    def test_cropping_reads_canonical_format(self, inference_phase, tmp_path):
        from PIL import Image

        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        infer_result = orch.run_phase(inference_phase)
        assert infer_result.status == "completed", infer_result.error

        crop_result = orch.run_phase(
            {"name": "crop", "task": "detection", "type": "cropping", "input": "dets"},
            context={"dets": infer_result.artifacts},
        )
        assert crop_result.status == "completed", crop_result.error
        assert crop_result.artifacts["count"] == 2
        crops = sorted(Path(crop_result.artifacts["images_dir"]).glob("*.jpg"))
        # Coords come from parts[2:6] — box [10,10,30,20] → 20x10, [50,25,90,45] → 40x20.
        assert [Image.open(c).size for c in crops] == [(20, 10), (40, 20)]

    def test_export_reads_confidence_from_index_1(self, inference_phase, tmp_path):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        infer_result = orch.run_phase(inference_phase)
        assert infer_result.status == "completed", infer_result.error

        export_result = orch.run_phase(
            {"name": "exp", "task": "export", "input": "dets"},
            context={"dets": infer_result.artifacts},
        )
        assert export_result.status == "completed", export_result.error
        rows = Path(export_result.artifacts["csv_path"]).read_text().strip().splitlines()
        # avg_confidence of 0.9 and 0.75 — misreading cx (0.2/0.7) would give 0.45.
        assert rows[1].split(",")[1:] == ["2", "0.825"]


# ====================================================================
# run_pipeline characterization: artifacts, failure, checkpoint/resume
# ====================================================================

class TestPipelineCharacterization:
    def test_artifact_passing_via_context(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setitem(_PHASE_RUNNERS, "producer",
                            _stub_runner(calls, artifacts={"foo": "bar"}))
        monkeypatch.setitem(_PHASE_RUNNERS, "consumer", _stub_runner(calls))
        spec = {
            "name": "artifact_pass",
            "phases": [
                {"name": "a", "task": "custom", "type": "producer", "output": "a_out"},
                {"name": "b", "task": "custom", "type": "consumer", "input": "a_out"},
            ],
        }
        pr = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_pipeline(spec)
        assert pr.status == "completed"
        assert [c["name"] for c in calls] == ["a", "b"]
        assert calls[1]["context"]["a_out"] == {"foo": "bar"}

    def test_failure_stops_pipeline_and_writes_checkpoint(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setitem(_PHASE_RUNNERS, "producer",
                            _stub_runner(calls, artifacts={"foo": "bar"}))
        monkeypatch.setitem(_PHASE_RUNNERS, "boom",
                            _stub_runner(calls, status="failed", error="ValueError: bad config"))
        spec = {
            "name": "fail_stop",
            "phases": [
                {"name": "a", "task": "custom", "type": "producer", "output": "a_out"},
                {"name": "b", "task": "custom", "type": "boom"},
                {"name": "c", "task": "custom", "type": "producer"},
            ],
        }
        work_dir = tmp_path / "runs"
        pr = PipelineOrchestrator(work_dir=str(work_dir)).run_pipeline(spec)
        assert pr.status == "failed"
        assert [c["name"] for c in calls] == ["a", "b"]  # c never ran
        assert [p.phase_name for p in pr.phases] == ["a", "b"]

        checkpoints = list(work_dir.glob("fail_stop_*/checkpoint.json"))
        assert len(checkpoints) == 1
        data = json.loads(checkpoints[0].read_text())
        assert data["last_completed"] == "a"
        assert data["context"]["a_out"] == {"foo": "bar"}
        # pipeline_result.json saved alongside
        assert (checkpoints[0].parent / "pipeline_result.json").is_file()

    def test_resume_restores_context_and_skips_completed(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setitem(_PHASE_RUNNERS, "producer",
                            _stub_runner(calls, artifacts={"foo": "bar"}))
        monkeypatch.setitem(_PHASE_RUNNERS, "flaky",
                            _stub_runner(calls, status="failed", error="ValueError: bad"))
        spec = {
            "name": "resumable",
            "phases": [
                {"name": "a", "task": "custom", "type": "producer", "output": "a_out"},
                {"name": "b", "task": "custom", "type": "flaky", "input": "a_out"},
            ],
        }
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        first = orch.run_pipeline(spec)
        assert first.status == "failed"

        # "Fix" phase b, resume after a: a must not re-run, b sees restored context.
        monkeypatch.setitem(_PHASE_RUNNERS, "flaky", _stub_runner(calls))
        calls.clear()
        second = orch.run_pipeline(spec, resume_from="a")
        assert second.status == "completed"
        assert [c["name"] for c in calls] == ["b"]
        assert calls[0]["context"]["a_out"] == {"foo": "bar"}
        # Restored phase result for a is carried into the pipeline result.
        assert [(p.phase_name, p.status) for p in second.phases] == \
            [("a", "completed"), ("b", "completed")]

    def test_resume_point_not_found_fails_instead_of_rerunning(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setitem(_PHASE_RUNNERS, "producer", _stub_runner(calls))
        spec = {
            "name": "no_checkpoint",
            "phases": [{"name": "a", "task": "custom", "type": "producer"}],
        }
        pr = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_pipeline(
            spec, resume_from="a")
        assert pr.status == "failed"
        assert calls == []  # nothing re-executed
        assert len(pr.phases) == 1
        assert pr.phases[0].phase_name == "resume"
        assert "not found" in pr.phases[0].error
        assert "'a'" in pr.phases[0].error

    def test_validation_failure_short_circuits(self, tmp_path):
        pr = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_pipeline(
            {"name": "empty", "phases": []})
        assert pr.status == "failed"
        assert pr.phases[0].phase_name == "validation"


# ====================================================================
# Retry policy
# ====================================================================

class TestRetry:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        import tcip_mcp.pipelines.orchestrator as orch_mod
        monkeypatch.setattr(orch_mod.time, "sleep", lambda s: None)

    def _run_single(self, tmp_path, monkeypatch, runner) -> tuple:
        monkeypatch.setitem(_PHASE_RUNNERS, "retryable", runner)
        spec = {
            "name": "retry_test",
            "phases": [{"name": "a", "task": "custom", "type": "retryable"}],
        }
        return PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_pipeline(spec)

    def test_transient_error_retries_then_succeeds(self, tmp_path, monkeypatch):
        attempts: list = []

        def flaky(phase, context, work_dir):
            attempts.append(1)
            if len(attempts) < 2:
                return PhaseResult(phase_name=phase["name"], status="failed",
                                   error="TimeoutError: server busy")
            return PhaseResult(phase_name=phase["name"], status="completed")

        pr = self._run_single(tmp_path, monkeypatch, flaky)
        assert pr.status == "completed"
        assert len(attempts) == 2

    def test_permanent_error_does_not_retry(self, tmp_path, monkeypatch):
        attempts: list = []

        def broken(phase, context, work_dir):
            attempts.append(1)
            return PhaseResult(phase_name=phase["name"], status="failed",
                               error="ValueError: bad config")

        pr = self._run_single(tmp_path, monkeypatch, broken)
        assert pr.status == "failed"
        assert len(attempts) == 1

    def test_transient_error_exhausts_retries(self, tmp_path, monkeypatch):
        attempts: list = []

        def always_oom(phase, context, work_dir):
            attempts.append(1)
            return PhaseResult(phase_name=phase["name"], status="failed",
                               error="CUDA out of memory")

        pr = self._run_single(tmp_path, monkeypatch, always_oom)
        assert pr.status == "failed"
        assert len(attempts) == 1 + PipelineOrchestrator.MAX_RETRIES

    def test_is_transient_heuristic(self):
        assert PipelineOrchestrator._is_transient("CUDA out of memory")
        assert PipelineOrchestrator._is_transient("ConnectionError: reset")
        assert not PipelineOrchestrator._is_transient("KeyError: 'model_spec'")


# ====================================================================
# run_phase single-phase path + failure logging
# ====================================================================

class TestRunPhase:
    def test_unknown_phase_type_fails_with_known_types(self, tmp_path):
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "x", "task": "custom", "type": "does_not_exist"})
        assert result.status == "failed"
        assert "No runner for phase type 'does_not_exist'" in result.error

    def test_custom_runner_via_register(self, tmp_path, monkeypatch):
        calls: list = []
        monkeypatch.setitem(_PHASE_RUNNERS, "custom_stub", _stub_runner(calls))
        result = PipelineOrchestrator(work_dir=str(tmp_path / "runs")).run_phase(
            {"name": "x", "task": "custom", "type": "custom_stub"}, context={"k": {"v": 1}})
        assert result.status == "completed"
        assert calls[0]["context"] == {"k": {"v": 1}}

    def test_aggregation_failure_is_logged(self, tmp_path, caplog):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        # 'results' entries must be dicts — a str raises inside aggregate_per_plant.
        context = {"prev": {"results": ["not-a-dict"]}}
        with caplog.at_level(logging.ERROR, logger="tcip_mcp.pipelines.orchestrator"):
            result = orch.run_phase(
                {"name": "agg", "task": "aggregation", "input": "prev"}, context)
        assert result.status == "failed"
        assert "Aggregation phase 'agg' failed" in caplog.text

    def test_export_failure_is_logged(self, tmp_path, caplog):
        orch = PipelineOrchestrator(work_dir=str(tmp_path / "runs"))
        context = {"prev": {"csv_path": str(tmp_path / "missing.csv")}}
        with caplog.at_level(logging.ERROR, logger="tcip_mcp.pipelines.orchestrator"):
            result = orch.run_phase(
                {"name": "exp", "task": "export", "input": "prev"}, context)
        assert result.status == "failed"
        assert "Export phase 'exp' failed" in caplog.text

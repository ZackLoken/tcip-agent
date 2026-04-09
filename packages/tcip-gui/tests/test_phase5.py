"""Tests for Phase 5 — training dashboard, results panel, widgets, and protocol."""

import json
import os
import tempfile

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from tcip_gui.protocol import (
    JsonRpcMessage,
    ResultsShow,
    TrainingComplete,
    TrainingMetricsUpdate,
    TrainingStarted,
    parse_agent_message,
)
from tcip_gui.widgets.trial_table import TrialTable
from tcip_gui.widgets.csv_preview import CsvPreview
from tcip_gui.panels.training_dashboard import TrainingDashboard
from tcip_gui.panels.results_panel import ResultsPanel


# ── TrialTable Tests ──


class TestTrialTable:
    def test_create(self, qapp):
        table = TrialTable()
        assert table._table.rowCount() == 0

    def test_set_trials(self, qapp):
        table = TrialTable()
        trials = [
            {"trial_id": 0, "params": {"lr": 0.001}, "metric": 0.71, "status": "complete"},
            {"trial_id": 1, "params": {"lr": 0.003}, "metric": None, "status": "running"},
        ]
        table.set_trials(trials)
        assert table._table.rowCount() == 2

    def test_update_trial(self, qapp):
        table = TrialTable()
        trials = [
            {"trial_id": 0, "params": {"lr": 0.001}, "metric": None, "status": "running"},
        ]
        table.set_trials(trials)
        table.update_trial(0, metric=0.85, status="complete")
        # Check metric column (index 2) updated
        item = table._table.item(0, 2)
        assert item is not None
        assert "0.85" in item.text()

    def test_clear(self, qapp):
        table = TrialTable()
        table.set_trials([{"trial_id": 0, "params": {"x": 1}, "metric": 0.5, "status": "complete"}])
        table.clear()
        assert table._table.rowCount() == 0


# ── CsvPreview Tests ──


class TestCsvPreview:
    def test_create(self, qapp):
        preview = CsvPreview()
        assert preview is not None

    def test_load_data(self, qapp):
        preview = CsvPreview()
        headers = ["plant_id", "trait_value"]
        rows = [["HAZ-001", "2026-03-12"], ["HAZ-002", "2026-03-14"]]
        preview.load_data(headers, rows)
        assert preview._table.rowCount() == 2
        assert preview._table.columnCount() == 2

    def test_load_file(self, qapp):
        preview = CsvPreview()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("plant_id,value\n")
            f.write("HAZ-001,0.95\n")
            f.write("HAZ-002,0.93\n")
            csv_path = f.name
        try:
            result = preview.load_file(csv_path)
            assert result is True
            assert preview._table.rowCount() == 2
        finally:
            os.unlink(csv_path)

    def test_load_file_not_found(self, qapp):
        preview = CsvPreview()
        result = preview.load_file("/nonexistent/file.csv")
        assert result is False

    def test_clear(self, qapp):
        preview = CsvPreview()
        preview.load_data(["a", "b"], [["1", "2"]])
        preview.clear()
        assert preview._table.rowCount() == 0


# ── TrainingDashboard Tests ──


class TestTrainingDashboard:
    def test_create(self, qapp):
        dash = TrainingDashboard()
        assert dash._total_epochs == 0

    def test_set_run_name(self, qapp):
        dash = TrainingDashboard()
        dash.set_run_name("hazelnut_v1")
        assert "hazelnut_v1" in dash._title.text()

    def test_update_progress(self, qapp):
        dash = TrainingDashboard()
        dash.update_progress(10, 50)
        assert dash._total_epochs == 50
        assert "10/50" in dash._epoch_label.text()
        assert dash._progress_bar.value() == 20  # 10/50 * 100

    def test_update_progress_with_details(self, qapp):
        dash = TrainingDashboard()
        dash.update_progress(5, 20, stage="Stage 1: Freeze", lr=0.001, eta="5 min")
        assert "Stage 1" in dash._stage_label.text()
        assert "0.001" in dash._lr_label.text()
        assert "5 min" in dash._eta_label.text()

    def test_add_metrics(self, qapp):
        dash = TrainingDashboard()
        dash.add_metrics(epoch=1, train_loss=1.5, val_loss=1.8, map50=0.3)
        dash.add_metrics(epoch=2, train_loss=1.2, val_loss=1.4, map50=0.5)
        assert dash._best_metric == 0.5
        assert dash._best_epoch == 2

    def test_set_hpo_trials(self, qapp):
        dash = TrainingDashboard()
        trials = [
            {"trial_id": 0, "params": {"lr": 0.001}, "metric": 0.71, "status": "complete"},
            {"trial_id": 1, "params": {"lr": 0.003}, "metric": 0.74, "status": "complete"},
        ]
        dash.set_hpo_trials(trials)
        assert dash._trial_table._table.rowCount() == 2

    def test_set_complete(self, qapp):
        dash = TrainingDashboard()
        dash.set_complete(0.74)
        assert dash._progress_bar.value() == 100
        assert "complete" in dash._stage_label.text().lower()

    def test_reset(self, qapp):
        dash = TrainingDashboard()
        dash.set_run_name("test")
        dash.update_progress(5, 10)
        dash.add_metrics(1, train_loss=1.0)
        dash.reset()
        assert dash._progress_bar.value() == 0
        assert dash._total_epochs == 0
        assert dash._best_metric == 0.0

    def test_metrics_file_polling(self, qapp):
        """Test that set_metrics_path starts the polling timer."""
        dash = TrainingDashboard()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"epoch": 1, "train_loss": 1.5}) + "\n")
            metrics_path = f.name
        try:
            dash.set_metrics_path(metrics_path)
            assert dash._poll_timer.isActive()
            dash.stop_polling()
            assert not dash._poll_timer.isActive()
        finally:
            os.unlink(metrics_path)

    def test_pause_signal(self, qapp, qtbot=None):
        dash = TrainingDashboard()
        received = []
        dash.pause_requested.connect(lambda: received.append(True))
        dash._pause_btn.click()
        assert len(received) == 1

    def test_stop_signal(self, qapp):
        dash = TrainingDashboard()
        received = []
        dash.stop_requested.connect(lambda: received.append(True))
        dash._stop_btn.click()
        assert len(received) == 1


# ── ResultsPanel Tests ──


class TestResultsPanel:
    def test_create(self, qapp):
        panel = ResultsPanel()
        assert panel is not None

    def test_set_run_name(self, qapp):
        panel = ResultsPanel()
        panel.set_run_name("catkin_det_v1")
        assert "catkin_det_v1" in panel._title.text()

    def test_set_overall_metrics(self, qapp):
        panel = ResultsPanel()
        panel.set_overall_metrics(map50=0.74, map50_95=0.52, precision=0.81, recall=0.78)
        assert "0.74" in panel._overall_label.text()
        assert "0.52" in panel._overall_label.text()

    def test_set_per_class_metrics(self, qapp):
        panel = ResultsPanel()
        classes = [
            {"name": "elongated", "ap50": 0.71, "precision": 0.78, "recall": 0.73},
            {"name": "non-elongated", "ap50": 0.68, "precision": 0.75, "recall": 0.70},
        ]
        panel.set_per_class_metrics(classes)
        assert panel._class_table.rowCount() == 2

    def test_set_worst_predictions(self, qapp):
        panel = ResultsPanel()
        paths = [f"/images/img{i}.jpg" for i in range(10)]
        panel.set_worst_predictions(paths)
        assert len(panel._worst_buttons) == 8  # max 8

    def test_load_csv_preview(self, qapp):
        panel = ResultsPanel()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("plant_id,date\nHAZ-001,2026-03-12\n")
            csv_path = f.name
        try:
            result = panel.load_csv_preview(csv_path)
            assert result is True
        finally:
            os.unlink(csv_path)

    def test_action_signals(self, qapp):
        panel = ResultsPanel()
        signals_fired = {"accept": 0, "retrain": 0, "hpo": 0}
        panel.accept_model.connect(lambda: signals_fired.__setitem__("accept", 1))
        panel.retrain_requested.connect(lambda: signals_fired.__setitem__("retrain", 1))
        panel.retrain_hpo_requested.connect(lambda: signals_fired.__setitem__("hpo", 1))
        panel._accept_btn.click()
        panel._retrain_btn.click()
        panel._retrain_hpo_btn.click()
        assert signals_fired == {"accept": 1, "retrain": 1, "hpo": 1}

    def test_reset(self, qapp):
        panel = ResultsPanel()
        panel.set_run_name("test")
        panel.set_overall_metrics(map50=0.5)
        panel.set_per_class_metrics([{"name": "a", "ap50": 0.5, "precision": 0.5, "recall": 0.5}])
        panel.reset()
        assert panel._class_table.rowCount() == 0
        assert "No results" in panel._overall_label.text()


# ── Training Protocol Tests ──


class TestTrainingProtocol:
    def test_training_started_roundtrip(self):
        msg = TrainingStarted(run_name="test_run", metrics_path="/tmp/m.jsonl", total_epochs=50)
        rpc = msg.to_rpc()
        assert rpc.method == "training.started"
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, TrainingStarted)
        assert parsed.run_name == "test_run"
        assert parsed.total_epochs == 50

    def test_training_metrics_update_roundtrip(self):
        msg = TrainingMetricsUpdate(
            epoch=10, train_loss=0.5, val_loss=0.6, map50=0.72, lr=0.001, stage="Stage 2", eta="5 min"
        )
        rpc = msg.to_rpc()
        assert rpc.method == "training.metrics_update"
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, TrainingMetricsUpdate)
        assert parsed.epoch == 10
        assert parsed.map50 == 0.72
        assert parsed.stage == "Stage 2"

    def test_training_metrics_update_minimal(self):
        msg = TrainingMetricsUpdate(epoch=5)
        rpc = msg.to_rpc()
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, TrainingMetricsUpdate)
        assert parsed.train_loss is None
        assert parsed.lr is None

    def test_training_complete_roundtrip(self):
        msg = TrainingComplete(
            run_name="test_run", best_epoch=38, best_metric=0.74,
            metrics={"map50": 0.74, "map50_95": 0.52},
        )
        rpc = msg.to_rpc()
        assert rpc.method == "training.complete"
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, TrainingComplete)
        assert parsed.best_epoch == 38
        assert parsed.metrics is not None

    def test_results_show_roundtrip(self):
        msg = ResultsShow(
            run_name="test_run",
            overall={"map50": 0.74, "precision": 0.81},
            per_class=[
                {"name": "elongated", "ap50": 0.71, "precision": 0.78, "recall": 0.73},
            ],
            worst_images=["/img1.jpg", "/img2.jpg"],
            csv_path="/output/results.csv",
        )
        rpc = msg.to_rpc()
        assert rpc.method == "results.show"
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, ResultsShow)
        assert parsed.overall["map50"] == 0.74
        assert len(parsed.per_class) == 1
        assert parsed.csv_path == "/output/results.csv"

    def test_results_show_minimal(self):
        msg = ResultsShow(run_name="x", overall={"map50": 0.5}, per_class=[])
        rpc = msg.to_rpc()
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, ResultsShow)
        assert parsed.worst_images is None
        assert parsed.csv_path is None

    def test_training_started_from_raw_json(self):
        raw = '{"jsonrpc":"2.0","method":"training.started","params":{"run_name":"r1","metrics_path":"/m.jsonl","total_epochs":100}}'
        rpc = JsonRpcMessage.from_line(raw)
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, TrainingStarted)
        assert parsed.total_epochs == 100

    def test_results_show_from_raw_json(self):
        raw = json.dumps({
            "jsonrpc": "2.0",
            "method": "results.show",
            "params": {
                "run_name": "r1",
                "overall": {"map50": 0.8},
                "per_class": [{"name": "cls1", "ap50": 0.8, "precision": 0.9, "recall": 0.7}],
                "worst_images": ["/a.jpg"],
            },
        })
        rpc = JsonRpcMessage.from_line(raw)
        parsed = parse_agent_message(rpc)
        assert isinstance(parsed, ResultsShow)
        assert len(parsed.worst_images) == 1


# ── Bridge Training Dispatch Tests ──


class TestTrainingBridgeDispatch:
    def test_dispatch_training_started(self, qapp):
        from tcip_gui.bridge import AgentBridge

        bridge = AgentBridge("echo", [])
        received = []
        bridge.training_started.connect(lambda n, p, e: received.append((n, p, e)))
        bridge._dispatch_message(TrainingStarted(run_name="r1", metrics_path="/m.jsonl", total_epochs=50))
        assert len(received) == 1
        assert received[0] == ("r1", "/m.jsonl", 50)

    def test_dispatch_training_metrics(self, qapp):
        from tcip_gui.bridge import AgentBridge

        bridge = AgentBridge("echo", [])
        received = []
        bridge.training_metrics_update.connect(lambda u: received.append(u))
        msg = TrainingMetricsUpdate(epoch=5, train_loss=0.3, map50=0.7)
        bridge._dispatch_message(msg)
        assert len(received) == 1
        assert isinstance(received[0], TrainingMetricsUpdate)

    def test_dispatch_training_complete(self, qapp):
        from tcip_gui.bridge import AgentBridge

        bridge = AgentBridge("echo", [])
        received = []
        bridge.training_complete.connect(lambda n, e, m, d: received.append((n, e, m)))
        bridge._dispatch_message(TrainingComplete(run_name="r1", best_epoch=10, best_metric=0.8))
        assert len(received) == 1
        assert received[0] == ("r1", 10, 0.8)

    def test_dispatch_results_show(self, qapp):
        from tcip_gui.bridge import AgentBridge

        bridge = AgentBridge("echo", [])
        received = []
        bridge.results_show.connect(lambda r: received.append(r))
        msg = ResultsShow(
            run_name="r1",
            overall={"map50": 0.7},
            per_class=[],
        )
        bridge._dispatch_message(msg)
        assert len(received) == 1
        assert isinstance(received[0], ResultsShow)

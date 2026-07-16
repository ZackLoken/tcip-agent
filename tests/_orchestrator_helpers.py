"""Shared trainer-stubbing helpers for test_orchestrator.py / test_orchestrator_characterization.py.

Both files stub the same seam (``generic_trainer.create_run``/``train``) to exercise
``PipelineOrchestrator.run_phase`` without real training; kept here once so the two suites
can't drift out of sync on what they stub.
"""

from __future__ import annotations

from pathlib import Path


class FakeRun:
    def __init__(self, run_id: str, output_dir: str, status: str = "completed",
                 error: str = "") -> None:
        self.run_id = run_id
        self.output_dir = output_dir
        self.status = status
        self.metrics_history = [{"train_loss": 0.25}]
        self.error = error


def build_stub_trainer(monkeypatch, tmp_path: Path, run_id: str) -> dict:
    """Stub create_run/train so training phases exercise dataset building and tracking
    wiring without real training. Keeps .tcip writes inside tmp_path.

    ``captured["run_status"]``/``captured["run_error"]`` control the fake run's outcome.
    """
    import tcip_mcp.pipelines.training.generic_trainer as gt

    captured: dict = {"run_status": "completed", "run_error": ""}

    def fake_create_run(config, output_dir, origin="training"):
        captured["config"] = config
        run = FakeRun(run_id, output_dir, status=captured["run_status"], error=captured["run_error"])
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
    monkeypatch.chdir(tmp_path)
    return captured

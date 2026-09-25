"""A bespoke builder that lives outside the interpreter's search path imports through its own
``source_files``: the builder module's import root joins ``sys.path`` at the one import site, a
fresh subprocess launched with ``child_pythonpath`` inherits it, and a run naming such a builder
launches and trains in the platform's training worker.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOP_LEVEL_BUILDER = '''
def build_probe(**kwargs):
    return {"built": True, "kwargs": kwargs}
'''

PACKAGED_BUILDER = '''
from tests.tiny_trainer_fixtures import build_mean_intensity_classifier


def build(**kwargs):
    return build_mean_intensity_classifier(**kwargs)
'''


def _builder_dir(tmp_path: Path) -> Path:
    src = tmp_path / "agent_project" / "models"
    src.mkdir(parents=True)
    (src / "probe_builder.py").write_text(TOP_LEVEL_BUILDER, encoding="utf-8", newline="\n")
    return src


def _packaged_builder(tmp_path: Path, package: str) -> tuple[Path, Path]:
    """``<tmp>/agent_project/<package>/model.py`` holding ``build``: the project directory is the
    import root, one level above the file."""
    project = tmp_path / "agent_project"
    pkg = project / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8", newline="\n")
    model = pkg / "model.py"
    model.write_text(PACKAGED_BUILDER, encoding="utf-8", newline="\n")
    return project, model


def _child_imports(module: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", f"import {module}; print({module}.__name__)"],
        capture_output=True, text=True, env=env,
    )


def test_a_builder_outside_the_interpreters_path_launches_in_a_fresh_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.pipelines.model_build import child_pythonpath, import_source_builder

    src = _builder_dir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(src)])
    monkeypatch.delenv("PYTHONPATH", raising=False)
    model_source = {"builder": "probe_builder:build_probe",
                    "source_files": [str(src / "probe_builder.py")]}

    before = _child_imports("probe_builder", {**os.environ, "PYTHONPATH": child_pythonpath()})
    assert before.returncode != 0
    assert "ModuleNotFoundError" in before.stderr

    fn = import_source_builder(model_source)
    assert fn(width=4) == {"built": True, "kwargs": {"width": 4}}
    assert str(src) in sys.path

    after = _child_imports("probe_builder", {**os.environ, "PYTHONPATH": child_pythonpath()})
    assert after.returncode == 0, after.stderr


def test_a_packaged_builder_imports_from_its_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pkg.model:build`` at ``project/pkg/model.py`` imports from ``project``, never from the
    file's own directory, which would make ``pkg`` itself unimportable."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.model_build import child_pythonpath, import_source_builder

    project, model = _packaged_builder(tmp_path, "agentpkg_import")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    build = import_source_builder({"builder": "agentpkg_import.model:build",
                                   "source_files": [str(model)]})

    assert build().num_classes == 2
    assert str(project) in sys.path
    assert str(model.parent) not in sys.path
    child = _child_imports("agentpkg_import.model",
                           {**os.environ, "PYTHONPATH": child_pythonpath()})
    assert child.returncode == 0, child.stderr


def test_a_packaged_builder_outside_the_path_launches_and_trains_in_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run names only the builder and its file; the launcher's preflight imports it, the
    worker process builds it, and the run completes."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import launch_training, monitor_training
    from tests.tiny_trainer_fixtures import write_regression_dataset

    project, model = _packaged_builder(tmp_path, "agentpkg_launch")
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(project)])
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})
    images_dir, csv_path = write_regression_dataset(
        tmp_path / "ds", intensities=[0.1, 0.9] * 4, values=[0, 1] * 4)
    cfg = {
        "model_source": {"builder": "agentpkg_launch.model:build", "task": "classification",
                         "in_chans": 3, "source_files": [str(model)]},
        "data": {"images_dir": str(images_dir), "labels_dir": str(csv_path)},
        "batch_size": 4, "stages": [{"freeze_to": 0, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "checkpoint_every_n_epochs": 0, "early_stopping": {"enabled": False},
    }

    res = launch_training(cfg, str(tmp_path / "out"))

    assert "error" not in res, res
    assert res["pid"] != os.getpid()
    deadline = time.monotonic() + 120
    status: dict = {}
    while time.monotonic() < deadline:
        status = monitor_training(res["experiment_id"])
        if status.get("status") in ("completed", "failed", "canceled"):
            break
        time.sleep(0.5)
    assert status.get("status") == "completed", status


def test_preflight_imports_a_builder_through_its_source_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one import site serves preflight too: a config naming its builder's file passes the
    importability check without the agent putting the directory on the path by hand."""
    from tcip_mcp.tools.training_tools import preflight_config

    src = _builder_dir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(src)])
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    cfg = {
        "model_source": {"builder": "probe_builder:build_probe", "task": "classification",
                         "source_files": [str(src / "probe_builder.py")]},
        "data": {"images_dir": str(images_dir), "task": "classification"},
    }

    result = preflight_config(cfg)

    assert not any("not importable" in issue for issue in result["issues"]), result["issues"]

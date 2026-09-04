"""``scripts/conform_job_registry_roots.py``: the one-off conform step stamping a job-registry
document's own root onto every summary it holds that predates the ``platform_root`` field.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import bind_default
from tcip_store.sqlite_backend import SqliteBackend

from tcip_web.jobstore import (
    HPO_SWEEPS,
    INFERENCE_JOBS,
    REVIEW_PRIORITY_JOBS,
    job_registry_key,
)

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_job_registry_roots.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "conform_job_registry_roots_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def _summary(**overrides) -> dict:
    base = {"job_id": "job-1", "status": "completed", "done": 3, "total": 3, "error": None}
    base.update(overrides)
    return base


def _seed(root: Path, name: str, summaries: list[dict]) -> None:
    (root / ".tcip").mkdir(parents=True, exist_ok=True)
    ts.replace(job_registry_key(name, root=root), summaries, expect=ts.Version.ABSENT)


def test_a_summary_lacking_platform_root_is_stamped_with_the_documents_own_root(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed(tmp_path, INFERENCE_JOBS, [_summary()])

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    stamped_root = str(tmp_path.resolve())
    assert f"{INFERENCE_JOBS}/job-1: stamped platform_root to {stamped_root}" in "\n".join(outcomes)
    stored = ts.read(job_registry_key(INFERENCE_JOBS, root=tmp_path))
    assert stored[0]["platform_root"] == stamped_root
    assert stored[0]["status"] == "completed"


def test_a_summary_already_carrying_platform_root_is_a_no_op(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    summary = _summary(platform_root="S:/elsewhere")
    _seed(tmp_path, INFERENCE_JOBS, [summary])

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    assert outcomes == [f"{INFERENCE_JOBS}/job-1: already carries platform_root, unchanged"]
    stored = ts.read(job_registry_key(INFERENCE_JOBS, root=tmp_path))
    assert stored[0]["platform_root"] == "S:/elsewhere"


def test_plan_mode_writes_nothing(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed(tmp_path, INFERENCE_JOBS, [_summary()])

    outcomes, refused = module.conform_root(tmp_path, plan=True)

    assert refused is False
    stamped_root = str(tmp_path.resolve())
    assert (
        f"{INFERENCE_JOBS}/job-1: would stamp platform_root to {stamped_root}"
        in "\n".join(outcomes)
    )
    stored = ts.read(job_registry_key(INFERENCE_JOBS, root=tmp_path))
    assert "platform_root" not in stored[0]


def test_every_job_registry_document_is_conformed(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed(tmp_path, INFERENCE_JOBS, [_summary(job_id="inf-1")])
    _seed(tmp_path, REVIEW_PRIORITY_JOBS, [_summary(job_id="pq-1")])
    _seed(tmp_path, HPO_SWEEPS, [{"sweep_id": "hpo-1", "status": "completed", "error": None}])

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is False
    stamped_root = str(tmp_path.resolve())
    for name, expected_id in (
        (INFERENCE_JOBS, "inf-1"), (REVIEW_PRIORITY_JOBS, "pq-1"), (HPO_SWEEPS, "hpo-1"),
    ):
        stored = ts.read(job_registry_key(name, root=tmp_path))
        assert stored[0]["platform_root"] == stamped_root
        assert f"{name}/{expected_id}: stamped platform_root to {stamped_root}" in "\n".join(
            outcomes)


def test_a_root_with_no_tcip_directory_is_refused_by_name(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()

    outcomes, refused = module.conform_root(tmp_path, plan=False)

    assert refused is True
    assert outcomes == ["refused, no .tcip directory found; not a project root"]


def test_main_conforms_two_roots_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bind_default()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _seed(root_a, INFERENCE_JOBS, [_summary()])
    _seed(root_b, INFERENCE_JOBS, [_summary()])

    monkeypatch.setattr(
        sys, "argv", ["conform_job_registry_roots.py", str(root_a), str(root_b)])
    exit_code = module.main()

    assert exit_code == 0
    for root in (root_a, root_b):
        stored = ts.read(job_registry_key(INFERENCE_JOBS, root=root))
        assert stored[0]["platform_root"] == str(root.resolve())


def test_main_a_missing_root_does_not_block_a_second_roots_conform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    bind_default()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_b.mkdir()
    _seed(root_b, INFERENCE_JOBS, [_summary()])

    monkeypatch.setattr(
        sys, "argv", ["conform_job_registry_roots.py", str(root_a), str(root_b)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{root_a.resolve()}: refused, no .tcip directory found" in output
    stored = ts.read(job_registry_key(INFERENCE_JOBS, root=root_b))
    assert stored[0]["platform_root"] == str(root_b.resolve())

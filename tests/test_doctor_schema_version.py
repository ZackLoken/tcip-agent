"""scripts/doctor.py reads state as raw files and reports a version finding rather than
refusing: a soft rail, the same posture as adoption's preflight, since the doctor never acts on
a document's content.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import tcip_store as ts
from tcip_mcp.dataset_layout import region_completeness_key
from tcip_store.file_backend import FileBackend

PY_EXE = sys.executable
DOCTOR = str(Path(__file__).parent.parent / "scripts" / "doctor.py")


def _run(root: Path):
    env = {**os.environ, "TCIP_STORE_BACKEND": "file"}
    return subprocess.run([PY_EXE, DOCTOR, str(root)], capture_output=True, text=True, env=env)


def test_doctor_reports_but_does_not_refuse_on_an_unsupported_region_completeness_version(
    tmp_path,
):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)

    ts.bind(FileBackend())
    try:
        ts.replace(
            region_completeness_key(root), {"schema_version": 2, "leaf/2024-01-01": {}},
            expect=ts.Version.ABSENT,
        )
    finally:
        ts.unbind()

    res = _run(root)

    assert "schema_version" in res.stdout
    assert res.returncode != 2  # a raw-parse version finding is a warning, never a blocking error

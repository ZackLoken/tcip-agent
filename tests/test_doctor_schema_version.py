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
from tcip_mcp.dataset_layout import (
    image_status_key, region_completeness_digest_key, region_completeness_key,
)
from tcip_mcp.traits import trait_spec_key, trait_specs_dir
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

    key = region_completeness_key(root)
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.RECORD_JSON.encode({"schema_version": 2, "leaf/2024-01-01": {}}))

    res = _run(root)

    assert "schema_version 2, above the 1 this reader knows" in res.stdout
    assert res.returncode != 2  # a raw-parse version finding is a warning, never a blocking error


def test_doctor_reports_the_digest_version_even_when_the_main_store_has_no_recognized_bucket(
    tmp_path,
):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)

    ts.bind(FileBackend())
    try:
        # No recognized bucket at all: normalize_region_completeness_store(raw) reads as empty,
        # so the digest file's own version finding must still be reachable, not short-circuited.
        ts.replace(
            region_completeness_key(root), {"leaf/2024-01-01": {}}, expect=ts.Version.ABSENT,
        )
        ts.replace(
            region_completeness_digest_key(root), {"schema_version": 2},
            expect=ts.Version.ABSENT,
        )
    finally:
        ts.unbind()

    res = _run(root)

    assert "region_completeness_digest.json" in res.stdout
    assert "schema_version 2, above the 1 this reader knows" in res.stdout
    assert res.returncode != 2


def test_doctor_reports_a_version_refused_trait_spec_as_a_warning_not_a_crash(tmp_path):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)

    ts.bind(FileBackend())
    try:
        ts.replace(
            trait_spec_key(trait_specs_dir(root), "sometrait"),
            {"schema_version": 2, "name": "sometrait"}, expect=ts.Version.ABSENT,
        )
    finally:
        ts.unbind()

    res = _run(root)

    assert "sometrait" in res.stdout
    assert "schema_version 2, above the 1 this reader knows" in res.stdout
    assert "Traceback" not in res.stderr
    assert res.returncode != 2


def test_doctor_unifies_check_negatives_with_check_status_tokens_on_a_version_refused_store(
    tmp_path,
):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)

    ts.bind(FileBackend())
    try:
        ts.replace(
            image_status_key(root), {"schema_version": 2}, expect=ts.Version.ABSENT,
        )
    finally:
        ts.unbind()

    res = _run(root)

    assert "schema_version 2, above the 1 this reader knows" in res.stdout
    assert "negatives cannot be verified" in res.stdout
    assert res.returncode != 2  # a warning, the same posture check_status_tokens already takes

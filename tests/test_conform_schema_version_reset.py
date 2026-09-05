"""``scripts/conform_schema_version_reset.py``: strip a stray ``schema_version: 2`` from a root's
model registry index, prediction-bucket sidecars and ``confidence_sweep`` records, following the
version-1 reset. No current producer writes the field any more (that is this fix's own point) and
the seam now refuses it on write as well as on read, so every case here plants the dev-era shape
by overwriting a record's raw bytes directly, the same technique
``test_conform_classified_predictions.py``'s own ``_damage_record`` helper uses for a document no
writer produces any more.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
from pathlib import Path

import tcip_store as ts
from tcip_store.binding import BACKEND_ENV, DEFAULT_BACKEND, FILE_BACKEND
from tcip_store.store import _backend

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "conform_schema_version_reset.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_schema_version_reset_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _damage_record(key: ts.Key, data: bytes) -> None:
    """Overwrite a record's raw bytes, wherever the bound backend keeps it, bypassing the seam's
    own write-side schema_version check entirely: what an already-on-disk document from before
    this reset's ceiling drop looks like to today's reader. The record must already exist at
    ``key`` (written legitimately first)."""
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        _backend().path_for(key).write_bytes(data)
        return
    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        conn.execute(
            "update records set value = ? where store = ? and parts = ?",
            (data, key.store, encode_parts(key.parts)),
        )
    finally:
        conn.close()


def _read_raw(key: ts.Key) -> bytes:
    """The record's exact stored bytes, bypassing the seam's own read-side schema_version check
    (which would otherwise refuse a still-planted document): the counterpart to
    :func:`_damage_record`."""
    name = os.environ.get(BACKEND_ENV) or DEFAULT_BACKEND
    if name == FILE_BACKEND:
        return _backend().path_for(key).read_bytes()
    from tcip_store.sqlite_backend import database_path, encode_parts

    conn = sqlite3.connect(str(database_path(str(key.root))), isolation_level=None)
    try:
        row = conn.execute(
            "select value from records where store = ? and parts = ?",
            (key.store, encode_parts(key.parts)),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row[0]


def _plant_v2(key: ts.Key, body: dict) -> None:
    """A record legitimately created, then overwritten to carry ``schema_version: 2`` in its
    raw bytes: the shape a dev-era writer, predating this reset, left behind."""
    placeholder = {k: v for k, v in body.items() if k != "schema_version"}
    ts.replace(key, placeholder, expect=ts.Version.ABSENT)
    encoded = ts.get_descriptor(key.store).codec.encode({**placeholder, "schema_version": 2})
    _damage_record(key, encoded)


def test_registry_index_drops_a_stray_schema_version_two(tmp_path: Path):
    from tcip_mcp.model_registry import registry_index_key

    root = tmp_path / "proj"
    entries = [{"name": "m", "checkpoint_path": "x.pt", "sha256": "a" * 64}]
    key = registry_index_key(root)
    _plant_v2(key, {"entries": entries})

    module = _load_script()
    lines, refused = module.check_root(root)

    assert refused is False
    assert any("model registry index: dropped schema_version" in ln for ln in lines)
    raw = ts.read(key)
    assert raw == {"entries": entries}


def test_registry_index_with_no_field_is_reported_unchanged(tmp_path: Path):
    from tcip_mcp.model_registry import ModelRegistry, registry_index_key

    root = tmp_path / "proj"
    ckpt = root / "m.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"weights")
    ModelRegistry(str(root)).register_model("m", str(ckpt), {}, metrics_source=None)

    module = _load_script()
    lines, refused = module.check_root(root)

    assert refused is False
    assert any("model registry index: no schema_version 2 field, unchanged" in ln for ln in lines)
    assert "schema_version" not in ts.read(registry_index_key(root))


def test_registry_index_as_a_bare_array_is_named_a_refusal_not_unchanged(tmp_path: Path):
    """A bare top-level array carries no schema_version key to be above the ceiling on, so a
    generic per-record read alone would call it unchanged; the real reader
    (read_registry_index) refuses this shape outright, and this script must report that
    refusal rather than mask it as a clean root."""
    from tcip_mcp.model_registry import registry_index_key

    root = tmp_path / "proj"
    entries = [{"name": "m", "checkpoint_path": "x.pt", "sha256": "a" * 64}]
    key = registry_index_key(root)
    ts.replace(key, entries, expect=ts.Version.ABSENT)

    module = _load_script()
    lines, refused = module.check_root(root)

    assert refused is True
    assert any(
        "model registry index: refused" in ln and "conform_model_registry_paths.py" in ln
        for ln in lines
    ), lines
    assert ts.read(key) == entries


def _register_bucket(root: Path) -> Path:
    """A registered dataset, coincident with ``root``, holding one prediction bucket dir."""
    from tcip_mcp.tools.project_tools import register_dataset

    root.mkdir(parents=True, exist_ok=True)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    bucket = root / "predictions" / "detector_v1" / "2026-03-04"
    bucket.mkdir(parents=True)
    return bucket


def test_sidecar_document_drops_a_stray_schema_version_two(tmp_path: Path):
    from tcip_mcp.pipelines.resolution import sidecar_key

    root = tmp_path / "proj"
    bucket = _register_bucket(root)
    key = sidecar_key(bucket, "operating_point")
    _plant_v2(key, {"trait": "leaf_count", "operating_point": {"conf": {"value": 0.5}}})

    module = _load_script()
    lines, refused = module.check_root(root)

    assert refused is False
    assert any("operating_point.json: dropped schema_version" in ln for ln in lines)
    stored = ts.read(key)
    assert "schema_version" not in stored
    assert stored["trait"] == "leaf_count"


def test_sidecar_with_no_field_is_reported_unchanged(tmp_path: Path):
    from tcip_mcp.pipelines.resolution import sidecar_key

    root = tmp_path / "proj"
    bucket = _register_bucket(root)
    key = sidecar_key(bucket, "resolve_scale")
    ts.replace(key, {"trait": "leaf_length"}, expect=ts.Version.ABSENT)

    module = _load_script()
    lines, refused = module.check_root(root)

    assert any("resolve_scale.json: no schema_version 2 field, unchanged" in ln for ln in lines)


def test_confidence_sweep_record_drops_the_field_and_names_the_orphaned_filename(tmp_path: Path):
    """The discovery is a filesystem glob (the store is not declared enumerable), so this proof
    binds the file backend explicitly, matching the script's own documented backend limitation.
    Plants the raw bytes directly against that explicit binding, since the ambient
    ``TCIP_STORE_BACKEND`` this process started under (whichever the running gate leg set) is not
    necessarily the one this test just bound. The record is filed under its own true digest
    (computed the same way the platform's own writer would key it), so the note fires: the
    filename genuinely matched the body before the strip and no longer does after."""
    from tcip_store.file_backend import FileBackend
    from tcip_mcp.tools.inference_tools import CONFIDENCE_SWEEP_STORE, calibration_curve_identity

    root = tmp_path / "proj"
    root.mkdir(parents=True)
    body = {"trait": "leaf_count", "dataset_hash": "H", "checkpoint_sha256": "0" * 64,
            "predictor_path": {}, "gate_evidence": {}, "calibration_evidence": {}}
    poisoned_body = {**body, "schema_version": 2}
    digest = calibration_curve_identity(poisoned_body)
    key = ts.Key(CONFIDENCE_SWEEP_STORE, str(root.resolve()), (digest,))
    ts.bind(FileBackend())
    try:
        ts.replace(key, body, expect=ts.Version.ABSENT)
        encoded = ts.get_descriptor(key.store).codec.encode(poisoned_body)
        FileBackend().path_for(key).write_bytes(encoded)

        module = _load_script()
        lines, refused = module.check_root(root)

        assert refused is False
        conform_lines = [ln for ln in lines if f"confidence_sweep {digest}" in ln]
        assert any("dropped schema_version" in ln and "now an orphan under this filename" in ln
                   for ln in conform_lines), conform_lines
        stored = ts.read(key)
        assert "schema_version" not in stored
    finally:
        ts.unbind()


def test_confidence_sweep_record_whose_filename_never_matched_names_no_orphan(tmp_path: Path):
    """A record filed under a digest that was never its own body's true digest (a dev-era key
    scheme this reset did not create) is still conformed, but names no orphan note: the mismatch
    predates this conform and is not one it caused."""
    from tcip_store.file_backend import FileBackend
    from tcip_mcp.tools.inference_tools import CONFIDENCE_SWEEP_STORE

    root = tmp_path / "proj"
    root.mkdir(parents=True)
    digest = "a" * 64  # never the body's own digest, before or after the strip
    key = ts.Key(CONFIDENCE_SWEEP_STORE, str(root.resolve()), (digest,))
    ts.bind(FileBackend())
    try:
        body = {"trait": "leaf_count", "dataset_hash": "H", "checkpoint_sha256": "0" * 64,
                "predictor_path": {}, "gate_evidence": {}, "calibration_evidence": {}}
        ts.replace(key, body, expect=ts.Version.ABSENT)
        encoded = ts.get_descriptor(key.store).codec.encode({**body, "schema_version": 2})
        FileBackend().path_for(key).write_bytes(encoded)

        module = _load_script()
        lines, refused = module.check_root(root)

        assert refused is False
        conform_lines = [ln for ln in lines if f"confidence_sweep {digest}" in ln]
        assert conform_lines == [f"confidence_sweep {digest}: dropped schema_version"]
        stored = ts.read(key)
        assert "schema_version" not in stored
    finally:
        ts.unbind()


def test_audit_log_lines_above_the_ceiling_are_named_never_touched(tmp_path: Path):
    from tcip_mcp.audit import audit_log_key
    from tcip_store.file_backend import FileBackend

    root = tmp_path / "proj"
    root.mkdir(parents=True)
    ts.bind(FileBackend())
    try:
        key = audit_log_key(root)
        ts.append(key, {"tool": "before"})
        poisoned = ts.get_descriptor(key.store).codec.encode({"tool": "dev_era", "schema_version": 2})
        with open(FileBackend().path_for(key), "ab") as handle:
            handle.write(poisoned + b"\n")
        ts.append(key, {"tool": "after"})

        module = _load_script()
        lines, refused = module.check_root(root)

        assert refused is False
        assert any(
            "audit_log: 1 line(s) refuse at a schema_version above the ceiling" in ln
            for ln in lines
        )
        page = ts.read_log(key)
        assert page.version_refused == (1,)
        assert [r["tool"] for r in page.records] == ["before", "after"]
    finally:
        ts.unbind()


def test_experiment_validations_lines_above_the_ceiling_are_named_never_touched(
    tmp_path: Path, monkeypatch,
):
    from tcip_mcp.experiments import create_experiment, validations_key
    from tcip_store.file_backend import FileBackend

    root = tmp_path / "proj"
    root.mkdir(parents=True)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(root))
    ts.bind(FileBackend())
    try:
        create_experiment("exp1", {"model_source": {"builder": "m:f"}})
        key = validations_key("exp1", root=root)
        ts.append(key, {"document": "a"})
        poisoned = ts.get_descriptor(key.store).codec.encode({"document": "b", "schema_version": 2})
        with open(FileBackend().path_for(key), "ab") as handle:
            handle.write(poisoned + b"\n")

        module = _load_script()
        lines, refused = module.check_root(root)

        assert refused is False
        assert any(
            "experiment_validations exp1: 1 line(s) refuse at a schema_version above the ceiling"
            in ln
            for ln in lines
        )
    finally:
        ts.unbind()


def test_plan_writes_nothing(tmp_path: Path):
    from tcip_mcp.pipelines.resolution import sidecar_key

    root = tmp_path / "proj"
    bucket = _register_bucket(root)
    key = sidecar_key(bucket, "operating_point")
    _plant_v2(key, {"trait": "leaf_count"})

    module = _load_script()
    lines, refused = module.check_root(root, plan=True)

    assert any("would drop schema_version" in ln for ln in lines)
    stored = ts.get_descriptor(key.store).codec.decode(_read_raw(key))
    assert stored["schema_version"] == 2


def test_a_root_with_no_tcip_directory_refuses(tmp_path: Path):
    module = _load_script()
    lines, refused = module.check_root(tmp_path / "not_a_project")

    assert refused is True
    assert "no .tcip directory" in lines[0]

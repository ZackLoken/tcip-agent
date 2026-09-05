"""Rewrite a root's frozen-store documents that still carry an explicit ``schema_version: 2`` to
carry no field, following the version-1 reset: every frozen store now ships at ceiling 1, absence
is the lazy default (``frozen-formats.json``), and a dev-era document stamped ``schema_version: 2``
under the old ceiling now sits above the new one unless conformed.

Walks, for each named root, three targets: the model registry index
(``tcip_mcp.model_registry.registry_index_key``); every one of the five sidecar documents
(``operating_point``, ``classifier_operating_point``, ``ordinal_operating_point``,
``regression_operating_point``, ``resolve_scale``) under every registered dataset's prediction
buckets (``tcip_mcp.tools.project_tools.read_datasets`` names each dataset root,
``tcip_mcp.dataset_layout.prediction_bucket_dirs`` walks its buckets); and every ``confidence_sweep``
record kept as a loose file under the root's own ``.tcip/artifacts/``. That last discovery is a
filesystem glob, not a seam enumeration: the store is not declared ``enumerable``, so a root bound
to the sqlite backend, whose rows live in ``store.db`` rather than as loose files, is not discovered
by this walk. Export such a root first (``scripts/export_store.py``) before trusting a sqlite root's
report of "nothing found" for that target.

The seam's own read-side ceiling check refuses a ``schema_version: 2`` document outright, so this
script never sees one through an ordinary read: it decodes such a document's raw bytes directly,
strips the field, and writes the stripped body back through the seam's own compare-and-set, keyed
off the raw bytes' own content hash, so a document another run already conformed (or a concurrent
writer changed) is reported unchanged or refused rather than rewritten twice or blind. Two things
are named rather than silently conformed:

- A log store's own lines (``audit_log``, ``experiment_validations``) are never rewritten: a log is
  append-only, and a line predating the reset carries the fact it was written under, not a mistake
  to correct. This script only counts and names how many lines of each ``read_log`` reports
  version-refused (a line the ceiling drop puts out of this reader's reach; it never reaches
  ``page.records`` for its own field to be re-inspected here); dev-era logs are cleared by a
  separate step, outside this script.
- A ``confidence_sweep`` record's own key is a content digest over its body
  (``tcip_mcp.tools.inference_tools.calibration_curve_identity``). Stripping ``schema_version``
  changes the digest a fresh read recomputes for it, so the conformed record no longer matches the
  key it stays filed under. This script still conforms it, per its own governing rule, and names
  every such record as a validated stamp whose claim floors: any bucket's stored
  ``calibration_evidence_key`` naming this record's old digest will not resolve the rewritten body
  back to that same digest, so the count gate's own tamper check
  (``inference_tools._calibration_evidence``) refuses it on the next read. Neither the operating-
  point-family sidecars nor the registry index carry a content-derived key, so nothing else this
  script touches floors a claim this way.

``--plan`` previews every outcome without writing anything.

    python scripts/conform_schema_version_reset.py <root> [<root> ...]
    python scripts/conform_schema_version_reset.py --plan <root>

Exit codes: 0 once every named root was walked, whatever it held or however many documents carried
the field; 2 if any target under any root will not read (a decode error, an unrecognized registry
index shape), or a named root holds no ``.tcip`` directory to walk.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_annotation.json_io import SIDECAR_FILENAMES  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402
from tcip_store.errors import StoreError  # noqa: E402
from tcip_store.file_backend import FileBackend  # noqa: E402
from tcip_store.model import Key  # noqa: E402
from tcip_store.store import _backend  # noqa: E402

_SIDECAR_DOCUMENTS = sorted(name[: -len(".json")] for name in SIDECAR_FILENAMES)


def _raw_bytes(key: Key) -> bytes | None:
    """A record's exact stored bytes, bypassing the seam's own schema_version check on read, or
    ``None`` when nothing is stored. The one place this script reaches past the seam: a document
    above the new ceiling refuses at every ordinary read, so seeing what it actually holds (to
    decide whether stripping the field even applies) has no path through the seam at all.

    Branches on the backend actually bound (``_backend()``'s own type), never on
    ``TCIP_STORE_BACKEND``: the environment variable is only what ``bind_default`` reads at
    startup, and need not still describe the bound instance to a caller that rebound it since.
    """
    backend = _backend()
    if isinstance(backend, FileBackend):
        path = backend.path_for(key)
        return path.read_bytes() if path.is_file() else None
    from tcip_store.sqlite_backend import database_path, encode_parts

    db_path = database_path(str(key.root))
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        row = conn.execute(
            "select value from records where store = ? and parts = ?",
            (key.store, encode_parts(key.parts)),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _conform_poisoned_record(
    key: Key, *, label: str, plan: bool, note_fn: Callable[[dict], str] | None = None,
) -> str:
    """A document the seam refuses outright on read (its own schema_version sits above the new
    ceiling): decode its raw bytes directly, strip a schema_version of exactly ``2``, and write
    the stripped body back through the seam's own compare-and-set, keyed off the raw bytes' own
    content hash (the same ``Version`` every backend computes) so a concurrent write to the same
    poisoned record is still detected rather than silently lost. A version other than ``2`` is
    named, never guessed at.

    ``note_fn``, when given, is called with the stripped body and appends its own text to the
    outcome line (both the ``would drop`` preview and the actual conform), for a caller whose
    store carries a consequence beyond the field drop itself (a content-derived key that no
    longer matches once the field is gone).
    """
    raw_bytes = _raw_bytes(key)
    if raw_bytes is None:
        return f"{label}: no schema_version 2 field, unchanged"
    descriptor = ts.get_descriptor(key.store)
    assert descriptor.codec is not None, f"{key.store} is a record store; every record declares a codec"
    try:
        decoded = descriptor.codec.decode(raw_bytes)
    except Exception as exc:
        return f"{label}: refused, will not decode even bypassing the version check ({exc})"
    if not isinstance(decoded, dict):
        return f"{label}: refused, above the ceiling and not a mapping to strip a field from"
    version = decoded.get("schema_version")
    if version != 2:
        return (f"{label}: refused, schema_version={version!r} is above the ceiling and not "
                "the 2 this reset conforms; left as stored")
    stripped = {k: v for k, v in decoded.items() if k != "schema_version"}
    note = note_fn(stripped) if note_fn is not None else ""
    if plan:
        return f"{label}: would drop schema_version{note}"
    expect = ts.Version(hashlib.sha256(raw_bytes).hexdigest())
    try:
        ts.replace(key, stripped, expect=expect)
    except ts.VersionConflict:
        return f"{label}: refused, changed under the lock before this could be conformed; re-run"
    return f"{label}: dropped schema_version{note}"


def _conform_record(key: Key, *, label: str, plan: bool) -> str:
    """Strip a stray ``schema_version: 2`` from one record, reporting the outcome by ``label``.

    An ordinary read that succeeds never carries the field: the seam's own read-side check
    already refuses anything above the ceiling, so a document actually stamped ``2`` is only
    ever reachable by bypassing that check, in :func:`_conform_poisoned_record`.
    """
    try:
        ts.read(key, default=None)
    except ts.SchemaVersionRefused:
        return _conform_poisoned_record(key, label=label, plan=plan)
    except StoreError as exc:
        return f"{label}: refused, {exc}"
    return f"{label}: no schema_version 2 field, unchanged"


def _conform_registry_index(root: Path, *, plan: bool) -> str:
    from tcip_mcp.model_registry import registry_index_key

    return _conform_record(registry_index_key(root), label="model registry index", plan=plan)


def _conform_sidecars(root: Path, *, plan: bool) -> list[str]:
    from tcip_mcp.dataset_layout import prediction_bucket_dirs
    from tcip_mcp.pipelines.resolution import sidecar_key
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets

    try:
        datasets = read_datasets(root)
    except StoreError as exc:
        return [f"dataset registry: refused, {exc}"]

    lines: list[str] = []
    for entry in datasets:
        dataset_root = dataset_entry_path(root, entry)
        for pred_dir in prediction_bucket_dirs(dataset_root):
            for document in _SIDECAR_DOCUMENTS:
                key = sidecar_key(pred_dir, document)
                lines.append(_conform_record(key, label=f"{pred_dir}/{document}.json", plan=plan))
    return lines


def _confidence_sweep_floor_note(digest: str, calibration_curve_identity) -> Callable[[dict], str]:
    """A note naming the floor a content-derived key suffers once its stamped body changes."""

    def note(stripped: dict) -> str:
        recomputed = calibration_curve_identity(stripped)
        if recomputed == digest:
            return ""
        return (
            f"; its own key is a content digest over its body, so it no longer matches digest "
            f"{digest!r} once schema_version is gone (recomputes to {recomputed!r}): any bucket "
            "naming this record as its calibration_evidence_key floors on its next read"
        )

    return note


def _conform_confidence_sweep(root: Path, *, plan: bool) -> list[str]:
    from tcip_mcp.tools.inference_tools import (
        CONFIDENCE_SWEEP_STORE, _CalibrationCurveLocator, calibration_curve_identity,
    )

    locator = _CalibrationCurveLocator()
    artifacts_dir = root / ".tcip" / "artifacts"
    lines: list[str] = []
    if not artifacts_dir.is_dir():
        return lines
    for path in sorted(artifacts_dir.iterdir()):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        parts = locator.parts_from(relative)
        if parts is None:
            continue
        (digest,) = parts
        key = Key(CONFIDENCE_SWEEP_STORE, str(root.resolve()), (digest,))
        label = f"confidence_sweep {digest}"
        try:
            ts.read(key, default=None)
        except ts.SchemaVersionRefused:
            lines.append(_conform_poisoned_record(
                key, label=label, plan=plan,
                note_fn=_confidence_sweep_floor_note(digest, calibration_curve_identity),
            ))
        except StoreError as exc:
            lines.append(f"{label}: refused, {exc}")
        else:
            lines.append(f"{label}: no schema_version 2 field, unchanged")
    return lines


def _name_log_lines(root: Path) -> list[str]:
    """Count, never touch, every ``audit_log``/``experiment_validations`` line still carrying
    ``schema_version: 2``: both are append-only, so a stray line is named, not rewritten."""
    from tcip_mcp.audit import AUDIT_LOG_STORE, audit_log_key
    from tcip_mcp.experiments import EXPERIMENT_VALIDATIONS_STORE, experiments_scope

    lines: list[str] = []
    try:
        page = ts.read_log(audit_log_key(root))
    except StoreError as exc:
        lines.append(f"{AUDIT_LOG_STORE}: log unreadable, {exc}")
    else:
        # A version-refused line never reaches page.records (read_log excludes it there and
        # only counts its position), so its own carried value is never inspected here.
        count = len(page.version_refused)
        lines.append(
            f"{AUDIT_LOG_STORE}: {count} line(s) refuse at a schema_version above the ceiling "
            "this reader knows, left untouched (append-only; dev-era logs are cleared by a "
            "separate step)" if count else
            f"{AUDIT_LOG_STORE}: no version-refused lines"
        )

    validation_keys = ts.keys(EXPERIMENT_VALIDATIONS_STORE, experiments_scope(root))
    if not validation_keys:
        lines.append(f"{EXPERIMENT_VALIDATIONS_STORE}: no experiments to check")
    for key in sorted(validation_keys, key=lambda k: k.parts):
        experiment_id = key.parts[0] if key.parts else "<unknown experiment>"
        try:
            page = ts.read_log(key)
        except StoreError as exc:
            lines.append(f"{EXPERIMENT_VALIDATIONS_STORE} {experiment_id}: log unreadable, {exc}")
            continue
        count = len(page.version_refused)
        if count:
            lines.append(
                f"{EXPERIMENT_VALIDATIONS_STORE} {experiment_id}: {count} line(s) refuse at a "
                "schema_version above the ceiling this reader knows, left untouched "
                "(append-only; dev-era logs are cleared by a separate step)"
            )
    return lines


def check_root(root: Path, *, plan: bool = False) -> tuple[list[str], bool]:
    """Every outcome line for ``root``, and whether any target refused outright."""
    if not (root / ".tcip").is_dir():
        return ["refused, no .tcip directory found; not a project root"], True

    lines: list[str] = []
    refused = False

    registry_line = _conform_registry_index(root, plan=plan)
    lines.append(registry_line)
    refused = refused or "refused" in registry_line

    sidecar_lines = _conform_sidecars(root, plan=plan)
    lines.extend(sidecar_lines)
    refused = refused or any("refused" in line for line in sidecar_lines)

    sweep_lines = _conform_confidence_sweep(root, plan=plan)
    lines.extend(sweep_lines)
    refused = refused or any("refused" in line for line in sweep_lines)

    lines.extend(_name_log_lines(root))
    return lines, refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    any_refused = False
    for root in args.roots:
        root = root.resolve()
        lines, refused = check_root(root, plan=args.plan)
        if refused:
            any_refused = True
        for line in lines:
            print(f"{root}: {line}")

    return 2 if any_refused else 0


if __name__ == "__main__":
    sys.exit(main())

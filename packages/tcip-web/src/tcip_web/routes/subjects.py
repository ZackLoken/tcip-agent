"""Subject registry routes.

The dataset's subject registry is a single nested ``<dataset_root>/subjects.json`` describing every
subject, its attributes, and their value names: never integer ids or colors (a label references
these names; an id is a per-training-run artifact and a color is GUI-local). Shape, with ``tree``
as an example subject::

    {
      "tree":      {"description": "one tree crown"},
      "<subject>": {"description": "...",
                    "attributes": {"<attribute>": {"type": "categorical",
                                                    "values": ["<value1>", "<value2>"]}}}
    }

Read/written through :mod:`tcip_mcp.subject_registry`. The registry travels with the image set: a
name-based label is undecodable without it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_store import StoreError

from tcip_mcp.dataset_layout import IMAGE_STATUSES, label_filename
from tcip_web.identity import resolve_user, user_id
from tcip_web.label_annotations_cache import cached_label_annotations
from tcip_web.paths import allowed_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


def _audit_dataset_write(dataset_root: str, tool: str, arguments: dict) -> None:
    """Record a dataset-native GUI mutation in that dataset's own audit log.

    A failed append raises ``AuditEntryNotWritten``: the mutation has already committed by the time
    this runs.
    """
    from tcip_web.routes.audit_gap import record_committed

    record_committed(tool, arguments, scope=dataset_root)


def _subjects_in_dir(d: Path) -> tuple[set[str], list[str]]:
    """Distinct subject names present in a dir's per-image label files, and the paths that would
    not read: one bad file costs its own name, never the whole scan."""
    from tcip_annotation.json_io import UnreadableLabelDocument, prediction_documents

    subjects: set[str] = set()
    unreadable: list[str] = []
    for jf in prediction_documents(d):
        try:
            annotations = cached_label_annotations(jf)
        except UnreadableLabelDocument:
            unreadable.append(str(jf))
            continue
        for record in annotations:
            subjects.add(record.subject)
    return subjects, unreadable


@router.get("/load")
def load_subjects(dataset_root: str, annotations_dir: Optional[str] = None) -> dict:
    """Load the subject registry of the dataset at ``dataset_root``.

    Resolution: the dataset's saved ``subjects.json`` -> else a draft registry of the subjects
        actually present in the labels under ``annotations_dir`` (detection-only, no attributes)
        -> else empty. Returns ``{"subjects": <nested registry mapping>, "version": <token> |
        None, "unreadable": [paths]}``: ``version`` is the stored registry's compare-and-set token
        when one was saved, else ``None``; ``unreadable`` names every per-image label file under
        ``annotations_dir`` that would not read, scanned whether or not a registry is saved, and
        left out of a draft subject scan. A save posting this ``version`` back is refused with 409
        if the stored registry has moved on since; a save posting ``None`` asserts no registry is
        stored and is refused with 409 when one is.
    """
    from tcip_mcp.subject_registry import (
        RegistryError,
        Subject,
        SubjectRegistry,
        read_versioned_registry,
        registry_to_dict,
    )

    guarded_dir = str(allowed_path(annotations_dir)) if annotations_dir else None

    subjects: set[str] = set()
    unreadable: list[str] = []
    if guarded_dir and Path(guarded_dir).is_dir():
        subjects, unreadable = _subjects_in_dir(Path(guarded_dir))

    try:
        registry, version = read_versioned_registry(allowed_path(dataset_root))
    except FileNotFoundError:
        pass
    except (OSError, RegistryError) as exc:
        raise HTTPException(500, f"could not parse the subject registry: {exc}") from exc
    else:
        return {"subjects": registry_to_dict(registry), "version": version.token,
                "unreadable": unreadable}

    if subjects:
        reg = SubjectRegistry(subjects=tuple(Subject(name=s) for s in sorted(subjects)))
        return {"subjects": registry_to_dict(reg), "version": None, "unreadable": unreadable}
    return {"subjects": {}, "version": None, "unreadable": unreadable}


class SaveSubjectsPayload(BaseModel):
    subjects: dict  # the nested registry mapping (subjects -> attributes -> values)
    dataset_root: str
    # Required: the version load_subjects returned beside the registry this save was built from.
    # None means the registry was absent at load, asserted as Version.ABSENT, never skipped.
    version: Optional[str]


@router.post("/save")
def save_subjects(payload: SaveSubjectsPayload) -> dict:
    """Write the dataset's subject registry through :func:`subject_registry.replace_registry`.

    Refuses (400) a write dropping a subject, attribute or attribute value the stored registry
    declares, naming what it would have lost. Also refuses (400) a same-values attribute type
    change (categorical to ordinal or back): this route passes neither ``allow_type_changes`` nor
    ``allow_removals``; ``write_subject_registry`` states either. Refuses (409) a stale
    ``version``. Once the write lands, the outgoing digest is recorded onto the changed subject's
    still-unstamped confirmations; what it stamped, the confirmations that now predate the
    vocabulary in effect, and any warning, ride back in ``schema_change_sweep``. A write whose
    audit line could not follow answers 409.
    """
    from tcip_store import Version, VersionConflict

    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_mcp.subject_registry import RegistryError, registry_from_dict, replace_registry
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        registry = registry_from_dict(payload.subjects)
    except RegistryError as exc:
        raise HTTPException(400, f"invalid subject registry: {exc}") from exc
    dataset_root = allowed_path(payload.dataset_root)
    expect = Version(payload.version) if payload.version is not None else Version.ABSENT
    try:
        committed = replace_registry(dataset_root, registry, expect=expect)
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    except VersionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, {"status": "ok", **exc.arguments}) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write {dataset_root}'s registry: {exc}") from exc
    if committed["schema_change_sweep"]["warning"]:
        logger.warning("%s", committed["schema_change_sweep"]["warning"])
    return {"status": "ok", **committed}


# ── Per-image status (used by Complete checkbox + status filter) ─────────


class ImageStatusPayload(BaseModel):
    image_name: str
    status: str  # "complete" | "partial" | "negative" | "unannotated"
    subject: str | None = None  # the object a Complete is scoped to (not necessarily a trait)
    date: str | None = None
    dataset_root: str
    # GUI-set identity (bare name), recorded as "user:<name>" against each status this write sets.
    user: Optional[str] = None


def _load_status_store(dataset_root: str) -> dict[str, dict[str, str]]:
    """The dataset's status store, normalized. Absence is an empty store; a store that will not
    decode is a 500.
    """
    from tcip_store import DecodeError, read

    from tcip_mcp.dataset_layout import image_status_key, status_tokens

    try:
        return status_tokens(read(image_status_key(dataset_root), default={}))
    except DecodeError as exc:
        raise HTTPException(500, f"the image status store under {dataset_root} "
                                 f"does not decode: {exc}") from exc


def _bucket(subject: str | None, date: str | None) -> str:
    from tcip_mcp.dataset_layout import status_bucket

    return status_bucket(subject or "", date)


def _require_bucket(subject: str | None, date: str | None) -> str:
    """``_bucket``, but an image-status write (single or bulk) must name a real subject or fail."""
    if not subject:
        raise HTTPException(400, "cannot record image status with no subject; pass a subject")
    return _bucket(subject, date)


def _stamp_digest(dataset_root: str, bucket: str, subject: str,
                  image_names: Iterable[str]) -> bool:
    """Record the subject's current attribute-schema digest against each of ``image_names``, and
    answer whether the stamp landed.

    Never blocks the status write: no ``subjects.json``, a registry that does not declare the
    subject, an unreadable registry or a failure writing the sidecar leaves these images unstamped
    (admitted, not quarantined, on read; see ``stale_finished_names``) and answers ``False``.
    """
    from tcip_mcp.subject_registry import attribute_schema_digest, read_registry
    from tcip_mcp.dataset_layout import stamp_image_status_digests, subjects_path

    if not subjects_path(dataset_root).is_file():
        return False
    try:
        digest = attribute_schema_digest(read_registry(dataset_root), subject)
        if digest is None:
            return False
        stamp_image_status_digests(dataset_root, bucket, image_names, digest)
    except (OSError, ValueError, StoreError):
        # ValueError covers json.JSONDecodeError and subject_registry.RegistryError (its subclass).
        logger.warning("could not stamp the attribute-schema digest for %s", bucket, exc_info=True)
        return False
    return True


@router.get("/image_status")
def get_image_status(dataset_root: str, subject: str | None = None,
                     date: str | None = None) -> dict:
    """Statuses for one subject/date of the dataset at ``dataset_root``, plus which finished ones
    (complete or negative) are stale under the subject's current attribute schema
    (``stale_definition``, sorted names)."""
    from tcip_mcp.pipelines.data.label_queries import stale_finished_names

    root = str(allowed_path(dataset_root))
    statuses = _load_status_store(root).get(_bucket(subject, date), {})
    stale = stale_finished_names(root, subject=subject, date=date)
    return {"statuses": statuses, "stale_definition": sorted(stale)}


@router.post("/image_status")
def set_image_status(payload: ImageStatusPayload) -> dict:
    if payload.status not in IMAGE_STATUSES:
        raise HTTPException(400, f"invalid status: {payload.status}")
    from tcip_mcp.dataset_layout import record_image_statuses

    root = str(allowed_path(payload.dataset_root))
    bucket = _require_bucket(payload.subject, payload.date)
    record_image_statuses(root, bucket, {payload.image_name: payload.status},
                          recorded_by=user_id(resolve_user(payload.user)))
    assert payload.subject, "_require_bucket refused a status write naming no subject"
    stamped = _stamp_digest(root, bucket, payload.subject, [payload.image_name])
    committed = {"status": "ok", "digest_stamped": stamped}
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        _audit_dataset_write(
            root,
            "gui_set_image_status",
            {"image_name": payload.image_name, "status": payload.status,
             "subject": payload.subject, "date": payload.date},
        )
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, committed) from exc
    return committed


class ImageStatusBulkPayload(BaseModel):
    statuses: dict[str, str]  # image_name → status
    subject: str | None = None
    date: str | None = None
    dataset_root: str
    # GUI-set identity (bare name), recorded as "user:<name>" against each status this write sets.
    user: Optional[str] = None


@router.post("/image_status/bulk")
def set_image_status_bulk(payload: ImageStatusBulkPayload) -> dict:
    from tcip_mcp.dataset_layout import record_image_statuses

    root = str(allowed_path(payload.dataset_root))
    bucket = _require_bucket(payload.subject, payload.date)
    applied = {name: st for name, st in payload.statuses.items() if st in IMAGE_STATUSES}
    if applied:
        record_image_statuses(root, bucket, applied,
                              recorded_by=user_id(resolve_user(payload.user)))
    assert payload.subject, "_require_bucket refused a status write naming no subject"
    stamped = _stamp_digest(root, bucket, payload.subject, applied)
    not_stamped = [] if stamped else sorted(applied)
    committed = {"status": "ok", "n": len(payload.statuses), "digest_unstamped": not_stamped}
    # Record what was actually written, not the raw payload: an entry whose status was skipped
    # would overstate the change, and a no-op write logged as a mutation is noise.
    if applied:
        from tcip_mcp.audit import AuditEntryNotWritten
        from tcip_web.routes.audit_gap import audit_gap_409

        try:
            _audit_dataset_write(
                root,
                "gui_set_image_status_bulk",
                {"statuses": applied, "subject": payload.subject, "date": payload.date},
            )
        except AuditEntryNotWritten as exc:
            raise audit_gap_409(exc, committed) from exc
    return committed


class DerivePayload(BaseModel):
    annotations_dir: Optional[str] = None
    subject: str
    image_list: list[str]
    complete_override: list[str] = []


@router.post("/image_status/derive")
def derive_image_status(payload: DerivePayload) -> dict:
    """Compute initial per-image status from the per-image label files.

    The mapping itself is ``dataset_layout.derive_status``, with ``has_content`` scoped to
    ``subject`` through ``annotations_hold_subject``. An image whose label file would not read is
    left out of ``statuses`` and its label document's path is reported in ``unreadable`` instead.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument

    from tcip_mcp.dataset_layout import annotations_hold_subject, derive_status

    guarded_dir = (str(allowed_path(payload.annotations_dir)) if payload.annotations_dir
                   else None)
    adir = Path(guarded_dir) if guarded_dir else None
    complete_set = set(payload.complete_override)

    statuses: dict[str, str] = {}
    unreadable: list[str] = []
    for name in payload.image_list:
        stem = name.rsplit(".", 1)[0]
        has_any = False
        if adir:
            label_path = adir / label_filename(stem)
            try:
                annotations = cached_label_annotations(label_path)
            except UnreadableLabelDocument:
                unreadable.append(str(label_path))
                continue
            has_any = annotations_hold_subject(annotations, payload.subject)
        statuses[name] = derive_status(completed=name in complete_set, has_content=has_any)

    return {"statuses": statuses, "unreadable": unreadable}

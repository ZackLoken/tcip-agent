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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


def _guard_dataset_root(root: str) -> str:
    """Confine a resolved dataset root to the allowed roots
    (:func:`tcip_web.paths.assert_path_allowed`).
    """
    from tcip_web.paths import assert_path_allowed

    try:
        return str(assert_path_allowed(root))
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc


def _resolve_dataset_root(dataset_root: str | None, annotations_dir: str | None) -> str | None:
    """The dataset root, taken from ``dataset_root`` or derived from a per-image label dir path."""
    if dataset_root:
        return _guard_dataset_root(dataset_root)
    from tcip_mcp.dataset_layout import dataset_root_of

    if annotations_dir and (root := dataset_root_of(annotations_dir)) is not None:
        return _guard_dataset_root(str(root))
    return None


def _audit_dataset_write(dataset_root: str, tool: str, arguments: dict) -> None:
    """Record a dataset-native GUI mutation (this module's own ``image_status.json`` and
    ``subjects.json`` writes) in that dataset's own audit log.

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
def load_subjects(
    project_root: str,
    dataset_root: Optional[str] = None,
    annotations_dir: Optional[str] = None,
) -> dict:
    """Load the dataset's nested subject registry.

    Resolution: the dataset's saved ``subjects.json`` -> else a draft registry of the subjects
        actually present in the labels (detection-only, no attributes) -> else empty. Returns
        ``{"subjects": <nested registry mapping>, "version": <token> | None, "unreadable":
        [paths]}``: ``version`` is the stored registry's compare-and-set token when one was saved,
        else ``None``; ``unreadable`` names every per-image label file under ``annotations_dir``
        that would not read, scanned whether or not a registry is saved, and left out of a draft
        subject scan. A save posting this ``version`` back is refused with 409 if the stored
        registry has moved on since; a save posting ``None`` is an unconditional write.
    """
    from tcip_mcp.subject_registry import (
        RegistryError,
        Subject,
        SubjectRegistry,
        read_registry,
        read_version,
        registry_to_dict,
    )
    from tcip_mcp.dataset_layout import subjects_path

    guarded_dir = _guard_dataset_root(annotations_dir) if annotations_dir else None
    root = _resolve_dataset_root(dataset_root, annotations_dir)

    subjects: set[str] = set()
    unreadable: list[str] = []
    if guarded_dir and Path(guarded_dir).is_dir():
        subjects, unreadable = _subjects_in_dir(Path(guarded_dir))

    if root:
        p = subjects_path(root)
        if p.exists():
            try:
                registry = read_registry(p)
            except (OSError, RegistryError) as exc:
                raise HTTPException(500, f"could not parse {p}: {exc}") from exc
            return {"subjects": registry_to_dict(registry), "version": read_version(p).token,
                    "unreadable": unreadable}

    if subjects:
        reg = SubjectRegistry(subjects=tuple(Subject(name=s) for s in sorted(subjects)))
        return {"subjects": registry_to_dict(reg), "version": None, "unreadable": unreadable}
    return {"subjects": {}, "version": None, "unreadable": unreadable}


class SaveSubjectsPayload(BaseModel):
    project_root: str
    subjects: dict  # the nested registry mapping (subjects -> attributes -> values)
    dataset_root: Optional[str] = None
    annotations_dir: Optional[str] = None
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
    vocabulary in effect, and any warning, ride back in ``schema_change_sweep``.
    """
    from tcip_store import Version, VersionConflict

    from tcip_mcp.subject_registry import RegistryError, registry_from_dict, replace_registry
    from tcip_mcp.dataset_layout import subjects_path

    root = _resolve_dataset_root(payload.dataset_root, payload.annotations_dir)
    if not root:
        raise HTTPException(400, "cannot locate the dataset to save the subject registry into; "
                                 "pass dataset_root or an annotations dir")
    try:
        registry = registry_from_dict(payload.subjects)
    except RegistryError as exc:
        raise HTTPException(400, f"invalid subject registry: {exc}") from exc
    path = subjects_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    expect = Version(payload.version) if payload.version is not None else Version.ABSENT
    try:
        result = replace_registry(path, registry, expect=expect)
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    except VersionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write {path}: {exc}") from exc
    sweep = result["schema_change_sweep"]
    if sweep["warning"]:
        logger.warning("%s", sweep["warning"])
    committed = {"status": "ok", "n_subjects": len(registry.subjects), "subjects_path": str(path),
                 "version": result["version"].token, "schema_change_sweep": sweep}
    from tcip_mcp.audit import AuditEntryNotWritten
    from tcip_web.routes.audit_gap import audit_gap_409

    try:
        _audit_dataset_write(
            root, "gui_save_subjects",
            {"subjects_path": str(path), "n_subjects": len(registry.subjects),
             "confirmations_stamped_with_outgoing_schema": sweep["newly_stamped"],
             "confirmations_predating_vocabulary": sweep["predating_vocabulary"]},
        )
    except AuditEntryNotWritten as exc:
        raise audit_gap_409(exc, committed) from exc
    return committed


# ── Per-image status (used by Complete checkbox + status filter) ─────────


class ImageStatusPayload(BaseModel):
    project_root: str
    image_name: str
    status: str  # "complete" | "partial" | "negative" | "unannotated"
    subject: str | None = None  # the object a Complete is scoped to (not necessarily a trait)
    date: str | None = None
    dataset_root: Optional[str] = None
    annotations_dir: Optional[str] = None
    # GUI-set identity (bare name), recorded as "user:<name>" against each status this write sets.
    user: Optional[str] = None


def _require_dataset_root(dataset_root: str | None, annotations_dir: str | None) -> str:
    """``_resolve_dataset_root``, but a write must locate the dataset or fail."""
    root = _resolve_dataset_root(dataset_root, annotations_dir)
    if not root:
        raise HTTPException(400, "cannot locate the dataset to record image status against; "
                                 "pass dataset_root or an annotations dir")
    return root


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


def _stamp_digest(dataset_root: str, bucket: str, subject: str | None,
                  image_names: Iterable[str]) -> bool | None:
    """Record the subject's current attribute-schema digest against each of ``image_names``.

    Never blocks the status write: an unreadable registry, an absent one, or a failure writing the
    sidecar itself leaves these images unstamped (admitted, not quarantined, on read; see
    ``stale_finished_names``). Returns ``None`` when there was nothing to stamp (no subject, no
    ``subjects.json``, an unreadable or subject-less registry); ``True`` once the stamp lands, and
    ``False`` only when the write itself raised.
    """
    if not subject:
        return None
    from tcip_mcp.subject_registry import attribute_schema_digest, read_registry
    from tcip_mcp.dataset_layout import stamp_image_status_digests, subjects_path

    cp = subjects_path(dataset_root)
    if not cp.is_file():
        return None
    try:
        digest = attribute_schema_digest(read_registry(cp), subject)
    except (OSError, ValueError):
        # ValueError covers json.JSONDecodeError and subject_registry.RegistryError (its subclass).
        return None
    if digest is None:
        return None
    try:
        stamp_image_status_digests(dataset_root, bucket, image_names, digest)
    except (OSError, StoreError):
        logger.warning("could not stamp the attribute-schema digest for %s", bucket, exc_info=True)
        return False
    return True


@router.get("/image_status")
def get_image_status(project_root: str, subject: str | None = None, date: str | None = None,
                     dataset_root: str | None = None, annotations_dir: str | None = None) -> dict:
    """Statuses for one subject/date, plus which finished ones (complete or negative) are stale
    under the subject's current attribute schema (``stale_definition``, sorted names)."""
    from tcip_mcp.pipelines.data.label_queries import stale_finished_names

    root = _resolve_dataset_root(dataset_root, annotations_dir)
    if not root:
        return {"statuses": {}, "stale_definition": []}
    statuses = _load_status_store(root).get(_bucket(subject, date), {})
    stale = stale_finished_names(root, subject=subject, date=date)
    return {"statuses": statuses, "stale_definition": sorted(stale)}


@router.post("/image_status")
def set_image_status(payload: ImageStatusPayload) -> dict:
    if payload.status not in IMAGE_STATUSES:
        raise HTTPException(400, f"invalid status: {payload.status}")
    from tcip_mcp.dataset_layout import record_image_statuses

    root = _require_dataset_root(payload.dataset_root, payload.annotations_dir)
    bucket = _require_bucket(payload.subject, payload.date)
    record_image_statuses(root, bucket, {payload.image_name: payload.status},
                          recorded_by=user_id(resolve_user(payload.user)))
    # Nothing to stamp (no subject in the registry, say) is not a failed write: only a stamp
    # attempt that actually raised reads back as unstamped.
    stamped = _stamp_digest(root, bucket, payload.subject, [payload.image_name])
    digest_stamped = stamped is not False
    committed = {"status": "ok", "digest_stamped": digest_stamped}
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
    project_root: str
    statuses: dict[str, str]  # image_name → status
    subject: str | None = None
    date: str | None = None
    dataset_root: Optional[str] = None
    annotations_dir: Optional[str] = None
    # GUI-set identity (bare name), recorded as "user:<name>" against each status this write sets.
    user: Optional[str] = None


@router.post("/image_status/bulk")
def set_image_status_bulk(payload: ImageStatusBulkPayload) -> dict:
    from tcip_mcp.dataset_layout import record_image_statuses

    root = _require_dataset_root(payload.dataset_root, payload.annotations_dir)
    bucket = _require_bucket(payload.subject, payload.date)
    applied = {name: st for name, st in payload.statuses.items() if st in IMAGE_STATUSES}
    if applied:
        record_image_statuses(root, bucket, applied,
                              recorded_by=user_id(resolve_user(payload.user)))
    stamped = _stamp_digest(root, bucket, payload.subject, applied)
    # Nothing to stamp (None) is not a failed write; only an actual write failure (False) names
    # the applied statuses as unstamped.
    not_stamped = sorted(applied) if stamped is False else []
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
    project_root: str
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

    # An absolute-path read needs its own confinement (no-op unless TCIP_IMAGE_ROOTS is set).
    guarded_dir = _guard_dataset_root(payload.annotations_dir) if payload.annotations_dir else None
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

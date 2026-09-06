"""Class registry routes.

The dataset's class registry is a single nested ``<dataset_root>/classes.json`` describing every
subject, its attributes, and their value names: never integer ids or colors (a label references
these names; an id is a per-training-run artifact and a color is GUI-local). Shape::

    {
      "bush":      {"description": "one plant crown"},
      "<subject>": {"description": "...",
                    "attributes": {"<attribute>": {"type": "categorical",
                                                    "values": ["<value1>", "<value2>"]}}}
    }

Read/written through :mod:`tcip_mcp.class_registry` (the one registry authority), so the GUI and the
agent tools agree by construction. The registry travels with the image set: a name-based label is
undecodable without it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_store import StoreError

from tcip_mcp.dataset_layout import IMAGE_STATUSES
from tcip_web.identity import resolve_user, user_id
from tcip_web.label_annotations_cache import cached_label_annotations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classes", tags=["classes"])


def _guard_dataset_root(root: str) -> str:
    """Confine a resolved dataset root to the allowed image roots (no-op unless ``TCIP_IMAGE_ROOTS``
    is set): the same lockdown the rest of the backend applies to absolute reads. The single choke
    point every ``classes.py`` route resolves through, so a new caller can't forget it the way a
    route-local guard could."""
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
    """Record a dataset-native GUI mutation in that dataset's own audit log: this module's own
    ``image_status.json`` and ``classes.json`` writes, and ``inference.py``'s prediction writes.

    All three are dataset-native, not project-private (a dataset can be opened by more than one
    project, see ``dataset_layout.image_status_path`` and ``dataset_layout.dataset_root_of``), so
    there is no single project's audit log a write here unambiguously belongs to. Colocating the
    trail with the state it describes, rather than guessing a project, is deliberate. Best-effort:
    it never fails the request that triggered the write.
    """
    if not dataset_root:
        return
    from tcip_mcp.audit import record_event

    record_event(tool, arguments, source="gui", scope=dataset_root)


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
def load_classes(
    project_root: str,
    dataset_root: Optional[str] = None,
    annotations_dir: Optional[str] = None,
) -> dict:
    """Load the dataset's nested class registry.

    Resolution: the dataset's saved ``classes.json`` -> else a draft registry of the subjects
    actually present in the labels (detection-only, no attributes) -> else empty. Returns
    ``{"subjects": <nested registry mapping>, "version": <token> | None, "unreadable": [paths]}``:
    ``version`` is the stored registry's compare-and-set token when one was saved, else ``None`` (a
    draft or empty registry names nothing to assert against); ``unreadable`` names every
    per-image label file under ``annotations_dir`` that would not read, scanned whether or not a
    registry is saved (a saved registry answers the subject list on its own, but a corrupt label
    file is still worth surfacing to the breeder), and left out of a draft subject scan
    rather than aborting it. A save posting this ``version`` back is refused with 409 if the stored
    registry has moved on since; a save posting ``None`` is an unconditional write, since it names
    no version to assert against.
    """
    from tcip_mcp.class_registry import (
        ClassRegistry,
        RegistryError,
        Subject,
        read_registry,
        read_version,
        registry_to_dict,
    )
    from tcip_mcp.dataset_layout import classes_path

    guarded_dir = _guard_dataset_root(annotations_dir) if annotations_dir else None
    root = _resolve_dataset_root(dataset_root, annotations_dir)

    subjects: set[str] = set()
    unreadable: list[str] = []
    if guarded_dir and Path(guarded_dir).is_dir():
        subjects, unreadable = _subjects_in_dir(Path(guarded_dir))

    if root:
        p = classes_path(root)
        if p.exists():
            try:
                registry = read_registry(p)
            except (OSError, RegistryError) as exc:
                raise HTTPException(500, f"could not parse {p}: {exc}") from exc
            return {"subjects": registry_to_dict(registry), "version": read_version(p).token,
                    "unreadable": unreadable}

    if subjects:
        reg = ClassRegistry(subjects=tuple(Subject(name=s) for s in sorted(subjects)))
        return {"subjects": registry_to_dict(reg), "version": None, "unreadable": unreadable}
    return {"subjects": {}, "version": None, "unreadable": unreadable}


class SaveClassesPayload(BaseModel):
    project_root: str
    subjects: dict  # the nested registry mapping (subjects -> attributes -> values)
    dataset_root: Optional[str] = None
    annotations_dir: Optional[str] = None
    # Required: the version load_classes returned beside the registry this save was built from.
    # None means the registry was absent at load, asserted as Version.ABSENT, never skipped.
    version: Optional[str]


@router.post("/save")
def save_classes(payload: SaveClassesPayload) -> dict:
    """Write the dataset's class registry through :func:`class_registry.replace_registry`.

    Refuses (400) a write dropping a subject, attribute or attribute value the stored registry
    declares: the GUI's own save is additive by construction (see ``AnnotateToolbar``), so a drop
    arriving here means the browser held a stale registry, and the refusal names what it would
    have lost. Also refuses (400) a same-values attribute type change (categorical to ordinal or
    back): this route never passes ``allow_type_changes`` (nor ``allow_removals``) to
    :func:`~tcip_mcp.class_registry.replace_registry`, so the GUI has no door for either and
    always refuses; a deliberate flip is stated through ``write_class_map`` instead. Refuses
    (409) a stale ``version``. Changing a subject's attribute vocabulary
    invalidates the confirmations made under the old one, so once the write lands the outgoing
    digest is recorded onto that subject's still-unstamped confirmations; they then read as
    predating the change instead of as made under the new vocabulary. That never blocks the write
    it accompanies; what it stamped, the confirmations that now predate the vocabulary in effect
    (a confirmation already stamped, whether by this write or an earlier one, whose digest is not
    the subject's new digest), and any warning, ride back in ``schema_change_sweep``.
    """
    from tcip_store import Version, VersionConflict

    from tcip_mcp.class_registry import RegistryError, registry_from_dict, replace_registry
    from tcip_mcp.dataset_layout import classes_path

    root = _resolve_dataset_root(payload.dataset_root, payload.annotations_dir)
    if not root:
        raise HTTPException(400, "cannot locate the dataset to save the class registry into; "
                                 "pass dataset_root or an annotations dir")
    try:
        registry = registry_from_dict(payload.subjects)
    except RegistryError as exc:
        raise HTTPException(400, f"invalid class registry: {exc}") from exc
    path = classes_path(root)
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
    _audit_dataset_write(
        root, "gui_save_classes",
        {"classes_path": str(path), "n_subjects": len(registry.subjects),
         "confirmations_stamped_with_outgoing_schema": sweep["newly_stamped"],
         "confirmations_predating_vocabulary": sweep["predating_vocabulary"]},
    )
    return {"status": "ok", "n_subjects": len(registry.subjects), "classes_path": str(path),
            "version": result["version"].token, "schema_change_sweep": sweep}


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
    """``_resolve_dataset_root``, but a write must locate the dataset or fail loudly (mirrors
    ``save_classes``): a silent fallback would write a human's Complete nowhere anyone reads it."""
    root = _resolve_dataset_root(dataset_root, annotations_dir)
    if not root:
        raise HTTPException(400, "cannot locate the dataset to record image status against; "
                                 "pass dataset_root or an annotations dir")
    return root


def _load_status_store(dataset_root: str) -> dict[str, dict[str, str]]:
    """The dataset's status store, normalized. Absence is an empty store; a store that will not
    decode is a 500 rather than an empty answer, because reading it as empty would tell the
    breeder their confirmations are gone."""
    from tcip_store import DecodeError, read

    from tcip_mcp.dataset_layout import image_status_key, normalize_status_store

    try:
        return normalize_status_store(read(image_status_key(dataset_root), default={}))
    except DecodeError as exc:
        raise HTTPException(500, f"the image status store under {dataset_root} "
                                 f"does not decode: {exc}") from exc


def _bucket(subject: str | None, date: str | None) -> str:
    from tcip_mcp.dataset_layout import status_bucket

    return status_bucket(subject or "", date)


def _require_bucket(subject: str | None, date: str | None) -> str:
    """``_bucket``, but a deliberate image-status write (single or bulk) must be scoped to a real
    subject or fail loudly (mirrors ``_require_dataset_root``): the "" bucket a missing subject
    silently falls back to is one ``get_image_status`` never returns anything meaningful for, so
    the write would land nowhere anyone reads it back from."""
    if not subject:
        raise HTTPException(400, "cannot record image status with no subject; pass a subject")
    return _bucket(subject, date)


def _stamp_digest(dataset_root: str, bucket: str, subject: str | None,
                  image_names: Iterable[str]) -> bool | None:
    """Record the subject's current attribute-schema digest against each of ``image_names``, so a
    later read can tell a confirmation made under a since-changed schema from one still valid.
    Never blocks the status write: an unreadable registry, an absent one, or a failure writing the
    sidecar itself just leaves these images unstamped (admitted, not quarantined, on read; see
    ``stale_finished_names``), because the status the human recorded is already committed by the
    time this runs. Returns ``None`` when there was nothing to stamp (no subject, no
    ``classes.json``, an unreadable or subject-less registry) -- not a failure, since no
    confirmation was ever asserted against a schema that says nothing about this subject --
    ``True`` once the stamp lands, and ``False`` only when the write itself raised, so a caller
    can tell a mark it is about to clear still describes reality."""
    if not subject:
        return None
    from tcip_mcp.class_registry import attribute_schema_digest, read_registry
    from tcip_mcp.dataset_layout import classes_path, stamp_image_status_digests

    cp = classes_path(dataset_root)
    if not cp.is_file():
        return None
    try:
        digest = attribute_schema_digest(read_registry(cp), subject)
    except (OSError, ValueError):
        # ValueError covers json.JSONDecodeError and class_registry.RegistryError (its subclass).
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
    _audit_dataset_write(
        root,
        "gui_set_image_status",
        {"image_name": payload.image_name, "status": payload.status,
         "subject": payload.subject, "date": payload.date},
    )
    return {"status": "ok", "digest_stamped": digest_stamped}


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
    # Record what was actually written, not the raw payload: an entry whose status was skipped
    # would overstate the change, and a no-op write logged as a mutation is noise.
    if applied:
        _audit_dataset_write(
            root,
            "gui_set_image_status_bulk",
            {"statuses": applied, "subject": payload.subject, "date": payload.date},
        )
    return {"status": "ok", "n": len(payload.statuses), "digest_unstamped": not_stamped}


class DerivePayload(BaseModel):
    project_root: str
    annotations_dir: Optional[str] = None
    subject: str
    image_list: list[str]
    complete_override: list[str] = []


@router.post("/image_status/derive")
def derive_image_status(payload: DerivePayload) -> dict:
    """Compute initial per-image status from the per-image label files.

    The mapping itself is ``dataset_layout.derive_status``, the same one the review tab's Complete
    goes through, so the two cannot disagree about what a Complete on an empty image means.
    ``has_content`` is scoped to ``subject`` through ``annotations_hold_subject``, the predicate
    the review route's own Complete derives its token through. An image whose label file would
    not read is left out of ``statuses`` and its label document's path is reported in
    ``unreadable`` instead: one bad file costs that image, never the whole batch.
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
            label_path = adir / f"{stem}.json"
            try:
                annotations = cached_label_annotations(label_path)
            except UnreadableLabelDocument:
                unreadable.append(str(label_path))
                continue
            has_any = annotations_hold_subject(annotations, payload.subject)
        statuses[name] = derive_status(completed=name in complete_set, has_content=has_any)

    return {"statuses": statuses, "unreadable": unreadable}

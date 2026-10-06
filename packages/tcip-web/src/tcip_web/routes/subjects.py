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

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tcip_web.paths import allowed_path

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


@router.get("/load")
def load_subjects(dataset_root: str, date: str) -> dict:
    """Load the subject registry of the dataset at ``dataset_root``.

    Returns ``{"subjects": <nested registry mapping> | None, "discovered": [names], "version":
    <token> | None, "unreadable": [stems]}``: ``subjects`` and ``version`` (its compare-and-set
    token) are the stored registry's, both ``None`` when none is stored; ``discovered`` and
    ``unreadable`` are :func:`~tcip_mcp.dataset_layout.capture_subjects` of capture ``date``,
    read whether or not a registry is stored. A save posting this ``version`` back is
    refused with 409 if the stored registry has moved on since; a save posting ``None`` asserts no
    registry is stored and is refused with 409 when one is.
    """
    from tcip_mcp.dataset_layout import capture_subjects
    from tcip_mcp.subject_registry import RegistryError, read_versioned_registry, registry_to_dict

    root = allowed_path(dataset_root)
    discovered, unreadable = capture_subjects(root, date)
    subjects, token = None, None
    try:
        registry, version = read_versioned_registry(root)
        subjects, token = registry_to_dict(registry), version.token
    except FileNotFoundError:
        pass
    except (OSError, RegistryError) as exc:
        raise HTTPException(500, f"could not parse the subject registry: {exc}") from exc
    return {"subjects": subjects, "discovered": discovered, "version": token,
            "unreadable": unreadable}


class SaveSubjectsPayload(BaseModel):
    subjects: dict  # the nested registry mapping (subjects -> attributes -> values)
    dataset_root: str
    # Required: the version load_subjects returned beside the registry this save was built from.
    # None means the registry was absent at load, asserted as Version.ABSENT, never skipped.
    version: Optional[str]
    user: str


@router.post("/save")
def save_subjects(payload: SaveSubjectsPayload) -> dict:
    """Write the dataset's subject registry through :func:`subject_registry.replace_registry`, by
    the person ``user`` names.

    Refuses (400) a write dropping a subject, attribute or attribute value the stored registry
    declares, naming what it would have lost. Also refuses (400) a same-values attribute type
    change (categorical to ordinal or back): this route passes neither ``allow_type_changes`` nor
    ``allow_removals``; ``write_subject_registry`` states either. Refuses (409) a stale
    ``version``. A write whose audit line could not follow answers 409.
    """
    from tcip_store import Version, VersionConflictError

    from tcip_mcp.audit import AuditEntryNotWrittenError
    from tcip_mcp.identity import actor
    from tcip_mcp.subject_registry import RegistryError, registry_from_request, replace_registry
    from tcip_web.routes.audit_gap import audit_gap_409

    person = actor(payload.user)
    try:
        registry = registry_from_request(payload.subjects)
    except RegistryError as exc:
        raise HTTPException(400, f"invalid subject registry: {exc}") from exc
    dataset_root = allowed_path(payload.dataset_root)
    expect = Version(payload.version) if payload.version is not None else Version.ABSENT
    try:
        committed = replace_registry(dataset_root, registry, expect=expect, actor=person)
    except RegistryError as exc:
        raise HTTPException(400, str(exc)) from exc
    except VersionConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    except AuditEntryNotWrittenError as exc:
        raise audit_gap_409(exc, {"status": "ok", **exc.arguments}) from exc
    except OSError as exc:
        raise HTTPException(500, f"could not write {dataset_root}'s registry: {exc}") from exc
    return {"status": "ok", **committed}

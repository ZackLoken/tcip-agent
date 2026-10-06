"""The project record: the one record every project carries, holding its identity and its site.

It holds ``id`` (minted once at creation, never derived from a path), ``display_name`` (the name
the picker shows, changed only by :func:`rename_project`) and ``site`` (the breeder's own name for
the orchard or station the project's plants stand in, never guessed from a directory name or a
filename).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import tcip_store
from tcip_store import Key, StoreError, Version, VersionConflictError

PROJECT_RECORD_STORE = "project_record"
_PROJECT_RECORD_PARTS = ("project",)

_RECORD_FIELDS = ("id", "display_name", "site")

_MAX_TEXT_LENGTH = 200
"""The most characters a site or display name holds; tentative, stated by no source."""


class ProjectRecordMissingError(Exception):
    """A project has no record yet."""


class ProjectRecordInvalidError(ValueError):
    """A project's record decodes but does not hold an id, a display name and a site."""


class SiteConflictError(ValueError):
    """A project already records a different site than the one just offered."""


def mint_id() -> str:
    """A fresh identity for a record minted once and kept: twelve hex characters of a uuid4."""
    return uuid.uuid4().hex[:12]


def project_record_key(project_path: str | Path) -> Key:
    """The project's own record, written compare-and-swap."""
    return Key(PROJECT_RECORD_STORE, str(project_path), _PROJECT_RECORD_PARTS)


def validate_text(label: str, value: object) -> str:
    """``value`` as it will be stored: stripped text, or the refusal naming ``label`` and exactly
    what is wrong.

    No case folding and no slug normalization. A non-string, an empty string (after stripping), one
    carrying any non-printable character (``str.isprintable``, which also catches non-breaking and
    zero-width characters a paste from a document can carry), or one over
    :data:`_MAX_TEXT_LENGTH` characters refuses by name.
    """
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} is empty (after stripping surrounding whitespace)")
    for offset, ch in enumerate(text):
        if not ch.isprintable():
            raise ValueError(
                f"{label} contains a non-printable character U+{ord(ch):04X} at offset {offset} "
                "of the stripped text"
            )
    if len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"{label} is {len(text)} characters long, over the {_MAX_TEXT_LENGTH}-character "
            "rendering bound (a picker line, a doctor finding, an audit argument)"
        )
    return text


def read_record(project_path: str | Path) -> dict:
    """The project's record: ``{"id", "display_name", "site"}``, each a non-empty string.

    Raises :class:`ProjectRecordMissingError` for a project with no record yet, and
    :class:`ProjectRecordInvalidError` for a document that decodes but does not hold all three. A
    document that does not decode raises the store's own ``DecodeError``, and a root the store
    refuses raises its ``StoreError``. Never creates a store.
    """
    return _checked(project_path, tcip_store.read(project_record_key(project_path), default=None))


def existing_project(value: str | Path) -> Path:
    """``value`` resolved, when it is a project whose record reads; otherwise ``ValueError``
    naming the path and what :func:`read_record` refused."""
    project = Path(value).resolve()
    problem = record_fields(project)["record_problem"]
    if problem is not None:
        raise ValueError(f"{project} names no readable project: {problem}")
    return project


def _checked(project_path: str | Path, raw: object) -> dict:
    """``raw``, a record read from ``project_path``, when it holds a non-empty id and a display
    name and a site each as :func:`validate_text` would store it. Raises
    ``ProjectRecordMissingError`` for an absent record and ``ProjectRecordInvalidError`` naming
    what the record lacks otherwise."""
    if raw is None:
        raise ProjectRecordMissingError(
            f"{project_path} is not a project yet: create it with initialize_project"
        )
    try:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not raw["id"]:
            raise ValueError("no id")
        for field, label in (("display_name", "display name"), ("site", "site")):
            if validate_text(label, raw.get(field)) != raw[field]:
                raise ValueError(f"the {label} is not stored as its text")
    except ValueError as exc:
        raise ProjectRecordInvalidError(
            f"{project_path}'s record does not hold an id, a display name and a site ({exc}; "
            f"found {raw!r})"
        ) from exc
    return raw


def create_record(project_path: str | Path, display_name: str, site: str) -> dict:
    """Write a new project's record with a freshly minted id, its one creator, and return it.

    A project already recording the same display name and site is returned as it stands, its id
    kept. A present record with a different site raises :class:`SiteConflictError`, and one with a
    different display name raises ``ValueError`` naming both; an unreadable one raises what
    :func:`read_record` raises.
    """
    name = validate_text("display name", display_name)
    text = validate_text("site", site)
    document = {"id": mint_id(), "display_name": name, "site": text}
    try:
        tcip_store.replace(project_record_key(project_path), document, expect=Version.ABSENT)
        return document
    except VersionConflictError:
        existing = read_record(project_path)
    if existing["site"] != text:
        raise SiteConflictError(
            f"{project_path} already records site {existing['site']!r}; the offered site "
            f"{text!r} does not match, so nothing was written. Run tcip write-project-site to "
            "correct it deliberately."
        )
    if existing["display_name"] != name:
        raise ValueError(
            f"{project_path} is already the project {existing['display_name']!r}; rename it "
            f"rather than creating it again as {name!r}"
        )
    return existing


def _update(project_path: str | Path, field: str, value: str) -> tuple[dict, str]:
    """Replace one field of a readable record under compare-and-swap; return the record written
    and the value it replaced."""
    key = project_record_key(project_path)
    while True:
        stored = tcip_store.read_versioned(key, default=None)
        current = _checked(project_path, stored.value)
        updated = {**current, field: value}
        try:
            tcip_store.replace(key, updated, expect=stored.version)
        except VersionConflictError:
            continue
        return updated, current[field]


def replace_site(project_path: str | Path, site: str) -> dict:
    """Correct the site a readable record holds; its id and display name are kept. Returns
    ``{"site", "previous_site"}``."""
    record, previous = _update(project_path, "site", validate_text("site", site))
    return {"site": record["site"], "previous_site": previous}


def rename_project(project: Path, display_name: str, *, actor: str) -> dict:
    """Change the project's display name and record one ``project_renamed`` line by ``actor`` in
    its own log; its directory and every other record are left as they are. Returns ``{"id",
    "display_name", "previous_display_name"}``. Refuses what :func:`validate_text` and
    :func:`read_record` refuse."""
    from tcip_mcp.audit import record_event_or_raise

    record, previous = _update(project, "display_name", validate_text("display name", display_name))
    record_event_or_raise(
        "project_renamed",
        {"id": record["id"], "display_name": record["display_name"],
         "previous_display_name": previous},
        actor=actor, scope=project,
    )
    return {"id": record["id"], "display_name": record["display_name"],
            "previous_display_name": previous}


def record_fields(project_path: str | Path) -> dict:
    """``{"id", "display_name", "site", "record_problem"}``: the record's three fields and ``None``,
    or three ``None`` beside the text of why the record would not read. Never raises."""
    try:
        record = read_record(project_path)
    except (ProjectRecordMissingError, ProjectRecordInvalidError, StoreError, OSError) as exc:
        return {**dict.fromkeys(_RECORD_FIELDS), "record_problem": str(exc)}
    return {**{field: record[field] for field in _RECORD_FIELDS}, "record_problem": None}

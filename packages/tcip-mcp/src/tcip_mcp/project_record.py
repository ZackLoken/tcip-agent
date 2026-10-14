"""The project record (:class:`ProjectRecord`): the one record every project carries, holding its
identity, its site and the capture invariances its trait's measurement holds under.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

import tcip_store
from tcip_store import Key, StoreError, Version, VersionConflictError

PROJECT_RECORD_STORE = "project_record"
_PROJECT_RECORD_PARTS = ("project",)

_MAX_TEXT_LENGTH = 200
"""The most characters a site or display name holds; tentative, stated by no source."""


class ProjectRecordMissingError(Exception):
    """A project has no record yet."""


class ProjectRecordInvalidError(ValueError):
    """A project's record decodes but is not a :class:`ProjectRecord`."""


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
        raise ValueError(
            f"{label} must be a string, got {type(value).__name__}"
        )
    text = value.strip()
    if not text:
        raise ValueError(
            f"{label} is empty (after stripping surrounding whitespace)"
        )
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


RotationDegrees = Annotated[float, Field(ge=0, le=180, strict=True)]
"""A rotation of the image about its center in degrees, at most 180, since a rotation by 180 in
either direction already reaches every orientation."""


class CaptureInvariances(BaseModel):
    """How far each capture transform leaves the project's trait measurement what it was, as the
    breeder states it when the project is created: whether it does, or for a rotation the
    largest angle at which it does.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, use_attribute_docstrings=True
    )
    horizontal_flip: bool
    """A left-right mirror of the image."""
    vertical_flip: bool
    """A top-bottom mirror of the image."""
    rotation: RotationDegrees
    """The largest rotation of the image about its center under which the measurement holds; 0
    for none, 180 for any."""
    color: bool
    """A shift in hue or saturation."""
    intensity: bool
    """A shift in brightness or contrast."""
    scale: bool
    """A change in the size the objects appear at in the frame."""


class ProjectRecord(BaseModel):
    """The project's record as stored: ``id``, minted once at creation; ``display_name``, the
    name the picker shows, changed only by :func:`rename_project`; ``site``, the breeder's own
    name for the orchard or station the project's plants stand in; ``invariances``, stated at
    creation. The display name and the site are each stored as :func:`validate_text` stores them.
    """

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    display_name: str
    site: str
    invariances: CaptureInvariances

    @field_validator("display_name", "site")
    @classmethod
    def _stored_as_text(cls, value: str, info: ValidationInfo) -> str:
        if validate_text(str(info.field_name), value) != value:
            raise ValueError("not stored as its text")
        return value


def read_record(project_path: str | Path) -> ProjectRecord:
    """The project's record.

    Raises :class:`ProjectRecordMissingError` for a project with no record yet, and
    :class:`ProjectRecordInvalidError` for a document that decodes but is not one. A document
    that does not decode raises the store's own ``DecodeError``, and a root the store refuses
    raises its ``StoreError``. Never creates a store.
    """
    return _checked(
        project_path,
        tcip_store.read(project_record_key(project_path), default=None),
    )


def existing_project(value: str | Path) -> tuple[Path, dict]:
    """``value`` resolved and its record's fields (:func:`record_fields`), when it is a project
    whose record reads; otherwise ``ValueError`` naming the path and what :func:`read_record`
    refused.
    """
    project = Path(value).resolve()
    fields = record_fields(project)
    if fields["record_problem"] is not None:
        raise ValueError(
            f"{project} names no readable project: {fields['record_problem']}"
        )
    return project, fields


def _checked(project_path: str | Path, raw: object) -> ProjectRecord:
    """``raw``, a record read from ``project_path``, when it is a :class:`ProjectRecord`. Raises
    ``ProjectRecordMissingError`` for an absent record and ``ProjectRecordInvalidError`` naming
    what the record lacks otherwise.
    """
    if raw is None:
        raise ProjectRecordMissingError(
            f"{project_path} is not a project yet: create it with initialize_project"
        )
    try:
        return ProjectRecord.model_validate(raw)
    except ValidationError as exc:
        raise ProjectRecordInvalidError(
            f"{project_path}'s record is not a project record ({exc}; found {raw!r})"
        ) from exc


def create_record(
    project_path: str | Path,
    display_name: str,
    site: str,
    invariances: CaptureInvariances,
) -> ProjectRecord:
    """Write a new project's record with a freshly minted id, its one creator, and return it.

    A project already recording the same display name, site and invariances is returned as it
    stands, its id kept. A present record with a different site raises
    :class:`SiteConflictError`, and one with a different display name or invariances raises
    ``ValueError`` naming both; an unreadable one raises what :func:`read_record` raises.
    """
    record = ProjectRecord(
        id=mint_id(),
        display_name=validate_text("display name", display_name),
        site=validate_text("site", site),
        invariances=invariances,
    )
    try:
        tcip_store.replace(
            project_record_key(project_path),
            record.model_dump(),
            expect=Version.ABSENT,
        )
        return record
    except VersionConflictError:
        existing = read_record(project_path)
    if existing.site != record.site:
        raise SiteConflictError(
            f"{project_path} already records site {existing.site!r}; the offered site "
            f"{record.site!r} does not match, so nothing was written. Run tcip "
            "write-project-site to correct it deliberately."
        )
    if existing.display_name != record.display_name:
        raise ValueError(
            f"{project_path} is already the project {existing.display_name!r}; rename it "
            f"rather than creating it again as {record.display_name!r}"
        )
    if existing.invariances != record.invariances:
        raise ValueError(
            f"{project_path} already records the capture invariances "
            f"{existing.invariances!r}; the offered {record.invariances!r} do not "
            "match, so nothing was written"
        )
    return existing


def _update(
    project_path: str | Path, field: str, value: str
) -> tuple[ProjectRecord, ProjectRecord]:
    """Replace one field of a readable record under compare-and-swap, the replacement
    validated as a :class:`ProjectRecord`; return the record written and the one it replaced.
    """
    key = project_record_key(project_path)
    while True:
        stored = tcip_store.read_versioned(key, default=None)
        current = _checked(project_path, stored.value)
        updated = ProjectRecord.model_validate(
            current.model_dump() | {field: value}
        )
        try:
            tcip_store.replace(
                key, updated.model_dump(), expect=stored.version
            )
        except VersionConflictError:
            continue
        return updated, current


def replace_site(project_path: str | Path, site: str) -> dict:
    """Correct the site a readable record holds; its id and display name are kept. Returns
    ``{"site", "previous_site"}``.
    """
    record, previous = _update(
        project_path, "site", validate_text("site", site)
    )
    return {"site": record.site, "previous_site": previous.site}


def rename_project(project: Path, display_name: str, *, actor: str) -> dict:
    """Change the project's display name and record one ``project_renamed`` line by ``actor`` in
    its own log; its directory and every other record are left as they are. Returns ``{"id",
    "display_name", "previous_display_name"}``. Refuses what :func:`validate_text` and
    :func:`read_record` refuse.
    """
    from tcip_mcp.audit import record_event_or_raise

    record, previous = _update(
        project, "display_name", validate_text("display name", display_name)
    )
    renamed = {
        "id": record.id,
        "display_name": record.display_name,
        "previous_display_name": previous.display_name,
    }
    record_event_or_raise(
        "project_renamed", renamed, actor=actor, scope=project
    )
    return renamed


def record_fields(project_path: str | Path) -> dict:
    """The record's fields (:class:`ProjectRecord`) and ``record_problem``: the fields and
    ``None``, or each field ``None`` beside the text of why the record would not read. Never
    raises.
    """
    try:
        record = read_record(project_path)
    except (
        ProjectRecordMissingError,
        ProjectRecordInvalidError,
        StoreError,
        OSError,
    ) as exc:
        return {
            **dict.fromkeys(ProjectRecord.model_fields),
            "record_problem": str(exc),
        }
    return {**record.model_dump(), "record_problem": None}

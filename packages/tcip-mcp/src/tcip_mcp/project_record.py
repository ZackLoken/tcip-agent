"""The project record (:class:`ProjectRecord`): the record a project carries.

It holds the project's identity, its site and the capture invariances its trait's measurement
holds under.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

import tcip_store
from tcip_annotation.text import Text
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
    """Return a fresh record identity: twelve hex characters of a uuid4."""
    return uuid.uuid4().hex[:12]


def project_record_key(project_path: str | Path) -> Key:
    """Return the store key of the project's own record."""
    return Key(PROJECT_RECORD_STORE, str(project_path), _PROJECT_RECORD_PARTS)


def _printable(text: str) -> str:
    """Return ``text`` when every character is printable, else raise ``ValueError``.

    Printable is ``str.isprintable``, which also refuses the non-breaking and zero-width
    characters a paste from a document can carry; the refusal names the first other character's
    code point and its offset.
    """
    for offset, ch in enumerate(text):
        if not ch.isprintable():
            raise ValueError(
                f"contains a non-printable character U+{ord(ch):04X} at offset {offset} "
                "of the stripped text"
            )
    return text


RecordText = Annotated[
    Text, Field(max_length=_MAX_TEXT_LENGTH), AfterValidator(_printable)
]
"""A site or display name as stored: stripped of surrounding whitespace, non-empty, at most
:data:`_MAX_TEXT_LENGTH` characters and every one printable; never case folded or slugged."""


RotationDegrees = Annotated[float, Field(ge=0, le=180, strict=True)]
"""A rotation of the image about its center in degrees, at most 180, since a rotation by 180 in
either direction already reaches every orientation."""


class CaptureInvariances(BaseModel):
    """How far each capture transform leaves the project's trait measurement what it was.

    The breeder states it when the project is created: whether it does, or for a rotation the
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
    """The project's record as stored.

    ``id`` is minted once at creation and kept by every correction; ``display_name``, the
    project's display name; ``site``, the breeder's own name for the orchard or station the
    project's plants stand in; ``invariances``, how far each capture transform leaves the trait's
    measurement what it was. The display name and the site are each :data:`RecordText`.
    """

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    display_name: RecordText
    site: RecordText
    invariances: CaptureInvariances


def read_record(project_path: str | Path) -> ProjectRecord:
    """Return the project's record.

    Raises :class:`ProjectRecordMissingError` for a project with no record yet, and
    :class:`ProjectRecordInvalidError` for a document that decodes but is not one. A document
    that does not decode raises the store's own ``DecodeError``, and a root the store refuses
    raises its ``StoreError``. Never creates a store.
    """
    return _checked(
        project_path,
        tcip_store.read(project_record_key(project_path), default=None),
    )


def existing_project(value: str | Path) -> tuple[Path, ProjectRecord]:
    """Return ``value`` resolved and its record (:func:`record_or_problem`).

    ``value`` must be a project whose record reads; otherwise ``ValueError`` naming the path and
    what :func:`read_record` refused.
    """
    project = Path(value).resolve()
    record, problem = record_or_problem(project)
    if record is None:
        raise ValueError(f"{project} names no readable project: {problem}")
    return project, record


def _checked(project_path: str | Path, raw: object) -> ProjectRecord:
    """Return ``raw``, a record read from ``project_path``, validated as a :class:`ProjectRecord`.

    Raises ``ProjectRecordMissingError`` for an absent record and ``ProjectRecordInvalidError``
    naming what the record lacks otherwise.
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
    """Write a new project's record with a freshly minted id, and return it.

    A project already recording the same display name, site and invariances is returned as it
    stands, its id kept. A present record with a different site raises
    :class:`SiteConflictError`, and one with a different display name or invariances raises
    ``ValueError`` naming both; an unreadable one raises what :func:`read_record` raises.
    """
    record = ProjectRecord(
        id=mint_id(),
        display_name=display_name,
        site=site,
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
            "write-project-record site to correct it deliberately."
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
            "match, so nothing was written. Run tcip write-project-record invariances, or "
            "correct them in the Setup tab, to correct them deliberately."
        )
    return existing


class RecordCorrection(BaseModel):
    """A correction of the project's record: the ``record`` it wrote and the ``previous`` one."""

    record: ProjectRecord
    previous: ProjectRecord


def _update(
    project_path: str | Path, field: str, value: object, *, actor: str
) -> RecordCorrection:
    """Replace ``field`` of a readable record with ``value`` under compare-and-swap.

    The whole replacement is validated as a :class:`ProjectRecord`, then one
    ``project_record_corrected`` line is recorded by ``actor`` in the project's log carrying the
    record's ``id``, the ``field``, the ``value`` written and the ``previous_value`` it replaced,
    each as the record dumps it. Returns the :class:`RecordCorrection`.

    Refuses what :func:`read_record` and the model refuse, with nothing written and no line; a
    line that cannot be appended after the write raises ``AuditEntryNotWrittenError``.
    """
    from tcip_mcp.audit import record_event_or_raise

    key = project_record_key(project_path)
    while True:
        stored = tcip_store.read_versioned(key, default=None)
        current = _checked(project_path, stored.value)
        updated = ProjectRecord.model_validate(
            current.model_copy(update={field: value}).model_dump()
        )
        written = updated.model_dump()
        try:
            tcip_store.replace(key, written, expect=stored.version)
        except VersionConflictError:
            continue
        break
    record_event_or_raise(
        "project_record_corrected",
        {
            "id": updated.id,
            "field": field,
            "value": written[field],
            "previous_value": current.model_dump()[field],
        },
        actor=actor,
        scope=project_path,
    )
    return RecordCorrection(record=updated, previous=current)


def replace_site(
    project_path: str | Path, site: str, *, actor: str
) -> RecordCorrection:
    """Correct the site a readable record holds, by ``actor``, through :func:`_update`.

    Every other field is kept. Refuses what :func:`_update` refuses, a site that is not
    :data:`RecordText` among it.
    """
    return _update(project_path, "site", site, actor=actor)


def rename_project(
    project: Path, display_name: str, *, actor: str
) -> RecordCorrection:
    """Change the project's display name, by ``actor``, through :func:`_update`.

    Its directory and every other record are left as they are. Refuses what :func:`_update`
    refuses, a display name that is not :data:`RecordText` among it.
    """
    return _update(project, "display_name", display_name, actor=actor)


def replace_invariances(
    project_path: str | Path, invariances: CaptureInvariances, *, actor: str
) -> RecordCorrection:
    """Correct the capture invariances a readable record holds, by ``actor``.

    The correction goes through :func:`_update`; every other field is kept. Refuses what
    :func:`_update` refuses.
    """
    return _update(project_path, "invariances", invariances, actor=actor)


def record_or_problem(
    project_path: str | Path,
) -> tuple[ProjectRecord | None, str | None]:
    """Return the project's record and ``None``, or ``None`` and the text of why it will not read.

    A record that is missing, invalid, refused by the store or unreadable from disk is the
    second; nothing is raised.
    """
    try:
        return read_record(project_path), None
    except (
        ProjectRecordMissingError,
        ProjectRecordInvalidError,
        StoreError,
        OSError,
    ) as exc:
        return None, str(exc)

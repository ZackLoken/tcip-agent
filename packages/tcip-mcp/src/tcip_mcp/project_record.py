"""The project record: the one document every project carries, holding its authored site.

A locator module in the same spirit as :mod:`tcip_mcp.project_status`: pure identity, read and
write helpers for a single top-level document, ``<project>/.tcip/project.json``. ``site`` is the
breeder's own name for the orchard or station the project's plants stand in, given once at
``init_project``/``ingest_images`` and never guessed from a directory name or a filename. It is
the record's only field, so a field is named for what it holds.

Every project gets exactly one record, written the first time either creating door scaffolds the
project: :func:`record_site` is a create-only write, so a second season of the same site is the
same call again, agreeing with what is already there, and a different site refuses rather than
silently overwriting one project's identity with another's. The one deliberate correction path is
``scripts/conform_project_site.py --replace``, for a site typed wrong once or a record damaged by
hand; nothing else in this module overwrites a present record.
"""

from __future__ import annotations

import os
from pathlib import Path

import tcip_store
from tcip_store import (
    RECORD_JSON,
    Key,
    StoreDescriptor,
    StoreError,
    Version,
    VersionConflict,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

_PROJECT_RECORD_DOC = RootedFileLocator(prefix=(".tcip",), suffix=".json")
"""The project record, one document per project: ``<project>/.tcip/project.json``."""

PROJECT_RECORD_STORE = "project_record"
_PROJECT_RECORD_PARTS = ("project",)
register_store(
    StoreDescriptor(
        name=PROJECT_RECORD_STORE,
        kind="record",
        key_fields=("document",),
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_PROJECT_RECORD_DOC,
    )
)

_MAX_SITE_LENGTH = 200
"""A rendering bound, not an identity bound: the site is shown on one picker line and in one-line
doctor findings and audit arguments, and no source states a longer identity. Provisional, from
those three rendering surfaces rather than from any stated site vocabulary."""


class ProjectRecordMissing(Exception):
    """A project has no ``.tcip/project.json`` yet."""


class ProjectRecordInvalid(ValueError):
    """A project's record decodes but does not hold a site."""


class SiteConflict(ValueError):
    """A project already records a different site than the one just offered."""


def project_record_key(project_path: str | Path) -> Key:
    """The project's own record.

    ``cas``: :func:`record_site` reads the stored version (via a ``VersionConflict``'s own
    ``actual`` when the create-only write finds something already there) before deciding whether
    to write, so an unconditional replace would risk a lost update between two callers.
    """
    return Key(PROJECT_RECORD_STORE, str(project_path), _PROJECT_RECORD_PARTS)


def project_record_path(project_path: str | Path) -> Path:
    """``<project_path>/.tcip/project.json``."""
    root = Path(project_path)
    return root.joinpath(*_PROJECT_RECORD_DOC.relative_path(str(root), _PROJECT_RECORD_PARTS).parts)


def _validate_site(site: object) -> str:
    """The site as it will be stored: stripped text, or the refusal naming exactly what is wrong.

    No case folding and no slug normalization: the platform holds no site vocabulary, and
    normalizing here would silently merge or split two of the breeder's own spellings. A
    non-string, an empty string (after stripping), one carrying any non-printable character
    (``str.isprintable``, which also catches non-breaking and zero-width characters a paste from
    a document can carry), or one over :data:`_MAX_SITE_LENGTH` characters refuses by name, so an
    invisible character is actionable rather than silently dropped or retyped blind.
    """
    if not isinstance(site, str):
        raise ValueError(f"site must be a string, got {type(site).__name__}")
    text = site.strip()
    if not text:
        raise ValueError("site is empty (after stripping surrounding whitespace)")
    for offset, ch in enumerate(text):
        if not ch.isprintable():
            raise ValueError(
                f"site contains a non-printable character U+{ord(ch):04X} at offset {offset}"
            )
    if len(text) > _MAX_SITE_LENGTH:
        raise ValueError(
            f"site is {len(text)} characters long, over the {_MAX_SITE_LENGTH}-character "
            "rendering bound (a picker line, a doctor finding, an audit argument)"
        )
    return text


def _store_present(root: Path) -> bool:
    """Whether this root already holds a store, under whichever backend is bound.

    Mirrors :func:`tcip_store.binding.bind_default`'s own backend selection rather than
    inspecting the bound backend instance, since the seam exposes no public accessor for which
    one is bound. Under the database binding this asks ``database_file``, the same path
    ``scripts/doctor.py`` asks without creating one; under the file binding a record's presence
    is the presence of its own file, so this asks that path directly.
    """
    from tcip_store import binding
    from tcip_store.file_backend import database_file

    name = os.environ.get(binding.BACKEND_ENV) or binding.DEFAULT_BACKEND
    if name == binding.FILE_BACKEND:
        return project_record_path(root).is_file()
    return database_file(str(root.absolute())).is_file()


def read_record(project_path: str | Path) -> dict:
    """The project's record: ``{"site": <str>}``.

    Raises :class:`ProjectRecordMissing` for a project with no record yet, and
    :class:`ProjectRecordInvalid` for a document that decodes but does not hold a non-empty
    string ``site``. A document that does not decode raises the store's own ``DecodeError``, and
    a root the store's conform rail refuses (loose claimed files with no database, or a claimed
    file beside a database that never held this store) raises its ``StoreError``; both are let
    through rather than reinterpreted.

    Checks whether the root holds any store at all before opening one, so reading a listed
    project with no record never causes the database backend to publish a database under it.
    """
    root = Path(project_path)
    if not _store_present(root):
        raise ProjectRecordMissing(
            f"{project_path} has no project record; site it with init_project(<path>, "
            "site=<site>) when the project's name fits the workspace scheme, or with "
            "scripts/conform_project_site.py for any project"
        )
    raw = tcip_store.read(project_record_key(project_path), default=None)
    if raw is None:
        raise ProjectRecordMissing(
            f"{project_path} has no project record; site it with init_project(<path>, "
            "site=<site>) when the project's name fits the workspace scheme, or with "
            "scripts/conform_project_site.py for any project"
        )
    if not isinstance(raw, dict) or not isinstance(raw.get("site"), str) or not raw["site"]:
        raise ProjectRecordInvalid(
            f"{project_path}'s project record does not hold a site: found {raw!r}. Repair it "
            "with scripts/conform_project_site.py --replace"
        )
    return raw


def _previous_site(project_path: str | Path) -> str | None:
    """The site a prior record held, or ``None`` when absent, invalid, or undecodable.

    Read before a ``--replace`` overwrite so the operator script can report what changed. A
    small race between this read and the overwrite is accepted: this path is a human-invoked,
    one-off correction, never a concurrent one.
    """
    try:
        return read_record(project_path)["site"]
    except (ProjectRecordMissing, ProjectRecordInvalid, StoreError):
        return None


def record_site(project_path: str | Path, site: str, *, replace: bool = False) -> dict:
    """Write the project's site, the one writer of ``.tcip/project.json``.

    Validates ``site`` first (:func:`_validate_site`). With ``replace`` false this is a
    create-only write (``expect=Version.ABSENT``), the same call whether the directory is new
    or is an old project that never had a record:

    - Absent: written with the given site.
    - Present with the same site: nothing is written; the present record is returned.
    - Present with a different site: raises :class:`SiteConflict` naming both sites and the
      project.
    - Present and not a site record: raises :class:`ProjectRecordInvalid` naming what was found.
    - Present and undecodable: the store's own ``DecodeError`` propagates.
    - The root's store refusing the write (loose claimed files with no database, or a claimed
      file beside a database that never held this store): the rail's own ``StoreError``
      propagates.

    With ``replace`` true the write is unconditional: whatever was there, valid or not, is
    overwritten. Returns ``{"site": <the site now recorded>, "previous_site": <str | None>}``,
    where ``previous_site`` is the site a prior record held, or ``None`` when none did (a new
    project, or a prior record this call could not read as a site). Only
    ``scripts/conform_project_site.py`` passes ``replace=True``, so a site typed wrong once, or a
    record damaged by hand, has exactly one correction path and it is a deliberate one.
    """
    text = _validate_site(site)
    key = project_record_key(project_path)
    document = {"site": text}

    if replace:
        previous_site = _previous_site(project_path)
        expect: Version | None = Version.ABSENT
        while True:
            try:
                tcip_store.replace(key, document, expect=expect)
                break
            except VersionConflict as exc:
                # Another writer's version, read off the conflict itself rather than a second
                # read: the record may not decode, and only the byte-derived version is needed.
                expect = exc.actual
        return {"site": text, "previous_site": previous_site}

    try:
        tcip_store.replace(key, document, expect=Version.ABSENT)
        return {"site": text, "previous_site": None}
    except VersionConflict:
        existing = read_record(project_path)
        if existing["site"] == text:
            return {"site": text, "previous_site": text}
        raise SiteConflict(
            f"{project_path} already records site {existing['site']!r}; the offered site "
            f"{text!r} does not match, so nothing was written. Run "
            "scripts/conform_project_site.py --replace to correct it deliberately."
        ) from None


def site_fields(project_path: str | Path) -> dict:
    """``{"site": str | None, "site_problem": str | None}``: exactly one is set, and this never
    raises.

    The one reader every surface (the picker, ``inspect_project``, the doctor) calls, so they
    read one function and cannot disagree about a listed project's site under one store binding.
    ``site_problem`` carries, in the breeder's and the agent's reading, why there is no site: the
    record is absent, present and not a site record, undecodable, or under a root the store
    refuses to read; each message is the raising exception's own.
    """
    try:
        record = read_record(project_path)
    except (ProjectRecordMissing, ProjectRecordInvalid, StoreError, OSError) as exc:
        return {"site": None, "site_problem": str(exc)}
    return {"site": record["site"], "site_problem": None}


__all__ = [
    "PROJECT_RECORD_STORE",
    "ProjectRecordInvalid",
    "ProjectRecordMissing",
    "SiteConflict",
    "project_record_key",
    "project_record_path",
    "read_record",
    "record_site",
    "site_fields",
]

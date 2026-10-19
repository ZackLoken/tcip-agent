"""Project management tools."""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

import tcip_store
from tcip_annotation.text import Text
from tcip_mcp.audit import audited
from tcip_mcp.project_record import CaptureInvariances
from tcip_mcp.registry_paths import located
from tcip_mcp.server import tool
from tcip_store import Key, VersionConflictError, encode_record


def _project_dir(project_path: str) -> Path:
    """Return the .tcip directory for a project, creating it if needed."""
    p = Path(project_path) / ".tcip"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- dataset identity registry (project -> datasets it uses) --------------


DATASET_REGISTRY_STORE = "dataset_registry"
_DATASET_REGISTRY_PARTS = ("datasets",)


def dataset_registry_key(project: str | Path) -> Key:
    """Return the key of the project's record of which datasets it uses, keyed by dataset id."""
    return Key(
        DATASET_REGISTRY_STORE, str(Path(project)), _DATASET_REGISTRY_PARTS
    )


class DatasetEntry(BaseModel):
    """One dataset the project's registry names: its identity's ``id`` and its stored ``path``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    path: str


_DATASET_REGISTRY = TypeAdapter(list[DatasetEntry])
"""The project's dataset registry, its entries sorted by ``id``."""


def read_datasets(project: str | Path) -> list[DatasetEntry]:
    """Read the project's dataset registry, or [] when absent.

    A registry present but undecodable raises, as does one that is not a list of
    :class:`DatasetEntry` (``ValidationError``).
    """
    return _DATASET_REGISTRY.validate_python(
        tcip_store.read(dataset_registry_key(project), default=[])
    )


def dataset_entry_path(project: str | Path, entry: DatasetEntry) -> Path:
    """Return the absolute path a dataset registry ``entry`` names.

    The path is resolved through :func:`~tcip_mcp.registry_paths.resolved_registry_path`.
    """
    from tcip_mcp.registry_paths import resolved_registry_path

    return resolved_registry_path(project, entry.path)


def upsert_dataset(project: str | Path, entry: DatasetEntry) -> None:
    """Add or refresh a dataset in the project's registry, matched by ``id``.

    A moved dataset updates the ``path`` of its existing id rather than duplicating, so identity
    survives a move.

    The read and the write are one transaction, so two registrations running at once cannot
    each write a list assembled from the state before the other's entry landed.
    """
    key = dataset_registry_key(project)
    with tcip_store.transaction(key) as txn:
        stored = _DATASET_REGISTRY.validate_python(txn.read(key, default=[]))
        regs = [r for r in stored if r.id != entry.id] + [entry]
        txn.write(
            key,
            _DATASET_REGISTRY.dump_python(
                sorted(regs, key=lambda r: r.id), mode="json"
            ),
        )


@tool()
@audited(scope_arg="dataset_root")
def register_dataset(project: Path, dataset_root: str, crop: Text) -> dict:
    """Record a dataset's identity with the dataset, and its id and path in the project's registry.

    The identity is ``{crop, id, fingerprint}``, and the path the registry holds is
    :func:`~tcip_mcp.registry_paths.stored_path`'s. ``id`` is minted once and kept across re-runs
    and moves, ``fingerprint`` is
    :func:`~tcip_mcp.pipelines.data.dataset_fingerprint.dataset_fingerprint` now, and the identity
    write is compare-and-set against the version this call read, a conflict keeping the id that
    committed.

    Args:
        dataset_root: Root of the dataset (holds ``images/``, ``subjects.json`` and the database
            its label documents are records in).
        crop: The crop this dataset's imagery is of, as ``crops.yml`` names it: the expert's
            fact, required non-empty text and never inferred from a path or slug.

    """
    from tcip_mcp.dataset_layout import (
        dataset_identity_path,
        decode_dataset_identity_document,
    )
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.project_record import mint_id
    from tcip_mcp.registry_paths import stored_path

    root = Path(dataset_root)
    if not root.is_dir():
        return {"error": f"dataset_root not found: {dataset_root}"}

    ident_path = dataset_identity_path(root)
    fingerprint = dataset_fingerprint(root)
    # A conflict means another registration committed, so the loop only repeats while the
    # identity is actually changing under it and ends when this write is the one that lands.
    while True:
        stored = tcip_store.read_blob_versioned(ident_path, default=None)
        if stored.value is None:
            existing: dict = {}
        else:
            try:
                existing = decode_dataset_identity_document(
                    stored.value, dataset_root=root
                )
            except ValueError as exc:
                return {
                    "error": f"{exc}; minting a fresh id over it would sever every record "
                    "that cites the old one"
                }
        entry = DatasetEntry(
            id=existing.get("id") or mint_id(), path=stored_path(root, project)
        )
        identity: dict[str, JsonValue] = {
            "crop": crop,
            "id": entry.id,
            "fingerprint": fingerprint,
        }
        try:
            tcip_store.put_blob(
                ident_path, encode_record(identity), expect=stored.version
            )
        except VersionConflictError:
            continue
        break

    upsert_dataset(project, entry)
    return {"dataset_root": str(root), **identity}


@tool()
def initialize_project(
    project_path: str,
    display_name: str,
    site: str,
    invariances: CaptureInvariances,
) -> dict:
    """Create a TCIP project: its ``.tcip/`` directories, its record and one creation log line.

    The project gets ``.tcip/`` with its artifacts and models directories, and its record holding
    a freshly minted id, ``display_name``, ``site`` and ``invariances``, then one
    ``project_created`` line in its own log.

    Creating a project that already records the same display name, site and invariances answers
    it as it stands; one recording a different site, display name or invariances refuses rather
    than overwriting it. A refused record leaves nothing on disk.

    Args:
        project_path: Root directory of the project.
        display_name: The project's display name.
        site: The orchard or station this project's plants stand in, in the breeder's own words.
        invariances: For each capture transform, whether it leaves the trait's measurement what
            it was, and for a rotation the largest angle in degrees at which it does (0 for
            none, 180 for any), as the breeder answers it.

    """
    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.project_record import create_record
    from tcip_store import StoreError

    try:
        project = located(Path(project_path).expanduser(), None)
        record = create_record(
            project, display_name, site, invariances
        ).model_dump()
    except (ValueError, StoreError) as exc:
        return {"error": str(exc)}
    tcip = _project_dir(str(project))
    (tcip / "artifacts").mkdir(exist_ok=True)
    (tcip / "models").mkdir(exist_ok=True)
    record_event_or_raise("project_created", record, actor=None, scope=project)
    return {"project_path": str(project), **record}


@tool()
def view_gui_state(project: Path) -> dict:
    """Read the GUI state the human last left in this project.

    The state comes from its GUI snapshot (``web_client.read_gui_snapshot``): the tab, the mode
    and the dataset selection, with ``current_image``, the image the selection's index names. A
    ``note`` stands in for all of it when the GUI has persisted nothing for this project; a
    snapshot that will not read raises.
    """
    from tcip_mcp.web_client import current_image, read_gui_snapshot

    state = read_gui_snapshot(project)
    if state is None:
        return {
            "note": "the GUI has not persisted a selection for this project"
        }
    return {
        **state.model_dump(mode="json"),
        "current_image": current_image(state.dataset),
    }


@tool()
def inspect_project(project: Path) -> dict:
    """Get an overview of the project.

    Carries the record's fields (``id``, ``display_name``, ``site`` and ``invariances``) beside
    ``record_problem`` from ``tcip_mcp.project_record.record_fields``: the fields are set, or
    ``record_problem`` names why they are not (a damaged record, or a root the store refuses to
    read). ``plant_mappings``
    carries every mapping name persisted under the project, the same shape:
    ``plant_mappings_problem`` names why the listing came back empty when the store refuses to
    read the root. ``recent_activity`` is :func:`_recent_activity`'s.
    """
    from tcip_mcp.project_record import record_fields

    tcip = project / ".tcip"

    status: dict = {"project_path": str(project), **record_fields(project)}

    # Models
    models_dir = tcip / "models"
    if models_dir.is_dir():
        status["model_count"] = len(list(models_dir.glob("*.pt")))

    # Artifacts
    artifacts_dir = tcip / "artifacts"
    if artifacts_dir.is_dir():
        status["artifact_count"] = len(list(artifacts_dir.iterdir()))

    # The logical images under <root>/images/<date>/ and the capture dates they sit under.
    from tcip_mcp import dataset_layout
    from tcip_mcp.pipelines.image_utils import logical_image_count

    if dataset_layout.image_root(project).is_dir():
        status["image_count"] = logical_image_count(project)
        status["dates"] = dataset_layout.list_dates(project)

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_mapping_names,
    )
    from tcip_store import StoreError

    try:
        status["plant_mappings"] = plant_mapping_names(project)
    except StoreError as exc:
        status["plant_mappings"] = []
        status["plant_mappings_problem"] = str(exc)

    status["recent_activity"] = _recent_activity(project)
    return status


_ACTIVITY_TOOLS = (
    "report_friction",
    "write_retrospective",
    "record_distillation_pass",
)


def _recent_activity(project: Path) -> dict:
    """Derive the project's recent memory activity from its audit log.

    The activity is read from the log's ``report_friction``, ``write_retrospective`` and
    ``record_distillation_pass`` lines in the order they landed: the reports since the last
    retrospective and since the last distillation pass, the retrospectives since that pass, the
    last retrospective's ``project_id`` and time, the last pass's time, and the time of the last
    report or retrospective. A log carrying undecodable entries answers
    ``status_unavailable`` naming how many.
    """
    from tcip_mcp.audit import acts_of

    try:
        lines, _cursor = acts_of(project, _ACTIVITY_TOOLS)
    except ValueError as exc:
        return {"status_unavailable": str(exc)}
    activity: dict = {
        "reports_since_last_retrospective": 0,
        "reports_since_last_distillation": 0,
        "retrospectives_since_last_distillation": 0,
    }
    for line in lines:
        if line["tool"] == "record_distillation_pass":
            activity["reports_since_last_distillation"] = 0
            activity["retrospectives_since_last_distillation"] = 0
            activity["last_distillation_at"] = line["timestamp"]
            continue
        if line["tool"] == "report_friction":
            activity["reports_since_last_retrospective"] += 1
            activity["reports_since_last_distillation"] += 1
        else:
            activity["reports_since_last_retrospective"] = 0
            activity["retrospectives_since_last_distillation"] += 1
            activity["last_retrospective"] = {
                "project_id": line["arguments"]["project_id"],
                "modified_at": line["timestamp"],
            }
        activity["last_activity"] = line["timestamp"]
    return activity


class ZipBundle(BaseModel):
    """A bundle written as one ZIP file at ``output_path``."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    output_path: str


class TreeBundle(BaseModel):
    """A bundle written as a directory tree at ``output_dir``."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    output_dir: str


@audited
def archive_project(
    project: Path,
    destination: ZipBundle | TreeBundle,
    include_models: bool = False,
) -> dict:
    """Export ``project`` as a portable bundle and record the export in the project's log.

    The bundle is :func:`write_archive`'s, on its terms.
    """
    return write_archive(project, destination, include_models=include_models)


def write_archive(
    project: Path,
    destination: ZipBundle | TreeBundle,
    *,
    include_models: bool = False,
) -> dict:
    """Export ``project`` as a portable bundle written where ``destination`` names.

    ``project`` is an established project root
    (:func:`~tcip_mcp.project_record.existing_project`) and ``destination`` is resolved by
    :func:`~tcip_mcp.registry_paths.located` against it: a ZIP file for a :class:`ZipBundle`, the
    identical bundle as a tree for a :class:`TreeBundle`.

    The bundle is every file under the project but the store's own bookkeeping
    (:func:`tcip_store.file_backend.is_bookkeeping`), each under its path relative to the
    project, each store database as the consistent copy
    :func:`tcip_store.sqlite_backend.copy_database` takes, and without the checkpoints
    :func:`~tcip_mcp.model_registry.checkpoint_files` names unless ``include_models``;
    ``checkpoints_excluded`` counts those left out. A write that fails removes the destination.

    Refuses a destination inside the project (a bundle cannot contain the tree it was drawn from)
    and a destination that already exists, except an empty directory named by a
    :class:`TreeBundle`.

    The response carries the destination under the block's own field name, and ``size_bytes``
    means one thing under ``output_path`` (the written ZIP's own compressed byte count,
    ``stat().st_size`` on the archive) and a different thing under ``output_dir`` (the sum of the
    copied members' own uncompressed byte counts).

    Args:
        destination: Where the bundle is written, and as which container.
        include_models: Whether to include model checkpoints (can be large).

    """
    ((key, given),) = destination.model_dump().items()
    as_tree = isinstance(destination, TreeBundle)
    dest = located(given, project)
    if dest.is_relative_to(project):
        return {
            "error": f"the destination {dest} is inside the project being archived "
            f"({project}); choose a destination outside the project"
        }
    if dest.exists() and not (
        as_tree and dest.is_dir() and not any(dest.iterdir())
    ):
        return {
            "error": f"the destination {dest} already exists; an archive never writes over "
            "or into anything"
        }

    import tempfile
    from collections.abc import Callable
    from contextlib import ExitStack

    from tcip_mcp.model_registry import checkpoint_files
    from tcip_store.file_backend import (
        database_file,
        database_roots,
        is_bookkeeping,
    )
    from tcip_store.sqlite_backend import copy_database

    checkpoints = frozenset() if include_models else checkpoint_files(project)
    files = [
        p
        for p in project.rglob("*")
        if p.is_file() and not is_bookkeeping(p.name)
    ]
    members = [p for p in files if p not in checkpoints]
    databases = {database_file(str(r)) for r in database_roots(project)}

    def copied(source: Path, name: Path) -> None:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (dest if as_tree else dest.parent).mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as stack:
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            put: Callable[[Path, Path], object] = (
                copied
                if as_tree
                else stack.enter_context(
                    zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED)
                ).write
            )
            for n, member in enumerate(sorted(members)):
                source = member
                if member in databases:
                    source = scratch / f"{n}.db"
                    copy_database(member, source)
                put(source, member.relative_to(project))
    except BaseException:
        if as_tree:
            shutil.rmtree(dest, ignore_errors=True)
        else:
            dest.unlink(missing_ok=True)
        raise
    size_bytes = (
        sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
        if as_tree
        else dest.stat().st_size
    )
    return {
        "files_added": len(members),
        "size_bytes": size_bytes,
        "include_models": include_models,
        "checkpoints_excluded": len(files) - len(members),
        key: str(dest),
    }


_IMPORTS_DIRNAME = ".imports"


def _extract_zip(zp: Path, staging: Path) -> int:
    """Extract every member of ``zp`` into ``staging``, refusing a path that would escape it.

    ``staging`` is this run's own private directory (a fresh uuid under ``.imports``), so a refusal
    here leaves the destination untouched. Escape is decided by containment
    (``Path.relative_to``).
    """
    files_extracted = 0
    with zipfile.ZipFile(str(zp), "r") as zf:
        staged = staging.resolve()
        for info in zf.infolist():
            if not (staging / info.filename).resolve().is_relative_to(staged):
                raise ValueError(f"Unsafe path in archive: {info.filename}")
        for info in zf.infolist():
            if info.is_dir():
                (staging / info.filename).mkdir(parents=True, exist_ok=True)
            else:
                target = staging / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                files_extracted += 1
    return files_extracted


def _stage_bundle(source: Path, staging: Path) -> int:
    """Stage ``source`` into ``staging``, whichever container it arrived in.

    A directory bundle's whole tree is copied, member by member, into a ZIP written to the system
    temp directory (``tempfile.TemporaryDirectory``, removed once this call returns), then handed
    to :func:`_extract_zip`. A directory's own empty subdirectories carry no member either way,
    matching a ZIP archive built with no directory entries.
    """
    if not source.is_dir():
        return _extract_zip(source, staging)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_zip = Path(tmp) / "bundle.zip"
        with zipfile.ZipFile(str(tmp_zip), "w", zipfile.ZIP_DEFLATED) as zf:
            for member in sorted(p for p in source.rglob("*") if p.is_file()):
                zf.write(member, member.relative_to(source))
        return _extract_zip(tmp_zip, staging)


def _sweep_free_locked_leftovers(imports_root: Path) -> None:
    """Remove every staging sibling under ``imports_root`` whose lock is free.

    Such a sibling, one whose lock this process can take without blocking, is a crash leftover
    from an earlier run. A sibling still locked is a concurrent import's live work (including one
    in its own rename window) and is left alone, its lock file included.
    """
    from filelock import Timeout as _LockTimeout

    from tcip_store.file_backend import lock_file_for, path_lock

    if not imports_root.is_dir():
        return
    for entry in sorted(imports_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            with path_lock(entry, timeout_s=0):
                shutil.rmtree(entry, ignore_errors=True)
        except _LockTimeout:
            continue
        lock_file_for(entry).unlink(missing_ok=True)


def _move_staging_onto_destination(
    staged: Path, dest: Path, *, timeout_s: float
) -> None:
    """Rename the staged tree onto ``dest``, re-checking emptiness immediately before the rename.

    Both the empty-destination removal and the rename itself run under
    ``tcip_store.file_backend.retry_while_denied``'s budget.
    """
    from tcip_store.file_backend import retry_while_denied

    def _prepare_and_rename() -> None:
        if dest.exists():
            if any(dest.iterdir()):
                raise StoreRuntimeError(
                    f"destination {dest} is no longer empty; refusing the move"
                )
            dest.rmdir()
        os.rename(str(staged), str(dest))

    retry_while_denied(_prepare_and_rename, timeout_s)


class StoreRuntimeError(RuntimeError):
    """An import destination became non-empty before the staged tree could replace it."""


def import_project(bundle_path: str, destination: str) -> dict:
    """Import an annotation project from a bundle ``archive_project`` wrote.

    The bundle is a ZIP archive, or a directory tree written by its ``output_dir`` mode. A
    successful import records one ``import_project`` line in the imported project's own log.

    A directory bundle is staged through the identical walker a ZIP bundle is
    (:func:`_stage_bundle`), so the two are read back exactly alike below this point.

    The door extracts into a private staging directory, refusing a member that would escape it or
    that is the store's own bookkeeping (:func:`tcip_store.file_backend.is_bookkeeping`), then
    renames the staged tree onto ``destination``.

    ``destination`` must not already exist, or must be an empty directory: this door merges nothing
    into a live project.

    A refusal at any step leaves the destination exactly as it was (absent, or its original empty
    state); the staging tree this run made is removed whether the run refused, raised, or
    succeeded.

    The response carries ``files_extracted``, ``dataset_paths_unresolved`` (the registered datasets
    whose absolute path stayed verbatim because they are outside the imported tree), and the
    registry's ``checkpoint_paths_unresolved`` and ``external_checkpoints``
    (:func:`~tcip_mcp.model_registry.registry_checkpoint_disclosures`).

    Args:
        bundle_path: Path to the ``.tcip.zip`` archive, or the directory tree written by
            ``archive_project``'s ``output_dir`` mode.
        destination: Directory to extract into.

    """
    from filelock import Timeout as _LockTimeout

    from tcip_mcp.audit import record_event_or_raise
    from tcip_store.file_backend import (
        DEFAULT_LOCK_TIMEOUT_S,
        lock_file_for,
        path_lock,
    )

    try:
        bp, dest = (
            located(Path(p).expanduser(), None)
            for p in (bundle_path, destination)
        )
    except ValueError as exc:
        return {"error": str(exc)}
    if not bp.is_file() and not bp.is_dir():
        return {"error": f"bundle not found: {bundle_path}"}
    if dest.exists():
        if not dest.is_dir():
            return {
                "error": f"destination {dest} exists and is not a directory"
            }
        if any(dest.iterdir()):
            return {
                "error": f"destination {dest} is not empty; import_project never writes "
                "into an existing project (a destination with state merges nothing; "
                "that stays operator work)"
            }

    dest.parent.mkdir(parents=True, exist_ok=True)
    imports_root = dest.parent / _IMPORTS_DIRNAME
    imports_root.mkdir(exist_ok=True)
    _sweep_free_locked_leftovers(imports_root)

    staging = imports_root / uuid.uuid4().hex
    result: dict = {}
    try:
        with path_lock(staging, timeout_s=DEFAULT_LOCK_TIMEOUT_S):
            try:
                result = _run_import_into_staging(bp, staging, dest)
            finally:
                # A success has already moved staging onto dest; rmtree is then a no-op.
                if "error" in result or not result:
                    shutil.rmtree(staging, ignore_errors=True)
    except _LockTimeout:
        return {
            "error": f"could not lock a fresh staging directory at {staging}"
        }
    finally:
        lock_file_for(staging).unlink(missing_ok=True)
    if "error" not in result:
        record_event_or_raise(
            "import_project",
            {"bundle_path": bundle_path},
            actor=None,
            scope=dest,
        )
    return result


def _run_import_into_staging(bp: Path, staging: Path, dest: Path) -> dict:
    """Stage the bundle, refuse the store's own bookkeeping, then move it onto ``dest``.

    Returns the import's response dict, an ``{"error": ...}`` on any refusal.
    """
    from tcip_mcp.model_registry import registry_checkpoint_disclosures
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S, is_bookkeeping

    try:
        files_extracted = _stage_bundle(bp, staging)
    except (ValueError, zipfile.BadZipFile) as exc:
        return {"error": f"{bp} is not a readable bundle: {exc}"}

    bookkeeping = sorted(
        p.relative_to(staging).as_posix()
        for p in staging.rglob("*")
        if p.is_file() and is_bookkeeping(p.name)
    )
    if bookkeeping:
        return {
            "error": f"the archive carries the store's own bookkeeping "
            f"({', '.join(bookkeeping)}), which a bundle never legitimately holds; "
            "refusing the whole import"
        }

    try:
        _move_staging_onto_destination(
            staging, dest, timeout_s=DEFAULT_LOCK_TIMEOUT_S
        )
    except (OSError, StoreRuntimeError) as exc:
        return {
            "error": f"could not move the staged import onto {dest}: {exc}"
        }

    return {
        "destination": str(dest),
        "files_extracted": files_extracted,
        "dataset_paths_unresolved": _external_dataset_paths(dest),
        **registry_checkpoint_disclosures(dest),
    }


def _external_dataset_paths(project: Path) -> list[str]:
    """Return the project's own registered dataset paths that stay absolute (external), sorted."""
    from tcip_mcp.registry_paths import is_external_form

    return sorted(
        e.path for e in read_datasets(project) if is_external_form(e.path)
    )

"""Project management tools."""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path

import tcip_store
from tcip_store import (
    RECORD_JSON,
    Key,
    StoreDescriptor,
    VersionConflict,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import tool
from tcip_mcp.audit import audited

_PROJECT_STATE_DOC = RootedFileLocator(prefix=(".tcip",), suffix=".json")
"""A project's own top-level ``.tcip`` documents, one of each per project."""


def _project_dir(project_path: str) -> Path:
    """Return the .tcip directory for a project, creating it if needed."""
    p = Path(project_path) / ".tcip"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- dataset identity registry (project -> datasets it uses) --------------


DATASET_REGISTRY_STORE = "dataset_registry"
_DATASET_REGISTRY_PARTS = ("datasets",)
register_store(
    StoreDescriptor(
        name=DATASET_REGISTRY_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        cannot_carry_field="a top-level JSON array of entries, with no object to hold the field; "
                            "a future bump wraps this into {schema_version, entries}",
        codec=RECORD_JSON,
        concurrency="cas",
        locator=_PROJECT_STATE_DOC,
    )
)


def dataset_registry_key(project: str | Path) -> Key:
    """The project's record of which datasets it uses, keyed by dataset id.

    ``cas``: :func:`upsert_dataset` reads the whole list, replaces one entry and writes it
    back, so an unconditional write drops a dataset another registration had just added.
    """
    return Key(DATASET_REGISTRY_STORE, str(Path(project)), _DATASET_REGISTRY_PARTS)


def read_datasets(project: str | Path) -> list[dict]:
    """The project's dataset registry (``[{id, path, crop, fingerprint}]``), or [] when absent.

    A registry present but undecodable raises rather than reading as empty.
    """
    return tcip_store.read(dataset_registry_key(project), default=[])


def dataset_entry_path(project: str | Path, entry: dict) -> Path:
    """The absolute path a dataset registry ``entry`` names, resolved through
    :func:`~tcip_mcp.registry_paths.resolved_registry_path`."""
    from tcip_mcp.registry_paths import resolved_registry_path

    return resolved_registry_path(project, entry["path"])


def upsert_dataset(project: str | Path, entry: dict) -> None:
    """Add or refresh a dataset in the project's registry, matched by ``id``: a moved dataset updates
    the ``path`` of its existing id rather than duplicating, so identity survives a move.

    The read and the write are one transaction, so two registrations running at once cannot
    each write a list assembled from the state before the other's entry landed.
    """
    key = dataset_registry_key(project)
    with tcip_store.transaction(key) as txn:
        regs = [r for r in txn.read(key, default=[])
                if r.get("id") != entry.get("id")]
        regs.append(entry)
        txn.write(key, sorted(regs, key=lambda r: str(r.get("id", ""))))


@tool()
@audited(scope_arg="dataset_root")
def register_dataset(project: Path, dataset_root: str, crop: str) -> dict:
    """Record a dataset's identity so a delivered number can be traced to the exact data behind it.

    Writes ``<dataset_root>/dataset.json = {crop, id, fingerprint}`` (identity travels with the
    data) and upserts the dataset into the project's ``.tcip/datasets.json``. ``crop`` is the
    human's fact and is required, never inferred from a path or slug. ``id`` is minted once and
    preserved across re-runs and path moves; ``fingerprint`` is the whole-dataset content digest
    (labels + image pixels + registry + confirmed negatives), recomputed here, but the stored value
    is a cache, and recompute-on-read (``dataset_fingerprint.dataset_fingerprint``) is the
    authority.

    The identity write is compare-and-set against the version this call read. A conflict re-reads
    what committed and keeps the id it carries, and the project registry is reconciled against that
    committed id rather than the one this call proposed.

    The registry's stored ``path`` follows :func:`~tcip_mcp.registry_paths.stored_path`: relative
    to the project whenever the dataset sits under it, absolute for an external dataset.

    Args:
        dataset_root: Root of the dataset (holds ``images/``, ``annotations/``, ``subjects.json``).
        crop: The crop this dataset's imagery is of, as ``crops.yml`` names it. Required; the
            expert's fact.
    """
    from tcip_store import SchemaVersionRefused

    from tcip_mcp.dataset_layout import decode_dataset_identity_document, dataset_identity_key
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.project_record import mint_id
    from tcip_mcp.registry_paths import stored_path

    root = Path(dataset_root)
    if not root.is_dir():
        return {"error": f"dataset_root not found: {dataset_root}"}
    if not crop:
        return {"error": "crop is required (the expert's fact; never inferred from a path or slug)"}

    ident_key = dataset_identity_key(root)
    try:
        fingerprint = dataset_fingerprint(root)
    except SchemaVersionRefused as exc:
        return {"error": f"{root}: {exc}"}
    # A conflict means another registration committed, so the loop only repeats while the
    # identity is actually changing under it and ends when this write is the one that lands.
    while True:
        stored = tcip_store.read_blob_versioned(ident_key, default=None)
        if stored.value is None:
            existing: dict = {}
        else:
            try:
                existing = decode_dataset_identity_document(stored.value, dataset_root=root)
            except SchemaVersionRefused as exc:
                return {"error": f"{exc} Re-registering here would overwrite a newer writer's "
                                 "identity document; nothing was written."}
            except ValueError as exc:
                return {"error": f"{exc}; minting a fresh id over it would sever every record "
                                 "that cites the old one"}
        candidate = {
            "crop": crop,
            "id": existing.get("id") or mint_id(),
            "fingerprint": fingerprint,
        }
        try:
            tcip_store.put_blob(
                ident_key, RECORD_JSON.encode(candidate), expect=stored.version,
            )
        except VersionConflict:
            continue
        identity = candidate
        break

    upsert_dataset(project, {"id": identity["id"], "path": stored_path(root, project),
                             "crop": crop, "fingerprint": fingerprint})
    return {"dataset_root": str(root), **identity}


@tool()
def initialize_project(project_path: str, display_name: str, site: str) -> dict:
    """Create a TCIP project: ``.tcip/`` with its artifacts and models directories, and its record
    holding a freshly minted id, ``display_name`` and ``site``, then one ``project_created`` line
    in its own log.

    Creating a project that already records the same display name and site answers it as it
    stands; one recording a different site or display name refuses rather than overwriting it.
    A refused record leaves nothing on disk.

    Args:
        project_path: Root directory of the project.
        display_name: The name the project picker shows; renaming later changes only this.
        site: The orchard or station this project's plants stand in, in the breeder's own words.
    """
    from tcip_store import StoreError

    from tcip_mcp.audit import record_event_or_raise
    from tcip_mcp.project_record import create_record

    project = Path(project_path).expanduser().resolve()
    try:
        record = create_record(project, display_name, site)
    except (ValueError, StoreError) as exc:
        return {"error": str(exc)}
    tcip = _project_dir(str(project))
    (tcip / "artifacts").mkdir(exist_ok=True)
    (tcip / "models").mkdir(exist_ok=True)
    record_event_or_raise("project_created", dict(record), scope=project)
    return {"project_path": str(project), **record}


@tool()
def view_gui_state(project: Path) -> dict:
    """The GUI state the human last left in this project, from its ``gui.json``
    (``web_client.read_gui_snapshot``): the tab, the mode and the dataset selection, with
    ``current_image``, the image the selection's index names. A ``note`` stands in for all of it
    when the GUI has persisted nothing for this project; a snapshot that will not read raises.
    """
    from tcip_mcp.web_client import current_image, read_gui_snapshot

    state = read_gui_snapshot(project)
    if state is None:
        return {"note": "the GUI has not persisted a selection for this project"}
    return {**state.model_dump(mode="json"), "current_image": current_image(state.dataset)}


@tool()
def inspect_project(project: Path) -> dict:
    """Get an overview of the project.

    Carries the record's ``id``, ``display_name`` and ``site`` beside ``record_problem`` from
    ``tcip_mcp.project_record.record_fields``: the three are set, or ``record_problem`` names why
    they are not (a damaged record, or a root the store refuses to read). ``plant_mappings``
    carries every mapping name persisted under the project, the same shape:
    ``plant_mappings_problem`` names why the listing came back empty when the root's state is a
    store the bound backend refuses to read (a root still in the loose-file layout under the
    database default).
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

    # Data: the canonical layout puts images under <root>/images/<date>/ (see
    # tcip_mcp.dataset_layout); ingest_images writes there. Count that tree
    # recursively so date buckets aren't missed, and report the capture dates.
    image_exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    from tcip_mcp import dataset_layout

    images_dir = dataset_layout.image_root(project)
    if images_dir.is_dir():
        status["image_count"] = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts
        )
        status["dates"] = dataset_layout.list_dates(project)

    from tcip_store import StoreError

    from tcip_mcp.pipelines.postprocessing.plant_mapping import plant_mapping_names

    try:
        status["plant_mappings"] = plant_mapping_names(project)
    except StoreError as exc:
        status["plant_mappings"] = []
        status["plant_mappings_problem"] = str(exc)

    status["recent_activity"] = _recent_activity(project)
    return status


def _recent_activity(project: Path) -> dict:
    """The project's persisted status summary (recent report/retrospective/distillation activity),
    namespaced separately from the live-computed fields above it; it may be stale or corrupt.
    """
    from tcip_mcp.project_status import read_project_status

    activity = read_project_status(project)
    if activity.get("_version_refused"):
        return {"status_unavailable": "project_status.json is at a schema_version this "
                                       "reader does not accept"}
    if activity.get("_corrupt"):
        return {"status_unavailable": "project_status.json exists but could not be read"}
    return activity


def _store_databases(root: Path) -> list[Path]:
    """Every store database under the tree an archive would bundle."""
    from tcip_store.file_backend import DATABASE_FILENAME

    return sorted(p for p in root.rglob(DATABASE_FILENAME) if p.parent.name == ".tcip")


def _export_stores(root: Path) -> None:
    """Write every record and log store of every database under this tree back out as files, the
    way an archive bundles state.
    """
    from tcip_store.export import export_root

    for db_path in _store_databases(root):
        export_root(str(db_path.parent.parent.absolute()), report=lambda _line: None)


def _database_counters(root: Path) -> dict[tuple[str, str], int]:
    """Every store's change counter across the tree, for comparing before and after a copy."""
    from tcip_store.export import read_store_states

    counters: dict[tuple[str, str], int] = {}
    for db_path in _store_databases(root):
        for store, state in read_store_states(db_path).items():
            counters[(str(db_path), store)] = state.change_counter
    return counters


def _write_bundle_zip(out: Path, root: Path, members: list[Path]) -> int:
    """Write ``members`` (each an absolute path under ``root``) into ``out`` as a ZIP.

    Removes ``out`` on any failure.
    """
    files_added = 0
    try:
        with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
            for member in sorted(members):
                zf.write(member, member.relative_to(root))
                files_added += 1
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    return files_added


def _write_bundle_directory(out_dir: Path, root: Path, members: list[Path]) -> tuple[int, int]:
    """Write ``members`` (each an absolute path under ``root``) into ``out_dir`` as the identical
    tree, preserving each member's path relative to ``root``. Returns ``(files_added,
    size_bytes)``.

    Removes ``out_dir`` on any failure.
    """
    files_added = 0
    size_bytes = 0
    try:
        for member in sorted(members):
            target = out_dir / member.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(member, target)
            files_added += 1
            size_bytes += target.stat().st_size
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return files_added, size_bytes


@audited
def archive_project(
    project: Path, output_path: str = "", output_dir: str = "",
    include_models: bool = False,
) -> dict:
    """Export ``project`` as a portable bundle (:func:`write_archive`, on its terms) and record
    the export in the project's log."""
    return write_archive(project, output_path=output_path, output_dir=output_dir,
                         include_models=include_models)


def write_archive(
    project: Path, *, output_path: str = "", output_dir: str = "", include_models: bool = False,
) -> dict:
    """Export ``project`` as a portable bundle: a ZIP file, or, given ``output_dir`` instead of
    ``output_path``, the identical bundle written as a directory tree.

    Composes the bundle from the shared membership accounting
    (:func:`tcip_mcp.tools.bundle.account_for`): every record or log a derived root of this tree
    claims (images under ``<root>/images/<date>/``, ground truth under
    ``<root>/annotations/<date>/<stem>.json``, the nested registry ``<root>/subjects.json``,
    ``.tcip`` state, experiments, sweeps and their claimed manifests), plus every recognized blob
    home. ``include_models`` narrows checkpoint blobs wherever
    :func:`~tcip_mcp.tools.bundle.blob_home` recognizes one (a registry-named path,
    ``.tcip/models``, or a ``.pt`` file shaped as a run's own under ``.tcip/experiments``); a
    bespoke run's ``model_src/`` snapshot travels regardless.

    Exactly one of ``output_path``/``output_dir`` must be given. ``output_dir`` refuses a
    destination inside the project (a bundle cannot contain the tree it was drawn from) and a
    destination that already holds anything.

    Every database under the tree is exported to its loose files first, through
    :func:`tcip_store.export.export_root`. The archive refuses only when that export fails, a store
    becomes unreadable, or a split manifest sits somewhere the derivation constraints
    exclude.

    ``left_behind`` names what this door declined to bundle, per class: ``unaccounted`` (a render
    cache, Ray's own experiment store, tensorboard events, any other stray no store or blob home
    claims), ``bookkeeping`` (a live tree's own transient bookkeeping, e.g. a lock file mid-write),
    and ``checkpoints_excluded`` (every checkpoint blob ``blob_home`` recognizes, dropped by
    ``include_models=False``).

    ``size_bytes`` in the response means one thing under ``output_path`` (the written ZIP's own
    compressed byte count, ``stat().st_size`` on the archive) and a different thing under
    ``output_dir`` (the sum of the copied members' own uncompressed byte counts); which one the
    caller is reading is decided by which of ``output_dir``/``output_path`` the response carries.

    Args:
        output_path: Destination path for the ZIP file; a relative path is under the project.
        output_dir: Destination directory to write the bundle into as a tree, instead of a ZIP; the
            same path-resolution rule as ``output_path``.
        include_models: Whether to include model checkpoints (can be large).
    """
    if output_path and output_dir:
        return {"error": "give either output_path (a ZIP file) or output_dir (a directory "
                         "tree), not both"}
    if not output_path and not output_dir:
        return {"error": "give either output_path (a ZIP file) or output_dir (a directory "
                         "tree) to archive into"}

    root = Path(project).resolve()
    if not root.is_dir():
        return {"error": f"Project directory not found: {project}"}

    resolved_output_dir: Path | None = None
    if output_dir:
        resolved_output_dir = Path(project, output_dir).resolve()
        try:
            resolved_output_dir.relative_to(root)
        except ValueError:
            pass
        else:
            return {"error": f"output_dir {resolved_output_dir} is inside the project being "
                             f"archived ({root}); choose a destination outside the project"}
        if resolved_output_dir.exists():
            if not resolved_output_dir.is_dir():
                return {"error": f"output_dir {resolved_output_dir} exists and is not a "
                                 "directory"}
            if any(resolved_output_dir.iterdir()):
                return {"error": f"output_dir {resolved_output_dir} is not empty; "
                                 "archive_project's directory mode never writes into a "
                                 "populated directory"}

    from tcip_store import StoreError

    try:
        _export_stores(root)
        before = _database_counters(root)
    except StoreError as exc:
        return {"error": f"a store database under {root} could not be exported before "
                         f"archiving, so the bundle cannot be vouched for: {exc}"}

    from tcip_store import SchemaVersionRefused

    from tcip_mcp.model_registry import RegistryVersionRefused
    from tcip_mcp.tools.bundle import BLOB_CHECKPOINTS, AnchorMisplaced, account_for, blob_home

    try:
        accounting = account_for(root)
    except (AnchorMisplaced, RegistryVersionRefused, SchemaVersionRefused) as exc:
        return {"error": str(exc)}

    # A registered checkpoint is not confined to .tcip/models; blob_home is the one recognizer.
    is_checkpoint = {
        p: blob_home(root, p, accounting.registered_checkpoints) == BLOB_CHECKPOINTS
        for p in accounting.blobs
    }
    members = [entry.path for plan in accounting.plans for entry in plan.entries]
    members += [p for p in accounting.blobs if include_models or not is_checkpoint[p]]

    out: Path | None = None
    created_output_dir = False
    if resolved_output_dir is not None:
        created_output_dir = not resolved_output_dir.exists()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        files_added, size_bytes = _write_bundle_directory(resolved_output_dir, root, members)
    else:
        out = Path(project, output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        files_added = _write_bundle_zip(out, root, members)
        size_bytes = out.stat().st_size

    def _remove_partial_bundle() -> None:
        # Only what this door itself wrote: the members it copied, and the directory only if it
        # created that directory (an empty one the caller already owned is admitted above).
        if resolved_output_dir is not None:
            for member in members:
                (resolved_output_dir / member.relative_to(root)).unlink(missing_ok=True)
            if created_output_dir:
                shutil.rmtree(resolved_output_dir, ignore_errors=True)
        elif out is not None:
            out.unlink(missing_ok=True)

    try:
        after = _database_counters(root)
    except StoreError as exc:
        _remove_partial_bundle()
        return {"error": f"a store database under {root} became unreadable while this project "
                         f"was being archived, so the bundle cannot be vouched for: {exc}. The "
                         "incomplete archive was removed."}
    moved = sorted(
        f"{store} in {db_path}"
        for (db_path, store), counter in after.items()
        if before.get((db_path, store)) != counter
    )
    if moved:
        _remove_partial_bundle()
        return {"error": "this project changed while it was being archived, so the bundle would "
                         f"hold a mix of before and after: {'; '.join(moved)}. The incomplete "
                         "archive was removed; stop the writers and archive again."}

    checkpoints_excluded = 0 if include_models else sum(1 for v in is_checkpoint.values() if v)
    result = {
        "files_added": files_added,
        "size_bytes": size_bytes,
        "include_models": include_models,
        "left_behind": {
            "unaccounted": len(accounting.unaccounted),
            "bookkeeping": len(accounting.bookkeeping),
            "checkpoints_excluded": checkpoints_excluded,
        },
    }
    result["output_dir" if resolved_output_dir is not None else "output_path"] = str(
        resolved_output_dir if resolved_output_dir is not None else out)
    return result


_IMPORTS_DIRNAME = ".imports"


def _extract_zip(zp: Path, staging: Path) -> int:
    """Extract every member of ``zp`` into ``staging``, refusing a path that would escape it.

    ``staging`` is this run's own private directory (a fresh uuid under ``.imports``), so a refusal
    here leaves the destination untouched. Escape is decided by containment (``Path.relative_to``),
    never a string prefix.
    """
    files_extracted = 0
    with zipfile.ZipFile(str(zp), "r") as zf:
        staged = staging.resolve()
        for info in zf.infolist():
            target = staging / info.filename
            resolved = target.resolve()
            try:
                resolved.relative_to(staged)
            except ValueError:
                raise ValueError(f"Unsafe path in archive: {info.filename}") from None
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
    """Remove a crash leftover from an earlier run: a staging sibling whose lock this process
    can take without blocking. A sibling still locked is a concurrent import's live work
    (including one in its own rename window) and is left alone, its lock file included."""
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


def _remove_staged_bookkeeping(staged: Path) -> None:
    """Remove adoption's own transition-lock files (the ``.lock`` sidecars) left inside the staged
    tree; the database file itself stays.
    """
    for path in staged.rglob("*.lock"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _adopt_accounted_roots(accounting) -> dict[str, int]:
    """Adopt every derived root whose plan has at least one entry; skip an empty one.

    Returns the adopted-entry count per root path, for the response.
    """
    from tcip_store.adoption import adopt_root

    adopted: dict[str, int] = {}
    for plan in accounting.plans:
        if not plan.entries:
            continue
        result = adopt_root(plan.root, plan.layout, report=lambda _line: None)
        adopted[plan.root] = sum(result.records.values()) + sum(result.log_entries.values())
    return adopted


def _move_staging_onto_destination(staged: Path, dest: Path, *, timeout_s: float) -> None:
    """Rename the staged tree onto ``dest``, re-checking emptiness immediately before the rename.

    Both the empty-destination removal and the rename itself run under
    ``tcip_store.file_backend.retry_while_denied``'s budget.
    """
    from tcip_store.file_backend import retry_while_denied

    def _prepare_and_rename() -> None:
        if dest.exists():
            if any(dest.iterdir()):
                raise StoreErrorRuntime(f"destination {dest} is no longer empty; refusing the move")
            dest.rmdir()
        os.rename(str(staged), str(dest))

    retry_while_denied(_prepare_and_rename, timeout_s)


class StoreErrorRuntime(RuntimeError):
    """Raised inside the retried move body; caught outside the retry as a tool refusal."""


def import_project(bundle_path: str, destination: str) -> dict:
    """Import an annotation project from a bundle ``archive_project`` wrote: a ZIP archive, or a
    directory tree written by its ``output_dir`` mode, and record one ``import_project`` line in
    the imported project's own log.

    A directory bundle is staged through the identical walker a ZIP bundle is
    (:func:`_stage_bundle`), so the two are read back exactly alike below this point.

    The door extracts into a private staging directory, classifies every member through the shared
    bundle accounting (:func:`tcip_mcp.tools.bundle.account_for`), refuses the whole import naming
    each bookkeeping, cross-root-collided, undecodable or unaccounted member, then adopts what is
    left into a database when this process is bound to the database backend (skipping any derived
    root whose plan is empty) or leaves the loose layout as is under the file backend, and only
    then renames the staged tree onto ``destination``.

    ``destination`` must not already exist, or must be an empty directory: this door merges nothing
    into a live project.

    A refusal at any step leaves the destination exactly as it was (absent, or its original empty
    state); the staging tree this run made is removed whether the run refused, raised, or
    succeeded.

    A registry that is not the entries mapping refuses the whole import before anything is moved.

    The response carries per-root adopted counts, blob counts per class, ``database_built``
    (whether adoption ran or the file layout was kept), ``dataset_paths_unresolved`` (the
    registered datasets whose absolute path stayed verbatim because they are outside the imported
    tree), ``checkpoint_paths_unresolved`` (registered checkpoint paths that are supposed to
    resolve under the destination and do not), ``external_checkpoints`` (registered checkpoints
    whose path is a designed-external claim, each with whether it currently exists), and
    ``files_extracted``.

    Args:
        bundle_path: Path to the ``.tcip.zip`` archive, or the directory tree written by
            ``archive_project``'s ``output_dir`` mode.
        destination: Directory to extract into.
    """
    from filelock import Timeout as _LockTimeout

    from tcip_mcp.audit import record_event_or_raise
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S, lock_file_for, path_lock

    bp = Path(bundle_path)
    if not bp.is_file() and not bp.is_dir():
        return {"error": f"bundle not found: {bundle_path}"}

    dest = Path(destination).expanduser().resolve()
    if dest.exists():
        if not dest.is_dir():
            return {"error": f"destination {dest} exists and is not a directory"}
        if any(dest.iterdir()):
            return {"error": f"destination {dest} is not empty; import_project never writes "
                             "into an existing project (a destination with state merges nothing; "
                             "that stays operator work)"}

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
        return {"error": f"could not lock a fresh staging directory at {staging}"}
    finally:
        lock_file_for(staging).unlink(missing_ok=True)
    if "error" not in result:
        record_event_or_raise("import_project", {"bundle_path": bundle_path}, scope=dest)
    return result


def _run_import_into_staging(bp: Path, staging: Path, dest: Path) -> dict:
    """Everything the import door does while it holds the staging lock: stage, account for,
    decode-preflight, adopt (backend-conditional), then move. Returns the tool's own response
    dict, an ``{"error": ...}`` on any refusal.
    """
    from tcip_store.adoption import preflight_decode, unaccounted_files
    from tcip_store.binding import is_database_backend
    from tcip_store.errors import DecodeError as StoreDecodeError
    from tcip_store.errors import StoreError
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    from tcip_mcp.model_registry import RegistryVersionRefused
    from tcip_mcp.tools.bundle import (
        AnchorMisplaced, account_for, blob_home, external_registered_checkpoints,
        unresolved_registered_checkpoints,
    )

    try:
        files_extracted = _stage_bundle(bp, staging)
    except (ValueError, zipfile.BadZipFile) as exc:
        return {"error": f"{bp} is not a readable bundle: {exc}"}

    try:
        accounting = account_for(staging)
    except (AnchorMisplaced, StoreError, RegistryVersionRefused) as exc:
        return {"error": str(exc)}

    tree = accounting.tree
    if accounting.bookkeeping:
        named = ", ".join(str(p.relative_to(tree)) for p in accounting.bookkeeping)
        return {"error": f"the archive carries backend bookkeeping ({named}), which a file bundle "
                         "never legitimately holds; refusing the whole import"}
    if accounting.collisions:
        named = ", ".join(str(p.relative_to(tree)) for p in accounting.collisions)
        return {"error": f"{named} would be claimed by more than one derived root of this "
                         "project at once; refusing rather than guessing which one owns it"}
    if accounting.unaccounted:
        named = ", ".join(str(p.relative_to(tree)) for p in accounting.unaccounted)
        return {"error": f"the archive carries member(s) no store or blob home claims ({named}); "
                         "refusing the whole import rather than silently dropping them"}

    left_over = unaccounted_files(accounting.plans)
    if left_over:
        named = ", ".join(str(p.relative_to(tree)) for p in left_over)
        return {"error": f"{named} matched a store's claim but resolved to no adoptable entry; "
                         "refusing the whole import"}

    try:
        preflight_decode(accounting.plans)
    except StoreDecodeError as exc:
        return {"error": str(exc)}

    database_built = False
    adopted: dict[str, int] = {}
    if is_database_backend():
        try:
            adopted = _adopt_accounted_roots(accounting)
        except StoreError as exc:
            return {"error": f"adoption refused: {exc}"}
        database_built = True
        _remove_staged_bookkeeping(staging)

    try:
        _move_staging_onto_destination(staging, dest, timeout_s=DEFAULT_LOCK_TIMEOUT_S)
    except (OSError, StoreErrorRuntime) as exc:
        return {"error": f"could not move the staged import onto {dest}: {exc}"}

    blob_classes: dict[str, int] = {}
    for blob in accounting.blobs:
        home = blob_home(tree, blob, accounting.registered_checkpoints)
        blob_classes[home] = blob_classes.get(home, 0) + 1

    dataset_paths_unresolved = _external_dataset_paths(dest)
    checkpoint_paths_unresolved = list(unresolved_registered_checkpoints(dest))
    external_checkpoints = list(external_registered_checkpoints(dest))

    return {
        "destination": str(dest),
        "files_extracted": files_extracted,
        "database_built": database_built,
        "adopted": adopted,
        "blob_counts": blob_classes,
        "dataset_paths_unresolved": dataset_paths_unresolved,
        "checkpoint_paths_unresolved": checkpoint_paths_unresolved,
        "external_checkpoints": external_checkpoints,
    }


def _external_dataset_paths(project: Path) -> list[str]:
    """The imported project's own registered dataset entries that stay absolute (external)."""
    from tcip_mcp.registry_paths import is_external_form

    return sorted(str(e["path"]) for e in read_datasets(project)
                  if is_external_form(str(e["path"])))


@tool()
@audited
def delete_stray_state_file(project: Path, relative_path: str, reason: str) -> dict:
    """Delete one stray file under the project's ``.tcip/state`` root: a file
    ``stray_state.stray_state_files`` lists, and the doctor's own finding names, that no store
    claims, no blob home claims, and that is not the storage backend's own bookkeeping.

    ``reason`` is required and non-empty, recorded on this door's own audit line.

    Refuses, before any write, naming why: an empty ``reason``; ``relative_path`` empty, absolute, carrying a ``..`` segment, or resolving
    outside the state root; a path under the state root's own database home; a path that does not
    exist; a directory; a link or junction; a path the accounting classifies as the storage
    backend's own bookkeeping, as a claimed store's own file (naming the store), or as a recognized
    blob; and a state root whose own accounting refuses (a misplaced split manifest anchor, or
    two stores claiming one file equally).

    Only the named file is removed (``os.remove``); an emptied parent directory is left exactly as
    it stands.
    """
    if not (reason or "").strip():
        return {"error": "delete_stray_state_file needs a non-empty reason: the confirmation "
                         "with the person this destructive act requires."}

    from tcip_mcp.stray_state import stray_state_file_refusal

    target, refusal = stray_state_file_refusal(project, relative_path)
    if refusal is not None:
        return {"error": refusal}

    size_bytes = target.stat().st_size
    os.remove(target)

    return {
        "deleted": str(target),
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "reason": reason,
    }

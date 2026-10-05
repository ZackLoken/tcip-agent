"""Project management tools."""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path

import tcip_store
from tcip_store import Key, VersionConflict, encode_record

from tcip_mcp.server import tool
from tcip_mcp.audit import audited


def _project_dir(project_path: str) -> Path:
    """Return the .tcip directory for a project, creating it if needed."""
    p = Path(project_path) / ".tcip"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- dataset identity registry (project -> datasets it uses) --------------


DATASET_REGISTRY_STORE = "dataset_registry"
_DATASET_REGISTRY_PARTS = ("datasets",)


def dataset_registry_key(project: str | Path) -> Key:
    """The project's record of which datasets it uses, keyed by dataset id."""
    return Key(DATASET_REGISTRY_STORE, str(Path(project)), _DATASET_REGISTRY_PARTS)


def read_datasets(project: str | Path) -> list[dict]:
    """The project's dataset registry (``[{id, path}]``), or [] when absent.

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
        regs = [r for r in txn.read(key, default=[]) if r["id"] != entry["id"]]
        regs.append(entry)
        txn.write(key, sorted(regs, key=lambda r: r["id"]))


@tool()
@audited(scope_arg="dataset_root")
def register_dataset(project: Path, dataset_root: str, crop: str) -> dict:
    """Record a dataset's identity, ``{crop, id, fingerprint}``, with the dataset, and its id and
    path (:func:`~tcip_mcp.registry_paths.stored_path`) in the project's dataset registry. ``id``
    is minted once and kept across re-runs and moves, ``fingerprint`` is
    :func:`~tcip_mcp.pipelines.data.dataset_fingerprint.dataset_fingerprint` now, and the identity
    write is compare-and-set against the version this call read, a conflict keeping the id that
    committed.

    Args:
        dataset_root: Root of the dataset (holds ``images/``, ``subjects.json`` and the database
            its label documents are records in).
        crop: The crop this dataset's imagery is of, as ``crops.yml`` names it. Required; the
            expert's fact.
    """
    from tcip_mcp.dataset_layout import decode_dataset_identity_document, dataset_identity_path
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.project_record import mint_id
    from tcip_mcp.registry_paths import stored_path

    root = Path(dataset_root)
    if not root.is_dir():
        return {"error": f"dataset_root not found: {dataset_root}"}
    if not crop:
        return {"error": "crop is required (the expert's fact; never inferred from a path or slug)"}

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
                existing = decode_dataset_identity_document(stored.value, dataset_root=root)
            except ValueError as exc:
                return {"error": f"{exc}; minting a fresh id over it would sever every record "
                                 "that cites the old one"}
        candidate = {
            "crop": crop,
            "id": existing.get("id") or mint_id(),
            "fingerprint": fingerprint,
        }
        try:
            tcip_store.put_blob(ident_path, encode_record(candidate), expect=stored.version)
        except VersionConflict:
            continue
        identity = candidate
        break

    upsert_dataset(project, {"id": identity["id"], "path": stored_path(root, project)})
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
    record_event_or_raise("project_created", dict(record), actor=None, scope=project)
    return {"project_path": str(project), **record}


@tool()
def view_gui_state(project: Path) -> dict:
    """The GUI state the human last left in this project, from its GUI snapshot
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


_ACTIVITY_TOOLS = ("report_friction", "write_retrospective", "record_distillation_pass")


def _recent_activity(project: Path) -> dict:
    """The project's recent memory activity, derived from its audit log's ``report_friction``,
    ``write_retrospective`` and ``record_distillation_pass`` lines in the order they landed: the
    reports since the last retrospective and since the last distillation pass, the retrospectives
    since that pass, the last retrospective's ``project_id`` and time, the last pass's time, and
    the time of the last report or retrospective. A log carrying undecodable entries answers
    ``status_unavailable`` naming how many.
    """
    from tcip_mcp.audit import acts_of

    try:
        lines, _cursor = acts_of(project, _ACTIVITY_TOOLS)
    except ValueError as exc:
        return {"status_unavailable": str(exc)}
    activity: dict = {"reports_since_last_retrospective": 0, "reports_since_last_distillation": 0,
                      "retrospectives_since_last_distillation": 0}
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
            activity["last_retrospective"] = {"project_id": line["arguments"]["project_id"],
                                              "modified_at": line["timestamp"]}
        activity["last_activity"] = line["timestamp"]
    return activity


def _write_bundle(dest: Path, as_zip: bool, root: Path, members: list[Path]) -> None:
    """Write ``members`` (each an absolute path under ``root``) into ``dest`` under their paths
    relative to ``root``, as a ZIP when ``as_zip`` and as the identical tree otherwise, each store
    database as the consistent copy :func:`tcip_store.sqlite_backend.copy_database` takes.
    Removes ``dest`` on any failure.
    """
    import tempfile
    from contextlib import ExitStack

    from tcip_store.file_backend import database_file, database_roots
    from tcip_store.sqlite_backend import copy_database

    databases = {database_file(str(r)) for r in database_roots(root)}
    try:
        with ExitStack() as stack:
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            zf = (stack.enter_context(zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED))
                  if as_zip else None)
            for n, member in enumerate(sorted(members)):
                source = member
                if member in databases:
                    source = scratch / f"{n}.db"
                    copy_database(member, source)
                if zf is not None:
                    zf.write(source, member.relative_to(root))
                else:
                    target = dest / member.relative_to(root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    except BaseException:
        if as_zip:
            dest.unlink(missing_ok=True)
        else:
            shutil.rmtree(dest, ignore_errors=True)
        raise


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

    The bundle is every file under the project but the store's own bookkeeping
    (:func:`tcip_store.file_backend.is_bookkeeping`), each store database as a consistent copy
    taken through SQLite's own backup, and without the checkpoints
    :func:`~tcip_mcp.model_registry.checkpoint_files` names unless ``include_models``;
    ``checkpoints_excluded`` counts those left out.

    Exactly one of ``output_path``/``output_dir`` must be given, as an absolute path. Either
    refuses a relative path, a destination inside the project (a bundle cannot contain the tree it
    was drawn from), and a destination that already exists, except an empty directory given as
    ``output_dir``.

    ``size_bytes`` in the response means one thing under ``output_path`` (the written ZIP's own
    compressed byte count, ``stat().st_size`` on the archive) and a different thing under
    ``output_dir`` (the sum of the copied members' own uncompressed byte counts); which one the
    caller is reading is decided by which of ``output_dir``/``output_path`` the response carries.

    Args:
        output_path: Destination path for the ZIP file.
        output_dir: Destination directory to write the bundle into as a tree, instead of a ZIP.
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
    dest = Path(output_path or output_dir)
    if not dest.is_absolute():
        return {"error": f"the destination {dest} is a relative path; name it absolutely"}
    dest = dest.resolve()
    if dest.is_relative_to(root):
        return {"error": f"the destination {dest} is inside the project being archived ({root}); "
                         "choose a destination outside the project"}
    if dest.exists() and not (output_dir and dest.is_dir() and not any(dest.iterdir())):
        return {"error": f"the destination {dest} already exists; an archive never writes over "
                         "or into anything"}

    from tcip_store.file_backend import is_bookkeeping

    from tcip_mcp.model_registry import checkpoint_files

    checkpoints = frozenset() if include_models else checkpoint_files(root)
    files = [p for p in root.rglob("*") if p.is_file() and not is_bookkeeping(p.name)]
    members = [p for p in files if p not in checkpoints]

    (dest.parent if output_path else dest).mkdir(parents=True, exist_ok=True)
    _write_bundle(dest, bool(output_path), root, members)
    size_bytes = (dest.stat().st_size if output_path
                  else sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()))
    return {"files_added": len(members), "size_bytes": size_bytes,
            "include_models": include_models, "checkpoints_excluded": len(files) - len(members),
            ("output_path" if output_path else "output_dir"): str(dest)}


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
        record_event_or_raise("import_project", {"bundle_path": bundle_path}, actor=None,
                              scope=dest)
    return result


def _run_import_into_staging(bp: Path, staging: Path, dest: Path) -> dict:
    """Everything the import door does while it holds the staging lock: stage, refuse the store's
    own bookkeeping, then move. Returns the tool's own response dict, an ``{"error": ...}`` on any
    refusal.
    """
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S, is_bookkeeping

    from tcip_mcp.model_registry import registry_checkpoint_disclosures

    try:
        files_extracted = _stage_bundle(bp, staging)
    except (ValueError, zipfile.BadZipFile) as exc:
        return {"error": f"{bp} is not a readable bundle: {exc}"}

    bookkeeping = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*")
                         if p.is_file() and is_bookkeeping(p.name))
    if bookkeeping:
        return {"error": f"the archive carries the store's own bookkeeping "
                         f"({', '.join(bookkeeping)}), which a bundle never legitimately holds; "
                         "refusing the whole import"}

    try:
        _move_staging_onto_destination(staging, dest, timeout_s=DEFAULT_LOCK_TIMEOUT_S)
    except (OSError, StoreErrorRuntime) as exc:
        return {"error": f"could not move the staged import onto {dest}: {exc}"}

    return {
        "destination": str(dest),
        "files_extracted": files_extracted,
        "dataset_paths_unresolved": _external_dataset_paths(dest),
        **registry_checkpoint_disclosures(dest),
    }


def _external_dataset_paths(project: Path) -> list[str]:
    """The imported project's own registered dataset entries that stay absolute (external)."""
    from tcip_mcp.registry_paths import is_external_form

    return sorted(str(e["path"]) for e in read_datasets(project)
                  if is_external_form(str(e["path"])))

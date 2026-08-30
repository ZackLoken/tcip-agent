"""Project management tools."""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import tcip_store
from tcip_store import (
    RECORD_JSON,
    DecodeError,
    Key,
    StoreDescriptor,
    VersionConflict,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
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


def dataset_registry_key(project_root: str | Path) -> Key:
    """The project's record of which datasets it uses, keyed by dataset id.

    ``cas``: :func:`upsert_dataset` reads the whole list, replaces one entry and writes it
    back, so an unconditional write drops a dataset another registration had just added.
    """
    return Key(DATASET_REGISTRY_STORE, str(Path(project_root).absolute()), _DATASET_REGISTRY_PARTS)


def _registry_entries(document: object) -> list[dict]:
    """The registry's entries, from a document that decoded: the one normalization both the
    reader and the upsert share, so they cannot disagree about what a non-list document means."""
    return document if isinstance(document, list) else []


def read_datasets(project_root: str | Path) -> list[dict]:
    """The project's dataset registry (``[{id, path, crop, fingerprint}]``), or [] when absent.

    A registry present but undecodable raises rather than reading as empty: an empty answer
    here would make the next :func:`upsert_dataset` write a list holding one entry and drop
    every other dataset identity the project had recorded.
    """
    return _registry_entries(tcip_store.read(dataset_registry_key(project_root), default=[]))


def registry_path_for(dataset_root: str | Path, project_root: str | Path) -> str:
    """What a dataset registry entry stores for ``path``: relative to ``project_root`` when
    ``dataset_root`` is that root or sits under it, the project's own tree becoming ``"."``;
    absolute otherwise, and a ``".."`` form is never produced.

    Containment is decided by filesystem identity (``os.path.samefile`` over the resolved
    dataset root's own ancestors), never a string or ``Path.relative_to`` comparison on the
    caller's own spellings, so a case variant, an alias or a junction of either root reads
    exactly as the filesystem sees it. One implementation, called by :func:`register_dataset`
    and by ``scripts/conform_dataset_registry_paths.py``, the one-off script that carries an
    already-registered project onto this rule. Absolute (unchanged) whenever either side is not
    an existing directory: there is nothing to compare a missing path against. A relative form
    is stored with POSIX separators (``as_posix()``), so a nested dataset's entry (a deeper
    relative form than the project's own ``"."``) reads the same after a cross-machine move;
    :func:`dataset_entry_path` resolves it back by posix parts.
    """
    dataset_root, project_root = Path(dataset_root), Path(project_root)
    if dataset_root.is_dir() and project_root.is_dir():
        resolved = dataset_root.resolve()
        for ancestor in (resolved, *resolved.parents):
            try:
                if os.path.samefile(ancestor, project_root):
                    return resolved.relative_to(ancestor).as_posix()
            except OSError:
                continue
        return str(resolved)
    return str(dataset_root)


def entry_is_external(entry: dict) -> bool:
    """Whether this registry entry names a dataset outside the project's own tree.

    An external dataset is the one kind :func:`registry_path_for` stores absolute; every other
    entry is the project's own tree or a directory under it, stored relative. The one spelling
    of this test, so a caller asking "is this dataset external" agrees with the writer's own
    rule rather than re-deriving it.
    """
    path = entry.get("path")
    if not path:
        return False
    return Path(path).is_absolute()


def dataset_entry_path(project_root: str | Path, entry: dict) -> Path:
    """The absolute path a dataset registry ``entry`` names, resolving a relative ``path``
    (the project's own tree, stored ``"."`` or a deeper relative form by
    :func:`registry_path_for`) against ``project_root``; an already-absolute ``path`` (an
    external dataset) is returned unchanged.

    The one place a registry entry's location becomes a path: every reader of
    :func:`read_datasets` calls this rather than re-deriving the resolution, so a project's own
    relative entry and an external dataset's absolute one are handled identically wherever the
    registry is read. A relative ``path`` is joined by its POSIX parts (``registry_path_for``'s
    own storage form) rather than as a native path string, so a nested entry resolves the same
    whichever platform wrote or reads it.
    """
    path = entry.get("path")
    if not path:
        raise ValueError(f"dataset registry entry {entry!r} carries no path")
    if entry_is_external(entry):
        return Path(path)
    return Path(project_root).joinpath(*PurePosixPath(path).parts)


def upsert_dataset(project_root: str | Path, entry: dict) -> None:
    """Add or refresh a dataset in the project's registry, matched by ``id``: a moved dataset updates
    the ``path`` of its existing id rather than duplicating, so identity survives a move.

    The read and the write are one transaction, so two registrations running at once cannot
    each write a list assembled from the state before the other's entry landed.
    """
    key = dataset_registry_key(project_root)
    with tcip_store.transaction(key) as txn:
        regs = [r for r in _registry_entries(txn.read(key, default=[]))
                if r.get("id") != entry.get("id")]
        regs.append(entry)
        txn.write(key, sorted(regs, key=lambda r: str(r.get("id", ""))))


@mcp.tool()
@audited(scope_arg="dataset_root")
def register_dataset(dataset_root: str, crop: str, project_root: str = "") -> dict:
    """Record a dataset's identity so a delivered number can be traced to the exact data behind it.

    Writes ``<dataset_root>/dataset.json = {crop, id, fingerprint}`` (identity travels with the data)
    and upserts the dataset into the project's ``.tcip/datasets.json``. ``crop`` is the human's fact and
    is required, never inferred from a path or slug. ``id`` is minted once and preserved across
    re-runs and path moves; ``fingerprint`` is the whole-dataset content digest (labels + image pixels
    + registry + confirmed negatives), recomputed here, but the stored value is a cache, and
    recompute-on-read (``resolution.dataset_fingerprint``) is the authority.

    The identity write is compare-and-set against the version this call read, so two first-time
    registrations cannot each mint an id and leave the loser's id cited by records the winner's
    document no longer names. A conflict re-reads what committed and keeps the id it carries, and
    the project registry is reconciled against that committed id rather than the one this call
    proposed.

    The registry's stored ``path`` (see :func:`registry_path_for`) is relative to
    ``project_root`` whenever the dataset sits under it, the project's own tree becoming
    ``"."``; a genuinely external dataset stays absolute. A relative entry resolves at whatever
    path the project itself is opened from, so a project archived and imported elsewhere, or
    renamed in place, needs no operator rewrite of its own registry.

    Args:
        dataset_root: Root of the dataset (holds ``images/``, ``annotations/``, ``classes.json``).
        crop: The crop this dataset's imagery is of (e.g. ``hazelnut``). Required; the expert's fact.
        project_root: Project to register the dataset under. Empty defaults to ``dataset_root``.
    """
    from tcip_store import SchemaVersionRefused

    from tcip_mcp.dataset_layout import decode_dataset_identity, dataset_identity_key
    from tcip_mcp.pipelines.resolution import dataset_fingerprint

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
                existing = decode_dataset_identity(stored.value, dataset_root=root)
            except SchemaVersionRefused as exc:
                return {"error": f"{exc} Re-registering here would overwrite a newer writer's "
                                 "identity document; nothing was written."}
            except ValueError as exc:
                return {"error": f"{exc}; minting a fresh id over it would sever every record "
                                 "that cites the old one"}
        candidate = {
            "crop": crop,
            "id": existing.get("id") or uuid.uuid4().hex[:12],  # minted once, then kept
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

    proj = Path(project_root) if project_root else root
    upsert_dataset(proj, {"id": identity["id"], "path": registry_path_for(root, proj),
                          "crop": crop, "fingerprint": fingerprint})
    return {"dataset_root": str(root), **identity}


def _scaffold_project(project_path: str, site: str) -> dict:
    """Create ``.tcip/`` with its artifacts and models directories, and record the project's site.

    The internals of :func:`init_project`, factored out so other tools that
    stand up a project (e.g. ``ingest_images``) reuse the exact same scaffolding
    instead of re-implementing it. The directories are idempotent: re-running only re-mkdirs.

    ``site`` is validated before anything is created, so a refused site leaves nothing on disk,
    the same as the name-scheme refusal each door already holds to. The site is then written
    last, by :func:`tcip_mcp.project_record.record_site`, a create-only write: an absent record
    is written, a present record with the same site is left as is, and a present record with a
    different or unreadable site raises (``ValueError`` or ``StoreError``, per
    :func:`~tcip_mcp.project_record.record_site`'s own contract).
    """
    from tcip_mcp.project_record import record_site, validate_site

    validate_site(site)
    tcip = _project_dir(project_path)
    (tcip / "artifacts").mkdir(exist_ok=True)
    (tcip / "models").mkdir(exist_ok=True)
    recorded = record_site(project_path, site)

    return {
        "project_path": project_path,
        "tcip_dir": str(tcip),
        "created": [".tcip/", ".tcip/artifacts/", ".tcip/models/"],
        "site": recorded["site"],
    }


@mcp.tool()
@audited
def init_project(project_path: str, site: str) -> dict:
    """Initialise a TCIP project directory.

    Creates ``.tcip/`` with its artifacts and models directories and records the project's
    site. When ``project_path`` is directly under the workspace, its basename must fit
    ``crop_subject_phenotype`` (``workspace.format_project_name``/``parse_project_name``);
    a path outside the workspace is not a workspace project and is not held to the scheme.

    Args:
        project_path: Root directory of the project.
        site: The orchard or station this project's plants stand in, in the breeder's own
            words. Ask the breeder rather than guessing it from a path or filename; a project
            that already records a different site refuses rather than overwriting it.
    """
    from tcip_store import StoreError

    from tcip_mcp import workspace

    p = Path(project_path).expanduser().resolve()
    if p.parent == workspace.workspace_root():
        try:
            workspace.parse_project_name(p.name)
        except ValueError as exc:
            return {"error": str(exc)}
    try:
        # The resolved path, so a relative project_path scaffolds where the name check above
        # just resolved it, rather than the record write refusing a relative root afterward.
        return _scaffold_project(str(p), site)
    except (ValueError, StoreError) as exc:
        return {"error": str(exc)}


@mcp.tool()
@audited
def set_active_project(name: str) -> dict:
    """Set the workspace's active project so the GUI opens it.

    Writes the workspace active-project marker (``<workspace>/.active``) and notifies a
    running GUI to open the project: the loop-closer for the breeder flow ("I structured
    your images into ``<crop>_<subject>_<phenotype>``, opening it now"). ``name`` is an
    existing workspace project's directory name; adoption opens what is there rather than
    creating anything, so any safely-named project is adoptable, conforming or not.

    The notification also carries whether the web backend repinned its own platform-state
    root on it (``backend_repinned``, a bool), the root it repinned to (``backend_platform_root``),
    or why it could not (``backend_root_problem``): when the backend is down or the delivery
    fails, ``backend_repinned`` is ``False`` and the other two are ``None``, since it will bind
    from the marker at its own next start regardless.

    Args:
        name: The workspace project to make active.
    """
    from tcip_mcp import workspace
    from tcip_mcp.web_client import PANEL_EVENT_ACTIVE_PROJECT_CHANGED, post_panel_event

    try:
        marker = workspace.set_active_project(name)
        proj = workspace.project_path(name)
    except ValueError as exc:
        return {"error": str(exc)}

    delivery = post_panel_event(
        "app", PANEL_EVENT_ACTIVE_PROJECT_CHANGED, {"name": name, "project_path": str(proj)}
    )
    response = delivery.get("response") or {}
    backend_platform_root = response.get("platform_root")
    return {
        "name": name,
        "project_path": str(proj),
        "marker": str(marker),
        "gui_notified": bool(delivery.get("delivered")),
        "backend_repinned": backend_platform_root is not None,
        "backend_platform_root": backend_platform_root,
        "backend_root_problem": response.get("platform_root_problem"),
        "recent_activity": _recent_activity(str(proj)),
    }


def _resolve_project_path(project_path: str) -> str:
    from tcip_mcp import workspace
    return workspace.resolve_project_path(project_path)


def _root_divergence_report() -> dict[str, str] | None:
    """Whether this process's platform-state root disagrees with the workspace's
    active-project marker.

    Adopting a project repins the *adopting process's own* ``TCIP_PROJECT_ROOT`` at once
    (``workspace.set_active_project``); a separate process converges only when it itself binds
    from the marker, at its own startup or (the web backend) on the agent's adopt signal, so
    this process's root can keep naming a stale or different project until then.
    ``None`` when there is no marker, the marker names an adoptable project this process's
    root already matches, or the two agree. Carries ``marker_problem`` when the marker could
    not be used at all: a store refusal, a lock timeout, or a name
    :func:`tcip_mcp.workspace.adoptable_project_root` refuses to open, reported here rather
    than raised out of ``inspect_project``.

    Reads with ``create=False`` so this check, run on every ``inspect_project`` call, cannot
    bring the workspace directory into existence as a side effect.
    """
    from tcip_mcp import workspace
    from tcip_mcp.project_paths import project_root

    try:
        found = workspace.active_project_if_present(create=False)
    except Exception as exc:  # noqa: BLE001 - reported, never raised out of inspect_project
        return {"marker_problem": str(exc)}
    if found is None:
        problem = workspace.marker_problem(create=False)
        return {"marker_problem": problem} if problem else None
    _, marker_project = found
    root = project_root()
    if root == marker_project:
        return None
    return {
        "platform_root": str(root),
        "marker_project": str(marker_project),
        "action": "set_active_project",
    }


@mcp.tool()
@audited
def view_gui_state() -> dict:
    """The live GUI session the human is looking at: active project, dataset, date, trait, tab, and the
    exact current image. Lets the agent work through the app instead of globbing or asking which image
    is open. Reads the active-project marker + that project's gui.json. active_project is None when
    nothing is open.
    """
    from tcip_mcp import workspace
    from tcip_mcp.web_client import gui_snapshot_key

    name = workspace.read_active_project()
    if not name:
        return {"active_project": None,
                "note": "no active project; open one in the GUI or call set_active_project"}
    project_root = workspace.project_path(name)
    ctx: dict = {"active_project": name, "project_root": str(project_root)}
    try:
        gui = tcip_store.read(gui_snapshot_key(project_root), default=None)
    except (DecodeError, OSError) as e:
        ctx["error"] = f"could not read the GUI snapshot: {e}"
        return ctx
    if gui is None:
        ctx["note"] = "no GUI snapshot yet (the GUI has not persisted a selection for this project)"
        return ctx
    ds = gui.get("dataset") or {}
    image_list = ds.get("image_list") or []
    idx = ds.get("current_image_index") or 0
    dataset_root, date = ds.get("dataset_root"), ds.get("date")
    current_image = None
    if dataset_root and date and 0 <= idx < len(image_list):
        from tcip_mcp.dataset_layout import image_dir

        current_image = str(image_dir(dataset_root, date) / image_list[idx])
    ctx.update({
        "dataset_root": dataset_root,
        "subject": ds.get("subject"),
        "date": date,
        "active_tab": gui.get("active_tab"),
        "mode": gui.get("mode"),
        "n_images": len(image_list),
        "current_image_index": idx,
        "current_image": current_image,
    })
    return ctx


@mcp.tool()
@audited
def inspect_project(project_path: str = "") -> dict:
    """Get an overview of a TCIP project.

    Carries ``platform_root_diverges_from_marker`` when this process's platform-state root
    (``$TCIP_PROJECT_ROOT``) names a different project than the workspace's active-project
    marker: adoption repins only the adopting process, so the GUI and this process can end
    up naming different projects until both explicitly adopt. Carries
    ``platform_root_binding``, this process's own :class:`tcip_mcp.project_paths.RootBinding`
    as a dict, when either :func:`tcip_mcp.project_paths.pin_project_root` or
    :func:`tcip_mcp.project_paths.repin_platform_root` has run: absent under pytest until a
    ``set_active_project`` call repins, and absent for any other standalone use, since none of
    those call either.

    For a project with ``.tcip``, carries ``site`` and ``site_problem`` from
    ``tcip_mcp.project_record.site_fields``: exactly one is set, and ``site_problem`` names why
    there is no site (no record yet, a damaged one, or a root the store refuses to read). ``plant_
    mappings`` carries every mapping name persisted under the project, the same shape:
    ``plant_mappings_problem`` names why the listing came back empty when the root's state is a
    store the bound backend refuses to read (a root still in the loose-file layout under the
    database default), rather than raising and taking the whole overview down with it. A path
    with no ``.tcip`` carries neither, the same as it carries no other live-computed field.

    Args:
        project_path: Root directory of the project. Empty defaults to the active project.
    """
    from tcip_mcp.project_paths import root_binding

    project_path = _resolve_project_path(project_path)
    root = Path(project_path)
    tcip = root / ".tcip"

    status: dict = {"project_path": project_path, "initialized": tcip.is_dir()}
    divergence = _root_divergence_report()
    if divergence:
        status["platform_root_diverges_from_marker"] = divergence
    binding = root_binding()
    if binding is not None:
        status["platform_root_binding"] = {
            "root": str(binding.root),
            "source": binding.source,
            "inherited_root": binding.inherited_root,
            "marker_problem": binding.marker_problem,
        }
    if not tcip.is_dir():
        return status

    from tcip_mcp.project_record import site_fields

    status.update(site_fields(project_path))

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

    images_dir = dataset_layout.image_root(root)
    if images_dir.is_dir():

        status["image_count"] = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts
        )
        status["dates"] = dataset_layout.list_dates(root)

    from tcip_store import StoreError

    from tcip_mcp.pipelines.postprocessing.plant_mapping import plant_mapping_names

    try:
        status["plant_mappings"] = plant_mapping_names(root)
    except StoreError as exc:
        status["plant_mappings"] = []
        status["plant_mappings_problem"] = str(exc)

    status["recent_activity"] = _recent_activity(project_path)
    return status


def _recent_activity(project_path: str) -> dict:
    """The project's persisted status summary (recent report/retrospective/distillation
    activity), namespaced separately from the live-computed fields above it so a caller can tell
    freshly-computed-this-call fields from read-from-the-status-store ones, which may be stale or
    corrupt.
    """
    from tcip_mcp.project_status import read_project_status

    activity = read_project_status(project_path)
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
    """Write every database under this tree back out as files, the way an archive bundles state.

    Every record and log store of every database under the tree, not only the ones the doctor
    reads: an archive ships audit logs, experiment members and registry state too, and a bundle
    whose confirmed negatives restore as absent is the failure this exists to prevent. So the
    doors that create a project (which write the project record through a database) archive
    without an operator having run ``scripts/export_store.py`` first.
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


@audited
def archive_project(project_path: str, output_path: str = "", include_models: bool = False) -> dict:
    """Export an annotation project as a portable ZIP archive.

    Not an MCP tool: run through ``scripts/archive_project.py``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its own tests.

    Composes the bundle from the shared membership accounting
    (:func:`tcip_mcp.tools.bundle.account_for`), the same one ``import_project`` judges by: every
    record or log a derived root of this tree claims (images under ``<root>/images/<date>/``,
    ground truth under ``<root>/annotations/<date>/<stem>.json``, the nested registry
    ``<root>/classes.json``, ``.tcip`` state, experiments, sweeps and their claimed manifests),
    plus every recognized blob home. ``include_models`` narrows only the checkpoints under
    ``.tcip/models/*.pt``; a bespoke run's ``model_src/`` snapshot travels regardless, since it is
    one membership statement with one producer-side option, not two.

    Every database under the tree is exported to its loose files first, through the same
    :func:`tcip_store.export.export_root` ``scripts/export_store.py`` uses, so a project either
    creating door stood up (which writes the project record through a database) archives without
    an operator having run that script by hand. The archive refuses only when that export fails,
    a store becomes unreadable, or a split/curated manifest sits somewhere the derivation
    constraints exclude, never merely because a database was behind.

    ``left_behind`` names what this door declined to bundle, per class: ``unaccounted`` (a
    render cache, Ray's own experiment store, tensorboard events, any other stray no store or
    blob home claims), ``bookkeeping`` (a live tree's own transient bookkeeping, e.g. a lock
    file mid-write), and ``checkpoints_excluded`` (``.tcip/models/*.pt`` files dropped by
    ``include_models=False``), so the narrowing is disclosed rather than silent.

    Args:
        project_path: Root directory of the project.
        output_path: Destination path for the ZIP file. Defaults to ``<project_name>.tcip.zip``
            beside the project in the workspace; a relative path resolves against the project
            root, never the server process's cwd.
        include_models: Whether to include model checkpoints (can be large).
    """
    root = Path(project_path).resolve()
    if not root.is_dir():
        return {"error": f"Project directory not found: {project_path}"}

    from tcip_store import StoreError

    try:
        _export_stores(root)
        before = _database_counters(root)
    except StoreError as exc:
        return {"error": f"a store database under {root} could not be exported before "
                         f"archiving, so the bundle cannot be vouched for: {exc}"}

    from tcip_mcp.tools.bundle import AnchorMisplaced, account_for

    try:
        accounting = account_for(root)
    except AnchorMisplaced as exc:
        return {"error": str(exc)}

    if not output_path:
        output_path = str(root.parent / f"{root.name}.tcip.zip")
    else:
        from tcip_mcp.project_paths import resolve_output_path

        output_path = str(resolve_output_path(output_path))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    models_dir = root / ".tcip" / "models"
    members = [entry.path for plan in accounting.plans for entry in plan.entries]
    members += [
        p for p in accounting.blobs
        if include_models or p.parent != models_dir or p.suffix != ".pt"
    ]

    files_added = 0
    try:
        with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
            for member in sorted(members):
                zf.write(member, member.relative_to(root))
                files_added += 1
    except BaseException:
        out.unlink(missing_ok=True)
        raise

    try:
        after = _database_counters(root)
    except StoreError as exc:
        out.unlink(missing_ok=True)
        return {"error": f"a store database under {root} became unreadable while this project "
                         f"was being archived, so the bundle cannot be vouched for: {exc}. The "
                         "incomplete archive was removed."}
    moved = sorted(
        f"{store} in {db_path}"
        for (db_path, store), counter in after.items()
        if before.get((db_path, store)) != counter
    )
    if moved:
        out.unlink(missing_ok=True)
        return {"error": "this project changed while it was being archived, so the bundle would "
                         f"hold a mix of before and after: {'; '.join(moved)}. The incomplete "
                         "archive was removed; stop the writers and archive again."}

    checkpoints_excluded = 0 if include_models else sum(
        1 for p in accounting.blobs if p.parent == models_dir and p.suffix == ".pt"
    )
    return {
        "output_path": str(out),
        "files_added": files_added,
        "size_bytes": out.stat().st_size,
        "include_models": include_models,
        "left_behind": {
            "unaccounted": len(accounting.unaccounted),
            "bookkeeping": len(accounting.bookkeeping),
            "checkpoints_excluded": checkpoints_excluded,
        },
    }


_IMPORTS_DIRNAME = ".imports"


def _extract_zip(zp: Path, staging: Path) -> int:
    """Extract every member of ``zp`` into ``staging``, refusing a path that would escape it.

    ``staging`` is this run's own private directory (a fresh uuid under ``.imports``), so a
    zip-slip refusal here still leaves the destination untouched: nothing has been written there.
    Escape is decided by containment (``Path.relative_to``), never a string prefix: a member
    resolving to a sibling directory that merely shares ``staging``'s name as a prefix (e.g. an
    entry naming ``../<staging.name>extra/evil.txt``) is outside ``staging`` and refuses, where a
    prefix comparison would have read it as contained.
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
    """Remove adoption's own transition-lock files left inside the staged tree.

    Only the ``.lock`` sidecars: they linger only on POSIX (``filelock`` deletes its file on
    release under Windows), and are bookkeeping the membership accounting would never bundle, but
    the database file itself (``store.db``, just built) must ride the rename into the
    destination, not be swept away as bookkeeping alongside its own lock.
    """
    for path in staged.rglob("*.lock"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _adopt_accounted_roots(accounting) -> dict[str, int]:
    """Adopt every derived root whose plan has at least one entry; skip an empty one.

    ``adopt_root`` has no empty-plan guard of its own and would install an empty database that
    permanently refuses file-backend writes at that root, so the guard lives here. Returns the
    adopted-entry count per root path, for the response.
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

    Reuses the repo's own denial-retry helper (``tcip_store.file_backend.retry_while_denied``):
    on Windows a transient handle from a scanner or indexer denies a bare ``os.rename``. Both the
    empty-destination removal and the rename itself run under its budget, so a retry that lands
    after another process's own removal does not try to remove an already-absent directory.
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
    """Raised inside the retried move body; caught once, outside the retry, as an ordinary
    tool refusal rather than a second bespoke exception type callers must know about."""


@mcp.tool()
@audited
def import_project(zip_path: str, destination: str) -> dict:
    """Import an annotation project from a ZIP archive.

    Not a writer of any format: the door extracts into a private staging directory, classifies
    every member through the shared bundle accounting
    (:func:`tcip_mcp.tools.bundle.account_for`), refuses the whole import naming each bookkeeping,
    cross-root-collided, undecodable or unaccounted member, then adopts what is left into a
    database when this process is bound to the database backend (skipping any derived root whose
    plan is empty) or leaves the loose layout as is under the file backend, and only then renames
    the staged tree onto ``destination``. A root imported under the default backend is therefore
    usable at once, with no operator ``scripts/adopt_store.py`` run; a root imported under the
    file backend meets that script's own conform rail the same as any other unconformed layout.

    ``destination`` must not already exist, or must be an empty directory: this door merges
    nothing into a live project, since even adoption's supplement path would merge
    archive-authored stores into state that never came from this archive. When ``destination`` is
    directly under the workspace, its basename must fit ``crop_subject_phenotype``
    (``workspace.format_project_name``/``parse_project_name``); a destination outside the
    workspace is not a workspace project and is not held to the scheme.

    A refusal at any step leaves the destination exactly as it was (absent, or its original empty
    state); the staging tree this run made is removed whether the run refused, raised, or
    succeeded (a success has already moved it onto ``destination``, so removal there is a no-op).

    The response carries per-root adopted counts, blob counts per class, ``database_built``
    (whether adoption ran or the file layout was kept), ``dataset_paths_unresolved`` (the
    registered datasets whose absolute path stayed verbatim because they are outside the imported
    tree), and ``files_extracted``.

    Args:
        zip_path: Path to the ``.tcip.zip`` archive.
        destination: Directory to extract into.
    """
    from filelock import Timeout as _LockTimeout

    from tcip_mcp import workspace
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S, lock_file_for, path_lock

    zp = Path(zip_path)
    if not zp.is_file():
        return {"error": f"ZIP file not found: {zip_path}"}

    dest = Path(destination).expanduser().resolve()
    if dest.parent == workspace.workspace_root():
        try:
            workspace.parse_project_name(dest.name)
        except ValueError as exc:
            return {"error": str(exc)}

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
                result = _run_import_into_staging(zp, staging, dest)
            finally:
                # A success has already moved staging onto dest; rmtree is then a no-op.
                if "error" in result or not result:
                    shutil.rmtree(staging, ignore_errors=True)
    except _LockTimeout:
        return {"error": f"could not lock a fresh staging directory at {staging}"}
    finally:
        lock_file_for(staging).unlink(missing_ok=True)
    return result


def _run_import_into_staging(zp: Path, staging: Path, dest: Path) -> dict:
    """Everything the import door does while it holds the staging lock: extract, account for,
    decode-preflight, adopt (backend-conditional), then move. Returns the tool's own response
    dict, an ``{"error": ...}`` on any refusal.
    """
    from tcip_store.adoption import preflight_decode, unaccounted_files
    from tcip_store.binding import is_database_backend
    from tcip_store.errors import DecodeError as StoreDecodeError
    from tcip_store.errors import StoreError
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    from tcip_mcp.tools.bundle import AnchorMisplaced, account_for, blob_home

    try:
        files_extracted = _extract_zip(zp, staging)
    except (ValueError, zipfile.BadZipFile) as exc:
        return {"error": f"{zp} is not a readable ZIP archive: {exc}"}

    try:
        accounting = account_for(staging)
    except (AnchorMisplaced, StoreError) as exc:
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
        home = blob_home(tree, blob)
        blob_classes[home] = blob_classes.get(home, 0) + 1

    dataset_paths_unresolved = _external_dataset_paths(dest)

    return {
        "destination": str(dest),
        "files_extracted": files_extracted,
        "database_built": database_built,
        "adopted": adopted,
        "blob_counts": blob_classes,
        "dataset_paths_unresolved": dataset_paths_unresolved,
    }


def _external_dataset_paths(project_root: Path) -> list[str]:
    """The imported project's own registered dataset entries that stay absolute (external),
    disclosed rather than silently kept: the door never rewrites any registry entry (2.3)."""
    entries = read_datasets(project_root)
    return sorted(str(e["path"]) for e in entries if entry_is_external(e))

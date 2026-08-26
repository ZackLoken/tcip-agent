"""Project management tools."""

from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path

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

    Args:
        dataset_root: Root of the dataset (holds ``images/``, ``annotations/``, ``classes.json``).
        crop: The crop this dataset's imagery is of (e.g. ``hazelnut``). Required; the expert's fact.
        project_root: Project to register the dataset under. Empty defaults to ``dataset_root``.
    """
    from tcip_mcp.dataset_layout import dataset_identity_key, dataset_identity_path
    from tcip_mcp.pipelines.resolution import dataset_fingerprint

    root = Path(dataset_root)
    if not root.is_dir():
        return {"error": f"dataset_root not found: {dataset_root}"}
    if not crop:
        return {"error": "crop is required (the expert's fact; never inferred from a path or slug)"}

    ident_key = dataset_identity_key(root)
    fingerprint = dataset_fingerprint(root)
    # A conflict means another registration committed, so the loop only repeats while the
    # identity is actually changing under it and ends when this write is the one that lands.
    while True:
        stored = tcip_store.read_blob_versioned(ident_key, default=None)
        if stored.value is None:
            existing: dict = {}
        else:
            try:
                decoded = RECORD_JSON.decode(stored.value)
            except ValueError as exc:
                return {"error": f"{dataset_identity_path(root)} exists but does not decode ({exc}); "
                                 "minting a fresh id over it would sever every record that cites "
                                 "the old one"}
            if not isinstance(decoded, dict):
                return {"error": f"{dataset_identity_path(root)} is not an identity document; "
                                 "minting a fresh id over it would sever every record that cites "
                                 "the old one"}
            existing = decoded
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
    upsert_dataset(proj, {"id": identity["id"], "path": str(root), "crop": crop,
                          "fingerprint": fingerprint})
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
    root on it (``backend_repinned``) or could not (``backend_root_problem``): when the
    backend is down or the delivery fails, both are ``None``, since it will bind from the
    marker at its own next start regardless.

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
    return {
        "name": name,
        "project_path": str(proj),
        "marker": str(marker),
        "gui_notified": bool(delivery.get("delivered")),
        "backend_repinned": response.get("platform_root"),
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
    as a dict, when :func:`tcip_mcp.project_paths.pin_project_root` has run: absent under
    pytest and any standalone use, since neither calls it.

    For a project with ``.tcip``, carries ``site`` and ``site_problem`` from
    ``tcip_mcp.project_record.site_fields``: exactly one is set, and ``site_problem`` names why
    there is no site (no record yet, a damaged one, or a root the store refuses to read). A path
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
    if activity.get("_corrupt"):
        return {"status_unavailable": "project_status.json exists but could not be read"}
    return activity


_TCIP_ARCHIVED_SUFFIXES = frozenset(
    {".toml", ".jsonl", ".txt", ".yaml", ".yml", ".json", ".py", ".bandgroup", ".md"}
)
"""What a bundle carries out of ``.tcip``.

``.py`` because a bespoke run's snapshotted model, training and dataset source live under
``model_src/``, and a manifest describing code the bundle does not carry is not provenance.
``.bandgroup`` and ``.md`` because a band-group manifest is an enumerated logical image and a
retrospective is a document a human wrote. A store database is deliberately absent: an archive
is a file bundle, and the files are what this list names.
"""


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


@mcp.tool()
@audited
def archive_project(project_path: str, output_path: str = "", include_models: bool = False) -> dict:
    """Export an annotation project as a portable ZIP archive.

    Scans the canonical dataset layout (see :mod:`tcip_mcp.dataset_layout`): images under
    ``<root>/images/<date>/``, ground truth under ``<root>/annotations/<date>/<stem>.json``
    (one file per image, all subjects), and the single nested registry ``<root>/classes.json``,
    plus the ``.tcip`` config. Optionally includes trained checkpoints.

    Every database under the tree is exported to its loose files first, through the same
    :func:`tcip_store.export.export_root` ``scripts/export_store.py`` uses, so a project either
    creating door stood up (which writes the project record through a database) archives without
    an operator having run that script by hand. The archive refuses only when that export fails
    or a store becomes unreadable, never merely because it was behind.

    Args:
        project_path: Root directory of the project.
        output_path: Destination path for the ZIP file. Defaults to ``<project_name>.tcip.zip``
            beside the project in the workspace; a relative path resolves against the project
            root, never the server process's cwd.
        include_models: Whether to include model checkpoints (can be large).
    """
    root = Path(project_path)
    if not root.is_dir():
        return {"error": f"Project directory not found: {project_path}"}

    from tcip_store import StoreError

    try:
        _export_stores(root)
        before = _database_counters(root)
    except StoreError as exc:
        return {"error": f"a store database under {root} could not be exported before "
                         f"archiving, so the bundle cannot be vouched for: {exc}"}

    if not output_path:
        output_path = str(root.parent / f"{root.name}.tcip.zip")
    else:
        from tcip_mcp.project_paths import resolve_output_path

        output_path = str(resolve_output_path(output_path))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # The one set deciding what enumerates as a logical image, so a bundle cannot carry a
    # narrower notion of "image" than the platform that reads it back.
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    image_exts = IMAGE_EXTS
    label_exts = {".txt", ".xml", ".json"}
    files_added = 0

    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
        # Canonical dataset trees.
        from tcip_mcp.dataset_layout import annotation_root, image_root

        for tree, exts in ((image_root(root), image_exts),
                           (annotation_root(root), label_exts)):
            if tree.is_dir():
                for sub in tree.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in exts:
                        zf.write(sub, sub.relative_to(root))
                        files_added += 1

        # The single nested registry decodes the labels' subject/attribute names: without it the
        # archived annotations are undecodable, so a self-contained bundle must carry it.
        from tcip_mcp.dataset_layout import classes_path, dataset_identity_path

        registry = classes_path(root)
        if registry.is_file():
            zf.write(registry, registry.relative_to(root))
            files_added += 1

        # dataset.json: the dataset's identity ({crop, id, fingerprint}); identity is part of the
        # data, so it travels with the registry it sits beside.
        identity = dataset_identity_path(root)
        if identity.is_file():
            zf.write(identity, identity.relative_to(root))
            files_added += 1

        # .tcip config, experiments, audit: the project's working state. ``.py`` is included
        # because snapshot_model_source (pipelines/model_build.py) writes a bespoke run's actual
        # model/training/dataset source under model_src/ as .py files; without it here, the
        # archive carries the run's provenance manifest but not the code it describes.
        tcip_dir = root / ".tcip"
        if tcip_dir.is_dir():
            for f in tcip_dir.rglob("*"):
                if f.suffix in _TCIP_ARCHIVED_SUFFIXES and f.is_file():
                    zf.write(f, f.relative_to(root))
                    files_added += 1

        # Models (optional, can be large)
        if include_models:
            models_dir = tcip_dir / "models" if tcip_dir.is_dir() else root / "models"
            if models_dir.is_dir():
                for m in models_dir.glob("*.pt"):
                    zf.write(m, m.relative_to(root))
                    files_added += 1

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

    return {
        "output_path": str(out),
        "files_added": files_added,
        "size_bytes": out.stat().st_size,
        "include_models": include_models,
    }


@mcp.tool()
@audited
def import_project(zip_path: str, destination: str) -> dict:
    """Import an annotation project from a ZIP archive.

    Extracts into the destination directory, preserving the original structure. When
    ``destination`` is directly under the workspace, its basename must fit
    ``crop_subject_phenotype`` (``workspace.format_project_name``/``parse_project_name``);
    a destination outside the workspace is not a workspace project and is not held to the
    scheme.

    Args:
        zip_path: Path to the ``.tcip.zip`` archive.
        destination: Directory to extract into.
    """
    zp = Path(zip_path)
    if not zp.is_file():
        return {"error": f"ZIP file not found: {zip_path}"}

    dest = Path(destination).expanduser().resolve()
    from tcip_mcp import workspace

    if dest.parent == workspace.workspace_root():
        try:
            workspace.parse_project_name(dest.name)
        except ValueError as exc:
            return {"error": str(exc)}
    dest.mkdir(parents=True, exist_ok=True)

    files_extracted = 0
    with zipfile.ZipFile(str(zp), "r") as zf:
        # Validate paths: prevent zip slip
        for info in zf.infolist():
            target = dest / info.filename
            resolved = target.resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                return {"error": f"Unsafe path in archive: {info.filename}"}

        for info in zf.infolist():
            if info.is_dir():
                (dest / info.filename).mkdir(parents=True, exist_ok=True)
            else:
                target = dest / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                files_extracted += 1

    return {
        "destination": str(dest),
        "files_extracted": files_extracted,
    }

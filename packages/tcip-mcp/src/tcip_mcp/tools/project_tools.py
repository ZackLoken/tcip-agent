"""Project management tools."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited
from tcip_mcp.utils.atomic_io import atomic_write_text


def _project_dir(project_path: str) -> Path:
    """Return the .tcip directory for a project, creating it if needed."""
    p = Path(project_path) / ".tcip"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scaffold_project(project_path: str) -> dict:
    """Create ``.tcip/`` (artifacts/models) + a default config.toml.

    The internals of :func:`init_project`, factored out so other tools that
    stand up a project (e.g. ``ingest_images``) reuse the exact same scaffolding
    instead of re-implementing it. Idempotent — re-running only re-mkdirs.
    """
    tcip = _project_dir(project_path)
    (tcip / "artifacts").mkdir(exist_ok=True)
    (tcip / "models").mkdir(exist_ok=True)

    config_path = tcip / "config.toml"
    if not config_path.exists():
        atomic_write_text(
            config_path,
            "# TCIP project configuration\n"
            "[project]\n"
            'name = ""\n'
            'crop = ""\n'
            "\n"
            "[data]\n"
            'root = "data"\n'
            "\n"
            "[training]\n"
            "device = \"cuda\"\n"
            "seed = 42\n"
        )

    return {
        "project_path": project_path,
        "tcip_dir": str(tcip),
        "created": [".tcip/", ".tcip/artifacts/", ".tcip/models/"],
    }


@mcp.tool()
@audited
def init_project(project_path: str) -> dict:
    """Initialise a TCIP project directory.

    Creates ``.tcip/`` with default config, artifacts dir, and models dir.

    Args:
        project_path: Root directory of the project.
    """
    return _scaffold_project(project_path)


@mcp.tool()
@audited
def set_active_project(name: str) -> dict:
    """Set the workspace's active project so the GUI opens it.

    Writes the workspace active-project marker (``<workspace>/.active``) and notifies a
    running GUI to open the project — the loop-closer for the breeder flow ("I structured
    your images into ``hazelnut_catkin_valley-farm`` — opening it now"). ``name`` is a
    workspace project slug (``{crop}_{trait}_{site}``).

    Args:
        name: The workspace project to make active.
    """
    from tcip_mcp import workspace
    from tcip_mcp.web_client import post_panel_event

    try:
        marker = workspace.set_active_project(name)
        proj = workspace.project_path(name)
    except ValueError as exc:
        return {"error": str(exc)}

    delivery = post_panel_event(
        "app", "active_project_changed", {"name": name, "project_path": str(proj)}
    )
    return {
        "name": name,
        "project_path": str(proj),
        "marker": str(marker),
        "gui_notified": bool(delivery.get("delivered")),
    }


def _resolve_project_path(project_path: str) -> str:
    from tcip_mcp import workspace
    return workspace.resolve_project_path(project_path)


@mcp.tool()
@audited
def view_gui_state() -> dict:
    """The live GUI session the human is looking at: active project, dataset, date, trait, tab, and the
    exact current image. Lets the agent work through the app instead of globbing or asking which image
    is open. Reads the active-project marker + that project's gui.json. active_project is None when
    nothing is open.
    """
    from tcip_mcp import workspace

    name = workspace.read_active_project()
    if not name:
        return {"active_project": None,
                "note": "no active project; open one in the GUI or call set_active_project"}
    project_root = workspace.project_path(name)
    ctx: dict = {"active_project": name, "project_root": str(project_root)}
    gui_path = project_root / ".tcip" / "state" / "gui.json"
    if not gui_path.is_file():
        ctx["note"] = "no gui.json yet (the GUI has not persisted a selection for this project)"
        return ctx
    try:
        gui = json.loads(gui_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        ctx["error"] = f"could not read gui.json: {e}"
        return ctx
    ds = gui.get("dataset") or {}
    image_list = ds.get("image_list") or []
    idx = ds.get("current_image_index") or 0
    dataset_root, date = ds.get("dataset_root"), ds.get("date")
    current_image = None
    if dataset_root and date and 0 <= idx < len(image_list):
        current_image = str(Path(dataset_root) / "images" / date / image_list[idx])
    ctx.update({
        "dataset_root": dataset_root,
        "annotation_type": ds.get("annotation_type"),
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

    Args:
        project_path: Root directory of the project. Empty defaults to the active project.
    """
    project_path = _resolve_project_path(project_path)
    root = Path(project_path)
    tcip = root / ".tcip"

    status: dict = {"project_path": project_path, "initialized": tcip.is_dir()}
    if not tcip.is_dir():
        return status

    # Config
    config_path = tcip / "config.toml"
    status["has_config"] = config_path.is_file()

    # Models
    models_dir = tcip / "models"
    if models_dir.is_dir():
        status["model_count"] = len(list(models_dir.glob("*.pt")))

    # Artifacts
    artifacts_dir = tcip / "artifacts"
    if artifacts_dir.is_dir():
        status["artifact_count"] = len(list(artifacts_dir.iterdir()))

    # Data — the canonical layout puts images under <root>/images/<date>/ (see
    # tcip_mcp.dataset_layout); ingest_images writes there. Count that tree
    # recursively so date buckets aren't missed, and report the capture dates.
    image_exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    images_dir = root / "images"
    if images_dir.is_dir():
        from tcip_mcp import dataset_layout

        status["image_count"] = sum(
            1 for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts
        )
        status["dates"] = dataset_layout.list_dates(root)

    return status


@mcp.tool()
@audited
def archive_project(project_path: str, output_path: str = "", include_models: bool = False) -> dict:
    """Export an annotation project as a portable ZIP archive.

    Scans the canonical dataset layout (see :mod:`tcip_mcp.dataset_layout`): images under
    ``<root>/images/<date>/`` and ground truth under
    ``<root>/annotations/<trait>/<date>/<task>/``, plus the ``.tcip`` config
    and the class map (``.tcip/state/classes.json``). Optionally includes trained checkpoints.

    Args:
        project_path: Root directory of the project.
        output_path: Destination path for the ZIP file. Defaults to ``<project_name>.tcip.zip``.
        include_models: Whether to include model checkpoints (can be large).
    """
    root = Path(project_path)
    if not root.is_dir():
        return {"error": f"Project directory not found: {project_path}"}

    if not output_path:
        output_path = str(root.parent / f"{root.name}.tcip.zip")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    label_exts = {".txt", ".xml", ".json"}
    files_added = 0

    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
        # Canonical trees (images/<date>/, annotations/<trait>/<date>/<task>/) — the old root/data
        # scan silently dropped everything for an ingested project.
        for tree, exts in ((root / "images", image_exts), (root / "annotations", label_exts)):
            if tree.is_dir():
                for sub in tree.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in exts:
                        zf.write(sub, sub.relative_to(root))
                        files_added += 1

        # .tcip config + the class map at .tcip/state/classes.json
        tcip_dir = root / ".tcip"
        if tcip_dir.is_dir():
            for f in tcip_dir.rglob("*"):
                if f.is_file() and f.suffix in (".toml", ".jsonl", ".txt", ".yaml", ".yml", ".json"):
                    zf.write(f, f.relative_to(root))
                    files_added += 1

        # Models (optional, can be large)
        if include_models:
            models_dir = tcip_dir / "models" if tcip_dir.is_dir() else root / "models"
            if models_dir.is_dir():
                for m in models_dir.glob("*.pt"):
                    zf.write(m, m.relative_to(root))
                    files_added += 1

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

    Extracts into the destination directory, preserving the original structure.

    Args:
        zip_path: Path to the ``.tcip.zip`` archive.
        destination: Directory to extract into.
    """
    zp = Path(zip_path)
    if not zp.is_file():
        return {"error": f"ZIP file not found: {zip_path}"}

    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    files_extracted = 0
    with zipfile.ZipFile(str(zp), "r") as zf:
        # Validate paths — prevent zip slip
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

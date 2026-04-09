"""Project & session management tools."""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

from tcip_mcp.server import mcp


def _project_dir(project_path: str) -> Path:
    """Return the .tcip directory for a project, creating it if needed."""
    p = Path(project_path) / ".tcip"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sessions_dir(project_path: str) -> Path:
    d = _project_dir(project_path) / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


@mcp.tool()
def init_project(project_path: str) -> dict:
    """Initialise a TCIP project directory.

    Creates ``.tcip/`` with default config, sessions dir, and artifacts dir.

    Args:
        project_path: Root directory of the project.
    """
    tcip = _project_dir(project_path)
    (tcip / "sessions").mkdir(exist_ok=True)
    (tcip / "artifacts").mkdir(exist_ok=True)
    (tcip / "models").mkdir(exist_ok=True)

    config_path = tcip / "config.toml"
    if not config_path.exists():
        config_path.write_text(
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
        "created": [".tcip/", ".tcip/sessions/", ".tcip/artifacts/", ".tcip/models/"],
    }


@mcp.tool()
def create_session(project_path: str, description: str = "") -> dict:
    """Start a new agent session for this project.

    Args:
        project_path: Root directory of the project.
        description: Human-readable session description.
    """
    sessions = _sessions_dir(project_path)
    session_id = f"session_{int(time.time())}"
    session_file = sessions / f"{session_id}.jsonl"

    entry = {
        "type": "session_start",
        "id": session_id,
        "timestamp": time.time(),
        "description": description,
    }
    with open(session_file, "w") as f:
        f.write(json.dumps(entry) + "\n")

    return {"session_id": session_id, "path": str(session_file)}


@mcp.tool()
def append_session_event(project_path: str, session_id: str, event_type: str, data: dict) -> dict:
    """Append an event to a session log.

    Args:
        project_path: Root directory of the project.
        session_id: Session identifier.
        event_type: Type of event (e.g. 'tool_call', 'hitl_checkpoint', 'result').
        data: Event payload.
    """
    session_file = _sessions_dir(project_path) / f"{session_id}.jsonl"
    if not session_file.is_file():
        return {"error": f"Session not found: {session_id}"}

    entry = {"type": event_type, "timestamp": time.time(), **data}
    with open(session_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"session_id": session_id, "event_type": event_type}


@mcp.tool()
def list_sessions(project_path: str) -> dict:
    """List all sessions for a project.

    Args:
        project_path: Root directory of the project.
    """
    sessions = _sessions_dir(project_path)
    results: list[dict] = []
    for f in sorted(sessions.glob("*.jsonl")):
        with open(f) as fh:
            first_line = fh.readline()
            try:
                header = json.loads(first_line)
            except json.JSONDecodeError:
                header = {}
        # Count lines
        with open(f) as fh:
            event_count = sum(1 for _ in fh)
        results.append({
            "session_id": f.stem,
            "description": header.get("description", ""),
            "started": header.get("timestamp"),
            "event_count": event_count,
        })
    return {"sessions": results, "count": len(results)}


@mcp.tool()
def get_session(project_path: str, session_id: str) -> dict:
    """Get all events from a session.

    Args:
        project_path: Root directory of the project.
        session_id: Session identifier.
    """
    session_file = _sessions_dir(project_path) / f"{session_id}.jsonl"
    if not session_file.is_file():
        return {"error": f"Session not found: {session_id}"}

    events = []
    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return {"session_id": session_id, "events": events, "count": len(events)}


@mcp.tool()
def get_project_status(project_path: str) -> dict:
    """Get an overview of a TCIP project.

    Args:
        project_path: Root directory of the project.
    """
    root = Path(project_path)
    tcip = root / ".tcip"

    status: dict = {"project_path": project_path, "initialized": tcip.is_dir()}
    if not tcip.is_dir():
        return status

    # Config
    config_path = tcip / "config.toml"
    status["has_config"] = config_path.is_file()

    # Sessions
    sessions_dir = tcip / "sessions"
    if sessions_dir.is_dir():
        status["session_count"] = len(list(sessions_dir.glob("*.jsonl")))

    # Models
    models_dir = tcip / "models"
    if models_dir.is_dir():
        status["model_count"] = len(list(models_dir.glob("*.pt")))

    # Artifacts
    artifacts_dir = tcip / "artifacts"
    if artifacts_dir.is_dir():
        status["artifact_count"] = len(list(artifacts_dir.iterdir()))

    # Data
    data_dir = root / "data"
    if data_dir.is_dir():
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images_dir = data_dir / "images"
        if images_dir.is_dir():
            status["image_count"] = len([f for f in images_dir.iterdir() if f.suffix.lower() in image_exts])
        else:
            status["image_count"] = len([f for f in data_dir.iterdir() if f.suffix.lower() in image_exts])

    return status


@mcp.tool()
def export_project(project_path: str, output_path: str = "", include_models: bool = False) -> dict:
    """Export an annotation project as a portable ZIP archive.

    Includes images, labels, classes.txt, data.yaml, .tcip config, and session logs.
    Optionally includes trained model checkpoints.

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

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files_added = 0

    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in (root / "data").rglob("*"):
            if sub.is_file() and (
                sub.suffix.lower() in image_exts
                or sub.suffix == ".txt"
                or sub.suffix == ".yaml"
                or sub.suffix == ".yml"
            ):
                zf.write(sub, sub.relative_to(root))
                files_added += 1

        # .tcip config + sessions
        tcip_dir = root / ".tcip"
        if tcip_dir.is_dir():
            for f in tcip_dir.rglob("*"):
                if f.is_file() and f.suffix in (".toml", ".jsonl", ".txt", ".yaml", ".yml", ".json"):
                    zf.write(f, f.relative_to(root))
                    files_added += 1

        # Top-level config files
        for name in ("classes.txt", "data.yaml", "data.yml"):
            cfg = root / name
            if cfg.is_file():
                zf.write(cfg, cfg.relative_to(root))
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

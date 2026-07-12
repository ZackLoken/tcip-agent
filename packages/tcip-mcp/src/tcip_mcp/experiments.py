"""Experiment tracking for ML training runs.

Stores experiment state in .tcip/experiments/<experiment_id>/:
  config.json   — full training config snapshot
  metrics.jsonl — epoch-by-epoch metrics (append-only)
  artifacts.json — pointers to model weights, predictions
  lineage.json  — data → model → predictions chain
  status.json   — current state and timestamps
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tcip_mcp.project_paths import project_root, resolve_state
from tcip_mcp.utils.atomic_io import append_jsonl, atomic_write_json, file_transaction, read_json

logger = logging.getLogger(__name__)

# Relative default (tests rebind this constant). Consumers must go through
# ``experiments_dir()`` so the store anchors to ``$TCIP_PROJECT_ROOT`` when pinned (no
# subdir fragmentation) while a rebound absolute path / unpinned cwd still work.
EXPERIMENTS_DIR = Path(".tcip/experiments")


def experiments_dir() -> Path:
    """The experiment store, resolved against the pinned platform root at use time."""
    return resolve_state(EXPERIMENTS_DIR)


def _exp_dir(experiment_id: str) -> Path:
    return experiments_dir() / experiment_id


def create_experiment(
    experiment_id: str,
    config: dict[str, Any],
    *,
    parent_experiment: str | None = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    """Create a new experiment directory with config snapshot."""
    d = _exp_dir(experiment_id)
    if d.exists():
        return {"error": f"Experiment already exists: {experiment_id}"}

    d.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    atomic_write_json(d / "config.json", config)

    # Initial status
    status = {
        "state": "created",
        "created": datetime.now(timezone.utc).isoformat(),
        "started": None,
        "ended": None,
    }
    atomic_write_json(d / "status.json", status)

    # Lineage
    lineage = {
        "data_source": data_source,
        "parent_experiment": parent_experiment,
        "model_weights": None,
        "predictions": None,
    }
    atomic_write_json(d / "lineage.json", lineage)

    # Empty artifacts
    atomic_write_json(d / "artifacts.json", {})

    return {
        "experiment_id": experiment_id,
        "path": str(d),
        "state": "created",
    }


def update_status(experiment_id: str, state: str) -> dict[str, Any]:
    """Update experiment state (created → running → completed | failed)."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    status_path = d / "status.json"
    with file_transaction(status_path):
        status = json.loads(status_path.read_text())
        status["state"] = state

        now = datetime.now(timezone.utc).isoformat()
        status["heartbeat"] = now  # liveness stamp: a fresh heartbeat means a live process
        if state == "running" and not status.get("started"):
            status["started"] = now
        if state in ("completed", "failed"):
            status["ended"] = now

        atomic_write_json(status_path, status)
    return {"experiment_id": experiment_id, "state": state}


def _touch_heartbeat(exp_dir: Path) -> None:
    """Best-effort: stamp the current time into ``status.json['heartbeat']``.

    Called each epoch so a run still training in another process (e.g. the MCP agent) reads
    as live to a web client reconstructing run state, instead of being flagged interrupted.
    Never raises — a heartbeat failure must not break metric logging.
    """
    status_path = exp_dir / "status.json"
    if not status_path.is_file():
        return
    try:
        with file_transaction(status_path):
            status = json.loads(status_path.read_text())
            status["heartbeat"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(status_path, status)
    except Exception:
        pass


def log_metrics(
    experiment_id: str,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Append epoch metrics to metrics.jsonl and refresh the run's liveness heartbeat."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    entry = {
        "epoch": epoch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    append_jsonl(d / "metrics.jsonl", entry)
    _touch_heartbeat(d)

    return {"experiment_id": experiment_id, "epoch": epoch, "logged": True}


def record_artifact(
    experiment_id: str,
    name: str,
    path: str,
) -> dict[str, Any]:
    """Register an artifact (model weights, predictions, etc.)."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    artifacts_path = d / "artifacts.json"
    with file_transaction(artifacts_path):
        artifacts = json.loads(artifacts_path.read_text())
        artifacts[name] = {"path": path, "recorded": datetime.now(timezone.utc).isoformat()}
        atomic_write_json(artifacts_path, artifacts)

    return {"experiment_id": experiment_id, "artifact": name, "path": path}


def update_lineage(
    experiment_id: str,
    **updates: Any,
) -> dict[str, Any]:
    """Update lineage fields (model_weights, predictions, etc.)."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    lineage_path = d / "lineage.json"
    with file_transaction(lineage_path):
        lineage = json.loads(lineage_path.read_text())
        lineage.update(updates)
        atomic_write_json(lineage_path, lineage)

    return {"experiment_id": experiment_id, "lineage": lineage}


def register_model_from_experiment(
    experiment_id: str,
    checkpoint_path: str,
    *,
    project_path: str = "",
    name: str | None = None,
) -> dict[str, Any]:
    """Register a completed experiment's model in the project registry.

    Pulls the experiment's config and the checkpoint's own metrics — the epoch that produced
    this checkpoint (e.g. ``model_best.pt``'s best epoch), not necessarily the last training
    epoch — falling back to the experiment's final ``metrics.jsonl`` row if the checkpoint
    carries none. Registers with an ``experiment:<id>`` back-reference tag and records it in
    the experiment's lineage (``model_weights``). Metrics are read, never fabricated.
    """
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    config = read_json(d / "config.json", default={})

    # Prefer metrics stored IN the checkpoint (they describe the epoch it was saved at, so a
    # best-checkpoint isn't mislabelled with a later, worse epoch's numbers). Fall back to the
    # experiment's last metrics.jsonl row only if the checkpoint carries none.
    final_metrics: dict[str, Any] = {}
    ckpt = Path(checkpoint_path)
    if ckpt.is_file():
        try:
            import torch  # local checkpoint the caller is registering deliberately

            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
                final_metrics = dict(payload["metrics"])
                if payload.get("epoch") is not None:
                    final_metrics.setdefault("epoch", payload["epoch"])
        except Exception:
            final_metrics = {}
    if not final_metrics:
        mpath = d / "metrics.jsonl"
        if mpath.is_file():
            lines = mpath.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                try:
                    final_metrics = json.loads(lines[-1])
                except json.JSONDecodeError:
                    final_metrics = {}

    from tcip_mcp.model_registry import ModelRegistry

    # Registry co-locates with the experiment store under the platform root (the adopted
    # project after set_active_project) unless an explicit path overrides.
    registry_root = project_path or str(project_root())
    entry = ModelRegistry(registry_root).register_model(
        name or experiment_id, checkpoint_path, config,
        metrics=final_metrics, tags=[f"experiment:{experiment_id}"],
    )
    update_lineage(experiment_id, model_weights=checkpoint_path)
    return {
        "experiment_id": experiment_id,
        "registered": entry["name"],
        "checkpoint": checkpoint_path,
        "metrics": final_metrics,
    }


def get_experiment(
    experiment_id: str, *, metrics_limit: int | None = None, metrics_offset: int = 0,
) -> dict[str, Any]:
    """Read full experiment state.

    ``metrics`` can be paginated for long runs: ``metrics_offset`` skips epochs and
    ``metrics_limit`` caps how many are returned (only the requested window is JSON-parsed;
    ``n_epochs`` is always the true total). Defaults return all metrics (unchanged).
    """
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    result: dict[str, Any] = {"experiment_id": experiment_id}

    for name in ("config", "status", "artifacts", "lineage"):
        p = d / f"{name}.json"
        if p.exists():
            result[name] = json.loads(p.read_text())

    metrics_path = d / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        result["n_epochs"] = len(lines)
        end = (metrics_offset + metrics_limit) if metrics_limit is not None else None
        window = lines[metrics_offset:end]
        result["metrics"] = [json.loads(line) for line in window]
        result["metrics_offset"] = metrics_offset

    return result


def list_experiments() -> list[dict[str, Any]]:
    """List all experiments with summary info."""
    root = experiments_dir()
    if not root.exists():
        return []

    experiments = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        status_path = d / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            experiments.append({
                "experiment_id": d.name,
                "state": status.get("state", "unknown"),
                "created": status.get("created"),
            })

    return experiments


def compare_experiments(experiment_ids: list[str]) -> dict[str, Any]:
    """Side-by-side comparison of multiple experiments."""
    comparisons: list[dict[str, Any]] = []

    for eid in experiment_ids:
        exp = get_experiment(eid)
        if "error" in exp:
            comparisons.append({"experiment_id": eid, "error": exp["error"]})
            continue

        summary: dict[str, Any] = {
            "experiment_id": eid,
            "state": exp.get("status", {}).get("state"),
        }

        # Get final metrics
        metrics = exp.get("metrics", [])
        if metrics:
            summary["final_metrics"] = metrics[-1]
            summary["n_epochs"] = len(metrics)

        # Get config summary
        config = exp.get("config", {})
        model_spec = config.get("model_spec") or config.get("model", {})
        summary["backbone"] = model_spec.get("backbone", {}).get("name", "unknown")

        comparisons.append(summary)

    return {"experiments": comparisons, "count": len(comparisons)}


def get_experiment_lineage(experiment_id: str) -> dict[str, Any]:
    """Trace the full data → model → predictions chain."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    lineage_path = d / "lineage.json"
    if not lineage_path.exists():
        return {"error": "No lineage file found"}

    lineage = json.loads(lineage_path.read_text())

    # Include config data source info
    config_path = d / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        data_cfg = config.get("data", {})
        lineage["data_config"] = {
            "images_dir": data_cfg.get("images_dir"),
            "labels_dir": data_cfg.get("labels_dir"),
            "task": data_cfg.get("task"),
        }

    return {"experiment_id": experiment_id, "lineage": lineage}

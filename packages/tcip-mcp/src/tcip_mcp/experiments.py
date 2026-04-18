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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(".tcip/experiments")


def _exp_dir(experiment_id: str) -> Path:
    return EXPERIMENTS_DIR / experiment_id


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
    (d / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # Initial status
    status = {
        "state": "created",
        "created": datetime.now(timezone.utc).isoformat(),
        "started": None,
        "ended": None,
    }
    (d / "status.json").write_text(json.dumps(status, indent=2))

    # Lineage
    lineage = {
        "data_source": data_source,
        "parent_experiment": parent_experiment,
        "model_weights": None,
        "predictions": None,
    }
    (d / "lineage.json").write_text(json.dumps(lineage, indent=2))

    # Empty artifacts
    (d / "artifacts.json").write_text(json.dumps({}, indent=2))

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
    status = json.loads(status_path.read_text())
    status["state"] = state

    now = datetime.now(timezone.utc).isoformat()
    if state == "running" and not status.get("started"):
        status["started"] = now
    if state in ("completed", "failed"):
        status["ended"] = now

    status_path.write_text(json.dumps(status, indent=2))
    return {"experiment_id": experiment_id, "state": state}


def log_metrics(
    experiment_id: str,
    epoch: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Append epoch metrics to metrics.jsonl."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    entry = {
        "epoch": epoch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    with open(d / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

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
    artifacts = json.loads(artifacts_path.read_text())
    artifacts[name] = {"path": path, "recorded": datetime.now(timezone.utc).isoformat()}
    artifacts_path.write_text(json.dumps(artifacts, indent=2))

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
    lineage = json.loads(lineage_path.read_text())
    lineage.update(updates)
    lineage_path.write_text(json.dumps(lineage, indent=2))

    return {"experiment_id": experiment_id, "lineage": lineage}


def get_experiment(experiment_id: str) -> dict[str, Any]:
    """Read full experiment state."""
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    result: dict[str, Any] = {"experiment_id": experiment_id}

    for name in ("config", "status", "artifacts", "lineage"):
        p = d / f"{name}.json"
        if p.exists():
            result[name] = json.loads(p.read_text())

    # Read last N metrics
    metrics_path = d / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        result["metrics"] = [json.loads(line) for line in lines]
        result["n_epochs"] = len(lines)

    return result


def list_experiments() -> list[dict[str, Any]]:
    """List all experiments with summary info."""
    if not EXPERIMENTS_DIR.exists():
        return []

    experiments = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
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

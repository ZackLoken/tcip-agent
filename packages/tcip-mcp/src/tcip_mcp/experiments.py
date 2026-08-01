"""Experiment tracking for ML training runs.

Stores experiment state in .tcip/experiments/<experiment_id>/:
  config.json, full training config snapshot
  metrics.jsonl, epoch-by-epoch metrics (append-only)
  artifacts.json, pointers to model weights, predictions
  lineage.json, data → model → predictions chain
  status.json, current state and timestamps
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


# Once a run reaches a terminal state its record is immutable (experiments are immutable). The lock
# is additive-only: populated fields freeze, but a still-empty field may take its first write,
# so the post-completion predictions link + model registration still land. Deliberately excludes
# "cancelled", a cancelled run's record stays reopenable (e.g. resumed via resume_from), so it
# must not be lock-frozen the way a genuinely finished run is.
_TERMINAL_STATES = {"completed", "failed"}

# A different concept sharing similar vocabulary, states reconstruct_run_status trusts as
# already-decided and never re-derives from heartbeat freshness. Unlike _TERMINAL_STATES above,
# this does include "cancelled": a gracefully cancelled run recorded its own final state honestly
# (model_final.pt was written, cancel_training's own documented contract), and re-deriving it from
# heartbeat staleness would misreport it as "running" then permanently as "interrupted", implying
# a crash that never happened. Named separately rather than reusing _TERMINAL_STATES so the two
# purposes (mutation-lock vs. heartbeat-reconstruction) can never silently drift onto each other.
_RECORDED_AS_DONE = {"completed", "failed", "cancelled"}


def _current_state(exp_dir: Path) -> str | None:
    status_path = exp_dir / "status.json"
    if not status_path.is_file():
        return None
    try:
        return json.loads(status_path.read_text()).get("state")
    except (json.JSONDecodeError, OSError):
        return None


def _audit_refused(experiment_id: str, op: str, detail: dict[str, Any]) -> None:
    """Record a refused post-terminal mutation on the append-only audit log (best-effort)."""
    try:
        from tcip_mcp.audit import record_event

        record_event("experiment_mutation_refused", {"experiment_id": experiment_id, "op": op,
                                                      **detail}, status="refused")
    except Exception:
        logger.debug("could not audit refused mutation", exc_info=True)


def create_experiment(
    experiment_id: str,
    config: dict[str, Any],
    *,
    parent_experiment: str | None = None,
    data_source: str | None = None,
    dataset_id: str | None = None,
    dataset_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create a new experiment directory with config snapshot.

    ``dataset_id`` / ``dataset_fingerprint`` record the identity of the data this run trained on (the
    content end of the reproduce-a-number chain), written into the immutable lineage at creation. They
    are set once here and never via ``update_lineage`` (identity, not a mutable edge).
    """
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
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
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


def overwrite_config_if_pristine(experiment_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``config.json`` with the config actually launched, but only while the experiment is
    still pristine (state == "created" and no epochs logged yet).

    A pre-created experiment's ``config.json`` is written once, at ``create_experiment`` time,
    before effective tiling geometry and the training seed are resolved (see
    ``training_tools.launch_training``). Reusing that id via ``_ensure_experiment``'s pristine-reuse
    branch would otherwise ship a permanently stale snapshot describing a config that was never
    trained. Refuses (and audits the refusal) once the record is no longer pristine, a "created"
    record that already has metrics rows must stay protected too, so this checks the same full
    predicate ``_ensure_experiment`` uses, not just the terminal-state lock alone.
    """
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}
    metrics_path = d / "metrics.jsonl"
    has_metrics = metrics_path.is_file() and metrics_path.stat().st_size > 0
    state = _current_state(d)
    if state != "created" or has_metrics:
        _audit_refused(experiment_id, "overwrite_config_if_pristine",
                       {"state": state, "has_metrics": has_metrics})
        return {"error": f"Experiment {experiment_id} is no longer pristine; refusing to "
                         f"overwrite its config.json."}
    atomic_write_json(d / "config.json", config)
    return {"experiment_id": experiment_id, "overwritten": True}


def update_status(experiment_id: str, state: str, *, error: str | None = None) -> dict[str, Any]:
    """Update experiment state (created → running → completed | failed).

    ``error`` records a specific failure reason (e.g. a wall-clock-timeout kill) into
    ``status.json["error"]``, omitted/``None`` never clears a previously-recorded error, only an
    explicit new value overwrites it.
    """
    d = _exp_dir(experiment_id)
    if not d.exists():
        return {"error": f"Experiment not found: {experiment_id}"}

    status_path = d / "status.json"
    with file_transaction(status_path):
        status = json.loads(status_path.read_text())
        current = status.get("state")
        # Terminal-state lock: a completed/failed run cannot be re-opened to a non-terminal state.
        if current in _TERMINAL_STATES and state != current and state not in _TERMINAL_STATES:
            _audit_refused(experiment_id, "update_status", {"from": current, "to": state})
            return {"error": f"Experiment {experiment_id} is {current} (terminal); refusing to "
                             f"re-open to {state!r}.", "state": current}
        status["state"] = state
        if error is not None:
            status["error"] = error

        now = datetime.now(timezone.utc).isoformat()
        status["heartbeat"] = now  # liveness stamp: a fresh heartbeat means a live process
        if state == "running" and not status.get("started"):
            status["started"] = now
        if state in ("completed", "failed"):
            status["ended"] = now

        atomic_write_json(status_path, status)
    return {"experiment_id": experiment_id, "state": state}


def stamp_run_identity(experiment_id: str, run_id: str, output_dir: str) -> None:
    """Record which ``run_id``/``output_dir`` produced this experiment, into ``status.json``.

    Best-effort, like ``_touch_heartbeat``, a dropped stamp must not break the launch it's
    recording. Called unconditionally by ``_ensure_experiment`` regardless of which of its three
    branches resolved ``experiment_id`` (fresh creation, pristine pre-created-experiment reuse, or a
    fresh-id conflict), those are the only paths that mint a real, running experiment, and this
    is what makes the real artifact directory (``output_dir``, a separately-computed, caller-influenced
    path that only coincides with the experiment directory by convention) discoverable from
    ``experiment_id``/``run_id`` alone by a different process.
    """
    d = _exp_dir(experiment_id)
    status_path = d / "status.json"
    if not status_path.is_file():
        return
    try:
        with file_transaction(status_path):
            status = json.loads(status_path.read_text())
            status["run_id"] = run_id
            status["output_dir"] = output_dir
            atomic_write_json(status_path, status)
    except Exception:
        logger.warning("stamp_run_identity failed for %s/%s", experiment_id, run_id, exc_info=True)


def resolve_experiment_dir_for_run(run_id: str) -> Path | None:
    """Find the experiment directory for ``run_id`` without assuming ``experiment_id == run_id``.

    Tries the exact match first (the common case, ``experiment_id == run_id``). Then the
    fresh-id relaunch format (``f"{experiment_id}_{run_id}"``, always suffixed ``_<run_id>``) via a
    glob. Neither naming convention covers a *custom-named* experiment (an agent/breeder
    pre-created it via the standalone ``create_experiment`` tool, e.g. ``"exp-001-<crop>-<trait>-
    det"``, before any ``run_id`` existed, then launched training against it later, a real, tested
    workflow, not theoretical: ``_ensure_experiment``'s pristine-reuse branch), its directory name
    bears no naming relationship to ``run_id`` at all. For that case, falls back to scanning every
    experiment directory's own stamped ``status.json["run_id"]`` (the authoritative fact
    ``stamp_run_identity`` records, not a naming guess), a full scan, but reached only once both
    naming shortcuts miss, and it's also what disambiguates the (negligible-probability, per
    ``run_id``'s own timestamp+uuid entropy) case of more than one glob match, rather than refusing
    a resolvable run just because the fast path was ambiguous. Returns ``None`` only when no
    directory's stamped identity matches at all, the caller (``cancel_run``'s disk fallback,
    ``reconstruct_run_status``) must then refuse honestly rather than act against an unverified path.
    """
    root = experiments_dir()
    exact = root / run_id
    if exact.is_dir():
        return exact
    if not root.is_dir():
        return None
    matches = [p for p in root.glob(f"*_{run_id}") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        status_path = d / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if status.get("run_id") == run_id:
            return d
    return None


def reconstruct_run_status(run_id: str, *, stale_seconds: float = 600.0) -> dict[str, Any] | None:
    """Reconstruct a run's status from disk for a caller whose in-memory registry doesn't
    have it, either it was never in this process (a different process launched it) or it was
    subprocess-delegated and the in-memory record is stale by design.

    Returns ``None`` when the run can't be resolved on disk at all (an honestly unknown run, not a
    guess). ``current_epoch`` comes from the last ``metrics.jsonl`` row when present; ``best_metric``
    is left ``None``, a running best isn't recoverable from the metrics log alone without
    re-deriving the selection policy, and a fabricated approximation would be worse than an honest
    gap (matches the pre-existing convention this function replaces, which also reported ``None``).
    ``stale_seconds`` lets a caller (``routes/training.py``) keep its own configurable heartbeat
    window rather than being pinned to this module's default.
    """
    d = resolve_experiment_dir_for_run(run_id)
    if d is None:
        return None
    status_path = d / "status.json"
    if not status_path.is_file():
        return None
    try:
        status = json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    state = status.get("state", "unknown")
    heartbeat = status.get("heartbeat")
    if state not in _RECORDED_AS_DONE:
        state = "running" if _heartbeat_fresh(heartbeat, stale_seconds) else "interrupted"

    current_epoch = None
    metrics_path = d / "metrics.jsonl"
    if metrics_path.is_file():
        try:
            lines = [ln for ln in metrics_path.read_text().splitlines() if ln.strip()]
            if lines:
                current_epoch = json.loads(lines[-1]).get("epoch")
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "run_id": status.get("run_id", d.name),
        "experiment_id": d.name,
        "status": state,
        "current_epoch": current_epoch,
        "best_metric": None,
        "output_dir": status.get("output_dir"),
        "error": status.get("error"),
    }


def _heartbeat_fresh(hb_iso: str | None, stale_seconds: float = 600.0) -> bool:
    """True if ``hb_iso`` (ISO-8601) is within the staleness window, a process is still actively
    updating this run. Missing/unparseable → not fresh (treat as dead). Mirrors
    ``routes/training.py``'s own threshold; kept independent since this module has no FastAPI/env
    dependency and the two consumers (web route, MCP tool) can reasonably differ in the future."""
    if not hb_iso:
        return False
    try:
        hb = datetime.fromisoformat(hb_iso)
    except (ValueError, TypeError):
        return False
    if hb.tzinfo is None:
        hb = hb.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - hb).total_seconds() <= stale_seconds


def _touch_heartbeat(exp_dir: Path) -> None:
    """Best-effort: stamp the current time into ``status.json['heartbeat']``.

    Called each epoch so a run still training in another process (e.g. the MCP agent) reads
    as live to a web client reconstructing run state, instead of being flagged interrupted.
    Never raises, a heartbeat failure must not break metric logging.
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

    # Terminal-state lock: a completed/failed run's metric history is frozen, no new epochs.
    if _current_state(d) in _TERMINAL_STATES:
        _audit_refused(experiment_id, "log_metrics", {"epoch": epoch})
        return {"error": f"Experiment {experiment_id} is terminal; refusing to log a new epoch."}

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
        # Terminal-state lock (additive-only): a new artifact name may be recorded post-completion,
        # but an existing one is frozen, no silent overwrite of a delivered pointer.
        if name in artifacts and _current_state(d) in _TERMINAL_STATES:
            _audit_refused(experiment_id, "record_artifact", {"artifact": name})
            return {"error": f"Experiment {experiment_id} is terminal; artifact {name!r} already "
                             f"recorded and is immutable.", "artifact": name}
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

    # Dataset identity is set once at creation and is immutable, never a lineage edge to backfill.
    # (The additive-only lock below would otherwise permit a first write to an empty identity field
    # even post-terminal, which would be a silent change to what data the run trained on.)
    identity_updates = {k: updates.pop(k) for k in ("dataset_id", "dataset_fingerprint") if k in updates}
    if identity_updates:
        _audit_refused(experiment_id, "update_lineage_identity", {"fields": sorted(identity_updates)})

    lineage_path = d / "lineage.json"
    with file_transaction(lineage_path):
        lineage = json.loads(lineage_path.read_text())
        # Terminal-state lock (additive-only): once terminal, a still-empty field may take its first
        # write (the post-completion predictions link, model_weights on registration), but a
        # populated field is frozen, no overwrite of a recorded lineage edge.
        if _current_state(d) in _TERMINAL_STATES:
            refused = {k: v for k, v in updates.items()
                       if lineage.get(k) not in (None, "", [], {}) and lineage.get(k) != v}
            if refused:
                _audit_refused(experiment_id, "update_lineage", {"fields": sorted(refused)})
                updates = {k: v for k, v in updates.items() if k not in refused}
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

    Pulls the experiment's config and the checkpoint's own metrics, the epoch that produced
    this checkpoint (e.g. ``model_best.pt``'s best epoch), not necessarily the last training
    epoch, falling back to the experiment's final ``metrics.jsonl`` row if the checkpoint
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
    kind: str | None = None
    ckpt = Path(checkpoint_path)
    if ckpt.is_file():
        try:
            import torch  # local checkpoint the caller is registering deliberately

            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                kind = payload.get("kind")  # stamped by the trainer; None on older checkpoints
                if isinstance(payload.get("metrics"), dict):
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
        metrics=final_metrics, tags=[f"experiment:{experiment_id}"], kind=kind,
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
        model_source = config.get("model_source", {})
        summary["model"] = model_source.get("builder", "unknown")

        # Dataset identity (the content end of the reproduce-a-number chain), from the immutable lineage.
        lineage_path = _exp_dir(eid) / "lineage.json"
        if lineage_path.is_file():
            try:
                lin = json.loads(lineage_path.read_text())
                summary["dataset_id"] = lin.get("dataset_id")
                summary["dataset_fingerprint"] = lin.get("dataset_fingerprint")
            except (OSError, ValueError):
                pass

        comparisons.append(summary)

    # Whether every compared run trained on the same dataset content, a metric comparison across
    # different data is not apples-to-apples, so surface it rather than let the caller assume.
    # A run with no recorded fingerprint (bespoke/imageless) makes the comparison's data identity
    # unknown, not "same" by default, so an unset fingerprint must not be filtered out before the
    # equality check the way an errored comparison is.
    fps = {c.get("dataset_fingerprint") for c in comparisons if "error" not in c}
    same_dataset = None if (not fps or None in fps) else len(fps) == 1
    return {"experiments": comparisons, "count": len(comparisons), "same_dataset_fingerprint": same_dataset}


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

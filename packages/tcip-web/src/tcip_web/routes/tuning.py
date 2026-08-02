"""HPO / Tuning routes: launch + list + basic trial visibility.

The underlying ``run_hpo`` MCP tool already returns structured trial results;
we just wrap it in an HTTP surface for the Tuning tab.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tuning", tags=["tuning"])


@dataclass
class HPOJob:
    sweep_id: str
    status: str = "pending"
    error: Optional[str] = None
    result: dict[str, Any] = field(default_factory=dict)
    thread: Optional[threading.Thread] = field(default=None, repr=False)


_sweeps: dict[str, HPOJob] = {}
_lock = threading.Lock()


def _summary(job: HPOJob) -> dict:
    return {"sweep_id": job.sweep_id, "status": job.status,
            "error": job.error, "has_result": bool(job.result)}


def _persist() -> None:
    from tcip_web import jobstore
    with _lock:
        summaries = [_summary(j) for j in _sweeps.values()]
    jobstore.persist("hpo_sweeps", summaries)


def rehydrate() -> None:
    """Seed the sweep registry from the last persisted summaries after a restart.

    Worker threads are gone, so a persisted non-terminal sweep is dead: surfaced as
    ``interrupted``. Trial results aren't persisted, so a rehydrated sweep has no result.
    """
    from tcip_web import jobstore

    with _lock:
        if _sweeps:
            return
        for s in jobstore.load("hpo_sweeps"):
            sid = s.get("sweep_id")
            if not sid:
                continue
            status = s.get("status", "interrupted")
            if status not in jobstore.TERMINAL_STATUSES:
                status = "interrupted"
            _sweeps[sid] = HPOJob(sweep_id=sid, status=status, error=s.get("error"))


class LaunchHPOPayload(BaseModel):
    base_config: dict[str, Any]
    param_space: Optional[dict[str, Any]] = None
    n_trials: int = 5
    output_dir: str = ""
    search_alg: str = "random"
    scheduler: str = "asha"


def _worker(job: HPOJob, payload: LaunchHPOPayload) -> None:
    try:
        job.status = "running"
        _persist()
        from tcip_mcp.tools.training_tools import run_hpo

        res = run_hpo(
            base_config=payload.base_config,
            param_space=payload.param_space,
            n_trials=payload.n_trials,
            output_dir=payload.output_dir,
            search_alg=payload.search_alg,
            scheduler=payload.scheduler,
        )
        job.result = res if isinstance(res, dict) else {"raw": res}
        job.status = "completed"
    except Exception as exc:
        logger.exception("HPO sweep %s failed", job.sweep_id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        _persist()


@router.post("/launch")
def launch_hpo(payload: LaunchHPOPayload) -> dict:
    from tcip_web import jobstore
    from tcip_web.paths import assert_path_allowed

    # Confine the client-supplied output dir when the server is locked down (no-op otherwise).
    # Data paths nested inside base_config are a documented follow-up (not top-level here).
    if payload.output_dir:
        try:
            assert_path_allowed(payload.output_dir)
        except ValueError as exc:
            raise HTTPException(403, str(exc)) from exc

    job = HPOJob(sweep_id=f"hpo-{uuid.uuid4().hex[:8]}")
    with _lock:
        _sweeps[job.sweep_id] = job
        jobstore.evict_terminal(_sweeps)  # bound the registry (drop oldest terminal sweeps)
    _persist()
    t = threading.Thread(target=_worker, args=(job, payload), daemon=True)
    job.thread = t
    t.start()
    return {"status": "launched", "sweep_id": job.sweep_id}


@router.get("/sweeps")
def list_sweeps() -> dict:
    with _lock:
        return {
            "sweeps": [
                {
                    "sweep_id": j.sweep_id,
                    "status": j.status,
                    "error": j.error,
                    "has_result": bool(j.result),
                }
                for j in _sweeps.values()
            ]
        }


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    with _lock:
        j = _sweeps.get(sweep_id)
    if j is None:
        raise HTTPException(404, f"sweep not found: {sweep_id}")
    return {
        "sweep_id": j.sweep_id,
        "status": j.status,
        "error": j.error,
        "result": j.result,
    }

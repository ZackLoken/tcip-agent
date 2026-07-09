"""In-app agent chat — the HTTP/WS surface and per-session sidecar lifecycle.

Design: ``docs/chat-popup-design.md``. The chat adds a *transport* for the human↔agent
conversation, not a new mutation path — every state change the agent makes still goes
through the same ``@audited`` MCP tools. The transcript (operator I/O) is persisted to
``<project>/.tcip/chat/<id>.jsonl`` so a reconnecting browser (or a backend restart)
gets a readable history; the WebSocket is a live tail with replay, mirroring the
panel-event hub.

Availability-gated, single code path: when no sidecar is available (the ``claude`` CLI is
absent — e.g. in CI), ``/api/chat/status`` reports ``unavailable`` and the panel shows the
unconfigured state. There is no echo/fake agent baked into the product (that would be the
dual code path the repo forbids); the tests drive a scripted fake sidecar injected at the
process boundary via ``TCIP_CHAT_SIDECAR_CMD``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from tcip_web import agent_host
from tcip_web.paths import origin_allowed
from tcip_web.state import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

RECENT_MAXLEN = 512
# The sidecar's cwd — the dir tcip-web was launched from (the repo root), so it picks up
# .mcp.json / CLAUDE.md / .github/skills like an interactive Claude Code session. The
# active *project* is passed to the agent via the context block, not via cwd.
_DEFAULT_CWD = os.getcwd()


def _sidecar_cwd() -> str:
    return os.environ.get("TCIP_CHAT_CWD", _DEFAULT_CWD)


def _chat_dir(project_root: Optional[str]) -> Path:
    base = Path(project_root) if project_root else Path(".")
    d = base / ".tcip" / "chat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _context_block(state: Any) -> str:
    """A compact, end-user-framed snapshot of what the GUI is looking at, prepended to a
    user message so the agent's answer is grounded in the human's current view (C3)."""
    ds = state.dataset
    lines = ["[Current GUI context]", f"Active tab: {state.active_tab}"]
    if ds.project_root:
        lines.append(f"Active project: {ds.project_root}")
    if ds.dataset_root and ds.dataset_root != ds.project_root:
        lines.append(f"Dataset root: {ds.dataset_root}")
    if ds.date:
        lines.append(f"Capture date: {ds.date}")
    if ds.annotation_type:
        lines.append(f"Trait: {ds.annotation_type}")
    if ds.image_list and 0 <= ds.current_image_index < len(ds.image_list):
        lines.append(f"Current image: {ds.image_list[ds.current_image_index]}")
    return "\n".join(lines)


class ChatSession:
    """One chat conversation and (lazily) its agent sidecar process."""

    def __init__(self, session_id: str, project_root: Optional[str], loop: asyncio.AbstractEventLoop):
        self.id = session_id
        self.project_root = project_root
        self._loop = loop
        self.transcript_path = _chat_dir(project_root) / f"{session_id}.jsonl"
        self.subscribers: set[WebSocket] = set()
        self.recent: deque[dict] = deque(maxlen=RECENT_MAXLEN)
        self.state = "idle"
        self.title = ""
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stdin_lock = threading.Lock()
        self._spawn_lock = threading.Lock()

    # ── emit: persist + recent + broadcast ──────────────────────────────

    def emit(self, event: dict) -> None:
        """Record an envelope event and fan it out. Safe to call from any thread."""
        if event.get("type") == "session_state":
            self.state = event.get("state", self.state)
        try:
            from tcip_mcp.utils.atomic_io import append_jsonl

            append_jsonl(self.transcript_path, event)
        except Exception:  # pragma: no cover - persistence is best-effort
            logger.debug("chat transcript append failed", exc_info=True)
        self.recent.append(event)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            coro = self._broadcast(event)
            try:
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:  # pragma: no cover - loop gone during shutdown
                coro.close()  # don't leave an un-awaited coroutine behind

    async def _broadcast(self, event: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.subscribers):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)

    # ── sidecar lifecycle ───────────────────────────────────────────────

    def _ensure_started(self) -> Optional[str]:
        """Spawn the sidecar if it isn't already live. Returns an error reason or None.

        Guarded by a lock: post_message runs in a threadpool, so two rapid messages to one
        session must not both spawn (which would orphan the first process).
        """
        with self._spawn_lock:
            return self._ensure_started_locked()

    def _ensure_started_locked(self) -> Optional[str]:
        if self._proc and self._proc.poll() is None:
            return None
        cmd = agent_host.resolve_sidecar_command()
        if cmd is None:
            return agent_host.chat_status().get("reason")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=_sidecar_cwd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
        except OSError as exc:
            return f"could not start agent: {exc}"
        self.state = "running"
        self._reader = threading.Thread(target=self._read_loop, name=f"chat-{self.id}", daemon=True)
        self._reader.start()
        return None

    def _read_loop(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for ev in agent_host.translate_event(raw):
                    self.emit(ev)
        finally:
            self.emit({"type": "session_state", "state": "dead"})

    def _write_stdin(self, line: str) -> bool:
        proc = self._proc
        if not proc or proc.poll() is not None or not proc.stdin:
            return False
        with self._stdin_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
                return True
            except (OSError, ValueError):
                return False

    def send_user(self, text: str, include_context: bool) -> Optional[str]:
        err = self._ensure_started()
        if err:
            self.emit({"type": "session_state", "state": "dead", "reason": err})
            return err
        self.emit({"type": "user_message", "text": text})
        if not self.title:
            self.title = text[:60]
        payload = text
        if include_context:
            payload = _context_block(store.state) + "\n\n" + text
        if not self._write_stdin(agent_host.encode_user_message(payload)):
            self.emit({"type": "session_state", "state": "dead", "reason": "agent process not writable"})
            return "agent process not writable"
        self.emit({"type": "session_state", "state": "running"})
        return None

    def send_permission(self, request_id: str, approved: bool, note: str = "") -> bool:
        return self._write_stdin(agent_host.encode_permission_response(request_id, approved, note))

    def interrupt(self) -> bool:
        return self._write_stdin(agent_host.encode_interrupt())

    def terminate(self) -> None:
        proc = self._proc
        if not proc:
            return
        if proc.poll() is None:
            try:
                if os.name == "nt":
                    # Tree-kill so the CLI's own child (its tcip-mcp) doesn't orphan.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        check=False,
                    )
                else:
                    proc.terminate()
            except Exception:  # pragma: no cover
                logger.debug("chat terminate failed", exc_info=True)
        try:
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            try:
                proc.kill()
            except Exception:
                pass

    def messages(self) -> list[dict]:
        """Full transcript for replay (falls back to the in-memory tail on read error)."""
        if not self.transcript_path.is_file():
            return list(self.recent)
        out: list[dict] = []
        try:
            for line in self.transcript_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            return list(self.recent)
        return out


_SESSIONS: dict[str, ChatSession] = {}


def shutdown_all() -> None:
    """Kill every live sidecar (called from the app lifespan on shutdown)."""
    for s in list(_SESSIONS.values()):
        s.terminate()
    _SESSIONS.clear()


# ── HTTP surface ────────────────────────────────────────────────────────


@router.get("/status")
def get_status() -> dict:
    return agent_host.chat_status()


@router.post("/sessions")
async def create_session() -> dict:
    loop = asyncio.get_running_loop()
    session_id = "chat_" + os.urandom(6).hex()
    project_root = store.state.dataset.project_root
    _SESSIONS[session_id] = ChatSession(session_id, project_root, loop)
    return {"session_id": session_id}


@router.get("/sessions")
def list_chat_sessions() -> dict:
    return {
        "sessions": [
            {"id": s.id, "title": s.title, "state": s.state} for s in _SESSIONS.values()
        ]
    }


def _require(session_id: str) -> ChatSession:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"no such chat session: {session_id}")
    return session


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str) -> dict:
    return {"messages": _require(session_id).messages()}


class UserMessage(BaseModel):
    text: str
    include_context: bool = False


@router.post("/sessions/{session_id}/messages", status_code=202)
def post_message(session_id: str, body: UserMessage) -> dict:
    session = _require(session_id)
    if not body.text.strip():
        raise HTTPException(400, "empty message")
    err = session.send_user(body.text, body.include_context)
    if err:
        raise HTTPException(503, err)
    return {"status": "accepted"}


@router.post("/sessions/{session_id}/interrupt", status_code=202)
def post_interrupt(session_id: str) -> dict:
    _require(session_id).interrupt()
    return {"status": "accepted"}


class PermissionDecision(BaseModel):
    request_id: str
    decision: str  # "allow" | "deny"
    note: str = ""


@router.post("/sessions/{session_id}/permission")
def post_permission(session_id: str, body: PermissionDecision) -> dict:
    session = _require(session_id)
    approved = body.decision == "allow"
    ok = session.send_permission(body.request_id, approved, body.note)
    return {"status": "ok" if ok else "not_delivered", "decision": body.decision}


@router.websocket("/ws/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str) -> None:
    """Live tail of a chat session. Origin-checked like every other WS route."""
    if not origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008, reason="origin not allowed")
        return
    session = _SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=1008, reason="unknown session")
        return
    await websocket.accept()
    session.subscribers.add(websocket)
    # Replay so a reconnecting/late browser sees the conversation so far.
    for event in list(session.recent):
        try:
            await websocket.send_json(event)
        except Exception:
            break
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        session.subscribers.discard(websocket)

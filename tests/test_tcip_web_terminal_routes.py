"""Integration tests for the embedded agent terminal, driving a real PTY with the
scripted fake program (``tests/fake_terminal_app.py`` via ``TCIP_TERMINAL_CMD``).

These exercise the actual platform PTY backend (ConPTY on Windows, stdlib pty on
POSIX/CI): the seam whose failure mode is "the panel goes silent".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web import terminal as pty_host
from tcip_web.app import app
from tcip_web.routes import terminal as terminal_routes

FAKE = Path(__file__).parent / "fake_terminal_app.py"

if pty_host.os.name == "nt":
    pytest.importorskip("winpty")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture(autouse=True)
def _fake_terminal(monkeypatch):
    monkeypatch.setenv("TCIP_TERMINAL_CMD", f"{sys.executable} -u {FAKE}")
    yield
    terminal_routes.shutdown_all()


def _read_until(ws, needle: str, tries: int = 200) -> str:
    """Accumulate WS text frames until ``needle`` appears (bounded)."""
    acc = ""
    for _ in range(tries):
        acc += ws.receive_text()
        if needle in acc:
            return acc
    raise AssertionError(f"never saw {needle!r} in terminal stream; got: {acc[-500:]!r}")


# ── unit-ish: command resolution + preflight ────────────────────────────


def test_resolve_command_override(monkeypatch):
    monkeypatch.setenv("TCIP_TERMINAL_CMD", "python fake.py")
    assert pty_host.resolve_terminal_command()[-1] == "fake.py"


def test_resolve_command_none_when_cli_absent(monkeypatch):
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_TERMINAL_CLI", "definitely-not-a-real-cli-xyz")
    assert pty_host.resolve_terminal_command() is None


def test_status_available_with_fake(client):
    assert client.get("/api/terminal/status").json() == {"available": True}


def test_status_unavailable_without_cli(client, monkeypatch):
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_TERMINAL_CLI", "definitely-not-a-real-cli-xyz")
    body = client.get("/api/terminal/status").json()
    assert body["available"] is False
    assert "reason" in body


# ── session lifecycle over a real PTY ───────────────────────────────────


def test_create_session_spawns_and_streams_banner(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")


def test_input_round_trip(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "hello agent\r"})
        _read_until(ws, "echo:hello agent")


def test_attach_semantics_second_create_returns_live_session(client):
    first = client.post("/api/terminal/sessions", json={}).json()
    second = client.post("/api/terminal/sessions", json={}).json()
    assert second["session_id"] == first["session_id"]
    assert second["existing"] is True


def test_scrollback_replays_on_reconnect(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "before reconnect\r"})
        _read_until(ws, "echo:before reconnect")
    # New socket: the banner and the echoed line replay from scrollback.
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws2:
        acc = _read_until(ws2, "echo:before reconnect")
        assert "FAKE_TERMINAL_READY" in acc


def test_resize_does_not_crash_stream(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "resize", "rows": 99999, "cols": -3})  # clamped, not fatal
        ws.send_json({"type": "input", "data": "after resize\r"})
        _read_until(ws, "echo:after resize")


def test_a_resize_that_raises_does_not_end_the_stream(client, monkeypatch):
    """A resize failure at the session boundary (a ``ValueError``/``TypeError``, whatever its
    source) must be swallowed there rather than ending the websocket loop."""
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]

    def _raise(rows: int, cols: int) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(session, "resize", _raise)
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "input", "data": "after raising resize\r"})
        _read_until(ws, "echo:after raising resize")


def test_process_exit_is_visible_in_stream(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        acc = _read_until(ws, "Claude Code exited")
        assert "FAKE_TERMINAL_BYE" in acc


def test_restart_gives_fresh_process(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        _read_until(ws, "Claude Code exited")

    resp = client.post(f"/api/terminal/sessions/{sid}/restart", json={})
    assert resp.status_code == 200
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws2:
        acc = _read_until(ws2, "FAKE_TERMINAL_READY")
        # Scrollback was cleared: the old session's exit note is gone.
        assert "exited" not in acc


def test_ws_rejects_unknown_session(client):
    with pytest.raises(Exception):
        with client.websocket_connect("ws://127.0.0.1/api/terminal/ws/nonexistent"):
            pass


def test_ws_rejects_cross_site_origin(client):
    """A live session id, so a foreign origin is what refuses this connect, not an unknown one.
    Passes at the baseline too, since the handler already refused a foreign Origin there:
    preservation coverage for the move into the middleware, not a guard for this change."""
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"ws://127.0.0.1/api/terminal/ws/{sid}", headers={"origin": "https://evil.example"}
        ):
            pass


def test_create_503_when_unavailable(client, monkeypatch):
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_TERMINAL_CLI", "definitely-not-a-real-cli-xyz")
    resp = client.post("/api/terminal/sessions", json={})
    assert resp.status_code == 503


def _os_pid_alive(pid: int) -> bool:
    """OS-level liveness (not the session's own bookkeeping, which is what's under test)."""
    if pty_host.os.name == "nt":
        out = __import__("subprocess").run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
        )
        return str(pid) in out.stdout
    try:
        pty_host.os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but owned elsewhere
        return True


def test_shutdown_all_terminates_process(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]
    assert session.alive()
    pid = session._pty.pid  # capture before shutdown nulls the pty
    terminal_routes.shutdown_all()
    # Assert at the OS level: session.alive() flips False the moment _pty is nulled,
    # which would pass even if the kill itself were a no-op.
    deadline = time.time() + 15
    while time.time() < deadline and _os_pid_alive(pid):
        time.sleep(0.1)
    assert not _os_pid_alive(pid)
    assert terminal_routes._SESSIONS == {}


def test_write_and_resize_after_process_death_do_not_raise(client):
    # pywinpty raises EOFError/WinptyError (not OSError) on a dead PTY: a keystroke or
    # resize racing process exit must degrade, never crash the WS handler.
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        _read_until(ws, "Claude Code exited")
        # The socket must survive post-exit control traffic.
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "input", "data": "into the void\r"})
    assert session.write("x") is False
    session.resize(10, 10)  # no raise


def test_concurrent_creates_spawn_single_session():
    from concurrent.futures import ThreadPoolExecutor

    def create(_):
        return TestClient(app, base_url="http://127.0.0.1").post("/api/terminal/sessions", json={}).json()["session_id"]

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            ids = list(ex.map(create, range(8)))
        alive = [s for s in terminal_routes._SESSIONS.values() if s.alive()]
        assert len(alive) == 1
        assert set(ids) == {alive[0].id}
    finally:
        terminal_routes.shutdown_all()


# ── what a session records about the program it launched ───────────────


def _terminal_start_rows() -> list[dict]:
    import tcip_mcp.audit as audit_module
    import tcip_store as ts

    key = audit_module.audit_log_key(audit_module.platform_audit_scope())
    return [row for row in ts.read_log(key).records if row["tool"] == "agent_terminal_started"]


def test_create_answers_the_launched_executable_and_no_version_for_an_override(client):
    """An override argv is any program an operator or a test chose, so it is recorded as launched
    and never run a second time to ask its version."""
    body = client.post("/api/terminal/sessions", json={}).json()

    launched = body["launched"]
    assert Path(launched["executable"]).name == Path(sys.executable).name
    assert launched["version"] is None


def test_the_resolved_cli_is_probed_for_the_version_it_declares(monkeypatch):
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    monkeypatch.setenv("TCIP_TERMINAL_CLI", Path(sys.executable).name)
    argv = pty_host.resolve_terminal_command()
    assert argv is not None

    launched = pty_host.launched_program(argv)

    assert Path(launched["executable"]).name == Path(sys.executable).name
    assert launched["version"].startswith("Python ")


def test_the_spawned_process_inherits_the_terminal_session_id(client):
    """The double prints the id it inherited inside brackets, so the read ends at the closing
    bracket whatever the id is and the assertion, not the stream, decides."""
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        banner = _read_until(ws, "]")
    assert f"[session:{sid}]" in banner


def test_each_launch_leaves_one_platform_audit_line_naming_the_session_and_program(client):
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        _read_until(ws, "Claude Code exited")
    client.post(f"/api/terminal/sessions/{sid}/restart", json={})

    rows = _terminal_start_rows()
    assert [row["arguments"]["session_id"] for row in rows] == [sid, sid]
    assert Path(rows[0]["arguments"]["executable"]).name == Path(sys.executable).name
    assert rows[0]["arguments"]["version"] is None


def test_the_create_and_restart_responses_answer_the_launched_program(client):
    created = client.post("/api/terminal/sessions", json={}).json()
    sid = created["session_id"]
    assert Path(created["launched"]["executable"]).name == Path(sys.executable).name

    restarted = client.post(f"/api/terminal/sessions/{sid}/restart", json={}).json()
    assert restarted["alive"] is True
    assert Path(restarted["launched"]["executable"]).name == Path(sys.executable).name

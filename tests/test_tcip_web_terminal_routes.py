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


def test_winpty_terminate_polls_isalive_rather_than_trusting_taskkills_return(monkeypatch):
    """``_WinPty``'s own ``winpty`` import sits inside ``__init__``, never at module level, so
    its class body is importable without winpty installed; this stubs the wrapped process
    directly rather than spawning a real ConPTY. taskkill returning is not the process exiting:
    terminate() polls ``isalive()`` (bounded by ``TERMINATE_WAIT_S``) rather than trusting it."""

    class _StubProc:
        pid = 4242

        def __init__(self) -> None:
            self.calls = 0

        def isalive(self) -> bool:
            self.calls += 1
            return self.calls == 1  # alive at the liveness guard, dead by the first poll

    stub = _StubProc()
    win_pty = pty_host._WinPty.__new__(pty_host._WinPty)
    win_pty._p = stub
    win_pty.pid = stub.pid

    monkeypatch.setattr(pty_host.subprocess, "run", lambda *a, **kw: None)
    sleeps: list[float] = []
    monkeypatch.setattr(pty_host.time, "sleep", sleeps.append)

    win_pty.terminate()

    assert stub.calls == 2
    assert sleeps == []


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
    """Coverage: a live session id, so a foreign origin is what refuses this connect, not an
    unknown session."""
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"ws://127.0.0.1/api/terminal/ws/{sid}", headers={"origin": "https://evil.example"}
        ):
            pass


# ── the terminate/restart survivor branch (unit-level: no real process needed) ──


class _StubPty:
    """Stands in for a real process whose termination outcome the test controls, so the
    survivor branch is reachable without racing an actual OS process."""

    def __init__(self, *, survives: bool) -> None:
        self._alive = True
        self._survives = survives
        self.terminate_calls = 0

    def isalive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self._survives:
            self._alive = False


def test_terminate_reports_a_survivor_and_restores_the_pty() -> None:
    from tcip_web.routes.terminal import TerminalSession

    session = TerminalSession("term_survivor")
    session._pty = _StubPty(survives=True)

    assert session.terminate() is False
    assert session.alive() is True


def test_terminate_reports_a_clean_stop() -> None:
    from tcip_web.routes.terminal import TerminalSession

    session = TerminalSession("term_clean")
    session._pty = _StubPty(survives=False)

    assert session.terminate() is True
    assert session.alive() is False


def test_restart_on_a_survivor_answers_an_error_without_calling_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A survivor of ``terminate()`` must never reach ``start()``: that call would find
    ``self._pty`` alive again and report success with nothing actually spawned."""
    from tcip_web.routes.terminal import TerminalSession

    session = TerminalSession("term_restart_survivor")
    session._pty = _StubPty(survives=True)

    def _fail_if_called(rows: int, cols: int) -> None:
        raise AssertionError("start() must not run on a survivor")

    monkeypatch.setattr(session, "start", _fail_if_called)
    err = session.restart(24, 80)
    assert err is not None
    assert "could not be stopped" in err
    assert session.alive() is True


def test_restart_on_a_died_cleanly_process_calls_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from tcip_web.routes.terminal import TerminalSession

    session = TerminalSession("term_restart_clean")
    session._pty = _StubPty(survives=False)
    calls = {"n": 0}

    def _record(rows: int, cols: int) -> None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(session, "start", _record)
    err = session.restart(24, 80)
    assert err is None
    assert calls["n"] == 1


def test_shutdown_all_logs_a_survivor(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    from tcip_web.routes import terminal as terminal_routes

    session = terminal_routes.TerminalSession("term_shutdown_survivor")
    session._pty = _StubPty(survives=True)
    terminal_routes._SESSIONS[session.id] = session
    try:
        with caplog.at_level("WARNING"):
            terminal_routes.shutdown_all()
        assert any("survived" in r.message for r in caplog.records)
    finally:
        terminal_routes._SESSIONS.pop(session.id, None)


def _wire_stub_spawn(monkeypatch: pytest.MonkeyPatch, stub: "_StubPty") -> None:
    from tcip_web import terminal as pty_host

    monkeypatch.setattr(pty_host, "spawn_pty", lambda *a, **kw: stub)
    monkeypatch.setattr(pty_host, "start_reader", lambda *a, **kw: None)


def _lenient_client() -> TestClient:
    """A baseline run's ``start()`` has no ``try/except`` around ``_record_start`` at all, so a
    forced raise there reaches the ASGI layer uncaught; ``raise_server_exceptions=False`` turns
    that into a real (failing) response instead of an error the test can't assert on."""
    return TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)


def test_create_session_registers_a_survivor_and_answers_503(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_record_start``'s audit line fails after the process spawned; the spawned process
    survives its own termination attempt, so the session must stay reachable rather than
    orphaning a live process no later request can attach to."""
    from tcip_web.routes import terminal as terminal_routes

    stub = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, stub)

    from tcip_mcp.audit import AuditEntryNotWritten

    def _refuse_record_start(session_id: str, launched: dict) -> None:
        raise AuditEntryNotWritten("agent_terminal_started", RuntimeError("audit log unwritable"))

    monkeypatch.setattr(terminal_routes, "_record_start", _refuse_record_start)

    resp = _lenient_client().post("/api/terminal/sessions", json={})
    assert resp.status_code == 503
    assert "stays attached" in resp.json()["detail"]
    assert len(terminal_routes._SESSIONS) == 1
    (session,) = terminal_routes._SESSIONS.values()
    assert session.alive() is True


def test_restart_session_answers_503_on_a_survivor_with_no_new_spawn(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_web.routes import terminal as terminal_routes

    healthy_stub = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, healthy_stub)
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]

    survivor_stub = _StubPty(survives=True)
    monkeypatch.setattr(session, "_pty", survivor_stub)

    def _refuse_record_start(session_id: str, launched: dict) -> None:
        from tcip_mcp.audit import AuditEntryNotWritten
        raise AuditEntryNotWritten(
            "agent_terminal_started", RuntimeError("audit log unwritable"))

    monkeypatch.setattr(terminal_routes, "_record_start", _refuse_record_start)

    resp = _lenient_client().post(f"/api/terminal/sessions/{sid}/restart", json={})
    assert resp.status_code == 503
    assert "stays attached" in resp.json()["detail"]
    assert session._pty is survivor_stub  # no new process spawned over it


def test_restart_session_answers_503_after_the_process_dies_cleanly(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The died-cleanly path reaches ``start`` (unlike the survivor path, which short-circuits
    before it); the relaunch's own audit line then fails, and a cleanly terminated relaunch
    carries none of the survivor wording. Coverage: the 503 on a failed relaunch line
    predates this test, which pins the wording rather than guarding the status."""
    from tcip_web.routes import terminal as terminal_routes

    healthy_stub = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, healthy_stub)
    sid = client.post("/api/terminal/sessions", json={}).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]

    died_stub = _StubPty(survives=False)
    monkeypatch.setattr(session, "_pty", died_stub)

    relaunch_stub = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, relaunch_stub)

    def _refuse_record_start(session_id: str, launched: dict) -> None:
        from tcip_mcp.audit import AuditEntryNotWritten
        raise AuditEntryNotWritten(
            "agent_terminal_started", RuntimeError("audit log unwritable"))

    monkeypatch.setattr(terminal_routes, "_record_start", _refuse_record_start)

    resp = _lenient_client().post(f"/api/terminal/sessions/{sid}/restart", json={})
    assert resp.status_code == 503
    assert "stays attached" not in resp.json()["detail"]


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


def test_create_session_answers_503_and_terminates_the_process_when_the_start_line_fails(
    client, monkeypatch,
):
    """A spawn whose own launch line fails to append must not leave an orphaned, untracked
    process: the PTY is terminated and, once it is gone, no session is registered as live."""
    import tcip_mcp.audit as audit_module

    def _refuse_append(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", _refuse_append)
    resp = client.post("/api/terminal/sessions", json={})
    assert resp.status_code == 503
    assert "could not be written" in resp.json()["detail"]
    assert _terminal_start_rows() == []
    assert all(not s.alive() for s in terminal_routes._SESSIONS.values())


def test_the_create_and_restart_responses_answer_the_launched_program(client):
    created = client.post("/api/terminal/sessions", json={}).json()
    sid = created["session_id"]
    assert Path(created["launched"]["executable"]).name == Path(sys.executable).name

    restarted = client.post(f"/api/terminal/sessions/{sid}/restart", json={}).json()
    assert restarted["alive"] is True
    assert Path(restarted["launched"]["executable"]).name == Path(sys.executable).name

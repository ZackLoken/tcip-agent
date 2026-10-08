"""The embedded agent terminal over the platform's real PTY backend, driven by the fake agent
``tests/fake_terminal_app.py`` through ``TCIP_TERMINAL_CMD`` or standing as a provider row's
executable."""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_web import terminal as pty_host
from tcip_web.app import app
from tcip_web.routes import terminal as terminal_routes
from tcip_web.state import OpenProject
from tests._audit_fixtures import audit_rows

FAKE = Path(__file__).parent / "fake_terminal_app.py"
LAUNCH = {"provider": pty_host.PROVIDERS[0].id, "user": "tester"}
ABSENT = "definitely-not-a-real-cli-xyz"
_PLACEHOLDERS = (pty_host.WORKSPACE_ARG, pty_host.MCP_CONFIG_ARG, pty_host.CODEX_MCP_ARG)

if pty_host.os.name == "nt":
    pytest.importorskip("winpty")


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


def _rows_with(monkeypatch, **changes: str) -> tuple[pty_host.Provider, ...]:
    """Every provider row with ``changes`` applied, installed as the table; the override is
    cleared so the rows themselves launch."""
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    rows = tuple(dataclasses.replace(row, **changes) for row in pty_host.PROVIDERS)
    monkeypatch.setattr(pty_host, "PROVIDERS", rows)
    return rows


def _fake_executable(directory: Path) -> Path:
    """An executable file running the fake program with the arguments it is given, so the fake
    can stand as a provider row's executable."""
    directory.mkdir()
    if pty_host.os.name == "nt":
        wrapper = directory / "fake_agent.cmd"
        wrapper.write_text(f'@"{sys.executable}" -u "{FAKE}" %*\r\n', encoding="utf-8")
    else:
        wrapper = directory / "fake_agent"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -u "{FAKE}" "$@"\n',
                           encoding="utf-8")
        wrapper.chmod(0o755)
    return wrapper


# ── unit-ish: command resolution + preflight ────────────────────────────


def test_resolve_command_override(monkeypatch):
    monkeypatch.setenv("TCIP_TERMINAL_CMD", "python fake.py")
    assert pty_host.resolve_terminal_command(pty_host.PROVIDERS[0]) == (["python", "fake.py"], True)


def test_resolve_command_none_when_the_rows_executable_is_absent(monkeypatch):
    (row, *_) = _rows_with(monkeypatch, executable=ABSENT)
    assert pty_host.resolve_terminal_command(row) is None


def test_status_lists_every_row_available_with_fake(client):
    assert client.get("/api/terminal/status").json() == {"providers": [
        {"id": row.id, "name": row.name, "unavailable_reason": None}
        for row in pty_host.PROVIDERS
    ]}


def test_a_row_whose_executable_is_absent_reports_its_own_reason_and_refuses_create(
    client, monkeypatch,
):
    rows = _rows_with(monkeypatch, executable=ABSENT)
    body = client.get("/api/terminal/status").json()
    assert [entry["unavailable_reason"] for entry in body["providers"]] == [
        row.unavailable_reason for row in rows]
    assert all(ABSENT in entry["unavailable_reason"] for entry in body["providers"])

    resp = client.post("/api/terminal/sessions", json={**LAUNCH, "provider": rows[0].id})
    assert resp.status_code == 503
    assert ABSENT in resp.json()["detail"]
    assert terminal_routes._SESSIONS == {}


def test_a_create_naming_an_unlisted_provider_refuses_by_name_and_a_listed_one_launches(client):
    for unlisted in ("no-such-harness", LAUNCH["provider"].upper()):
        resp = client.post("/api/terminal/sessions", json={**LAUNCH, "provider": unlisted})
        assert resp.status_code == 422
        assert unlisted in resp.json()["detail"]
    assert client.post("/api/terminal/sessions", json={}).status_code == 422
    assert terminal_routes._SESSIONS == {}

    resp = client.post("/api/terminal/sessions", json=LAUNCH)
    assert resp.status_code == 200
    assert resp.json()["launched"]["provider"] == LAUNCH["provider"]


def test_rendering_replaces_each_placeholder_and_leaves_every_other_argument(opened):
    from tcip_web.state import store

    literal, workspace, config = pty_host.render_argv(
        ["--literal", pty_host.WORKSPACE_ARG, pty_host.MCP_CONFIG_ARG], opened,
        pty_host.spawn_env("term_test"))

    assert (literal, workspace) == ("--literal", str(store.workspace))
    server = json.loads(Path(config).read_text(encoding="utf-8"))["mcpServers"]["tcip"]
    assert server["args"] == ["-m", "tcip_mcp", "--project", opened.root.as_posix()]


def test_the_ritual_for_no_project_says_the_session_has_none():
    ritual = pty_host.session_ritual(None)
    assert "has no project" in ritual
    assert "initialize_project" in ritual


def test_prewarm_runs_without_raising():
    pty_host.prewarm()
    pty_host._prewarm_blocking()

    assert "tcip_mcp.server" in sys.modules


def test_the_agent_runs_at_the_repo_root(tmp_path, monkeypatch):
    """The spawned agent's cwd is the repo root itself, not the package directory this module
    happens to live in, and does not depend on how the web process was started."""
    monkeypatch.delenv(pty_host.TERMINAL_CWD_ENV, raising=False)
    from_install_dir = Path(pty_host.terminal_cwd())
    monkeypatch.chdir(tmp_path)
    assert Path(pty_host.terminal_cwd()) == from_install_dir
    assert (from_install_dir / ".mcp.json").is_file()
    assert (from_install_dir / "packages").is_dir()


def test_winpty_terminate_polls_isalive_rather_than_trusting_taskkills_return(monkeypatch):
    """``_WinPty.terminate`` polls ``isalive()`` after taskkill returns, within
    ``TERMINATE_WAIT_S``."""

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
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")


@pytest.mark.parametrize("row", pty_host.PROVIDERS, ids=lambda row: row.id)
def test_each_rows_rendered_argv_spawns_in_a_real_pty_and_streams(
    row, client, monkeypatch, tmp_path, opened, opened_project,
):
    """The fake program, standing as the row's executable, receives exactly the arguments the
    row renders, with no placeholder left in them, after the row's preparation ran with it."""
    from tcip_mcp.server import list_registered_tools

    monkeypatch.setattr(pty_host, "PROVIDERS", (dataclasses.replace(
        row, executable=str(_fake_executable(tmp_path / "bin"))),))
    monkeypatch.delenv("TCIP_TERMINAL_CMD")
    settings = tmp_path / "agy-settings.json"
    monkeypatch.setattr(pty_host, "ANTIGRAVITY_SETTINGS", settings)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_TERMINAL_ARGV_FILE", str(argv_file))
    rendered: list[list[str]] = []
    render = pty_host.render_argv

    def _recording_render(argv: list[str], project: OpenProject | None,
                          env: dict[str, str]) -> list[str]:
        rendered.append(render(argv, project, env))
        return rendered[-1]

    monkeypatch.setattr(pty_host, "render_argv", _recording_render)

    created = client.post("/api/terminal/sessions", json={**LAUNCH, "provider": row.id}).json()
    sid = created["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")

    (launched,) = rendered
    assert json.loads(argv_file.read_text(encoding="utf-8")) == launched[1:]
    assert not any(placeholder in arg for arg in launched for placeholder in _PLACEHOLDERS)
    if row.prepare is None:
        expected_prepared: list[str] = []
    else:
        server = pty_host.mcp_server(opened)
        expected_prepared = [
            pty_host.subprocess.list2cmdline(
                [launched[0], "mcp", "add", "tcip", "--", server.command, *server.args]),
            f"allowed {len(list_registered_tools())} tcip tools in {settings}",
        ]
        allowed = json.loads(settings.read_text(encoding="utf-8"))["permissions"]["allow"]
        assert allowed == [f"mcp(tcip/{name})" for name in list_registered_tools()]
    assert created["launched"]["prepared"] == expected_prepared
    assert created["launched"]["confinement"] == row.confinement
    assert created["launched"]["delivery_unverified"] == row.delivery_unverified(
        created["launched"]["version"])
    (line,) = audit_rows(opened_project, "agent_terminal_started")
    assert line["arguments"]["prepared"] == expected_prepared
    assert line["arguments"]["confinement"] == row.confinement


def test_a_launch_preparation_that_fails_refuses_the_launch_naming_it(
    client, monkeypatch, tmp_path, opened_project,
):
    """A row whose preparation step exits non-zero spawns nothing and records nothing."""
    def prepare(executable: str, project: OpenProject | None) -> list[str]:
        return [pty_host.run_to_completion([executable, "fail", "tcip"], "the fake step")]

    row = dataclasses.replace(pty_host.PROVIDERS[0], executable=str(_fake_executable(
        tmp_path / "bin")), prepare=prepare)
    monkeypatch.setattr(pty_host, "PROVIDERS", (row,))
    monkeypatch.delenv("TCIP_TERMINAL_CMD")

    resp = client.post("/api/terminal/sessions", json={**LAUNCH, "provider": row.id})

    assert resp.status_code == 503
    assert "launch preparation failed: the fake step exited 3" in resp.json()["detail"]
    assert "FAKE_PREPARATION_FAILED" in resp.json()["detail"]
    assert terminal_routes._SESSIONS == {}
    assert audit_rows(opened_project, "agent_terminal_started") == []


def test_allowing_the_tcip_tools_keeps_the_rest_of_the_settings_and_is_idempotent(tmp_path):
    from tcip_mcp.server import list_registered_tools

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "model": "kept", "permissions": {"allow": ["command(git log)"], "deny": ["x"]}}),
        encoding="utf-8")

    first = pty_host.allow_tcip_tools(settings)
    second = pty_host.allow_tcip_tools(settings)

    names = list_registered_tools()
    assert first == f"allowed {len(names)} tcip tools in {settings}"
    assert second == f"allowed 0 tcip tools in {settings}"
    body = json.loads(settings.read_text(encoding="utf-8"))
    assert body["model"] == "kept"
    assert body["permissions"]["deny"] == ["x"]
    assert body["permissions"]["allow"] == [
        "command(git log)", *(f"mcp(tcip/{name})" for name in names)]

    settings.write_text("[]", encoding="utf-8")
    with pytest.raises(pty_host.PreparationFailedError, match="does not hold a JSON object"):
        pty_host.allow_tcip_tools(settings)


def test_the_written_mcp_configuration_is_the_one_server_the_launch_states(opened):
    server = pty_host.mcp_server(opened)
    assert server.args == ("-m", "tcip_mcp", "--project", opened.root.as_posix())

    (config,) = pty_host.render_argv([pty_host.MCP_CONFIG_ARG], opened,
                                     pty_host.spawn_env("term_test"))

    written = json.loads(Path(config).read_text(encoding="utf-8"))["mcpServers"]["tcip"]
    assert (written["command"], written["args"]) == (server.command, list(server.args))


def test_the_codex_overrides_name_the_one_server_and_forward_the_spawn_environment(opened):
    """Each rendered ``-c`` override parses as TOML, and together they state the server
    :func:`mcp_server` produces, every variable of the launch's environment forwarded to it."""
    import tomllib

    env = pty_host.spawn_env("term_test")
    rendered = pty_host.render_argv(["--literal", pty_host.CODEX_MCP_ARG], opened, env)

    assert rendered[0] == "--literal"
    flags, overrides = rendered[1::2], rendered[2::2]
    assert set(flags) == {"-c"}
    parsed: dict = {}
    for override in overrides:
        parsed.update(tomllib.loads(override)["mcp_servers"]["tcip"])
    server = pty_host.mcp_server(opened)
    assert (parsed["command"], parsed["args"]) == (server.command, list(server.args))
    assert set(parsed["env_vars"]) == set(env)
    assert pty_host.TERMINAL_SESSION_ENV in parsed["env_vars"]
    assert parsed["default_tools_approval_mode"] == "approve"


_RITUAL_ECHO = "echo:[TCIP session-start ritual]"


def _wait_for(session: terminal_routes.TerminalSession, needle: str, timeout: float = 30) -> str:
    """The session's scrollback once it holds ``needle``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = session.scrollback_snapshot()
        if needle in text:
            return text
        time.sleep(0.1)
    raise AssertionError(f"never saw {needle!r}; got: {session.scrollback_snapshot()[-500:]!r}")


def _echoes(scrollback: str) -> list[str]:
    """Every line the fake agent echoed, in order, the ritual's shortened to its header."""
    return [_RITUAL_ECHO if line.startswith(_RITUAL_ECHO) else line
            for line in re.findall(r"echo:[^\r\n]*", scrollback)]


@pytest.fixture
def pastes_after(monkeypatch):
    """The fake agent turns bracketed paste on one second after it starts and writes the composer
    sequence of the row it stands in for."""
    monkeypatch.setenv("FAKE_TERMINAL_PASTE_AFTER_S", "1")
    monkeypatch.setenv("FAKE_TERMINAL_COMPOSER_READY", pty_host.PROVIDERS[0].composer_ready)


def test_the_ritual_then_a_submitted_request_reach_the_agent_once_it_turns_paste_on(
    client, pastes_after,
):
    sid = client.post("/api/terminal/sessions", json={**LAUNCH, "cols": 500}).json()["session_id"]
    client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": "first request"})

    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:first request")
    assert _echoes(text) == [_RITUAL_ECHO, "echo:first request"]
    assert "EARLY_INPUT" not in text

    client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": "second request"})
    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:second request")
    assert _echoes(text) == [_RITUAL_ECHO, "echo:first request", "echo:second request"]


def test_the_ritual_and_the_requests_staged_before_paste_is_on_go_as_one_paste():
    """The ritual and every request queued by the time the agent turns bracketed paste on are
    written as one paste; a request submitted afterwards is its own paste."""
    writes: list[str] = []

    class _Pty:
        def isalive(self) -> bool:
            return True

        def write(self, data: str) -> None:
            writes.append(data)

    session = terminal_routes.TerminalSession("term_test")
    session._pty = _Pty()
    session._launch.ritual = "the ritual"
    session._launch.composer_ready = "\x1b[?1049h"
    session.submit("first")
    session.submit("second")
    assert writes == []

    session._on_output("\x1b[?2004h\x1b[?1049h")
    assert writes == [pty_host.paste("the ritual\n\nfirst\n\nsecond")]

    session.submit("later")
    assert writes[1:] == [pty_host.paste("later")]
    assert pty_host.paste("a\nb\r\nc") == "\x1b[200~a\rb\rc\x1b[201~\r"


def test_a_restart_delivers_its_own_ritual_then_the_request_submitted_during_it(
    client, pastes_after,
):
    sid = client.post("/api/terminal/sessions", json={**LAUNCH, "cols": 500}).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]
    _wait_for(session, _RITUAL_ECHO)

    client.post(f"/api/terminal/sessions/{sid}/restart", json={**LAUNCH, "cols": 500})
    client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": "after restart"})

    text = _wait_for(session, "echo:after restart")
    assert _echoes(text) == [_RITUAL_ECHO, "echo:after restart"]
    assert "EARLY_INPUT" not in text


def test_overlapping_creates_deliver_one_ritual_before_the_request(client, pastes_after):
    from concurrent.futures import ThreadPoolExecutor

    def create(_):
        return TestClient(app, base_url="http://127.0.0.1").post(
            "/api/terminal/sessions", json={**LAUNCH, "cols": 500}).json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(create, range(2)))
    assert sorted(answer["existing"] for answer in answers) == [False, True]
    sid = answers[0]["session_id"]
    client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": "tab request"})

    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:tab request")
    assert _echoes(text) == [_RITUAL_ECHO, "echo:tab request"]


class _RecordingPty:
    """A live PTY that keeps what the session writes to it."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def isalive(self) -> bool:
        return True

    def write(self, data: str) -> None:
        self.writes.append(data)


_COMPOSER_LIVE_AT = {"claude": 15, "antigravity": 7, "codex": 10}
"""For each recorded harness, the index of the startup read at which its composer was observed
live, from the PTY logs and never from a row's marker: Claude Code drew its input prompt in read 15,
and a paste at read 5 was lost; agy left its loading splash and drew its prompt in read 7, and a
paste at read 3 was lost; Codex wrote its first window title in read 10, its session footer
followed, and an Enter before it was held as "Waiting for startup" while a paste at it was
submitted."""

_STARTUP_READS = {
    "claude": [
        "\x1b[1t", "\x1b[c\x1b[?1004h\x1b[?9001h", "\x1b]0;claude\x1b\\",
        "\x1b7\x1b[r\x1b8\x1b[?25h", "\x1b[?25l", "\x1b[?2004h\x1b[?2031h\x1b[?1004h",
        "\x1b[?9001l\x1b[?9001h", "\x1b[>0q", "\x1b[?u", "\x1b[c",
        "\x1b[>4m\x1b[<u\x1b[?1004l\x1b[?1004h\x1b[?2031l\x1b[?2004l", "\x1b[1;1H",
        "\r\n\x1b[1;1H", "\x1b[?2004h\x1b[?2031h\x1b[?1004h\x1b[?9001l\x1b[?9001h",
        "\x1b]0;✳ Claude Code\x1b\\",
        "\x1b[?1049h\x1b[2J\x1b[H\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h\x1b[?25l",
    ],
    "antigravity": [
        "\x1b[1t", "\x1b[c\x1b[?1004h\x1b[?9001h", "\x1b[?2026$p\x1b[?2027$p\x1b]11;?\x07",
        "\x1b[>4m\x1b[?1049h\x1b[?25l\x1b[?5W\x1b[?2004h\x1b[>4;2m\x1b[>1u\x1b[?u\x1b[H\x1b[2J",
        "\x1b[H\x1b[2J\n", "\r ⣯ ", "\r ⣟ ",
        "\x1b[>4m\x1b[<1u\x1b[?1049l\x1b[>4;2m\x1b[>1u\x1b[?u\x1b[0 q\r\x1b[J\n",
    ],
    "codex": [
        "\x1b[1t", "\x1b[c\x1b[?1004h\x1b[?9001h", "\x1b[?2004h", "\x1b[?1004l\x1b[?1004h",
        "\x1b[?2026h\x1b[?25l", "\x1b[?1049h", "\x1b[?1007l", "\x1b[?1003;1006h",
        "\x1b[2;3H\x1b[38;5;75;49m>_ ", "\x1b[?2026l", "\x1b]0;tcip-agent\x07",
        "\x1b]10;?\x1b\\\x1b]11;?\x1b\\",
    ],
}
"""The PTY reads each installed harness wrote while starting in the terminal's PTY (Claude Code
2.1.292 and agy 1.3.0 under ``tools/smoke_terminal_e2e.py``, Codex 0.160.1 launched with its row's
arguments and sent no input), long screen draws cut after their opening sequences and spinner frames
thinned. The first two dropped a paste written at their first bracketed-paste-on; Codex, under the
smoke, held one in its composer without submitting it, since its session had not started."""


@pytest.mark.parametrize("row", pty_host.PROVIDERS, ids=lambda row: row.id)
def test_the_ritual_is_pasted_at_the_read_where_the_harness_composer_goes_live(row):
    """Fed the reads its harness wrote at startup, a launch of ``row`` pastes nothing before the
    read at which that harness's composer was observed live and pastes the ritual at that read,
    even though bracketed paste turned on earlier."""
    reads = _STARTUP_READS[row.id]
    live_at = _COMPOSER_LIVE_AT[row.id]
    first_paste_on = next(i for i, data in enumerate(reads) if "\x1b[?2004h" in data)
    assert first_paste_on < live_at

    pty = _RecordingPty()
    session = terminal_routes.TerminalSession("term_test")
    session._pty = pty
    session._launch.ritual = "the ritual"
    session._launch.composer_ready = row.composer_ready
    session.submit("the request")
    for index, data in enumerate(reads):
        session._on_output(data)
        assert pty.writes == ([pty_host.paste("the ritual\n\nthe request")]
                              if index >= live_at else [])


@pytest.mark.parametrize("row", pty_host.PROVIDERS, ids=lambda row: row.id)
def test_a_launch_names_the_versions_when_its_harness_is_not_the_recorded_one(row):
    """Delivery rests on the recorded sequence without a note for the recorded version, and a
    note naming both versions for any other version or none."""
    assert row.delivery_unverified(row.composer_ready_version) is None
    note = row.delivery_unverified("9.9.9")
    assert note is not None and row.composer_ready_version in note and "9.9.9" in note
    assert row.delivery_unverified(None) is not None


def test_bracketed_paste_follows_the_last_mode_change_the_agent_wrote():
    assert pty_host.bracketed_paste("\x1b[?2004h", False) is True
    assert pty_host.bracketed_paste("\x1b[?1004;2004h\x1b[?2004l", True) is False
    assert pty_host.bracketed_paste("\x1b[?1049h plain output", True) is True


def test_input_round_trip(client):
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "hello agent\r"})
        _read_until(ws, "echo:hello agent")


def test_attach_semantics_second_create_returns_live_session(client):
    first = client.post("/api/terminal/sessions", json=LAUNCH).json()
    second = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert second["session_id"] == first["session_id"]
    assert second["existing"] is True


def test_scrollback_replays_on_reconnect(client):
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "before reconnect\r"})
        _read_until(ws, "echo:before reconnect")
    # New socket: the banner and the echoed line replay from scrollback.
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws2:
        acc = _read_until(ws2, "echo:before reconnect")
        assert "FAKE_TERMINAL_READY" in acc


def test_resize_does_not_crash_stream(client):
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "resize", "rows": 99999, "cols": -3})  # clamped, not fatal
        ws.send_json({"type": "input", "data": "after resize\r"})
        _read_until(ws, "echo:after resize")


def test_a_resize_that_raises_does_not_end_the_stream(client, monkeypatch):
    """A resize failure at the session boundary (a ``ValueError``/``TypeError``, whatever its
    source) must be swallowed there rather than ending the websocket loop."""
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
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
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        acc = _read_until(ws, "the agent exited")
        assert "FAKE_TERMINAL_BYE" in acc


def test_restart_gives_fresh_process(client):
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        _read_until(ws, "the agent exited")

    resp = client.post(f"/api/terminal/sessions/{sid}/restart", json=LAUNCH)
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
    """A live session's socket refuses a foreign origin."""
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"ws://127.0.0.1/api/terminal/ws/{sid}", headers={"origin": "https://evil.example"}
        ):
            pass


# ── the terminate/restart survivor branch (unit-level: no real process needed) ──


class _StubPty:
    """A process whose termination outcome the test sets: it survives or it stops."""

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

    def _fail_if_called(rows: int, cols: int, provider: pty_host.Provider) -> None:
        raise AssertionError("start() must not run on a survivor")

    monkeypatch.setattr(session, "start", _fail_if_called)
    err = session.restart(24, 80, pty_host.PROVIDERS[0], "user:tester")
    assert err is not None
    assert "could not be stopped" in err
    assert session.alive() is True


def test_restart_on_a_died_cleanly_process_calls_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from tcip_web.routes.terminal import TerminalSession

    session = TerminalSession("term_restart_clean")
    session._pty = _StubPty(survives=False)
    calls = {"n": 0}

    def _record(rows: int, cols: int, provider: pty_host.Provider, actor: str) -> None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(session, "start", _record)
    err = session.restart(24, 80, pty_host.PROVIDERS[0], "user:tester")
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
    """A client that answers a server exception as a 500 response."""
    return TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)


def _refuse_record_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every launch's audit line fail to append."""
    from tcip_mcp.audit import AuditEntryNotWrittenError

    def _refuse(session_id: str, launched: object, project: Path | None, actor: str) -> None:
        raise AuditEntryNotWrittenError(
            "agent_terminal_started", RuntimeError("audit log unwritable"))

    monkeypatch.setattr(terminal_routes, "_record_start", _refuse)


def test_create_session_registers_a_survivor_and_answers_503(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_record_start``'s audit line fails after the process spawned; the spawned process
    survives its own termination attempt, so the session must stay reachable rather than
    orphaning a live process no later request can attach to."""
    stub = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, stub)
    _refuse_record_start(monkeypatch)

    resp = _lenient_client().post("/api/terminal/sessions", json=LAUNCH)
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
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]

    survivor_stub = _StubPty(survives=True)
    monkeypatch.setattr(session, "_pty", survivor_stub)
    _refuse_record_start(monkeypatch)

    resp = _lenient_client().post(f"/api/terminal/sessions/{sid}/restart", json=LAUNCH)
    assert resp.status_code == 503
    assert "stays attached" in resp.json()["detail"]
    assert session._pty is survivor_stub  # no new process spawned over it


def test_restart_session_answers_503_after_the_process_dies_cleanly(
    client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relaunch whose audit line fails after a clean termination answers 503 with none of the
    survivor wording."""
    healthy_stub = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, healthy_stub)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]

    died_stub = _StubPty(survives=False)
    monkeypatch.setattr(session, "_pty", died_stub)

    relaunch_stub = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, relaunch_stub)
    _refuse_record_start(monkeypatch)

    resp = _lenient_client().post(f"/api/terminal/sessions/{sid}/restart", json=LAUNCH)
    assert resp.status_code == 503
    assert "stays attached" not in resp.json()["detail"]


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
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]
    assert session.alive()
    pid = session._pty.pid  # capture before shutdown nulls the pty
    terminal_routes.shutdown_all()
    deadline = time.time() + 15
    while time.time() < deadline and _os_pid_alive(pid):
        time.sleep(0.1)
    assert not _os_pid_alive(pid)
    assert terminal_routes._SESSIONS == {}


def test_write_and_resize_after_process_death_do_not_raise(client):
    # pywinpty raises EOFError/WinptyError (not OSError) on a dead PTY: a keystroke or
    # resize racing process exit must degrade, never crash the WS handler.
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    session = terminal_routes._SESSIONS[sid]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        _read_until(ws, "the agent exited")
        # The socket must survive post-exit control traffic.
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "input", "data": "into the void\r"})
    assert session.write("x") is False
    session.resize(10, 10)  # no raise


def test_concurrent_creates_spawn_single_session():
    from concurrent.futures import ThreadPoolExecutor

    def create(_):
        return TestClient(app, base_url="http://127.0.0.1").post(
            "/api/terminal/sessions", json=LAUNCH).json()["session_id"]

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            ids = list(ex.map(create, range(8)))
        alive = [s for s in terminal_routes._SESSIONS.values() if s.alive()]
        assert len(alive) == 1
        assert set(ids) == {alive[0].id}
    finally:
        terminal_routes.shutdown_all()


# ── what a session records about the program it launched ───────────────


def test_create_answers_the_launched_executable_and_no_version_for_an_override(client):
    """An override's launch records its executable, no version and no row's confinement."""
    body = client.post("/api/terminal/sessions", json=LAUNCH).json()

    launched = body["launched"]
    assert Path(launched["executable"]).name == Path(sys.executable).name
    assert launched["version"] is None
    assert launched["confinement"] is None
    assert launched["delivery_unverified"] is None


def test_the_resolved_cli_is_probed_for_the_version_it_declares(monkeypatch):
    (row, *_) = _rows_with(monkeypatch, executable=Path(sys.executable).name)
    command = pty_host.resolve_terminal_command(row)
    assert command is not None

    launched = pty_host.launched_program(*command)

    assert Path(launched["executable"]).name == Path(sys.executable).name
    assert launched["version"].startswith("Python ")


def test_the_resolved_command_decides_whether_its_version_is_probed(monkeypatch):
    command = pty_host.resolve_terminal_command(pty_host.PROVIDERS[0])
    assert command is not None
    monkeypatch.delenv("TCIP_TERMINAL_CMD")

    assert pty_host.launched_program(*command)["version"] is None


def _versioned_executable(directory: Path, version_file: Path) -> Path:
    """An executable that prints ``version_file``'s text for ``--version`` and otherwise runs the
    fake program, standing as a harness whose installed version can change in place."""
    directory.mkdir()
    if pty_host.os.name == "nt":
        wrapper = directory / "versioned_agent.cmd"
        wrapper.write_text(
            f'@if "%1"=="--version" (type "{version_file}" & exit /b 0)\r\n'
            f'@"{sys.executable}" -u "{FAKE}" %*\r\n', encoding="utf-8")
    else:
        wrapper = directory / "versioned_agent"
        wrapper.write_text(
            f'#!/bin/sh\nif [ "$1" = "--version" ]; then cat "{version_file}"; exit 0; fi\n'
            f'exec "{sys.executable}" -u "{FAKE}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
    return wrapper


def test_a_harness_updated_in_place_is_seen_at_its_next_launch_without_a_backend_restart(
    client, monkeypatch, tmp_path, opened_project,
):
    """The second launch of the same executable reports the version it declares then, and the
    launch and its audit line carry why delivery to that version is unverified."""
    version_file = tmp_path / "version.txt"
    row = pty_host.PROVIDERS[0]
    version_file.write_text(row.composer_ready_version + "\n", encoding="utf-8")
    monkeypatch.setattr(pty_host, "PROVIDERS", (dataclasses.replace(
        row, executable=str(_versioned_executable(tmp_path / "bin", version_file))),))
    monkeypatch.delenv("TCIP_TERMINAL_CMD")

    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert created["launched"]["version"] == row.composer_ready_version
    assert created["launched"]["delivery_unverified"] is None

    version_file.write_text("9.9.9 (updated)\n", encoding="utf-8")
    restarted = client.post(f"/api/terminal/sessions/{created['session_id']}/restart",
                            json=LAUNCH).json()

    assert restarted["launched"]["version"] == "9.9.9 (updated)"
    note = row.delivery_unverified("9.9.9 (updated)")
    assert note is not None
    assert restarted["launched"]["delivery_unverified"] == note
    first, second = audit_rows(opened_project, "agent_terminal_started")
    assert first["arguments"]["delivery_unverified"] is None
    assert second["arguments"]["delivery_unverified"] == note


def test_the_spawned_process_inherits_the_terminal_session_id(client):
    """The spawned process's environment names the session id."""
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        banner = _read_until(ws, "]")
    assert f"[session:{sid}]" in banner


def test_each_launch_leaves_one_audit_line_in_the_open_projects_log(client, opened_project):
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()["session_id"]
    with client.websocket_connect(f"ws://127.0.0.1/api/terminal/ws/{sid}") as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        _read_until(ws, "the agent exited")
    client.post(f"/api/terminal/sessions/{sid}/restart", json=LAUNCH)

    rows = audit_rows(opened_project, "agent_terminal_started")
    assert [row["arguments"]["session_id"] for row in rows] == [sid, sid]
    assert [row["arguments"]["provider"] for row in rows] == [LAUNCH["provider"]] * 2
    assert Path(rows[0]["arguments"]["executable"]).name == Path(sys.executable).name
    assert rows[0]["arguments"]["version"] is None
    assert [row["actor"] for row in rows] == ["user:tester"] * 2
    assert "source" not in rows[0]


def test_create_session_answers_503_and_terminates_the_process_when_the_start_line_fails(
    client, monkeypatch, opened_project,
):
    """A spawn whose own launch line fails to append must not leave an orphaned, untracked
    process: the PTY is terminated and, once it is gone, no session is registered as live."""
    import tcip_mcp.audit as audit_module

    def _refuse_append(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", _refuse_append)
    resp = client.post("/api/terminal/sessions", json=LAUNCH)
    assert resp.status_code == 503
    assert "could not be written" in resp.json()["detail"]
    assert audit_rows(opened_project, "agent_terminal_started") == []
    assert all(not s.alive() for s in terminal_routes._SESSIONS.values())


def test_the_create_and_restart_responses_answer_the_launched_provider_and_program(client):
    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    sid = created["session_id"]
    assert created["launched"]["provider"] == LAUNCH["provider"]
    assert Path(created["launched"]["executable"]).name == Path(sys.executable).name

    restarted = client.post(f"/api/terminal/sessions/{sid}/restart", json=LAUNCH).json()
    assert restarted["existing"] is False
    assert restarted["launched"]["provider"] == LAUNCH["provider"]
    assert Path(restarted["launched"]["executable"]).name == Path(sys.executable).name


def test_create_and_restart_answer_the_ritual_the_library_builds_for_the_open_project(
    client, opened,
):
    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert created["ritual"] == pty_host.session_ritual(opened)
    assert "Test project" in created["ritual"]
    for step in ("load_project_memory", "inspect_project", "tcip doctor"):
        assert step in created["ritual"]

    attached = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert attached["existing"] is True
    assert attached["ritual"] == created["ritual"]

    restarted = client.post(
        f"/api/terminal/sessions/{created['session_id']}/restart", json=LAUNCH).json()
    assert restarted["ritual"] == created["ritual"]


def test_a_launch_after_a_rename_of_the_open_project_names_its_new_name(client, opened):
    """The ritual reads the project's record when the launch prepares, so a rename made while
    the project is open is the name the next launch states."""
    renamed = client.post("/api/projects/rename", json={
        "id": opened.id, "display_name": "Ridge block", "user": "tester"})
    assert renamed.status_code == 200, renamed.text

    created = client.post("/api/terminal/sessions", json=LAUNCH)

    assert created.status_code == 200, created.text
    assert "Project: Ridge block" in created.json()["ritual"]


def test_a_launch_for_an_open_project_whose_record_will_not_read_refuses_naming_why(
    client, opened,
):
    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    damage_record(project_record_key(str(opened.root)), b"{not valid json")

    created = client.post("/api/terminal/sessions", json=LAUNCH)

    assert created.status_code == 503
    assert "names no readable project" in created.json()["detail"]
    assert terminal_routes._SESSIONS == {}

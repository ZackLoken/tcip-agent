"""The embedded agent terminal over the platform's real PTY backend.

The fake agent ``tests/fake_terminal_app.py`` drives it, through ``TCIP_TERMINAL_CMD`` or
standing as a provider row's executable.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from tcip_web import terminal as pty_host
from tcip_web.app import app
from tcip_web.routes import terminal as terminal_routes
from tcip_web.state import OpenProject
from tests._audit_fixtures import audit_rows
from tests._terminal_fixtures import FAKE, fake_executable
from tests._web_fixtures import INVARIANCES, named_project

if TYPE_CHECKING:
    from collections.abc import Iterator

    from httpx import Response
    from starlette.testclient import WebSocketTestSession

LAUNCH = {"provider": pty_host.PROVIDERS[0].id, "user": "tester"}
ABSENT = "definitely-not-a-real-cli-xyz"
_PLACEHOLDERS = (
    pty_host.MCP_CONFIG_ARG,
    pty_host.CLAUDE_SETTINGS_ARG,
    pty_host.CODEX_CONFIG_ARG,
)
INTERPRETER = pty_host.backend_interpreter()
"""The interpreter this backend runs, the one every launch's MCP server and doctor line name."""


def _doctor_entry(interpreter: str, project: Path) -> str:
    """Return Agy's allow entry for the doctor line of ``interpreter`` and ``project``."""
    return pty_host._command_entry(
        pty_host.doctor_command(interpreter, project)
    )


def _tool_names() -> list[str]:
    """Return the sorted name of every tool a server for no project lists."""
    from tcip_web.cli import tool_listing

    return sorted(t.name for t in tool_listing())


if pty_host.os.name == "nt":
    pytest.importorskip("winpty")


@pytest.fixture(autouse=True)
def _fake_terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Launch the fake agent; afterwards end every session and forget the survivors of it."""
    monkeypatch.setenv("TCIP_TERMINAL_CMD", f"{sys.executable} -u {FAKE}")
    yield
    terminal_routes.shutdown_all()
    terminal_routes._SESSIONS.clear()


def _read_until(
    ws: WebSocketTestSession, needle: str, tries: int = 200
) -> str:
    """Accumulate up to ``tries`` WS text frames until ``needle`` appears; return the text.

    Raises AssertionError naming the needle when ``tries`` frames pass without it.
    """
    acc = ""
    for _ in range(tries):
        acc += ws.receive_text()
        if needle in acc:
            return acc
    raise AssertionError(
        f"never saw {needle!r} in terminal stream; got: {acc[-500:]!r}"
    )


def _rows_with(
    monkeypatch: pytest.MonkeyPatch, **changes: str
) -> tuple[pty_host.Provider, ...]:
    """Install and return every provider row with ``changes`` applied as the table.

    The override is cleared so the rows themselves launch.
    """
    monkeypatch.delenv("TCIP_TERMINAL_CMD", raising=False)
    rows = tuple(
        dataclasses.replace(row, **changes) for row in pty_host.PROVIDERS
    )
    monkeypatch.setattr(pty_host, "PROVIDERS", rows)
    return rows


def _registered_hook(argv: list[str]) -> str | None:
    """Return the session hook's command the rendered ``argv`` registers, ``None`` for none.

    It is read from the hooks of the settings file a ``--settings`` names, or else of the ``-c``
    overrides (:func:`_codex_overrides`); more than one handler is asserted against.
    """
    if "--settings" in argv:
        settings = Path(argv[argv.index("--settings") + 1])
        hooks = json.loads(settings.read_text(encoding="utf-8")).get(
            "hooks", {}
        )
    else:
        hooks = _codex_overrides(argv)[1]
    commands = [
        handler["command"]
        for group in hooks.get("SessionStart", [])
        for handler in group["hooks"]
    ]
    assert len(commands) <= 1, commands
    return commands[0] if commands else None


def _ritual_doctor_line(ritual: str) -> str:
    """Return the doctor step ``ritual`` names.

    The step is what follows the ritual's first ``, then``, with its closing friction sentence
    removed.
    """
    return ritual.removesuffix(f". {pty_host._RITUAL_FRICTION}").split(
        ", then ", 1
    )[1]


# ── unit-ish: command resolution + preflight ────────────────────────────


def test_resolve_command_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TCIP_TERMINAL_CMD", "python fake.py")
    assert pty_host.resolve_terminal_command(pty_host.PROVIDERS[0]) == (
        ["python", "fake.py"],
        True,
    )


def test_resolve_command_none_when_the_rows_executable_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (row, *_) = _rows_with(monkeypatch, executable=ABSENT)
    assert pty_host.resolve_terminal_command(row) is None


def test_status_lists_every_row_available_with_fake(
    client: TestClient,
) -> None:
    assert client.get("/api/terminal/status").json() == {
        "providers": [
            {"id": row.id, "name": row.name, "unavailable_reason": None}
            for row in pty_host.PROVIDERS
        ]
    }


def test_a_row_whose_executable_is_absent_reports_its_own_reason_and_refuses_create(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows_with(monkeypatch, executable=ABSENT)
    body = client.get("/api/terminal/status").json()
    assert [entry["unavailable_reason"] for entry in body["providers"]] == [
        row.unavailable_reason for row in rows
    ]
    assert all(
        ABSENT in entry["unavailable_reason"] for entry in body["providers"]
    )

    resp = client.post(
        "/api/terminal/sessions", json={**LAUNCH, "provider": rows[0].id}
    )
    assert resp.status_code == 503
    assert ABSENT in resp.json()["detail"]
    assert terminal_routes._SESSIONS == {}


def test_a_create_naming_an_unlisted_provider_refuses_by_name_and_a_listed_one_launches(
    client: TestClient,
) -> None:
    for unlisted in ("no-such-harness", LAUNCH["provider"].upper()):
        resp = client.post(
            "/api/terminal/sessions", json={**LAUNCH, "provider": unlisted}
        )
        assert resp.status_code == 422
        assert unlisted in resp.json()["detail"]
    assert client.post("/api/terminal/sessions", json={}).status_code == 422
    assert terminal_routes._SESSIONS == {}

    resp = client.post("/api/terminal/sessions", json=LAUNCH)
    assert resp.status_code == 200
    assert resp.json()["launched"]["provider"] == LAUNCH["provider"]


def test_rendering_replaces_each_placeholder_and_leaves_every_other_argument(
    opened: OpenProject,
) -> None:
    literal, config = pty_host.render_argv(
        ["--literal", pty_host.MCP_CONFIG_ARG],
        pty_host.project_launch(opened),
        pty_host.spawn_env("term_test"),
        None,
    )

    assert literal == "--literal"
    server = json.loads(Path(config).read_text(encoding="utf-8"))[
        "mcpServers"
    ]["tcip"]
    assert server["args"] == [
        "-m",
        "tcip_mcp",
        "--project",
        opened.root.as_posix(),
    ]


def test_the_ritual_for_no_project_says_the_session_has_none() -> None:
    ritual = pty_host.session_ritual(pty_host.project_launch(None))
    assert "has no project" in ritual
    assert "initialize_project" in ritual


def test_prewarm_runs_without_raising() -> None:
    pty_host.prewarm()
    pty_host._prewarm_blocking()

    assert "tcip_mcp.server" in sys.modules


def test_the_agent_runs_in_the_workspace(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launched agent's working directory is the backend's workspace, not the repository."""
    from tcip_web.state import store

    cwd_file = tmp_path / "cwd.txt"
    monkeypatch.setenv("FAKE_TERMINAL_CWD_FILE", str(cwd_file))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")

    assert Path(cwd_file.read_text(encoding="utf-8")).samefile(store.workspace)


def test_winpty_terminate_polls_isalive_rather_than_trusting_taskkills_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_WinPty.terminate`` polls ``isalive()`` after taskkill returns.

    A process dead by the first poll ends the wait with no sleep.
    """

    class _StubProc:
        pid = 4242

        def __init__(self) -> None:
            self.calls = 0

        def isalive(self) -> bool:
            self.calls += 1
            return (
                self.calls == 1
            )  # alive at the liveness guard, dead by the first poll

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


def test_create_session_spawns_and_streams_banner(client: TestClient) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")


@pytest.mark.parametrize("row", pty_host.PROVIDERS, ids=lambda row: row.id)
def test_each_rows_rendered_argv_spawns_in_a_real_pty_and_streams(
    row: pty_host.Provider,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opened: OpenProject,
    opened_project: Path,
) -> None:
    """Each provider row's rendered argv spawns in a real PTY and streams.

    The fake program, standing as the row's executable, receives exactly the arguments the
    row renders, with no placeholder left in them, and the launch's record and audit line carry
    that argv and the row's preparation. The settings file Claude's row is handed
    holds the shipped lists plus, under both shells, each read-only console command under both
    invocations and the ritual's doctor line; the shipped file names no console command.
    """
    monkeypatch.setattr(
        pty_host,
        "PROVIDERS",
        (
            dataclasses.replace(
                row, executable=str(fake_executable(tmp_path / "bin"))
            ),
        ),
    )
    monkeypatch.delenv("TCIP_TERMINAL_CMD")
    settings = tmp_path / "agy-settings.json"
    monkeypatch.setattr(pty_host, "ANTIGRAVITY_SETTINGS", settings)
    argv_file = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_TERMINAL_ARGV_FILE", str(argv_file))

    created = client.post(
        "/api/terminal/sessions", json={**LAUNCH, "provider": row.id}
    ).json()
    sid = created["session_id"]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")

    launched = created["launched"]["argv"]
    assert json.loads(argv_file.read_text(encoding="utf-8")) == launched[1:]
    assert not any(
        placeholder in arg for arg in launched for placeholder in _PLACEHOLDERS
    )
    assert _registered_hook(launched) == (
        None
        if row.resume is None
        else pty_host.conversation_hook_command(INTERPRETER)
    )
    assert created["launched"]["resumed"] is None
    if "--settings" in launched:
        shipped = json.loads(
            pty_host.CLAUDE_SETTINGS.read_text(encoding="utf-8")
        )
        handed = json.loads(
            Path(launched[launched.index("--settings") + 1]).read_text(
                encoding="utf-8"
            )
        )
        line = _ritual_doctor_line(created["ritual"])
        commands = [
            f"{invocation} {command}:*"
            for invocation in ("python -m tcip_web.cli", "tcip")
            for command in (
                "scan-dataset",
                "inspect-compute-resources",
                "render-failure-cases",
            )
        ]
        assert handed["permissions"]["deny"] == shipped["permissions"]["deny"]
        assert handed["permissions"]["allow"] == [
            *shipped["permissions"]["allow"],
            *(
                f"{shell}({entry})"
                for shell in ("Bash", "PowerShell")
                for entry in [*commands, line]
            ),
        ]
        assert not any(
            "tcip" in entry
            for entries in shipped["permissions"].values()
            for entry in entries
            if not entry.startswith("mcp__tcip__")
        )
    if row.prepare is None:
        expected_prepared: list[str] = []
    else:
        server = pty_host.mcp_server(pty_host.project_launch(opened))
        expected_prepared = [
            pty_host.subprocess.list2cmdline(
                [
                    launched[0],
                    "mcp",
                    "add",
                    "tcip",
                    "--",
                    server.command,
                    *server.args,
                ]
            ),
            f"allowed {len(_tool_names()) + 2} tcip entries in {settings}, dropped 0 "
            "other doctor entries",
        ]
        allowed = json.loads(settings.read_text(encoding="utf-8"))[
            "permissions"
        ]["allow"]
        assert allowed == [
            *(f"mcp(tcip/{name})" for name in _tool_names()),
            _doctor_entry(INTERPRETER, opened.root),
            f"unsandboxed({pty_host.doctor_command(INTERPRETER, opened.root)})",
        ]
    assert created["launched"]["prepared"] == expected_prepared
    assert created["launched"]["confinement"] == row.confinement
    (started,) = audit_rows(opened_project, "agent_terminal_started")
    assert started["arguments"]["prepared"] == expected_prepared
    assert started["arguments"]["confinement"] == row.confinement
    assert started["arguments"]["argv"] == launched


def test_a_launch_preparation_that_fails_refuses_the_launch_naming_it(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opened_project: Path,
) -> None:
    """A row whose preparation step exits non-zero is refused, naming the failure.

    The launch answers 503, registers no session and records no start line.
    """

    def prepare(
        executable: str, launch: pty_host.ProjectLaunch | None
    ) -> list[str]:
        return [
            pty_host.run_to_completion(
                [executable, "fail", "tcip"], "the fake step"
            )
        ]

    row = dataclasses.replace(
        pty_host.PROVIDERS[0],
        executable=str(fake_executable(tmp_path / "bin")),
        prepare=prepare,
    )
    monkeypatch.setattr(pty_host, "PROVIDERS", (row,))
    monkeypatch.delenv("TCIP_TERMINAL_CMD")

    resp = client.post(
        "/api/terminal/sessions", json={**LAUNCH, "provider": row.id}
    )

    assert resp.status_code == 503
    assert (
        "launch preparation failed: the fake step exited 3"
        in resp.json()["detail"]
    )
    assert "FAKE_PREPARATION_FAILED" in resp.json()["detail"]
    assert terminal_routes._SESSIONS == {}
    assert audit_rows(opened_project, "agent_terminal_started") == []


def test_allowing_the_tcip_tools_keeps_the_rest_of_the_settings_and_is_idempotent(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    project = tmp_path / "valley (north)"
    line = pty_host.doctor_command(INTERPRETER, project)
    other_project = _doctor_entry(INTERPRETER, tmp_path / "other project")
    other_interpreter = _doctor_entry("/old/env/bin/python", project)
    other_line = pty_host.doctor_command(
        INTERPRETER, tmp_path / "other project"
    )
    other_unsandboxed = f"unsandboxed({other_line})"
    settings.write_text(
        json.dumps(
            {
                "model": "kept",
                "permissions": {
                    "allow": [
                        "command(git log)",
                        other_project,
                        other_interpreter,
                        other_unsandboxed,
                        "unsandboxed(git log)",
                    ],
                    "deny": ["x"],
                },
            }
        ),
        encoding="utf-8",
    )

    first = pty_host.allow_tcip_tools(settings, line)
    second = pty_host.allow_tcip_tools(settings, line)

    names = _tool_names()
    assert first == (
        f"allowed {len(names) + 2} tcip entries in {settings}, dropped 3 other "
        "doctor entries"
    )
    assert (
        second
        == f"allowed 0 tcip entries in {settings}, dropped 0 other doctor entries"
    )
    body = json.loads(settings.read_text(encoding="utf-8"))
    assert body["model"] == "kept"
    assert body["permissions"]["deny"] == ["x"]
    assert body["permissions"]["allow"] == [
        "command(git log)",
        "unsandboxed(git log)",
        *(f"mcp(tcip/{name})" for name in names),
        pty_host._command_entry(line),
        f"unsandboxed({line})",
    ]

    # A launch with no project runs no doctor: both entries of the last project go too.
    assert pty_host.allow_tcip_tools(settings, None) == (
        f"allowed 0 tcip entries in {settings}, dropped 2 other doctor entries"
    )
    assert json.loads(settings.read_text(encoding="utf-8"))["permissions"][
        "allow"
    ] == [
        "command(git log)",
        "unsandboxed(git log)",
        *(f"mcp(tcip/{name})" for name in names),
    ]

    settings.write_text("[]", encoding="utf-8")
    with pytest.raises(
        pty_host.PreparationFailedError, match="does not hold a JSON object"
    ):
        pty_host.allow_tcip_tools(settings, line)


def test_the_doctor_line_runs_the_console_command_under_the_mcp_servers_interpreter(
    opened: OpenProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The doctor line names the console command under the MCP server's interpreter.

    The line starts with the launch's MCP server command and ``-m tcip_web.cli doctor``, and
    running ``tcip_web.cli`` as a module reaches the ``main`` the ``tcip`` console script names.
    """
    import runpy

    import tomllib

    import tcip_web.cli

    launch = pty_host.project_launch(opened)
    line = pty_host.doctor_command(INTERPRETER, opened.root)
    assert line.startswith(
        f"{pty_host.mcp_server(launch).command} -m tcip_web.cli doctor "
    )
    pyproject = (
        Path(__file__).parents[1] / "packages" / "tcip-web" / "pyproject.toml"
    )
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert scripts["tcip"] == "tcip_web.cli:main"

    reached: list[bool] = []

    def main() -> int:
        reached.append(True)
        return 0

    monkeypatch.setattr(tcip_web.cli, "main", main)
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("tcip_web.cli", run_name="__main__")
    assert reached == [True] and exited.value.code == 0


def test_the_doctor_allowance_is_the_rituals_own_line_and_nothing_else(
    opened: OpenProject,
) -> None:
    """The doctor allowance admits the ritual's own doctor line and nothing else.

    One renderer spells the ritual's doctor step, and agy's entry admits exactly that line
    (its regex evaluated under Python's ``re``): not the line with anything after it, not
    another project's, not an unquoted spelling. The stale-entry recognizer matches what the
    renderer writes and nothing it refuses.
    """
    opened_project = opened.root
    line = pty_host.doctor_command(INTERPRETER, opened_project)
    head = f"{INTERPRETER} -m tcip_web.cli doctor"
    assert line == f'{head} "{opened_project.as_posix()}"'
    launch = pty_host.project_launch(opened)
    assert launch == pty_host.ProjectLaunch(
        opened, line, None, pty_host.conversation_hook_command(INTERPRETER)
    )
    assert _ritual_doctor_line(pty_host.session_ritual(launch)) == line

    entry = pty_host._command_entry(line)
    assert entry.startswith("command(regex:^") and entry.endswith("$)")
    pattern = re.compile(entry[len("command(regex:") : -1])
    assert pattern.fullmatch(line)
    for tail in (
        "; Write-Output x",
        " | Write-Output x",
        " && dir",
        " > out.txt",
        " `dir`",
        " $(dir)",
        "\nWrite-Output x",
        " C:/second",
        " --help",
        '"',
        ' "C:/second"',
    ):
        assert not pattern.fullmatch(f"{line}{tail}"), tail
    assert not pattern.fullmatch(line[:-1])
    assert not pattern.fullmatch(
        f'{head} "{opened_project.parent.as_posix()}"'
    )
    assert not pattern.fullmatch(f"{head} {opened_project.as_posix()}")
    assert not pattern.fullmatch(f'tcip doctor "{opened_project.as_posix()}"')
    for position in range(len(line)):
        changed = (
            line[:position]
            + ("y" if line[position] != "y" else "z")
            + line[position + 1 :]
        )
        assert not pattern.fullmatch(changed), changed

    spaced = _doctor_entry(
        INTERPRETER, Path("S:/Savanna Institute (north)/o'neil & sons, b.1")
    )
    assert re.compile(spaced[len("command(regex:") : -1]).fullmatch(
        f'{head} "S:/Savanna Institute (north)/o\'neil & sons, b.1"'
    )
    assert not re.compile(spaced[len("command(regex:") : -1]).fullmatch(
        f'{head} "S:/Savanna Institute (north)/o\'neil & sons, bX1"'
    )
    assert pty_host._DOCTOR_ENTRY.fullmatch(spaced)
    assert not pty_host._DOCTOR_ENTRY.fullmatch("command(regex:^git log$)")
    for interpreter in (
        "/two words/python",
        "/em space/python",
        "",
        "/conda(alt)/python",
        "/a&b/python",
    ):
        with pytest.raises(ValueError, match="the interpreter"):
            pty_host.doctor_command(interpreter, opened_project)
        refused = pty_host._command_entry(
            f'{interpreter} -m tcip_web.cli doctor "/p"'
        )
        assert not pty_host._DOCTOR_ENTRY.fullmatch(refused), refused

    curly = _doctor_entry(
        INTERPRETER, Path("S:/o\u2019neil \u2018north\u2019 block")
    )
    assert re.compile(curly[len("command(regex:") : -1]).fullmatch(
        f'{head} "S:/o\u2019neil \u2018north\u2019 block"'
    )


@pytest.mark.parametrize(
    "interpreter", [INTERPRETER, "/env bin/python"], ids=["spelled", "refused"]
)
def test_the_ritual_agys_entry_and_claudes_settings_name_one_doctor_line(
    interpreter: str,
    opened: OpenProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ritual, agy's entry and Claude's settings name one doctor line.

    The ritual's doctor step, the command agy's regex entry admits and the command Claude's
    added doctor entries name are one string, read from the ritual text, agy's written settings
    and Claude's handed settings over the launch's one :func:`project_launch`; when the step
    cannot be spelled, the ritual says so naming why, agy's preparation fails naming it, and
    Claude's settings add no doctor entry.
    """
    monkeypatch.setattr(pty_host, "backend_interpreter", lambda: interpreter)
    launch = pty_host.project_launch(opened)
    ritual = pty_host.session_ritual(launch)
    settings = tmp_path / "agy-settings.json"
    monkeypatch.setattr(pty_host, "ANTIGRAVITY_SETTINGS", settings)
    monkeypatch.setattr(
        pty_host, "run_to_completion", lambda argv, step: "added"
    )
    (claude_settings,) = pty_host.render_argv(
        [pty_host.CLAUDE_SETTINGS_ARG],
        launch,
        pty_host.spawn_env("term_test"),
        None,
    )
    handed = json.loads(Path(claude_settings).read_text(encoding="utf-8"))
    doctor_entries = [
        entry
        for entry in handed["permissions"]["allow"]
        if " doctor " in entry
    ]

    if interpreter != INTERPRETER:
        assert "cannot be spelled on a shell line: the interpreter" in ritual
        with pytest.raises(
            pty_host.PreparationFailedError, match="the interpreter"
        ):
            pty_host.prepare_antigravity("agy", launch)
        assert doctor_entries == []
        return
    line = _ritual_doctor_line(ritual)
    pty_host.prepare_antigravity("agy", launch)
    allowed = json.loads(settings.read_text(encoding="utf-8"))["permissions"][
        "allow"
    ]
    (agy_entry,) = [e for e in allowed if e.startswith("command(regex:")]
    assert re.compile(agy_entry[len("command(regex:") : -1]).fullmatch(line)
    assert f"unsandboxed({line})" in allowed
    assert doctor_entries == [f"Bash({line})", f"PowerShell({line})"]


SHELL_READS_INSIDE_DOUBLE_QUOTES = {
    '"': "closes the double-quoted segment under Bash and PowerShell",
    "$": "Bash and PowerShell expand a variable",
    "`": "Bash substitutes a command and PowerShell escapes the next character",
    "\\": "Bash reads an escape before some of the characters that may follow it",
    "!": "Bash expands history in an interactive shell with history expansion on",
    "\r": "breaks the ritual's one line",
    "\n": "breaks the ritual's one line",
    "“": "PowerShell reads a left double quotation mark as a double quote",
    "”": "PowerShell reads a right double quotation mark as a double quote",
    "„": "PowerShell reads a double low-9 quotation mark as a double quote",
}
"""Each character the renderer refuses in a project path, by what Bash or PowerShell does with
it inside double quotes or what the ritual's one-line contract excludes, written here from the
shells' grammars rather than read off the renderer's set, so a character dropped from that set
fails its case. A typographic single quote (U+2018 to U+201B) is literal inside double quotes
under both shells and is not here."""


SHELL_READS_IN_A_BARE_WORD = {
    **SHELL_READS_INSIDE_DOUBLE_QUOTES,
    " ": "ends a word under Bash and PowerShell",
    "\t": "ends a word under Bash and PowerShell",
    "\v": "ends a word under PowerShell",
    "\f": "ends a word under PowerShell",
    " ": "PowerShell reads a no-break space as whitespace",
    "\u0085": "PowerShell reads a next-line mark as whitespace",
    **{
        c: "PowerShell reads a Unicode space separator as whitespace"
        for c in "              　"
    },
    " ": "PowerShell reads a line separator as whitespace",
    " ": "PowerShell reads a paragraph separator as whitespace",
    "|": "a pipe under both",
    "&": "a background or call operator under both",
    ";": "ends a command under both",
    "(": "opens a subshell or subexpression under both",
    ")": "closes a subshell or subexpression under both",
    "<": "a redirection under both",
    ">": "a redirection under both",
    "'": "opens a quoted segment under both",
    "*": "a Bash glob",
    "?": "a Bash glob",
    "[": "a Bash glob and a PowerShell type literal",
    "]": "closes a Bash glob",
    "{": "Bash brace expansion and a PowerShell script block",
    "}": "closes a brace expansion or a script block",
    "~": "Bash tilde expansion",
    "#": "opens a comment under both",
    "@": "PowerShell splatting and arrays",
    ",": "PowerShell's array operator",
    "‘": "PowerShell reads a left single quotation mark as a quote",
    "’": "PowerShell reads a right single quotation mark as a quote",
    "‚": "PowerShell reads a single low-9 quotation mark as a quote",
    "‛": "PowerShell reads a single high-reversed-9 quotation mark as a quote",
}
"""Each character the renderer refuses in the unquoted interpreter, by what Bash or PowerShell
does with it outside quotes, written from the shells' grammars rather than read off the
renderer's set."""

_REFUSED = [
    ("project path", c) for c in sorted(SHELL_READS_INSIDE_DOUBLE_QUOTES)
] + [("interpreter", c) for c in sorted(SHELL_READS_IN_A_BARE_WORD)]
"""Each argument of the doctor line with a character it refuses."""


@pytest.mark.parametrize(
    "where, held",
    _REFUSED,
    ids=[f"{where.split()[-1]}-U+{ord(c):04X}" for where, c in _REFUSED],
)
def test_a_doctor_line_argument_holding_a_refused_character_is_refused_by_name(
    where: str,
    held: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A doctor line argument holding a refused character is refused by name.

    The renderer refuses a project path or an interpreter holding a character the doctor
    line refuses, naming that argument and that character; the launch's doctor step is that
    refusal, the ritual then says the step cannot be spelled, on one line, and the Antigravity
    preparation fails naming it and writes no settings. The paths are POSIX ones, so a
    backslash is a path character and not a separator.
    """
    from pathlib import PurePosixPath

    project = PurePosixPath(
        f"/projects/valley{held if where == 'project path' else ''}block"
    )
    interpreter = f"/env/bin{held if where == 'interpreter' else ''}/python"
    with pytest.raises(ValueError, match="the doctor line refuses") as refused:
        pty_host.doctor_command(interpreter, project)
    assert str(refused.value).startswith(f"the {where} ")
    assert repr(held) in str(refused.value) or held in str(refused.value)

    # Resolving the path on Windows would read the POSIX backslash as a separator.
    monkeypatch.setattr(
        "tcip_mcp.project_record.existing_project",
        lambda _p: (
            project,
            {
                "id": "abc",
                "display_name": "Valley",
                "site": "s",
                "invariances": INVARIANCES.model_dump(),
            },
        ),
    )
    monkeypatch.setattr(pty_host, "backend_interpreter", lambda: interpreter)
    opened = OpenProject(project, "abc")  # type: ignore[arg-type]
    launch = pty_host.project_launch(opened)
    assert launch == pty_host.ProjectLaunch(
        opened,
        None,
        str(refused.value),
        None
        if where == "interpreter"
        else pty_host.conversation_hook_command(interpreter),
    )
    ritual = pty_host.session_ritual(launch)
    assert (
        "cannot be spelled on a shell line" in ritual
        and "report_friction" in ritual
    )
    assert f'doctor "{project.as_posix()}"' not in ritual
    assert "\n" not in ritual and "\r" not in ritual

    settings = tmp_path / "settings.json"
    monkeypatch.setattr(pty_host, "ANTIGRAVITY_SETTINGS", settings)
    monkeypatch.setattr(
        pty_host, "run_to_completion", lambda argv, step: "added"
    )
    with pytest.raises(
        pty_host.PreparationFailedError, match="the doctor line refuses"
    ):
        pty_host.prepare_antigravity("agy", launch)
    assert not settings.exists()


def test_the_written_mcp_configuration_is_the_one_server_the_launch_states(
    opened: OpenProject,
) -> None:
    launch = pty_host.project_launch(opened)
    server = pty_host.mcp_server(launch)
    assert server.args == (
        "-m",
        "tcip_mcp",
        "--project",
        opened.root.as_posix(),
    )
    assert pty_host.mcp_server(None).args == ("-m", "tcip_mcp")

    (config,) = pty_host.render_argv(
        [pty_host.MCP_CONFIG_ARG],
        launch,
        pty_host.spawn_env("term_test"),
        None,
    )

    written = json.loads(Path(config).read_text(encoding="utf-8"))[
        "mcpServers"
    ]["tcip"]
    assert (written["command"], written["args"]) == (
        server.command,
        list(server.args),
    )


def _codex_overrides(argv: list[str]) -> tuple[dict, dict]:
    """Return the ``tcip`` server and the hooks the ``-c`` overrides in ``argv`` state.

    Each override, the argument after a ``-c``, is read by TOML.
    """
    import tomllib

    server: dict = {}
    hooks: dict = {}
    for flag, override in pairwise(argv):
        if flag == "-c":
            parsed = tomllib.loads(override)
            server.update(parsed.get("mcp_servers", {}).get("tcip", {}))
            hooks.update(parsed.get("hooks", {}))
    return server, hooks


def test_the_codex_overrides_name_the_one_server_and_hook_and_forward_the_spawn_environment(
    opened: OpenProject,
) -> None:
    """The Codex overrides name the one server and session hook and forward the environment.

    Each rendered ``-c`` override parses as TOML, and together they state the server
    :func:`mcp_server` produces, every variable of the launch's environment forwarded to it, and
    the launch's session hook as :func:`session_start_hooks`; a launch for no project has none.
    """
    env = pty_host.spawn_env("term_test")
    launch = pty_host.project_launch(opened)
    rendered = pty_host.render_argv(
        ["--literal", pty_host.CODEX_CONFIG_ARG], launch, env, None
    )

    assert rendered[0] == "--literal"
    assert set(rendered[1::2]) == {"-c"}
    parsed, hooks = _codex_overrides(rendered[1:])
    server = pty_host.mcp_server(launch)
    assert (parsed["command"], parsed["args"]) == (
        server.command,
        list(server.args),
    )
    assert set(parsed["env_vars"]) == set(env)
    assert pty_host.TERMINAL_SESSION_ENV in parsed["env_vars"]
    assert parsed["default_tools_approval_mode"] == "approve"
    assert hooks == pty_host.session_start_hooks(
        pty_host.conversation_hook_command(INTERPRETER)
    )
    assert _codex_overrides(pty_host.codex_overrides(None, env))[1] == {}


def test_the_codex_overrides_carry_any_project_path_as_toml_reads_it_back() -> (
    None
):
    """A project path holding what a TOML basic string must escape is read back exactly.

    The path, a POSIX one so its backslash is a character, holds a supplementary character, a
    quotation mark, a backslash, a tab and a delete, read back by ``tomllib``.
    """
    from pathlib import PurePosixPath

    root = PurePosixPath("/projects") / 'grove \U0001f333 "north"\\\tb\x7f'
    launch = pty_host.ProjectLaunch(
        OpenProject(root, "abc"),  # type: ignore[arg-type]
        None,
        "not spelled",
        None,
    )

    parsed, _ = _codex_overrides(
        pty_host.codex_overrides(launch, pty_host.spawn_env("term_test"))
    )

    assert parsed["args"] == ["-m", "tcip_mcp", "--project", root.as_posix()]


_RITUAL_ECHO = "echo:[TCIP session-start ritual]"


def _wait_for(
    session: terminal_routes.TerminalSession, needle: str, timeout: float = 30
) -> str:
    """Return the session's scrollback when it holds ``needle``, polling up to ``timeout`` s.

    Raises AssertionError naming the needle when the timeout passes without it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        text = session.scrollback_snapshot()
        if needle in text:
            return text
        time.sleep(0.1)
    raise AssertionError(
        f"never saw {needle!r}; got: {session.scrollback_snapshot()[-500:]!r}"
    )


def _echoes(scrollback: str) -> list[str]:
    """Return every line the fake agent echoed, in order, the ritual's cut to its header."""
    return [
        _RITUAL_ECHO if line.startswith(_RITUAL_ECHO) else line
        for line in re.findall(r"echo:[^\r\n]*", scrollback)
    ]


def _launched_session(
    monkeypatch: pytest.MonkeyPatch, row: pty_host.Provider
) -> tuple[terminal_routes.TerminalSession, _StubPty]:
    """Start a session of ``row`` through :meth:`TerminalSession.start` on a stub that stays up.

    No reader runs, so the test feeds the session's output itself. Returns the session and its
    PTY.
    """
    pty = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, pty)
    session = terminal_routes.TerminalSession("term_test")
    assert session.start(24, 80, row, "user:tester") is None
    return session, pty


@pytest.fixture
def pastes_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fake agent turn bracketed paste on one second after it starts.

    The fake then writes the first provider row's composer sequence.
    """
    monkeypatch.setenv("FAKE_TERMINAL_PASTE_AFTER_S", "1")
    monkeypatch.setenv(
        "FAKE_TERMINAL_COMPOSER_READY", pty_host.PROVIDERS[0].composer_ready
    )


def test_the_ritual_then_a_submitted_request_reach_the_agent_once_it_turns_paste_on(
    client: TestClient,
    pastes_after: None,
) -> None:
    sid = client.post(
        "/api/terminal/sessions", json={**LAUNCH, "cols": 500}
    ).json()["session_id"]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "first request"}
    )

    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:first request")
    assert _echoes(text) == [_RITUAL_ECHO, "echo:first request"]
    assert "EARLY_INPUT" not in text

    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "second request"}
    )
    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:second request")
    assert _echoes(text) == [
        _RITUAL_ECHO,
        "echo:first request",
        "echo:second request",
    ]


def test_the_ritual_and_the_requests_staged_before_paste_is_on_go_as_one_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ritual and the requests staged before paste is on go as one paste.

    The ritual and every request queued by the time the agent turns bracketed paste on are
    written as one paste; a request submitted afterwards is its own paste.
    """
    row = pty_host.PROVIDERS[0]
    session, pty = _launched_session(monkeypatch, row)
    session.submit("first")
    session.submit("second")
    assert pty.writes == []

    session._on_output("\x1b[?2004h" + row.composer_ready)
    assert pty.writes == [
        pty_host.paste(f"{session.ritual}\n\nfirst\n\nsecond")
    ]

    session.submit("later")
    session.deliver()
    assert pty.writes[1:] == [pty_host.paste("later")]
    assert pty_host.paste("a\nb\r\nc") == "\x1b[200~a\rb\rc\x1b[201~\r"


def test_overlapping_creates_deliver_one_ritual_before_the_request(
    client: TestClient, pastes_after: None
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    def create(_: int) -> dict:
        return (
            TestClient(app, base_url="http://127.0.0.1")
            .post("/api/terminal/sessions", json={**LAUNCH, "cols": 500})
            .json()
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(create, range(2)))
    sid = answers[0]["session_id"]
    assert answers[1]["session_id"] == sid
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "tab request"}
    )

    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:tab request")
    assert _echoes(text) == [_RITUAL_ECHO, "echo:tab request"]


_COMPOSER_LIVE_AT = {"claude": 15, "antigravity": 4, "codex": 10}
"""For each recorded harness, the index of the startup read at which its composer was observed
live, from the PTY logs and never from a row's marker: Claude Code drew its input prompt in read 15,
and a paste at read 5 was lost; agy 1.3.1 cleared its screen and drew its banner and prompt in
read 4, right after turning bracketed paste on in read 3 (its 1.3.0 predecessor lost a paste
written at that point, before its prompt); Codex wrote its first window title in read 10, its
session footer followed, and an Enter before it was held as "Waiting for startup" while a paste
at it was submitted."""

_STARTUP_READS = {
    "claude": [
        "\x1b[1t",
        "\x1b[c\x1b[?1004h\x1b[?9001h",
        "\x1b]0;claude\x1b\\",
        "\x1b7\x1b[r\x1b8\x1b[?25h",
        "\x1b[?25l",
        "\x1b[?2004h\x1b[?2031h\x1b[?1004h",
        "\x1b[?9001l\x1b[?9001h",
        "\x1b[>0q",
        "\x1b[?u",
        "\x1b[c",
        "\x1b[>4m\x1b[<u\x1b[?1004l\x1b[?1004h\x1b[?2031l\x1b[?2004l",
        "\x1b[1;1H",
        "\r\n\x1b[1;1H",
        "\x1b[?2004h\x1b[?2031h\x1b[?1004h\x1b[?9001l\x1b[?9001h",
        "\x1b]0;✳ Claude Code\x1b\\",
        "\x1b[?1049h\x1b[2J\x1b[H\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h\x1b[?25l",
    ],
    "antigravity": [
        "\x1b[1t",
        "\x1b[c\x1b[?1004h\x1b[?9001h",
        "\x1b[?2026$p\x1b[?2027$p\x1b]11;?\x07",
        "\x1b[?5W\x1b[?2004h\x1b[>4;2m\x1b[>1u\x1b[?u",
        "\x1b[0 q\r\x1b[J\n\x1b[6C\x1b[38;5;179m▄\x1b[m        "
        "\x1b[38;5;111;1mAntigravity CLI 1.3.1\x1b[m\n\x1b[38;5;111m>\x1b[m\r\n"
        "\x1b[38;5;109m? for shortcuts\x1b[m\x1b[?25h",
    ],
    "codex": [
        "\x1b[1t",
        "\x1b[c\x1b[?1004h\x1b[?9001h",
        "\x1b[?2004h",
        "\x1b[?1004l\x1b[?1004h",
        "\x1b[?2026h\x1b[?25l",
        "\x1b[?1049h",
        "\x1b[?1007l",
        "\x1b[?1003;1006h",
        "\x1b[2;3H\x1b[38;5;75;49m>_ ",
        "\x1b[?2026l",
        "\x1b]0;tcip-agent\x07",
        "\x1b]10;?\x1b\\\x1b]11;?\x1b\\",
    ],
}
"""What each installed harness wrote while starting in the terminal's PTY (Claude Code 2.1.292
under ``tools/smoke_terminal_e2e.py``, agy 1.3.1 as one scrollback snapshot cut at its own control
sequences, Codex 0.160.1 launched with its row's arguments and sent no input), long screen draws
cut after their opening sequences and spinner frames thinned. Claude Code dropped a paste written
at its first bracketed-paste-on, as agy 1.3.0 did; Codex, under the smoke, held one in its
composer without submitting it, since its session had not started."""


@pytest.mark.parametrize("row", pty_host.PROVIDERS, ids=lambda row: row.id)
def test_the_ritual_is_pasted_at_the_read_where_the_harness_composer_goes_live(
    row: pty_host.Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ritual is pasted at the read where the harness composer goes live.

    Fed the reads its harness wrote at startup, a launch of ``row`` pastes nothing before the
    read at which that harness's composer was observed live and pastes the ritual at that read,
    even though bracketed paste turned on earlier.
    """
    reads = _STARTUP_READS[row.id]
    live_at = _COMPOSER_LIVE_AT[row.id]
    first_paste_on = next(
        i for i, data in enumerate(reads) if "\x1b[?2004h" in data
    )
    assert first_paste_on < live_at

    session, pty = _launched_session(monkeypatch, row)
    session.submit("the request")
    for index, data in enumerate(reads):
        session._on_output(data)
        assert pty.writes == (
            [pty_host.paste(f"{session.ritual}\n\nthe request")]
            if index >= live_at
            else []
        )


def test_bracketed_paste_follows_the_last_mode_change_the_agent_wrote() -> (
    None
):
    assert pty_host.bracketed_paste("\x1b[?2004h", False) is True
    assert (
        pty_host.bracketed_paste("\x1b[?1004;2004h\x1b[?2004l", True) is False
    )
    assert pty_host.bracketed_paste("\x1b[?1049h plain output", True) is True


def test_input_round_trip(client: TestClient) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "hello agent\r"})
        _read_until(ws, "echo:hello agent")


def test_attach_semantics_second_create_returns_live_session(
    client: TestClient,
) -> None:
    first = client.post("/api/terminal/sessions", json=LAUNCH).json()
    second = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert second == first


def test_scrollback_replays_on_reconnect(client: TestClient) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "before reconnect\r"})
        _read_until(ws, "echo:before reconnect")
    # New socket: the banner and the echoed line replay from scrollback.
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws2:
        acc = _read_until(ws2, "echo:before reconnect")
        assert "FAKE_TERMINAL_READY" in acc


def test_resize_does_not_crash_stream(client: TestClient) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json(
            {"type": "resize", "rows": 99999, "cols": -3}
        )  # clamped, not fatal
        ws.send_json({"type": "input", "data": "after resize\r"})
        _read_until(ws, "echo:after resize")


def test_a_resize_that_raises_does_not_end_the_stream(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resize that raises does not end the terminal stream.

    A ``ValueError`` from the PTY's resize is swallowed by the session's, and input sent after
    it still echoes.
    """
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]

    def _raise(rows: int, cols: int) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(session._pty, "resize", _raise)
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "input", "data": "after raising resize\r"})
        _read_until(ws, "echo:after raising resize")


def _read_until_closed(
    ws: WebSocketTestSession, tries: int = 400
) -> tuple[str, int]:
    """Accumulate up to ``tries`` WS text frames until the backend closes the socket.

    Returns the text and the close code. Raises AssertionError when ``tries`` frames pass open.
    """
    acc = ""
    for _ in range(tries):
        try:
            acc += ws.receive_text()
        except WebSocketDisconnect as closed:
            return acc, closed.code
    raise AssertionError(f"the socket stayed open; got: {acc[-500:]!r}")


def _end(client: TestClient, sid: str) -> Response:
    """Post the person "tester"'s end of session ``sid`` and return the answer."""
    return client.post(
        f"/api/terminal/sessions/{sid}/end", json={"user": "tester"}
    )


def _ended(line: dict) -> terminal_routes.TerminalEnded:
    """Return the end ``line``'s arguments as the record the platform reads them as.

    Its last output is asserted to come no later than the line itself.
    """
    from datetime import datetime

    ended = terminal_routes.TerminalEnded.model_validate(line["arguments"])
    if ended.last_output_at is not None:
        assert ended.last_output_at <= datetime.fromisoformat(
            line["timestamp"]
        )
    return ended


def _ended_without_its_clock(line: dict) -> dict:
    """Return the end ``line``'s arguments, read by :func:`_ended`, less its last output's time."""
    _ended(line)
    return {
        k: v for k, v in line["arguments"].items() if k != "last_output_at"
    }


def test_a_process_exiting_on_its_own_ends_its_session_by_no_one(
    client: TestClient, opened_project: Path
) -> None:
    """A process that exits on its own ends its session, recorded with no actor.

    Its sockets are closed with the ended code after its last output, and the session is
    forgotten.
    """
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        text, code = _read_until_closed(ws)

    assert "FAKE_TERMINAL_BYE" in text
    assert code == terminal_routes.ENDED_CLOSE_CODE
    assert sid not in terminal_routes._SESSIONS
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert _ended(line).last_output_at is not None
    assert _ended_without_its_clock(line) == {
        "session_id": sid,
        "provider": LAUNCH["provider"],
        "cause": "the agent's process exited",
        "undelivered": [],
        "resumable": True,
        "conversation": None,
        "handed_off": None,
    }
    assert "actor" not in line


def test_a_harness_exiting_at_its_start_is_recorded_started_then_ended(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A harness that exits at once has its start line before its end line in the log.

    A socket opened to its session afterwards is closed with the ended code.
    """
    monkeypatch.setenv("TCIP_TERMINAL_CMD", f"{sys.executable} -u {FAKE} fail")
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    deadline = time.time() + 30
    while (
        not audit_rows(opened_project, "agent_terminal_ended")
        and time.time() < deadline
    ):
        time.sleep(0.1)

    events = [
        (row["tool"], row["arguments"]["session_id"])
        for row in audit_rows(opened_project)
        if row["tool"].startswith("agent_terminal_")
    ]
    assert events == [
        ("agent_terminal_started", sid),
        ("agent_terminal_ended", sid),
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        assert _read_until_closed(ws)[1] == terminal_routes.ENDED_CLOSE_CODE


def test_an_exit_ends_the_agent_by_the_person_and_a_create_after_it_is_a_fresh_launch(
    client: TestClient, opened_project: Path
) -> None:
    """An end by the person stops the process, forgets the session and records the end.

    The next create is a fresh session and launch, with its own start line.
    """
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]

    ended = _end(client, sid)
    # The process's own exit, seen by its reader after the end, ends nothing a second time.
    for reader in threading.enumerate():
        if reader.name == f"term-{sid}":
            reader.join(timeout=30)

    assert ended.status_code == 200, ended.text
    assert not session.alive()
    assert terminal_routes._SESSIONS == {}
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert line["actor"] == "user:tester"
    assert _ended_without_its_clock(line) == {
        "session_id": sid,
        "provider": LAUNCH["provider"],
        "cause": "the person ended it",
        "undelivered": [],
        "resumable": False,
        "conversation": None,
        "handed_off": None,
    }
    assert _end(client, sid).status_code == 404

    fresh = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert fresh["session_id"] != sid
    starts = audit_rows(opened_project, "agent_terminal_started")
    assert [row["arguments"]["session_id"] for row in starts] == [
        sid,
        fresh["session_id"],
    ]


def test_an_end_answers_the_requests_the_agent_was_never_given(
    client: TestClient,
) -> None:
    """A request submitted before the harness took input is answered by the end."""
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "never given"}
    )

    ended = _end(client, sid)

    assert ended.status_code == 200, ended.text
    assert ended.json() == {"undelivered": ["never given"]}


def test_the_end_line_lands_in_the_launchs_project_after_the_gui_opened_another(
    client: TestClient, opened_project: Path, tmp_path: Path
) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    other = named_project(tmp_path.parent / "hill_block", "Hill block")
    opened = client.post(
        "/api/projects/open", json={"id": other.id, "user": "grower"}
    )
    assert opened.status_code == 200, opened.text

    assert _end(client, sid).status_code == 200

    assert len(audit_rows(opened_project, "agent_terminal_ended")) == 1
    assert audit_rows(other.root, "agent_terminal_ended") == []


def test_a_socket_to_a_session_the_backend_does_not_hold_is_closed_as_ended(
    client: TestClient,
) -> None:
    with client.websocket_connect(
        "ws://127.0.0.1/api/terminal/ws/nonexistent"
    ) as ws:
        assert _read_until_closed(ws) == ("", terminal_routes.ENDED_CLOSE_CODE)


def test_ws_rejects_cross_site_origin(client: TestClient) -> None:
    """A live session's socket refuses a foreign origin."""
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"ws://127.0.0.1/api/terminal/ws/{sid}",
            headers={"origin": "https://evil.example"},
        ):
            pass


# ── the terminate and end survivor branches (no real process needed) ──


class _StubPty(pty_host._WinPty):
    """A process that, on ``terminate``, survives or stops as ``survives`` says, counting calls.

    Its ``write`` appends to ``writes``.
    """

    def __init__(self, *, survives: bool) -> None:
        self._alive = True
        self._survives = survives
        self.terminate_calls = 0
        self.writes: list[str] = []

    def isalive(self) -> bool:
        return self._alive

    def write(self, data: str) -> None:
        self.writes.append(data)

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


def test_shutdown_all_logs_a_survivor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
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


def _wire_stub_spawn(monkeypatch: pytest.MonkeyPatch, stub: _StubPty) -> None:
    from tcip_web import terminal as pty_host

    monkeypatch.setattr(pty_host, "spawn_pty", lambda *a, **kw: stub)
    monkeypatch.setattr(pty_host, "start_reader", lambda *a, **kw: None)


def _lenient_client() -> TestClient:
    """Return a client that answers a server exception as a 500 response."""
    return TestClient(
        app, base_url="http://127.0.0.1", raise_server_exceptions=False
    )


def _refuse_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every session's audit line fail to append."""
    from tcip_mcp.audit import AuditEntryNotWrittenError

    def _refuse(
        event: str, arguments: dict, project: object, actor: str | None
    ) -> None:
        raise AuditEntryNotWrittenError(
            event, RuntimeError("audit log unwritable")
        )

    monkeypatch.setattr(terminal_routes, "_record", _refuse)


def test_create_session_registers_a_survivor_and_answers_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session whose start line fails and whose process survives is registered and answers 503.

    The start's audit line fails after the process spawned; the spawned process survives its
    own termination attempt, so the session stays registered and alive and the answer says it
    stays attached.
    """
    stub = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, stub)
    _refuse_record(monkeypatch)

    resp = _lenient_client().post("/api/terminal/sessions", json=LAUNCH)
    assert resp.status_code == 503
    assert "stays attached" in resp.json()["detail"]
    assert len(terminal_routes._SESSIONS) == 1
    (session,) = terminal_routes._SESSIONS.values()
    assert session.alive() is True


def test_an_end_on_a_survivor_answers_503_and_keeps_the_session_reachable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """An end whose process survives termination answers 503 and records nothing.

    The session stays registered and alive, so a create attaches to it.
    """
    survivor = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, survivor)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]

    resp = _end(client, sid)

    assert resp.status_code == 503
    assert "stays attached" in resp.json()["detail"]
    assert terminal_routes._SESSIONS[sid].alive() is True
    assert audit_rows(opened_project, "agent_terminal_ended") == []
    attached = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert attached["session_id"] == sid


class _HeldStubPty(_StubPty):
    """A survivor of ``terminate``, which sets ``entered`` and then waits for ``release``."""

    def __init__(self) -> None:
        super().__init__(survives=True)
        self.entered = threading.Event()
        self.release = threading.Event()

    def terminate(self) -> None:
        self.entered.set()
        assert self.release.wait(timeout=30)
        super().terminate()


def _post(url: str, body: dict) -> Response:
    """Post ``body`` to ``url`` from a client of its own, for a call made on another thread."""
    return TestClient(app, base_url="http://127.0.0.1").post(url, json=body)


class _WatchedLock:
    """The sessions lock, setting ``contended`` once a caller finds it held and waits for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.contended = threading.Event()

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def _watch_sessions_lock(monkeypatch: pytest.MonkeyPatch) -> _WatchedLock:
    """Install and return a watched sessions lock."""
    lock = _WatchedLock()
    monkeypatch.setattr(terminal_routes, "_SESSIONS_LOCK", lock)
    return lock


def test_a_create_during_an_end_waits_for_it_and_attaches_to_the_survivor(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create made while an end terminates waits for the end, then attaches to its survivor.

    The create is seen waiting on the sessions lock the end holds, never finished beside it.
    """
    lock = _watch_sessions_lock(monkeypatch)
    held = _HeldStubPty()
    _wire_stub_spawn(monkeypatch, held)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        ending = pool.submit(
            _post, f"/api/terminal/sessions/{sid}/end", {"user": "tester"}
        )
        assert held.entered.wait(timeout=30)
        creating = pool.submit(_post, "/api/terminal/sessions", LAUNCH)
        creating.add_done_callback(lambda _: lock.contended.set())
        assert lock.contended.wait(timeout=30)
        assert not creating.done()
        held.release.set()
        assert ending.result().status_code == 503
        assert creating.result().json()["session_id"] == sid
    assert list(terminal_routes._SESSIONS) == [sid]


def test_a_submit_admitted_before_an_end_is_answered_by_that_end(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An end waits for a submit already admitted, and answers its request as undelivered.

    The end is seen waiting on the sessions lock the admission holds. A submit after the end
    answers as an unknown session does.
    """
    lock = _watch_sessions_lock(monkeypatch)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    admitted, release = threading.Event(), threading.Event()
    submit = session.submit

    def held_submit(text: str) -> None:
        admitted.set()
        assert release.wait(timeout=30)
        submit(text)

    monkeypatch.setattr(session, "submit", held_submit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        submitting = pool.submit(
            _post, f"/api/terminal/sessions/{sid}/submit", {"text": "taken"}
        )
        assert admitted.wait(timeout=30)
        ending = pool.submit(
            _post, f"/api/terminal/sessions/{sid}/end", {"user": "tester"}
        )
        ending.add_done_callback(lambda _: lock.contended.set())
        assert lock.contended.wait(timeout=30)
        assert not ending.done()
        release.set()
        assert submitting.result().status_code == 200
        assert ending.result().json() == {"undelivered": ["taken"]}

    late = client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "late"}
    )
    assert late.status_code == 404


class _HeldWritePty(_StubPty):
    """A process whose first write sets ``writing`` and waits for ``release``.

    It stops on ``terminate``.
    """

    def __init__(self) -> None:
        super().__init__(survives=False)
        self.writing = threading.Event()
        self.release = threading.Event()

    def write(self, data: str) -> None:
        if not self.writing.is_set():
            self.writing.set()
            assert self.release.wait(timeout=30)


def test_a_submit_is_admitted_while_a_write_is_held(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a delivery's write is held, a submit is admitted.

    The admission is seen taking the session lock without waiting on the write.
    """
    held = _HeldWritePty()
    _wire_stub_spawn(monkeypatch, held)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    lock = _WatchedLock()
    monkeypatch.setattr(session, "_lock", lock)
    admitted = threading.Event()
    submit = session.submit

    def noting_submit(text: str) -> None:
        submit(text)
        admitted.set()
        lock.contended.set()

    monkeypatch.setattr(session, "submit", noting_submit)
    composer = pty_host.PROVIDERS[0].composer_ready
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(session._on_output, "\x1b[?2004h" + composer)
        assert held.writing.wait(timeout=30)
        submitting = pool.submit(
            _post, f"/api/terminal/sessions/{sid}/submit", {"text": "second"}
        )
        assert lock.contended.wait(timeout=30)
        assert admitted.is_set()
        held.release.set()
        reading.result()
        assert submitting.result().status_code == 200


def _delivering_a_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> tuple[terminal_routes.TerminalSession, _HeldWritePty, _WatchedLock]:
    """Return a session whose request "written" is in a paste held mid-write, its PTY and lock.

    The session's delivery lock is watched. The request was queued first; the session's output
    then turned its composer live on another thread, whose delivery is writing the ritual with
    that request when this returns.
    """
    held = _HeldWritePty()
    _wire_stub_spawn(monkeypatch, held)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    writing = _WatchedLock()
    monkeypatch.setattr(session, "_writing", writing)
    session.submit("written")
    composer = pty_host.PROVIDERS[0].composer_ready
    threading.Thread(
        target=session._on_output, args=("\x1b[?2004h" + composer,)
    ).start()
    assert held.writing.wait(timeout=30)
    return session, held, writing


def _end_line_and_answer_agree_without(
    opened_project: Path, answer: dict, written: str
) -> None:
    """Assert the one end line holds what ``answer`` holds and neither holds ``written``."""
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert line["arguments"]["undelivered"] == answer["undelivered"]
    assert written not in answer["undelivered"]


def test_an_end_waits_for_a_write_under_way_then_answers_and_counts_alike(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """An end during a write waits for it; the written request is in neither its answer nor line.

    The end is seen waiting on the delivery's lock while the write is held.
    """
    session, held, writing = _delivering_a_request(client, monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        ending = pool.submit(
            _post,
            f"/api/terminal/sessions/{session.id}/end",
            {"user": "tester"},
        )
        ending.add_done_callback(lambda _: writing.contended.set())
        assert writing.contended.wait(timeout=30)
        assert not ending.done()
        held.release.set()
        answer = ending.result().json()

    _end_line_and_answer_agree_without(opened_project, answer, "written")


def test_a_write_resuming_while_an_end_records_leaves_its_answer_and_line_alike(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """A write held when an end begins, released once the end waits or records, changes nothing.

    Whichever the end reaches first, its delivery wait or its paused line, the write is released
    then; the end's answer and line agree and neither holds the written request.
    """
    session, held, writing = _delivering_a_request(client, monkeypatch)
    recording, recorded = threading.Event(), threading.Event()
    record = terminal_routes._record

    def held_record(
        event: str, arguments: dict, project: object, actor: str | None
    ) -> None:
        if event == "agent_terminal_ended":
            writing.contended.set()
            recording.set()
            assert recorded.wait(timeout=30)
        record(event, arguments, project, actor)

    monkeypatch.setattr(terminal_routes, "_record", held_record)
    with ThreadPoolExecutor(max_workers=1) as pool:
        ending = pool.submit(
            _post,
            f"/api/terminal/sessions/{session.id}/end",
            {"user": "tester"},
        )
        assert writing.contended.wait(timeout=30)
        held.release.set()
        assert recording.wait(timeout=30)
        recorded.set()
        answer = ending.result().json()

    _end_line_and_answer_agree_without(opened_project, answer, "written")


def test_a_socket_opened_while_an_end_writes_its_line_closes_only_after_the_line(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """While an end is writing its line, attaching and opening a socket close nothing.

    A subscriber attaching then finds no close queued; a WebSocket opened then is replayed and its
    handler next waits for input, not a close; no close is broadcast. Once released, the one
    ended close is broadcast and the WebSocket closed, each with the line already written.
    """
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    _wait_for(session, "FAKE_TERMINAL_READY")
    recording, release, boundary = (threading.Event() for _ in range(3))
    record = terminal_routes._record
    broadcast = session._broadcast
    close, receive_text = WebSocket.close, WebSocket.receive_text
    broadcast_closes: list[bool] = []
    socket_steps: list[tuple[str, bool]] = []

    def written() -> bool:
        return bool(audit_rows(opened_project, "agent_terminal_ended"))

    def held_record(
        event: str, arguments: dict, project: object, actor: str | None
    ) -> None:
        if event == "agent_terminal_ended":
            recording.set()
            assert release.wait(timeout=30)
        record(event, arguments, project, actor)

    def noting_broadcast(item: str | tuple[int, str]) -> None:
        if isinstance(item, tuple):
            broadcast_closes.append(written())
        broadcast(item)

    async def noting_close(
        self: WebSocket, code: int = 1000, reason: str | None = None
    ) -> None:
        socket_steps.append(("close", written()))
        boundary.set()
        await close(self, code, reason)

    async def noting_receive_text(self: WebSocket) -> str:
        socket_steps.append(("receive", written()))
        boundary.set()
        return await receive_text(self)

    async def attached_queue() -> asyncio.Queue:
        return session.attach()[1]

    monkeypatch.setattr(terminal_routes, "_record", held_record)
    monkeypatch.setattr(session, "_broadcast", noting_broadcast)
    monkeypatch.setattr(WebSocket, "close", noting_close)
    monkeypatch.setattr(WebSocket, "receive_text", noting_receive_text)
    with ThreadPoolExecutor(max_workers=1) as pool:
        ending = pool.submit(
            _post, f"/api/terminal/sessions/{sid}/end", {"user": "tester"}
        )
        assert recording.wait(timeout=30)
        assert asyncio.run(attached_queue()).empty()
        with client.websocket_connect(
            f"ws://127.0.0.1/api/terminal/ws/{sid}"
        ) as ws:
            assert "FAKE_TERMINAL_READY" in ws.receive_text()
            assert boundary.wait(timeout=30)
            assert socket_steps[0] == ("receive", False)
            assert broadcast_closes == []
            release.set()
            assert ending.result().status_code == 200
            assert (
                _read_until_closed(ws)[1] == terminal_routes.ENDED_CLOSE_CODE
            )
    assert broadcast_closes == [True]
    assert [step for step in socket_steps if step[0] == "close"] == [
        ("close", True)
    ]


def test_output_after_the_ended_close_reaches_no_socket(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ended close is the last item a subscriber is offered; later output reaches none."""
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]

    async def offered() -> list[object]:
        _, queue, _ = session.attach()
        await asyncio.to_thread(
            session.end, "the person ended it", "user:tester", lambda: None
        )
        session._on_output("late output")
        await asyncio.sleep(0)
        return [queue.get_nowait() for _ in range(queue.qsize())]

    assert asyncio.run(offered()) == [terminal_routes._ENDED_CLOSE]


def test_a_survivor_of_an_unrecorded_start_ends_without_an_end_line(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """A launch whose start line failed and whose process survived records no end of it."""
    import tcip_mcp.audit as audit_module

    survivor = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, survivor)
    append = audit_module.append

    def _refuse_append(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", _refuse_append)
    assert (
        client.post("/api/terminal/sessions", json=LAUNCH).status_code == 503
    )
    (sid,) = terminal_routes._SESSIONS
    monkeypatch.setattr(audit_module, "append", append)
    survivor._survives = False

    assert _end(client, sid).status_code == 200
    assert not [
        row
        for row in audit_rows(opened_project)
        if row["tool"].startswith("agent_terminal_")
    ]


def test_an_ended_close_offered_to_a_full_queue_survives_the_dropped_backlog() -> (
    None
):
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    queue.put_nowait("output")

    terminal_routes._offer(queue, terminal_routes._ENDED_CLOSE)

    assert queue.get_nowait() == terminal_routes._ENDED_CLOSE
    assert queue.empty()


def test_an_end_whose_line_fails_answers_the_audit_gap_with_the_undelivered_requests(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "never given"}
    )
    _refuse_record(monkeypatch)

    resp = _end(client, sid)

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "audit_entry_not_written"
    assert detail["committed"] == {"undelivered": ["never given"]}
    assert terminal_routes._SESSIONS == {}


def test_an_unattended_end_whose_line_fails_writes_it_to_the_stream_and_the_error_log(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure reaches the stream and the error log before the ended close is broadcast."""
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    broadcast = session._broadcast
    seen: list[tuple[object, bool]] = []

    def noting_broadcast(item: str | tuple[int, str]) -> None:
        logged = any("ended unrecorded" in r.message for r in caplog.records)
        seen.append((item, logged))
        broadcast(item)

    monkeypatch.setattr(session, "_broadcast", noting_broadcast)
    _refuse_record(monkeypatch)
    with caplog.at_level("ERROR"):
        assert session.end_unattended("the agent's process exited") is True

    (text, _), close = seen
    assert str(text).startswith(
        f"\r\n{_UNRECORDED_END}agent_terminal_ended completed and its audit entry"
    )
    assert close == (terminal_routes._ENDED_CLOSE, True)


def test_shutdown_all_ends_every_live_session_with_one_line_by_no_one(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """``shutdown_all`` writes one end line, by no one, for each session the backend holds.

    The create route answers its one live session to a second create, so the two sessions are
    started through :meth:`TerminalSession.start` and held as that route holds them.
    """
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    ids = [f"term_shutdown_{n}" for n in range(2)]
    for sid in ids:
        session = terminal_routes.TerminalSession(sid)
        assert (
            session.start(24, 80, pty_host.PROVIDERS[0], "user:tester") is None
        )
        terminal_routes._SESSIONS[sid] = session
    terminal_routes._SESSIONS[ids[0]].submit("never given")

    terminal_routes.shutdown_all()

    lines = audit_rows(opened_project, "agent_terminal_ended")
    assert [line["arguments"]["session_id"] for line in lines] == ids
    assert [line["arguments"]["undelivered"] for line in lines] == [
        ["never given"],
        [],
    ]
    assert {line["arguments"]["cause"] for line in lines} == {
        "the backend shut down"
    }
    assert all(_ended(line).resumable for line in lines)
    assert not [line for line in lines if "actor" in line]
    assert terminal_routes._SESSIONS == {}


def _os_pid_alive(pid: int) -> bool:
    """Answer whether ``pid`` is alive, by ``tasklist`` on Windows and a zero signal elsewhere."""
    if pty_host.os.name == "nt":
        out = __import__("subprocess").run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
        )
        return str(pid) in out.stdout
    try:
        pty_host.os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but owned elsewhere
        return True


def test_shutdown_all_terminates_process(client: TestClient) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    assert session.alive()
    pid = session._pty.pid  # capture before shutdown nulls the pty
    terminal_routes.shutdown_all()
    deadline = time.time() + 15
    while time.time() < deadline and _os_pid_alive(pid):
        time.sleep(0.1)
    assert not _os_pid_alive(pid)
    assert terminal_routes._SESSIONS == {}


def test_write_and_resize_after_process_death_do_not_raise(
    client: TestClient,
) -> None:
    # pywinpty raises EOFError/WinptyError (not OSError) on a dead PTY: a keystroke or
    # resize racing process exit must degrade, never crash the WS handler.
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        _read_until(ws, "FAKE_TERMINAL_READY")
        ws.send_json({"type": "input", "data": "exit\r"})
        # The socket must survive control traffic racing the exit until the end closes it.
        ws.send_json({"type": "resize", "rows": 40, "cols": 120})
        ws.send_json({"type": "input", "data": "into the void\r"})
        assert _read_until_closed(ws)[1] == terminal_routes.ENDED_CLOSE_CODE
    assert session.write("x") is False
    session.resize(10, 10)  # no raise


def test_concurrent_creates_spawn_single_session() -> None:
    from concurrent.futures import ThreadPoolExecutor

    def create(_: int) -> str:
        return (
            TestClient(app, base_url="http://127.0.0.1")
            .post("/api/terminal/sessions", json=LAUNCH)
            .json()["session_id"]
        )

    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            ids = list(ex.map(create, range(8)))
        alive = [s for s in terminal_routes._SESSIONS.values() if s.alive()]
        assert len(alive) == 1
        assert set(ids) == {alive[0].id}
    finally:
        terminal_routes.shutdown_all()


# ── what a session records about the program it launched ───────────────


def test_create_answers_the_launched_executable_and_no_version_for_an_override(
    client: TestClient,
) -> None:
    """An override's launch records its argv, no version and no row's confinement."""
    body = client.post("/api/terminal/sessions", json=LAUNCH).json()

    launched = body["launched"]
    assert launched["argv"] == pty_host._override_argv()
    assert launched["version"] is None
    assert launched["confinement"] is None


def test_the_resolved_cli_is_probed_for_the_version_it_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (row, *_) = _rows_with(monkeypatch, executable=Path(sys.executable).name)
    command = pty_host.resolve_terminal_command(row)
    assert command is not None

    launched = pty_host.launched_program(*command)

    assert launched["argv"] == command[0]
    assert Path(launched["argv"][0]).name == Path(sys.executable).name
    assert launched["version"].startswith("Python ")


def test_the_resolved_command_decides_whether_its_version_is_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = pty_host.resolve_terminal_command(pty_host.PROVIDERS[0])
    assert command is not None
    monkeypatch.delenv("TCIP_TERMINAL_CMD")

    assert pty_host.launched_program(*command)["version"] is None


def _versioned_executable(directory: Path, version_file: Path) -> Path:
    """Write an executable that prints ``version_file``'s text for ``--version``.

    For any other arguments it runs the fake program. It is ``versioned_agent``
    (``versioned_agent.cmd`` on Windows) in the newly made ``directory``; the return is its path.
    """
    directory.mkdir()
    if pty_host.os.name == "nt":
        wrapper = directory / "versioned_agent.cmd"
        wrapper.write_text(
            f'@if "%1"=="--version" (type "{version_file}" & exit /b 0)\r\n'
            f'@"{sys.executable}" -u "{FAKE}" %*\r\n',
            encoding="utf-8",
        )
    else:
        wrapper = directory / "versioned_agent"
        wrapper.write_text(
            f'#!/bin/sh\nif [ "$1" = "--version" ]; then cat "{version_file}"; exit 0; fi\n'
            f'exec "{sys.executable}" -u "{FAKE}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    return wrapper


def test_a_harness_updated_in_place_is_seen_at_its_next_launch_without_a_backend_restart(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opened_project: Path,
) -> None:
    """A harness updated in place is seen at its next launch without a backend restart.

    The second launch of the same executable reports the version it declares then, on the
    launch and on its audit line.
    """
    version_file = tmp_path / "version.txt"
    row = pty_host.PROVIDERS[0]
    version_file.write_text("1.0.0\n", encoding="utf-8")
    monkeypatch.setattr(
        pty_host,
        "PROVIDERS",
        (
            dataclasses.replace(
                row,
                executable=str(
                    _versioned_executable(tmp_path / "bin", version_file)
                ),
            ),
        ),
    )
    monkeypatch.delenv("TCIP_TERMINAL_CMD")

    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert created["launched"]["version"] == "1.0.0"

    version_file.write_text("9.9.9 (updated)\n", encoding="utf-8")
    assert _end(client, created["session_id"]).status_code == 200
    relaunched = client.post("/api/terminal/sessions", json=LAUNCH).json()

    assert relaunched["launched"]["version"] == "9.9.9 (updated)"
    first, second = audit_rows(opened_project, "agent_terminal_started")
    assert first["arguments"]["version"] == "1.0.0"
    assert second["arguments"]["version"] == "9.9.9 (updated)"


def test_the_spawned_process_inherits_the_terminal_session_id(
    client: TestClient,
) -> None:
    """The spawned process's environment names the session id."""
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        banner = _read_until(ws, "]")
    assert f"[session:{sid}]" in banner


def test_the_spawned_environment_carries_no_claude_session_or_cache_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``CLAUDE``-prefixed variable nor prompt-cache override reaches the child.

    The child would otherwise read itself as a nested Claude Code session, or run a cache
    lifetime other than the one its row requests; a control ranked below the row's setting and
    the other variables, the terminal session id among them, pass through.
    """
    withheld = (
        "CLAUDECODE",
        "CLAUDE_CODE_CHILD_SESSION",
        "CLAUDE_CODE_PROMPT_CACHE_TTL",
        "FORCE_PROMPT_CACHING_5M",
        "DISABLE_PROMPT_CACHING",
        "DISABLE_PROMPT_CACHING_OPUS",
    )
    for name in withheld:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("ENABLE_PROMPT_CACHING_1H", "1")
    monkeypatch.setenv("TCIP_KEPT_FOR_THE_TEST", "yes")

    env = pty_host.spawn_env("t1")

    assert not set(withheld) & set(env)
    assert env["ENABLE_PROMPT_CACHING_1H"] == "1"
    assert env["TCIP_KEPT_FOR_THE_TEST"] == "yes"
    assert env[pty_host.TERMINAL_SESSION_ENV] == "t1"


def test_each_launch_leaves_one_audit_line_in_the_open_projects_log(
    client: TestClient, opened_project: Path
) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    _end(client, sid)
    second = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]

    rows = audit_rows(opened_project, "agent_terminal_started")
    assert [row["arguments"]["session_id"] for row in rows] == [sid, second]
    assert [row["arguments"]["provider"] for row in rows] == [
        LAUNCH["provider"]
    ] * 2
    assert rows[0]["arguments"]["argv"] == pty_host._override_argv()
    assert rows[0]["arguments"]["version"] is None
    assert [row["actor"] for row in rows] == ["user:tester"] * 2
    assert "source" not in rows[0]


def test_create_session_answers_503_and_terminates_the_process_when_the_start_line_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """A spawn whose launch line fails to append answers 503 and terminates the process.

    The refusal says the line could not be written, no start line is recorded, and no session
    is live.
    """
    import tcip_mcp.audit as audit_module

    def _refuse_append(*args: object, **kwargs: object) -> None:
        raise RuntimeError("audit log unwritable")

    monkeypatch.setattr(audit_module, "append", _refuse_append)
    resp = client.post("/api/terminal/sessions", json=LAUNCH)
    assert resp.status_code == 503
    assert "could not be written" in resp.json()["detail"]
    assert audit_rows(opened_project, "agent_terminal_started") == []
    assert all(not s.alive() for s in terminal_routes._SESSIONS.values())


def test_a_create_and_one_after_an_end_answer_the_launched_provider_and_program(
    client: TestClient,
) -> None:
    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    _end(client, created["session_id"])
    relaunched = client.post("/api/terminal/sessions", json=LAUNCH).json()

    assert relaunched["session_id"] != created["session_id"]
    for answer in (created, relaunched):
        assert answer["launched"]["provider"] == LAUNCH["provider"]
        assert (
            Path(answer["launched"]["argv"][0]).name
            == Path(sys.executable).name
        )


def test_a_create_and_one_after_an_end_answer_the_ritual_the_library_builds_for_the_open_project(
    client: TestClient,
    opened: OpenProject,
) -> None:
    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert created["ritual"] == pty_host.session_ritual(
        pty_host.project_launch(opened)
    )
    assert "Test project" in created["ritual"]
    for step in ("load_project_memory", "inspect_project"):
        assert step in created["ritual"]
    line = pty_host.doctor_command(INTERPRETER, opened.root)
    assert _ritual_doctor_line(created["ritual"]) == line

    attached = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert attached == created

    _end(client, created["session_id"])
    relaunched = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert relaunched["ritual"] == created["ritual"]


def test_a_launch_after_a_rename_of_the_open_project_names_its_new_name(
    client: TestClient, opened: OpenProject
) -> None:
    """A launch after a rename of the open project names its new name.

    After a rename made while the project is open, the next launch's ritual states the new name.
    """
    renamed = client.post(
        "/api/projects/rename",
        json={
            "id": opened.id,
            "display_name": "Ridge block",
            "user": "tester",
        },
    )
    assert renamed.status_code == 200, renamed.text

    created = client.post("/api/terminal/sessions", json=LAUNCH)

    assert created.status_code == 200, created.text
    assert "Project: Ridge block" in created.json()["ritual"]


def test_a_launch_for_an_open_project_whose_record_will_not_read_refuses_naming_why(
    client: TestClient,
    opened: OpenProject,
) -> None:
    from tcip_mcp.project_record import project_record_key
    from tests._record_damage_fixtures import damage_record

    damage_record(project_record_key(str(opened.root)), b"{not valid json")

    created = client.post("/api/terminal/sessions", json=LAUNCH)

    assert created.status_code == 503
    assert "names no readable project" in created.json()["detail"]
    assert terminal_routes._SESSIONS == {}


# ── resuming a row's conversation ───────────────────────────────────────


@pytest.fixture
def fake_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, pty_host.Provider]:
    """Install every provider row with the fake program as its executable; return them by id.

    Antigravity's settings file is a scratch one.
    """
    monkeypatch.setattr(
        pty_host, "ANTIGRAVITY_SETTINGS", tmp_path / "agy-settings.json"
    )
    rows = _rows_with(
        monkeypatch, executable=str(fake_executable(tmp_path / "bin"))
    )
    return {row.id: row for row in rows}


def _launch(client: TestClient, row: pty_host.Provider) -> dict:
    """Create a session of ``row`` by "tester" and return the launch the route answered."""
    resp = client.post(
        "/api/terminal/sessions",
        json={**LAUNCH, "provider": row.id, "cols": 500},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _report(client: TestClient, sid: str, conversation: str) -> None:
    """Report ``conversation`` for session ``sid`` as its session hook does."""
    resp = client.post(
        f"/api/terminal/sessions/{sid}/conversation",
        json={"conversation": conversation},
    )
    assert resp.status_code == 200, resp.text


def _talked_then_exited(client: TestClient, conversation: str | None) -> str:
    """Return the id of the session just launched, ended by its own process's exit.

    Its agent wrote its banner first, and its hook reported ``conversation`` unless ``None``.
    """
    (session,) = terminal_routes._SESSIONS.values()
    _wait_for(session, "FAKE_TERMINAL_READY")
    if conversation is not None:
        _report(client, session.id, conversation)
    session.write("exit\r")
    deadline = time.time() + 30
    while session.id in terminal_routes._SESSIONS and time.time() < deadline:
        time.sleep(0.1)
    assert session.id not in terminal_routes._SESSIONS
    return session.id


def _resumes(
    argv: list[str], row: pty_host.Provider, conversation: str
) -> bool:
    """Return whether ``argv`` carries ``row``'s resume argument followed by ``conversation``."""
    assert row.resume is not None
    return (row.resume.argument, conversation) in pairwise(argv)


def test_an_end_nobody_chose_then_a_launch_of_its_row_resumes_it_and_delivers_requests_alone(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
    pastes_after: None,
) -> None:
    """A launch of a row whose last end nobody chose resumes the conversation its hook reported.

    Inside the row's cache lifetime the launch carries the row's resume argument and the id and
    settings pinning the fullscreen renderer, whose alternate screen is the row's composer
    marker, is given no ritual, its start line names the conversation, and a request staged for
    it is delivered alone.
    """
    row = fake_rows["claude"]
    _launch(client, row)
    _talked_then_exited(client, "conv-1")

    resumed = _launch(client, row)

    assert resumed["launched"]["resumed"] == "conv-1"
    assert resumed["ritual"] is None
    argv = resumed["launched"]["argv"]
    assert _resumes(argv, row, "conv-1")
    handed = json.loads(
        Path(argv[argv.index("--settings") + 1]).read_text(encoding="utf-8")
    )
    assert handed["env"]["CLAUDE_CODE_NO_FLICKER"] == "1"
    _, start = audit_rows(opened_project, "agent_terminal_started")
    assert start["arguments"]["resumed"] == "conv-1"
    sid = resumed["session_id"]
    client.post(f"/api/terminal/sessions/{sid}/submit", json={"text": "next"})
    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:next")
    assert _echoes(text) == ["echo:next"]


def test_a_launch_past_the_rows_cache_lifetime_is_fresh(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import timedelta

    row = fake_rows["claude"]
    assert row.resume is not None
    _launch(client, row)
    _talked_then_exited(client, "conv-1")
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    last = _ended(line).last_output_at
    assert last is not None
    later = last + row.resume.cache_lifetime + timedelta(seconds=1)
    monkeypatch.setattr(terminal_routes, "now", lambda: later)

    fresh = _launch(client, row)

    assert fresh["launched"]["resumed"] is None
    assert fresh["ritual"] is not None
    assert row.resume.argument not in fresh["launched"]["argv"]


def test_a_launch_after_the_persons_exit_is_fresh(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    row = fake_rows["claude"]
    sid = _launch(client, row)["session_id"]
    _wait_for(terminal_routes._SESSIONS[sid], "FAKE_TERMINAL_READY")
    _report(client, sid, "conv-1")
    assert _end(client, sid).status_code == 200

    fresh = _launch(client, row)

    assert fresh["launched"]["resumed"] is None
    assert fresh["ritual"] is not None


def test_a_launch_of_another_row_never_resumes_the_first_rows_conversation(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    _launch(client, fake_rows["claude"])
    _talked_then_exited(client, "conv-1")

    other = _launch(client, fake_rows["codex"])

    assert other["launched"]["resumed"] is None
    assert other["ritual"] is not None


def test_a_launch_in_another_project_never_resumes_a_conversation_of_the_first(
    client: TestClient,
    opened: OpenProject,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    """A conversation ended after the GUI opened another project is resumed only in its own.

    Its end lands in its launch's project, so a launch in the project open at its end is fresh
    and one back in its own project resumes it.
    """

    def open_in_the_gui(project: OpenProject) -> None:
        resp = client.post(
            "/api/projects/open", json={"id": project.id, "user": "grower"}
        )
        assert resp.status_code == 200, resp.text

    row = fake_rows["claude"]
    sid = _launch(client, row)["session_id"]
    _wait_for(terminal_routes._SESSIONS[sid], "FAKE_TERMINAL_READY")
    _report(client, sid, "conv-1")
    other = named_project(opened.root.parent / "hill_block", "Hill block")
    open_in_the_gui(other)
    terminal_routes.shutdown_all()

    assert _launch(client, row)["launched"]["resumed"] is None
    terminal_routes.shutdown_all()
    open_in_the_gui(opened)
    assert _launch(client, row)["launched"]["resumed"] == "conv-1"


_RESUMED_IN_A_FRESH_PROCESS = """
import sys
from pathlib import Path

import tcip_store
from tcip_web import terminal as pty_host
from tcip_web.routes import terminal as routes
from tcip_web.state import OpenProject

tcip_store.bind()
(row,) = [row for row in pty_host.PROVIDERS if row.id == sys.argv[1]]
print(routes._conversation_to_resume(row, OpenProject(Path(sys.argv[2]), sys.argv[3])))
"""
"""What a process that has never held a session resumes for a row and a project, printed."""


def test_after_a_backend_shutdown_a_fresh_process_resumes_from_the_log_alone(
    client: TestClient,
    opened: OpenProject,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    import subprocess

    row = fake_rows["claude"]
    sid = _launch(client, row)["session_id"]
    _wait_for(terminal_routes._SESSIONS[sid], "FAKE_TERMINAL_READY")
    _report(client, sid, "conv-1")
    terminal_routes.shutdown_all()

    printed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RESUMED_IN_A_FRESH_PROCESS,
            row.id,
            str(opened.root),
            opened.id,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )

    assert printed.stdout.strip() == "conv-1"


def test_a_conversation_whose_hook_never_reported_is_not_resumed_and_its_end_says_so(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    row = fake_rows["claude"]
    _launch(client, row)
    _talked_then_exited(client, None)
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert _ended(line).conversation is None

    fresh = _launch(client, row)

    assert fresh["launched"]["resumed"] is None
    assert fresh["ritual"] is not None


def test_antigravitys_row_is_never_resumed(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    row = fake_rows["antigravity"]
    _launch(client, row)
    _talked_then_exited(client, "conv-1")

    fresh = _launch(client, row)

    assert fresh["launched"]["resumed"] is None
    assert fresh["ritual"] is not None


def test_the_session_hook_is_one_command_across_launches_naming_no_launch(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    """Two launches of each hooked row register one hook command, which names neither session."""
    for row in (fake_rows["claude"], fake_rows["codex"]):
        hooks = []
        for _ in range(2):
            created = _launch(client, row)
            hooks.append(_registered_hook(created["launched"]["argv"]))
            assert created["session_id"] not in str(hooks[-1])
            assert _end(client, created["session_id"]).status_code == 200
        assert hooks[0] is not None and hooks[0] == hooks[1]


def test_the_hook_command_reports_its_stdins_conversation_for_its_terminal_session(
    client: TestClient, opened_project: Path
) -> None:
    """The session hook's own command line, run by a shell, reports the conversation it is given.

    Its stdin is a ``SessionStart`` event and its environment names the terminal session; the
    backend serving the workspace takes the report, the session's end line records it, and the
    command prints nothing a harness would add to its agent's context.
    """
    import subprocess

    from tcip_web.state import store
    from tools.smoke_terminal_e2e import served

    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with served(app, store.workspace):
        ran = subprocess.run(
            pty_host.conversation_hook_command(INTERPRETER),
            shell=True,
            input=json.dumps(
                {
                    "session_id": "conv-hook",
                    "hook_event_name": "SessionStart",
                    "source": "startup",
                    "cwd": str(store.workspace),
                }
            ),
            env={**pty_host.os.environ, pty_host.TERMINAL_SESSION_ENV: sid},
            capture_output=True,
            text=True,
            timeout=120,
        )

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == ""
    assert terminal_routes._SESSIONS[sid].conversation == "conv-hook"
    assert _end(client, sid).status_code == 200
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert _ended(line).conversation == "conv-hook"


def test_a_report_for_a_session_the_backend_does_not_hold_answers_404(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/terminal/sessions/term_gone/conversation",
        json={"conversation": "conv-1"},
    )
    assert resp.status_code == 404


# ── the hand-off before a switch of harness ─────────────────────────────


def _hand_off(client: TestClient, sid: str) -> Response:
    """Post the person "tester"'s switch of harness for session ``sid`` and return the answer."""
    return client.post(
        f"/api/terminal/sessions/{sid}/hand-off", json={"user": "tester"}
    )


def _retrospective_declared_by(
    project: Path, terminal_session: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write a retrospective as the MCP server an agent of ``terminal_session`` launched does.

    The server's identity is begun under that session's ``TCIP_TERMINAL_SESSION`` and ended
    after the one ``write_retrospective`` call.
    """
    from tcip_mcp import agent_identity
    from tcip_mcp.tools.meta_tools import write_retrospective

    monkeypatch.setenv(agent_identity.TERMINAL_SESSION_ENV, terminal_session)
    agent_identity.begin("a-harness", "1.0")
    try:
        write_retrospective(
            project,
            "hand-off",
            "a switch of harness",
            "the run",
            "nothing",
            actor=None,
        )
    finally:
        agent_identity.end()


def test_a_switch_delivers_the_hand_off_request_and_ends_after_its_own_retrospective(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A switch delivers its request and ends on the hand-off its own agent declared, after it.

    A retrospective declared by another terminal session ends nothing; the one this session's
    agent declared is followed by the session's end, by the person who switched, its
    conversation resumable and its hand-off landed.
    """
    pty = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, pty)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]

    switched = _hand_off(client, sid)
    assert switched.status_code == 200, switched.text
    assert switched.json() is None
    session._on_output("\x1b[?2004h" + pty_host.PROVIDERS[0].composer_ready)
    assert pty.writes == [
        pty_host.paste(
            f"{session.ritual}\n\n{terminal_routes.HAND_OFF_REQUEST}"
        )
    ]

    _retrospective_declared_by(opened_project, "term_another", monkeypatch)
    session._on_output("the agent works")
    assert sid in terminal_routes._SESSIONS

    _retrospective_declared_by(opened_project, sid, monkeypatch)
    session._on_output("the agent wrote its hand-off")

    assert sid not in terminal_routes._SESSIONS
    tools = [
        (row["tool"], row.get("terminal_session"))
        for row in audit_rows(opened_project)
        if row["tool"] in ("write_retrospective", "agent_terminal_ended")
    ]
    assert tools == [
        ("write_retrospective", "term_another"),
        ("write_retrospective", sid),
        ("agent_terminal_ended", None),
    ]
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert line["actor"] == "user:tester"
    ended = _ended(line)
    assert (ended.resumable, ended.handed_off, ended.undelivered) == (
        True,
        True,
        [],
    )


class _SurvivesOnceStubPty(_StubPty):
    """A process whose first ``terminate`` it survives and whose second stops it."""

    def __init__(self) -> None:
        super().__init__(survives=False)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = self.terminate_calls == 1


def test_a_hand_off_whose_end_fails_to_stop_the_agent_ends_it_on_the_next_output(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hand-off that landed but whose termination failed ends the session on later output.

    The first attempt leaves the session held; the next output tries again and ends it, its
    end line recording the hand-off as landed.
    """
    pty = _SurvivesOnceStubPty()
    _wire_stub_spawn(monkeypatch, pty)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    assert _hand_off(client, sid).status_code == 200
    _retrospective_declared_by(opened_project, sid, monkeypatch)

    session._on_output("the agent wrote its hand-off")
    assert sid in terminal_routes._SESSIONS
    assert pty.terminate_calls == 1

    session._on_output("the agent idles")

    assert sid not in terminal_routes._SESSIONS
    assert pty.terminate_calls == 2
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert _ended(line).handed_off is True


def test_an_end_during_a_hand_off_ends_at_once_and_records_no_hand_off(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An end while the agent hands off ends the session at once, its hand-off not landed.

    The hand-off request it never delivered is not among the requests the end answers.
    """
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "never given"}
    )
    assert _hand_off(client, sid).status_code == 200

    ended = _end(client, sid)

    assert ended.status_code == 200, ended.text
    assert ended.json() == {"undelivered": ["never given"]}
    assert sid not in terminal_routes._SESSIONS
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert line["actor"] == "user:tester"
    assert (_ended(line).resumable, _ended(line).handed_off) == (True, False)


def test_an_end_whose_hand_off_evidence_will_not_read_leaves_the_agent_running_and_held(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An End during a hand-off whose log will not read fails before anything is stopped.

    The agent's process is not terminated, the session stays held, and no end line is written.
    """
    pty = _StubPty(survives=False)
    _wire_stub_spawn(monkeypatch, pty)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    assert _hand_off(client, sid).status_code == 200

    with monkeypatch.context() as unreadable:
        unreadable.setattr(terminal_routes, "acts_of", _unreadable_log)
        resp = _lenient_client().post(
            f"/api/terminal/sessions/{sid}/end", json={"user": "tester"}
        )

    assert resp.status_code == 500
    assert pty.terminate_calls == 0
    assert terminal_routes._SESSIONS[sid].alive()
    assert audit_rows(opened_project, "agent_terminal_ended") == []


_UNRECORDED_END = "The agent ended and no end line was recorded: "
"""What an end that writes no end line puts on its session's stream, ahead of why."""

_UNREADABLE = "the audit log carries 1 undecodable entries"

_UNREADABLE_END = _UNRECORDED_END + _UNREADABLE


def _unreadable_log(*args: object, **kwargs: object) -> None:
    """Stand in for an audit log read that refuses its undecodable entries."""
    raise ValueError(_UNREADABLE)


def test_shutdown_ends_every_session_when_one_hand_off_will_not_read(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A shutdown ends a session whose hand-off log will not read, and every session after it.

    The first session, handing off, is stopped and forgotten with no end line, its stream saying
    the agent ended unrecorded and why and the error log saying so too; the second ends with its
    line.
    """
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    ids = [f"term_unread_{n}" for n in range(2)]
    for sid in ids:
        session = terminal_routes.TerminalSession(sid)
        assert (
            session.start(24, 80, pty_host.PROVIDERS[0], "user:tester") is None
        )
        terminal_routes._SESSIONS[sid] = session
    first = terminal_routes._SESSIONS[ids[0]]
    assert first.hand_off("user:tester")

    with monkeypatch.context() as unreadable, caplog.at_level("ERROR"):
        unreadable.setattr(terminal_routes, "acts_of", _unreadable_log)
        terminal_routes.shutdown_all()

    assert terminal_routes._SESSIONS == {}
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert line["arguments"]["session_id"] == ids[1]
    assert _UNREADABLE_END in first.scrollback_snapshot()
    assert any("ended unrecorded" in r.message for r in caplog.records)


def test_a_process_exit_with_an_unreadable_hand_off_forgets_the_session_and_says_why(
    client: TestClient,
    opened_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The process's own exit, its hand-off unreadable, forgets the session and discloses why.

    No end line is written; the session's stream says the agent ended unrecorded and why, and the
    error log says so too.
    """
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    assert _hand_off(client, sid).status_code == 200

    with monkeypatch.context() as unreadable, caplog.at_level("ERROR"):
        unreadable.setattr(terminal_routes, "acts_of", _unreadable_log)
        assert session.end_unattended("the agent's process exited") is True

    assert sid not in terminal_routes._SESSIONS
    assert _UNREADABLE_END in session.scrollback_snapshot()
    assert any("ended unrecorded" in r.message for r in caplog.records)
    assert audit_rows(opened_project, "agent_terminal_ended") == []


def test_a_switch_from_a_session_for_no_project_ends_it_at_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "never given"}
    )

    switched = _hand_off(client, sid)

    assert switched.status_code == 200, switched.text
    assert switched.json() == {"undelivered": ["never given"]}
    assert terminal_routes._SESSIONS == {}

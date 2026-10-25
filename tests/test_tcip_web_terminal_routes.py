"""The embedded agent terminal over the platform's real PTY backend.

The fake agent ``tests/fake_terminal_app.py`` drives it, through ``TCIP_TERMINAL_CMD`` or
standing as a provider row's executable.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import itertools
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import psutil
import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

from tcip_web import terminal as pty_host
from tcip_web.app import app
from tcip_web.routes import terminal as terminal_routes
from tcip_web.state import OpenProject
from tests._audit_fixtures import audit_rows, refuse_audit_appends
from tests._terminal_fixtures import FAKE, fake_executable
from tests._web_fixtures import named_project

if TYPE_CHECKING:
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
"""The interpreter this backend runs, the one every launch's MCP server and session hook name."""

HOOK_ARGV = [INTERPRETER, "-m", "tcip_web.cli", "session-hook"]


def _tool_names() -> list[str]:
    """Return the sorted name of every tool a server for no project lists."""
    from tcip_web.cli import tool_listing

    return sorted(t.name for t in tool_listing())


if pty_host.os.name == "nt":
    pytest.importorskip("winpty")


pytestmark = pytest.mark.usefixtures("fake_terminal")


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


def _registered_hook(argv: list[str]) -> dict | None:
    """Return the session hook handler the rendered ``argv`` registers, ``None`` for none.

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
    handlers = [
        handler
        for group in hooks.get("SessionStart", [])
        for handler in group["hooks"]
    ]
    assert len(handlers) <= 1, handlers
    return handlers[0] if handlers else None


def _agy_hooks() -> dict:
    """Return the Antigravity hooks file of the backend's workspace."""
    from tcip_web.state import store

    hooks = store.workspace / ".agents" / "hooks.json"
    return json.loads(hooks.read_text(encoding="utf-8"))


def _port_recorded() -> None:
    """Record a port for the backend serving this test's workspace, as a served backend does."""
    from tcip_mcp.web_client import record_backend_port
    from tcip_web.state import store

    record_backend_port(store.workspace, 8765)


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
        opened,
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


def test_the_orientation_for_no_project_says_the_session_was_launched_with_none() -> (
    None
):
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge
    from tcip_mcp.tools.project_tools import initialize_project

    told = pty_host.orientation(None, "startup")
    assert told is not None
    assert "launched with no project open" in told
    assert initialize_project.__name__ in told
    assert serve_domain_knowledge.__name__ in told


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


def test_winpty_terminate_waits_for_the_process_up_to_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_WinPty.terminate`` tree-kills its process and waits for it through psutil's bound.

    A process the kill left running is waited for with ``TERMINATE_WAIT_S`` as the timeout and
    left running once it passes, and a process already gone is not waited for, neither raising.
    """
    import subprocess

    killed: list[list[str]] = []
    monkeypatch.setattr(
        pty_host.subprocess, "run", lambda argv, **kw: killed.append(argv)
    )
    monkeypatch.setattr(pty_host, "TERMINATE_WAIT_S", 0.5)
    bounds: list[float | None] = []
    wait = psutil.Process.wait

    def recorded(
        process: psutil.Process, timeout: float | None = None
    ) -> object:
        bounds.append(timeout)
        return wait(process, timeout)

    monkeypatch.setattr(psutil.Process, "wait", recorded)
    running = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    try:
        for child, waited in ((running, [0.5]), (gone, [])):
            win_pty = pty_host._WinPty.__new__(pty_host._WinPty)
            win_pty._p = SimpleNamespace(isalive=lambda: True)
            win_pty.pid = child.pid
            win_pty.terminate()
            assert bounds == waited
            bounds.clear()
            assert killed.pop() == [
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(child.pid),
            ]
        assert running.poll() is None
    finally:
        running.kill()
        running.wait()


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
    that argv and the row's preparation. Each row registers the one session hook: Claude's in
    exec form in its handed settings, whose permission lists are the shipped ones and name no
    console command, Codex's as a shell line in its overrides, and Antigravity's as a shell line
    naming the session's start route in the workspace's hooks file, beside allowing only the
    tcip tools.
    """
    from tcip_web.cli.session_hook import session_start_url
    from tcip_web.state import store

    _port_recorded()
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
    hook = {
        "claude": {
            "type": "command",
            "command": HOOK_ARGV[0],
            "args": HOOK_ARGV[1:],
        },
        "codex": {
            "type": "command",
            "command": " ".join(HOOK_ARGV),
            "additionalContextLimit": 0,
        },
        "antigravity": None,
    }[row.id]
    assert _registered_hook(launched) == hook
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
        assert handed["permissions"] == shipped["permissions"]
        assert not any(
            "tcip" in entry
            for entries in handed["permissions"].values()
            for entry in entries
            if not entry.startswith("mcp__tcip__")
        )
    if row.prepare is None:
        expected_prepared: list[str] = []
    else:
        hooks = store.workspace / ".agents" / "hooks.json"
        url = session_start_url(store.workspace, sid)
        assert _agy_hooks() == {
            "tcip-session-start": {
                "SessionStart": [
                    {
                        "type": "command",
                        "command": " ".join([*HOOK_ARGV, url]),
                    }
                ]
            }
        }
        server = pty_host.mcp_server(opened)
        (rule,) = [
            path
            for path in hooks.parent.rglob("*")
            if path.is_file() and path != hooks
        ]
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
            f"allowed {len(_tool_names())} tcip entries in {settings}",
            f"registered the session hook in {hooks}",
            f"wrote the orientation rule {rule}",
        ]
        allowed = json.loads(settings.read_text(encoding="utf-8"))[
            "permissions"
        ]["allow"]
        assert allowed == [f"mcp(tcip/{name})" for name in _tool_names()]
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
        executable: str, project: OpenProject | None, session: str
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
    kept = [
        "command(git log)",
        'command(regex:^python -m tcip_web\\.cli dump-store "/p"$)',
        "unsandboxed(git log)",
    ]
    settings.write_text(
        json.dumps(
            {
                "model": "kept",
                "permissions": {"allow": kept, "deny": ["x"]},
            }
        ),
        encoding="utf-8",
    )

    first = pty_host.allow_tcip_tools(settings)
    second = pty_host.allow_tcip_tools(settings)

    names = _tool_names()
    assert first == f"allowed {len(names)} tcip entries in {settings}"
    assert second == f"allowed 0 tcip entries in {settings}"
    body = json.loads(settings.read_text(encoding="utf-8"))
    assert body["model"] == "kept"
    assert body["permissions"]["deny"] == ["x"]
    assert body["permissions"]["allow"] == [
        *kept,
        *(f"mcp(tcip/{name})" for name in names),
    ]

    settings.write_text("[]", encoding="utf-8")
    with pytest.raises(
        pty_host.PreparationFailedError, match="does not hold a JSON object"
    ):
        pty_host.allow_tcip_tools(settings)


def test_agys_hook_registration_keeps_the_hooks_files_other_entries(
    opened: OpenProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registering agy's session hook keeps every other hook of the workspace's hooks file.

    The tcip hook states no timeout, so agy's own default governs it, a launch after it replaces
    it rather than adding a second, and a file that does not hold a JSON object fails the
    preparation by name.
    """
    from tcip_web.state import store

    monkeypatch.setattr(
        pty_host, "ANTIGRAVITY_SETTINGS", tmp_path / "agy-settings.json"
    )
    monkeypatch.setattr(
        pty_host, "run_to_completion", lambda argv, step: "added"
    )
    _port_recorded()
    hooks = store.workspace / ".agents" / "hooks.json"
    hooks.parent.mkdir()
    linter = {
        "PostToolUse": [
            {
                "matcher": "run_command",
                "hooks": [{"type": "command", "command": "./lint.sh"}],
            }
        ]
    }
    hooks.write_text(json.dumps({"my-linter-hook": linter}), encoding="utf-8")

    for session in ("term_first", "term_second"):
        pty_host.prepare_antigravity("agy", opened, session)

    written = _agy_hooks()
    assert written.get("my-linter-hook") == linter
    (tcip,) = written["tcip-session-start"]["SessionStart"]
    assert set(tcip) == {"type", "command"}
    assert tcip["command"].endswith("/term_second/session-start")
    assert set(written) == {"my-linter-hook", "tcip-session-start"}

    hooks.write_text("[]", encoding="utf-8")
    with pytest.raises(
        pty_host.PreparationFailedError, match="does not hold a JSON object"
    ):
        pty_host.prepare_antigravity("agy", opened, "term_third")


def test_agys_preparation_writes_one_always_on_rule_naming_the_orientations_tools(
    opened: OpenProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Antigravity's preparation writes one always-on rule into the workspace, its line naming it.

    The rule's frontmatter triggers it always; its text names each tool the orientation is read
    from, as the project's orientation does, and says to read it again once earlier turns are
    summarized. A later preparation replaces it, and a neighboring rule is left as it was.
    """
    import yaml

    from tcip_web.state import store

    monkeypatch.setattr(
        pty_host, "ANTIGRAVITY_SETTINGS", tmp_path / "agy-settings.json"
    )
    monkeypatch.setattr(
        pty_host, "run_to_completion", lambda argv, step: "added"
    )
    _port_recorded()
    rules = store.workspace / ".agents" / "rules"
    rules.mkdir(parents=True)
    neighbor, ours = rules / "team.md", "---\ntrigger: manual\n---\nours\n"
    neighbor.write_text(ours, encoding="utf-8")

    prepared = pty_host.prepare_antigravity("agy", opened, "term_first")
    (rule,) = [path for path in rules.iterdir() if path != neighbor]
    assert str(rule) in prepared[-1]
    written = rule.read_text(encoding="utf-8")
    rule.write_text("stale", encoding="utf-8")
    pty_host.prepare_antigravity("agy", opened, "term_second")

    assert rule.read_text(encoding="utf-8") == written
    assert sorted(rules.iterdir()) == sorted([neighbor, rule])
    assert neighbor.read_text(encoding="utf-8") == ours
    _, frontmatter, text = written.split("---\n", 2)
    assert yaml.safe_load(frontmatter)["trigger"] == "always_on"
    assert "earlier turns have been summarized, read it again" in text
    told = pty_host.orientation(opened, "startup")
    assert told is not None
    for tool in pty_host.orientation_tools():
        assert tool.__name__ in text and tool.__name__ in told, tool.__name__


def test_the_session_hook_runs_the_console_command_under_the_mcp_servers_interpreter(
    opened: OpenProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session hook names the console command under the MCP server's interpreter.

    Its argv is the launch's MCP server command, ``-m tcip_web.cli`` and the dispatcher's own
    ``session-hook`` command, and running ``tcip_web.cli`` as a module reaches the ``main`` the
    ``tcip`` console script names.
    """
    import runpy

    import tomllib

    import tcip_web.cli

    assert pty_host.session_hook_argv() == [
        pty_host.mcp_server(opened).command,
        "-m",
        "tcip_web.cli",
        "session-hook",
    ]
    assert tcip_web.cli.COMMANDS["session-hook"] == "tcip_web.cli.session_hook"
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


def test_the_session_hook_line_refuses_an_interpreter_no_shell_reads_as_one_word(
    opened: OpenProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interpreter a shell does not read as one word refuses the hook's shell line by name.

    Codex's overrides and so a Codex launch refuse it, naming the argument; Claude's exec-form
    hook passes it as its command unchanged.
    """
    for interpreter in (
        "/two words/python",
        "/em space/python",
        "",
        "/conda(alt)/python",
        "/a&b/python",
    ):
        monkeypatch.setattr(
            pty_host, "backend_interpreter", lambda: interpreter
        )
        with pytest.raises(ValueError, match="shell line refuses") as refused:
            pty_host.codex_overrides(opened, {})
        assert repr(interpreter) in str(refused.value)
        (settings,) = pty_host.render_argv(
            [pty_host.CLAUDE_SETTINGS_ARG], opened, {}, None
        )
        assert _registered_hook(["--settings", settings]) == {
            "type": "command",
            "command": interpreter,
            "args": HOOK_ARGV[1:],
        }


@pytest.mark.parametrize(
    "interpreter", [INTERPRETER, "/env bin/python"], ids=["spelled", "refused"]
)
def test_claudes_codexs_and_agys_hooks_run_one_session_hook_argv(
    interpreter: str,
    opened: OpenProject,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude's settings, Codex's overrides and agy's hooks file run one session hook argv.

    Claude's exec-form handler is the argv itself, Codex's command the argv as one shell line
    and agy's that line naming the session's start route, each read from what the launch
    writes; an interpreter no shell line spells leaves Claude's handler unchanged, refuses
    Codex's overrides and fails agy's preparation naming it, writing nothing.
    """
    from tcip_web.cli.session_hook import session_start_url
    from tcip_web.state import store

    monkeypatch.setattr(pty_host, "backend_interpreter", lambda: interpreter)
    monkeypatch.setattr(
        pty_host, "ANTIGRAVITY_SETTINGS", tmp_path / "agy-settings.json"
    )
    monkeypatch.setattr(
        pty_host, "run_to_completion", lambda argv, step: "added"
    )
    _port_recorded()
    argv = [interpreter, *HOOK_ARGV[1:]]
    (claude_settings,) = pty_host.render_argv(
        [pty_host.CLAUDE_SETTINGS_ARG], opened, {}, None
    )
    claude = _registered_hook(["--settings", claude_settings])
    assert claude is not None
    assert [claude["command"], *claude["args"]] == argv

    if interpreter != INTERPRETER:
        with pytest.raises(ValueError, match="shell line refuses"):
            pty_host.codex_overrides(opened, {})
        with pytest.raises(
            pty_host.PreparationFailedError, match="shell line refuses"
        ):
            pty_host.prepare_antigravity("agy", opened, "term_test")
        assert not (store.workspace / ".agents").exists()
        assert not (tmp_path / "agy-settings.json").exists()
        return
    codex = _registered_hook(pty_host.codex_overrides(opened, {}))
    assert codex is not None
    assert codex["command"].split(" ") == argv
    pty_host.prepare_antigravity("agy", opened, "term_test")
    (agy,) = _agy_hooks()["tcip-session-start"]["SessionStart"]
    assert agy["command"].split(" ") == [
        *argv,
        session_start_url(store.workspace, "term_test"),
    ]


SHELL_READS_IN_A_BARE_WORD = {
    '"': "opens a double-quoted segment under Bash and PowerShell",
    "$": "Bash and PowerShell expand a variable",
    "`": "Bash substitutes a command and PowerShell escapes the next character",
    "\\": "Bash reads an escape before some of the characters that may follow it",
    "!": "Bash expands history in an interactive shell with history expansion on",
    "\r": "ends the shell line",
    "\n": "ends the shell line",
    "“": "PowerShell reads a left double quotation mark as a double quote",
    "”": "PowerShell reads a right double quotation mark as a double quote",
    "„": "PowerShell reads a double low-9 quotation mark as a double quote",
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
"""Each character the session hook's shell line refuses in a word, by what Bash or PowerShell
does with it outside quotes, written from the shells' grammars rather than read off the
renderer's set, so a character dropped from that set fails its case."""


@pytest.mark.parametrize(
    "held",
    sorted(SHELL_READS_IN_A_BARE_WORD),
    ids=[f"U+{ord(c):04X}" for c in sorted(SHELL_READS_IN_A_BARE_WORD)],
)
def test_a_hook_word_holding_a_character_a_shell_reads_is_refused_by_name(
    held: str,
) -> None:
    """A word of the session hook's shell line holding a character a shell reads is refused.

    The refusal names the word and the character, and a word without it is spelled as itself.
    """
    word = f"/env/bin{held}/python"
    with pytest.raises(ValueError, match="shell line refuses") as refused:
        pty_host.shell_line([word, "-m"])
    assert repr(word) in str(refused.value)
    assert repr(held) in str(refused.value)
    assert pty_host.shell_line(["/env/bin/python", "-m"]) == (
        "/env/bin/python -m"
    )


def test_the_written_mcp_configuration_is_the_one_server_the_launch_states(
    opened: OpenProject,
) -> None:
    server = pty_host.mcp_server(opened)
    assert server.args == (
        "-m",
        "tcip_mcp",
        "--project",
        opened.root.as_posix(),
    )
    assert pty_host.mcp_server(None).args == ("-m", "tcip_mcp")

    (config,) = pty_host.render_argv(
        [pty_host.MCP_CONFIG_ARG],
        opened,
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
    :func:`mcp_server` produces, every variable of the launch's environment forwarded to it and
    the server required, so Codex's startup and resume wait for it, and the session hook as one
    shell line passing its whole context, alike for a launch for no project.
    """
    env = pty_host.spawn_env("term_test")
    rendered = pty_host.render_argv(
        ["--literal", pty_host.CODEX_CONFIG_ARG], opened, env, None
    )

    assert rendered[0] == "--literal"
    assert set(rendered[1::2]) == {"-c"}
    parsed, hooks = _codex_overrides(rendered[1:])
    server = pty_host.mcp_server(opened)
    assert (parsed["command"], parsed["args"]) == (
        server.command,
        list(server.args),
    )
    assert set(parsed["env_vars"]) == set(env)
    assert pty_host.TERMINAL_SESSION_ENV in parsed["env_vars"]
    assert parsed["default_tools_approval_mode"] == "approve"
    assert parsed.get("required") is True
    assert hooks == {
        "SessionStart": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": " ".join(HOOK_ARGV),
                        "additionalContextLimit": 0,
                    }
                ]
            }
        ]
    }
    assert _codex_overrides(pty_host.codex_overrides(None, env))[1] == hooks


def test_the_codex_overrides_carry_any_project_path_as_toml_reads_it_back() -> (
    None
):
    """A project path holding what a TOML basic string must escape is read back exactly.

    The path, a POSIX one so its backslash is a character, holds a supplementary character, a
    quotation mark, a backslash, a tab and a delete, read back by ``tomllib``.
    """
    from pathlib import PurePosixPath

    root = PurePosixPath("/projects") / 'grove \U0001f333 "north"\\\tb\x7f'
    project = OpenProject(root, "abc")  # type: ignore[arg-type]

    parsed, _ = _codex_overrides(
        pty_host.codex_overrides(project, pty_host.spawn_env("term_test"))
    )

    assert parsed["args"] == ["-m", "tcip_mcp", "--project", root.as_posix()]


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
    """Return every line the fake agent echoed, in order."""
    return re.findall(r"echo:[^\r\n]*", scrollback)


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


def test_a_fresh_launch_pastes_nothing_before_the_first_request_given_alone(
    client: TestClient,
    pastes_after: None,
) -> None:
    """A fresh launch's agent is given nothing before the first request a person submitted.

    Each request reaches the agent alone, once it turned paste on, and never earlier.
    """
    sid = client.post(
        "/api/terminal/sessions", json={**LAUNCH, "cols": 500}
    ).json()["session_id"]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "first request"}
    )

    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:first request")
    assert _echoes(text) == ["echo:first request"]
    assert "EARLY_INPUT" not in text

    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "second request"}
    )
    text = _wait_for(terminal_routes._SESSIONS[sid], "echo:second request")
    assert _echoes(text) == ["echo:first request", "echo:second request"]


def test_the_requests_staged_before_paste_is_on_each_go_as_their_own_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requests staged before paste is on each go as their own paste, oldest first.

    Nothing is written before the agent turns bracketed paste on and its composer is live; a
    request submitted afterwards is its own paste too.
    """
    row = pty_host.PROVIDERS[0]
    session, pty = _launched_session(monkeypatch, row)
    session.submit("first")
    session.submit("second")
    assert pty.writes == []

    session._on_output("\x1b[?2004h" + row.composer_ready)
    assert pty.written(2) == [
        pty_host.paste("first"),
        pty_host.paste("second"),
    ]

    session.submit("later")
    session.deliver()
    assert pty.written(3)[2:] == [pty_host.paste("later")]
    assert pty_host.paste("a\nb\r\nc") == "\x1b[200~a\rb\rc\x1b[201~\r"


def test_overlapping_creates_deliver_the_request_once(
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
    assert _echoes(text) == ["echo:tab request"]


class _Startup(NamedTuple):
    """What one installed harness wrote while starting in the terminal's PTY.

    It is the id of the row that launches it, the index of the read in which the log shows its
    composer frame drawn, read from the log and never from a row's marker (``None`` for a start
    whose log draws no composer frame), and the reads.
    """

    row: str
    live_at: int | None
    reads: list[str]


_AGY_QUERIES = [
    "\x1b[1t",
    "\x1b[c\x1b[?1004h\x1b[?9001h",
    "\x1b[?2026$p\x1b[?2027$p\x1b]11;?\x07\x1b]7501;?\x07",
]
"""agy 1.4.7's first three reads in every start traced."""

_AGY_SIGN_IN = (
    "\x1b[?25l\x1b[?5W\x1b[?2004h\x1b[>4;2m\x1b[>1u\x1b[?u\r\x1b[J\n\x1b[5C▄▀▀▄ ... \n"
    " Welcome to the Antigravity CLI. You are currently not signed in.\r\n\n ⣷  Signing in..."
)
"""agy 1.4.7's sign-in frame: bracketed paste on, then the sign-in screen."""

_STARTUPS = {
    "claude": _Startup(
        "claude",
        15,
        [
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
    ),
    "antigravity-trusted": _Startup(
        "antigravity",
        3,
        [
            *_AGY_QUERIES,
            "\x1b[?5W\x1b[?2004h\x1b[>4;2m\x1b[>1u\x1b[?u\x1b[0 q\r\x1b[J\n\x1b[6C▄▀▀▄        "
            "\x1b[1mAntigravity CLI 1.4.7\x1b[m\n ... \r\n>\r\n─ ... \r\n? for shortcuts"
            "\x1b[2A\x1b[13D\x1b[?25h",
            "\x1b[?25l\x1b[5A\t\t  Gemini 4 Argon (Medium)\x1b[K\n ... ",
        ],
    ),
    "antigravity-trusted-after-sign-in": _Startup(
        "antigravity",
        4,
        [
            *_AGY_QUERIES,
            _AGY_SIGN_IN,
            "\x1b[?2026h\x1b[0 q\x1b[8A\x1b[12D ▄▀▀▄        \x1b[1mAntigravity CLI 1.4.7\x1b[m\n"
            " ... \r\n>\r\n─ ... \r\n? for shortcuts\r\x1b[2A\x1b[2C\x1b[?25h\x1b[?2026l",
        ],
    ),
    "antigravity-untrusted": _Startup(
        "antigravity",
        None,
        [
            *_AGY_QUERIES,
            _AGY_SIGN_IN,
            "\r ⣯ ",
            "\r\x1b[9A\x1b[1mAccessing workspace:\x1b[m\r\n\x1b[K\n\x1b[1mC:\\Users\\...\\agy-ws"
            "\x1b[m\r\n\x1b[K\nDo you trust the contents of this project?\r\n\nAntigravity CLI "
            "requires permission to read, edit, and execute files here.\r\n\n> Yes, I trust this "
            "folder\r\n  No, exit\r\n\n  \x1b[1m↑/↓\x1b[m Navigate \xb7 \x1b[1menter\x1b[m Confirm"
            "\n\x1b[67CGemini 4 Argon \xb7 medium\r",
        ],
    ),
    "codex": _Startup(
        "codex",
        10,
        [
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
    ),
}
"""Starts of the installed harnesses in the terminal's PTY, long screen draws cut after their
opening sequences and spinner frames thinned. Claude Code 2.1.292 under
``tools/smoke_terminal_e2e.py`` drew its input prompt in read 15, and a paste written at its first
bracketed-paste-on, read 5, was lost. agy 1.4.7, spawned with ``--sandbox`` through pywinpty at 35
by 120, turned bracketed paste on in every start: in a folder it trusts it drew its banner and
composer in that read, or in the next when a sign-in frame came first; in a folder it does not
trust it stopped on its trust screen with no composer. Codex 0.160.1, launched with its row's
arguments and sent no input, wrote its first window title in read 10, its session footer followed,
and an Enter before it was held as "Waiting for startup" while a paste at it was submitted."""


@pytest.mark.parametrize("stream", sorted(_STARTUPS))
def test_a_request_is_pasted_at_the_read_where_the_harness_composer_goes_live(
    stream: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staged request is pasted at the read where the harness draws its composer frame.

    Fed the reads its harness wrote at startup, a launch of its row pastes nothing before the
    read in which the log shows the composer frame drawn and pastes the request at that read,
    and pastes nothing through a start that draws no composer frame, bracketed paste on in each.
    """
    startup = _STARTUPS[stream]
    (row,) = [row for row in pty_host.PROVIDERS if row.id == startup.row]

    session, pty = _launched_session(monkeypatch, row)
    session.submit("the request")
    for index, data in enumerate(startup.reads):
        session._on_output(data)
        if startup.live_at is not None and index >= startup.live_at:
            assert pty.written(1) == [pty_host.paste("the request")]
        else:
            assert (session._pasting, pty.writes) == (False, [])
    assert session._ready


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

    Its ``write`` takes the whole of what it is given and appends it, decoded, to ``writes``.
    """

    def __init__(self, *, survives: bool) -> None:
        self._alive = True
        self._survives = survives
        self.terminate_calls = 0
        self.writes: list[str] = []
        self._written = threading.Condition()

    def isalive(self) -> bool:
        return self._alive

    def write(self, data: bytes) -> int:
        with self._written:
            self.writes.append(data.decode("utf-8"))
            self._written.notify_all()
        return len(data)

    def written(self, count: int) -> list[str]:
        """Return ``writes`` once it holds ``count`` entries, waiting up to thirty seconds."""
        with self._written:
            assert self._written.wait_for(
                lambda: len(self.writes) >= count, timeout=30
            ), self.writes
            return list(self.writes)

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


def test_create_session_registers_a_survivor_and_answers_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """A session whose start line fails and whose process survives is registered and answers 503.

    The start's audit line fails after the process spawned; the spawned process survives its
    own termination attempt, so the session stays registered and alive and the answer says it
    stays attached.
    """
    stub = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, stub)
    refuse_audit_appends(monkeypatch)

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
    """A process whose first write sets ``writing`` and waits for ``release`` before it is taken.

    It stops on ``terminate``, and a resize sets ``resized``.
    """

    def __init__(self) -> None:
        super().__init__(survives=False)
        self.writing = threading.Event()
        self.release = threading.Event()
        self.resized = threading.Event()

    def write(self, data: bytes) -> int:
        if not self.writing.is_set():
            self.writing.set()
            assert self.release.wait(timeout=30)
        return super().write(data)

    def resize(self, rows: int, cols: int) -> None:
        self.resized.set()


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

    session.submit("first")
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


def _pasting(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    held: _HeldWritePty,
    *requests: str,
) -> terminal_routes.TerminalSession:
    """Return a session on ``held`` whose first of ``requests`` is in a paste held mid-write.

    The requests were queued, then the session's output turned its composer live, so its writer
    is writing the first one's paste when this returns.
    """
    _wire_stub_spawn(monkeypatch, held)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    for text in requests:
        session.submit(text)
    session._on_output("\x1b[?2004h" + pty_host.PROVIDERS[0].composer_ready)
    assert held.writing.wait(timeout=30)
    return session


class _MidPasteWritePty(_HeldWritePty):
    """A process whose ``write`` takes one byte per call, its second call held for ``release``.

    ``calls`` counts every write begun.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def write(self, data: bytes) -> int:
        self.calls += 1
        if self.calls == 2:
            self.writing.set()
            assert self.release.wait(timeout=30)
        return _StubPty.write(self, data[:1])


def _writer_of(session: terminal_routes.TerminalSession) -> threading.Thread:
    """Return the thread writing ``session``'s input."""
    (writer,) = [
        t
        for t in threading.enumerate()
        if t.name == f"term-{session.id}-input"
    ]
    return writer


def test_an_end_during_a_paste_answers_at_once_and_its_writer_writes_nothing_more(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    opened_project: Path,
) -> None:
    """An End while a paste is held mid-payload answers within ten seconds, the paste undelivered.

    The process is terminated, and the answer and the end line both hold the held request and
    the one queued behind it. Once the held write is released the writer stops, beginning no
    further write, and the rest of the paste it held is dropped with no report from a
    finalizer, its objects collected. Ten seconds is far above the end's own work and below the
    stub's thirty-second hold.
    """
    reports: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", reports.append)
    pty = _MidPasteWritePty()
    session = _pasting(client, monkeypatch, pty, "held", "behind")
    writer = _writer_of(session)
    with ThreadPoolExecutor(max_workers=1) as pool:
        ending = pool.submit(
            _post,
            f"/api/terminal/sessions/{session.id}/end",
            {"user": "tester"},
        )
        try:
            answer = ending.result(timeout=10).json()
        finally:
            pty.release.set()
    writer.join(timeout=30)
    gc.collect()

    assert reports == []
    assert answer == {"undelivered": ["held", "behind"]}
    assert pty.terminate_calls == 1 and not pty.isalive()
    (line,) = audit_rows(opened_project, "agent_terminal_ended")
    assert line["arguments"]["undelivered"] == answer["undelivered"]
    assert not writer.is_alive()
    assert pty.calls == 2


@pytest.mark.parametrize(
    ("stub", "taken"),
    [
        (_MidPasteWritePty, ["\x1b", "["]),
        (_HeldWritePty, [pty_host.paste("held")]),
    ],
    ids=["mid-paste", "last-write"],
)
def test_a_paste_released_before_an_end_records_stays_among_its_undelivered(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stub: type[_HeldWritePty],
    taken: list[str],
) -> None:
    """A held paste released while an End reads its hand-off stays among its undelivered.

    Held mid-payload, the writer begins no write after the held one; held at its last write,
    the paste is written whole and still not taken off the queue. Either way the writer has
    stopped by the time the End holds what its agent was never given.
    """
    held = stub()
    session = _pasting(client, monkeypatch, held, "held")
    writer = _writer_of(session)

    def released() -> None:
        held.release.set()
        writer.join(timeout=30)

    assert session.end("the person ended it", "user:tester", released)

    assert session.ended.undelivered == ["held"]
    assert not writer.is_alive()
    assert held.writes == taken


class _OneBytePty(_StubPty):
    """A process whose ``write`` takes one byte per call, noting ``queued`` at each call."""

    def __init__(self) -> None:
        super().__init__(survives=False)
        self.queued: list[str] = []
        self.queued_at: list[list[str]] = []

    def write(self, data: bytes) -> int:
        self.queued_at.append(list(self.queued))
        return super().write(data[:1])


def test_a_paste_leaves_the_queue_only_once_all_of_it_reached_the_pty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PTY taking one byte per write is given each paste whole, in order.

    The first request is still queued at every write its paste takes, and has left the queue by
    the first write of the next request's paste.
    """
    pty = _OneBytePty()
    _wire_stub_spawn(monkeypatch, pty)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    pty.queued = session._queued
    session.submit("the request")
    session.submit("next")

    session._on_output("\x1b[?2004h" + pty_host.PROVIDERS[0].composer_ready)

    first, second = pty_host.paste("the request"), pty_host.paste("next")
    assert "".join(pty.written(len(first) + len(second))) == first + second
    assert pty.queued_at == [["the request", "next"]] * len(first) + [
        ["next"]
    ] * len(second)


class _Descriptors:
    """Stands in for ``os`` and ``pty`` in the PTY module, its descriptors numbers open or closed.

    Each write is noted with its number and whether that number was open. The first write sets
    ``writing``, waits for ``release`` and takes one byte; every later one takes all it is given.
    ``openpty`` and ``dup`` open numbers none of ``descriptors`` holds.
    """

    def __init__(self, *descriptors: int) -> None:
        self.open = set(descriptors)
        self.writes: list[tuple[int, bool]] = []
        self.writing = threading.Event()
        self.release = threading.Event()
        self._numbers = itertools.count(max(descriptors, default=2) + 1)

    def openpty(self) -> tuple[int, int]:
        return self.dup(0), self.dup(0)

    def dup(self, fd: int) -> int:
        number = next(self._numbers)
        self.open.add(number)
        return number

    def write(self, fd: int, data: bytes) -> int:
        self.writes.append((fd, fd in self.open))
        if self.writing.is_set():
            return len(data)
        self.writing.set()
        assert self.release.wait(timeout=30)
        return 1

    def close(self, fd: int) -> None:
        self.open.discard(fd)


def test_a_partial_write_outstanding_at_the_readers_close_never_lands_on_the_closed_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write still under way when the reader closes the master ends on the writer's own number.

    The POSIX adapter, which cannot spawn here, is given the two descriptors its ``__init__``
    holds: the master and the writer's duplicate of it. The reader's close lands while the first
    write, which takes one byte, is held; every write, the rest of the payload included, lands
    on a number still open, and the writer closes its own once it stops.
    """
    fds = _Descriptors(3, 4)
    monkeypatch.setattr(pty_host, "os", fds)
    adapter = pty_host._PosixPty.__new__(pty_host._PosixPty)
    adapter._master, adapter._input = 3, 4
    inputs: Queue[pty_host.Input] = Queue()
    written = threading.Event()
    pty_host.start_writer(adapter, inputs, name="term-descriptors-input")
    (writer,) = [
        t for t in threading.enumerate() if t.name == "term-descriptors-input"
    ]
    inputs.put(("ab", written.set))
    assert fds.writing.wait(timeout=30)

    adapter.close()
    fds.release.set()
    assert written.wait(timeout=30)
    inputs.shutdown(immediate=True)
    writer.join(timeout=30)

    assert fds.writes == [(4, True), (4, True)]
    assert fds.open == set()


def test_a_posix_spawn_that_fails_leaves_none_of_its_descriptors_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A POSIX launch whose spawn raises closes its master, its slave and the writer's duplicate.

    The platform guard, the pseudo-terminal and the window size are stood in for, since the
    adapter cannot spawn here.
    """

    def refused(*args: object, **kwargs: object) -> None:
        raise OSError("the spawn was refused")

    fds = _Descriptors()
    monkeypatch.setattr(pty_host, "os", fds)
    monkeypatch.setitem(sys.modules, "pty", fds)
    monkeypatch.setattr(pty_host, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(pty_host, "_set_winsize", lambda *args: None)
    monkeypatch.setattr(pty_host, "subprocess", SimpleNamespace(Popen=refused))

    with pytest.raises(OSError, match="the spawn was refused"):
        pty_host._PosixPty(["agent"], "/", 24, 80, {})

    assert fds.open == set()


def test_keystrokes_the_socket_takes_during_a_paste_reach_the_pty_after_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keystrokes sent while a paste is being written reach the PTY after the whole paste.

    The writer writes its inputs in the order it took them: the paste, then the keystrokes the
    socket took while the paste's write was held, which the resize served after them shows.
    """
    held = _HeldWritePty()
    session = _pasting(client, monkeypatch, held, "pasted")
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{session.id}"
    ) as ws:
        try:
            ws.send_json({"type": "input", "data": "typed"})
            ws.send_json({"type": "resize", "rows": 40, "cols": 120})
            assert held.resized.wait(timeout=30)
        finally:
            held.release.set()

    assert held.written(2) == [pty_host.paste("pasted"), "typed"]


def test_a_keystroke_whose_write_is_held_leaves_the_socket_serving_its_frames(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a keystroke's write is held, a resize sent after it is served within ten seconds."""
    held = _HeldWritePty()
    _wire_stub_spawn(monkeypatch, held)
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        try:
            ws.send_json({"type": "input", "data": "typed"})
            assert held.writing.wait(timeout=30)
            ws.send_json({"type": "resize", "rows": 40, "cols": 120})
            assert held.resized.wait(timeout=10)
        finally:
            held.release.set()


def test_output_past_the_scrollback_cap_keeps_its_newest_characters_for_a_reconnect(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output past ``SCROLLBACK_MAX_CHARS`` keeps exactly its newest that many characters.

    A socket opened afterwards is replayed them first.
    """
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    cap = terminal_routes.SCROLLBACK_MAX_CHARS

    session._on_output("a" * cap)
    session._on_output("tail")

    kept = "a" * (cap - len("tail")) + "tail"
    assert session.scrollback_snapshot() == kept
    with client.websocket_connect(
        f"ws://127.0.0.1/api/terminal/ws/{sid}"
    ) as ws:
        assert ws.receive_text() == kept


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
    survivor = _StubPty(survives=True)
    _wire_stub_spawn(monkeypatch, survivor)
    with monkeypatch.context() as refusing:
        refuse_audit_appends(refusing)
        created = client.post("/api/terminal/sessions", json=LAUNCH)
    assert created.status_code == 503
    (sid,) = terminal_routes._SESSIONS
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
    opened_project: Path,
) -> None:
    _wire_stub_spawn(monkeypatch, _StubPty(survives=False))
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    client.post(
        f"/api/terminal/sessions/{sid}/submit", json={"text": "never given"}
    )
    refuse_audit_appends(monkeypatch)

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
    opened_project: Path,
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
    refuse_audit_appends(monkeypatch)
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


def test_shutdown_all_terminates_process(client: TestClient) -> None:
    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    session = terminal_routes._SESSIONS[sid]
    assert session.alive()
    pid = session._pty.pid  # capture before shutdown nulls the pty
    terminal_routes.shutdown_all()
    with suppress(psutil.NoSuchProcess):
        psutil.Process(pid).wait(timeout=15)
    assert not psutil.pid_exists(pid)
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
    refuse_audit_appends(monkeypatch)
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


def _started(
    client: TestClient, sid: str, source: str = "startup"
) -> str | None:
    """Report a start of session ``sid`` by ``source`` as its hook does; return its orientation."""
    resp = client.post(
        f"/api/terminal/sessions/{sid}/session-start",
        json={"conversation": "conv-start", "source": source},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["text"]


def test_a_session_start_is_answered_the_orientation_of_the_sessions_own_project(
    client: TestClient,
    opened: OpenProject,
) -> None:
    """A session's start is answered the orientation of the project it launched for.

    After the GUI opened another project, the session's start is still told its own project,
    as :func:`orientation` tells it; a launch's answer names no orientation.
    """
    created = client.post("/api/terminal/sessions", json=LAUNCH).json()
    assert set(created) == {"session_id", "launched"}
    other = named_project(opened.root.parent / "hill_block", "Hill block")
    resp = client.post(
        "/api/projects/open", json={"id": other.id, "user": "grower"}
    )
    assert resp.status_code == 200, resp.text

    told = _started(client, created["session_id"])

    assert told == pty_host.orientation(opened, "startup")
    assert told is not None and "Test project" in told
    assert "Hill block" not in told


def test_a_session_launched_for_no_project_stays_one_after_its_agent_creates_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session launched for no project is told a fact that still holds once its agent made one.

    The project is created and written to through the MCP server's own creation and project
    adapters, the server's identity begun under the session's id as the launch declares it; the
    session then holds no project, and its compaction is told it was launched with none, never
    that it has none.
    """
    import anyio

    from tcip_mcp import agent_identity
    from tcip_mcp.server import Session, build_server
    from tcip_mcp.tools.project_tools import initialize_project
    from tcip_web.state import store
    from tests._web_fixtures import INVARIANCES

    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    before = _started(client, sid, "compact")
    assert before is not None
    monkeypatch.setenv(agent_identity.TERMINAL_SESSION_ENV, sid)
    server = build_server(Session())
    agent_identity.begin("a-harness", "1.0")
    try:
        anyio.run(
            server.call_tool,
            initialize_project.__name__,
            {
                "project_path": str(store.workspace / "created"),
                "display_name": "Created block",
                "site": "south orchard",
                "invariances": INVARIANCES.model_dump(),
            },
        )
        anyio.run(
            server.call_tool,
            "report_friction",
            {"category": "missing_tool", "detail": "written after creation"},
        )
    finally:
        agent_identity.end()

    after = _started(client, sid, "compact")

    assert after == before
    assert after is not None
    assert "launched with no project open" in after
    assert "has no project" not in after
    assert terminal_routes._SESSIONS[sid]._project is None


def test_a_launch_after_a_rename_of_the_open_project_names_its_new_name(
    client: TestClient, opened: OpenProject
) -> None:
    """A launch after a rename of the open project names its new name.

    After a rename made while the project is open, the next session's orientation states the
    new name.
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
    told = _started(client, created.json()["session_id"])
    assert told is not None and '"display_name": "Ridge block"' in told


def test_the_orientation_of_a_project_whose_record_will_not_read_states_why(
    client: TestClient,
    opened: OpenProject,
) -> None:
    """A session for a project whose record will not read is told why, in the overview."""
    from tcip_mcp.project_record import project_record_key, record_fields
    from tests._record_damage_fixtures import damage_record

    damage_record(project_record_key(str(opened.root)), b"{not valid json")
    problem = record_fields(opened.root)["record_problem"]
    assert problem is not None

    created = client.post("/api/terminal/sessions", json=LAUNCH)

    assert created.status_code == 200, created.text
    told = _started(client, created.json()["session_id"])
    assert told is not None
    assert f'"record_problem": {json.dumps(problem)}' in told


# ── resuming a row's conversation ───────────────────────────────────────


@pytest.fixture
def fake_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, pty_host.Provider]:
    """Install every provider row with the fake program as its executable; return them by id.

    Antigravity's settings file is a scratch one, and a port is recorded for its session hook.
    """
    monkeypatch.setattr(
        pty_host, "ANTIGRAVITY_SETTINGS", tmp_path / "agy-settings.json"
    )
    _port_recorded()
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
        f"/api/terminal/sessions/{sid}/session-start",
        json={"conversation": conversation, "source": "startup"},
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
    marker, its start line names the conversation, and a request staged for it is delivered
    alone.
    """
    row = fake_rows["claude"]
    _launch(client, row)
    _talked_then_exited(client, "conv-1")

    resumed = _launch(client, row)

    assert resumed["launched"]["resumed"] == "conv-1"
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


def test_a_launch_of_another_row_never_resumes_the_first_rows_conversation(
    client: TestClient,
    opened_project: Path,
    fake_rows: dict[str, pty_host.Provider],
) -> None:
    _launch(client, fake_rows["claude"])
    _talked_then_exited(client, "conv-1")

    other = _launch(client, fake_rows["codex"])

    assert other["launched"]["resumed"] is None


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


def test_a_report_for_a_session_the_backend_does_not_hold_answers_404(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/terminal/sessions/term_gone/session-start",
        json={"conversation": "conv-1", "source": "startup"},
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
    assert pty.written(1) == [pty_host.paste(terminal_routes.HAND_OFF_REQUEST)]

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

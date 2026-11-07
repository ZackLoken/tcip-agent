"""Live smoke of one provider row's real harness through the embedded agent terminal.

Runs the smoke ``packages/tcip-web/CLAUDE.md`` states for the row, in the workspace
``--workspace`` names or a fresh one, and exits 0 on a pass and 1 otherwise. What only the real
harness decides is checked here, each check a printed line with its result: that the harness
read a workspace instruction file (Claude Code's ``CLAUDE.md``, Codex's discovery of ``AGENTS.md``
in a workspace that is no git root, Antigravity's reading of a workspace ``AGENTS.md``), shown by
a report carrying the token only those files state, and that the row's rendered instruction file
is unchanged after the agent was asked to edit it. For Claude Code's row the harness's own
transcript must also record, after the edit request was submitted, the edit attempted and then
denied by its permission settings, and the same holds for a scratch file the smoke creates outside
the workspace; for the other rows, whose transcripts the smoke does not read, the unchanged file
cannot tell a denied edit from one never tried.

Usage (from the repo root, tcip-agent env):
    python tools/smoke_terminal_e2e.py <provider id> [--workspace <trusted workspace>]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.testclient import WebSocketTestSession

    from tcip_web.terminal import Provider

sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src")
)
sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src")
)

ANSWER_TIMEOUT_S = 240


def terminal_ws_url(client: TestClient, path: str) -> str:
    """Return the absolute ``ws://`` URL of the socket ``path`` on ``client``'s host."""
    parts = urlsplit(str(client.base_url))
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


REQUESTED_CATEGORY = "unexpected_behavior"
"""The ``report_friction`` category the request names, one of that tool's own categories."""

INSTRUCTION_TOKEN = "The terminal smoke's instruction token is {token}."
"""The line the smoke adds to the workspace's instruction files, outside the contract's block."""


def request_for(token: str) -> str:
    """Return the request asking for one ``report_friction`` call with ``token`` and two facts.

    They are the open project's display name and the instruction token, neither of which the
    request states.
    """
    return (
        "This is a smoke test of the agent terminal. Without calling any other tool, call "
        f"the tcip report_friction tool exactly once, with category {REQUESTED_CATEGORY} and, "
        f"as detail, {token} followed by this session's project's display name and then the "
        "terminal smoke's instruction token your instruction file states, then stop."
    )


def edit_request(token: str, path: Path) -> str:
    """Return the request asking the agent to try once to edit ``path``, then to report ``token``.

    For a file that exists the edit appends the line ``token``; for one that does not, it creates
    the file holding that line, since Claude Code's ``Edit`` and ``Write`` tools refuse an existing
    file the agent has not read before they check where it lies. The report says only that the
    agent is done; whether an edit was attempted and denied is read from the harness's own record
    (:func:`edit_answers`, :func:`denial`) and the file's bytes.
    """
    action = (
        f"append the line {token} to the file"
        if path.exists()
        else f"create the file, holding the line {token},"
    )
    return (
        "This is a smoke test of the agent terminal's confinement. Try once to "
        f"{action} {path.as_posix()} with your own file-editing tool, without reading it first, "
        "then, without calling any other tool, call the tcip report_friction tool exactly once, "
        f"with category {REQUESTED_CATEGORY} and, as detail, {token}, then stop."
    )


TRANSCRIPT_ROW = "claude"
"""The row whose harness's transcript the smoke reads: Claude Code's JSONL record of a
conversation, which its ``SessionStart`` hook input names as ``transcript_path``."""

EDITING_TOOLS = ("Edit", "Write")
"""The Claude Code tools whose use names the file it edits as ``input.file_path``, the two
built-in file-editing tools an ``Edit`` permission rule covers (Claude Code's permissions page)."""

DENIED_BY_SETTINGS = "denied by your permission settings"
"""The text of the error result Claude Code 2.1.316 records for an edit its permission settings
deny, in an entry carrying no ``toolDenialKind``; a denial by a rule may instead carry a non-null
``toolDenialKind``, and the check admits either."""


def transcript_length(transcript: Path) -> int:
    """Return how many bytes ``transcript`` holds, none when the harness has not yet written it."""
    return transcript.stat().st_size if transcript.is_file() else 0


def edit_answers(
    transcript: Path, start: int, target: Path
) -> list[tuple[dict, dict]]:
    """Return each tool result the Claude Code transcript ``transcript`` records for an edit.

    Only the entries written past its first ``start`` bytes are read, in the order written. The
    edits are the uses of one of :data:`EDITING_TOOLS` whose ``file_path`` is ``target``; a result
    answers one when it carries that use's ``tool_use_id`` and was written after it. Each answer
    is the transcript entry holding the result and the result itself.
    """
    items = [
        (entry, item)
        for entry in map(
            json.loads, transcript.read_bytes()[start:].decode().splitlines()
        )
        if isinstance(entry.get("message"), dict)
        and isinstance(entry["message"].get("content"), list)
        for item in entry["message"]["content"]
    ]
    attempts: set[str] = set()
    answers: list[tuple[dict, dict]] = []
    for entry, item in items:
        if (
            item.get("type") == "tool_use"
            and item.get("name") in EDITING_TOOLS
            and Path(item["input"]["file_path"]).resolve() == target.resolve()
        ):
            attempts.add(item["id"])
        elif (
            item.get("type") == "tool_result"
            and item.get("tool_use_id") in attempts
        ):
            answers.append((entry, item))
    return answers


def denial(entry: dict, result: dict) -> bool:
    """Return whether ``result``, held by the transcript ``entry``, is Claude Code's denial.

    It is an error result in an entry whose ``toolDenialKind`` is not null, or one stating
    :data:`DENIED_BY_SETTINGS`.
    """
    return result.get("is_error") is True and (
        entry.get("toolDenialKind") is not None
        or DENIED_BY_SETTINGS in str(result.get("content"))
    )


def rendered(workspace: Path, row: Provider) -> bool:
    """Return whether ``workspace`` holds ``row``'s rendered contract and every knowledge skill.

    The instruction file must carry the contract document's text and each skill must be its
    renderer's text; one line per check is printed with its result.
    """
    from tcip_mcp.knowledge import list_documents, read_document
    from tcip_mcp.knowledge.harness_files import CONTRACT, render_skill

    instructions = workspace / row.files.instructions
    contract = instructions.is_file() and read_document(
        CONTRACT
    ) in instructions.read_text(encoding="utf-8")
    skills = workspace / row.files.skills
    every_skill = all(
        (skills / d.name / "SKILL.md").is_file()
        and (skills / d.name / "SKILL.md").read_text(encoding="utf-8")
        == render_skill(d)
        for d in list_documents()
    )
    print(f"    the operating contract in {instructions}: {contract}")
    print(f"    every knowledge skill under {skills}: {every_skill}")
    return contract and every_skill


CONTINUED_SUFFIX = "-continued"
"""What the resumed agent adds to the token of its earlier report."""


def continuation_request() -> str:
    """Return the request asking a resumed harness to report its earlier token, continued.

    It names no token; the token is the first word of the detail :func:`request_for` asks for.
    """
    return (
        "This is a smoke test of a resumed conversation. Without calling any other tool, call "
        f"the tcip report_friction tool exactly once, with category {REQUESTED_CATEGORY} and, "
        "as detail, the first word of the detail of the first report_friction call you made "
        f"in this conversation followed at once by {CONTINUED_SUFFIX}, then stop."
    )


def reported(client: TestClient, token: str, *facts: str) -> bool:
    """Return whether exactly one friction report of the open project carries ``token``.

    The token is looked for in its detail, and that report must have the requested category
    and carry each of ``facts`` in its detail too.
    """
    entries = client.get("/api/meta/reports").json()["entries"]
    matching = [
        entry["record"]
        for entry in entries
        if "reason" not in entry["record"]
        and token in entry["record"]["detail"]
    ]
    return (
        len(matching) == 1
        and matching[0]["category"] == REQUESTED_CATEGORY
        and all(fact in matching[0]["detail"] for fact in facts)
    )


def keep_held(ws: WebSocketTestSession, held: list[int]) -> None:
    """Append to ``held`` the count of every held frame ``ws`` serves, until the socket ends."""
    from anyio import ClosedResourceError, EndOfStream
    from fastapi import WebSocketDisconnect
    from pydantic import TypeAdapter

    from tcip_web.routes.terminal import TerminalServedFrame

    frames: TypeAdapter[TerminalServedFrame] = TypeAdapter(TerminalServedFrame)
    with suppress(WebSocketDisconnect, EndOfStream, ClosedResourceError):
        while True:
            frame = frames.validate_json(ws.receive_text())
            if frame.type == "held":
                held.append(frame.count)


@contextmanager
def planted(workspace: Path, token: str) -> Iterator[None]:
    """Add the instruction token's line to the workspace's instruction files until the block ends.

    The files are Claude Code's ``CLAUDE.md`` and the ``AGENTS.md`` Codex reads; each is restored
    to its earlier text afterwards, or removed when it had none.
    """
    from tcip_mcp.knowledge.harness_files import (
        AGENTS,
        CLAUDE_CODE,
        write_text,
    )

    before = {
        path: path.read_text(encoding="utf-8") if path.is_file() else None
        for path in (workspace / f.instructions for f in (CLAUDE_CODE, AGENTS))
    }
    for path, text in before.items():
        write_text(
            path, f"{text or ''}{INSTRUCTION_TOKEN.format(token=token)}\n"
        )
    try:
        yield
    finally:
        for path, text in before.items():
            if text is None:
                path.unlink(missing_ok=True)
            else:
                write_text(path, text)


SERVE_TIMEOUT_S = 10
"""Provisional: seconds the served backend is given to start listening."""


@contextmanager
def served(app: FastAPI, workspace: Path) -> Iterator[None]:
    """Serve ``app`` on a free loopback port recorded for ``workspace``, until the block ends.

    The port is :func:`~tcip_mcp.web_client.free_port`'s, recorded as ``python -m tcip_web``
    records it, so any process under the workspace (a harness's hook, an MCP server's push)
    reaches this backend through ``backend_url``. Raises ``RuntimeError`` when the server does not
    start listening within :data:`SERVE_TIMEOUT_S`.
    """
    import uvicorn

    from tcip_mcp.web_client import (
        LOOPBACK_HOST,
        free_port,
        record_backend_port,
    )

    port = free_port(0)
    record_backend_port(workspace, port)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=LOOPBACK_HOST,
            port=port,
            log_level="error",
            lifespan="off",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + SERVE_TIMEOUT_S
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError(f"the backend did not listen on port {port}")
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=SERVE_TIMEOUT_S)


def main(
    provider: str,
    workspace: str | None = None,
    answer_timeout_s: float = ANSWER_TIMEOUT_S,
) -> int:
    """Drive the smoke for the ``provider`` row and return 0 on a pass.

    ``TCIP_WORKSPACE`` is set to ``workspace`` (a fresh temp directory when omitted) and the
    backend started on it and :func:`served` for the harness's own processes, waiting
    ``answer_timeout_s`` for each report.
    """
    os.environ["TCIP_WORKSPACE"] = workspace or tempfile.mkdtemp(
        prefix="terminal-smoke-ws-"
    )

    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import workspace_from_environment
    from tcip_web.app import app
    from tcip_web.state import store

    store.configure(workspace_from_environment(), ())
    client = TestClient(app, base_url="http://127.0.0.1")
    with served(app, store.workspace):
        return _smoke(client, provider, answer_timeout_s)


def _smoke(client: TestClient, provider: str, answer_timeout_s: float) -> int:
    """Run the smoke for the ``provider`` row through ``client``; return 0 on a pass."""
    from tcip_mcp.project_record import CaptureInvariances
    from tcip_mcp.tools.project_tools import initialize_project
    from tcip_web.app import app
    from tcip_web.routes import terminal as terminal_routes
    from tcip_web.routes.terminal import (
        RecordedSession,
        TerminalSession,
        TerminalStatus,
    )
    from tcip_web.state import store
    from tcip_web.terminal import Resumed

    status = TerminalStatus.model_validate(
        client.get("/api/terminal/status").json()
    )
    print(f"[1] status: {status}")
    rows = {row.id: row for row in status.providers}
    if provider not in rows or rows[provider].kind != "launchable":
        print(
            f"FAIL: provider {provider!r} cannot launch here: {rows.get(provider, rows)}"
        )
        return 1

    project = tempfile.mkdtemp(
        prefix="terminal-smoke-project-", dir=store.workspace
    )
    # A sample answer for the scratch project; nothing in the smoke trains under it.
    invariances = CaptureInvariances(
        horizontal_flip=True,
        vertical_flip=True,
        rotation=180,
        color=True,
        intensity=True,
        scale=True,
    )
    name = f"Terminal smoke {secrets.token_hex(4)}"
    created = initialize_project(
        project,
        name,
        "the terminal smoke's scratch workspace",
        invariances,
        actor=None,
    )
    if "error" in created:
        print(f"FAIL: the scratch project was refused: {created['error']}")
        return 1
    user = getpass.getuser()
    opened = client.post(
        "/api/projects/open",
        json={"id": created["record"]["id"], "user": user},
    )
    if opened.status_code != 200:
        print(
            f"FAIL: the scratch project did not open ({opened.status_code}): {opened.text}"
        )
        return 1
    print(f"[2] project opened: {opened.json()}")
    token = f"ack-{secrets.token_hex(6)}"

    def launched() -> TerminalSession | None:
        """Launch the row by ``user``; return its session, or ``None`` printing the refusal."""
        resp = client.post(
            "/api/terminal/sessions",
            json={"provider": provider, "rows": 35, "cols": 120, "user": user},
        )
        if resp.status_code != 200:
            print(
                f"FAIL: the launch was refused ({resp.status_code}): {resp.text}"
            )
            return None
        created = resp.json()
        print(
            f"    session spawned: {created['session_id']} {created['launched']}"
        )
        return terminal_routes._SESSIONS[created["session_id"]]

    def answered(
        session: TerminalSession, request: str, signal: str, *facts: str
    ) -> bool:
        """Submit ``request`` to ``session``; return whether ``signal`` with ``facts`` came in time.

        The signal and facts are what :func:`reported` looks for.
        """
        started = time.time()
        socket_path = app.url_path_for("terminal_ws", session_id=session.id)
        held: list[int] = []
        with client.websocket_connect(
            terminal_ws_url(client, socket_path)
        ) as ws:
            threading.Thread(
                target=keep_held, args=(ws, held), daemon=True
            ).start()
            ws.send_json(
                {"type": "input", "data": "\x1b[?1;2c"}
            )  # a terminal's DA reply
            client.post(
                f"/api/terminal/sessions/{session.id}/submit",
                json={"text": request},
            )
            print(
                f"    request submitted; waiting for a report of {signal}..."
            )
            while time.time() - started < answer_timeout_s:
                time.sleep(1.0)
                if reported(client, signal, *facts):
                    return True
            print(f"    requests the socket last counted held: {held[-1:]}")
            print(
                f"    stream tail: {ascii(session.scrollback_snapshot()[-1500:])}"
            )
        ended = client.post(
            f"/api/terminal/sessions/{session.id}/end", json={"user": user}
        )
        print(f"    ended, the agent never given: {ended.json()}")
        return False

    def recorded(session: TerminalSession) -> RecordedSession | None:
        """Return ``session`` as the scratch project's log records it, ``None`` before it does."""
        assert store.opened is not None
        found = terminal_routes.recorded_session(provider, store.opened)
        return found if found and found.session_id == session.id else None

    row = terminal_routes._provider(provider)

    def attempt_edit(current: TerminalSession, target: Path) -> bool:
        """Ask ``current``'s agent to edit ``target`` and return whether the edit was refused.

        It passes when the agent reports done and ``target`` is as it was (its bytes, or absent)
        and, for the row whose transcript is read, that transcript records the edit attempted and
        denied.
        """
        edit = f"edit-{secrets.token_hex(6)}"
        held = target.read_bytes() if target.exists() else None
        record = recorded(current)
        transcript = (
            None
            if record is None
            or record.report is None
            or record.report.transcript_path is None
            else Path(record.report.transcript_path)
        )
        before = (
            None
            if transcript is None
            else (transcript, transcript_length(transcript))
        )
        done = answered(current, edit_request(edit, target), edit)
        unchanged = (target.read_bytes() if target.exists() else None) == held
        print(f"    reported done: {done}; the file unchanged: {unchanged}")
        refused = done and unchanged
        if row.id == TRANSCRIPT_ROW:
            answers = [] if before is None else edit_answers(*before, target)
            for entry, result in answers:
                print(
                    f"    an edit answered: is_error {result.get('is_error')}, "
                    f"toolDenialKind {entry.get('toolDenialKind')}, "
                    f"{ascii(str(result.get('content'))[:200])}"
                )
            denied = any(denial(*answer) for answer in answers)
            print(
                f"    the harness's transcript {transcript} records the edit "
                f"attempted and denied: {denied}"
            )
            return refused and denied
        print(
            f"    {row.name} records no transcript this smoke reads; the "
            "unchanged file is the whole check"
        )
        return refused

    instruction = f"instruction-{secrets.token_hex(6)}"
    print("[3] the first launch")
    with planted(store.workspace, instruction):
        first = launched()
        if first is None:
            return 1
        try:
            files = rendered(store.workspace, row)
            passed = answered(first, request_for(token), token, name)
            print(
                f"    reported naming {name!r} from its orientation: {passed}"
            )
            read = reported(client, token, name, instruction)
            print(
                "    reported the instruction token, so the harness read a workspace "
                f"instruction file: {read}"
            )
            passed = passed and files and read
            current = first
            if passed and row.resume is not None:
                started, conversation = time.time(), None
                while (
                    conversation is None
                    and time.time() - started < answer_timeout_s
                ):
                    time.sleep(1.0)
                    record = recorded(first)
                    conversation = (
                        None
                        if record is None or record.report is None
                        else record.report.conversation
                    )
                print(f"[4] the harness reported conversation {conversation}")
                terminal_routes.shutdown_all()
                record = recorded(first)
                print(f"    the shutdown's end: {record and record.end}")
                print("[5] the launch after the shutdown")
                second = launched()
                if second is not None:
                    current = second
                passed = (
                    second is not None
                    and conversation is not None
                    and second.launched.resumed
                    == Resumed(conversation=conversation, session=first.id)
                    and answered(
                        second,
                        continuation_request(),
                        token + CONTINUED_SUFFIX,
                    )
                )
                print(
                    f"    resumed {conversation} and reported its token continued: {passed}"
                )
            if passed:
                print("[6] the edit of the row's rendered instruction file")
                passed = attempt_edit(
                    current, store.workspace / row.files.instructions
                )
            if passed and row.id == TRANSCRIPT_ROW:
                print("[6b] the creation of a file outside the workspace")
                with tempfile.TemporaryDirectory(
                    prefix="terminal-smoke-outside-"
                ) as outside:
                    passed = attempt_edit(
                        current, Path(outside) / "scratch.txt"
                    )
        finally:
            terminal_routes.shutdown_all()
    print(
        f"[7] after shutdown_all: sessions still held {len(terminal_routes._SESSIONS)}"
    )

    print("SMOKE PASS" if passed else "SMOKE FAIL")
    return 0 if passed else 1


def command(argv: list[str] | None = None) -> int:
    """Parse the smoke's command line ``argv`` and return what :func:`main` returns for it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "provider", help="the id of the provider row to launch"
    )
    parser.add_argument(
        "--workspace",
        help="the workspace to run in, one the person trusted by hand for a harness that "
        "asks to trust its workspace (default: a fresh temporary directory)",
    )
    parser.add_argument(
        "--answer-timeout",
        type=float,
        default=ANSWER_TIMEOUT_S,
        help=f"seconds to wait for the report (default {ANSWER_TIMEOUT_S})",
    )
    args = parser.parse_args(argv)
    return main(
        args.provider,
        workspace=args.workspace,
        answer_timeout_s=args.answer_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(command())

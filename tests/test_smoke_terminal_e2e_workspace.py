"""``tools/smoke_terminal_e2e.py`` runs against the workspace its command line names.

It sets ``TCIP_WORKSPACE`` to the workspace ``--workspace`` names and opens nothing for a row that
cannot launch.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from tcip_web import terminal as pty_host
from tests.fake_terminal_app import transcript_entry
from tools import smoke_terminal_e2e

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from fastapi.testclient import TestClient


def test_runs_under_the_given_workspace_and_opens_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The smoke binds the workspace ``--workspace`` names and opens nothing for a failed row."""
    from tcip_web.state import store

    monkeypatch.delenv(pty_host.TERMINAL_CMD_ENV, raising=False)
    row = dataclasses.replace(
        pty_host.PROVIDERS[0], executable="tcip-smoke-test-nonexistent-cli"
    )
    monkeypatch.setattr(pty_host, "PROVIDERS", (row,))

    scratch_ws = tmp_path.parent.parent / "scratch-workspace"
    result = smoke_terminal_e2e.command(
        [row.id, "--workspace", str(scratch_ws)]
    )

    assert result == 1
    assert os.environ["TCIP_WORKSPACE"] == str(scratch_ws)
    assert store.workspace == scratch_ws.resolve()
    assert store.opened is None


def test_the_signal_is_a_friction_report_of_the_open_project_carrying_the_token(
    opened_project: Path, opened_client: TestClient
) -> None:
    """The smoke signal is one friction report of the open project carrying the token.

    ``reported`` answers only while ``report_friction`` has recorded, in the open project,
    exactly one report whose detail carries the token, of the requested category: the token
    anywhere else, a report of another category, or a second call is not it.
    """
    from tcip_mcp.tools.meta_tools import report_friction

    token = "ack-0123456789ab"
    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        "an unrelated report",
        context={"note": token},
        actor=None,
    )
    assert not smoke_terminal_e2e.reported(opened_client, token)

    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        f"smoke {token}",
        actor=None,
    )
    assert smoke_terminal_e2e.reported(opened_client, token)

    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        f"again {token}",
        actor=None,
    )
    assert not smoke_terminal_e2e.reported(opened_client, token)

    other = "ack-ba9876543210"
    report_friction(opened_project, "missing_tool", other, actor=None)
    assert not smoke_terminal_e2e.reported(opened_client, other)


def test_the_first_signal_carries_the_display_name_its_request_never_states(
    opened_project: Path, opened_client: TestClient
) -> None:
    """The first launch's signal carries the project's display name, which its request omits.

    A report of the token alone is not the signal once the name is asked for; a report of the
    token followed by the name is.
    """
    from tcip_mcp.tools.meta_tools import report_friction

    name = "Terminal smoke 0a1b2c3d"
    alone, named = "ack-0123456789ab", "ack-ba9876543210"
    assert alone in smoke_terminal_e2e.request_for(alone)
    assert name not in smoke_terminal_e2e.request_for(alone)

    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        f"smoke {alone}",
        actor=None,
    )
    assert smoke_terminal_e2e.reported(opened_client, alone)
    assert not smoke_terminal_e2e.reported(opened_client, alone, name)

    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        f"smoke {named} {name}",
        actor=None,
    )
    assert smoke_terminal_e2e.reported(opened_client, named, name)


def test_the_smoke_against_a_harness_no_deny_list_binds_fails_on_the_edit_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Against the fake harness standing in for Claude's row, every check passes but the edit's.

    The fake runs its launch's ``SessionStart`` hook, whose command reaches the backend the smoke
    serves, reports the token with the display name the hook's orientation states and the
    instruction token its workspace ``CLAUDE.md`` states, then, resumed after the backend's
    shutdown, the token continued from its own memory of the conversation; the second launch
    resumed the conversation the hook reported. Asked to edit its ``CLAUDE.md``, it appends to it
    through its own process, which no deny list binds, and its transcript records the edit made:
    the smoke finds the file changed and no denial in the transcript the hook named, and fails.
    """
    from tests._terminal_fixtures import fake_executable

    monkeypatch.delenv(pty_host.TERMINAL_CMD_ENV, raising=False)
    (claude,) = [row for row in pty_host.PROVIDERS if row.id == "claude"]
    row = dataclasses.replace(
        claude, executable=str(fake_executable(tmp_path / "bin"))
    )
    monkeypatch.setattr(pty_host, "PROVIDERS", (row,))
    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.setenv("FAKE_TERMINAL_MEMORY_DIR", str(memory))
    monkeypatch.setenv("FAKE_TERMINAL_PASTE_AFTER_S", "1")
    monkeypatch.setenv("FAKE_TERMINAL_COMPOSER_READY", row.composer_ready)

    result = smoke_terminal_e2e.main(
        row.id,
        workspace=str(tmp_path.parent.parent / "smoke-workspace"),
        answer_timeout_s=90,
    )

    checks = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.endswith((": True", ": False"))
    ]
    assert result == 1
    assert len(checks) == 7, checks
    assert all(check.endswith(": True") for check in checks[:5]), checks
    assert checks[5].endswith("reported done: True; the file unchanged: False")
    assert checks[6].endswith("records the edit attempted and denied: False")
    conversation, transcript = sorted(memory.iterdir())
    assert transcript.name == f"{conversation.name}.jsonl"


OBSERVED_DENIAL = (
    "File is in a directory that is denied by your permission settings"
)
"""The error text Claude Code 2.1.316 records for an edit its settings deny."""


def _append(transcript: Path, *entries: dict) -> None:
    """Append ``entries`` to ``transcript``, one JSON line each."""
    with transcript.open("a", encoding="utf-8") as file:
        file.writelines(json.dumps(entry) + "\n" for entry in entries)


def _edit(target: Path, use: str) -> dict:
    """Return the transcript entry of the ``Edit`` use ``use`` of ``target``."""
    return transcript_entry(
        "assistant",
        {
            "type": "tool_use",
            "id": use,
            "name": "Edit",
            "input": {"file_path": str(target)},
        },
    )


def _answer(use: str, content: str, **fields: object) -> dict:
    """Return the transcript entry of the error result ``content`` of ``use``."""
    return transcript_entry(
        "user",
        {
            "type": "tool_result",
            "tool_use_id": use,
            "content": content,
            "is_error": True,
        },
        **fields,
    )


def test_only_an_edit_after_the_request_answered_after_it_is_read(
    tmp_path: Path,
) -> None:
    """The smoke reads an edit's answer past the request's boundary, written after its use.

    A denied edit written before the transcript's length was read is not the request's, and a
    result written before the use its id names answers nothing; an edit and its denial both
    past the boundary, in that order, are the one answer read.
    """
    target = tmp_path / "CLAUDE.md"
    transcript = tmp_path / "conversation.jsonl"
    _append(
        transcript,
        _edit(target, "toolu_earlier"),
        _answer("toolu_earlier", OBSERVED_DENIAL),
    )
    start = smoke_terminal_e2e.transcript_length(transcript)

    assert smoke_terminal_e2e.edit_answers(transcript, start, target) == []

    _append(
        transcript,
        _answer("toolu_reversed", OBSERVED_DENIAL),
        _edit(target, "toolu_reversed"),
    )
    assert smoke_terminal_e2e.edit_answers(transcript, start, target) == []

    _append(
        transcript,
        _edit(target, "toolu_current"),
        _answer("toolu_current", OBSERVED_DENIAL),
    )
    answers = smoke_terminal_e2e.edit_answers(transcript, start, target)
    assert [result["tool_use_id"] for _, result in answers] == [
        "toolu_current"
    ]
    assert smoke_terminal_e2e.denial(*answers[0])


def test_a_denial_is_an_error_with_a_kind_or_the_settings_text() -> None:
    """An error result is a denial when its entry's denial kind is set or it states the text.

    One whose ``toolDenialKind`` is null and states no settings text is not.
    """

    def read(entry: dict) -> bool:
        (result,) = entry["message"]["content"]
        return smoke_terminal_e2e.denial(entry, result)

    refused = "the edit was not made"
    assert not read(_answer("toolu_a", refused, toolDenialKind=None))
    assert read(_answer("toolu_a", refused, toolDenialKind="a kind"))
    assert read(_answer("toolu_a", OBSERVED_DENIAL))


def test_the_resumed_signal_is_the_token_continued_no_request_names(
    opened_project: Path, opened_client: TestClient
) -> None:
    """The resumed launch's signal is the first token continued, named by no request after it.

    The continuation request carries no token; ``reported`` answers for the continued token only
    once a report carries it, and the first token, then carried by two reports, no longer reads
    as the first launch's signal.
    """
    from tcip_mcp.tools.meta_tools import report_friction

    token = "ack-0123456789ab"
    continued = token + smoke_terminal_e2e.CONTINUED_SUFFIX
    assert "ack-" not in smoke_terminal_e2e.continuation_request()
    assert (
        smoke_terminal_e2e.CONTINUED_SUFFIX
        in smoke_terminal_e2e.continuation_request()
    )

    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        f"smoke {token}",
        actor=None,
    )
    assert smoke_terminal_e2e.reported(opened_client, token)
    assert not smoke_terminal_e2e.reported(opened_client, continued)

    report_friction(
        opened_project,
        smoke_terminal_e2e.REQUESTED_CATEGORY,
        continued,
        actor=None,
    )
    assert smoke_terminal_e2e.reported(opened_client, continued)
    assert not smoke_terminal_e2e.reported(opened_client, token)


def test_the_instruction_token_is_added_to_each_instruction_file_and_each_restored(
    tmp_path: Path,
) -> None:
    """``planted`` adds the token's line to both instruction files and restores both after.

    A file that held text gets that text back; one that did not exist is removed again.
    """
    from tcip_mcp.knowledge.harness_files import AGENTS, CLAUDE_CODE

    claude = tmp_path / CLAUDE_CODE.instructions
    agents = tmp_path / AGENTS.instructions
    claude.write_text("the person's own text\n", encoding="utf-8")
    token = "instruction-0a1b2c"
    line = smoke_terminal_e2e.INSTRUCTION_TOKEN.format(token=token)

    with smoke_terminal_e2e.planted(tmp_path, token):
        assert claude.read_text(encoding="utf-8") == (
            f"the person's own text\n{line}\n"
        )
        assert agents.read_text(encoding="utf-8") == f"{line}\n"

    assert claude.read_text(encoding="utf-8") == "the person's own text\n"
    assert not agents.exists()


def test_the_websocket_url_is_absolute_and_carries_the_served_host(
    tmp_path: Path, client: TestClient
) -> None:
    """``terminal_ws_url`` is absolute, names the client's own host and carries the socket path.

    The path is the one the app reverses for a session's socket, as the smoke produces it.
    """
    from tcip_web.app import app

    path = app.url_path_for("terminal_ws", session_id="abc123")

    url = smoke_terminal_e2e.terminal_ws_url(client, path)

    parsed = urlsplit(url)
    assert parsed.scheme == "ws"
    assert parsed.netloc == "127.0.0.1"
    assert parsed.path == path == "/api/terminal/ws/abc123"

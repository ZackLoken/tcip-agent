"""The ``session-hook`` console command, fed each harness's documented ``SessionStart`` input.

The command reports the start to the backend :func:`tools.smoke_terminal_e2e.served` serves and
prints the orientation answered, in the shape the harness reads; each row's own spelling of the
hook is run as its harness would run it.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tomllib
from PIL import Image

from tcip_mcp.agent_identity import TERMINAL_SESSION_ENV
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.subject_registry import Subject, SubjectRegistry
from tcip_mcp.tools.doctor_tools import diagnose_project
from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge
from tcip_mcp.tools.meta_tools import report_friction, write_retrospective
from tcip_web import terminal as pty_host
from tcip_web.app import app
from tcip_web.cli import session_hook
from tcip_web.cli.session_hook import session_start_url
from tcip_web.routes import terminal as terminal_routes
from tests._producer_fixtures import label_image, registry_over
from tools.smoke_terminal_e2e import served

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

pytestmark = pytest.mark.usefixtures("fake_terminal")

LAUNCH = {"provider": pty_host.PROVIDERS[0].id, "user": "tester"}


def _claude_start(conversation: str, source: str) -> dict:
    """Return Claude Code's ``SessionStart`` input, as its hooks page documents its fields."""
    return {
        "session_id": conversation,
        "transcript_path": "/home/user/.claude/projects/x/transcript.jsonl",
        "cwd": "/workspace",
        "hook_event_name": "SessionStart",
        "source": source,
        "model": "claude-opus-5",
    }


def _codex_start(conversation: str, source: str) -> dict:
    """Return Codex's ``SessionStart`` input, as its hooks page documents its fields."""
    return {
        "session_id": conversation,
        "transcript_path": None,
        "cwd": "/workspace",
        "hook_event_name": "SessionStart",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "source": source,
    }


def _agy_start(conversation: str) -> dict:
    """Return Antigravity's ``SessionStart`` input, as its hooks page documents its fields."""
    return {
        "conversationId": conversation,
        "workspacePaths": ["/workspace"],
        "transcriptPath": "~/.gemini/antigravity-cli/brain/x/transcript.jsonl",
        "artifactDirectoryPath": "~/.gemini/antigravity-cli/brain/x",
        "modelName": "gemini-3.6-flash-medium",
    }


def _additional_context(printed: str) -> str:
    """Return the ``additionalContext`` Claude Code and Codex read from a hook's stdout."""
    output = json.loads(printed)["hookSpecificOutput"]
    assert output["hookEventName"] == "SessionStart"
    return output["additionalContext"]


def _injected(printed: str) -> str:
    """Return the system message Antigravity injects from a ``SessionStart`` hook's stdout."""
    (step,) = json.loads(printed)["injectSteps"]
    return step["systemMessage"]["systemMessage"]


@pytest.fixture
def remembered(opened: OpenProject) -> OpenProject:
    """Give the open project a report, a retrospective and one data-state finding.

    The finding is an empty label document no mark finishes, under capture ``2026-03-04``.
    """
    root = opened.root
    report_friction(
        root, "missing_tool", "the newest report detail", actor=None
    )
    write_retrospective(
        root,
        "orientation-task",
        "what worked",
        "what did not",
        "nothing",
        actor=None,
    )
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="bud", description=""),))
    )
    image = image_dir(root, "2026-03-04") / "IMG_E.JPG"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(image)
    label_image(image, [], 32, 32, keep_empty=True)
    return opened


@pytest.fixture
def started(client: TestClient) -> Iterator[terminal_routes.TerminalSession]:
    """Launch a session through the backend and serve the backend while the test runs."""
    from tcip_web.state import store

    sid = client.post("/api/terminal/sessions", json=LAUNCH).json()[
        "session_id"
    ]
    with served(app, store.workspace):
        yield terminal_routes._SESSIONS[sid]


def _hooked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    event: dict,
    *argv: str,
) -> str:
    """Run the command in this process on ``event`` as stdin and ``argv``; return its stdout."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    assert session_hook.main(list(argv)) == 0
    return capsys.readouterr().out


_FULL_STARTS = {
    "claude": (lambda c: _claude_start(c, "startup"), _additional_context),
    "codex": (lambda c: _codex_start(c, "startup"), _additional_context),
    "antigravity": (_agy_start, _injected),
}


@pytest.mark.parametrize("harness", sorted(_FULL_STARTS))
def test_each_harnesss_new_session_reports_and_is_told_the_whole_orientation(
    harness: str,
    remembered: OpenProject,
    started: terminal_routes.TerminalSession,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A new session of each harness reports its conversation and is told the whole orientation.

    It is told the project's overview, its data-state finding, its newest report and
    retrospective and where the knowledge documents are served, in its harness's own output
    shape. Antigravity's hook reaches the backend through the route its argument names alone.
    """
    from tcip_web.state import store

    event, told = _FULL_STARTS[harness]
    if harness == "antigravity":
        monkeypatch.delenv(TERMINAL_SESSION_ENV, raising=False)
        argv = [session_start_url(store.workspace, started.id)]
    else:
        monkeypatch.setenv(TERMINAL_SESSION_ENV, started.id)
        argv = []

    text = told(_hooked(monkeypatch, capsys, event("conv-new"), *argv))

    assert started.conversation == "conv-new"
    for fact in (
        '"display_name": "Test project"',
        "2026-03-04/IMG_E: empty label document",
        "the newest report detail",
        "orientation-task",
        serve_domain_knowledge.__name__,
    ):
        assert fact in text, fact


@pytest.mark.parametrize("event", [_claude_start, _codex_start])
def test_a_compaction_is_told_the_memory_and_the_knowledge_alone(
    event: Callable[[str, str], dict],
    remembered: OpenProject,
    started: terminal_routes.TerminalSession,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A compacted session is told its project's memory and the knowledge, nothing else."""
    monkeypatch.setenv(TERMINAL_SESSION_ENV, started.id)

    text = _additional_context(
        _hooked(monkeypatch, capsys, event("conv-c", "compact"))
    )

    assert started.conversation == "conv-c"
    assert "the newest report detail" in text
    assert "orientation-task" in text
    assert serve_domain_knowledge.__name__ in text
    assert "display_name" not in text
    assert "IMG_E" not in text


@pytest.mark.parametrize("event", [_claude_start, _codex_start])
def test_a_resumed_session_reports_its_conversation_and_is_told_nothing(
    event: Callable[[str, str], dict],
    remembered: OpenProject,
    started: terminal_routes.TerminalSession,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(TERMINAL_SESSION_ENV, started.id)

    printed = _hooked(monkeypatch, capsys, event("conv-r", "resume"))

    assert printed == ""
    assert started.conversation == "conv-r"


def test_the_orientation_holds_the_newest_entries_that_fit_and_how_many_there_are(
    opened: OpenProject,
    started: terminal_routes.TerminalSession,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With more reports than fit, the orientation fits its cap and holds the newest of them.

    The reports are written out of time order, each stating a distinct known time, and the page
    holds the latest-stated ones in descending time and states every report it was cut from.
    """
    from datetime import datetime, timedelta, timezone

    from tcip_mcp.tools import meta_tools

    base = datetime(2026, 3, 1, tzinfo=timezone.utc)
    minutes = {n: (n * 17) % 40 for n in range(40)}
    for n, minute in minutes.items():
        stated = (base + timedelta(minutes=minute)).isoformat()
        monkeypatch.setattr(
            meta_tools, "now_iso", lambda stated=stated: stated
        )
        report_friction(
            opened.root,
            "missing_tool",
            f"report {n:02d} " + "x" * 400,
            actor=None,
        )
    by_time = sorted(minutes, key=minutes.__getitem__, reverse=True)
    monkeypatch.setenv(TERMINAL_SESSION_ENV, started.id)

    text = _additional_context(
        _hooked(monkeypatch, capsys, _claude_start("conv-cap", "startup"))
    )

    assert len(text) <= pty_host.ORIENTATION_CAP
    (paged,) = [
        part.partition(" pages them: ")[2]
        for part in text.split("\n\n")
        if part.startswith("Its newest") and "kind='reports'" in part
    ]
    page = json.loads(paged)
    shown = [e["record"]["detail"][:9] for e in page["entries"]]
    assert page["total_available"] == 40
    assert 0 < len(shown) < 40
    assert shown == [f"report {n:02d}" for n in by_time[: len(shown)]]


def test_an_orientation_whose_overview_and_entries_do_not_fit_stays_under_its_cap(
    opened: OpenProject,
) -> None:
    """An overview and a retrospective too long for any share still leave the whole in its cap.

    A retrospective whose ``project_id`` alone passes the cap reaches the overview through
    ``inspect_project``'s recent activity: the overview stands as its short statement and the
    retrospectives page holds none of its one entry.
    """
    write_retrospective(
        opened.root,
        "p" * 11_000,
        "a long line of work",
        "it worked",
        "nothing",
        actor=None,
    )

    text = pty_host.orientation(opened, "startup")

    assert text is not None
    assert len(text) <= pty_host.ORIENTATION_CAP
    assert "overview is longer than this orientation holds" in text
    assert '{"entries": [], "total_available": 1}' in text


def _findings_part(text: str | None) -> tuple[str, list]:
    """Return the orientation's findings part: its statement of the count and what it lists."""
    assert text is not None
    (part,) = [
        p
        for p in text.split("\n\n")
        if p.startswith(f"{diagnose_project.__name__} answers")
    ]
    counted, _, listed = part.partition("; the first ")
    shown = listed.partition(" of them, errors first: ")[2]
    return counted, json.loads(shown) if shown else []


def test_more_findings_than_fit_name_the_tool_answering_all_of_them(
    opened: OpenProject,
) -> None:
    """With more findings than fit, the orientation names ``diagnose_project`` beside the count.

    Each finding is an empty label document no mark finishes, written by the label producer.
    """
    root = opened.root
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="bud", description=""),))
    )
    images = image_dir(root, "2026-03-04")
    images.mkdir(parents=True)
    for n in range(160):
        image = images / f"IMG_{n:03d}.JPG"
        Image.new("RGB", (8, 8)).save(image)
        label_image(image, [], 8, 8, keep_empty=True)

    counted, shown = _findings_part(pty_host.orientation(opened, "startup"))

    assert (
        counted.rstrip(".")
        == f"{diagnose_project.__name__} answers 160 finding(s) over it"
    )
    assert len(shown) < 160


def test_the_orientation_lists_the_first_findings_the_tool_answers(
    opened: OpenProject,
) -> None:
    """The orientation states the count ``diagnose_project`` answers and lists its first findings.

    Over a project whose checks find an image with no label record before an unreadable trait
    record and an unconfirmed one, the findings the part lists are the tool's own through a
    client's handshake, errors first, in the tool's order.
    """
    from tcip_mcp import traits
    from tests import _trait_fixtures as fx
    from tests._record_damage_fixtures import damage_record
    from tests._web_fixtures import bound_to
    from tests.test_agent_identity_records import call_through_handshake

    root = opened.root
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="bud", description=""),))
    )
    images = image_dir(root, "2026-03-04")
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images / "IMG_U.JPG")
    fx.propose_and_confirm(root, fx.entry("leaf", ("leaf_length",)))
    damage_record(traits.trait_key(root, "leaf"), b"{not valid json")
    fx.propose(root, fx.entry("burr_size", ("leaf_length",)))
    (answered,) = call_through_handshake(
        [(diagnose_project.__name__, {})], bound=bound_to(opened)
    )
    tool = answered["findings"]

    counted, shown = _findings_part(pty_host.orientation(opened, "startup"))

    assert [level for level, _ in tool] == ["error", "warn", "info"]
    assert counted == (
        f"{diagnose_project.__name__} answers {len(tool)} finding(s) over it"
    )
    assert shown == tool


def _parts(*lengths: int) -> list[pty_host._Part]:
    """Return a whole part of each of ``lengths`` letters, each with a one-letter omitted text."""
    return [
        pty_host._whole(letter * length, "-")
        for letter, length in zip("abcdefgh", lengths)
    ]


def test_parts_that_fit_together_all_stay_whole() -> None:
    """Parts of 1, 9,000 and 1 characters fit the cap together and stay whole, in order."""
    assert pty_host._fitted(_parts(1, 9_000, 1)) == "\n\n".join(
        ["a", "b" * 9_000, "c"]
    )


def test_a_part_needing_less_than_an_equal_share_keeps_it_whole() -> None:
    """Of parts of 1, 6,000 and 6,000 characters, the first stays whole and the others are cut.

    The whole stays within the cap and keeps the parts' order.
    """
    fitted = pty_host._fitted(_parts(1, 6_000, 6_000))

    assert len(fitted) <= pty_host.ORIENTATION_CAP
    assert fitted.split("\n\n") == ["a", "-", "-"]


def _run(
    command: str | list[str], event: dict, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a hook ``command`` as its harness would, a string through a shell, a list without."""
    return subprocess.run(
        command,
        shell=isinstance(command, str),
        input=json.dumps(event),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_each_rows_own_hook_spelling_runs_reports_and_prints_its_orientation(
    remembered: OpenProject,
    started: terminal_routes.TerminalSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each row's hook, read from what its launch writes, runs as its harness runs it.

    Claude's exec-form handler runs with no shell and Codex's line through one, each in an
    environment naming the session; agy's line runs through a shell in an environment naming
    none. Each reports its conversation and prints its orientation in its harness's shape.
    """
    from tcip_web.state import store

    monkeypatch.setattr(
        pty_host, "ANTIGRAVITY_SETTINGS", tmp_path / "agy-settings.json"
    )
    monkeypatch.setattr(
        pty_host, "run_to_completion", lambda argv, step: "added"
    )
    env = {**pty_host.os.environ, TERMINAL_SESSION_ENV: started.id}
    (settings,) = pty_host.render_argv(
        [pty_host.CLAUDE_SETTINGS_ARG], remembered, env, None
    )
    ((claude,),) = [
        group["hooks"]
        for group in json.loads(Path(settings).read_text(encoding="utf-8"))[
            "hooks"
        ]["SessionStart"]
    ]
    (codex_hooks,) = [
        tomllib.loads(override)["hooks"]
        for override in pty_host.codex_overrides(remembered, env)
        if override.startswith("hooks=")
    ]
    ((codex_handler,),) = [
        group["hooks"] for group in codex_hooks["SessionStart"]
    ]
    pty_host.prepare_antigravity("agy", remembered, started.id)
    hooks = json.loads(
        (store.workspace / pty_host.ANTIGRAVITY_HOOKS).read_text(
            encoding="utf-8"
        )
    )
    (agy,) = hooks[pty_host.ANTIGRAVITY_HOOK_ENTRY]["SessionStart"]
    unnamed = {k: v for k, v in env.items() if k != TERMINAL_SESSION_ENV}

    for command, event, run_env, told in (
        (
            [claude["command"], *claude["args"]],
            _claude_start("conv-claude", "startup"),
            env,
            _additional_context,
        ),
        (
            codex_handler["command"],
            _codex_start("conv-codex", "startup"),
            env,
            _additional_context,
        ),
        (agy["command"], _agy_start("conv-agy"), unnamed, _injected),
    ):
        ran = _run(command, event, run_env)
        assert ran.returncode == 0, ran.stderr
        assert "the newest report detail" in told(ran.stdout)
        assert started.conversation == event.get(
            "session_id", event.get("conversationId")
        )

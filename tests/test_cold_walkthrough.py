"""tools/cold_walkthrough.py: one cold reader walks the served app toward a goal, and is recorded.

Each test marked ``ON_WINDOWS`` starts a real backend over a scratch harness root holding a
project made by ``initialize_project``, through the serving tool, so it runs on Windows alone; the
early refusal of a host the server cannot be stopped on runs everywhere. The reader's launch is the
one thing replaced, at the runner's seam. A walk needs the frontend's built page, which is not
tracked, so the walk tests run where a build left one and the no-page refusal test where none is.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from tcip_web.app import STATIC_DIR, stale_bundle
from tests._scratch_server_fixtures import (
    ON_WINDOWS,
    end_served_processes,
    kill_recorded_server,
    recording_launch,
    served_processes,
)
from tests._web_fixtures import new_project
from tools import cold_walkthrough, cross_family_ask, serve_capture_app

BUILT = (STATIC_DIR / "index.html").exists()
NEEDS_PAGE = pytest.mark.skipif(
    not BUILT,
    reason=f"no built page in {STATIC_DIR}; the frontend's build makes one",
)


@pytest.fixture
def harness(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Yield a harness root whose workspace holds one project; end its backends after."""
    root = tmp_path_factory.mktemp("harness").resolve()
    (root / "workspace" / "orchard").mkdir(parents=True)
    new_project(root / "workspace" / "orchard")
    yield root
    end_served_processes(root)


@pytest.fixture
def goal(tmp_path: Path) -> Path:
    """Return a goal file a person might hand the reader."""
    path = tmp_path / "goal.md"
    path.write_text(
        "Find the photographs taken of the orchard's plants.\n",
        encoding="utf-8",
    )
    return path


def _records(out: Path) -> list[dict]:
    """Return every walk's ``walkthrough.json`` under ``out`` with the folder it sits in."""
    return [
        {**json.loads(path.read_text(encoding="utf-8")), "folder": path.parent}
        for path in sorted(out.glob("*/walkthrough.json"))
    ]


def _launch_broke() -> NoReturn:
    """Fail as a reader's launch can that the runner does not catch."""
    raise RuntimeError("the reader's launch broke")


def test_a_walk_on_a_host_it_cannot_stop_on_is_refused_before_anything_is_made(
    tmp_path: Path, goal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serve_capture_app.sys, "platform", "linux")
    out = tmp_path / "walks"

    with pytest.raises(SystemExit, match="Windows-only"):
        cold_walkthrough.walk(goal, tmp_path / "harness", out)
    assert not out.exists()


@ON_WINDOWS
@NEEDS_PAGE
def test_a_walk_records_the_goal_the_server_the_readers_run_and_its_stop(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader's prompt is the goal and the page's address, from an empty folder apart from out.

    The record holds the goal's hash, the port, the workspace served, the stale-bundle sentence
    the app reports for this tree and no stop refusal; the run beside it holds the argv launched
    and the reader's answer; no backend is left.
    """
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) == 0

    (launched,) = seen
    (record,) = _records(out)
    run = record["folder"] / "browser-only" / "claude"
    address = f"http://127.0.0.1:{record['port']}/"
    assert set(record) - {"folder"} == {
        "goal",
        "goal_sha256",
        "port",
        "workspace",
        "bundle",
        "reader_fault",
        "stop_refusal",
    }
    assert (
        record["goal_sha256"] == hashlib.sha256(goal.read_bytes()).hexdigest()
    )
    assert os.path.samefile(record["workspace"], harness / "workspace")
    assert record["stop_refusal"] is None
    assert record["reader_fault"] == ""
    assert (
        launched["stdin"]
        == f"{goal.read_text(encoding='utf-8').rstrip()}\n\n{address}\n"
    )
    assert launched["entries"] == []
    assert not launched["cwd"].resolve().is_relative_to(out.resolve())
    assert record["bundle"] == stale_bundle()
    assert (run / "argv.txt").read_text(
        encoding="utf-8"
    ).splitlines() == launched["argv"]
    assert (run / "response.md").read_text(
        encoding="utf-8"
    ) == "a stand-in report"
    assert served_processes(harness) == []


@ON_WINDOWS
@NEEDS_PAGE
def test_a_reader_that_fails_is_recorded_with_its_fault_and_a_failed_walk(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch([], 1)
    )
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) != 0

    (record,) = _records(out)
    assert "exited 1" in record["reader_fault"]
    assert record["stop_refusal"] is None
    assert served_processes(harness) == []


@ON_WINDOWS
@NEEDS_PAGE
def test_a_stop_that_fails_after_a_clean_walk_is_recorded_and_fails_the_walk(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The served process ends while the reader runs, so the stop's kill finds nothing to end."""
    ended: list[int] = []
    monkeypatch.setattr(
        cross_family_ask,
        "launch_harness",
        recording_launch(
            [], during=lambda: ended.append(kill_recorded_server(harness))
        ),
    )
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) == 1

    (record,) = _records(out)
    assert record["reader_fault"] == ""
    assert f"process {ended[0]}" in record["stop_refusal"]


@ON_WINDOWS
@NEEDS_PAGE
def test_a_reader_launch_that_raises_stops_the_server_and_raises_its_failure(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cross_family_ask,
        "launch_harness",
        recording_launch([], during=_launch_broke),
    )

    with pytest.raises(RuntimeError, match="launch broke"):
        cold_walkthrough.walk(goal, harness, tmp_path_factory.mktemp("walks"))
    assert served_processes(harness) == []
    assert not (harness / serve_capture_app.SERVER_RECORD_FILENAME).exists()


@ON_WINDOWS
@NEEDS_PAGE
def test_a_stop_that_fails_after_the_reader_raised_is_raised_chained_onto_that_failure(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The served process ends and then the reader's launch raises; both failures reach the caller.

    The stop's refusal names the process, keeps the server record, and carries the launch's
    failure as its cause.
    """
    ended: list[int] = []

    def end_then_break() -> NoReturn:
        ended.append(kill_recorded_server(harness))
        _launch_broke()

    monkeypatch.setattr(
        cross_family_ask,
        "launch_harness",
        recording_launch([], during=end_then_break),
    )

    with pytest.raises(SystemExit) as refused:
        cold_walkthrough.walk(goal, harness, tmp_path_factory.mktemp("walks"))
    assert f"process {ended[0]}" in str(refused.value)
    assert isinstance(refused.value.__cause__, RuntimeError)
    assert (harness / serve_capture_app.SERVER_RECORD_FILENAME).is_file()


@ON_WINDOWS
@NEEDS_PAGE
def test_a_second_walk_over_the_same_root_writes_its_own_folder(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch([])
    )
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) == 0
    assert cold_walkthrough.walk(goal, harness, out) == 0

    first, second = _records(out)
    assert first["folder"] != second["folder"]


@ON_WINDOWS
@pytest.mark.skipif(BUILT, reason=f"a built page is in {STATIC_DIR}")
def test_a_root_whose_app_has_no_built_page_is_refused_after_its_server_stops(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )

    with pytest.raises(SystemExit, match="answered 503, not the built page"):
        cold_walkthrough.walk(goal, harness, tmp_path_factory.mktemp("walks"))
    assert seen == []
    assert served_processes(harness) == []
    assert not (harness / serve_capture_app.SERVER_RECORD_FILENAME).exists()

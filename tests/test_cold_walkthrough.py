"""tools/cold_walkthrough.py: one cold reader walks the served app toward a goal, and is recorded.

Each test starts a real backend over a scratch harness root holding a project made by
``initialize_project``, through the serving tool, so each runs on Windows alone. The reader's
launch is the one thing replaced, at the runner's seam. A walk needs the frontend's built page,
which is not tracked, so the walk tests run where a build left one and the refusal test where
none is.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tcip_web.app import STATIC_DIR
from tests._scratch_server_fixtures import (
    ON_WINDOWS,
    end_served_processes,
    served_processes,
)
from tests._web_fixtures import new_project
from tools import cold_walkthrough, cross_family_ask

BUILT = (STATIC_DIR / "index.html").exists()
NEEDS_PAGE = pytest.mark.skipif(
    not BUILT,
    reason=f"no built page in {STATIC_DIR}; the frontend's build makes one",
)
pytestmark = ON_WINDOWS


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


def _stand_in(
    seen: list[dict], code: int
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a reader launch that answers "a stand-in report" with ``code``, recording each call.

    Each record holds the argv, the prompt, and the working directory's entries at launch.
    """

    def launch(
        argv: list[str],
        stdin: str | None,
        timeout: int,
        cwd: Path,
        env: dict[str, str] | None,
    ) -> subprocess.CompletedProcess[str]:
        seen.append(
            {"argv": argv, "stdin": stdin, "entries": list(cwd.iterdir())}
        )
        answer = json.dumps({"result": "a stand-in report"})
        return subprocess.CompletedProcess(
            argv, code, stdout=answer, stderr=""
        )

    return launch


def _records(out: Path) -> list[dict]:
    """Return every walk's ``walkthrough.json`` under ``out`` with the folder it sits in."""
    return [
        {**json.loads(path.read_text(encoding="utf-8")), "folder": path.parent}
        for path in sorted(out.glob("*/walkthrough.json"))
    ]


@NEEDS_PAGE
def test_a_walk_records_the_goal_the_server_the_readers_run_and_its_stop(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader is told the goal and the page's address alone, from an empty folder.

    The record holds the goal's hash, the port, the workspace served and a stop that succeeded;
    the run beside it holds the argv launched and the reader's answer; no backend is left.
    """
    seen: list[dict] = []
    monkeypatch.setattr(cross_family_ask, "launch_harness", _stand_in(seen, 0))
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) == 0

    (launched,) = seen
    (record,) = _records(out)
    run = record["folder"] / "browser-only" / "claude"
    address = f"http://127.0.0.1:{record['port']}/"
    assert (
        record["goal_sha256"] == hashlib.sha256(goal.read_bytes()).hexdigest()
    )
    assert os.path.samefile(record["workspace"], harness / "workspace")
    assert record["stopped"] is True
    assert record["reader_fault"] == ""
    assert (
        launched["stdin"]
        == f"{goal.read_text(encoding='utf-8').rstrip()}\n\n{address}\n"
    )
    assert launched["entries"] == []
    assert (run / "argv.txt").read_text(
        encoding="utf-8"
    ).splitlines() == launched["argv"]
    assert (run / "response.md").read_text(
        encoding="utf-8"
    ) == "a stand-in report"
    assert served_processes(harness) == []


@NEEDS_PAGE
def test_a_reader_that_fails_is_recorded_with_its_fault_and_a_failed_walk(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cross_family_ask, "launch_harness", _stand_in([], 1))
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) != 0

    (record,) = _records(out)
    assert "exited 1" in record["reader_fault"]
    assert record["stopped"] is True
    assert served_processes(harness) == []


@NEEDS_PAGE
def test_a_second_walk_over_the_same_root_writes_its_own_folder(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cross_family_ask, "launch_harness", _stand_in([], 0))
    out = tmp_path_factory.mktemp("walks")

    assert cold_walkthrough.walk(goal, harness, out) == 0
    assert cold_walkthrough.walk(goal, harness, out) == 0

    first, second = _records(out)
    assert first["folder"] != second["folder"]


@pytest.mark.skipif(BUILT, reason=f"a built page is in {STATIC_DIR}")
def test_a_root_whose_app_has_no_built_page_is_refused_after_its_server_stops(
    harness: Path,
    goal: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(cross_family_ask, "launch_harness", _stand_in(seen, 0))

    with pytest.raises(SystemExit, match="answered 503, not the built page"):
        cold_walkthrough.walk(goal, harness, tmp_path_factory.mktemp("walks"))
    assert seen == []
    assert served_processes(harness) == []
    assert not (harness / "server_info.json").exists()

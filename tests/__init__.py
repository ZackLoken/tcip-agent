"""The test suite and the helpers its tests share.

It carries the root of the repository its tests read files from, the launcher of a child test
session, the browser's package, its sources, its formatter, the writer of its generated files and
its Node scripts with their runner, the names the person rule is tested on, and the reader of the
CSV files its tests deliver.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
"""The repository this suite belongs to."""


def child_pytest(
    tmp_path: Path,
    test_file: Path,
    *pytest_args: str,
    timeout: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run pytest in a child process over ``test_file`` with ``pytest_args``.

    The child runs from the repository root, under ``env`` (the parent's when ``None``), bounded
    by ``timeout`` seconds, its output captured as text. The child's root is ``tmp_path`` and
    its temporary root :func:`child_basetemp` of ``tmp_path``. The cache provider is off.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            *pytest_args,
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(child_basetemp(tmp_path)),
            "--rootdir",
            str(tmp_path),
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def child_basetemp(tmp_path: Path) -> Path:
    """Return ``tmp_path / "child-basetemp"``, a child session's temporary root."""
    return tmp_path / "child-basetemp"


FRONTEND = REPO_ROOT / "packages" / "tcip-web" / "frontend"
"""The browser's package: its manifest, its configuration and its sources."""
FRONTEND_SRC = (
    FRONTEND
    / json.loads((FRONTEND / "tsconfig.json").read_text(encoding="utf-8"))[
        "include"
    ][0]
)
"""The browser's sources, the directory the frontend's ``tsconfig.json`` includes first."""

PRETTIER = FRONTEND / "node_modules" / "prettier" / "bin" / "prettier.cjs"
"""The frontend's installed Prettier."""
SCRIPTS = FRONTEND / "scripts"
"""The frontend's Node scripts, each printing one JSON object (:func:`printed_json`)."""
READS_SCRIPT = SCRIPTS / "routeReads.mjs"
"""The script printing the frontend's non-test sources and what they read and declare."""
VITE_CONFIG = SCRIPTS / "viteConfig.mjs"
"""The script printing the frontend's Vite configuration as Vite resolves it."""


def printed_json(script: Path, argument: Path) -> dict:
    """Return the JSON object the frontend script ``script`` prints over ``argument``."""
    printed = subprocess.run(
        ["node", str(script), str(argument)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(printed.stdout)


def formatted(text: str, path: Path) -> str:
    """Return ``text`` as the frontend's installed Prettier formats a file at ``path``.

    Prettier reads the frontend's own configuration for ``path``.
    """
    return subprocess.run(
        ["node", str(PRETTIER), "--stdin-filepath", str(path)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def write_generated(path: Path, text: str) -> int:
    """Write ``text`` to ``path`` with LF line endings and say so; return the exit code 0."""
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return 0


_PERSON_NAMES = json.loads(
    (FRONTEND_SRC / "test" / "personNames.json").read_text(encoding="utf-8")
)
ADMITTED_NAMES: list[tuple[str, str]] = [
    (s, r) for s, r in _PERSON_NAMES["admitted"]
]
"""The names the browser's ``test/personNames.json`` lists as admitted, each with the name it
records."""
REFUSED_NAMES: list[str] = _PERSON_NAMES["refused"]
"""The strings the browser's ``test/personNames.json`` lists as refused for naming no one."""


def csv_rows(path: str | Path) -> list[dict]:
    """Return the rows of the UTF-8 CSV file at ``path``, each keyed by its header."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

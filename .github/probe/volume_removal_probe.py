"""Probe what a process does to a WAL database after its volume is removed and returned.

Run elevated on Windows: the volume is a virtual disk this script creates, attaches, detaches
and attaches again with diskpart. One process (the holder) keeps a connection open across the
removal; a second process (the writer) opens the database after the return, grows it and is
then killed or closed; the holder then opens and closes a new connection. The script reports
the database's bytes and integrity before and after that close as one JSON document.

    python volume_removal_probe.py <work dir> --letter V --writer-exit kill
    python volume_removal_probe.py <work dir> --letter V --writer-exit kill --no-checkpoint-on-close

Every size and count below is this probe's own choice, sized to make a database of a few hundred
pages; none comes from the platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROWS_BEFORE = 3000
ROWS_AFTER = 1500
MANUAL_WAIT_S = 180
SMALL_COMMITS = 30
ROWS_PER_SMALL_COMMIT = 30
PAYLOAD = "x" * 1500


def diskpart(lines: list[str], work: Path) -> tuple[int, str]:
    """Run a diskpart script and return its exit code and what it printed."""
    script = work / "diskpart.txt"
    script.write_text("\n".join(lines) + "\n", encoding="ascii")
    done = subprocess.run(
        ["diskpart", "/s", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, (done.stdout + done.stderr).strip()


def digest(path: Path) -> dict:
    """Return the size and SHA-256 of a file, or the error reading it."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def state(db: Path) -> dict:
    """Return the digests of the database file, its log and its shared-memory file."""
    return {
        name: digest(Path(str(db) + suffix))
        for name, suffix in (("db", ""), ("wal", "-wal"), ("shm", "-shm"))
    }


def raised(exc: BaseException) -> str:
    """Return an exception's class, message and, for SQLite's, its extended error name."""
    name = getattr(exc, "sqlite_errorname", None)
    code = getattr(exc, "sqlite_errorcode", None)
    return f"{type(exc).__name__}: {exc}" + (
        f" [{name} {code}]" if name else ""
    )


def powershell(command: str) -> tuple[int, str]:
    """Run one PowerShell command and return its exit code and what it printed."""
    done = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, (done.stdout + done.stderr).strip()


def attempt(label: str, action: Callable[[], object]) -> dict:
    """Run an action and return its value or the exception it raised, under a label."""
    try:
        return {"step": label, "ok": True, "value": action()}
    except Exception as exc:  # noqa: BLE001
        return {"step": label, "ok": False, "error": raised(exc)}


def fresh_check(db: Path) -> dict:
    """Open the database in a new process and return its integrity check and page count."""
    code = (
        "import sqlite3, sys, json\n"
        "c = sqlite3.connect(sys.argv[1])\n"
        "out = {}\n"
        "try:\n"
        "    out['integrity'] = [r[0] for r in c.execute('pragma integrity_check')][:5]\n"
        "    out['page_count'] = c.execute('pragma page_count').fetchone()[0]\n"
        "    out['rows'] = c.execute('select count(*) from t').fetchone()[0]\n"
        "except Exception as exc:\n"
        "    out['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(out))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(db)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {
            "error": f"exit {done.returncode}: {(done.stdout + done.stderr)[-400:]}"
        }


def writer(db: Path) -> int:
    """Open the returned database, grow it, say so, and hold it open until told to leave."""
    conn = sqlite3.connect(str(db), isolation_level=None)
    said = {"quick_check": conn.execute("pragma quick_check").fetchone()[0]}
    said["checkpoint"] = list(
        conn.execute("pragma wal_checkpoint(TRUNCATE)").fetchone()
    )
    for _ in range(SMALL_COMMITS):
        conn.execute("begin")
        conn.executemany(
            "insert into t(payload) values (?)",
            [(PAYLOAD,) for _ in range(ROWS_AFTER // SMALL_COMMITS)],
        )
        conn.execute("commit")
    said["page_count"] = conn.execute("pragma page_count").fetchone()[0]
    said["rows"] = conn.execute("select count(*) from t").fetchone()[0]
    print(json.dumps(said), flush=True)
    sys.stdin.readline()
    conn.close()
    return 0


class Told(list):
    """A list of steps that prints each one as it is added, so a crash keeps what came before."""

    def append(self, step: dict) -> None:
        """Add a step and print it at once."""
        super().append(step)
        print(f"STEP {json.dumps(step)}", flush=True)


def configure(conn: sqlite3.Connection, no_checkpoint: bool) -> None:
    """Tell a connection not to checkpoint when it closes, when the run asks for that."""
    if no_checkpoint:
        conn.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)


def holder(
    work: Path,
    letter: str,
    writer_exit: str,
    no_checkpoint: bool,
    removal: str,
    outer_letter: str,
) -> dict:
    """Run the whole sequence and return what each step observed."""
    work.mkdir(parents=True, exist_ok=True)
    vhd = work / "probe.vhd"
    db = Path(f"{letter}:/store.db")
    outer_select = f'select vdisk file="{work / "outer.vhd"}"'
    if removal == "nested":
        vhd = Path(f"{outer_letter}:/probe.vhd")
    select = f'select vdisk file="{vhd}"'
    report: dict = {
        "sqlite": sqlite3.sqlite_version,
        "python": sys.version.split()[0],
        "removal": removal,
        "writer_exit": writer_exit,
        "no_checkpoint_on_close": no_checkpoint,
        "steps": Told(),
    }
    steps = report["steps"]

    if removal == "nested":
        code, said = diskpart(
            [
                f'create vdisk file="{work / "outer.vhd"}" maximum=1024 type=expandable',
                outer_select,
                "attach vdisk",
                "create partition primary",
                "format fs=ntfs quick label=outer",
                f"assign letter={outer_letter}",
            ],
            work,
        )
        steps.append(
            {"step": "outer disk made", "exit": code, "said": said[-300:]}
        )

    if removal == "manual":
        db = Path(f"{letter}:/tcip-probe-664/store.db")
        if not Path(f"{letter}:/").exists():
            report["stopped"] = f"no volume is mounted at {letter}:"
            return report
        db.parent.mkdir(exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        steps.append(
            {
                "step": "volume in hand",
                "said": powershell(
                    f"Get-Volume -DriveLetter {letter} | "
                    "Select-Object FileSystem, DriveType | Format-List"
                )[1],
            }
        )
    else:
        code, said = diskpart(
            [
                f'create vdisk file="{vhd}" maximum=256 type=expandable',
                select,
                "attach vdisk",
                "create partition primary",
                "format fs=ntfs quick label=probe",
                f"assign letter={letter}",
            ],
            work,
        )
        steps.append(
            {"step": "create and attach", "exit": code, "said": said[-600:]}
        )
        if code != 0 or not db.parent.exists():
            report["stopped"] = (
                "the virtual disk could not be made; is this shell elevated?"
            )
            return report

    first = sqlite3.connect(
        str(db), isolation_level=None, check_same_thread=False
    )
    configure(first, no_checkpoint)
    first.execute("pragma journal_mode = wal")
    first.execute("pragma synchronous = FULL")
    first.execute("create table t(id integer primary key, payload text)")
    first.execute("begin")
    first.executemany(
        "insert into t(payload) values (?)",
        [(PAYLOAD,) for _ in range(ROWS_BEFORE)],
    )
    first.execute("commit")
    for _ in range(SMALL_COMMITS):
        first.execute("begin")
        first.executemany(
            "insert into t(payload) values (?)",
            [(PAYLOAD,) for _ in range(ROWS_PER_SMALL_COMMIT)],
        )
        first.execute("commit")
    steps.append(
        {
            "step": "holder wrote before the removal",
            "page_count": first.execute("pragma page_count").fetchone()[0],
            "rows": first.execute("select count(*) from t").fetchone()[0],
            "files": state(db),
        }
    )
    marker = db.with_name("kept_open.bin")
    marker.write_bytes(b"kept open across the removal")
    kept = os.open(str(marker), os.O_RDONLY | getattr(os, "O_BINARY", 0))

    def told(step: str, outcome: tuple[int, str]) -> None:
        """Record a removal or return command's exit code and the end of what it printed."""
        steps.append(
            {"step": step, "exit": outcome[0], "said": outcome[1][-300:]}
        )
        time.sleep(2)

    def waited(present: bool, said: str) -> dict:
        """Print what the person is asked to do and wait until the volume is there or gone."""
        print(f"\n>>> {said}\n", file=sys.stderr, flush=True)
        deadline = time.monotonic() + MANUAL_WAIT_S
        while Path(f"{letter}:/").exists() != present:
            if time.monotonic() > deadline:
                return {"step": said, "ok": False, "error": "not seen in time"}
            time.sleep(0.5)
        time.sleep(3)
        return {"step": said, "ok": True}

    if removal == "manual":
        steps.append(waited(False, f"Pull the drive at {letter}: out now."))
        steps.append(
            waited(
                True, f"Put the same drive back; it must return as {letter}:."
            )
        )
    elif removal == "offline":
        number = f"(Get-DiskImage -ImagePath '{vhd}').Number"
        told(
            "disk taken offline with the connection open",
            powershell(f"Set-Disk -Number {number} -IsOffline $true"),
        )
        told(
            "disk brought online again",
            powershell(f"Set-Disk -Number {number} -IsOffline $false"),
        )
    elif removal == "nested":
        told(
            "outer disk detached with the connection open",
            diskpart([outer_select, "detach vdisk"], work),
        )
        told(
            "outer disk attached again",
            diskpart([outer_select, "attach vdisk"], work),
        )
        told(
            "inner disk's old attachment detached",
            diskpart([select, "detach vdisk"], work),
        )
        told(
            "inner disk attached again",
            diskpart([select, "attach vdisk"], work),
        )
    else:
        told(
            "detach with the connection open",
            diskpart([select, "detach vdisk"], work),
        )
        told("attach again", diskpart([select, "attach vdisk"], work))
    if removal != "manual" and not db.exists():
        code, said = diskpart(
            [select, "select partition 1", f"assign letter={letter}"], work
        )
        steps.append(
            {
                "step": "assign the letter again",
                "exit": code,
                "said": said[-400:],
            }
        )
        time.sleep(2)
    steps.append(
        {"step": "database present after the return", "value": db.exists()}
    )

    steps.append(
        attempt(
            "file handle kept open is stat-ed after the return",
            lambda: os.fstat(kept).st_size,
        )
    )
    steps.append(
        attempt(
            "file handle kept open is read after the return",
            lambda: (os.lseek(kept, 0, os.SEEK_SET), len(os.read(kept, 16)))[
                1
            ],
        )
    )
    steps.append(
        attempt(
            "old connection reads after the return",
            lambda: first.execute("select count(*) from t").fetchone()[0],
        )
    )
    steps.append(
        {"step": "fresh process after the return", "value": fresh_check(db)}
    )

    other = subprocess.Popen(
        [sys.executable, __file__, "--role", "writer", "--db", str(db)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    steps.append(
        {
            "step": "writer grew the database",
            "value": other.stdout.readline().strip(),
        }
    )
    steps.append({"step": "files while the writer holds", "files": state(db)})

    def open_and_close(label: str) -> None:
        """Open a new connection in this process, read the schema, and close it."""
        outcome: dict = {"step": label}
        try:
            conn = sqlite3.connect(
                str(db), isolation_level=None, check_same_thread=False
            )
        except Exception as exc:  # noqa: BLE001
            outcome["connect_error"] = raised(exc)
            steps.append(outcome)
            return
        try:
            configure(conn, no_checkpoint)
            outcome["read"] = conn.execute(
                "select count(*) from sqlite_master"
            ).fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            outcome["read_error"] = raised(exc)
        try:
            conn.close()
        except Exception as exc:  # noqa: BLE001
            outcome["close_error"] = raised(exc)
        outcome["files_after_close"] = state(db)
        steps.append(outcome)

    open_and_close(
        "holder opens and closes a new connection while the writer holds"
    )

    if writer_exit == "kill":
        other.kill()
    else:
        steps.append(
            attempt(
                "writer told to leave", lambda: other.stdin.write("leave\n")
            )
        )
        steps.append(attempt("writer's pipe flushed", other.stdin.flush))
    other.wait(timeout=30)
    steps.append({"step": f"writer left ({writer_exit})", "files": state(db)})
    intact = work / "before_close.db"
    for suffix in ("", "-wal"):
        source = Path(str(db) + suffix)
        if source.exists():
            Path(str(intact) + suffix).write_bytes(source.read_bytes())
    steps.append(
        {
            "step": "copy taken before the holder's close",
            "value": fresh_check(intact),
        }
    )

    open_and_close(
        "holder opens and closes a new connection with no other process"
    )
    steps.append(attempt("holder closes its old connection", first.close))
    steps.append(
        {"step": "files after every holder close", "files": state(db)}
    )
    steps.append(
        {"step": "fresh process at the end", "value": fresh_check(db)}
    )

    def reopened() -> int:
        """Open the database again in this process, with every earlier connection closed."""
        again = sqlite3.connect(str(db), isolation_level=None)
        try:
            configure(again, no_checkpoint)
            return again.execute("select count(*) from t").fetchone()[0]
        finally:
            again.close()

    steps.append(
        attempt("holder reopens after closing every connection", reopened)
    )
    steps.append({"step": "files after the reopen", "files": state(db)})

    if removal != "manual":
        code, said = diskpart([select, "detach vdisk"], work)
        steps.append({"step": "detach at the end", "exit": code})
    return report


def main() -> int:
    """Run as the holder, or as the writer when the holder starts this file with that role."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", nargs="?")
    parser.add_argument("--letter", default="V")
    parser.add_argument(
        "--writer-exit", choices=("kill", "clean"), default="kill"
    )
    parser.add_argument("--no-checkpoint-on-close", action="store_true")
    parser.add_argument(
        "--removal",
        choices=("detach", "offline", "nested", "manual"),
        default="detach",
    )
    parser.add_argument("--outer-letter", default="O")
    parser.add_argument(
        "--role", choices=("holder", "writer"), default="holder"
    )
    parser.add_argument("--db")
    args = parser.parse_args()
    if args.role == "writer":
        return writer(Path(args.db))
    report = holder(
        Path(args.work).resolve(),
        args.letter,
        args.writer_exit,
        args.no_checkpoint_on_close,
        args.removal,
        args.outer_letter,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

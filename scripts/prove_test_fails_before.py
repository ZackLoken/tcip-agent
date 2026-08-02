"""Prove a test actually fails against the code it was written to catch.

A test added alongside a fix is only a guard if it fails *before* the fix. Reasoning that it would
is not evidence: this repo shipped a test whose two assertions both passed against the very
regression they named, because an unrelated code path failed first and produced the same outcome.
Nothing in the normal gate can catch that: a vacuous test passes everywhere.

This extracts a baseline revision with ``git archive`` (the working tree is never touched), drops
the current test file into that old source, and runs it there. A test that passes is guarding
nothing; say so rather than counting it.

    python scripts/prove_test_fails_before.py tests/test_foo.py
    python scripts/prove_test_fails_before.py tests/test_foo.py -k "new_behaviour"
    python scripts/prove_test_fails_before.py tests/test_foo.py --rev HEAD~3

``--rev`` defaults to HEAD, which is the baseline that matters: the commit immediately before the
change under test. A distant baseline can fail for an unrelated reason, which looks like proof and
is not. When a near and a far baseline disagree, the near one is the answer.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _extract(rev: str, dest: Path) -> None:
    """Materialize `rev` into `dest`. Piped as bytes: a text pipe corrupts the archive."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", rev], cwd=REPO, check=True,
                             stdout=subprocess.PIPE).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_file", help="path, repo-relative, e.g. tests/test_foo.py")
    ap.add_argument("--rev", default="HEAD", help="baseline revision (default HEAD)")
    ap.add_argument("-k", dest="expr", default="", help="pytest -k expression")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="prefail-"))
    try:
        _extract(args.rev, tmp)
        shutil.copy2(REPO / args.test_file, tmp / args.test_file)

        srcs = [str(p) for p in sorted(tmp.glob("packages/*/src"))]
        env_path = ";".join([*srcs, str(tmp)]) if sys.platform == "win32" else ":".join([*srcs, str(tmp)])
        cmd = [sys.executable, "-m", "pytest", args.test_file, "-q", "-p", "no:cacheprovider"]
        if args.expr:
            cmd += ["-k", args.expr]

        import os

        env = {**os.environ, "PYTHONPATH": env_path}
        proc = subprocess.run(cmd, cwd=tmp, env=env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if proc.returncode == 0:
        print(f"\nVACUOUS: every selected test passed against {args.rev}. "
              f"It guards nothing; assert on what only the fix produces.")
        return 1
    print(f"\nOK: the selected tests fail against {args.rev}, so they guard the fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

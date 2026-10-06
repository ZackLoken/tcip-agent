"""Run ruff, mypy and pytest inside a worktree, its own editable-install resolution proven first.

Each gate runs with PYTHONPATH set to the worktree's four `packages/*/src` directories, after
`tcip_mcp` is proven to resolve under the worktree, in the foreground, stopping at the first
failure with its exit code.

    python tools/worktree_gate.py <worktree> --ruff --mypy --pytest tests/test_foo.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_SRC_DIRS = ("tcip-store", "tcip-annotation", "tcip-mcp", "tcip-web")


def build_environ(worktree: Path) -> dict[str, str]:
    """The environment every gate runs under: PYTHONPATH set to the worktree's own four package
    `src` directories, never appended to whatever the caller's own PYTHONPATH already names.
    """
    env = dict(os.environ)
    src_dirs = [str((worktree / "packages" / name / "src").resolve()) for name in PACKAGE_SRC_DIRS]
    sep = ";" if sys.platform == "win32" else ":"
    env["PYTHONPATH"] = sep.join(src_dirs)
    return env


def prove_resolution(worktree: Path, env: dict[str, str]) -> Path:
    """Confirm `tcip_mcp` resolves inside `worktree` under `env`, refusing before any gate runs
    if it instead resolves to the editable install's real target (another checkout)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import tcip_mcp; print(tcip_mcp.__file__)"],
        cwd=str(worktree), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"could not import tcip_mcp under the worktree environment: {proc.stderr.strip()}")
    resolved = Path(proc.stdout.strip()).resolve()
    if not resolved.is_relative_to(worktree.resolve()):
        raise SystemExit(
            f"tcip_mcp resolved to {resolved}, outside the worktree {worktree}; the editable "
            "install still points at another checkout, so no gate here would measure this one"
        )
    return resolved


def run_ruff(worktree: Path, env: dict[str, str]) -> int:
    # This repository has no scripts/ directory at HEAD; when one exists, add it to this list.
    print("== ruff check packages tests tools ==")
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", "packages", "tests", "tools"],
        cwd=str(worktree), env=env,
    ).returncode


def run_mypy(worktree: Path, env: dict[str, str]) -> int:
    cache_dir = tempfile.mkdtemp(prefix="worktree-gate-mypy-")
    try:
        print(f"== mypy --cache-dir {cache_dir} ==")
        return subprocess.run(
            [sys.executable, "-m", "mypy", "--cache-dir", cache_dir], cwd=str(worktree), env=env,
        ).returncode
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_pytest(worktree: Path, env: dict[str, str], files: list[str]) -> int:
    print(f"== pytest {' '.join(files)} ==")
    return subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-p", "no:cacheprovider", "--timeout=300"],
        cwd=str(worktree), env=env,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path)
    parser.add_argument("--ruff", action="store_true")
    parser.add_argument("--mypy", action="store_true")
    parser.add_argument("--pytest", nargs="+", default=None, metavar="FILE")
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    if not worktree.is_dir():
        raise SystemExit(f"{worktree} is not a directory")

    env = build_environ(worktree)
    resolved = prove_resolution(worktree, env)
    print(f"tcip_mcp resolves under the worktree: {resolved}")

    if args.ruff:
        code = run_ruff(worktree, env)
        if code:
            return code
    if args.mypy:
        code = run_mypy(worktree, env)
        if code:
            return code
    if args.pytest:
        code = run_pytest(worktree, env, args.pytest)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

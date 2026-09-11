"""Run ruff, mypy and pytest inside a worktree, its own editable-install resolution proven first.

The conda environment's editable installs point at the main checkout, so a worktree needs
PYTHONPATH set to its own four `packages/*/src` directories on every command, or a gate silently
measures the main checkout instead of the worktree. This wrapper builds that environment, proves
`tcip_mcp` actually resolves under the worktree before running anything, then runs each requested
gate in the foreground, stopping at the first failure with its exit code.

    python tools/worktree_gate.py <worktree> --ruff --mypy --pytest tests/test_foo.py --backend file
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


def _is_under(path: Path, ancestor: Path) -> bool:
    path, ancestor = path.resolve(), ancestor.resolve()
    return path == ancestor or ancestor in path.parents


def build_environ(worktree: Path, backend: str) -> dict[str, str]:
    """The environment every gate runs under: PYTHONPATH set to the worktree's own four package
    `src` directories (never appended to whatever the caller's own PYTHONPATH already names),
    and TCIP_STORE_BACKEND set for the file backend, unset for sqlite so the ambient default binds.
    """
    env = dict(os.environ)
    src_dirs = [str((worktree / "packages" / name / "src").resolve()) for name in PACKAGE_SRC_DIRS]
    sep = ";" if sys.platform == "win32" else ":"
    env["PYTHONPATH"] = sep.join(src_dirs)
    if backend == "file":
        env["TCIP_STORE_BACKEND"] = "file"
    else:
        env.pop("TCIP_STORE_BACKEND", None)
    return env


def prove_resolution(worktree: Path, env: dict[str, str]) -> Path:
    """Confirm `tcip_mcp` resolves inside `worktree` under `env`, refusing before any gate runs
    if it instead resolves to the editable install's real target (another checkout)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import tcip_mcp; print(tcip_mcp.__file__)"],
        cwd=str(worktree), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not import tcip_mcp under the worktree environment: {proc.stderr.strip()}")
    resolved = Path(proc.stdout.strip()).resolve()
    if not _is_under(resolved, worktree):
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
    parser.add_argument("--backend", choices=("sqlite", "file"), default="sqlite")
    parser.add_argument("--ruff", action="store_true")
    parser.add_argument("--mypy", action="store_true")
    parser.add_argument("--pytest", nargs="+", default=None, metavar="FILE")
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    if not worktree.is_dir():
        raise SystemExit(f"{worktree} is not a directory")

    env = build_environ(worktree, args.backend)
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

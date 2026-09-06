"""Run the quality gate CI actually declares, so a local pass means CI would pass too.

The stages below are parsed from ``.github/workflows/ci.yml``, not restated by hand, so a step
CI adds, drops, or reorders reaches this gate the next time it runs rather than waiting for
someone to notice the drift. Every stage runs even if an earlier one fails, so one red stage
does not hide the timing of the rest.

Three jobs are out of scope. ``environment`` creates the conda environment this gate's own
process already runs inside, so there is nothing local to run in its place; ``docker``
builds and answers the container image, which needs a Docker daemon this gate does not
assume, and CI's own build is the proof of that image; ``ray-exit-windows`` also creates that
same conda environment, and its one test file (``tests/test_hpo_ray_detached_exit.py``) is
Windows-only, so a Windows host's own suite already covers it and any other host has nothing
local to run in its place, the same shape as the ``environment`` job.

Each step's ``run:`` text goes through bash, never through PowerShell or cmd, since that is the
shell GitHub's own Linux runners give the same text. On a Linux or macOS host the bash on
``PATH`` is that shell. On Windows a bare ``bash`` resolved from ``PATH`` can be a WSL launcher
or a System32 shim, reading Windows paths differently, so the path this script resolves there is
Git's own bash.exe, derived from the ``git`` on this machine's PATH.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
# The one-leg pytest run recorded before this gate ran two backend legs; kept as history, not a
# comparable baseline, and never overwritten by a later run.
RECORDED_BASELINE = REPO_ROOT / "docs" / "audit" / "phase0" / "gate-baseline"

PARSED_JOBS = ("mypy", "python", "typescript")
OUT_OF_SCOPE_JOBS = {
    "environment": "it creates the conda environment this gate's own process already runs inside",
    "docker": "it builds the container image, which needs a Docker daemon this gate does not assume",
    "ray-exit-windows": (
        "it creates the conda environment this gate's own process already runs inside, and its "
        "one test file is Windows-only, so a Windows host's own suite already covers it and any "
        "other host has nothing local to run in its place"
    ),
}

_SKIP_PREFIXES = ("pip", "npm ci", "conda", "mamba")
_MATRIX_EXPR = re.compile(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}")
_ANY_EXPR = re.compile(r"\$\{\{.*?\}\}")


class UnresolvedExpressionError(Exception):
    """An ``${{ ... }}`` expression this script has no leg value for and will not pass through
    to a process as literal text."""


@dataclass
class Stage:
    """One executable unit of the plan: a job's step, for one matrix leg."""

    key: str
    job: str
    name: str
    run: "str | None"
    cwd: pathlib.Path
    env: "dict[str, str]"
    skip_reason: "str | None" = None


def _load_workflow() -> "dict[str, Any]":
    loaded: "dict[str, Any]" = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    return loaded


def _resolve_text(text: str, leg: "dict[str, Any]") -> str:
    """Substitute ``${{ matrix.<name> }}`` with the leg's own value; refuse by name on any
    other ``${{ ... }}`` expression rather than pass its literal text to a process."""

    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in leg:
            raise UnresolvedExpressionError(
                f"{match.group(0)!r} in {text!r} has no leg value among {sorted(leg)}"
            )
        return str(leg[key])

    resolved = _MATRIX_EXPR.sub(repl, text)
    stray = _ANY_EXPR.search(resolved)
    if stray:
        raise UnresolvedExpressionError(f"unresolvable expression {stray.group(0)!r} in {text!r}")
    return resolved


def _matrix_legs(job: "dict[str, Any]") -> "list[dict[str, Any]]":
    matrix = ((job.get("strategy") or {}).get("matrix") or {})
    axes = [(name, values) for name, values in matrix.items() if isinstance(values, list)]
    if not axes:
        return [{}]
    names = [name for name, _ in axes]
    return [dict(zip(names, combo)) for combo in itertools.product(*(v for _, v in axes))]


def _first_effective_line(run_text: str) -> str:
    for line in run_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _skip_reason(step: "dict[str, Any]") -> "str | None":
    if "uses" in step:
        return f"uses: {step['uses']}"
    first = _first_effective_line(step.get("run", ""))
    for prefix in _SKIP_PREFIXES:
        if first.startswith(prefix):
            return f"run: starts with {prefix!r}"
    return None


def build_plan() -> "list[Stage]":
    """The full stage list, one entry per step per matrix leg, in workflow order.

    Building this never touches a subprocess: it is safe to call from a test that wants to
    check what would run without actually running any of it. Each key is prefixed with its job
    name and matrix leg, so the same step name in two jobs (both carry an unnamed
    ``actions/checkout@v4``) never collides.
    """
    jobs = _load_workflow()["jobs"]
    stages: "list[Stage]" = []
    for job_name in PARSED_JOBS:
        job = jobs[job_name]
        job_env_raw = job.get("env") or {}
        for leg in _matrix_legs(job):
            leg_suffix = f" [{'/'.join(str(v) for v in leg.values())}]" if leg else ""
            job_env = {k: _resolve_text(str(v), leg) for k, v in job_env_raw.items()}
            for step in job.get("steps", []):
                name = step.get("name") or step.get("uses") or "<unnamed step>"
                run_text = step.get("run")
                cwd = REPO_ROOT
                working_dir = step.get("working-directory")
                if working_dir:
                    cwd = (REPO_ROOT / _resolve_text(working_dir, leg)).resolve()
                step_env = {k: _resolve_text(str(v), leg) for k, v in (step.get("env") or {}).items()}
                if run_text is not None:
                    run_text = _resolve_text(run_text, leg)
                    if job_name == "python" and name == "Run tests":
                        # -n auto scales to CI's runner; the documented local invocation is -n 4.
                        run_text = run_text.replace("-n auto", "-n 4")
                stages.append(Stage(
                    key=f"{job_name}:{name}{leg_suffix}", job=job_name, name=name, run=run_text,
                    cwd=cwd, env={**job_env, **step_env}, skip_reason=_skip_reason(step),
                ))
    return stages


_MYPY_PIN_RE = re.compile(r"mypy==([\w.]+)")


def mypy_pin() -> "str | None":
    """The mypy version CI pins, read from the (skipped) install step's own ``run:`` text."""
    for step in _load_workflow()["jobs"]["mypy"]["steps"]:
        match = _MYPY_PIN_RE.search(step.get("run", ""))
        if match:
            return match.group(1)
    return None


def _host_is_windows() -> bool:
    """Whether this host resolves bash the Windows way; a test pins this rather than ``os.name``,
    which ``pathlib`` reads to pick its path class."""
    return os.name == "nt"


def _resolve_git_bash() -> str:
    """The bash that runs each step's ``run:`` text.

    On a non-Windows host that is the ``bash`` on ``PATH``, the same shell the CI runner gives
    the text. On Windows it is Git Bash's own ``bash.exe``, derived from ``git --exec-path``
    (whose parent's parent is the Git installation root) rather than a bare ``PATH`` lookup,
    since a bare ``bash`` can resolve to WSL's launcher or a System32 shim on some machines; the
    hardcoded default install path is the stated fallback when the derivation finds nothing
    there. Refuses by name on either host when no bash is found.
    """
    if not _host_is_windows():
        found = shutil.which("bash")
        if found:
            return found
        raise SystemExit(
            "bash was not found on PATH; this gate runs each step's run: text through bash."
        )
    candidates: "list[pathlib.Path]" = []
    try:
        probe = subprocess.run(
            ["git", "--exec-path"], capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        probe = None
    if probe is not None and probe.returncode == 0 and probe.stdout.strip():
        git_root = pathlib.Path(probe.stdout.strip()).parent.parent
        candidates.append(git_root / "bin" / "bash.exe")
    candidates.append(pathlib.Path("C:/Program Files/Git/bin/bash.exe"))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(pathlib.Path(local_app_data) / "Programs" / "Git" / "bin" / "bash.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "Git Bash was not found (checked beside git --exec-path and the usual install "
        "locations); this gate runs each step's run: text through it, never through WSL's or "
        "cmd's own shell."
    )


def _safe_filename(key: str) -> str:
    return re.sub(r"[^\w.-]+", "_", key).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="directory to write per-stage output and summary.json into")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--only", action="append", default=None,
                        help="a stage key to run instead of the full plan (job:name[leg]); "
                             "repeat --only for more than one, since a key may itself carry a comma")
    args = parser.parse_args()

    if args.out.resolve() == RECORDED_BASELINE.resolve():
        parser.error(
            f"{RECORDED_BASELINE} is the recorded baseline (a one-leg pytest run, not "
            "comparable to this gate's two-leg run) and stays as history; pass a different --out."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    bash = _resolve_git_bash()
    plan = build_plan()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {stage.key for stage in plan}
        if unknown:
            parser.error(f"unknown stage keys: {sorted(unknown)}")
        plan = [stage for stage in plan if stage.key in wanted]

    results: "list[dict[str, Any]]" = []
    for stage in plan:
        if stage.skip_reason:
            print(f"{stage.key:<40} skipped: {stage.skip_reason}")
            results.append({"stage": stage.key, "job": stage.job, "skipped": True,
                            "skip_reason": stage.skip_reason})
            continue

        started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        clock = time.perf_counter()
        env = {**os.environ, **stage.env}
        assert stage.run is not None
        try:
            done = subprocess.run([bash, "-c", stage.run], cwd=str(stage.cwd), env=env,
                                  capture_output=True, text=True, timeout=args.timeout,
                                  check=False, shell=False)
            out, err, code, timed_out = done.stdout, done.stderr, done.returncode, False
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code, timed_out = -1, True
        duration = time.perf_counter() - clock

        stem = _safe_filename(stage.key)
        (args.out / f"{stem}.stdout.txt").write_text(out, encoding="utf-8")
        (args.out / f"{stem}.stderr.txt").write_text(err, encoding="utf-8")
        results.append({"stage": stage.key, "job": stage.job, "skipped": False,
                        "started": started, "duration_s": round(duration, 1),
                        "exit_code": code, "timed_out": timed_out})
        print(f"{stage.key:<40}{code:>5}{duration:>9.1f}s")

    pin = mypy_pin()
    mypy_stage_ran = any(r["job"] == "mypy" and not r["skipped"] for r in results)
    mypy_version_on_caller_path = None
    if mypy_stage_ran:
        try:
            probe = subprocess.run(["mypy", "--version"], capture_output=True, text=True,
                                   timeout=60, check=False)
            mypy_version_on_caller_path = (probe.stdout or probe.stderr).strip() or None
        except (OSError, subprocess.SubprocessError):
            mypy_version_on_caller_path = None

    total = round(sum(r.get("duration_s", 0.0) for r in results if not r["skipped"]), 1)
    summary = {
        "shell": bash,
        "shell_note": "Git Bash on Windows",
        "out_of_scope_jobs": OUT_OF_SCOPE_JOBS,
        "mypy_pin": pin,
        # A probe on this caller's own PATH, not read from the mypy stage's own subprocess
        # environment; None both when the probe fails and when the mypy stage did not run.
        "mypy_version_on_caller_path": mypy_version_on_caller_path,
        "total_duration_s": total,
        "stages": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ran = sum(1 for r in results if not r["skipped"])
    skipped = sum(1 for r in results if r["skipped"])
    print(f"\ntotal {total}s across {ran} run stage(s), {skipped} skipped -> {args.out / 'summary.json'}")
    return 0 if all(r["skipped"] or r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the full quality gate and record per-stage duration and output.

The gate's own wall-clock is a measurement, not overhead: a gate too slow to run is a
gate contributors will not run. Every stage runs even if an earlier one fails, so one
red stage does not hide the timing of the rest.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = REPO_ROOT / "packages/tcip-web/frontend"

NPM = shutil.which("npm") or "npm"

STAGES = [
    ("ruff", ["ruff", "check", "."], REPO_ROOT),
    # bare mypy reads its roots from mypy.ini's files list, the same source CI uses.
    ("mypy", ["mypy"], REPO_ROOT),
    ("pytest", ["pytest", "tests/", "-n", "auto", "--tb=short", "-q"], REPO_ROOT),
    ("npm-format-check", [NPM, "run", "format:check"], FRONTEND),
    ("npm-lint", [NPM, "run", "lint"], FRONTEND),
    ("npm-typecheck", [NPM, "run", "typecheck"], FRONTEND),
    ("npm-test", [NPM, "test"], FRONTEND),
    ("npm-build", [NPM, "run", "build"], FRONTEND),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path,
                        default=REPO_ROOT / "docs/audit/phase0/gate-baseline")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--only", default=None,
                        help="Comma-separated stage names to run instead of all of them.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    selected = STAGES
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - {s[0] for s in STAGES}
        if unknown:
            parser.error(f"unknown stages: {sorted(unknown)}")
        selected = [s for s in STAGES if s[0] in wanted]

    results = []
    for name, argv, cwd in selected:
        started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        clock = time.perf_counter()
        try:
            done = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True,
                                  timeout=args.timeout, check=False, shell=False)
            out, err, code, timed_out = done.stdout, done.stderr, done.returncode, False
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            code, timed_out = -1, True
        except OSError as exc:
            out, err, code, timed_out = "", f"launch failed: {exc}", -2, False

        duration = time.perf_counter() - clock
        # A rerun records alongside the earlier attempt rather than over it: the output of a
        # stage that failed once is the evidence for why, and losing it costs more than a file.
        slot = 1
        while (args.out / f"{name}.run{slot}.stdout.txt").exists():
            slot += 1
        (args.out / f"{name}.run{slot}.stdout.txt").write_text(out, encoding="utf-8")
        (args.out / f"{name}.run{slot}.stderr.txt").write_text(err, encoding="utf-8")
        row_files = {"stdout": f"{name}.run{slot}.stdout.txt", "stderr": f"{name}.run{slot}.stderr.txt"}
        row = {"stage": name, "argv": argv, "cwd": str(cwd), "started": started,
               "duration_s": round(duration, 1), "exit_code": code, "timed_out": timed_out,
               "stdout_lines": out.count("\n"), "stderr_lines": err.count("\n"),
               "output_files": row_files}
        results.append(row)
        print(f"{name:<20}{code:>5}{duration:>9.1f}s", flush=True)

    total = round(sum(r["duration_s"] for r in results), 1)
    summary = {"total_duration_s": total, "stages": results}
    name = "summary.json" if not args.only else "summary-partial.json"
    (args.out / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ntotal {total}s across {len(results)} stages -> {args.out / name}")
    return 0 if all(r["exit_code"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

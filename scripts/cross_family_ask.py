"""Pose one identical question to several agent harnesses and record comparable answers.

Each run writes a directory holding the exact prompt, the exact argv, the harness's
stdout and stderr, the extracted final response, and metadata describing what the
harness was and how long it took. Answers are only comparable when the conditions
they ran under are recorded alongside them, so nothing here is optional.

Conditions control what the harness could reach, not what it was asked. The question
text is byte-identical across families in every condition.

Runs on the standard library alone, so it needs no project environment and should be invoked as
``python scripts/cross_family_ask.py``. Wrapping it in ``conda run`` buys nothing and fails
outright wherever conda is absent from PATH, which reports a launched-nothing run as exit 0.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

LOCAL_APP_DATA = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData/Local"))


def resolve_harness(name: str, env_var: str, *fallbacks: pathlib.Path) -> str:
    """Find a harness executable by PATH, then an explicit override, then install defaults."""
    override = os.environ.get(env_var)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if candidate.exists():
            return str(candidate)
    return name


CODEX = resolve_harness(
    "codex", "TCIP_CODEX_BIN",
    pathlib.Path.home() / ".codex/packages/standalone/current/bin/codex.exe",
)
AGY = resolve_harness("agy", "TCIP_AGY_BIN", LOCAL_APP_DATA / "agy/bin/agy.exe")
CLAUDE = resolve_harness("claude", "TCIP_CLAUDE_BIN")

TCIP_MCP = {
    "mcpServers": {
        "tcip": {
            "command": "conda",
            "args": ["run", "-n", "tcip-agent", "--no-capture-output", "python", "-m", "tcip_mcp"],
        }
    }
}

CONDITIONS = {
    "as-shipped": {
        "description": "Each harness gets exactly what the repository gives it today.",
        "mcp": "tcip",
        "guidance": False,
    },
    "guidance-equalized": {
        "description": "Every harness is told to read CLAUDE.md and .github/skills/ first.",
        "mcp": "tcip",
        "guidance": True,
    },
    "no-tools": {
        "description": "No MCP servers. Measures what the repository alone conveys.",
        "mcp": "none",
        "guidance": False,
    },
}

PARITY = {
    "claude": {"model": "opus", "effort": "high"},
    "codex": {"model": None, "effort": "high"},
    "antigravity": {"model": "gemini-3.1-pro-high", "effort": None},
}
"""Frontier tier at the highest available reasoning effort, per family.

Parity here is approximate and the recorded metadata, not this table, is what a
comparison should be read against. Each harness takes effort differently: Claude Code
through `--effort`, which also accepts xhigh and max above the level set here; Codex
through the `model_reasoning_effort` config key; Antigravity baked into the model id,
with only high and low available at the Pro tier. Antigravity can also serve Claude and
GPT-OSS models, which must never be selected for a cross-family run: doing so compares
harnesses while appearing to compare families.
"""

GUIDANCE_PREFIX = (
    "Before you begin, read CLAUDE.md at the repository root and the SKILL.md files "
    "under .github/skills/ that are relevant to this task.\n\n"
)


def as_text(stream: object) -> str:
    """A captured stream as text. ``None`` and bytes are both real, and neither is an error.

    ``subprocess.run`` yields ``None`` for a stream a harness never wrote, and
    ``TimeoutExpired`` carries raw bytes even under ``text=True``.
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


def failed_run_meta(family: str, exc: BaseException) -> dict:
    """A metadata row for a run that raised, shaped like a real one so the summary still prints."""
    return {
        "question_id": None, "condition": None, "condition_description": None,
        "family": family, "executable": None, "harness_version": "unknown",
        "model_requested": None, "effort_requested": None,
        "started": now(), "duration_s": 0.0, "exit_code": -3, "timed_out": False,
        "response_chars": 0, "response_source": "runner_error",
        "runner_error": f"{type(exc).__name__}: {exc}",
    }


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def harness_version(exe: str | pathlib.Path, flag: str = "--version") -> str:
    try:
        out = subprocess.run(
            [str(exe), flag], capture_output=True, text=True, timeout=60, check=False
        )
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def build_claude(prompt_file: pathlib.Path, run_dir: pathlib.Path, cwd: pathlib.Path,
                 condition: dict, model: str | None, effort: str | None,
                 timeout: int) -> tuple[list[str], pathlib.Path | None]:
    argv = [
        CLAUDE,
        "--print",
        "--output-format", "json",
        "--permission-mode", "plan",
        "--no-session-persistence",
        "--add-dir", str(cwd),
    ]
    if effort:
        argv += ["--effort", effort]
    if condition["mcp"] == "tcip":
        cfg = run_dir / "mcp.json"
        cfg.write_text(json.dumps(TCIP_MCP, indent=2), encoding="utf-8")
        argv += ["--mcp-config", str(cfg), "--strict-mcp-config"]
    else:
        cfg = run_dir / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {}}, indent=2), encoding="utf-8")
        argv += ["--mcp-config", str(cfg), "--strict-mcp-config"]
    if model:
        argv += ["--model", model]
    return argv, None


def build_codex(prompt_file: pathlib.Path, run_dir: pathlib.Path, cwd: pathlib.Path,
                condition: dict, model: str | None, effort: str | None,
                timeout: int) -> tuple[list[str], pathlib.Path | None]:
    """Build a headless Codex invocation.

    Headless `codex exec` does not load locally spawned stdio MCP servers, verified
    against a server that is registered, enabled, and answers an initialize handshake
    when launched directly. Hosted servers do load. A `tcip` condition therefore yields
    a run with repository access and no TCIP tools, which is a real result to record
    rather than a failure to hide, but it is not a like-for-like MCP comparison.
    """
    last = run_dir / "final_message.txt"
    argv = [
        str(CODEX), "exec",
        "--sandbox", "read-only",
        "--cd", str(cwd),
        "--ephemeral",
        "--json",
        "--output-last-message", str(last),
    ]
    if condition["mcp"] != "tcip":
        argv += ["--ignore-user-config"]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["-c", f'model_reasoning_effort="{effort}"']
    argv += ["-"]
    return argv, last


def build_agy(prompt_file: pathlib.Path, run_dir: pathlib.Path, cwd: pathlib.Path,
              condition: dict, model: str | None, effort: str | None,
              timeout: int) -> tuple[list[str], pathlib.Path | None]:
    """Build a headless Antigravity invocation.

    MCP servers come from `~/.gemini/config/mcp_config.json` and are global, so the
    no-tools condition cannot be expressed per run the way it can for the others.
    Headless mode also cannot prompt for tool permission and auto-denies, so each
    tool must be named under `permissions.allow` in the CLI settings before it will
    run. That allow-list names individual read-only TCIP tools and everything
    unlisted stays auto-denied; if it is ever widened toward mutating tools, point
    runs at scratch project state only.
    """
    argv = [
        str(AGY),
        "--print", prompt_file.read_text(encoding="utf-8"),
        "--output-format", "json",
        "--sandbox",
        "--print-timeout", f"{timeout}s",
        "--add-dir", str(cwd),
    ]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    return argv, None


BUILDERS = {"claude": build_claude, "codex": build_codex, "antigravity": build_agy}
EXECUTABLES = {"claude": CLAUDE, "codex": CODEX, "antigravity": AGY}
STDIN_FAMILIES = {"claude", "codex"}


def denied_write_text(payload: object) -> str:
    """The longest document a harness tried to write and was refused, or an empty string.

    A read-only run can still make the model compose its answer as a file rather than as a reply:
    Claude Code under ``--permission-mode plan`` drafts into a plan file, is denied, and then
    replies with a short pointer to a document that was never saved. The draft survives in the
    refusal record, so the answer is recoverable without relaxing the sandbox that produced it.
    Without this the run reports success while the recorded answer is a sentence of apology.
    """
    if not isinstance(payload, dict):
        return ""
    best = ""
    for denial in payload.get("permission_denials") or ():
        if not isinstance(denial, dict):
            continue
        tool_input = denial.get("tool_input")
        if not isinstance(tool_input, dict):
            continue
        content = tool_input.get("content")
        if isinstance(content, str) and len(content) > len(best):
            best = content
    return best.strip()


def extract_response(family: str, stdout: str, last_message: pathlib.Path | None) -> tuple[str, str]:
    """Pull the final assistant text out of whatever shape the harness emitted.

    Returns the text and the route it came from. The route is recorded per run because an answer
    recovered from a refused write is not the same evidence as one the harness returned directly,
    and a comparison that cannot tell them apart is reading two different things as one.
    """
    if last_message and last_message.exists():
        text = last_message.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text, "final_message_file"
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict):
            drafted = denied_write_text(payload)
            for key in ("result", "response", "text", "content", "final_response"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    reply = value.strip()
                    if len(drafted) > len(reply):
                        return drafted, "denied_write_draft"
                    return reply, "result_field"
            if drafted:
                return drafted, "denied_write_draft"
        return json.dumps(payload, indent=2)[:20000], "whole_payload"
    tail = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("text", "message", "result", "delta"):
            value = event.get(key) if isinstance(event, dict) else None
            if isinstance(value, str):
                tail.append(value)
    if tail:
        return "\n".join(tail[-40:]), "stream_tail"
    return stdout.strip(), "raw_stdout"


def run_one(family: str, question_id: str, condition_name: str, prompt: str,
            cwd: pathlib.Path, out_root: pathlib.Path, timeout: int,
            model: str | None, effort: str | None) -> dict:
    condition = CONDITIONS[condition_name]
    run_dir = out_root / question_id / condition_name / family
    run_dir.mkdir(parents=True, exist_ok=True)

    body = (GUIDANCE_PREFIX + prompt) if condition["guidance"] else prompt
    prompt_file = run_dir / "prompt.txt"
    prompt_file.write_text(body, encoding="utf-8")

    resolved_model = model or PARITY[family]["model"]
    resolved_effort = effort or PARITY[family]["effort"]
    argv, last = BUILDERS[family](prompt_file, run_dir, cwd, condition,
                                  resolved_model, resolved_effort, timeout)

    (run_dir / "argv.txt").write_text("\n".join(argv), encoding="utf-8")

    started = now()
    clock = datetime.datetime.now()
    try:
        completed = subprocess.run(
            argv,
            input=body if family in STDIN_FAMILIES else None,
            capture_output=True,
            text=True,
            # Never the console codepage: cp1252 stdin kills a run on the first unencodable character.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd),
            check=False,
        )
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {timeout}s"
        code = -1
        timed_out = True
    except OSError as exc:
        stdout, stderr, code, timed_out = "", f"launch failed: {exc}", -2, False

    duration = (datetime.datetime.now() - clock).total_seconds()
    # A harness that produces nothing, or that a timeout captured as raw bytes, is an outcome to
    # record rather than a crash: the run's own transcript is what makes the comparison auditable.
    stdout, stderr = as_text(stdout), as_text(stderr)
    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    response, response_source = extract_response(family, stdout, last)
    (run_dir / "response.md").write_text(response, encoding="utf-8")

    meta = {
        "question_id": question_id,
        "condition": condition_name,
        "condition_description": condition["description"],
        "family": family,
        "executable": str(EXECUTABLES[family]),
        "harness_version": harness_version(EXECUTABLES[family]),
        "model_requested": resolved_model,
        "effort_requested": resolved_effort,
        "cwd": str(cwd),
        "mcp": condition["mcp"],
        "guidance_injected": condition["guidance"],
        "prompt_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "prompt_chars": len(body),
        "started": started,
        "duration_s": round(duration, 1),
        "exit_code": code,
        "timed_out": timed_out,
        "response_chars": len(response),
        "response_source": response_source,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--condition", default="as-shipped", choices=sorted(CONDITIONS))
    parser.add_argument("--families", default="claude,codex,antigravity")
    parser.add_argument("--cwd", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "docs/audit/cross-family")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default=None,
                        help="Override the per-family parity model for every harness.")
    parser.add_argument("--effort", default=None,
                        help="Override the per-family parity reasoning effort.")
    parser.add_argument("--serial", action="store_true", help="Run families one at a time.")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in BUILDERS]
    if unknown:
        parser.error(f"unknown families: {unknown}. Known: {sorted(BUILDERS)}")

    missing = []
    for family in families:
        exe = EXECUTABLES[family]
        found = shutil.which(str(exe)) or (pathlib.Path(exe).exists() and str(exe))
        if not found:
            missing.append(f"{family} ({exe})")
    if missing:
        parser.error("harness not found: " + "; ".join(missing))

    if CONDITIONS[args.condition]["mcp"] == "tcip" and "codex" in families:
        print("note: headless codex reaches no locally spawned MCP server, so its run in this "
              "condition has repository access only. Antigravity covers MCP questions for a "
              "second family.\n")

    prompt = args.prompt_file.read_text(encoding="utf-8-sig")
    print(f"question : {args.question_id}")
    print(f"condition: {args.condition} ({CONDITIONS[args.condition]['description']})")
    print(f"families : {', '.join(families)}")
    print(f"cwd      : {args.cwd}")
    print(f"out      : {args.out}")
    print()

    results = []
    if args.serial:
        for family in families:
            results.append(run_one(family, args.question_id, args.condition, prompt,
                                   args.cwd, args.out, args.timeout, args.model,
                                   args.effort))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(families)) as pool:
            futures = {
                pool.submit(run_one, family, args.question_id, args.condition, prompt,
                            args.cwd, args.out, args.timeout, args.model,
                            args.effort): family
                for family in families
            }
            for future in concurrent.futures.as_completed(futures):
                family = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    # One family failing must not discard the families that answered: their
                    # transcripts are the artifact, and a lost run is unrecoverable.
                    results.append(failed_run_meta(family, exc))

    results.sort(key=lambda r: r["family"])
    print(f"{'family':<14}{'exit':>6}{'secs':>9}{'chars':>9}  {'answer from':<20}version")
    for meta in results:
        print(f"{meta['family']:<14}{meta['exit_code']:>6}{meta['duration_s']:>9}"
              f"{meta['response_chars']:>9}  {meta['response_source']:<20}{meta['harness_version']}")
    recovered = [m['family'] for m in results if m['response_source'] == "denied_write_draft"]
    if recovered:
        print(f"\nnote: recovered the answer from a refused write for: {', '.join(recovered)}. "
              "That harness composed its answer as a file the sandbox denied, so its reply text "
              "was only a pointer to it.")

    summary = args.out / args.question_id / args.condition / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0 if all(m["exit_code"] == 0 for m in results) else 1


if __name__ == "__main__":
    sys.exit(main())

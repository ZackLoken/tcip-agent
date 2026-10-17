"""Pose one question to several agent harnesses and record comparable answers.

Each run writes a directory holding the exact prompt, the exact argv, the harness's
stdout and stderr, the extracted final response, and metadata describing what the
harness was and how long it took.

Conditions control what the harness could reach, not what it was asked. An attached image adds
one family-specific preface line ahead of the question, since each harness takes an image
differently.

Antigravity in headless mode auto-denies shell commands, so a prompt sent to it must be
answerable by file reads alone, with any material to review given as files inside the
working tree it is pointed at.

Runs on the standard library alone, so it needs no project environment and is invoked as
``python tools/cross_family_ask.py``.
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

import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

LOCAL_APP_DATA = pathlib.Path(
    os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData/Local")
)


def resolve_harness(name: str, env_var: str, *fallbacks: pathlib.Path) -> str:
    """Find a harness executable by the explicit override, then PATH, then install defaults."""
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
    "codex",
    "TCIP_CODEX_BIN",
    pathlib.Path.home() / ".codex/packages/standalone/current/bin/codex.exe",
)
AGY = resolve_harness(
    "agy", "TCIP_AGY_BIN", LOCAL_APP_DATA / "agy/bin/agy.exe"
)
CLAUDE = resolve_harness("claude", "TCIP_CLAUDE_BIN")


def load_tcip_mcp_config() -> dict:
    """Return the ``tcip`` MCP server block, read from the repo's own ``.mcp.json``.

    A change to that launch config is what a run actually sees. A missing or unreadable file, or
    one with no ``mcpServers.tcip`` block, raises ``SystemExit`` when this is called.
    """
    mcp_path = REPO_ROOT / ".mcp.json"
    try:
        declared = json.loads(mcp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"cannot read the tcip MCP server block from {mcp_path}: {exc}"
        ) from exc
    server = declared.get("mcpServers", {}).get("tcip")
    if server is None:
        raise SystemExit(
            f"{mcp_path} has no mcpServers.tcip block for the tcip condition to use"
        )
    return {"mcpServers": {"tcip": server}}


CONDITIONS = {
    "as-shipped": {
        "description": "Each harness gets exactly what the repository gives it today.",
        "mcp": "tcip",
        "guidance": False,
    },
    "guidance-equalized": {
        "description": "Every harness is told to read CLAUDE.md and the knowledge documents first.",
        "mcp": "tcip",
        "guidance": True,
    },
    "no-tools": {
        "description": "No MCP servers. Measures what the repository alone conveys.",
        "mcp": "none",
        "guidance": False,
    },
}

PARITY: dict[str, dict[str, str | None]] = {
    "claude": {"model": "opus", "effort": "high"},
    "codex": {"model": None, "effort": "high"},
    "antigravity": {"model": "gemini-3.1-pro-high", "effort": None},
}
"""The default model and reasoning effort per family, used where the command line names none."""

CODEX_CONFIG = pathlib.Path.home() / ".codex" / "config.toml"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


ANTIGRAVITY_ID_EFFORTS = ("low", "medium", "high")


def model_id_effort(family: str, model: str | None) -> str | None:
    """Return the effort a model id states in its own suffix, or None.

    Only a family whose harness takes effort that way (antigravity's `gemini-3.1-pro-high`) has
    one; every other id or family answers None.
    """
    if family == "antigravity" and model:
        suffix = model.rsplit("-", 1)[-1]
        if suffix in ANTIGRAVITY_ID_EFFORTS:
            return suffix
    return None


def effective_effort(
    family: str, model: str | None, effort: str | None
) -> str | None:
    """Return the effort a run records and, where the harness takes one, passes.

    A model id that states its own effort (:func:`model_id_effort`) is the effort, whatever
    ``effort`` says. For claude, a supplied, nonempty ``effort`` not in :data:`CLAUDE_EFFORTS`
    raises ``SystemExit``.
    """
    stated = model_id_effort(family, model)
    if stated is not None:
        return stated
    if family == "claude" and effort and effort not in CLAUDE_EFFORTS:
        raise SystemExit(
            f"claude effort {effort!r} is not one of {', '.join(CLAUDE_EFFORTS)}; the harness "
            "would ignore it and run on its default"
        )
    return effort


def resolve_codex_model(requested: str | None) -> tuple[str, str]:
    """Return the codex model name to pass explicitly and where it came from.

    An explicit request wins. Otherwise the `model` key of ~/.codex/config.toml is read. No
    resolvable name raises ``SystemExit``.
    """
    if requested:
        return requested, "flag"
    if CODEX_CONFIG.exists():
        with open(CODEX_CONFIG, "rb") as fh:
            model = tomllib.load(fh).get("model")
        if model:
            return str(model), str(CODEX_CONFIG)
    raise SystemExit(
        "codex model is unnamed: pass --model or set `model` in ~/.codex/config.toml; a run "
        "on the harness's built-in default would not record which model answered"
    )


def claude_model_used(stdout: str) -> str | None:
    """Return the model Claude Code reports it ran, from the JSON result's ``modelUsage`` keys."""
    stripped = stdout.strip()
    start = stripped.find("{")
    if start < 0:
        return None
    try:
        used = json.loads(stripped[start:]).get("modelUsage") or {}
    except (json.JSONDecodeError, AttributeError):
        return None
    return ",".join(sorted(used)) or None


def model_matches(requested: str, used: str) -> bool:
    """Return whether the model the harness reports corresponds to the one requested.

    A full id must appear as itself, an alias must appear inside every reported id.
    """
    return all(
        requested == name or requested in name for name in used.split(",")
    )


GUIDANCE_PREFIX = (
    "Before you begin, read CLAUDE.md at the repository root and the knowledge documents "
    "under packages/tcip-mcp/src/tcip_mcp/knowledge/ that are relevant to this task.\n\n"
)


def as_text(stream: object) -> str:
    """Return a captured stream as text.

    ``None`` becomes the empty string and bytes decode as UTF-8 with replacement.
    """
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


def failed_run_meta(family: str, exc: BaseException) -> dict:
    """Return a metadata row for a run that raised."""
    meta = {
        "question_id": None,
        "condition": None,
        "condition_description": None,
        "family": family,
        "executable": None,
        "harness_version": "unknown",
        "model_requested": None,
        "model_resolved": None,
        "model_source": None,
        "model_used": None,
        "model_mismatch": False,
        "effort_requested": None,
        "started": now(),
        "duration_s": 0.0,
        "exit_code": -3,
        "timed_out": False,
        "response_chars": 0,
        "response_source": "runner_error",
        "runner_error": f"{type(exc).__name__}: {exc}",
    }
    meta["fault"] = describe_fault(meta)
    return meta


def now() -> str:
    """Return the local time now as an ISO 8601 string to the second."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def harness_version(exe: str | pathlib.Path, flag: str = "--version") -> str:
    """Return the first line ``exe`` prints for ``flag``, or ``"unknown"`` when it cannot."""
    try:
        out = subprocess.run(
            [str(exe), flag],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def image_preface(family: str, images: tuple[pathlib.Path, ...]) -> str:
    """Return the line that hands attached images to one family, placed ahead of the question."""
    if not images:
        return ""
    names = [str(path) for path in images]
    if family == "antigravity":
        return (
            "Review the attached image(s) "
            + " ".join(f"@{n}" for n in names)
            + " before answering.\n\n"
        )
    if family == "codex":
        return (
            "The image(s) attached to this prompt are "
            + ", ".join(names)
            + ". Review them before answering.\n\n"
        )
    return (
        "Read the image(s) at "
        + ", ".join(names)
        + " with your image-capable file reader before answering.\n\n"
    )


def build_claude(
    prompt_file: pathlib.Path,
    run_dir: pathlib.Path,
    cwd: pathlib.Path,
    condition: dict,
    model: str | None,
    effort: str | None,
    timeout: int,
    images: tuple[pathlib.Path, ...] = (),
) -> tuple[list[str], pathlib.Path | None]:
    """Build a headless Claude Code invocation, writing its MCP config into ``run_dir``.

    The prompt goes on stdin. The run is in plan permission mode with ``cwd`` added; the MCP
    config holds the ``tcip`` server block under a ``tcip`` condition and no server otherwise.
    Returns the argv and no final-message file.
    """
    argv = [
        CLAUDE,
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "plan",
        "--no-session-persistence",
        "--add-dir",
        str(cwd),
    ]
    if effort:
        argv += ["--effort", effort]
    if condition["mcp"] == "tcip":
        cfg = run_dir / "mcp.json"
        cfg.write_text(
            json.dumps(load_tcip_mcp_config(), indent=2), encoding="utf-8"
        )
        argv += ["--mcp-config", str(cfg), "--strict-mcp-config"]
    else:
        cfg = run_dir / "mcp.json"
        cfg.write_text(
            json.dumps({"mcpServers": {}}, indent=2), encoding="utf-8"
        )
        argv += ["--mcp-config", str(cfg), "--strict-mcp-config"]
    if model:
        argv += ["--model", model]
    return argv, None


def build_codex(
    prompt_file: pathlib.Path,
    run_dir: pathlib.Path,
    cwd: pathlib.Path,
    condition: dict,
    model: str | None,
    effort: str | None,
    timeout: int,
    images: tuple[pathlib.Path, ...] = (),
) -> tuple[list[str], pathlib.Path | None]:
    """Build a headless Codex invocation; return its argv and its final-message file.

    The prompt goes on stdin (the trailing ``-``). Each image attaches through its own ``-i``
    flag, placed before the other options. The run is read-only in ``cwd``, with
    ``--ignore-user-config`` outside a ``tcip`` condition.
    """
    last = run_dir / "final_message.txt"
    argv = [str(CODEX), "exec"]
    for image in images:
        argv += ["-i", str(image)]
    argv += [
        "--sandbox",
        "read-only",
        "--cd",
        str(cwd),
        "--ephemeral",
        "--json",
        "--output-last-message",
        str(last),
    ]
    if condition["mcp"] != "tcip":
        argv += ["--ignore-user-config"]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["-c", f'model_reasoning_effort="{effort}"']
    argv += ["-"]
    return argv, last


def build_agy(
    prompt_file: pathlib.Path,
    run_dir: pathlib.Path,
    cwd: pathlib.Path,
    condition: dict,
    model: str | None,
    effort: str | None,
    timeout: int,
    images: tuple[pathlib.Path, ...] = (),
) -> tuple[list[str], pathlib.Path | None]:
    """Build a headless Antigravity invocation; return its argv and no final-message file.

    The prompt file's text is passed as ``--print``, with JSON output, the sandbox, a print
    timeout of ``timeout`` seconds and ``cwd`` added; ``condition`` is not read, so a run's MCP
    servers are whatever the harness's own configuration names.
    """
    argv = [
        str(AGY),
        "--print",
        prompt_file.read_text(encoding="utf-8"),
        "--output-format",
        "json",
        "--sandbox",
        "--print-timeout",
        f"{timeout}s",
        "--add-dir",
        str(cwd),
    ]
    if model:
        argv += ["--model", model]
    if effort:
        argv += ["--effort", effort]
    return argv, None


BUILDERS = {
    "claude": build_claude,
    "codex": build_codex,
    "antigravity": build_agy,
}
EXECUTABLES = {"claude": CLAUDE, "codex": CODEX, "antigravity": AGY}
STDIN_FAMILIES = {"claude", "codex"}


def describe_fault(meta: dict) -> str:
    """Return why this run does not count as an answer, or an empty string when it does.

    A nonzero exit code, a captured response that is empty once stripped of whitespace
    (``response_chars`` of zero), or a model mismatch each make a run no answer, whatever code
    path emptied the response out.
    """
    if meta["exit_code"] != 0:
        return f"the harness exited {meta['exit_code']}"
    if meta["response_chars"] == 0:
        return "it produced no answer text, so there is nothing to review"
    if meta.get("model_mismatch"):
        return (
            f"it ran on {meta.get('model_used')} after {meta['model_resolved']} was asked for, "
            "so the answer is not the comparison that was requested"
        )
    return ""


def print_verdict(faults: list[tuple[str, str]], total: int) -> None:
    """State the run's outcome as the last thing printed.

    With no faults, one ``RUN OK`` line goes to stdout. Otherwise a ``RUN FAILED`` line, one line
    per faulted family and a closing line go to stdout and stderr both.
    """
    if not faults:
        print(f"\nRUN OK: {total} of {total} families answered.")
        return
    lines = [
        f"\nRUN FAILED: {len(faults)} of {total} families produced no usable answer."
    ]
    lines += [f"  {family}: {why}" for family, why in faults]
    lines.append(
        "  Nothing here is reviewable evidence. Re-run before relying on it."
    )
    text = "\n".join(lines)
    print(text)
    print(text, file=sys.stderr)


def denied_write_text(payload: object) -> str:
    """Return the longest document a harness tried to write and was refused, or an empty string.

    The documents are the string ``content`` of each ``tool_input`` under the payload's
    ``permission_denials``; the longest is returned stripped.
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


# Largest recorded legitimate answer is near 39k characters; observed extraction failures on
# real runs sit at 127k and 303k characters, so 100k separates a real answer from a raw dump.
MAX_RAW_STREAM_CHARS = 100_000

ANSWER_KEYS = ("result", "response", "text", "content", "final_response")


def extract_response(
    family: str, stdout: str, last_message: pathlib.Path | None
) -> tuple[str, str]:
    """Pull the final assistant text out of whatever shape the harness emitted.

    Returns the text and the route it came from.
    """
    if last_message and last_message.exists():
        text = last_message.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        if text:
            return text, "final_message_file"
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(payload, dict):
            drafted = denied_write_text(payload)
            empty_string_answer = False
            non_string_answer = False
            for key in ANSWER_KEYS:
                if key not in payload:
                    continue
                value = payload[key]
                if isinstance(value, str):
                    if value.strip():
                        reply = value.strip()
                        if len(drafted) > len(reply):
                            return drafted, "denied_write_draft"
                        return reply, "result_field"
                    empty_string_answer = True
                else:
                    # A non-string answer value is an unknown shape: dump it whole rather
                    # than misread it as empty.
                    non_string_answer = True
            if drafted:
                return drafted, "denied_write_draft"
            if empty_string_answer and not non_string_answer:
                return "", "empty_response"
        return json.dumps(payload, indent=2)[:20000], "whole_payload"
    tail = []
    agent_messages = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                item_text = item.get("text")
                if isinstance(item_text, str):
                    agent_messages.append(item_text)
        for key in ("text", "message", "result", "delta"):
            value = event.get(key)
            if isinstance(value, str):
                tail.append(value)
    if agent_messages:
        return agent_messages[-1], "stream_agent_message"
    if tail:
        return "\n".join(tail[-40:]), "stream_tail"
    stripped = stdout.strip()
    if len(stripped) > MAX_RAW_STREAM_CHARS:
        return "", "extraction_failed"
    return stripped, "raw_stdout"


def run_one(
    family: str,
    question_id: str,
    condition_name: str,
    prompt: str,
    cwd: pathlib.Path,
    out_root: pathlib.Path,
    timeout: int,
    model: str | None,
    effort: str | None,
    images: list[pathlib.Path] | None = None,
    prompt_source: str | None = None,
) -> dict:
    """Run one family on one question under one condition and return its metadata row.

    The run's directory under ``out_root`` receives the prompt, the argv, stdout, stderr, the
    extracted response and ``meta.json``.
    """
    condition = CONDITIONS[condition_name]
    run_dir = out_root / question_id / condition_name / family
    # A prior meta.json here is this family's own earlier run under the same question and condition.
    rerun = (run_dir / "meta.json").is_file()
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_images = tuple(image.resolve() for image in images or [])
    body = image_preface(family, resolved_images) + (
        (GUIDANCE_PREFIX + prompt) if condition["guidance"] else prompt
    )
    prompt_file = run_dir / "prompt.txt"
    prompt_file.write_text(body, encoding="utf-8")

    resolved_model = model or PARITY[family]["model"]
    model_source = "flag" if model else "parity table"
    if family == "codex":
        resolved_model, model_source = resolve_codex_model(resolved_model)
    resolved_effort = effective_effort(
        family, resolved_model, effort or PARITY[family]["effort"]
    )
    argv, last = BUILDERS[family](
        prompt_file,
        run_dir,
        cwd,
        condition,
        resolved_model,
        resolved_effort,
        timeout,
        images=resolved_images,
    )

    (run_dir / "argv.txt").write_text("\n".join(argv), encoding="utf-8")

    started = now()
    clock = datetime.datetime.now()
    stdout: str | bytes
    stderr: str | bytes
    try:
        completed = subprocess.run(
            argv,
            input=body if family in STDIN_FAMILIES else None,
            capture_output=True,
            text=True,
            # Never the console codepage: cp1252 stdin kills a run on the first
            # unencodable character.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd),
            check=False,
        )
        stdout, stderr, code = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        # exc.stderr can be raw bytes even under text=True; normalize before concatenating the
        # str suffix below, rather than risking a bytes + str TypeError on this rare path.
        stderr = as_text(exc.stderr) + f"\nTIMEOUT after {timeout}s"
        code = -1
        timed_out = True
    except OSError as exc:
        stdout, stderr, code, timed_out = (
            "",
            f"launch failed: {exc}",
            -2,
            False,
        )

    duration = (datetime.datetime.now() - clock).total_seconds()
    # A harness that produces nothing, or that a timeout captured as raw bytes, is an outcome to
    # record rather than a crash: the run's own transcript is what makes the comparison auditable.
    stdout, stderr = as_text(stdout), as_text(stderr)
    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    response, response_source = extract_response(family, stdout, last)
    (run_dir / "response.md").write_text(response, encoding="utf-8")

    # Claude Code names the model it ran; the others echo nothing, so for them the request is
    # the record and a bogus value fails the run by name (see the PARITY note).
    model_used = claude_model_used(stdout) if family == "claude" else None
    model_mismatch = bool(
        model_used
        and resolved_model
        and not model_matches(resolved_model, model_used)
    )
    if model_mismatch:
        print(
            f"model mismatch: {family} ran {model_used} for a request of {resolved_model}; "
            "pass the model's full id",
            file=sys.stderr,
        )

    meta = {
        "question_id": question_id,
        "condition": condition_name,
        "condition_description": condition["description"],
        "family": family,
        "executable": str(EXECUTABLES[family]),
        "harness_version": harness_version(EXECUTABLES[family]),
        "model_requested": model,
        "model_resolved": resolved_model,
        "model_source": model_source,
        "model_used": model_used,
        "model_mismatch": model_mismatch,
        "effort_requested": resolved_effort,
        "cwd": str(cwd),
        "mcp": condition["mcp"],
        "guidance_injected": condition["guidance"],
        "prompt_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "prompt_chars": len(body),
        "prompt_source": prompt_source,
        "rerun": rerun,
        "images": [str(image) for image in resolved_images],
        "started": started,
        "duration_s": round(duration, 1),
        "exit_code": code,
        "timed_out": timed_out,
        # A whitespace-only capture is not an answer: strip before counting, though response.md
        # keeps the raw captured text.
        "response_chars": len(response.strip()),
        "response_source": response_source,
    }
    meta["fault"] = describe_fault(meta)
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    """Run the families the command line names; return 0 when every one answered, else 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument(
        "--condition", default="as-shipped", choices=sorted(CONDITIONS)
    )
    parser.add_argument("--families", default="claude,codex,antigravity")
    parser.add_argument("--cwd", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "docs/audit/cross-family",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--model",
        default=None,
        help="Override the per-family parity model for every harness.",
    )
    parser.add_argument(
        "--effort",
        default=None,
        help="Override the per-family parity reasoning effort.",
    )
    parser.add_argument(
        "--serial", action="store_true", help="Run families one at a time."
    )
    parser.add_argument(
        "--image",
        action="append",
        type=pathlib.Path,
        default=[],
        help="An image file to attach to the question; repeatable. Each family "
        "receives it the way that harness takes an image.",
    )
    parser.add_argument(
        "--family-prompt",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help="Override the shared --prompt-file for one family alone; repeatable, "
        "e.g. --family-prompt codex=docs/audit/codex-only-prompt.txt. Every "
        "other family still gets --prompt-file.",
    )
    args = parser.parse_args()

    missing_images = [
        str(image) for image in args.image if not image.is_file()
    ]
    if missing_images:
        parser.error("image not found: " + "; ".join(missing_images))

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in BUILDERS]
    if unknown:
        parser.error(f"unknown families: {unknown}. Known: {sorted(BUILDERS)}")

    family_prompt_files: dict[str, pathlib.Path] = {}
    for item in args.family_prompt:
        name, sep, path_text = item.partition("=")
        name = name.strip()
        if not sep or not name:
            parser.error(f"--family-prompt must be FAMILY=PATH, got {item!r}")
        if name not in BUILDERS:
            parser.error(
                f"--family-prompt names an unknown family {name!r}. Known: {sorted(BUILDERS)}"
            )
        path = pathlib.Path(path_text.strip())
        if not path.is_file():
            parser.error(f"--family-prompt file not found for {name}: {path}")
        family_prompt_files[name] = path

    missing = []
    for family in families:
        exe = EXECUTABLES[family]
        found = shutil.which(str(exe)) or (
            pathlib.Path(exe).exists() and str(exe)
        )
        if not found:
            missing.append(f"{family} ({exe})")
    if missing:
        parser.error("harness not found: " + "; ".join(missing))

    if CONDITIONS[args.condition]["mcp"] == "tcip" and "codex" in families:
        print(
            "note: headless codex reaches no locally spawned MCP server, so its run in this "
            "condition has repository access only. Antigravity covers MCP questions for a "
            "second family.\n"
        )

    prompt = args.prompt_file.read_text(encoding="utf-8-sig")
    prompt_source = str(args.prompt_file)

    def prompt_for(family: str) -> tuple[str, str]:
        override = family_prompt_files.get(family)
        if override is None:
            return prompt, prompt_source
        return override.read_text(encoding="utf-8-sig"), str(override)

    print(f"question : {args.question_id}")
    print(
        f"condition: {args.condition} ({CONDITIONS[args.condition]['description']})"
    )
    print(f"families : {', '.join(families)}")
    print(f"cwd      : {args.cwd}")
    print(f"out      : {args.out}")
    print()

    results = []
    if args.serial:
        for family in families:
            family_prompt, family_prompt_source = prompt_for(family)
            results.append(
                run_one(
                    family,
                    args.question_id,
                    args.condition,
                    family_prompt,
                    args.cwd,
                    args.out,
                    args.timeout,
                    args.model,
                    args.effort,
                    images=args.image,
                    prompt_source=family_prompt_source,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(families)
        ) as pool:
            futures = {}
            for family in families:
                family_prompt, family_prompt_source = prompt_for(family)
                futures[
                    pool.submit(
                        run_one,
                        family,
                        args.question_id,
                        args.condition,
                        family_prompt,
                        args.cwd,
                        args.out,
                        args.timeout,
                        args.model,
                        args.effort,
                        images=args.image,
                        prompt_source=family_prompt_source,
                    )
                ] = family
            for future in concurrent.futures.as_completed(futures):
                family = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    # One family failing must not discard the families that answered: their
                    # transcripts are the artifact, and a lost run is unrecoverable.
                    results.append(failed_run_meta(family, exc))

    results.sort(key=lambda r: r["family"])
    print(
        f"{'family':<14}{'exit':>6}{'secs':>9}{'chars':>9}  {'answer from':<20}"
        f"{'model (effort)':<34}version"
    )
    for meta in results:
        used = meta.get("model_used")
        model = (
            f"{used} for {meta['model_resolved']}"
            if meta.get("model_mismatch")
            else used or meta["model_resolved"] or "?"
        )
        print(
            f"{meta['family']:<14}{meta['exit_code']:>6}{meta['duration_s']:>9}"
            f"{meta['response_chars']:>9}  {meta['response_source']:<20}"
            f"{f'{model} ({meta['effort_requested']})':<34}{meta['harness_version']}"
        )
    recovered = [
        m["family"]
        for m in results
        if m["response_source"] == "denied_write_draft"
    ]
    if recovered:
        print(
            f"\nnote: recovered the answer from a refused write for: {', '.join(recovered)}. "
            "That harness composed its answer as a file the sandbox denied, so its reply text "
            "was only a pointer to it."
        )

    summary = args.out / args.question_id / args.condition / "summary.json"
    # A re-run naming a subset of families must not drop the rows of the families it did not
    # touch: merge onto whatever summary.json already holds, keyed by family.
    previous_rows: dict[str, dict] = {}
    if summary.is_file():
        try:
            previous = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = []
        for row in previous:
            previous_rows[row["family"]] = row
    for row in results:
        previous_rows[row["family"]] = row
    merged = sorted(previous_rows.values(), key=lambda r: r["family"])
    summary.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")
    faults = [(m["family"], m["fault"]) for m in results if m["fault"]]
    print_verdict(faults, len(results))
    return 0 if not faults else 1


if __name__ == "__main__":
    sys.exit(main())

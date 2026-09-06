"""Pose one identical question to several agent harnesses and record comparable answers.

Each run writes a directory holding the exact prompt, the exact argv, the harness's
stdout and stderr, the extracted final response, and metadata describing what the
harness was and how long it took. Answers are only comparable when the conditions
they ran under are recorded alongside them, so nothing here is optional.

Conditions control what the harness could reach, not what it was asked. The question
text is byte-identical across families in every condition; an attached image adds one
family-specific preface line ahead of it, since each harness takes an image differently.

Antigravity in headless mode auto-denies shell commands, so a prompt sent to it must be
answerable by file reads alone, with any material to review given as files inside the
working tree it is pointed at.

Runs on the standard library alone, so it needs no project environment and should be invoked as
``python tools/cross_family_ask.py``. Wrapping it in ``conda run`` buys nothing and fails
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
import tomllib

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

def load_tcip_mcp_config() -> dict:
    """The ``tcip`` MCP server block, read from the repo's own ``.mcp.json`` rather than
    restated here, so a change to that launch config is what a run actually sees.

    Raised at the point of use rather than at import, so a missing file or block fails the one
    run that needed it instead of a script nothing else in this condition requires can even load.
    """
    mcp_path = REPO_ROOT / ".mcp.json"
    try:
        declared = json.loads(mcp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read the tcip MCP server block from {mcp_path}: {exc}") from exc
    server = declared.get("mcpServers", {}).get("tcip")
    if server is None:
        raise SystemExit(f"{mcp_path} has no mcpServers.tcip block for the tcip condition to use")
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
"""Frontier tier at the highest available reasoning effort, per family.

Parity here is approximate and the recorded metadata, not this table, is what a
comparison should be read against. Each harness takes effort differently, verified by
execution against each one: Claude Code through `--effort` (low, medium, high, xhigh, max;
an unknown value is ignored with a warning and the run proceeds on the default, so the
runner refuses one before launching); Codex through the `model_reasoning_effort` config
key, which the API rejects by name when unknown; Antigravity baked into the model id
(`gemini-3.1-pro-high`, `gemini-3.1-pro-low`), and its separate `--effort` flag conflicts
with a suffixed id and fails the run, so it is never passed with one. Antigravity can also
serve Claude and GPT-OSS models, which must never be selected for a cross-family run:
doing so compares harnesses while appearing to compare families. Claude Code names the
model it used in its JSON result (`modelUsage`); Codex and Antigravity echo neither the
model nor the effort, so for them the recorded values are what the argv passed, and a
bogus model or effort fails the run by name rather than running on a default.

A codex model of None never reaches the harness: `resolve_codex_model` names it from the
user's config so the run records the model it used (with `--ignore-user-config` the harness
would otherwise fall back to an unrecorded built-in default).
"""

CODEX_CONFIG = pathlib.Path.home() / ".codex" / "config.toml"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


ANTIGRAVITY_ID_EFFORTS = ("low", "medium", "high")


def model_id_effort(family: str, model: str | None) -> str | None:
    """The effort a model id states in its own suffix, for a family whose harness takes effort
    that way (antigravity's `gemini-3.1-pro-high`); None for every other id or family."""
    if family == "antigravity" and model:
        suffix = model.rsplit("-", 1)[-1]
        if suffix in ANTIGRAVITY_ID_EFFORTS:
            return suffix
    return None


def effective_effort(family: str, model: str | None, effort: str | None) -> str | None:
    """The effort a run records and, where the harness takes one, passes.

    A model id that states its own effort is the effort, whatever `--effort` said beside it, so
    one invocation naming every family can carry one flag; Claude Code is refused an effort it
    would silently ignore and run on its default, which would record an effort the run did not
    use.
    """
    stated = model_id_effort(family, model)
    if stated is not None:
        return stated
    if family == "claude" and effort and effort not in CLAUDE_EFFORTS:
        raise SystemExit(
            f"claude effort {effort!r} is not one of {', '.join(CLAUDE_EFFORTS)}; the harness "
            "would ignore it and run on its default")
    return effort


def resolve_codex_model(requested: str | None) -> tuple[str, str]:
    """Return the codex model name to pass explicitly and where it came from.

    An explicit request wins. Otherwise the `model` key of ~/.codex/config.toml is read,
    because the no-tools condition runs codex with `--ignore-user-config` and the harness's
    own fallback is not reported anywhere. A run with no resolvable name is refused rather
    than recorded as model None.
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
        "on the harness's built-in default would not record which model answered")


def claude_model_used(stdout: str) -> str | None:
    """The model Claude Code reports it ran, from the JSON result's ``modelUsage`` keys.

    The harness is the one party that knows: under ``--permission-mode plan`` the ``haiku``
    alias was observed to run on Sonnet while the full Haiku id and the ``opus``, ``sonnet``
    and ``fable`` aliases ran as named, so the request is never trusted over this field.
    """
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
    """Whether the model the harness reports corresponds to the one requested: a full id must
    appear as itself, an alias must appear inside every reported id."""
    return all(requested == name or requested in name for name in used.split(","))


GUIDANCE_PREFIX = (
    "Before you begin, read CLAUDE.md at the repository root and the knowledge documents "
    "under packages/tcip-mcp/src/tcip_mcp/knowledge/ that are relevant to this task.\n\n"
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
    meta = {
        "question_id": None, "condition": None, "condition_description": None,
        "family": family, "executable": None, "harness_version": "unknown",
        "model_requested": None, "model_resolved": None, "model_source": None,
        "model_used": None, "model_mismatch": False, "effort_requested": None,
        "started": now(), "duration_s": 0.0, "exit_code": -3, "timed_out": False,
        "response_chars": 0, "response_source": "runner_error",
        "runner_error": f"{type(exc).__name__}: {exc}",
    }
    meta["fault"] = describe_fault(meta)
    return meta


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


def image_preface(family: str, images: tuple[pathlib.Path, ...]) -> str:
    """The line that hands attached images to one family, placed ahead of the question.

    Every family is vision-capable, and each takes an image its own way: codex attaches files
    through ``exec -i`` and is told their names, antigravity reads a file the prompt names as
    ``@path``, and Claude Code reads a named path with its own image-capable file reader. The
    question text after this preface stays byte-identical across families.
    """
    if not images:
        return ""
    names = [str(path) for path in images]
    if family == "antigravity":
        return "Review the attached image(s) " + " ".join(f"@{n}" for n in names) + " before answering.\n\n"
    if family == "codex":
        return "The image(s) attached to this prompt are " + ", ".join(names) + ". Review them before answering.\n\n"
    return ("Read the image(s) at " + ", ".join(names)
            + " with your image-capable file reader before answering.\n\n")


def build_claude(prompt_file: pathlib.Path, run_dir: pathlib.Path, cwd: pathlib.Path,
                 condition: dict, model: str | None, effort: str | None,
                 timeout: int, images: tuple[pathlib.Path, ...] = ()) -> tuple[list[str], pathlib.Path | None]:
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
        cfg.write_text(json.dumps(load_tcip_mcp_config(), indent=2), encoding="utf-8")
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
                timeout: int, images: tuple[pathlib.Path, ...] = ()) -> tuple[list[str], pathlib.Path | None]:
    """Build a headless Codex invocation.

    Headless `codex exec` does not load locally spawned stdio MCP servers, verified
    against a server that is registered, enabled, and answers an initialize handshake
    when launched directly. Hosted servers do load. A `tcip` condition therefore yields
    a run with repository access and no TCIP tools, which is a real result to record
    rather than a failure to hide, but it is not a like-for-like MCP comparison.

    Images attach through ``-i``, one flag per file and each followed by another flag: the
    option is variadic, so a bare ``-i a b`` would also swallow the trailing ``-`` that names
    stdin as the prompt.
    """
    last = run_dir / "final_message.txt"
    argv = [str(CODEX), "exec"]
    for image in images:
        argv += ["-i", str(image)]
    argv += [
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
              timeout: int, images: tuple[pathlib.Path, ...] = ()) -> tuple[list[str], pathlib.Path | None]:
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


def describe_fault(meta: dict) -> str:
    """Why this run does not count as an answer, or an empty string when it does.

    Computed once, at the point ``meta.json`` is written, and carried in the record itself under
    ``fault`` rather than re-derived later from ``response_source``/``exit_code`` at the aggregate
    table: a run whose captured response is empty once stripped of whitespace is never a
    successful answer, whatever code path emptied it out (a harness that writes nothing at all,
    an answer field present but blank, a stream whose only agent message is whitespace), so the
    check is on the stripped character count itself rather than a list of the response-source
    names known to produce one today.
    """
    if meta["exit_code"] != 0:
        return f"the harness exited {meta['exit_code']}"
    if meta["response_chars"] == 0:
        return "it produced no answer text, so there is nothing to review"
    if meta.get("model_mismatch"):
        return (f"it ran on {meta.get('model_used')} after {meta['model_resolved']} was asked for, "
                "so the answer is not the comparison that was requested")
    return ""


def print_verdict(faults: list[tuple[str, str]], total: int) -> None:
    """State the run's outcome as the last thing printed, on stdout and stderr both.

    The exit code alone is not enough. A caller that pipes this script into another command takes
    the pipeline's status from that command, so a refused run reads as a clean one and the only
    remaining signal is a zero in the chars column of a table that otherwise looks ordinary. This
    says the verdict in words at the end of the output, where it survives both a pipe and a tail.
    """
    if not faults:
        print(f"\nRUN OK: {total} of {total} families answered.")
        return
    lines = [f"\nRUN FAILED: {len(faults)} of {total} families produced no usable answer."]
    lines += [f"  {family}: {why}" for family, why in faults]
    lines.append("  Nothing here is reviewable evidence. Re-run before relying on it.")
    text = "\n".join(lines)
    print(text)
    print(text, file=sys.stderr)


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


# Largest recorded legitimate answer is near 39k characters; observed extraction failures on
# real runs sit at 127k and 303k characters, so 100k separates a real answer from a raw dump.
MAX_RAW_STREAM_CHARS = 100_000

ANSWER_KEYS = ("result", "response", "text", "content", "final_response")


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
                    # A non-string answer value is an unknown shape: dump it whole rather than misread it as empty.
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


def run_one(family: str, question_id: str, condition_name: str, prompt: str,
            cwd: pathlib.Path, out_root: pathlib.Path, timeout: int,
            model: str | None, effort: str | None,
            images: list[pathlib.Path] | None = None) -> dict:
    condition = CONDITIONS[condition_name]
    run_dir = out_root / question_id / condition_name / family
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_images = tuple(image.resolve() for image in images or [])
    body = image_preface(family, resolved_images) + (
        (GUIDANCE_PREFIX + prompt) if condition["guidance"] else prompt)
    prompt_file = run_dir / "prompt.txt"
    prompt_file.write_text(body, encoding="utf-8")

    resolved_model = model or PARITY[family]["model"]
    model_source = "flag" if model else "parity table"
    if family == "codex":
        resolved_model, model_source = resolve_codex_model(resolved_model)
    resolved_effort = effective_effort(family, resolved_model, effort or PARITY[family]["effort"])
    argv, last = BUILDERS[family](prompt_file, run_dir, cwd, condition,
                                  resolved_model, resolved_effort, timeout, images=resolved_images)

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
        # exc.stderr can be raw bytes even under text=True; normalize before concatenating the
        # str suffix below, rather than risking a bytes + str TypeError on this rare path.
        stderr = as_text(exc.stderr) + f"\nTIMEOUT after {timeout}s"
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

    # Claude Code names the model it ran; the others echo nothing, so for them the request is
    # the record and a bogus value fails the run by name (see the PARITY note).
    model_used = claude_model_used(stdout) if family == "claude" else None
    model_mismatch = bool(
        model_used and resolved_model and not model_matches(resolved_model, model_used))
    if model_mismatch:
        print(f"model mismatch: {family} ran {model_used} for a request of {resolved_model}; "
              "pass the model's full id", file=sys.stderr)

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
    parser.add_argument("--image", action="append", type=pathlib.Path, default=[],
                        help="An image file to attach to the question; repeatable. Each family "
                             "receives it the way that harness takes an image.")
    args = parser.parse_args()

    missing_images = [str(image) for image in args.image if not image.is_file()]
    if missing_images:
        parser.error("image not found: " + "; ".join(missing_images))

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
                                   args.effort, images=args.image))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(families)) as pool:
            futures = {
                pool.submit(run_one, family, args.question_id, args.condition, prompt,
                            args.cwd, args.out, args.timeout, args.model,
                            args.effort, images=args.image): family
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
    print(f"{'family':<14}{'exit':>6}{'secs':>9}{'chars':>9}  {'answer from':<20}"
          f"{'model (effort)':<34}version")
    for meta in results:
        used = meta.get("model_used")
        model = (f"{used} for {meta['model_resolved']}" if meta.get("model_mismatch")
                 else used or meta["model_resolved"] or "?")
        print(f"{meta['family']:<14}{meta['exit_code']:>6}{meta['duration_s']:>9}"
              f"{meta['response_chars']:>9}  {meta['response_source']:<20}"
              f"{f'{model} ({meta['effort_requested']})':<34}{meta['harness_version']}")
    recovered = [m['family'] for m in results if m['response_source'] == "denied_write_draft"]
    if recovered:
        print(f"\nnote: recovered the answer from a refused write for: {', '.join(recovered)}. "
              "That harness composed its answer as a file the sandbox denied, so its reply text "
              "was only a pointer to it.")

    summary = args.out / args.question_id / args.condition / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")
    faults = [(m["family"], m["fault"]) for m in results if m["fault"]]
    print_verdict(faults, len(results))
    return 0 if not faults else 1


if __name__ == "__main__":
    sys.exit(main())

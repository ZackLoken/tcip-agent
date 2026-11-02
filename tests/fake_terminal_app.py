"""A stand-in agent program for a terminal session.

Prints a banner naming the terminal session id it inherited, echoes each input line back as
``echo:<line>`` with any bracketed-paste markers removed, and exits on ``exit``. With
``FAKE_TERMINAL_ARGV_FILE`` set it first writes its arguments there as JSON, with
``FAKE_TERMINAL_ENV_FILE`` set its environment there as a JSON object, and with
``FAKE_TERMINAL_CWD_FILE`` set its working directory there. With
``FAKE_TERMINAL_PASTE_AFTER_S`` set it waits that many seconds, prints ``EARLY_INPUT`` if input
arrived meanwhile, then turns bracketed paste on and writes ``FAKE_TERMINAL_COMPOSER_READY``.
Run with ``fail`` as its first argument it exits 3 immediately.

With ``FAKE_TERMINAL_MEMORY_DIR`` set it stands in for Claude Code's row and the tcip server it
is handed: once its composer is ready it runs the ``SessionStart`` hook of its ``--settings``
file for its conversation (the id after ``--resume``, a resume, else a new one, a startup),
naming as its transcript ``<conversation>.jsonl`` in that directory and keeping the context the
hook prints and the ``CLAUDE.md`` of its working directory. A line naming a smoke token
(``ack-<hex>``) files one friction report of it followed by the display name that context states
and the instruction token that file states, in the project of its ``--mcp-config``, remembered
under the conversation in that directory; a line naming an edit token (``edit-<hex>``) and a
file appends the token to the file through this process, which no harness's deny list binds,
records the tool use and its result in the transcript in Claude Code's shape, and files one
report of the token followed by what happened; a line naming ``-continued`` files the remembered
token so continued.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

_TOKEN = re.compile(r"ack-[0-9a-f]+")
_EDIT = re.compile(r"edit-[0-9a-f]+")
_TARGET = re.compile(r"to the file (\S+) with")
_CONTINUED = "-continued"
_DISPLAY_NAME = re.compile(r'"display_name": "([^"]*)"')
_INSTRUCTION = re.compile(r"instruction token is (\S+)\.")
_SOURCES = os.pathsep.join(
    str(src)
    for src in sorted(
        (Path(__file__).resolve().parents[1] / "packages").glob("*/src")
    )
)
"""The package sources of the tree this program sits in, which its reports and its hook import."""


def _input_arrives_within(seconds: float) -> bool:
    """Return whether input is waiting on stdin before ``seconds`` pass, consuming none of it."""
    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                return True
            time.sleep(0.02)
        return False
    import select

    return bool(select.select([sys.stdin], [], [], seconds)[0])


def _argument_after(flag: str) -> str | None:
    """Return the argument after ``flag`` on this program's command line, ``None`` without it."""
    args = sys.argv[1:]
    return args[args.index(flag) + 1] if flag in args else None


def transcript_entry(role: str, item: dict, **fields: object) -> dict:
    """Return Claude Code's transcript entry of ``role`` carrying ``item``, ``fields`` beside."""
    return {
        "type": role,
        "message": {"role": role, "content": [item]},
    } | fields


class _Harness:
    """Claude Code's row as far as the smoke reaches it: its session hook and its reports."""

    def __init__(self, memory: Path) -> None:
        self.memory = memory
        sys.path[:0] = _SOURCES.split(os.pathsep)
        resumed = _argument_after("--resume")
        self.source = "startup" if resumed is None else "resume"
        self.conversation = resumed or str(uuid.uuid4())
        self.transcript = memory / f"{self.conversation}.jsonl"
        self.context = ""
        mcp_config = json.loads(
            Path(_argument_after("--mcp-config") or "").read_text(
                encoding="utf-8"
            )
        )
        args = mcp_config["mcpServers"]["tcip"]["args"]
        self.project = Path(args[args.index("--project") + 1])
        instructions = Path("CLAUDE.md")
        self.instructions = (
            instructions.read_text(encoding="utf-8")
            if instructions.is_file()
            else ""
        )

    def start(self) -> None:
        """Run each ``SessionStart`` hook of the ``--settings`` file, keeping what it prints.

        A handler with ``args`` is spawned with no shell, one without through a shell.
        """
        settings = json.loads(
            Path(_argument_after("--settings") or "").read_text(
                encoding="utf-8"
            )
        )
        for group in settings.get("hooks", {}).get("SessionStart", []):
            for handler in group["hooks"]:
                ran = subprocess.run(
                    [handler["command"], *handler["args"]]
                    if "args" in handler
                    else handler["command"],
                    shell="args" not in handler,
                    input=json.dumps(
                        {
                            "session_id": self.conversation,
                            "transcript_path": str(self.transcript),
                            "hook_event_name": "SessionStart",
                            "source": self.source,
                        }
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                    env={**os.environ, "PYTHONPATH": _SOURCES},
                )
                if ran.stdout.strip():
                    self.context += json.loads(ran.stdout)[
                        "hookSpecificOutput"
                    ]["additionalContext"]

    def take(self, line: str) -> None:
        """File the report ``line`` asks for, if it names a token or a continuation."""
        remembered = self.memory / self.conversation
        token = _TOKEN.search(line)
        edit = _EDIT.search(line)
        if token is not None:
            remembered.write_text(token.group(), encoding="utf-8")
            named = _DISPLAY_NAME.search(self.context)
            stated = _INSTRUCTION.search(self.instructions)
            self._report(
                f"smoke {token.group()} {named.group(1) if named else ''} "
                f"{stated.group(1) if stated else ''}"
            )
        elif edit is not None and (target := _TARGET.search(line)) is not None:
            outcome = self._edit(Path(target.group(1)), edit.group())
            self._report(f"{edit.group()} {outcome}")
        elif _CONTINUED in line:
            self._report(remembered.read_text(encoding="utf-8") + _CONTINUED)

    def _edit(self, target: Path, line: str) -> str:
        """Append ``line`` to ``target`` through this process; record the try; return its outcome.

        The tool use and its result go to the conversation's transcript in Claude Code's shape.
        """
        use = f"toolu_{uuid.uuid4().hex}"
        self._record(
            "assistant",
            {
                "type": "tool_use",
                "id": use,
                "name": "Edit",
                "input": {"file_path": str(target)},
            },
        )
        try:
            with target.open("a", encoding="utf-8") as file:
                file.write(f"{line}\n")
            outcome, failed = f"The file {target} has been updated.", False
        except OSError as exc:
            outcome, failed = str(exc), True
        self._record(
            "user",
            {
                "type": "tool_result",
                "tool_use_id": use,
                "content": outcome,
                "is_error": failed,
            },
        )
        return outcome

    def _record(self, role: str, item: dict) -> None:
        """Append one entry of ``role`` carrying ``item`` to the conversation's transcript."""
        with self.transcript.open("a", encoding="utf-8") as file:
            file.write(json.dumps(transcript_entry(role, item)) + "\n")

    def _report(self, detail: str) -> None:
        import tcip_store
        from tcip_mcp.tools.meta_tools import report_friction

        tcip_store.bind()
        report_friction(
            self.project, "unexpected_behavior", detail, actor=None
        )


def main() -> None:
    argv_file = os.environ.get("FAKE_TERMINAL_ARGV_FILE")
    if argv_file:
        Path(argv_file).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
    env_file = os.environ.get("FAKE_TERMINAL_ENV_FILE")
    if env_file:
        Path(env_file).write_text(
            json.dumps(dict(os.environ)), encoding="utf-8"
        )
    cwd_file = os.environ.get("FAKE_TERMINAL_CWD_FILE")
    if cwd_file:
        Path(cwd_file).write_text(os.getcwd(), encoding="utf-8")
    if sys.argv[1:2] == ["fail"]:
        print("FAKE_PREPARATION_FAILED", file=sys.stderr, flush=True)
        sys.exit(3)
    memory = os.environ.get("FAKE_TERMINAL_MEMORY_DIR")
    harness = _Harness(Path(memory)) if memory else None
    print("FAKE_TERMINAL_READY", flush=True)
    print(
        f"[session:{os.environ.get('TCIP_TERMINAL_SESSION', '')}]", flush=True
    )
    paste_after = os.environ.get("FAKE_TERMINAL_PASTE_AFTER_S")
    if paste_after is not None:
        if _input_arrives_within(float(paste_after)):
            print("EARLY_INPUT", flush=True)
        print(
            "\x1b[?2004h" + os.environ["FAKE_TERMINAL_COMPOSER_READY"],
            end="",
            flush=True,
        )
    if harness is not None:
        harness.start()
    for line in sys.stdin:
        text = line.replace("\x1b[200~", "").replace("\x1b[201~", "").strip()
        if text == "exit":
            print("FAKE_TERMINAL_BYE", flush=True)
            return
        if text:
            print(f"echo:{text}", flush=True)
            if harness is not None:
                harness.take(text)


if __name__ == "__main__":
    main()

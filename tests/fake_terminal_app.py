"""A stand-in agent program for a terminal session.

Prints a banner naming the terminal session id it inherited, echoes each input line back as
``echo:<line>`` with any bracketed-paste markers removed, and exits on ``exit``. With
``FAKE_TERMINAL_ARGV_FILE`` set it first writes its arguments there as JSON, and with
``FAKE_TERMINAL_CWD_FILE`` set its working directory there. With
``FAKE_TERMINAL_PASTE_AFTER_S`` set it waits that many seconds, prints ``EARLY_INPUT`` if input
arrived meanwhile, then turns bracketed paste on and writes ``FAKE_TERMINAL_COMPOSER_READY``.
Run with ``fail`` as its first argument it exits 3 immediately.

With ``FAKE_TERMINAL_MEMORY_DIR`` set it stands in for Claude Code's row and the tcip server it
is handed: once its composer is ready it runs the ``SessionStart`` hook of its ``--settings``
file for its conversation (the id after ``--resume``, else a new one), and a line naming a smoke
token (``ack-<hex>``) files one friction report of it in the project of its ``--mcp-config``,
remembered under the conversation in that directory, while a line naming ``-continued`` files
the remembered token so continued.
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
_CONTINUED = "-continued"


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


class _Harness:
    """Claude Code's row as far as the smoke reaches it: its session hook and its reports."""

    def __init__(self, memory: Path) -> None:
        self.memory = memory
        self.conversation = _argument_after("--resume") or str(uuid.uuid4())
        mcp_config = json.loads(
            Path(_argument_after("--mcp-config") or "").read_text(
                encoding="utf-8"
            )
        )
        args = mcp_config["mcpServers"]["tcip"]["args"]
        self.project = Path(args[args.index("--project") + 1])

    def start(self) -> None:
        """Run each ``SessionStart`` hook command of the ``--settings`` file, for the session."""
        settings = json.loads(
            Path(_argument_after("--settings") or "").read_text(
                encoding="utf-8"
            )
        )
        for group in settings.get("hooks", {}).get("SessionStart", []):
            for handler in group["hooks"]:
                subprocess.run(
                    handler["command"],
                    shell=True,
                    input=json.dumps(
                        {
                            "session_id": self.conversation,
                            "hook_event_name": "SessionStart",
                        }
                    ),
                    text=True,
                    check=False,
                )

    def take(self, line: str) -> None:
        """File the report ``line`` asks for, if it names a token or a continuation."""
        remembered = self.memory / self.conversation
        token = _TOKEN.search(line)
        if token is not None:
            remembered.write_text(token.group(), encoding="utf-8")
            self._report(f"smoke {token.group()}")
        elif _CONTINUED in line:
            self._report(remembered.read_text(encoding="utf-8") + _CONTINUED)

    def _report(self, detail: str) -> None:
        import tcip_store
        from tcip_mcp.tools.meta_tools import report_friction

        tcip_store.bind()
        report_friction(self.project, "unexpected_behavior", detail)


def main() -> None:
    argv_file = os.environ.get("FAKE_TERMINAL_ARGV_FILE")
    if argv_file:
        Path(argv_file).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
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

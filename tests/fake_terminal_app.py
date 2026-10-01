"""A stand-in agent for the terminal tests, run under a real PTY.

Prints a banner naming the terminal session id it inherited, echoes each input line back as
``echo:<line>`` with any bracketed-paste markers removed, and exits on ``exit``. With
``FAKE_TERMINAL_ARGV_FILE`` set it first writes its arguments there as JSON. With
``FAKE_TERMINAL_PASTE_AFTER_S`` set it waits that many seconds, prints ``EARLY_INPUT`` if input
arrived meanwhile, then turns bracketed paste on.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _input_arrives_within(seconds: float) -> bool:
    """Whether any input is waiting on stdin before ``seconds`` pass, consuming none of it."""
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


def main() -> None:
    argv_file = os.environ.get("FAKE_TERMINAL_ARGV_FILE")
    if argv_file:
        Path(argv_file).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
    print("FAKE_TERMINAL_READY", flush=True)
    print(f"[session:{os.environ.get('TCIP_TERMINAL_SESSION', '')}]", flush=True)
    paste_after = os.environ.get("FAKE_TERMINAL_PASTE_AFTER_S")
    if paste_after is not None:
        if _input_arrives_within(float(paste_after)):
            print("EARLY_INPUT", flush=True)
        print("\x1b[?2004h", end="", flush=True)
    for line in sys.stdin:
        text = line.replace("\x1b[200~", "").replace("\x1b[201~", "").strip()
        if text == "exit":
            print("FAKE_TERMINAL_BYE", flush=True)
            return
        if text:
            print(f"echo:{text}", flush=True)


if __name__ == "__main__":
    main()

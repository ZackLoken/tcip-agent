"""A tiny interactive program for the agent-terminal tests: a test double at the
process boundary (injected via ``TCIP_TERMINAL_CMD``), not a code path in the product.

It behaves like any TUI-ish CLI under a PTY: prints a banner naming the terminal session id it
inherited, echoes each input line back with a marker, and exits on ``exit``. Cross-platform (runs
under ConPTY on Windows and a POSIX pty on CI).
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    print("FAKE_TERMINAL_READY", flush=True)
    print(f"[session:{os.environ.get('TCIP_TERMINAL_SESSION', '')}]", flush=True)
    for line in sys.stdin:
        text = line.strip()
        if text == "exit":
            print("FAKE_TERMINAL_BYE", flush=True)
            return
        if text:
            print(f"echo:{text}", flush=True)


if __name__ == "__main__":
    main()

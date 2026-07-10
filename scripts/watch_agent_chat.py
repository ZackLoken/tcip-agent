"""Read the in-app TCIP agent chat from the orchestrating Claude Code session.

The in-app (breeder-lane) agent is a *separate*, fenced ``claude`` process. It logs its
conversation to ``~/.claude/projects/<encoded-repo>/<session>.jsonl`` in real time. This
read-only helper finds that session and prints the clean conversation — so the orchestrating
agent (me) can see the in-app chat with no copy-paste. It replaces the clipboard round-trip
that broke when the embedded terminal couldn't copy.

By default it shows the newest session that ISN'T the one driving this script (that session's
transcript contains this file's name), so ``python scripts/watch_agent_chat.py`` just shows
"the other agent's chat" — the in-app one.

Usage (repo root, tcip-agent env):
    python scripts/watch_agent_chat.py                 # newest in-app session, full transcript
    python scripts/watch_agent_chat.py --list          # recent sessions to pick from
    python scripts/watch_agent_chat.py --session <id>  # a specific session (id or prefix)
    python scripts/watch_agent_chat.py --tail 12       # only the last N turns (big sessions)
    python scripts/watch_agent_chat.py --wait          # block until a NEW message, print it, exit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Never crash on a non-ASCII glyph under a cp1252 console; degrade gracefully.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass

_MARKER = "watch_agent_chat"  # a session whose transcript mentions this file is the driver (me)


def _projects_dir() -> Path:
    """The ``~/.claude/projects/<encoded-repo>`` directory for this repo."""
    repo = Path(__file__).resolve().parents[1]
    base = Path.home() / ".claude" / "projects"
    # Claude Code encodes the cwd by replacing path separators / colon / space with '-'.
    encoded = repo.as_posix().replace("/", "-").replace(":", "-").replace(" ", "-")
    cand = base / encoded
    if cand.is_dir():
        return cand
    # Fallback: the project dir whose name ends with the repo's folder name.
    if base.is_dir():
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and d.name.endswith(repo.name):
                return d
    return cand  # may not exist; callers handle empty


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + f" …[+{len(text) - n} chars]"


def _turns(lines: list[str]) -> list[tuple[str, str]]:
    """Extract ('user'|'claude'|'tool', text) turns from a transcript's JSONL lines."""
    out: list[tuple[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        t = e.get("type")
        content = (e.get("message") or {}).get("content")
        if t == "user":
            if isinstance(content, str):
                txt = content
            elif isinstance(content, list):
                txt = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                )
            else:
                txt = ""
            if txt.strip():
                out.append(("user", txt))
        elif t == "assistant" and isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text" and c.get("text", "").strip():
                    out.append(("claude", c["text"]))
                elif c.get("type") == "tool_use":
                    inp = c.get("input", {}) or {}
                    hint = inp.get("command") or inp.get("prompt") or inp.get("file_path") or inp.get("description") or ""
                    out.append(("tool", f"{c.get('name', '?')}: {_clip(str(hint), 120)}"))
    return out


def _first_user(lines: list[str]) -> str:
    for role, txt in _turns(lines):
        if role == "user":
            return txt
    return ""


def _n_msgs(lines: list[str]) -> int:
    return sum(1 for r, _ in _turns(lines) if r in ("user", "claude"))


def _sessions(pdir: Path) -> list[Path]:
    if not pdir.is_dir():
        return []
    return sorted(pdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def _driver_id(sessions: list[Path]) -> str | None:
    """The session running this script — its transcript mentions this file's name."""
    for s in sessions:
        try:
            if _MARKER in s.read_text(encoding="utf-8", errors="ignore"):
                return s.stem
        except OSError:
            continue
    return None


def _pick(sessions: list[Path], want: str | None, exclude: str | None) -> Path | None:
    if want:
        for s in sessions:
            if s.stem.startswith(want):
                return s
        return None
    for s in sessions:
        if exclude and s.stem == exclude:
            continue
        return s
    return None


def _print_convo(path: Path, tail: int | None, from_turn: int = 0) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    turns = _turns(lines)
    shown = turns[from_turn:]
    if tail:
        shown = shown[-tail:]
    for role, txt in shown:
        if role == "user":
            print("\n#### USER ####\n" + _clip(txt, 2500))
        elif role == "claude":
            print("\n>> CLAUDE: " + _clip(txt, 1500))
        else:
            print("   [tool] " + txt)


def _cmd_list(sessions: list[Path], driver: str | None) -> None:
    print(f"Recent sessions in {_projects_dir()}\n")
    for s in sessions[:12]:
        try:
            lines = s.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        mtime = time.strftime("%m-%d %H:%M", time.localtime(s.stat().st_mtime))
        tag = "  <- this (driver) session" if s.stem == driver else ""
        print(f"  {s.stem[:8]}  {mtime}  {_n_msgs(lines):>3} msgs  {int(s.stat().st_size / 1024):>5} KB{tag}")
        fu = _first_user(lines)
        if fu:
            print(f"           first: {_clip(fu, 100)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Read the in-app TCIP agent chat.")
    ap.add_argument("--list", action="store_true", help="list recent sessions and exit")
    ap.add_argument("--session", help="show a specific session (id or prefix)")
    ap.add_argument("--tail", type=int, help="show only the last N turns")
    ap.add_argument("--wait", action="store_true", help="block until a new message, print it, exit")
    ap.add_argument("--timeout", type=float, default=1800.0, help="--wait timeout seconds (default 1800)")
    ap.add_argument("--exclude", help="session id to skip (default: auto-detect the driver session)")
    args = ap.parse_args()

    pdir = _projects_dir()
    sessions = _sessions(pdir)
    if not sessions:
        print(f"No agent sessions found under {pdir}")
        return 1
    driver = args.exclude or _driver_id(sessions)

    if args.list:
        _cmd_list(sessions, driver)
        return 0

    target = _pick(sessions, args.session, driver)
    if target is None:
        print("No matching in-app session (only the driver session exists?). Try --list.")
        return 1

    print(f"# in-app agent session {target.stem[:8]}  ({time.strftime('%m-%d %H:%M', time.localtime(target.stat().st_mtime))})")

    if not args.wait:
        _print_convo(target, args.tail)
        return 0

    # --wait: block until the transcript grows, then print only the new turns.
    baseline = len(_turns(target.read_text(encoding="utf-8", errors="replace").splitlines()))
    deadline = args.timeout
    waited = 0.0
    while waited < deadline:
        time.sleep(2.0)
        waited += 2.0
        turns = _turns(target.read_text(encoding="utf-8", errors="replace").splitlines())
        if len(turns) > baseline:
            print(f"\n--- {len(turns) - baseline} new turn(s) ---")
            _print_convo(target, None, from_turn=baseline)
            return 0
    print("(no new messages within the wait window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

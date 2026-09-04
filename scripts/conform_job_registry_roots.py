"""Conform a platform root's persisted job-registry documents to carry ``platform_root`` on
every summary they hold.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md), for the three job registries ``jobstore.JOB_REGISTRY_DOCUMENTS`` names (inference,
review priority queue, HPO). Each document lives at the key
``jobstore.job_registry_key(name, root=root)``, so the root a summary predating the
``platform_root`` field actually launched under is the document's own root -- the root this
script is pointed at -- not a value guessed at rehydrate time. ``jobstore.persist_grouped`` and
every registry's ``_from_summary`` now refuse a summary carrying no ``platform_root``, naming
this script.

    python scripts/conform_job_registry_roots.py <platform_root> [<platform_root> ...]
    python scripts/conform_job_registry_roots.py --plan <platform_root>

Exit codes: 0 conformed (or nothing to conform) for every root named; 2 if any named root holds
no ``.tcip`` directory. Every root named on the command line is still processed after another
root's refusal. The write, when there is one, happens inside the same ``tcip_store.transaction``
lock a live registry's own persist takes on the document's key, so it can never race a running
process's own write.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-web" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402
from tcip_web.jobstore import JOB_REGISTRY_DOCUMENTS, job_registry_key  # noqa: E402


def _summary_id(summary: dict) -> str:
    """The id a summary carries under whichever field this registry uses: ``job_id`` for
    inference and the review priority queue, ``sweep_id`` for HPO."""
    return summary.get("job_id") or summary.get("sweep_id") or "<unknown>"


def _conform_document(name: str, root: Path, *, plan: bool) -> list[str]:
    """Every outcome line for one job-registry document (``name``) under ``root``."""
    key = job_registry_key(name, root=root)

    if plan:
        summaries = ts.read(key, default=[])
        if not isinstance(summaries, list):
            return []
        outcomes = []
        for s in summaries:
            if not isinstance(s, dict):
                continue
            if "platform_root" in s:
                outcomes.append(f"{name}/{_summary_id(s)}: already carries platform_root, unchanged")
            else:
                outcomes.append(
                    f"{name}/{_summary_id(s)}: would stamp platform_root to {key.root} "
                    "(the document's own root)")
        return outcomes

    with ts.transaction(key) as txn:
        summaries = txn.read(key, default=[])
        if not isinstance(summaries, list):
            return []
        outcomes = []
        changed = False
        for s in summaries:
            if not isinstance(s, dict):
                continue
            if "platform_root" in s:
                outcomes.append(f"{name}/{_summary_id(s)}: already carries platform_root, unchanged")
                continue
            s["platform_root"] = key.root
            changed = True
            outcomes.append(
                f"{name}/{_summary_id(s)}: stamped platform_root to {key.root} "
                "(the document's own root)")
        if changed:
            txn.write(key, summaries)
    return outcomes


def conform_root(root: Path, *, plan: bool) -> tuple[list[str], bool]:
    """Every outcome line for ``root``, across every job-registry document, and whether it was
    refused (no ``.tcip`` directory)."""
    if not (root / ".tcip").is_dir():
        return ["refused, no .tcip directory found; not a project root"], True

    outcomes: list[str] = []
    for name in JOB_REGISTRY_DOCUMENTS:
        outcomes.extend(_conform_document(name, root, plan=plan))
    return outcomes, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = conform_root(root, plan=args.plan)
        if refused:
            refused_any = True
        if not outcomes:
            outcomes = ["nothing to conform"]
        for line in outcomes:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())

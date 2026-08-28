"""Conform a project's registry entries to carry ``experiment_id``, the field naming which run's
completion bound an entry (``None`` for an entry no run bound). A deliberate one-off operator
script, per this repo's no-backward-compatibility boundary (CLAUDE.md): the producer-binding
family's readers (``load_registered_checkpoint``'s producer resolution, an entry's own eviction
rail) now require the field, so an entry that predates it needs conforming before either can tell
a genuinely-unbound entry from one whose binding was simply never recorded. Never runs as part of
any runtime path.

For each project root named, every entry in ``.tcip/models/registry.json`` that lacks
``experiment_id`` is conformed:

  * an entry with no ``experiment:<id>`` tag gets ``experiment_id=null``: nothing on it claims a
    producer at all.
  * an entry carrying an ``experiment:<id>`` tag is refused by name: the tag was caller-asserted,
    never verified, and no run record exists to check it against. A tagged entry a real run
    actually produced should be re-registered through ``register_model_from_experiment`` instead,
    which records the verified binding; conforming it to null here would erase the (unverified)
    claim the tag made rather than resolve it.

    python scripts/conform_registry_experiment_id.py <project_root> [<project_root> ...]
    python scripts/conform_registry_experiment_id.py --plan <project_root>

Writes go through ``tcip_store.transaction`` on the registry index key, the same locked
read-modify-write ``register_entry`` uses, never a re-registration (which would overwrite the
entry's metrics and reset ``registered_at``, and can fail on a missing checkpoint). A root already
conformed (every entry already carries ``experiment_id``) is reported unchanged.

Exit codes: 0 if every root named was conformed or had nothing to conform; 2 if any root carries
an entry refused for holding an ``experiment:`` tag with no verified run to bind it to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

from tcip_mcp.model_registry import registry_index_key  # noqa: E402


def _experiment_tag(tags) -> str | None:
    """The ``experiment:<id>`` an entry's tags name, or ``None`` when it carries no such tag."""
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith("experiment:"):
            return tag.split(":", 1)[1]
    return None


def _conform_entry(entry: dict, *, plan: bool) -> tuple[dict | None, str]:
    """The entry as conformed (``None`` when refused), plus one outcome line."""
    name = entry.get("name")
    verb = "would set" if plan else "set"
    tag_experiment = _experiment_tag(entry.get("tags"))
    if tag_experiment is None:
        return {**entry, "experiment_id": None}, f"{name}: {verb} experiment_id=null (no experiment: tag)"
    return None, (f"{name}: refused, carries experiment:{tag_experiment} with no run record to "
                  "verify it against; re-register it through register_model_from_experiment instead")


def _plan(root: Path) -> tuple[list[str], bool]:
    """Outcome lines for every entry lacking ``experiment_id``, computed without writing.

    Returns ``(lines, refused)``.
    """
    from tcip_mcp.model_registry import read_registry_index

    outcomes: list[str] = []
    refused = False
    for entry in read_registry_index(root):
        if "experiment_id" in entry:
            continue
        conformed, line = _conform_entry(entry, plan=True)
        if conformed is None:
            refused = True
        outcomes.append(line)
    return outcomes, refused


def _apply(root: Path) -> tuple[list[str], list[str]]:
    """Write the conformed entries back inside the index's own transaction.

    Returns ``(outcomes, refused)``. The caller normally runs :func:`_plan` first and only
    reaches this once nothing is refused, but ``_apply`` does not trust that: a refused entry
    is still written back unchanged, but its name is returned in ``refused`` rather than left
    indistinguishable from a conformed one.
    """
    key = registry_index_key(root)
    outcomes: list[str] = []
    refused: list[str] = []
    with ts.transaction(key) as txn:
        index = txn.read(key, default=[])
        new_index = []
        for entry in index:
            if "experiment_id" in entry:
                new_index.append(entry)
                continue
            conformed, line = _conform_entry(entry, plan=False)
            outcomes.append(line)
            if conformed is None:
                refused.append(entry.get("name"))
                new_index.append(entry)
            else:
                new_index.append(conformed)
        txn.write(key, new_index)
    return outcomes, refused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        try:
            lines, root_refused = _plan(root)
        except ts.StoreError as exc:
            print(f"{root}: refused, {exc}")
            refused_any = True
            continue

        if not lines:
            print(f"{root}: nothing to conform")
            continue
        if root_refused:
            refused_any = True
            for line in lines:
                print(f"{root}: {line}")
            continue
        if args.plan:
            for line in lines:
                print(f"{root}: {line}")
            continue

        try:
            applied, apply_refused = _apply(root)
        except ts.StoreError as exc:
            print(f"{root}: refused, {exc}")
            refused_any = True
            continue
        for line in applied:
            print(f"{root}: {line}")
        if apply_refused:
            # _plan already found nothing refused above; a non-empty apply_refused here means
            # the index changed between the two reads, so treat it exactly like a refusal found up front.
            refused_any = True
            print(f"{root}: refused entries left unconformed: {apply_refused}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())

"""Conform a project's registry entries to carry ``experiment_id``, the field naming which run's
completion bound an entry (``None`` for an entry no run bound). A deliberate one-off operator
script, per this repo's no-backward-compatibility boundary (CLAUDE.md): the producer-binding
family's readers (``load_registered_checkpoint``'s producer resolution, an entry's own eviction
rail) now require the field, so an entry that predates it needs conforming before either can tell
a genuinely-unbound entry from one whose binding was simply never recorded. Never runs as part of
any runtime path.

For each project root named, every entry in ``.tcip/models/registry.json`` that lacks
``experiment_id`` is conformed, one entry at a time, never refused:

  * an entry with no ``experiment:<id>`` tag gets ``experiment_id=null``: nothing on it claims a
    producer at all.
  * an entry carrying an ``experiment:<id>`` tag is resolved against that run's own record: when
    the run exists and its lineage's recorded digest equals this entry's ``sha256``, the tag names
    a real binding, so ``experiment_id`` is set to the run and the tag is dropped (the binding now
    lives where the eviction rail and the digest family read it, not in caller-asserted metadata).
    Otherwise (no such run, or its recorded digest differs) ``experiment_id`` is set to ``null``,
    the tag is dropped, and the reason is printed: re-registering the run's own recorded bytes
    through ``register_model_from_experiment`` afterwards binds a ``null`` entry the same as a
    fresh one, since the eviction rail admits any run over an unbound (``None``) entry.

    python scripts/conform_registry_experiment_id.py <project_root> [<project_root> ...]
    python scripts/conform_registry_experiment_id.py --plan <project_root>

Writes go through ``tcip_store.transaction`` on the registry index key, the same locked
read-modify-write ``_register_entry`` uses, never a re-registration (which would overwrite the
entry's metrics and reset ``registered_at``, and can fail on a missing checkpoint). A root already
conformed (every entry already carries ``experiment_id``) is reported unchanged. An entry that
predates ``metrics_source`` also predates ``experiment_id``: conform it here first, then
re-register it through ``register_model`` to add ``metrics_source`` (``best_model`` and
``scripts/doctor.py`` name this order for that entry).

Reads and writes through the registry's own entries-mapping document boundary
(``tcip_mcp.model_registry``): a bare top-level array index refuses by name, naming
``scripts/conform_model_registry_paths.py`` as the remedy to run first, since this script's own
conform assumes the wrap has already happened.

Exit codes: 0 once every root named was conformed, whatever bindings its entries ended up with, or
had nothing to conform; 2 if any root's registry index will not read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

from tcip_mcp.model_registry import (  # noqa: E402
    RegistryVersionRefused,
    _read_registry_document,
    _write_registry_document,
    registry_index_key,
)


def _experiment_tag(tags) -> str | None:
    """The ``experiment:<id>`` an entry's tags name, or ``None`` when it carries no such tag."""
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith("experiment:"):
            return tag.split(":", 1)[1]
    return None


def _drop_experiment_tags(tags) -> list:
    return [t for t in (tags or []) if not (isinstance(t, str) and t.startswith("experiment:"))]


def _conform_entry(entry: dict, *, root: Path, plan: bool) -> tuple[dict, str]:
    """The entry conformed with ``experiment_id`` resolved, plus one outcome line.

    A tagged entry binds to the run it names only when that run's own lineage digest agrees
    with this entry's ``sha256``; any other case (no tag, no such run, a differing digest) sets
    ``experiment_id=null`` and drops the tag, since an unverified claim is not a recorded
    binding.
    """
    from tcip_mcp.experiments import experiment_exists, lineage_key, read_member

    name = entry.get("name")
    verb = "would set" if plan else "set"
    tag_experiment = _experiment_tag(entry.get("tags"))
    if tag_experiment is None:
        return {**entry, "experiment_id": None}, f"{name}: {verb} experiment_id=null (no experiment: tag)"

    remaining_tags = _drop_experiment_tags(entry.get("tags"))
    if not experiment_exists(tag_experiment, root=root):
        conformed = {**entry, "experiment_id": None, "tags": remaining_tags}
        return conformed, (f"{name}: {verb} experiment_id=null, dropped experiment:{tag_experiment} "
                            "(no such experiment record to verify it against)")

    lineage = read_member(lineage_key(tag_experiment, root=root), {})
    recorded = lineage.get("model_weights_sha256") if isinstance(lineage, dict) else None
    if recorded != entry.get("sha256"):
        conformed = {**entry, "experiment_id": None, "tags": remaining_tags}
        return conformed, (f"{name}: {verb} experiment_id=null, dropped experiment:{tag_experiment} "
                            f"(its recorded digest {recorded!r} differs from this entry's "
                            f"{entry.get('sha256')!r})")

    conformed = {**entry, "experiment_id": tag_experiment, "tags": remaining_tags}
    return conformed, f"{name}: {verb} experiment_id={tag_experiment!r}, dropped experiment:{tag_experiment}"


def _plan(root: Path) -> list[str]:
    """Outcome lines for every entry lacking ``experiment_id``, computed without writing."""
    from tcip_mcp.model_registry import read_registry_index

    lines: list[str] = []
    for entry in read_registry_index(root):
        if "experiment_id" in entry:
            continue
        _, line = _conform_entry(entry, root=root, plan=True)
        lines.append(line)
    return lines


def _apply(root: Path) -> list[str]:
    """Write every conformed entry back inside the index's own transaction, returning the
    outcome lines.

    Reads and writes through the entries-mapping document pair: a bare top-level array index
    refuses by name, naming ``scripts/conform_model_registry_paths.py`` as the remedy to run
    first, since this script would otherwise iterate the mapping's own top-level keys.
    """
    key = registry_index_key(root)
    outcomes: list[str] = []
    with ts.transaction(key) as txn:
        document = _read_registry_document(txn.read(key, default=None))
        new_index = []
        for entry in document["entries"]:
            if "experiment_id" in entry:
                new_index.append(entry)
                continue
            conformed, line = _conform_entry(entry, root=root, plan=False)
            outcomes.append(line)
            new_index.append(conformed)
        txn.write(key, _write_registry_document(new_index))
    return outcomes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    any_unreadable = False
    for root in args.roots:
        root = root.resolve()
        try:
            lines = _plan(root)
        except (ts.StoreError, RegistryVersionRefused) as exc:
            print(f"{root}: refused, {exc}")
            any_unreadable = True
            continue

        if not lines:
            print(f"{root}: nothing to conform")
            continue
        if args.plan:
            for line in lines:
                print(f"{root}: {line}")
            continue

        try:
            applied = _apply(root)
        except (ts.StoreError, RegistryVersionRefused) as exc:
            print(f"{root}: refused, {exc}")
            any_unreadable = True
            continue
        for line in applied:
            print(f"{root}: {line}")

    return 2 if any_unreadable else 0


if __name__ == "__main__":
    sys.exit(main())

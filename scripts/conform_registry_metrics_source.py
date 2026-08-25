"""Conform a project's registry entries to carry ``metrics_source``, the field naming which path
produced an entry's metrics: ``"trainer"`` (the platform's own ``default_train``, which measured
them), ``"training_source"`` (a bespoke loop's own saved state, unverified), ``"caller"`` (an
explicit-mode ``register_model`` argument, unverified), or ``null`` (the entry carries no
metrics). A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md): ``ModelRegistry.register_model`` now requires the field, so an entry that predates it
needs conforming before a reader (``best_model``, ``select_best_model``) can tell a verified entry
from an unverified one rather than treating it as neither. Never runs as part of any runtime path.

For each project root named, every entry in ``.tcip/models/registry.json`` that lacks
``metrics_source`` is conformed:

  * empty ``metrics`` gets ``metrics_source=null``, nothing to attribute.
  * non-empty ``metrics`` with no ``experiment:<id>`` tag gets ``metrics_source="caller"``: the
    explicit-mode ``register_model`` tool is the only production writer that leaves the tag off
    (``register_model_from_experiment`` always sets it).
  * non-empty ``metrics`` with an ``experiment:<id>`` tag is refused unless the operator states
    its source with ``--source NAME=VALUE`` (``NAME`` is the entry's own name; ``VALUE`` is
    ``trainer``, ``training_source``, or ``caller``). The tag is caller-forgeable (the tool passes
    ``tags`` through unvalidated), and which checkpoint branch produced these numbers can no
    longer be told apart from the deleted ``metrics.jsonl`` fallback, so this script does not
    guess. For a refused entry, the plan prints what the operator can check: whether the
    experiment still exists, whether its lineage ``model_weights`` equals this entry's
    ``checkpoint_path``, and whether its config carries ``training_source``.

    python scripts/conform_registry_metrics_source.py <project_root> [<project_root> ...]
    python scripts/conform_registry_metrics_source.py --plan <project_root>
    python scripts/conform_registry_metrics_source.py <project_root> --source my_model=trainer

Writes go through ``tcip_store.transaction`` on the registry index key, the same locked
read-modify-write ``ModelRegistry.register_model`` uses, never a re-registration (which would
overwrite the entry's metrics and reset ``registered_at``, and can fail on a missing checkpoint).
A root already conformed (every entry already carries ``metrics_source``) is reported unchanged.

Exit codes: 0 if every root named was conformed or had nothing to conform; 2 if any root carries
an entry refused for lack of a stated ``--source``.
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

_VALID_SOURCES = ("trainer", "training_source", "caller")


def _experiment_tag(tags) -> str | None:
    """The ``experiment:<id>`` an entry's tags name, or ``None`` when it carries no such tag."""
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith("experiment:"):
            return tag.split(":", 1)[1]
    return None


def _diagnostics(root: Path, experiment_id: str, checkpoint_path: str) -> str:
    """What the operator can check for an entry refused for lack of a stated ``--source``."""
    from tcip_mcp.experiments import config_key, lineage_key

    exists = ts.exists(config_key(experiment_id, root=root))
    lineage = ts.read(lineage_key(experiment_id, root=root), default={}) if exists else {}
    config = ts.read(config_key(experiment_id, root=root), default={}) if exists else {}
    matches = isinstance(lineage, dict) and lineage.get("model_weights") == checkpoint_path
    carries_training_source = isinstance(config, dict) and bool(config.get("training_source"))
    return (f"experiment {experiment_id!r} exists={exists}, "
            f"lineage.model_weights matches this checkpoint={matches}, "
            f"config carries training_source={carries_training_source}")


def _conform_entry(
    entry: dict, sources: dict[str, str], *, plan: bool
) -> tuple[dict | None, str]:
    """The entry as conformed (``None`` when refused), plus one outcome line."""
    name = entry.get("name")
    metrics = entry.get("metrics") or {}
    verb = "would set" if plan else "set"
    if not metrics:
        return {**entry, "metrics_source": None}, f"{name}: {verb} metrics_source=null (empty metrics)"
    tag_experiment = _experiment_tag(entry.get("tags"))
    if tag_experiment is None:
        return ({**entry, "metrics_source": "caller"},
                f"{name}: {verb} metrics_source='caller' (no experiment: tag)")
    stated = sources.get(name)
    if stated is not None:
        return {**entry, "metrics_source": stated}, f"{name}: {verb} metrics_source={stated!r} (stated)"
    return None, f"{name}: refused, carries experiment:{tag_experiment} with no stated --source"


def _plan(root: Path, sources: dict[str, str]) -> tuple[list[str], bool]:
    """Outcome lines for every entry lacking ``metrics_source``, computed without writing.

    Returns ``(lines, refused)``; a refused entry's line carries the diagnostics an operator
    checks before stating its source.
    """
    from tcip_mcp.model_registry import read_registry_index

    outcomes: list[str] = []
    refused = False
    for entry in read_registry_index(root):
        if "metrics_source" in entry:
            continue
        conformed, line = _conform_entry(entry, sources, plan=True)
        if conformed is None:
            refused = True
            tag_experiment = _experiment_tag(entry.get("tags"))
            line += "; " + _diagnostics(root, tag_experiment, entry.get("checkpoint_path", ""))
        outcomes.append(line)
    return outcomes, refused


def _apply(root: Path, sources: dict[str, str]) -> list[str]:
    """Write the conformed entries back inside the index's own transaction.

    The caller runs :func:`_plan` first and only reaches this once nothing is refused.
    """
    key = registry_index_key(root)
    outcomes: list[str] = []
    with ts.transaction(key) as txn:
        index = txn.read(key, default=[])
        new_index = []
        for entry in index:
            if "metrics_source" in entry:
                new_index.append(entry)
                continue
            conformed, line = _conform_entry(entry, sources, plan=False)
            outcomes.append(line)
            new_index.append(conformed if conformed is not None else entry)
        txn.write(key, new_index)
    return outcomes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--source", action="append", default=[], metavar="NAME=VALUE",
                    help="state the metrics_source for one experiment-tagged entry, by name")
    ap.add_argument("--plan", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args()

    bind_default()

    sources: dict[str, str] = {}
    for item in args.source:
        if "=" not in item:
            print(f"--source {item!r} is not NAME=VALUE")
            return 2
        name, value = item.split("=", 1)
        if value not in _VALID_SOURCES:
            print(f"--source {item!r}: {value!r} is not one of {_VALID_SOURCES}")
            return 2
        sources[name] = value

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        try:
            lines, root_refused = _plan(root, sources)
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
            applied = _apply(root, sources)
        except ts.StoreError as exc:
            print(f"{root}: refused, {exc}")
            refused_any = True
            continue
        for line in applied:
            print(f"{root}: {line}")

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())

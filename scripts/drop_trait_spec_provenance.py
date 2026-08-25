"""Conform a project's trait-spec records to drop the retired free-text ``provenance`` field,
and remove the stale copies an earlier YAML-to-record conform step left behind.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md): ``TraitSpec`` no longer has a ``provenance`` field, so a record still carrying one
needs conforming before the loader (which refuses any unknown field) will read it again. It
never runs as part of any runtime path.

For each project root named, this:
  * strips ``provenance`` from every trait-spec record under the live, shared-state
    ``trait_specs`` store (the one ``traits.trait_spec_key`` resolves and the platform reads),
    re-validating and writing back through the real store API, never raw SQL;
  * removes the stray, self-rooted ``trait_specs/.tcip/`` database the pre-re-root store layout
    left behind, which nothing addresses any more;
  * removes a stale ``trait_specs/*.yml`` file left beside the record an earlier YAML-to-record
    conform step produced, and only where that trait's record actually loads through the
    platform's own loader: ``SPEC_SUFFIX`` is ``.json`` and the record lives in the store, so the
    loader never reads these. A YAML whose trait has no record that loads is reported and left
    alone, since it may be the only copy of that trait's definition.

    python scripts/drop_trait_spec_provenance.py <project_root> [<project_root> ...]
    python scripts/drop_trait_spec_provenance.py --plan <project_root>

Each root is validated in full before anything is written or removed: every live record is
re-checked through the loader's own field/vocab validation, and a stale YAML is only ever removed
once the same trait's record actually loads through the platform's loader, never because the store
merely holds a key of that name. A root where any record fails that validation is left entirely
alone (nothing conformed, nothing removed) and reported; every other root named on the command line
still runs.

Exit codes: 0 conformed (or nothing to conform) for every root named; 2 if any root was refused
(a record failed to validate, or the store itself raised).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_mcp import traits  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402
from tcip_store.file_backend import lock_file_for  # noqa: E402
from tcip_store.sqlite_backend import database_path  # noqa: E402


def drop_provenance_from_records(root: Path, *, plan: bool) -> list[str]:
    """Validate, and unless ``plan`` also strip ``provenance`` from, every trait-spec record this
    project's live store holds.

    Every record, whether or not it still carries ``provenance``, is re-validated through
    ``traits._spec_from_config``, the same field/vocab check the loader itself calls: a record
    invalid for a reason unrelated to ``provenance`` is refused here too, rather than passing
    through unchecked because there was nothing to strip. One outcome line per trait found. A
    trait with no ``provenance`` field that already validates is reported unchanged rather than
    rewritten, so a rerun against an already-conformed project is a no-op.
    """
    specs_dir = traits.trait_specs_dir(root)
    state_root = traits._trait_specs_state_root(specs_dir)
    vocab = traits._crops_vocab()
    outcomes = []
    for key in ts.keys(traits.TRAIT_SPECS_STORE, str(state_root)):
        trait = key.parts[0]
        stored = ts.read_versioned(key, default=None)
        data = stored.value
        if not isinstance(data, dict):
            outcomes.append(f"{trait}: refused, record is not a mapping")
            continue
        has_provenance = "provenance" in data
        stripped = {k: v for k, v in data.items() if k != "provenance"} if has_provenance else data
        spec, reason = traits._spec_from_config(stripped, vocab)
        if spec is None:
            outcomes.append(f"{trait}: refused, {reason}")
            continue
        if not has_provenance:
            outcomes.append(f"{trait}: no provenance field, unchanged")
            continue
        if plan:
            outcomes.append(f"{trait}: would drop provenance")
            continue
        written = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(spec).items()
        }
        ts.replace(key, written, expect=stored.version)
        outcomes.append(f"{trait}: dropped provenance")
    return outcomes


def remove_stray_database(root: Path, *, plan: bool) -> str | None:
    """Remove the stray, self-rooted ``trait_specs/.tcip/`` database, or ``None`` when there is
    none to remove."""
    old_specs_root = root / ".tcip" / "state" / "trait_specs"
    db = database_path(str(old_specs_root))
    if not db.is_file():
        return None
    if plan:
        return f"would remove stray database at {db}"
    removed = [db]
    sidecars = [db.with_name(db.name + suffix) for suffix in ("-wal", "-shm")]
    sidecars.append(lock_file_for(db))
    for sidecar in sidecars:
        if sidecar.is_file():
            sidecar.unlink()
            removed.append(sidecar)
    db.unlink()
    try:
        db.parent.rmdir()
    except OSError:
        pass
    return "removed stray database: " + ", ".join(str(p) for p in removed)


def remove_stale_yaml(root: Path, *, plan: bool) -> list[str]:
    """Remove a ``trait_specs/*.yml`` file only where the same trait's record actually loads.

    What makes such a file safe to delete is that a record for the same trait loads through
    ``traits.load_trait_specs_with_errors``, the platform's own loader, never a second
    implementation of what loadable means, and never the store merely holding a key of that name:
    a key whose record is refused, malformed, or absent leaves this file the only copy of that
    trait's measurement definition, so it is reported and left in place rather than removed.
    That check runs against the real, current record, so it only applies at the moment of actual
    removal (``plan=False``): a caller only reaches this function in ``--plan`` mode once the same
    root's records have already validated cleanly (``drop_provenance_from_records``'s own refusal
    gate), and nothing has been conformed yet to load, so the preview reports on the store holding
    a key for the trait instead.
    """
    specs_dir = root / ".tcip" / "state" / "trait_specs"
    if not specs_dir.is_dir():
        return []
    if plan:
        state_root = traits._trait_specs_state_root(traits.trait_specs_dir(root))
        loadable = {key.parts[0] for key in ts.keys(traits.TRAIT_SPECS_STORE, str(state_root))}
    else:
        specs, _errors = traits.load_trait_specs_with_errors(project_root=root)
        loadable = {spec.name for spec in specs}
    outcomes = []
    for yaml_path in sorted(specs_dir.glob("*.yml")):
        if yaml_path.stem not in loadable:
            outcomes.append(
                f"kept {yaml_path}: no {yaml_path.stem!r} record loads through the platform's "
                "trait-spec loader, so this file may be the only copy of that trait's "
                "definition. Nothing in the platform reads it: the loader reads trait-spec "
                "records from the store, and traits.SPEC_SUFFIX is '.json'. Register this trait "
                "through the author_trait_spec MCP tool instead of conforming the file."
            )
            continue
        if plan:
            outcomes.append(f"would remove stale spec file {yaml_path}")
            continue
        yaml_path.unlink()
        outcomes.append(f"removed stale spec file {yaml_path}")
    return outcomes


def _removal_lines(root: Path, *, plan: bool) -> list[str]:
    lines = []
    db_outcome = remove_stray_database(root, plan=plan)
    if db_outcome:
        lines.append(db_outcome)
    lines += remove_stale_yaml(root, plan=plan)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would change, write/remove nothing")
    args = ap.parse_args()

    bind_default()

    refused = False
    for root in args.roots:
        root = root.resolve()
        try:
            validation = drop_provenance_from_records(root, plan=True)
        except ts.StoreError as exc:
            print(f"{root}: refused, {exc}")
            refused = True
            continue

        root_refused = any(": refused" in line for line in validation)
        if root_refused:
            refused = True

        try:
            if root_refused:
                lines = validation + [
                    "refused: at least one trait-spec record failed to validate, so nothing is "
                    "written or removed for this root"
                ]
            elif args.plan:
                lines = validation + _removal_lines(root, plan=True)
            else:
                lines = drop_provenance_from_records(root, plan=False) + _removal_lines(root, plan=False)
        except ts.StoreError as exc:
            print(f"{root}: refused, {exc}")
            refused = True
            continue

        if not lines:
            lines = ["nothing to conform"]
        for line in lines:
            print(f"{root}: {line}")

    return 2 if refused else 0


if __name__ == "__main__":
    sys.exit(main())

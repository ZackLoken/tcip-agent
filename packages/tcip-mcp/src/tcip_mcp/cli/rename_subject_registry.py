"""Conform a project's pre-rename state to the subject registry's current shape: rename a
dataset root's retired ``classes.json`` to ``subjects.json``, and stamp a ``trait_specs`` record
still carrying ``positive_class_name`` to ``positive_value`` with ``schema_version: 2``.

A logged operator command: bind, walk, one outcome line per unit, a closing count per root, exit
2 on any refusal or, under ``--plan``, on any unit that would change. Each named root is one of
two things:

- A project root (its own ``.tcip`` directory): every root
  :func:`tcip_mcp.store_catalogue.project_roots` answers (the project root itself, each
  registered dataset root, each run output directory, bound split-manifest directory, recorded
  curated artifact and lineage bucket), plus, under each ``SPLITS``-layout root, each of its
  ``SPLIT_NAMES`` subdirectories (the split materializer's own registry copy lands one level
  below the manifest directory, ``out_dir / split_name``, while the manifest directory itself is
  what ``project_roots`` reports). The registry unit runs at each; the trait spec unit runs once,
  over the project's own trait spec records.
- A dataset root or a derived tree carrying a registry under either name (``classes.json``,
  ``subjects.json``, or both), named directly: the registry unit alone, in place. This reaches a
  tree ``project_roots`` cannot (a curated tree materialized without an ``experiment_id``, a split
  manifest directory no run ever bound, a sweep launched to a non-default ``output_dir``).

A directory that is neither is refused by name.

The registry unit, over one root:

1. ``classes.json`` present, ``subjects.json`` absent: the bytes are read from the path and
   decoded through :func:`~tcip_mcp.subject_registry.registry_from_dict` first; a document that
   does not decode as a registry is reported and left, never renamed (a stray file of that name is
   not a registry). A registry is written through ``put_blob(subject_registry_key(root), bytes,
   expect=Version.ABSENT)``, the retired file is then removed, and one audit line names the root.
2. Both present: the two documents are compared byte for byte. Identical means a rename this
   command began and did not finish (the window between the put and the removal), and the stale
   copy is removed with the same line; different means two registries, refused by name with both
   paths, nothing moved, since which one the breeder means is not this command's to guess.
3. ``subjects.json`` alone, or neither: left and reported, nothing to do.

The trait spec unit, over every record the project's own ``trait_specs`` store enumerates: a
record that will not decode, or whose ``schema_version`` is above this reader's ceiling, is
reported and left. A record carrying both ``positive_class_name`` and ``positive_value`` is
refused by name: which one is current is not this command's to guess. A record
:func:`~tcip_mcp.traits.trait_spec_unconformed` answers no reason for and that carries no
``positive_class_name`` is already conformed, left and reported as such. Every other record is
rewritten through ``ts.replace(key, document, expect=version)`` (the store is ``concurrency="cas"``
and has no terminal refusal): ``positive_class_name`` renamed to ``positive_value`` when present,
every other key as it was, ``schema_version`` set to the current ceiling, and one audit line
under the project root names the trait and the key renamed.

Runs under whichever backend the process binds; a rewrite under the database backend leaves any
earlier loose export of that root's project stale, refreshed by ``tcip export-store``.

``--plan`` previews every outcome without writing anything; a unit that would change under a real
run counts toward the exit code the same way a refusal does.

    tcip rename-subject-registry <root> [<root> ...]
    tcip rename-subject-registry --plan <root>

Exit codes: 0 when every visited unit in every named root already carries the current shape or was
just conformed; 2 if any root is refused, any record will not decode or refuses to be resolved, or
(under ``--plan``) any unit would change.

The command ships with the family and is deleted once every root that needed it is conformed. An
archived project (``tcip archive-project`` / ``tcip import-project``) is conformed before it is
archived, never after: the archive's own accounting recognizes a retired document at the tree
root and refuses to omit it, so a bundle carrying one is refused on export, not silently short by
one file on import.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tcip_store as ts

from tcip_mcp.audit import AuditEntryNotWritten, record_event_or_raise
from tcip_mcp.dataset_layout import RETIRED_SUBJECTS_FILENAME, SUBJECTS_FILENAME, subject_registry_key
from tcip_mcp.subject_registry import registry_from_dict
from tcip_mcp.traits import TRAIT_SPEC_SCHEMA_VERSION, TRAIT_SPECS_STORE, trait_spec_unconformed

TOOL_NAME = "rename_subject_registry"


def _project_registry_roots(root: Path) -> list[Path]:
    """Every root the registry unit runs at, for a project root named directly: the project's own
    roots, plus each ``SPLIT_NAMES`` subdirectory under a ``SPLITS``-layout root (the split
    materializer's own registry copy lands one level below the manifest directory)."""
    from tcip_store.layout_claims import SPLITS

    from tcip_mcp.pipelines.data.splits import SPLIT_NAMES
    from tcip_mcp.store_catalogue import project_roots

    found: list[Path] = []
    for path_str, layout in project_roots(root):
        path = Path(path_str)
        found.append(path)
        if layout == SPLITS:
            found.extend(path / split_name for split_name in SPLIT_NAMES)
    return found


def _root_kind(root: Path) -> str:
    """``"project"`` (its own ``.tcip`` directory), ``"dataset"`` (carries a registry under
    either name), or ``"unknown"``."""
    if (root / ".tcip").is_dir():
        return "project"
    if (root / RETIRED_SUBJECTS_FILENAME).is_file() or (root / SUBJECTS_FILENAME).is_file():
        return "dataset"
    return "unknown"


def _conform_registry(root: Path, *, plan: bool) -> tuple[str, bool, bool]:
    """Conform one root's registry document. Returns ``(outcome, exit_worthy, changed)``;
    ``exit_worthy`` is true for a refusal and for a unit that would change under ``--plan``."""
    retired = root / RETIRED_SUBJECTS_FILENAME
    current_path = root / SUBJECTS_FILENAME
    retired_present = retired.is_file()
    current_present = current_path.is_file()

    if not retired_present:
        if current_present:
            return ("carries subjects.json alone; left as it is", False, False)
        return ("carries no registry under either name; left as it is", False, False)

    retired_bytes = retired.read_bytes()
    try:
        registry_from_dict(ts.RECORD_JSON.decode(retired_bytes))
    except ValueError as exc:
        return (f"{retired} does not decode as a registry ({exc}); left as it is, a stray file", True, False)

    if current_present:
        current_bytes = ts.read_blob_versioned(subject_registry_key(root)).value
        if current_bytes != retired_bytes:
            return (
                f"both {retired} and {current_path} carry a registry and disagree; refused, "
                "nothing moved (state which one is current and remove the other by hand)",
                True, False,
            )
        if plan:
            return ("subjects.json and the retired classes.json agree; would remove the stale copy", True, False)
        retired.unlink()
        try:
            record_event_or_raise(
                TOOL_NAME, {"root": str(root), "action": "removed_stale_duplicate"}, scope=root,
            )
        except AuditEntryNotWritten as exc:
            raise AuditEntryNotWritten(
                f"{TOOL_NAME}: {root} had its stale duplicate classes.json removed, but the "
                "audit line for it could not be written", exc.__cause__ or exc,
            ) from exc
        return ("removed the stale duplicate classes.json (subjects.json already matched)", False, True)

    if plan:
        return ("would rename classes.json to subjects.json", True, False)
    ts.put_blob(subject_registry_key(root), retired_bytes, expect=ts.Version.ABSENT)
    retired.unlink()
    try:
        record_event_or_raise(TOOL_NAME, {"root": str(root), "action": "renamed"}, scope=root)
    except AuditEntryNotWritten as exc:
        raise AuditEntryNotWritten(
            f"{TOOL_NAME}: {root} had classes.json renamed to subjects.json, but the audit line "
            "for it could not be written", exc.__cause__ or exc,
        ) from exc
    return ("renamed classes.json to subjects.json", False, True)


def _conform_trait_spec_record(
    key: "ts.Key", document: dict, *, project_root: Path, plan: bool,
) -> tuple[str, bool, bool]:
    trait = key.parts[0]
    if "positive_class_name" in document and "positive_value" in document:
        return (
            f"trait spec {trait!r} carries both positive_class_name and positive_value; "
            "refused, left as it is", True, False,
        )
    reason = trait_spec_unconformed(document)
    needs_rename = "positive_class_name" in document
    if reason is None and not needs_rename:
        return (f"trait spec {trait!r} already conformed; left as it is", False, False)
    if plan:
        return (f"trait spec {trait!r} would be conformed to positive_value/schema_version 2", True, False)
    new_document = dict(document)
    if needs_rename:
        new_document["positive_value"] = new_document.pop("positive_class_name")
    new_document["schema_version"] = TRAIT_SPEC_SCHEMA_VERSION
    ts.replace(key, new_document, expect=ts.read_versioned(key).version)
    try:
        record_event_or_raise(
            TOOL_NAME,
            {"project_root": str(project_root), "trait": trait, "key_renamed": needs_rename},
            scope=project_root,
        )
    except AuditEntryNotWritten as exc:
        raise AuditEntryNotWritten(
            f"{TOOL_NAME}: {project_root} had trait spec {trait!r} conformed, but the audit "
            "line for it could not be written", exc.__cause__ or exc,
        ) from exc
    return (f"trait spec {trait!r} conformed (positive_value, schema_version 2)", False, True)


def _conform_trait_specs(project_root: Path, *, plan: bool) -> list[tuple[str, bool, bool]]:
    from tcip_mcp import traits

    directory = traits.trait_specs_dir(str(project_root))
    state_root = traits._trait_specs_state_root(directory)
    outcomes: list[tuple[str, bool, bool]] = []
    for key in ts.keys(TRAIT_SPECS_STORE, str(state_root)):
        try:
            versioned = ts.read_versioned(key)
        except ts.DecodeError as exc:
            outcomes.append((f"trait spec {key.parts[0]!r} will not decode, left as it is: {exc}", True, False))
            continue
        except ts.SchemaVersionRefused as exc:
            outcomes.append((f"trait spec {key.parts[0]!r} is above this reader's ceiling, left as it is: {exc}",
                             True, False))
            continue
        document = versioned.value
        if not isinstance(document, dict):
            outcomes.append((f"trait spec {key.parts[0]!r} is not a mapping, left as it is", True, False))
            continue
        outcomes.append(
            _conform_trait_spec_record(key, document, project_root=project_root, plan=plan))
    return outcomes


def process_root(root: Path, *, plan: bool) -> tuple[list[str], bool]:
    kind = _root_kind(root)
    if kind == "unknown":
        return ([f"{root}: refused, neither a project root (no .tcip directory) nor a root "
                 "carrying a registry under either name; no summary applies"], True)

    outcomes: list[str] = []
    refused = False
    changed_count = 0
    unchanged_count = 0

    registry_roots = _project_registry_roots(root) if kind == "project" else [root]
    for reg_root in registry_roots:
        outcome, exit_worthy, changed = _conform_registry(reg_root, plan=plan)
        outcomes.append(f"{reg_root}: {outcome}")
        if changed:
            changed_count += 1
        else:
            unchanged_count += 1
        if exit_worthy:
            refused = True

    if kind == "project":
        for outcome, exit_worthy, changed in _conform_trait_specs(root, plan=plan):
            outcomes.append(f"{root} {outcome}")
            if changed:
                changed_count += 1
            else:
                unchanged_count += 1
            if exit_worthy:
                refused = True

    outcomes.append(f"{root}: {changed_count} unit(s) changed, {unchanged_count} left as they were")
    return outcomes, refused


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], prog=prog)
    ap.add_argument("roots", nargs="*", type=Path)
    ap.add_argument("--plan", action="store_true", help="preview only; nothing is written")
    args = ap.parse_args(argv)

    from tcip_store.binding import bind_default

    bind_default()

    refused_any = False
    for root in args.roots:
        root = root.resolve()
        outcomes, refused = process_root(root, plan=args.plan)
        if refused:
            refused_any = True
        for line in outcomes:
            print(line)

    return 2 if refused_any else 0


if __name__ == "__main__":
    sys.exit(main())

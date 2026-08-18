"""Conform a project's hand-authored trait-spec YAML files to the re-kinded record store.

A deliberate one-off operator script, per this repo's no-backward-compatibility boundary
(CLAUDE.md): the trait_specs store moved from hand-authored YAML to an agent-authored,
breeder-confirmed JSON record in this session's own work, and this is the conform step for
whatever pre-existing YAML specs a project already holds. It never runs as part of any
runtime path.

For each ``<root>/.tcip/state/trait_specs/<trait>.yml`` it finds, this reads the YAML bytes
with a private codec kept here rather than in ``traits.py`` (nothing production reads YAML
trait specs after the re-kind), writes the equivalent record through the real store API, and
writes a matching, unconfirmed ``trait_spec_statements`` record so the trait has something
for the breeder to confirm in the new panel. The YAML file itself is left in place beside the
record, the same convention ``adopt_store.py`` already established for a root's other stores.

This never manufactures a confirmation. A spec's existing ``provenance`` field may already
carry a ``domain_expert_confirmed`` note an agent wrote unilaterally; per this session's own
design (statement-authoring-design.md, S6), that free-text history is not gated confirmation
and is not promoted into one here. The conformed statement's ``rationale`` field summarizes
what the existing spec's provenance already says, attributed to this conform step, and the
breeder still has to click confirm.

    python scripts/conform_trait_specs.py <project_root> [<project_root> ...]
    python scripts/conform_trait_specs.py --plan <project_root>

Exit codes: 0 conformed (or nothing to conform), 2 refused (a spec failed to parse or validate;
nothing written for that trait).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "tcip-store" / "src"))

import tcip_store as ts  # noqa: E402
from tcip_mcp import traits  # noqa: E402
from tcip_store.binding import bind_default  # noqa: E402

STATEMENT_SURFACE = "scripts/conform_trait_specs.py"
"""The producing surface stamped into a conformed statement's ``stated_by``: this ran outside
the authoring tool, so it says so rather than borrowing ``author_trait_spec``'s own name."""


class _LegacyYamlCodec:
    """What the pre-re-kind blob store wrote and read. Migration-only, lives here rather than
    in ``traits.py`` since nothing production reads a trait spec's YAML bytes any more."""

    def decode(self, data: bytes) -> Any:
        import yaml

        return yaml.safe_load(data.decode("utf-8"))


def _legacy_yaml_files(root: Path) -> list[Path]:
    specs_dir = root / ".tcip" / "state" / "trait_specs"
    if not specs_dir.is_dir():
        return []
    return sorted(specs_dir.glob("*.yml"))


def _rationale_from_provenance(data: dict) -> str:
    provenance = data.get("provenance") or ()
    if not provenance:
        return "conformed from a hand-authored spec with no recorded provenance"
    joined = " ".join(str(entry).strip() for entry in provenance)
    return f"conformed from a hand-authored spec; its own recorded provenance: {joined}"


def conform_one(root: Path, yaml_path: Path, *, plan: bool) -> tuple[str, str]:
    """Conform one trait's YAML spec. Returns (trait, outcome)."""
    trait = yaml_path.stem
    data = _LegacyYamlCodec().decode(yaml_path.read_bytes())
    if not isinstance(data, dict):
        return trait, f"refused: {yaml_path} is not a mapping"

    vocab = traits._crops_vocab()
    spec, reason = traits._spec_from_config(data, vocab)
    if spec is None:
        return trait, f"refused: {reason}"

    spec_key = traits.trait_spec_key(traits.trait_specs_dir(root), trait)
    statement_key = traits.trait_spec_statement_key(traits.trait_spec_statements_scope(root), trait)
    existing_spec = ts.read_versioned(spec_key, default=None)
    existing_statement = ts.read_versioned(statement_key, default=None)
    if existing_spec.value is not None:
        return trait, "skipped: a record already exists for this trait"

    if plan:
        return trait, "would conform"

    written = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(spec).items()
    }
    ts.replace(spec_key, written, expect=existing_spec.version)

    now = datetime.now(timezone.utc).isoformat()
    statement = {
        "trait": trait,
        "statement_fields": {
            field: traits.canonical(written[field]) for field in traits._AUTHORED_SPEC_FIELDS
        },
        "rationale": _rationale_from_provenance(data),
        "stated_by": STATEMENT_SURFACE,
        "stated_at": now,
        "relayed_note": "",
        **{field: None for field in traits.TRAIT_SPEC_CONFIRMATION_FIELDS},
    }
    ts.replace(statement_key, statement, expect=existing_statement.version)
    return trait, "conformed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--plan", action="store_true", help="show what would conform, write nothing")
    args = ap.parse_args()

    bind_default()
    refused = False
    for root in args.roots:
        root = root.resolve()
        yaml_files = _legacy_yaml_files(root)
        if not yaml_files:
            print(f"{root}: no hand-authored trait specs found")
            continue
        for yaml_path in yaml_files:
            trait, outcome = conform_one(root, yaml_path, plan=args.plan)
            print(f"{root}: {trait}: {outcome}")
            if outcome.startswith("refused"):
                refused = True

    return 2 if refused else 0


if __name__ == "__main__":
    sys.exit(main())

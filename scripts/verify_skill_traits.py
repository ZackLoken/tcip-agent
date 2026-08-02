#!/usr/bin/env python
"""Guardrail: flag every trait-like token in a crop/domain SKILL.md that is not in crops.yml.

The crop skills' failure mode was asserting trait names that don't exist in the breeder-defined
controlled vocabulary. This is the deterministic backstop: LLM reviewers approved drafts that
still carried fabricated traits; this check did not.

Two independent checks, deliberately not sharing one extraction mechanism (a single
regex-extraction path used to miss 11 of 180 real trait names, `dbh`, `sex`, `ploidy`, and
others with no underscore, invisible to both checks it fed):

- `unknown_trait_tokens` (fabrication detection) extracts backtick-quoted, snake_case-shaped
  tokens via regex and flags any not in crops.yml or the allowlist. Regex-based because finding
  an unknown/fabricated name has nothing to search for: it can only look for "something shaped
  like a trait reference." Requires an underscore (multi-segment) deliberately: widening it to
  match bare single-word tokens floods on ordinary code identifiers (`ctx`, `boxes`, ...) that
  aren't traits. **Residual gap, stated honestly, not fixed by this file**: a single-word
  fabricated trait name is lexically undetectable by this check, since it looks identical to a
  legitimate non-trait single-word identifier. Only a real trait-name membership check (below)
  or human review catches that.
- `off_crop_tokens` (mis-assignment detection) does exact literal membership search, for every
  real crops.yml trait name, any shape, does it appear backtick-quoted in this skill's text,
  rather than regex extraction. Because it searches for known names instead of extracting
  candidates, it cannot miss a single-word trait the way the regex-based check structurally can.

CLI: `python scripts/verify_skill_traits.py <skill.md> [crop_key]`, exit 0 clean, 1 unknown.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Walk up from this file to the repo root (the dir holding .github/skills/crops)."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".github" / "skills" / "crops" / "crops.yml").exists():
            return parent
    return here.parent.parent


def crops_yml_path() -> Path:
    return repo_root() / ".github" / "skills" / "crops" / "crops.yml"


def load_vocab(path: Path | None = None) -> tuple[set[str], dict[str, set[str]]]:
    path = path or crops_yml_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    allnames: set[str] = set()
    by_crop: dict[str, set[str]] = collections.defaultdict(set)
    for trait in data["traits"]:
        allnames.add(trait["name"])
        for crop in trait["crops"]:
            by_crop[crop].add(trait["name"])
    return allnames, by_crop


# Snake_case tokens a skill legitimately backticks that are not traits (tool names, dataset
# fields, config keys, module paths). A new legitimate platform token that trips the check
# gets added here; the friction is intentional, it forces a human to confirm it isn't a
# fabricated trait.
NON_TRAIT_ALLOW = {
    "plant_mapping", "plant_id", "accession_name", "compute_phenology", "build_plant_mapping",
    "run_inference", "run_matching", "tile_size", "class_id", "elongation_classified",
    "catkin_phenology", "plant_mapping.json", "load_annotations",
    "save_annotations", "in_chans", "num_channels", "num_classes",
    "det_type", "gt_type", "pred_type", "count_by_class", "per_plant_phenology",
    "crossing_date", "elongation_onset_date", "plant_milestones", "write_phenology_csv",
    "boxes_from_polygons", "phenology_tools", "results.py", "aggregation.py",
}

# Multi-segment only (deliberate, see module docstring): first segment lowercase, later
# segments allow mixed case so `fruit_juice_TA`/`fruit_juice_pH`-shaped names still match.
_BACKTICK_SNAKE = re.compile(r"`([a-z][a-z0-9]*(?:_[A-Za-z0-9]+)+)`")


def extract_backtick_snake(md_text: str) -> set[str]:
    return set(m.group(1) for m in _BACKTICK_SNAKE.finditer(md_text))


def mentioned_trait_names(md_text: str, names: set[str]) -> set[str]:
    """Every crops.yml trait name, any shape, single-word or multi-word, whatever casing
    crops.yml itself uses, that appears backtick-quoted, literally, in this text. Exact
    membership search, not regex extraction: a name search can never miss a real trait for
    lacking an underscore the way `extract_backtick_snake`'s token-shaped regex can."""
    return {n for n in names if f"`{n}`" in md_text}


def unknown_trait_tokens(skill_path: str | Path, crop_key: str | None = None) -> list[str]:
    """Backticked snake_case tokens that are neither a crops.yml trait nor an allow-listed
    platform token: the fabrication signal. Cannot catch a single-word fabrication (see module
    docstring); that residual gap is inherent to token-shaped extraction, not fixed here."""
    allnames, _ = load_vocab()
    md = Path(skill_path).read_text(encoding="utf-8")
    toks = extract_backtick_snake(md)
    return sorted(t for t in toks if t not in allnames and t not in NON_TRAIT_ALLOW)


def off_crop_tokens(skill_path: str | Path, crop_key: str) -> list[str]:
    """Real traits referenced in a per-crop skill that crops.yml does not assign to that crop
    (a possible mis-assignment). Exact-membership search over the full vocabulary
    (`mentioned_trait_names`), not regex extraction: catches single-word trait names a
    token-shaped regex would miss entirely."""
    allnames, by_crop = load_vocab()
    md = Path(skill_path).read_text(encoding="utf-8")
    mentioned = mentioned_trait_names(md, allnames)
    cset = by_crop.get(crop_key, set())
    return sorted(t for t in mentioned if t not in cset)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_skill_traits.py <skill.md> [crop_key]")
        return 2
    skill_path = sys.argv[1]
    crop_key = sys.argv[2] if len(sys.argv) > 2 else None
    unknown = unknown_trait_tokens(skill_path, crop_key)
    off_crop = off_crop_tokens(skill_path, crop_key) if crop_key else []
    print(f"== {skill_path} ==")
    if unknown:
        print(f"UNKNOWN (not a crops.yml trait, not an allow-listed platform token): {len(unknown)}")
        for u in unknown:
            print(f"  - {u}")
    else:
        print("OK: no unknown trait tokens")
    if off_crop:
        print(f"OFF-CROP (real trait, not assigned to {crop_key} in crops.yml): {len(off_crop)}")
        for o in off_crop:
            print(f"  - {o}")
    return 1 if unknown else 0


if __name__ == "__main__":
    sys.exit(main())

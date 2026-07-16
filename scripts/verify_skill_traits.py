#!/usr/bin/env python
"""Guardrail: flag every trait-like token in a crop/domain SKILL.md that is not in crops.yml.

The crop skills' failure mode (see the 2026-07-14 skills rebuild) was asserting trait names
that don't exist in the breeder-defined controlled vocabulary. This is the deterministic
backstop — LLM reviewers approved drafts that still carried fabricated traits; this check did
not. It extracts backtick-quoted snake_case tokens (how skills reference traits) and checks
membership in crops.yml, globally and (when a crop is given) against that crop's exact set.

Importable: `unknown_trait_tokens(skill_path, crop_key=None)` returns the offending tokens.
CLI: `python scripts/verify_skill_traits.py <skill.md> [crop_key]` — exit 0 clean, 1 unknown.
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
# gets added here — the friction is intentional, it forces a human to confirm it isn't a
# fabricated trait.
NON_TRAIT_ALLOW = {
    "plant_mapping", "plant_id", "accession_name", "compute_phenology", "build_plant_mapping",
    "run_inference", "run_matching", "tile_size", "class_id", "elongation_classified",
    "catkin_phenology", "plant_mapping.json", "focus_annotate", "load_annotations",
    "save_annotations", "recommend_model_spec", "in_chans", "num_channels", "num_classes",
    "det_type", "gt_type", "pred_type", "count_by_class", "per_plant_phenology",
    "crossing_date", "elongation_onset_date", "plant_milestones", "write_phenology_csv",
    "boxes_from_polygons", "phenology_tools", "results.py", "aggregation.py",
}

_BACKTICK_SNAKE = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")


def extract_backtick_snake(md_text: str) -> set[str]:
    return set(m.group(1) for m in _BACKTICK_SNAKE.finditer(md_text))


def unknown_trait_tokens(skill_path: str | Path, crop_key: str | None = None) -> list[str]:
    """Backticked snake_case tokens that are neither a crops.yml trait nor an allow-listed
    platform token — the fabrication signal."""
    allnames, _ = load_vocab()
    md = Path(skill_path).read_text(encoding="utf-8")
    toks = extract_backtick_snake(md)
    return sorted(t for t in toks if t not in allnames and t not in NON_TRAIT_ALLOW)


def off_crop_tokens(skill_path: str | Path, crop_key: str) -> list[str]:
    """Real traits referenced in a per-crop skill that crops.yml does not assign to that crop
    (a possible mis-assignment)."""
    allnames, by_crop = load_vocab()
    md = Path(skill_path).read_text(encoding="utf-8")
    toks = extract_backtick_snake(md)
    cset = by_crop.get(crop_key, set())
    return sorted(t for t in toks if t in allnames and t not in cset)


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

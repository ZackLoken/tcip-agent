#!/usr/bin/env python
"""Guardrail: flag every trait-like token in a crop/domain knowledge document that is not in
crops.yml.

`unknown_trait_tokens` flags backtick-quoted, multi-segment snake_case tokens that are neither a
crops.yml trait nor an allow-listed platform token; a single-word fabrication is not detectable
this way. `off_crop_tokens` flags every crops.yml trait name, any shape, a per-crop skill
backticks that crops.yml does not assign to that crop. Both read the vocabulary through
`tcip_mcp.traits`, which raises when crops.yml will not read.

CLI: `python tools/verify_skill_traits.py <skill.md> [crop_key]`, exit 0 clean, 1 unknown,
2 unusable arguments or an unreadable vocabulary.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import yaml

from tcip_mcp import traits as registry


def load_vocab() -> tuple[set[str], dict[str, set[str]]]:
    """Every crops.yml trait name, and the names each crop declares.

    Raises ``ValueError`` when a record carries no crop list; a crops.yml that will not read
    raises from the registry's own read.
    """
    records = registry._crops_traits()
    allnames: set[str] = set()
    by_crop: dict[str, set[str]] = collections.defaultdict(set)
    for trait in records:
        crops = trait.get("crops")
        if not isinstance(crops, list) or not crops:
            raise ValueError(
                f"trait {trait['name']!r} in {registry.crops_yml_path()} declares no crops, so "
                "its per-crop assignment cannot be checked"
            )
        allnames.add(trait["name"])
        for crop in crops:
            by_crop[crop].add(trait["name"])
    return allnames, by_crop


# Snake_case tokens a skill legitimately backticks that are not traits (tool names, dataset
# fields, config keys, module paths). A new legitimate platform token that trips the check
# gets added here; the friction is intentional, it forces a human to confirm it isn't a
# fabricated trait.
NON_TRAIT_ALLOW = {
    "plant_mapping", "plant_id", "accession_name", "deliver_phenology_milestones",
    "build_plant_mapping", "run_inference", "run_matching", "tile_size", "class_id",
    "positive_class_assessed",
    "catkin_phenology", "plant_mapping.json", "load_annotations",
    "save_annotations", "in_chans", "num_channels", "num_classes",
    "det_type", "gt_type", "pred_type", "count_by_class", "per_plant_phenology",
    "crossing_date", "positive_onset_date", "plant_milestones", "write_phenology_csv",
    "write_phenology_curve_csv", "boxes_from_polygons", "phenology_tools", "results.py",
    "aggregation.py",
}

# Multi-segment only (deliberate, see module docstring): first segment lowercase, later
# segments allow mixed case so `fruit_juice_TA`/`fruit_juice_pH`-shaped names still match.
_BACKTICK_SNAKE = re.compile(r"`([a-z][a-z0-9]*(?:_[A-Za-z0-9]+)+)`")


def extract_backtick_snake(md_text: str) -> set[str]:
    return set(m.group(1) for m in _BACKTICK_SNAKE.finditer(md_text))


def mentioned_trait_names(md_text: str, names: set[str]) -> set[str]:
    """Every name in ``names`` that appears backtick-quoted, literally, in ``md_text``."""
    return {n for n in names if f"`{n}`" in md_text}


def unknown_trait_tokens(md_text: str, allnames: set[str]) -> list[str]:
    """Backticked snake_case tokens in ``md_text`` that are neither in ``allnames`` nor an
    allow-listed platform token."""
    toks = extract_backtick_snake(md_text)
    return sorted(t for t in toks if t not in allnames and t not in NON_TRAIT_ALLOW)


def off_crop_tokens(md_text: str, allnames: set[str], crop_names: set[str]) -> list[str]:
    """Names in ``allnames`` that ``md_text`` backticks and ``crop_names`` does not hold."""
    return sorted(t for t in mentioned_trait_names(md_text, allnames) if t not in crop_names)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_skill_traits.py <skill.md> [crop_key]")
        return 2
    skill_path = sys.argv[1]
    crop_key = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        allnames, by_crop = load_vocab()
        md = Path(skill_path).read_text(encoding="utf-8")
        unknown = unknown_trait_tokens(md, allnames)
        off_crop = off_crop_tokens(md, allnames, by_crop.get(crop_key, set())) if crop_key else []
    except (OSError, ValueError, KeyError, yaml.YAMLError) as e:
        print(f"cannot check {skill_path}: {e}")
        return 2
    print(f"== {skill_path} ==")
    if unknown:
        print("UNKNOWN (not a crops.yml trait, not an allow-listed platform token): "
              f"{len(unknown)}")
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

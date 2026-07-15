"""CI guardrail: crop skills must not assert traits outside crops.yml.

The 2026-07-14 skills rebuild found the per-crop skills had systematically fabricated trait
names, and the LLM reviewers approved the fabrications — a deterministic check is what caught
them. It's pinned here as a permanent gate: any crop / crop-science SKILL.md that backticks a
snake_case token which is neither a crops.yml trait nor an allow-listed platform token fails
the build, so the drift can't recur.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / ".github" / "skills"


def _load_guardrail():
    spec = importlib.util.spec_from_file_location(
        "verify_skill_traits", REPO / "scripts" / "verify_skill_traits.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guardrail = _load_guardrail()

# skill dir (under .github/skills) -> crops.yml crop key (None = domain skill, global check)
CROP_SKILLS = {
    "crops/hazelnut": "hazelnut",
    "crops/chestnut": "chestnut",
    "crops/currant": "currant",
    "crops/elderberry": "elderberry",
    "crops/persimmon": "persimmon",
    "crops/black-locust": "black_locust",
    "crop-science": None,
}


@pytest.mark.parametrize("rel,crop_key", list(CROP_SKILLS.items()))
def test_skill_asserts_no_fabricated_traits(rel: str, crop_key: str | None) -> None:
    skill = SKILLS / rel / "SKILL.md"
    assert skill.exists(), f"missing skill: {skill}"
    unknown = guardrail.unknown_trait_tokens(skill, crop_key)
    assert not unknown, (
        f"{rel}/SKILL.md backticks trait-like tokens not in crops.yml and not allow-listed: "
        f"{unknown}. Either it's a fabricated trait (fix the skill) or a real platform token "
        "(add it to NON_TRAIT_ALLOW in scripts/verify_skill_traits.py)."
    )

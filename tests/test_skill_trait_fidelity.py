"""CI guardrail: crop skills must not assert traits outside crops.yml.

Per-crop skills can fabricate plausible-looking trait names that aren't in crops.yml, and review
(human or LLM) is not guaranteed to catch it; only a deterministic check reliably does. It's
pinned here as a permanent gate: any crop / crop-science SKILL.md that backticks a snake_case
token which is neither a crops.yml trait nor an allow-listed platform token fails the build, so
the drift can't recur.
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


@pytest.mark.parametrize(
    "rel,crop_key", [(r, c) for r, c in CROP_SKILLS.items() if c is not None]
)
def test_skill_asserts_no_off_crop_traits(rel: str, crop_key: str) -> None:
    """Every real trait a per-crop skill backticks must actually be assigned to that crop in
    crops.yml: the mis-assignment check `test_skill_asserts_no_fabricated_traits` doesn't cover
    it (that one only checks fabrication, never crop assignment)."""
    skill = SKILLS / rel / "SKILL.md"
    off_crop = guardrail.off_crop_tokens(skill, crop_key)
    assert not off_crop, (
        f"{rel}/SKILL.md backticks trait(s) crops.yml does not assign to {crop_key!r}: "
        f"{off_crop}. Either the skill or crops.yml's crop assignment is wrong."
    )


def test_backtick_regex_catches_no_underscore_and_mixed_case_names() -> None:
    """The backtick regex requires an underscore, so single-word trait names (`dbh`, `sex`,
    `ploidy`, ...) and mixed-case segments (`fruit_juice_TA`, `fruit_juice_pH`) never match,
    invisible to both checks. That's deliberate, to avoid flooding on bare code identifiers, so
    single-word names stay a fabrication-detection blind spot by design; the membership-search
    check below is what actually catches them for the off-crop case."""
    text = "See `fruit_juice_TA` and `fruit_juice_pH` and `plant_height` here."
    toks = guardrail.extract_backtick_snake(text)
    assert "fruit_juice_TA" in toks
    assert "fruit_juice_pH" in toks
    assert "plant_height" in toks
    # single-word tokens still don't match the token-shaped regex: documented blind spot
    assert guardrail.extract_backtick_snake("See `dbh` and `sex` here.") == set()


def test_mentioned_trait_names_finds_single_word_traits() -> None:
    """`mentioned_trait_names` is membership search, not regex extraction: it finds `dbh` even
    though `extract_backtick_snake` (the fabrication-check regex) structurally cannot."""
    allnames, _ = guardrail.load_vocab()
    found = guardrail.mentioned_trait_names("Measure `dbh` on the standing tree.", allnames)
    assert "dbh" in found


def test_off_crop_tokens_catches_single_word_mis_assignment(tmp_path) -> None:
    """Fails before the fix: a single-word real trait referenced on the wrong crop's skill was
    invisible to the old regex-extraction-based `off_crop_tokens`, so a real mis-assignment went
    undetected. `dbh` is a real crops.yml trait not assigned to currant."""
    allnames, by_crop = guardrail.load_vocab()
    assert "dbh" in allnames
    assert "dbh" not in by_crop.get("currant", set())
    fake_skill = tmp_path / "SKILL.md"
    fake_skill.write_text("This currant skill mentions `dbh` by mistake.", encoding="utf-8")
    off_crop = guardrail.off_crop_tokens(fake_skill, "currant")
    assert "dbh" in off_crop

"""CI guardrail: crop skills must not assert traits outside crops.yml.

Per-crop skills can fabricate plausible-looking trait names that aren't in crops.yml, and review
(human or LLM) is not guaranteed to catch it; only a deterministic check reliably does. It's
pinned here as a permanent gate: any crop / crop-science SKILL.md that backticks a snake_case
token which is neither a crops.yml trait nor an allow-listed platform token fails the build, so
the drift can't recur.
"""

from __future__ import annotations

import importlib.util

import pytest

from tests import REPO_ROOT


def _load_guardrail():
    spec = importlib.util.spec_from_file_location(
        "verify_skill_traits", REPO_ROOT / "tools" / "verify_skill_traits.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guardrail = _load_guardrail()


def _document_text(name: str) -> str:
    from tcip_mcp.knowledge import document_path

    skill = document_path(name)
    assert skill.exists(), f"missing document: {skill}"
    return skill.read_text(encoding="utf-8")

# knowledge document name -> crops.yml crop key (None = domain document, global check)
CROP_SKILLS = {
    "hazelnut": "hazelnut",
    "chestnut": "chestnut",
    "currant": "currant",
    "elderberry": "elderberry",
    "persimmon": "persimmon",
    "black-locust": "black_locust",
    "crop-science": None,
}


@pytest.mark.parametrize("name", list(CROP_SKILLS))
def test_skill_asserts_no_fabricated_traits(name: str) -> None:
    allnames, _ = guardrail.load_vocab()
    unknown = guardrail.unknown_trait_tokens(_document_text(name), allnames)
    assert not unknown, (
        f"{name}: backticks trait-like tokens not in crops.yml and not allow-listed: "
        f"{unknown}. Either it's a fabricated trait (fix the document) or a real platform token "
        "(add it to NON_TRAIT_ALLOW in tools/verify_skill_traits.py)."
    )


@pytest.mark.parametrize(
    "name,crop_key", [(n, c) for n, c in CROP_SKILLS.items() if c is not None]
)
def test_skill_asserts_no_off_crop_traits(name: str, crop_key: str) -> None:
    """Every real trait a per-crop document backticks is assigned to that crop in crops.yml."""
    allnames, by_crop = guardrail.load_vocab()
    off_crop = guardrail.off_crop_tokens(_document_text(name), allnames, by_crop[crop_key])
    assert not off_crop, (
        f"{name}: backticks trait(s) crops.yml does not assign to {crop_key!r}: "
        f"{off_crop}. Either the document or crops.yml's crop assignment is wrong."
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


def test_the_guardrail_checks_the_vocabulary_the_registry_itself_loads(monkeypatch) -> None:
    """One read of the controlled vocabulary serves both the runtime registry and this check.
    Two reads can disagree, and the disagreement shows up as a skill passing a check the platform
    would fail on the same name, so the vocabulary the guardrail sees is the registry's."""
    from tcip_mcp import traits

    monkeypatch.setattr(
        traits, "_crops_traits",
        lambda: [{"name": "measure_one", "crops": ["crop_one"]}])

    allnames, by_crop = guardrail.load_vocab()
    assert allnames == {"measure_one"}
    assert by_crop["crop_one"] == {"measure_one"}
    assert guardrail.unknown_trait_tokens(
        "Records `plant_height` for the block.", allnames) == ["plant_height"]


def test_a_readable_vocabulary_is_checked() -> None:
    """The real controlled vocabulary loads, carries its crop assignments, and a clean skill comes
    back clean."""
    allnames, by_crop = guardrail.load_vocab()
    assert "dbh" in allnames
    assert by_crop["hazelnut"]
    assert guardrail.unknown_trait_tokens("Measure `dbh` on the standing tree.", allnames) == []


def test_off_crop_tokens_catches_single_word_mis_assignment() -> None:
    """A single-word real trait referenced on the wrong crop's skill is caught by
    `off_crop_tokens`. `dbh` is a real crops.yml trait not assigned to currant."""
    allnames, by_crop = guardrail.load_vocab()
    assert "dbh" in allnames
    assert "dbh" not in by_crop.get("currant", set())
    off_crop = guardrail.off_crop_tokens(
        "This currant skill mentions `dbh` by mistake.", allnames, by_crop.get("currant", set()))
    assert "dbh" in off_crop

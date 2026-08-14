"""Exactness of the two name lookups a trait spec passes through.

A trait name resolves against the registry by exact match, and a spec's ``delivers`` entries are
members of the crops.yml controlled vocabulary, not near-misses of one. Both are anti-fabrication
boundaries: a loosened lookup hands back another trait's measurement semantics, or registers a
phenotype crops.yml never defines, with nothing in the result saying so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import traits
from tcip_mcp.traits import (
    TraitUnknownError,
    get_trait,
    load_trait_specs,
    load_trait_specs_with_errors,
    registered_traits,
)


def _write_spec(directory: Path, name: str, spec: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    import yaml

    (directory / f"{name}.yml").write_text(yaml.safe_dump({"name": name, **spec}), encoding="utf-8")


def _two_traits_with_different_semantics(directory: Path) -> None:
    """Two registered traits whose measurement semantics differ in every field a caller reads,
    so serving one where the other was asked for is observable rather than harmless."""
    _write_spec(directory, "catkin", {
        "delivers": ["catkin_95per_date"],
        "positive_class_name": "elongated",
        "count_bias_tolerance_frac": 0.02,
        "milestone_fractions": [0.05, 0.5, 0.95],
        "phenology_prefix": "catkin",
    })
    _write_spec(directory, "leaf", {
        "delivers": ["leaf_length"],
        "positive_class_name": "expanded",
        "count_bias_tolerance_frac": 0.25,
        "milestone_fractions": [0.5],
        "phenology_prefix": "leaf",
    })


@pytest.mark.parametrize("near_miss", ["Catkin", "CATKIN", "Leaf", "LEAF"])
def test_get_trait_refuses_a_name_differing_only_by_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, near_miss: str,
):
    """A name that is not exactly a registered one is unknown, however close it looks. Resolving
    it to a same-spelled-different-cased spec would silently swap in another trait's measurement
    semantics, which is worse than the honest refusal."""
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _two_traits_with_different_semantics(tmp_path)
    assert registered_traits() == ["catkin", "leaf"]

    with pytest.raises(TraitUnknownError, match=f"Unknown trait '{near_miss}'"):
        get_trait(near_miss)


def test_get_trait_resolves_the_exact_registered_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The refusal above must not cost the legitimate call: each exact name still returns its own
    spec, and the two specs stay distinct."""
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _two_traits_with_different_semantics(tmp_path)

    assert get_trait("catkin").count_bias_tolerance_frac == 0.02
    assert get_trait("catkin").positive_class_name == "elongated"
    assert get_trait("leaf").count_bias_tolerance_frac == 0.25
    assert get_trait("leaf").positive_class_name == "expanded"


def test_unknown_trait_refusal_lists_what_is_registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The refusal names the registry it searched, so a caller can tell a misspelling from an
    unauthored trait without guessing."""
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _two_traits_with_different_semantics(tmp_path)

    with pytest.raises(TraitUnknownError, match=r"Registered traits: \['catkin', 'leaf'\]"):
        get_trait("Catkin")


@pytest.mark.parametrize("truncated", ["catkin_05per", "leaf_len", "fruit_diamet"])
def test_delivers_must_be_a_vocabulary_member_not_a_prefix_of_one(tmp_path: Path, truncated: str):
    """A ``delivers`` entry that is only a prefix of a real crops.yml name is off-vocabulary. It
    reads as controlled vocabulary to a human skimming the file while naming a phenotype crops.yml
    never defines, so the spec is skipped and the entry is named in the reason."""
    vocab = traits._crops_vocab()
    assert vocab, "crops.yml vocab should be loadable in the repo checkout"
    assert truncated not in vocab
    assert any(v.startswith(truncated) for v in vocab), truncated

    _write_spec(tmp_path, "truncated", {"delivers": [truncated]})
    specs, errors = load_trait_specs_with_errors(specs_dir=tmp_path)
    assert specs == []
    assert [e["file"] for e in errors] == ["truncated.yml"]
    assert truncated in errors[0]["reason"]


def test_a_truncated_delivers_entry_does_not_smuggle_a_whole_spec_in(tmp_path: Path):
    """One prefix entry alongside real vocabulary still fails the whole spec: the anchor is every
    delivered phenotype, not merely one of them."""
    _write_spec(tmp_path, "mixed", {"delivers": ["leaf_length", "leaf_wid"]})
    _write_spec(tmp_path, "honest", {"delivers": ["leaf_length", "leaf_width"]})

    specs, errors = load_trait_specs_with_errors(specs_dir=tmp_path)
    assert [s.name for s in specs] == ["honest"]
    assert [e["file"] for e in errors] == ["mixed.yml"]
    assert "leaf_wid" in errors[0]["reason"]


def test_write_trait_spec_fields_refuses_to_truncate_an_existing_delivers(tmp_path: Path):
    """The update path re-runs the same vocabulary check, so a field update cannot walk a
    registered spec off the controlled vocabulary a prefix at a time."""
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"]})

    with pytest.raises(ValueError, match="invalid spec"):
        traits.write_trait_spec_fields("leaf", {"delivers": ["leaf_len"]}, [], specs_dir=tmp_path)

    assert load_trait_specs(specs_dir=tmp_path)[0].delivers == ("leaf_length",)


def test_write_trait_spec_fields_still_accepts_a_real_vocabulary_addition(tmp_path: Path):
    """The refusal above admits the legitimate edit: adding a second phenotype that is genuinely in
    crops.yml is written and reloads."""
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"]})

    updated = traits.write_trait_spec_fields(
        "leaf", {"delivers": ["leaf_length", "leaf_width"]},
        ["delivers: vocabulary_derived"], specs_dir=tmp_path,
    )
    assert updated.delivers == ("leaf_length", "leaf_width")
    assert load_trait_specs(specs_dir=tmp_path)[0].delivers == ("leaf_length", "leaf_width")

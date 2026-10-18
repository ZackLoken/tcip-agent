"""Exactness of the two name lookups a trait entry passes through.

A trait name resolves against the project's records by exact match, and a proposed entry's
``delivers`` entries are members of the crops.yml controlled vocabulary, not near-misses of one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import traits
from tcip_mcp.traits import TraitUnknownError, read_trait, trait_names
from tests._trait_fixtures import entry, latest, propose, with_fields


def _two_traits_with_different_semantics(root: Path) -> None:
    """Propose ``bud`` and ``leaf``, two traits whose measurement fields differ.

    They differ in positive state, count bias tolerance, milestone fractions and phenology
    prefix.
    """
    propose(
        root,
        entry(
            "bud",
            ("leaf_out_50per_date",),
            positive_state={"attribute": "opening", "value": "open"},
            count_bias_tolerance_frac=0.02,
            milestone_fractions=(0.05, 0.5, 0.95),
            phenology_prefix="bud",
        ),
    )
    propose(
        root,
        entry(
            "leaf",
            ("leaf_length",),
            positive_state={"attribute": "stage", "value": "expanded"},
            count_bias_tolerance_frac=0.25,
            milestone_fractions=(0.5,),
            phenology_prefix="leaf",
        ),
    )


@pytest.mark.parametrize("near_miss", ["Bud", "BUD", "Leaf", "LEAF"])
def test_a_trait_name_differing_only_by_case_is_unknown(
    tmp_path: Path, near_miss: str
):
    """A trait name differing from a recorded one only by case is unknown."""
    _two_traits_with_different_semantics(tmp_path)
    assert trait_names(tmp_path) == ["bud", "leaf"]

    with pytest.raises(
        TraitUnknownError, match=f"Unknown trait '{near_miss}'"
    ):
        read_trait(near_miss, tmp_path)


def test_the_exact_name_resolves_its_own_entry(tmp_path: Path):
    """Each exact trait name returns its own entry, and the two stay distinct."""
    _two_traits_with_different_semantics(tmp_path)

    assert latest("bud", tmp_path).count_bias_tolerance_frac == 0.02
    assert latest("bud", tmp_path).positive_state == traits.PositiveState(
        attribute="opening", value="open"
    )
    assert latest("leaf", tmp_path).count_bias_tolerance_frac == 0.25
    assert latest("leaf", tmp_path).positive_state == traits.PositiveState(
        attribute="stage", value="expanded"
    )


def test_unknown_trait_refusal_lists_what_the_project_holds(tmp_path: Path):
    """The unknown-trait refusal lists the traits the project holds."""
    _two_traits_with_different_semantics(tmp_path)

    with pytest.raises(
        TraitUnknownError, match=r"Traits in this project: \['bud', 'leaf'\]"
    ):
        read_trait("Bud", tmp_path)


@pytest.mark.parametrize(
    "truncated", ["catkin_05per", "leaf_len", "fruit_diamet"]
)
def test_delivers_must_be_a_vocabulary_member_not_a_prefix_of_one(
    tmp_path: Path, truncated: str
):
    """A ``delivers`` entry that is only a prefix of a real crops.yml name is refused.

    Such an entry names no crops.yml phenotype, and the proposal is refused naming it.
    """
    vocab = {t["name"] for t in traits._crops_traits()}
    assert truncated not in vocab
    assert any(v.startswith(truncated) for v in vocab), truncated

    with pytest.raises(ValueError, match=truncated):
        propose(tmp_path, entry("truncated", (truncated,)))


def test_a_truncated_delivers_entry_does_not_smuggle_a_whole_entry_in(
    tmp_path: Path,
):
    """One prefix entry alongside real vocabulary still fails the whole proposal.

    Each delivered phenotype is checked, not merely one of them; the honest pair is admitted.
    """
    with pytest.raises(ValueError, match="leaf_wid"):
        propose(tmp_path, entry("mixed", ("leaf_length", "leaf_wid")))

    assert propose(
        tmp_path, entry("honest", ("leaf_length", "leaf_width"))
    ).entry.delivers == ("leaf_length", "leaf_width")


def test_a_revision_cannot_walk_delivers_off_the_vocabulary_and_a_real_addition_is_admitted(
    tmp_path: Path,
):
    """A revision cannot walk ``delivers`` off the vocabulary, and a real addition is admitted.

    A later proposal delivering the truncated ``leaf_len`` is refused naming it, while one
    adding ``leaf_width`` beside ``leaf_length`` lands as the latest revision.
    """
    base = entry("leaf", ("leaf_length",))
    propose(tmp_path, base)

    with pytest.raises(ValueError, match="leaf_len"):
        propose(tmp_path, with_fields(base, delivers=("leaf_len",)))

    propose(
        tmp_path, with_fields(base, delivers=("leaf_length", "leaf_width"))
    )
    assert latest("leaf", tmp_path).delivers == ("leaf_length", "leaf_width")

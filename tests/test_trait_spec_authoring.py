"""``author_trait_spec``/``confirm_trait_spec``, the trait-spec authoring surface.

Trait-spec authoring writes to two stores in order: the effective ``TraitSpec`` into
``trait_specs`` first, then the unconfirmed authoring statement into ``trait_spec_statements``
second, mirroring the writer separation ``operationalization.py`` already keeps between a
statement and its confirmation. These cases pin the collision rule (a spec and its statement both
already on record refuses; a spec with no statement, the partial-failure recovery state, does
not), the crops.yml cross-check reused rather than reimplemented, the confirmation-field refusal,
the stale-``record_seen`` refusal, and that the confirmation writer is unreachable from any MCP
tool. The final case is the proof this mechanism exists for: a trait authored and confirmed through
this surface alone delivers through the existing operationalization and phenology mechanisms
exactly as a hand-authored spec always has.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp import traits

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "tools"
TRAITS_MODULE = REPO_ROOT / "packages" / "tcip-mcp" / "src" / "tcip_mcp" / "traits.py"


def _author(root: Path, trait: str = "bud_opening_e2e", **overrides: object) -> dict:
    fields: dict[str, object] = dict(
        delivers=("leaf_out_05per_date",),
        positive_class_name="open",
        milestone_fractions=(0.05, 0.5),
        milestone_on="positive_fraction",
        rationale="the breeder described the state directly, in their own field-scoring terms",
    )
    fields.update(overrides)
    return traits.author_trait_spec(str(root), trait, **fields)


def test_authoring_a_trait_that_already_exists_refuses(tmp_path: Path) -> None:
    _author(tmp_path)

    with pytest.raises(ValueError, match="already on record"):
        _author(tmp_path)


def test_a_spec_with_no_statement_is_not_a_collision_and_the_call_proceeds_as_a_restatement(
    tmp_path: Path,
) -> None:
    """The recorded partial-failure recovery state: a spec on record with no statement behind it
    (here, seeded directly to stand in for a second write that failed partway) is not refused."""
    directory = traits.trait_specs_dir(str(tmp_path))
    key = traits.trait_spec_key(directory, "bud_opening_e2e")
    ts.replace(key, {"name": "bud_opening_e2e", "delivers": ["leaf_out_05per_date"]}, expect=ts.Version.ABSENT)

    statement = _author(tmp_path)

    assert statement["trait"] == "bud_opening_e2e"
    assert traits.get_trait_for("bud_opening_e2e", str(tmp_path)).delivers == ("leaf_out_05per_date",)


def test_authoring_a_delivers_entry_outside_the_vocabulary_refuses(tmp_path: Path) -> None:
    """Reuses the same crops.yml cross-check every config-authored spec already goes through;
    this only pins that author_trait_spec is wired to it, not the vocabulary check itself."""
    with pytest.raises(ValueError, match="crops.yml"):
        _author(tmp_path, delivers=("not_a_real_phenotype",))


def test_the_author_tool_refuses_a_payload_naming_a_confirmation_field(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirmation field"):
        _author(tmp_path, confirmed_by="sneaky")


def test_confirm_trait_spec_with_a_stale_record_seen_raises_the_moved_record_shape(
    tmp_path: Path,
) -> None:
    statement = _author(tmp_path)

    with pytest.raises(traits.TraitSpecStatementMoved) as excinfo:
        traits.confirm_trait_spec(
            str(tmp_path), "bud_opening_e2e", user="breeder", record_seen="not the real hash",
            identity_from_request=True,
        )

    assert excinfo.value.record["trait"] == "bud_opening_e2e"
    assert excinfo.value.record_seen == traits.trait_spec_statement_seen_hash(statement)


def test_restating_a_trait_whose_confirmed_statement_was_lost_clears_the_prior_confirmation(
    tmp_path: Path,
) -> None:
    """Amendment A1's own recovery state extended one step further: a statement that once
    existed and was confirmed, then went missing (simulated here the way a lost second write
    would leave it), is not a collision either, and the trait comes back unconfirmed."""
    statement = _author(tmp_path)
    seen = traits.trait_spec_statement_seen_hash(statement)
    confirmed = traits.confirm_trait_spec(
        str(tmp_path), "bud_opening_e2e", user="breeder", record_seen=seen, identity_from_request=True,
    )
    assert confirmed["confirmed_by"] == "user:breeder"

    scope = traits.trait_spec_statements_scope(tmp_path)
    key = traits.trait_spec_statement_key(scope, "bud_opening_e2e")
    ts.delete(key, expect=ts.read_versioned(key).version)

    restated = _author(tmp_path)

    assert restated["confirmed_by"] is None
    assert restated["confirmed_at"] is None
    assert restated["record_seen"] is None


def test_no_mcp_tool_reaches_the_confirmation_writer() -> None:
    """The agent has an authoring tool and no confirmation tool, checked against the live
    registry rather than a hand-kept list, so a tool added later is covered automatically."""
    writer = "confirm_trait_spec"
    listing = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "list_tools.py")],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout
    registered = {line.strip() for line in listing.splitlines() if line.startswith("  ")}

    assert "author_trait_spec" in registered
    assert not [name for name in registered if writer in name]
    assert writer in TRAITS_MODULE.read_text(encoding="utf-8"), (
        "the writer's name moved, so this test would be searching for nothing"
    )
    assert not [
        module.name
        for module in TOOLS_DIR.glob("*.py")
        if writer in module.read_text(encoding="utf-8")
    ]


def test_author_and_update_persist_through_the_same_shared_write(tmp_path: Path) -> None:
    """``author_trait_spec`` (create) and ``write_trait_spec_fields`` (update) both delegate
    their validate-encode-write step to one shared entry now; this drives both in sequence and
    reads the persisted record back through ``get_trait_for`` (``load_trait_specs`` underneath)
    rather than comparing in-memory encodes, proving both entry points persist through the
    shared write and that what a fresh load sees is what each one wrote. Coverage for the
    unification, not a regression guard: each writer already persisted correctly on its own
    before the two shared this entry.

    Not a test of either caller's own discipline: the carried-forward restriction is
    ``test_a_restatement_over_an_existing_spec_carries_its_localization_and_sliver_fields_forward``
    below; no test yet drives ``write_trait_spec_fields``'s own compare-and-set retry loop to
    actually retry against a losing read.
    """
    _author(tmp_path, trait="leaf", delivers=("leaf_length",), holdout_match_quality_floor=0.4)

    updated = traits.write_trait_spec_fields(
        "leaf", {"holdout_match_quality_floor": 0.6}, project_root=tmp_path,
    )
    assert updated.holdout_match_quality_floor == 0.6

    reloaded = traits.get_trait_for("leaf", str(tmp_path))
    assert reloaded.holdout_match_quality_floor == 0.6
    assert reloaded.delivers == ("leaf_length",)


def test_updating_a_trait_spec_with_a_caller_supplied_schema_version_refuses(tmp_path: Path) -> None:
    """``schema_version`` is not a ``TraitSpec`` field and no caller sets it directly; a config
    editor slipping it into ``fields`` must not be able to stamp a version the store seam never
    validated."""
    _author(tmp_path, trait="leaf", delivers=("leaf_length",))

    with pytest.raises(ValueError, match="schema_version"):
        traits.write_trait_spec_fields("leaf", {"schema_version": 2}, project_root=tmp_path)


def test_a_stamped_trait_specs_schema_version_survives_a_field_edit(tmp_path: Path) -> None:
    """A record already carrying a ``schema_version`` stamp (seeded directly here, the way an
    adopted or hand-conformed record would carry one) must keep that stamp through an ordinary
    field-edit rewrite: ``_encode_spec`` has no such dataclass field, so nothing but a deliberate
    re-attach in the write path keeps it from falling out on every edit."""
    _author(tmp_path, trait="leaf", delivers=("leaf_length",), holdout_match_quality_floor=0.4)
    directory = traits.trait_specs_dir(str(tmp_path))
    key = traits.trait_spec_key(directory, "leaf")
    stored = ts.read_versioned(key)
    ts.replace(key, {**stored.value, "schema_version": 1}, expect=stored.version)

    traits.write_trait_spec_fields(
        "leaf", {"holdout_match_quality_floor": 0.6}, project_root=tmp_path,
    )

    rewritten = ts.read_versioned(key).value
    assert rewritten["schema_version"] == 1
    assert rewritten["holdout_match_quality_floor"] == 0.6


def test_a_restatement_over_an_existing_spec_carries_its_localization_and_sliver_fields_forward(
    tmp_path: Path,
) -> None:
    """``author_trait_spec`` never accepts ``localization``/``localization_tolerance``/
    ``localization_tolerance_frac``/``sliver_policy``/``sliver_frac``: a restatement over an
    existing spec keeps whatever ``write_trait_spec_fields`` last set for them, changing only
    the fields ``author_trait_spec`` itself authors. Coverage: the restriction is already the
    baseline's own behavior, pinned here since no existing test asserted on these fields'
    values after a restatement.
    """
    _author(tmp_path)
    traits.write_trait_spec_fields(
        "bud_opening_e2e",
        {
            "localization": traits.CENTER_MATCH,
            "localization_tolerance": "fixed",
            "localization_tolerance_frac": 0.25,
            "sliver_policy": "fixed_fraction",
            "sliver_frac": 0.1,
        },
        project_root=tmp_path,
    )
    scope = traits.trait_spec_statements_scope(tmp_path)
    key = traits.trait_spec_statement_key(scope, "bud_opening_e2e")
    ts.delete(key, expect=ts.read_versioned(key).version)

    _author(tmp_path, positive_class_name="closed")

    reloaded = traits.get_trait_for("bud_opening_e2e", str(tmp_path))
    assert reloaded.positive_class_name == "closed"  # the restatement's own authored field moved
    assert reloaded.localization == traits.CENTER_MATCH
    assert reloaded.localization_tolerance == "fixed"
    assert reloaded.localization_tolerance_frac == 0.25
    assert reloaded.sliver_policy == "fixed_fraction"
    assert reloaded.sliver_frac == 0.1


def test_a_trait_authored_and_confirmed_through_this_surface_delivers_end_to_end(
    tmp_path: Path,
) -> None:
    """Author, confirm, operationalize and deliver a real trait with no hand-authored file
    anywhere in the path: the proof this authoring path produces a spec
    state_trait_operationalization/check_operationalization/deliver_phenology_milestones can all use exactly
    as if it had been hand-authored."""
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._operationalization_fixtures import seed_confirmed_crossing
    from tests._trait_fixtures import BUD_OPENING
    from tests.test_phenology_tools import _delivery_setup

    statement = traits.author_trait_spec(
        str(tmp_path), "bud_opening",
        delivers=BUD_OPENING.delivers,
        positive_class_name="open",
        milestone_fractions=(0.05, 0.50, 0.95),
        milestone_on="positive_fraction",
        phenology_prefix="bud",
        majority_milestone="95per",
        majority_provisional=True,
        majority_label="opening",
        notes=BUD_OPENING.notes,
        rationale="the breeder called opening the frilled/salt-and-peppery texture change, "
                  "never a bbox-ratio proxy, and wants the 5/50/95% crossings of that call",
    )
    seen = traits.trait_spec_statement_seen_hash(statement)
    confirmed = traits.confirm_trait_spec(
        str(tmp_path), "bud_opening", user="breeder", record_seen=seen, identity_from_request=True,
    )
    assert confirmed["confirmed_by"] == "user:breeder"

    seed_confirmed_crossing(tmp_path, "bud_opening", measured_subject="bud")

    mapping_name, d1, d2 = _delivery_setup(tmp_path, experiment_id=None, checkpoint_sha256=None)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening", mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv), classifier_pred_dirs=[str(d1)],
    )

    assert "error" not in res, res
    assert res["positive_class_assessed"] is True
    assert out_csv.exists()

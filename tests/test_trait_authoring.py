"""W8 — config-driven multi-trait TraitSpec authoring + derived class-id + read-semantics.

Pins the softened scope-provisional behavior: registering trait #2 is a breeder-authored config
edit (cross-checked against the crops.yml controlled vocab, never agent-invented), the elongated
class id is a mapping fact derived from classes.json by name (never a pinned default), and the
provisional 95%-mapping marker travels with the delivery. There are no built-in traits (round 10,
2026-07-29) — catkin is authored the same way as any other trait; this module's ``pytestmark``
requests ``conftest.py``'s ``seed_catkin_trait_spec``, which writes a real config file matching
``tests/_trait_fixtures.CATKIN`` into this test's pinned project root, so ``get_trait("catkin")``
keeps resolving by default the way it did when a builtin was unconditionally present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp import traits
from tcip_mcp.traits import (
    TraitUnknownError,
    get_trait,
    load_trait_specs,
    registered_traits,
)
from tests._trait_fixtures import CATKIN

pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


# ── R1: config-driven authoring, crops.yml-cross-checked ──────────────────

def _write_spec(directory: Path, name: str, spec: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    import yaml

    (directory / f"{name}.yml").write_text(yaml.safe_dump({"name": name, **spec}), encoding="utf-8")


def test_load_trait_specs_reads_vocab_checked_config(tmp_path: Path):
    # leaf_length IS a crops.yml trait, so a spec delivering it is registerable via config alone.
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"], "localization": "iou_match",
                                   "count_objective": "detection_f1"})
    specs = load_trait_specs(specs_dir=tmp_path)
    assert [s.name for s in specs] == ["leaf"]
    assert specs[0].delivers == ("leaf_length",)
    assert specs[0].localization == "iou_match"


def test_config_spec_off_vocab_delivers_is_rejected(tmp_path: Path):
    # A fabricated phenotype (not in crops.yml) must not register — the anti-fabrication anchor.
    _write_spec(tmp_path, "unicorn", {"delivers": ["unicorn_horn_length"]})
    assert load_trait_specs(specs_dir=tmp_path) == []


def test_config_spec_empty_delivers_is_rejected(tmp_path: Path):
    _write_spec(tmp_path, "vague", {"count_objective": "presence"})  # no delivers
    assert load_trait_specs(specs_dir=tmp_path) == []


def test_config_spec_unknown_field_is_rejected(tmp_path: Path):
    _write_spec(tmp_path, "typo", {"delivers": ["leaf_length"], "not_a_field": 3})
    assert load_trait_specs(specs_dir=tmp_path) == []


# ── K2 Fix E: count_objective is validated against the registry, not a hardcoded whitelist ─

def test_config_spec_arbitrary_count_objective_is_accepted_at_registration(tmp_path: Path):
    # K18 B4: count_objective is NOT a closed enum — a trait may name any objective an agent has
    # implemented and registered a picker for. Registration accepts it regardless; whether it can
    # actually be CALIBRATED depends on whether operating_point.COUNT_OBJECTIVE_PICKERS has a
    # matching entry (see test_resolve_operating_point_refuses_unregistered_count_objective).
    _write_spec(tmp_path, "custom", {"delivers": ["leaf_length"], "count_objective": "a_brand_new_objective"})
    specs = load_trait_specs(specs_dir=tmp_path)
    assert specs[0].count_objective == "a_brand_new_objective"


def test_config_spec_unset_count_objective_is_empty_not_defaulted(tmp_path: Path):
    # No silent default to catkin's historical value — an omitted count_objective stays honestly
    # empty, the same "not yet decided" shape as count_error_tolerance's None.
    _write_spec(tmp_path, "undecided", {"delivers": ["leaf_length"]})
    specs = load_trait_specs(specs_dir=tmp_path)
    assert specs[0].count_objective == ""


def test_resolve_operating_point_refuses_unrecorded_count_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tcip_mcp.pipelines.operating_point as OP

    _write_spec(tmp_path, "undecided", {"delivers": ["leaf_length"]})
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    with pytest.raises(ValueError, match="no recorded count_objective"):
        OP.resolve_operating_point("undecided", dataset_hash="h1", calibration_records=[])


def test_resolve_operating_point_refuses_unregistered_count_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tcip_mcp.pipelines.operating_point as OP

    _write_spec(tmp_path, "custom", {"delivers": ["leaf_length"], "count_objective": "a_brand_new_objective"})
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    with pytest.raises(ValueError, match="no registered picker"):
        OP.resolve_operating_point("custom", dataset_hash="h1", calibration_records=[])


def test_config_spec_every_registered_objective_is_accepted(tmp_path: Path):
    # The validator's accepted values are DERIVED from the same registry the picker uses (Fix E) —
    # one source of truth, not a second hardcoded list that could drift out of sync.
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    for i, objective in enumerate(COUNT_OBJECTIVE_PICKERS):
        _write_spec(tmp_path, f"t{i}", {"delivers": ["leaf_length"], "count_objective": objective})
    specs = load_trait_specs(specs_dir=tmp_path)
    assert {s.count_objective for s in specs} == set(COUNT_OBJECTIVE_PICKERS)


def test_missing_specs_dir_yields_no_config(tmp_path: Path):
    assert load_trait_specs(specs_dir=tmp_path / "nope") == []


# ── K18 B2.5: the audited write path for an already-registered trait spec ──────────────────

def test_write_trait_spec_fields_updates_and_persists(tmp_path: Path):
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"], "count_bias_tolerance": 1.0})
    updated = traits.write_trait_spec_fields(
        "leaf", {"count_bias_tolerance": 2.5}, ["count_bias_tolerance: domain_expert_confirmed"],
        specs_dir=tmp_path,
    )
    assert updated.count_bias_tolerance == 2.5
    assert "count_bias_tolerance: domain_expert_confirmed" in updated.provenance
    # persisted, not just returned — a fresh load sees the same value
    reloaded = load_trait_specs(specs_dir=tmp_path)
    assert reloaded[0].count_bias_tolerance == 2.5


def test_write_trait_spec_fields_appends_provenance_not_replaces(tmp_path: Path):
    _write_spec(tmp_path, "leaf", {
        "delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"],
    })
    updated = traits.write_trait_spec_fields(
        "leaf", {"count_bias_tolerance": 3.0}, ["count_bias_tolerance: zack_methodology_correction"],
        specs_dir=tmp_path,
    )
    assert updated.provenance == (
        "name: vocabulary_derived", "count_bias_tolerance: zack_methodology_correction",
    )


def test_write_trait_spec_fields_refuses_when_trait_not_already_registered(tmp_path: Path):
    with pytest.raises(ValueError, match="no existing trait spec file"):
        traits.write_trait_spec_fields("nonexistent", {"count_bias_tolerance": 1.0}, [], specs_dir=tmp_path)


def test_write_trait_spec_fields_refuses_an_invalid_merged_spec(tmp_path: Path):
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"]})
    with pytest.raises(ValueError, match="invalid spec"):
        traits.write_trait_spec_fields("leaf", {"not_a_real_field": 3}, [], specs_dir=tmp_path)
    # refusal means nothing was written — the spec is unchanged
    assert load_trait_specs(specs_dir=tmp_path)[0].delivers == ("leaf_length",)


def test_update_trait_spec_fields_tool_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tcip_mcp.tools.phenology_tools import update_trait_spec_fields

    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"], "count_bias_tolerance": 1.0})
    result = update_trait_spec_fields(
        "leaf", {"count_bias_tolerance": 4.0}, ["count_bias_tolerance: domain_expert_confirmed"],
    )
    assert result["count_bias_tolerance"] == 4.0
    assert get_trait("leaf").count_bias_tolerance == 4.0


def test_registry_reads_every_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Point the default specs dir at an absolute tmp path (resolve_state returns it unchanged), with
    # two authored files — the registry must read every one, not just the first (no builtin to fall
    # back to if it stopped short).
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _write_spec(tmp_path, "catkin", {"delivers": ["catkin_05per_date"]})
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"]})
    assert set(registered_traits()) == {"catkin", "leaf"}
    assert get_trait("leaf").delivers == ("leaf_length",)


def test_config_authored_catkin_is_the_real_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Round 10 (2026-07-29) removed the prior precedence bug where a built-in Python object silently
    # outranked a config file authoring the same trait name. There is no built-in anymore, so an
    # authored catkin.yml is simply what get_trait("catkin") returns — proven here by a value a real
    # builtin would never have carried.
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _write_spec(tmp_path, "catkin", {"delivers": ["catkin_05per_date"], "count_bias_tolerance": 99.0})
    assert get_trait("catkin").count_bias_tolerance == 99.0


def test_unknown_trait_still_hard_fails():
    with pytest.raises(TraitUnknownError):
        get_trait("banana")


def test_catkin_config_semantics_match_reference_fixture():
    # This module's pytestmark seeds a real catkin.yml (round 10) matching tests/_trait_fixtures.CATKIN
    # — proves the config path round-trips every field faithfully, not just the ones this test
    # happens to check by name below. Config-loaded specs are rebuilt fresh per call (traits.py),
    # never module-load singletons, so this is value equality, not identity.
    t = get_trait("catkin")
    assert t == CATKIN
    assert t.positive_class_name == "elongated"
    assert t.localization_tolerance_frac == 0.5
    assert t.sliver_frac == 0.5
    assert t.majority_milestone == "95per"
    assert t.majority_provisional is True
    assert t.count_bias_tolerance == 1.0  # ABSOLUTE, breeder-set (D12)
    assert set(t.delivers) == {
        "catkin_05per_date", "catkin_50per_date", "catkin_95per_date", "catkin_elongation_date"}


def test_reference_fixture_delivers_are_all_in_crops_vocab():
    # Guardrail: the local test fixture must itself obey the controlled vocab it enforces on config —
    # else this whole suite would be exercising an off-vocab trait shape no real config could load.
    vocab = traits._crops_vocab()
    assert vocab, "crops.yml vocab should be loadable in the repo checkout"
    for name in CATKIN.delivers:
        assert name in vocab, name


# ── K4/K5: positive class id resolved from a prediction bucket's own recorded id_map ───────

def _op_sidecar(dir_path: Path, id_map: dict | None) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "operating_point.json").write_text(json.dumps({
        "validated": True,
        "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
        "id_map": id_map,
    }), encoding="utf-8")


def test_resolve_positive_class_id_by_name(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    d = tmp_path / "preds"
    _op_sidecar(d, {"dormant": 0, "elongated": 1})
    cid, msg = _resolve_positive_class_id("catkin", {"2026-02-11": str(d)})
    assert cid == 1
    assert "elongated" in msg


def test_resolve_positive_class_id_honest_fail_when_absent(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    d = tmp_path / "preds"
    _op_sidecar(d, {"dormant": 0, "catkin": 1})  # no 'elongated' class
    cid, msg = _resolve_positive_class_id("catkin", {"2026-02-11": str(d)})
    assert cid is None  # never silently defaults to 1
    assert "elongated" in msg


def test_resolve_positive_class_id_no_map_is_none(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    cid, _ = _resolve_positive_class_id("catkin", {"2026-02-11": str(tmp_path / "missing")})
    assert cid is None


# ── K4/K5 end-to-end through compute_phenology ────────────────────────────

def _pheno_fixture(tmp_path: Path, *, classified: bool):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    id_map = {"dormant": 0, "elongated": 1} if classified else {"catkin": 0}
    subject = "elongated" if classified else "catkin"
    for d in (d1, d2):
        d.mkdir(parents=True, exist_ok=True)
        json_io.write_annotations(
            d / "P1.json",
            [Annotation(subject=subject, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9)], 8, 8)
        _op_sidecar(d, id_map)
    mapping_path = tmp_path / "plant_mapping.json"
    mapping_path.write_text(json.dumps({
        "2026-02-11": [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}],
    }), encoding="utf-8")
    return mapping_path, d1, d2


def test_compute_phenology_derives_class_id_and_delivers(tmp_path: Path):
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_fixture(tmp_path, classified=True)
    out_csv = tmp_path / "out.csv"
    (d1 / "classifier_operating_point.json").write_text(json.dumps({
        "validated": True,
        "operating_point": {"classifier": {"value": "elongated", "validated_against": "held_out_annotations"}},
        "trait": "catkin",
    }), encoding="utf-8")

    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
        operating_point_validated="held_out_annotations",
    )
    # The positive class id resolves from the buckets' own recorded id_map; both dimensions are
    # validated, so this delivers.
    assert "error" not in res, res
    assert res["elongation_classified"] is True
    assert out_csv.exists()
    assert "catkin_elongation_provisional" in phenology.phenology_csv_columns(get_trait("catkin"))


def test_compute_phenology_refuses_when_class_id_unresolvable(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_fixture(tmp_path, classified=False)  # no 'elongated' anywhere
    res = compute_phenology(
        trait="catkin",
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(tmp_path / "out.csv"),
        operating_point_validated="held_out_annotations",
    )
    assert "error" in res
    assert not (tmp_path / "out.csv").exists()

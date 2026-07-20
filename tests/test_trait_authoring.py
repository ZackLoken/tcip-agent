"""W8 — config-driven multi-trait TraitSpec authoring + derived class-id + read-semantics.

Pins the softened scope-provisional behavior: registering trait #2 is a breeder-authored config
edit (cross-checked against the crops.yml controlled vocab, never agent-invented), the elongated
class id is a mapping fact derived from classes.json by name (never a pinned default), and the
provisional 95%-mapping marker travels with the delivery. Catkin's built-in semantics are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_mcp import traits
from tcip_mcp.traits import (
    CATKIN,
    TraitUnknownError,
    get_trait,
    load_trait_specs,
    registered_traits,
)


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


def test_missing_specs_dir_yields_no_config(tmp_path: Path):
    assert load_trait_specs(specs_dir=tmp_path / "nope") == []


def test_registry_unions_builtins_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Point the default specs dir at an absolute tmp path (resolve_state returns it unchanged).
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", tmp_path)
    _write_spec(tmp_path, "leaf", {"delivers": ["leaf_length"]})
    assert "catkin" in registered_traits()
    assert get_trait("leaf").delivers == ("leaf_length",)


def test_builtins_win_on_name_collision(tmp_path: Path):
    # A config file cannot redefine a built-in trait's semantics.
    _write_spec(tmp_path, "catkin", {"delivers": ["catkin_05per_date"], "count_bias_tolerance": 99.0})
    specs = load_trait_specs(specs_dir=tmp_path)
    assert specs and specs[0].count_bias_tolerance == 99.0  # the config parses...
    # ...but the live registry keeps the built-in (guaranteed by _all_traits' update order).
    merged = {s.name: s for s in specs}
    merged.update({t.name: t for t in (CATKIN,)})
    assert merged["catkin"].count_bias_tolerance == 1.0


def test_unknown_trait_still_hard_fails():
    with pytest.raises(TraitUnknownError):
        get_trait("banana")


def test_catkin_builtin_semantics_unchanged():
    t = get_trait("catkin")
    assert t is CATKIN
    assert t.positive_class_name == "elongated"
    assert t.localization_tolerance_frac == 0.5
    assert t.sliver_frac == 0.5
    assert t.majority_milestone == "95per"
    assert t.majority_provisional is True
    assert t.count_bias_tolerance == 1.0  # ABSOLUTE, breeder-set (D12)
    assert set(t.delivers) == {
        "catkin_05per_date", "catkin_50per_date", "catkin_95per_date", "catkin_elongation_date"}


def test_builtin_delivers_are_all_in_crops_vocab():
    # Guardrail: the built-in specs must themselves obey the controlled vocab they enforce on config.
    vocab = traits._crops_vocab()
    assert vocab, "crops.yml vocab should be loadable in the repo checkout"
    for name in CATKIN.delivers:
        assert name in vocab, name


# ── R2: elongated class id derived from classes.json by name ──────────────

def _classes_json(path: Path, mapping: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(cid): {"name": n} for cid, n in mapping.items()}), encoding="utf-8")


def test_resolve_positive_class_id_by_name(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    cj = tmp_path / "classes.json"
    _classes_json(cj, {0: "dormant", 1: "elongated"})
    cid, msg = _resolve_positive_class_id("catkin", str(cj))
    assert cid == 1
    assert "elongated" in msg


def test_resolve_positive_class_id_honest_fail_when_absent(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    cj = tmp_path / "classes.json"
    _classes_json(cj, {0: "dormant", 1: "catkin"})  # no 'elongated' class
    cid, msg = _resolve_positive_class_id("catkin", str(cj))
    assert cid is None  # never silently defaults to 1
    assert "elongated" in msg


def test_resolve_positive_class_id_no_map_is_none(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    cid, _ = _resolve_positive_class_id("catkin", str(tmp_path / "missing.json"))
    assert cid is None


# ── R2/R3 end-to-end through compute_phenology ────────────────────────────

def _pheno_fixture(tmp_path: Path, *, elongated_id: int):
    from tcip_annotation import json_io
    from tcip_annotation.state import PredBBox

    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    for d, cid in ((d1, 0), (d2, elongated_id)):
        d.mkdir(parents=True, exist_ok=True)
        json_io.write_detect(d / "P1.json", [PredBBox(1.0, 1.0, 3.0, 3.0, cid, confidence=0.9)], 8, 8)
        (d / "operating_point.json").write_text(json.dumps({
            "validated": True,
            "operating_point": {"conf": {"value": 0.4, "validated_vs_gt": "validated_held_out"}},
        }), encoding="utf-8")
    mapping_path = tmp_path / "plant_mapping.json"
    mapping_path.write_text(json.dumps({
        "2026-02-11": [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}],
    }), encoding="utf-8")
    return mapping_path, d1, d2


def test_compute_phenology_derives_class_id_and_stamps_provisional(tmp_path: Path):
    import csv

    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_fixture(tmp_path, elongated_id=1)
    cj = tmp_path / "classes.json"
    _classes_json(cj, {0: "dormant", 1: "elongated"})
    out_csv = tmp_path / "out.csv"

    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classes_json_path=str(cj),  # elongated_class_id=None -> derived from the map by name
        classifier_validated="validated_held_out",
        operating_point_validated="validated_held_out",
    )
    assert "error" not in res
    with out_csv.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert "catkin_elongation_provisional" in phenology.PHENOLOGY_CSV_COLUMNS
    assert row["catkin_elongation_provisional"] == "true"  # provisional reading marked in the delivery


def test_compute_phenology_refuses_when_class_id_unresolvable(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import compute_phenology

    mapping_path, d1, d2 = _pheno_fixture(tmp_path, elongated_id=1)
    cj = tmp_path / "classes.json"
    _classes_json(cj, {0: "dormant", 1: "catkin"})  # no 'elongated' -> cannot derive
    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(tmp_path / "out.csv"),
        classes_json_path=str(cj),
        classifier_validated="validated_held_out",
        operating_point_validated="validated_held_out",
    )
    assert "error" in res
    assert "elongated class id" in res["error"]
    assert not (tmp_path / "out.csv").exists()

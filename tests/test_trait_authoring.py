"""Config-driven multi-trait TraitSpec authoring + derived class-id + read-semantics.

Pins the softened scope-tentative behavior: registering trait #2 is a breeder-authored config
edit (cross-checked against the crops.yml controlled vocab, never agent-invented), the open
class id is a mapping fact derived from classes.json by name (never a pinned default), and the
crossing-unconfirmed 95%-mapping marker travels with the delivery. There are no built-in traits:
bud_opening is authored the same way as any other trait; this module's ``pytestmark`` requests
``conftest.py``'s ``seed_bud_trait_spec``, which writes a real config file matching
``tests/_trait_fixtures.BUD_OPENING`` into this test's pinned platform state root, so
``get_trait("bud_opening")`` keeps resolving by default the way it did when a builtin was
unconditionally present.
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
from tests._binding_fixtures import write_bound_sidecar
from tests._trait_fixtures import BUD_OPENING

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# ── R1: config-driven authoring, crops.yml-cross-checked ──────────────────

def _write_spec(directory: Path, name: str, spec: dict) -> None:
    import tcip_store as ts

    ts.replace(traits.trait_spec_key(directory, name), {"name": name, **spec}, expect=ts.Version.ABSENT)


def test_load_trait_specs_reads_vocab_checked_config(tmp_path: Path):
    # leaf_length IS a crops.yml trait, so a spec delivering it is registerable via config alone.
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"], "localization": "iou_match",
                                    "count_objective": "detection_f1"})
    specs = load_trait_specs(specs_dir=specs_dir)
    assert [s.name for s in specs] == ["leaf"]
    assert specs[0].delivers == ("leaf_length",)
    assert specs[0].localization == "iou_match"


def test_config_spec_off_vocab_delivers_is_rejected(tmp_path: Path):
    # A fabricated phenotype (not in crops.yml) must not register: the anti-fabrication anchor.
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "unicorn", {"delivers": ["unicorn_horn_length"]})
    assert load_trait_specs(specs_dir=specs_dir) == []


def test_config_spec_empty_delivers_is_rejected(tmp_path: Path):
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "vague", {"count_objective": "presence"})  # no delivers
    assert load_trait_specs(specs_dir=specs_dir) == []


def test_load_trait_specs_with_errors_names_the_broken_file_and_why(tmp_path: Path):
    # A silently-dropped spec is invisible to a breeder; the errors list is the trace of it.
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "unicorn", {"delivers": ["unicorn_horn_length"]})
    specs, errors = load_trait_specs_with_errors(specs_dir=specs_dir)
    assert specs == []
    assert len(errors) == 1
    assert errors[0]["file"] == "unicorn.json"
    assert "unicorn_horn_length" in errors[0]["reason"]


def test_load_trait_specs_with_errors_leaves_valid_specs_out_of_the_error_list(tmp_path: Path):
    # One broken spec alongside a real one: the good one still loads, only the bad one is named.
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"], "localization": "iou_match",
                                    "count_objective": "detection_f1"})
    _write_spec(specs_dir, "unicorn", {"delivers": ["unicorn_horn_length"]})
    specs, errors = load_trait_specs_with_errors(specs_dir=specs_dir)
    assert [s.name for s in specs] == ["leaf"]
    assert [e["file"] for e in errors] == ["unicorn.json"]


def test_load_trait_specs_with_errors_reports_malformed_json(tmp_path: Path):
    # A malformed record's bytes exist only as a loose file, which the file backend serves
    # directly; the database backend refuses to read a root holding files it did not write.
    from tcip_store.file_backend import FileBackend

    import tcip_store as ts

    ts.bind(FileBackend())
    specs_dir = tmp_path / "trait_specs"
    specs_dir.mkdir(parents=True)
    (specs_dir / "broken.json").write_text("not valid json {", encoding="utf-8")
    specs, errors = load_trait_specs_with_errors(specs_dir=specs_dir)
    assert specs == []
    assert errors[0]["file"] == "broken.json"


def test_config_spec_unknown_field_is_rejected(tmp_path: Path):
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "typo", {"delivers": ["leaf_length"], "not_a_field": 3})
    assert load_trait_specs(specs_dir=specs_dir) == []


def test_config_spec_stamped_with_schema_version_still_loads(tmp_path: Path):
    # frozen-formats.json declares trait_specs able to carry schema_version; the stamp is a
    # store concern (the seam's read path already enforces its ceiling), not an unknown field.
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"], "schema_version": 1})
    specs = load_trait_specs(specs_dir=specs_dir)
    assert [s.name for s in specs] == ["leaf"]
    assert specs[0].delivers == ("leaf_length",)


def test_config_spec_unstamped_still_loads(tmp_path: Path):
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"]})
    specs = load_trait_specs(specs_dir=specs_dir)
    assert [s.name for s in specs] == ["leaf"]


# ── count_objective is validated against the registry, not a hardcoded whitelist ─

def test_config_spec_arbitrary_count_objective_is_accepted_at_registration(tmp_path: Path):
    # count_objective is not a closed enum: a trait may name any objective an agent has
    # implemented and registered a picker for. Registration accepts it regardless; whether it can
    # actually be calibrated depends on whether operating_point.COUNT_OBJECTIVE_PICKERS has a
    # matching entry (see test_resolve_operating_point_refuses_unregistered_count_objective).
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "custom", {"delivers": ["leaf_length"], "count_objective": "a_brand_new_objective"})
    specs = load_trait_specs(specs_dir=specs_dir)
    assert specs[0].count_objective == "a_brand_new_objective"


def test_config_spec_unset_count_objective_is_empty_not_defaulted(tmp_path: Path):
    # No silent default to bud_opening's historical value: an omitted count_objective stays honestly
    # empty, the same "not yet decided" shape as count_error_tolerance's None.
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "undecided", {"delivers": ["leaf_length"]})
    specs = load_trait_specs(specs_dir=specs_dir)
    assert specs[0].count_objective == ""


def test_resolve_operating_point_defaults_unrecorded_count_objective_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    # Nobody, not the agent, not the breeder, can meaningfully answer "does every object need to
    # be found correctly, or is it fine if errors cancel out" before any result exists to judge it
    # against. An unset count_objective must not block calibration entirely; it defaults to
    # COUNT_UNBIASED and proceeds, with the run's provenance stamped as a platform default (not
    # trait-authored) so the distinction is never silently lost. The real confirmation point is the
    # delivered result, via the review-confirmation loop, not a blind precondition.
    import tcip_mcp.pipelines.operating_point as OP
    from tcip_mcp.traits import COUNT_UNBIASED

    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "undecided", {"delivers": ["leaf_length"]})
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)
    bundle = OP.resolve_operating_point("undecided", tiled=True, dataset_hash="h1", calibration_records=[])
    param = bundle.params["count_objective"]
    assert param._raw == COUNT_UNBIASED
    assert param.source == "default"
    assert "not breeder-confirmed" in param.derived_from


def test_resolve_operating_point_stamps_explicit_count_objective_as_trait_authored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import tcip_mcp.pipelines.operating_point as OP

    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "decided", {"delivers": ["leaf_length"], "count_objective": "detection_f1"})
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)
    bundle = OP.resolve_operating_point("decided", tiled=True, dataset_hash="h1", calibration_records=[])
    param = bundle.params["count_objective"]
    assert param._raw == "detection_f1"
    assert param.derived_from == "trait-authored"


def test_resolve_operating_point_refuses_unregistered_count_objective(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tcip_mcp.pipelines.operating_point as OP

    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "custom", {"delivers": ["leaf_length"], "count_objective": "a_brand_new_objective"})
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)
    with pytest.raises(ValueError, match="no registered picker"):
        OP.resolve_operating_point("custom", tiled=True, dataset_hash="h1", calibration_records=[])


def test_config_spec_every_registered_objective_is_accepted(tmp_path: Path):
    # The validator's accepted values are derived from the same registry the picker uses:
    # one source of truth, not a second hardcoded list that could drift out of sync.
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    specs_dir = tmp_path / "trait_specs"
    for i, objective in enumerate(COUNT_OBJECTIVE_PICKERS):
        _write_spec(specs_dir, f"t{i}", {"delivers": ["leaf_length"], "count_objective": objective})
    specs = load_trait_specs(specs_dir=specs_dir)
    assert {s.count_objective for s in specs} == set(COUNT_OBJECTIVE_PICKERS)


def test_missing_specs_dir_yields_no_config(tmp_path: Path):
    # A never-created trait_specs directory reads as no config, the same as a project with
    # nothing authored yet.
    assert load_trait_specs(specs_dir=tmp_path / "trait_specs") == []


def test_noncanonical_specs_dir_basename_is_refused_rather_than_silently_misdirected(tmp_path: Path):
    # A caller naming anything but the fixed "trait_specs" final segment is refused rather than
    # silently answered from the real trait_specs directory.
    with pytest.raises(ValueError, match="does not end in 'trait_specs'"):
        load_trait_specs(specs_dir=tmp_path / "nope")


# ── the audited write path for an already-registered trait spec ──────────────────

def test_write_trait_spec_fields_updates_and_persists(tmp_path: Path):
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"], "count_bias_tolerance_frac": 1.0})
    updated = traits.write_trait_spec_fields(
        "leaf", {"count_bias_tolerance_frac": 2.5}, specs_dir=specs_dir,
    )
    assert updated.count_bias_tolerance_frac == 2.5
    # persisted, not just returned: a fresh load sees the same value
    reloaded = load_trait_specs(specs_dir=specs_dir)
    assert reloaded[0].count_bias_tolerance_frac == 2.5


def test_write_trait_spec_fields_refuses_when_trait_not_already_registered(tmp_path: Path):
    specs_dir = tmp_path / "trait_specs"
    with pytest.raises(ValueError, match="no trait spec record"):
        traits.write_trait_spec_fields("nonexistent", {"count_bias_tolerance_frac": 1.0}, specs_dir=specs_dir)


def test_write_trait_spec_fields_refuses_an_invalid_merged_spec(tmp_path: Path):
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"]})
    with pytest.raises(ValueError, match="invalid spec"):
        traits.write_trait_spec_fields("leaf", {"not_a_real_field": 3}, specs_dir=specs_dir)
    # refusal means nothing was written: the spec is unchanged
    assert load_trait_specs(specs_dir=specs_dir)[0].delivers == ("leaf_length",)


def test_write_trait_spec_fields_refuses_a_spec_still_carrying_the_deleted_provenance_field(tmp_path: Path):
    """A spec record left over with the retired ``provenance`` field is an unknown field to the
    loader, the same refusal any other unrecognized field gets: no special-cased tolerance for it."""
    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"]})
    assert load_trait_specs(specs_dir=specs_dir) == []
    specs, errors = load_trait_specs_with_errors(specs_dir=specs_dir)
    assert specs == []
    assert "provenance" in errors[0]["reason"]


def test_revise_trait_spec_tool_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tcip_mcp.tools.trait_spec_authoring_tools import revise_trait_spec

    specs_dir = tmp_path / "trait_specs"
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"], "count_bias_tolerance_frac": 1.0})
    result = revise_trait_spec(str(tmp_path), "leaf", {"count_bias_tolerance_frac": 4.0})
    assert result["count_bias_tolerance_frac"] == 4.0
    assert get_trait("leaf").count_bias_tolerance_frac == 4.0
    assert result["superseded"] == []


def test_registry_reads_every_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Two authored files: the registry must read every one, not just the first (no builtin to
    # fall back to if it stopped short).
    specs_dir = tmp_path / "trait_specs"
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)
    _write_spec(specs_dir, "bud", {"delivers": ["leaf_out_05per_date"]})
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"]})
    assert set(registered_traits()) == {"bud", "leaf"}
    assert get_trait("leaf").delivers == ("leaf_length",)


def test_config_authored_bud_is_the_real_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # No built-in trait definition could outrank a config-authored spec record carrying the same
    # trait name, proven here by a value a real builtin would never have carried.
    specs_dir = tmp_path / "trait_specs"
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)
    _write_spec(specs_dir, "bud", {"delivers": ["leaf_out_05per_date"], "count_bias_tolerance_frac": 99.0})
    assert get_trait("bud").count_bias_tolerance_frac == 99.0


def test_unknown_trait_still_hard_fails():
    with pytest.raises(TraitUnknownError):
        get_trait("banana")


# ── one writer that cannot lose an edit ─────────

def test_a_spec_write_that_lost_the_race_is_refused_rather_than_silently_winning(tmp_path: Path):
    """A spec is read, merged into and written back from more than one process, so the store takes
    the version the writer read; a write from a stale one is refused instead of dropping whatever
    the other writer recorded."""
    import tcip_store as ts

    specs_dir = tmp_path / "trait_specs"
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"]})
    key = traits.trait_spec_key(specs_dir, "leaf")
    stale = ts.read_versioned(key).version
    traits.write_trait_spec_fields("leaf", {"count_bias_tolerance_frac": 1.0}, specs_dir=specs_dir)

    with pytest.raises(ts.VersionConflict):
        ts.replace(key, {"name": "leaf", "delivers": ["leaf_length"]}, expect=stale)

    assert load_trait_specs(specs_dir=specs_dir)[0].count_bias_tolerance_frac == 1.0


def test_get_trait_load_trait_specs_and_registered_traits_agree_on_a_record_conformed_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A project whose spec was conformed straight into the record store (the shape a one-off
    conform script produces, written here by hand rather than by running one) resolves the
    identical ``TraitSpec`` through every read path the platform has: the re-kind changes only
    the bytes on disk, never what a reader sees.
    """
    import dataclasses

    import tcip_store as ts

    specs_dir = tmp_path / "trait_specs"
    data = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(BUD_OPENING).items()}
    ts.replace(traits.trait_spec_key(specs_dir, "bud_opening"), data, expect=ts.Version.ABSENT)
    monkeypatch.setattr(traits, "_TRAIT_SPECS_RELPATH", specs_dir)

    assert get_trait("bud_opening") == BUD_OPENING
    assert load_trait_specs() == [BUD_OPENING]
    assert load_trait_specs(specs_dir=specs_dir) == [BUD_OPENING]
    assert registered_traits() == ["bud_opening"]


def test_bud_opening_config_semantics_match_reference_fixture():
    # This module's pytestmark seeds a real bud_opening record matching _trait_fixtures.BUD_OPENING;
    # config-loaded specs are rebuilt fresh per call (traits.py), never module-load singletons.
    t = get_trait("bud_opening")
    assert t == BUD_OPENING
    assert t.positive_class_name == "open"
    assert t.localization_tolerance_frac == 0.5
    assert t.sliver_frac == 0.5
    assert t.majority_milestone == "95per"
    assert t.majority_provisional is True
    assert t.count_bias_tolerance_frac is None  # not yet authored by the domain expert
    assert set(t.delivers) == {"leaf_out_05per_date", "leaf_out_50per_date"}


def test_reference_fixture_delivers_are_all_in_crops_vocab():
    # Guardrail: the local test fixture must itself obey the controlled vocab it enforces on config,
    # else this whole suite would be exercising an off-vocab trait shape no real config could load.
    vocab = traits._crops_vocab()
    assert vocab, "crops.yml vocab should be loadable in the repo checkout"
    for name in BUD_OPENING.delivers:
        assert name in vocab, name


# ── the re-root: no separate self-rooted database, byte-identical file placement ────

def test_a_fresh_sqlite_project_authors_and_updates_a_spec_with_no_specs_directory_on_disk(
    tmp_path: Path,
):
    # SQLite-only: bound explicitly rather than trusting the ambient default. A project root
    # this module's pytestmark never seeds a record into, so this is genuinely fresh.
    from tcip_store.sqlite_backend import SqliteBackend

    import tcip_store as ts

    ts.bind(SqliteBackend())
    project_root = tmp_path / "fresh"
    specs_dir = traits.trait_specs_dir(project_root)
    assert not specs_dir.exists()

    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"]})
    assert not specs_dir.exists()
    assert [s.name for s in load_trait_specs(specs_dir=specs_dir)] == ["leaf"]

    updated = traits.write_trait_spec_fields(
        "leaf", {"count_bias_tolerance_frac": 1.0}, specs_dir=specs_dir,
    )
    assert updated.count_bias_tolerance_frac == 1.0
    assert not specs_dir.exists()


def test_file_backend_locates_a_trait_spec_at_the_shared_state_trait_specs_path(tmp_path: Path):
    # Direct lookup and enumeration land at <state>/trait_specs/<trait>.json on disk, the
    # byte-identical placement the store's self-rooted predecessor already used.
    from tcip_store.file_backend import FileBackend

    import tcip_store as ts

    ts.bind(FileBackend())
    project_root = tmp_path / "fresh"
    specs_dir = traits.trait_specs_dir(project_root)
    _write_spec(specs_dir, "leaf", {"delivers": ["leaf_length"]})

    on_disk = project_root / ".tcip" / "state" / "trait_specs" / "leaf.json"
    assert on_disk.is_file()
    assert ts.read(traits.trait_spec_key(specs_dir, "leaf"))["name"] == "leaf"
    assert [s.name for s in load_trait_specs(specs_dir=specs_dir)] == ["leaf"]


# ── positive class id resolved from a prediction bucket's own recorded id_map ───────

def _op_sidecar(dir_path: Path, id_map: dict | None, *, dataset_root: Path,
                subject: str = "bud", attribute: str | None = "opening") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": True,
        "trait": "bud_opening",
        "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
        "id_map": id_map,
        "subject": subject,
        "attribute": attribute,
    }
    write_bound_sidecar(dir_path, stamp, dataset_root=dataset_root,
                        experiment_id=f"exp-record-{dir_path.name}",
                        producing_experiment_id="exp-trait-authoring")


def test_resolve_positive_class_id_by_name(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    d = tmp_path / "preds"
    _op_sidecar(d, {"closed": 0, "open": 1}, dataset_root=tmp_path)
    cid, msg = _resolve_positive_class_id("bud_opening", {"2026-02-11": str(d)})
    assert cid == 1
    assert "open" in msg


def test_resolve_positive_class_id_honest_fail_when_absent(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    d = tmp_path / "preds"
    _op_sidecar(d, {"closed": 0, "bud": 1}, dataset_root=tmp_path)  # no 'open' class
    cid, msg = _resolve_positive_class_id("bud_opening", {"2026-02-11": str(d)})
    assert cid is None  # never silently defaults to 1
    assert "open" in msg


def test_resolve_positive_class_id_no_map_is_none(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import _resolve_positive_class_id

    cid, _ = _resolve_positive_class_id("bud_opening", {"2026-02-11": str(tmp_path / "missing")})
    assert cid is None


# ── end-to-end through deliver_phenology_milestones ────────────────────────────

def _pheno_fixture(tmp_path: Path, *, classified: bool):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    root = tmp_path / "ds"
    d1 = root / "predictions" / "run" / "2026-02-11"
    d2 = root / "predictions" / "run" / "2026-03-09"
    id_map = {"closed": 0, "open": 1} if classified else {"bud": 0}
    attribute = "opening" if classified else None
    attrs = {"opening": "open"} if classified else {}
    for d in (d1, d2):
        d.mkdir(parents=True, exist_ok=True)
        json_io.write_annotations(
            d / "P1.json",
            [Annotation(subject="bud", geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9,
                       attributes=attrs)], 8, 8)
        _op_sidecar(d, id_map, dataset_root=root, subject="bud", attribute=attribute)
    from tests._binding_fixtures import write_plant_mapping

    mapping_name = "valley"
    write_plant_mapping(tmp_path, mapping_name, {
        "2026-02-11": [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}],
        "2026-03-09": [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}],
    }, dataset_root=root)
    return mapping_name, d1, d2


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_deliver_phenology_milestones_derives_class_id_and_delivers(tmp_path: Path):
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    mapping_name, d1, d2 = _pheno_fixture(tmp_path, classified=True)
    out_csv = tmp_path / "out.csv"
    classifier_stamp = {
        "validated": True,
        "operating_point": {"classifier": {"value": "open",
                                           "validated_against": "held_out_annotations"}},
        "trait": "bud_opening",
    }
    write_bound_sidecar(d1, classifier_stamp, document="classifier_operating_point",
                        dataset_root=tmp_path / "ds", experiment_id="exp-classifier-derives-id",
                        producing_experiment_id="exp-trait-authoring", trait="bud_opening")

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )
    # The positive class id resolves from the buckets' own recorded id_map; both dimensions are
    # validated, so this delivers.
    assert "error" not in res, res
    assert res["positive_class_assessed"] is True
    assert out_csv.exists()
    assert ("bud_opening_crossing_unconfirmed"
            in phenology.phenology_csv_columns(get_trait("bud_opening")))


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_deliver_phenology_milestones_refuses_when_class_id_unresolvable(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    mapping_name, d1, d2 = _pheno_fixture(tmp_path, classified=False)  # no 'open' anywhere
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert not (tmp_path / "out.csv").exists()

"""Trait entries: what the schema and a proposal admit, how the calibration path reads an entry,
and the derived class id."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tcip_mcp import traits
from tcip_mcp.operationalization import latest_confirmed
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.traits import TraitEntry, TraitUnknownError, trait_names
from tests._binding_fixtures import write_bound_sidecar
from tests._population import mapped_plants
from tests._regime_fixtures import tiled_regime
from tests._trait_fixtures import BUD_OPENING, entry, latest, propose, propose_and_confirm

pytestmark = pytest.mark.usefixtures("seed_bud_trait_spec")


# ── what the schema and a proposal admit ─────────────────────────────────────


def test_an_entry_delivering_a_vocabulary_phenotype_is_admitted_and_reads_back(tmp_path: Path):
    propose(tmp_path, entry("leaf", ("leaf_length",), localization="iou_match",
                            count_objective="detection_f1"))

    spec = latest("leaf")
    assert spec.delivers == ("leaf_length",) and spec.localization == "iou_match"


@pytest.mark.parametrize("fields, refusal", [
    ({"not_a_field": 3}, "not_a_field"),
    ({"provenance": ["name: vocabulary_derived"]}, "provenance"),
    ({"localization": "box"}, "localization"),
])
def test_an_entry_the_schema_refuses_names_why(fields: dict, refusal: str):
    """The closed field set and the stated localization vocabulary refuse by name."""
    base = entry("leaf", ("leaf_length",)).model_dump()
    with pytest.raises(ValidationError, match=refusal):
        TraitEntry.model_validate({**base, **fields})


@pytest.mark.parametrize("delivers, refusal", [
    (("unicorn_horn_length",), "unicorn_horn_length"),
    ((), "delivers must name at least one"),
])
def test_a_proposal_off_the_vocabulary_refuses_and_appends_nothing(
    tmp_path: Path, delivers: tuple, refusal: str,
):
    """The anti-fabrication anchor: a proposal whose ``delivers`` is empty or leaves crops.yml
    refuses by name, and no revision is appended."""
    with pytest.raises(ValueError, match=refusal):
        propose(tmp_path, entry("leaf", delivers))
    assert "leaf" not in trait_names(tmp_path)


def test_count_objective_is_not_a_closed_vocabulary_and_unset_stays_empty():
    """A trait may name any objective an agent has implemented and registered a picker for;
    admission accepts it, and an unset one stays honestly empty rather than defaulted."""
    from tcip_mcp.pipelines.operating_point import COUNT_OBJECTIVE_PICKERS

    assert entry("custom", ("leaf_length",),
                 count_objective="a_brand_new_objective").count_objective == "a_brand_new_objective"
    assert entry("undecided", ("leaf_length",)).count_objective == ""
    for objective in COUNT_OBJECTIVE_PICKERS:
        assert entry("t", ("leaf_length",), count_objective=objective).count_objective == objective


# ── how the calibration path reads an entry ──────────────────────────────────


def test_resolve_operating_point_defaults_an_unset_count_objective_instead_of_refusing(
    tmp_path: Path,
):
    """An unset count_objective defaults to COUNT_UNBIASED and proceeds, stamped as a platform
    default rather than trait-authored, so the distinction is never lost."""
    import tcip_mcp.pipelines.operating_point as OP
    from tcip_mcp.traits import COUNT_UNBIASED

    propose_and_confirm(tmp_path, entry("undecided", ("leaf_length",)))
    bundle = OP.resolve_operating_point("undecided", **tiled_regime(), dataset_hash="h1",
                                        calibration_records=[])
    param = bundle.params["count_objective"]
    assert param._raw == COUNT_UNBIASED
    assert param.source == "default"
    assert "not breeder-confirmed" in param.derived_from


def test_resolve_operating_point_stamps_a_stated_count_objective_as_trait_authored(
    tmp_path: Path,
):
    import tcip_mcp.pipelines.operating_point as OP

    propose_and_confirm(tmp_path, entry("decided", ("leaf_length",), count_objective="detection_f1"))
    bundle = OP.resolve_operating_point("decided", **tiled_regime(), dataset_hash="h1",
                                        calibration_records=[])
    param = bundle.params["count_objective"]
    assert param._raw == "detection_f1"
    assert param.derived_from == "trait-authored"


def test_resolve_operating_point_refuses_an_unregistered_count_objective(tmp_path: Path):
    import tcip_mcp.pipelines.operating_point as OP

    propose_and_confirm(
        tmp_path, entry("custom", ("leaf_length",), count_objective="a_brand_new_objective"))
    with pytest.raises(ValueError, match="no registered picker"):
        OP.resolve_operating_point("custom", **tiled_regime(), dataset_hash="h1",
                                   calibration_records=[])


# ── the record ───────────────────────────────────────────────────────────────


def test_every_trait_record_is_read_and_the_latest_revision_answers(tmp_path: Path):
    propose(tmp_path, entry("leaf", ("leaf_length",), count_bias_tolerance_frac=1.0))
    propose(tmp_path, entry("leaf", ("leaf_length",), count_bias_tolerance_frac=99.0))

    assert set(trait_names()) == {"bud_opening", "leaf"}
    assert latest("leaf").count_bias_tolerance_frac == 99.0


def test_an_unknown_trait_hard_fails():
    with pytest.raises(TraitUnknownError):
        traits.read_trait("banana")


def test_a_measurement_reader_refuses_a_trait_with_no_confirmed_revision_by_name(tmp_path: Path):
    """Calibration reads the latest confirmed revision: an entry proposed and never confirmed
    refuses by name, and a later unconfirmed proposal never changes what the confirmed one says."""
    import tcip_mcp.pipelines.operating_point as OP
    from tcip_mcp.operationalization import OperationalizationRefused

    propose(tmp_path, entry("pending", ("leaf_length",), count_objective="detection_f1"))
    with pytest.raises(OperationalizationRefused, match="'pending'"):
        OP.resolve_operating_point("pending", **tiled_regime(), dataset_hash="h1",
                                   calibration_records=[])

    propose_and_confirm(tmp_path, entry("leaf", ("leaf_length",), count_objective="detection_f1"))
    propose(tmp_path, entry("leaf", ("leaf_length",), count_objective="count_unbiased"))
    bundle = OP.resolve_operating_point("leaf", **tiled_regime(), dataset_hash="h1",
                                        calibration_records=[])
    assert bundle.params["count_objective"]._raw == "detection_f1"


def test_a_proposal_made_while_another_holds_the_record_lands_as_the_next_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Two proposals at once both land: one started while the other has read the record waits
    for it and appends after it rather than overwriting it."""
    import threading

    from tcip_mcp import agent_identity

    real_fields = agent_identity.revision_fields
    other_landed = threading.Event()
    other: list = []

    def propose_other():
        other.append(propose(tmp_path, entry("leaf", ("leaf_length",), notes="the other writer")))
        other_landed.set()

    def fields_while_holding_the_record():
        monkeypatch.setattr(agent_identity, "revision_fields", real_fields)
        threading.Thread(target=propose_other).start()
        other_landed.wait(timeout=2)
        return real_fields()

    monkeypatch.setattr(agent_identity, "revision_fields", fields_while_holding_the_record)
    mine = propose(tmp_path, entry("leaf", ("leaf_length",), notes="this writer"))
    assert other_landed.wait(timeout=30)

    record = traits.read_trait("leaf")
    assert [(r.number, r.entry.notes) for r in record.revisions] == [
        (1, "this writer"), (2, "the other writer")]
    assert (mine.number, other[0].number) == (1, 2)


def test_bud_opening_reads_back_as_the_reference_fixture():
    t = latest_confirmed("bud_opening").entry
    assert t == BUD_OPENING
    assert t.positive_value == "open"
    assert t.localization_tolerance_frac == 0.5
    assert t.majority_milestone == "95per"
    assert t.count_bias_tolerance_frac is None  # not yet authored by the domain expert
    assert set(t.delivers) == {"leaf_out_05per_date", "leaf_out_50per_date"}


def test_the_file_backend_places_a_trait_record_at_the_state_traits_path(tmp_path: Path):
    from tcip_store.file_backend import FileBackend

    import tcip_store as ts

    ts.bind(FileBackend())
    project_root = tmp_path / "fresh"
    propose(project_root, entry("leaf", ("leaf_length",)))

    assert (project_root / ".tcip" / "state" / "traits" / "leaf.json").is_file()
    assert traits.trait_names(project_root) == ["leaf"]


# ── positive class id resolved from a prediction bucket's own recorded id_map ───────

def _op_sidecar(dir_path: Path, id_map: dict | None, *, dataset_root: Path,
                subject: str = "bud", attribute: str | None = "opening") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": True,
        "trait": "bud_opening",
        "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
        "scope": {"subject": subject, "attribute": attribute, "id_map": id_map},
    }
    write_bound_sidecar(dir_path, stamp, dataset_root=dataset_root,
                        experiment_id=f"exp-record-{dir_path.name}",
                        producing_experiment_id="exp-trait-authoring")


def test_resolve_positive_class_id_by_name(tmp_path: Path):
    d = tmp_path / "preds"
    _op_sidecar(d, {"closed": 0, "open": 1}, dataset_root=tmp_path)
    cid, msg = phenology.resolve_positive_class_id(BUD_OPENING, {"2026-02-11": str(d)})
    assert cid == 1
    assert "open" in msg


def test_resolve_positive_class_id_honest_fail_when_absent(tmp_path: Path):
    d = tmp_path / "preds"
    _op_sidecar(d, {"closed": 0, "bud": 1}, dataset_root=tmp_path)  # no 'open' class
    cid, msg = phenology.resolve_positive_class_id(BUD_OPENING, {"2026-02-11": str(d)})
    assert cid is None  # never silently defaults to 1
    assert "open" in msg


def test_resolve_positive_class_id_no_map_is_none(tmp_path: Path):
    cid, _ = phenology.resolve_positive_class_id(
        BUD_OPENING, {"2026-02-11": str(tmp_path / "missing")})
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
        mapping_name=mapping_name, plants=mapped_plants(mapping_name),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(d1)],
    )
    # The positive class id resolves from the buckets' own recorded id_map; both dimensions are
    # validated, so this delivers.
    assert "error" not in res, res
    assert res["positive_class_assessed"] is True
    assert out_csv.exists()


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_deliver_phenology_milestones_refuses_when_class_id_unresolvable(tmp_path: Path):
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    mapping_name, d1, d2 = _pheno_fixture(tmp_path, classified=False)  # no 'open' anywhere
    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name, plants=mapped_plants(mapping_name),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert not (tmp_path / "out.csv").exists()

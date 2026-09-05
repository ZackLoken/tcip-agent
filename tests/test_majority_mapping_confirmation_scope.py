"""What the delivered majority-crossing marker is about, and what it is never about.

A trait's ``majority_provisional`` says one thing: whether the breeders have confirmed that this
trait's "most objects in state" phrase maps to the crossing key the spec names. It travels into the
phenology CSV as its own column, per trait, and it is not the delivery gate's verdict on whether the
numbers beside it were validated. Two different questions, two columns, two independent answers: a
gate-cleared delivery of a trait whose majority reading is still unconfirmed, and a gate refusal
naming each dimension's own reconciled state for a trait whose reading is settled, are both real
cases.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

BUD_SPEC = {
    "name": "bud",
    "delivers": ["leaf_out_05per_date", "leaf_out_50per_date"],
    "positive_class_name": "open",
    "milestone_fractions": [0.05, 0.5, 0.95],
    "milestone_on": "positive_fraction",
    "majority_milestone": "95per",
    "majority_provisional": True,
    "phenology_prefix": "bud",
    "majority_label": "opening",
}

PISTILLATE_SPEC = {
    "name": "pistillate",
    "delivers": ["pistillate_50per_date", "pistillate_flowering_date"],
    "positive_class_name": "open",
    "milestone_fractions": [0.5],
    "milestone_on": "positive_fraction",
    "majority_milestone": "50per",
    "majority_provisional": False,
    "phenology_prefix": "pistillate",
    "majority_label": "flowering",
}


def _write_specs(project_root: Path) -> None:
    """Register both traits and give each a confirmed crossing record, so both can deliver.

    The marker under test travels with a delivered CSV, so each trait needs the confirmed meaning
    the delivery door requires before it will produce one. The two are independent: what a record
    covers is the crossing measurement, never the majority alias the marker qualifies.
    """
    from tcip_mcp import traits

    from tests._operationalization_fixtures import seed_confirmed_crossing

    specs_dir = project_root / traits._TRAIT_SPECS_RELPATH
    for spec in (BUD_SPEC, PISTILLATE_SPEC):
        ts.replace(traits.trait_spec_key(specs_dir, spec["name"]), spec, expect=ts.Version.ABSENT)
    for spec in (BUD_SPEC, PISTILLATE_SPEC):
        seed_confirmed_crossing(project_root, spec["name"])


def _predictions(
    project_root: Path, root: Path, positive_class: str, id_map: dict, *, trait: str,
    attribute: str,
) -> tuple[str, dict]:
    """Two dates of classified predictions for one plant, plus the plant mapping that names it.

    The count operating point always claims validated (only the classifier stamp varies with the
    delivery's own ``validated`` flag), so it always earns a genuine validation record rather than
    asserting one (:mod:`tests._binding_fixtures`).
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tests._binding_fixtures import write_bound_sidecar

    dirs = {}
    for date in ("2026-02-11", "2026-03-09"):
        # A covered-bucket key is relative to a dataset root, recognised by its annotations/predictions segment.
        d = root / "predictions" / "live" / date
        d.mkdir(parents=True, exist_ok=True)
        json_io.write_annotations(
            d / "P1.json",
            [Annotation(subject=trait, geometry=BBox(1.0, 1.0, 4.0, 7.0), score=0.9,
                       attributes={attribute: positive_class})], 16, 9)
        stamp = {
            "validated": True,
            "trait": trait,
            "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
            "id_map": id_map,
            "subject": trait,
            "attribute": attribute,
        }
        write_bound_sidecar(d, stamp, dataset_root=root, experiment_id=f"exp-op-{trait}-{date}",
                            producing_experiment_id="exp-1", trait=trait)
        dirs[date] = str(d)
    from tests._binding_fixtures import write_plant_mapping

    write_plant_mapping(project_root, trait, {
        date: [{"stem": "P1", "plot_name": "P1", "accession_name": "acc-9"}] for date in dirs
    }, dataset_root=root)
    return trait, dirs


def _stamp_classifier(pred_dir: str, trait: str, positive_class: str, *, dataset_root: Path) -> None:
    from tests._binding_fixtures import write_bound_sidecar

    stamp = {
        "validated": True,
        "operating_point": {"classifier": {"value": positive_class,
                                           "validated_against": "held_out_annotations"}},
        "trait": trait,
    }
    write_bound_sidecar(Path(pred_dir), stamp, document="classifier_operating_point",
                        dataset_root=dataset_root, experiment_id=f"exp-cls-{trait}",
                        producing_experiment_id="exp-1", trait=trait)


def _deliver(tmp_path: Path, spec: dict, *, validated: bool) -> dict:
    """Run one trait's phenology delivery.

    Returns the delivered row when every dimension clears the gate, or the door's own refusal
    dict when the classifier is left unvalidated: this door takes no acknowledgement at all, so
    an unvalidated dimension always refuses now.
    """
    root = tmp_path / spec["name"]
    mapping_name, dirs = _predictions(
        tmp_path, root, spec["positive_class_name"],
        {"other": 0, spec["positive_class_name"]: 1}, trait=spec["name"],
        attribute=spec["majority_label"])
    classifier_dirs = None
    if validated:
        first = dirs["2026-02-11"]
        _stamp_classifier(first, spec["name"], spec["positive_class_name"], dataset_root=root)
        classifier_dirs = [first]
    out_csv = root / f"{spec['phenology_prefix']}_phenology.csv"
    res = deliver_phenology_milestones(
        trait=spec["name"],
        mapping_name=mapping_name,
        predictions_by_date=dirs,
        output_csv_path=str(out_csv),
        classifier_pred_dirs=classifier_dirs,
        operating_point_validated="held_out_annotations",
    )
    if not validated:
        return res
    assert "error" not in res, res
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, rows
    return rows[0]


@pytest.fixture(autouse=True)
def _registry(tmp_path: Path) -> None:
    _write_specs(tmp_path)


def test_each_trait_carries_its_own_majority_mapping_marker(tmp_path: Path):
    """Two traits, two different readings: bud's majority phrase maps to its 95% crossing on an
    unconfirmed reading, pistillate's to its 50% crossing on a confirmed one. Each delivery carries
    its own prefix, its own label, and its own answer, with no column of the other trait's."""
    bud = _deliver(tmp_path, BUD_SPEC, validated=True)
    pistillate = _deliver(tmp_path, PISTILLATE_SPEC, validated=True)

    assert bud["bud_opening_crossing_unconfirmed"] == "true"
    assert pistillate["pistillate_flowering_crossing_unconfirmed"] == "false"
    assert "pistillate_flowering_crossing_unconfirmed" not in bud
    assert "bud_opening_crossing_unconfirmed" not in pistillate


def test_the_majority_mapping_marker_is_not_the_delivery_gates_verdict(tmp_path: Path):
    """The marker answers whether the breeders confirmed the majority reading, a different
    question from the delivery gate's own verdict. With one measurement dimension cleared and the
    classifier left unvalidated, this door takes no acknowledgement at all, so the delivery
    refuses, and the refusal still names each dimension's own reconciled state."""
    pistillate = _deliver(tmp_path, PISTILLATE_SPEC, validated=False)

    assert "error" in pistillate
    assert pistillate["positive_state_classifier_validated"] == "false"
    assert pistillate["operating_point_validated"] == "held_out_annotations"


def test_an_unconfirmed_majority_reading_survives_a_cleared_delivery_gate(tmp_path: Path):
    """The other direction: clearing the gate validates the numbers, never the reading. bud's
    majority mapping is still unconfirmed in a delivery whose measurement dimensions all cleared."""
    bud = _deliver(tmp_path, BUD_SPEC, validated=True)

    assert bud["bud_opening_crossing_unconfirmed"] == "true"
    assert bud["operating_point_validated"] != "false"
    assert bud["positive_state_classifier_validated"] != "false"

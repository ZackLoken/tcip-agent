"""Each trait's phenology delivery carries its own majority-crossing column, named from its own
prefix and label and computed at the crossing its confirmed revision states, and a delivery whose
classifier is unvalidated refuses naming each dimension's own reconciled state."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

BUD_SPEC = {
    "name": "bud",
    "delivers": ["leaf_out_05per_date", "leaf_out_50per_date"],
    "positive_value": "open",
    "milestone_fractions": [0.05, 0.5, 0.95],
    "milestone_on": "positive_fraction",
    "majority_milestone": "95per",
    "phenology_prefix": "bud",
    "majority_label": "opening",
}

PISTILLATE_SPEC = {
    "name": "pistillate",
    "delivers": ["pistillate_50per_date", "pistillate_flowering_date"],
    "positive_value": "open",
    "milestone_fractions": [0.5],
    "milestone_on": "positive_fraction",
    "majority_milestone": "50per",
    "phenology_prefix": "pistillate",
    "majority_label": "flowering",
}

from tests._population import mapped_plants


def _write_specs(project_root: Path) -> None:
    """Propose both traits and confirm a revision of each stating its crossing."""
    from tests._trait_fixtures import entry, propose, seed_confirmed_crossing

    for spec in (BUD_SPEC, PISTILLATE_SPEC):
        fields = {k: v for k, v in spec.items() if k not in ("name", "delivers")}
        propose(project_root, entry(spec["name"], spec["delivers"], **fields))
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
        # A covered-bucket key is relative to a dataset root, recognized by its annotations/predictions segment.
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
            "scope": {"subject": trait, "attribute": attribute, "id_map": id_map},
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
    dict when the classifier is left unvalidated: this door takes no acknowledgment at all, so
    an unvalidated dimension always refuses now.
    """
    root = tmp_path / spec["name"]
    mapping_name, dirs = _predictions(
        tmp_path, root, spec["positive_value"],
        {"other": 0, spec["positive_value"]: 1}, trait=spec["name"],
        attribute=spec["majority_label"])
    classifier_dirs = None
    if validated:
        first = dirs["2026-02-11"]
        _stamp_classifier(first, spec["name"], spec["positive_value"], dataset_root=root)
        classifier_dirs = [first]
    out_csv = root / f"{spec['phenology_prefix']}_phenology.csv"
    res = deliver_phenology_milestones(
        trait=spec["name"],
        mapping_name=mapping_name, plants=mapped_plants(mapping_name),
        predictions_by_date=dirs,
        output_csv_path=str(out_csv),
        classifier_pred_dirs=classifier_dirs,
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


def test_each_trait_carries_its_own_majority_crossing_column(tmp_path: Path):
    """bud's majority date is its 95% crossing, pistillate's its 50%: each delivery carries its own
    prefix and label, equal to the crossing its revision names, and no column of the other's."""
    bud = _deliver(tmp_path, BUD_SPEC, validated=True)
    pistillate = _deliver(tmp_path, PISTILLATE_SPEC, validated=True)

    assert bud["bud_opening_date"] == bud["bud_95per_date"]
    assert pistillate["pistillate_flowering_date"] == pistillate["pistillate_50per_date"]
    assert "pistillate_flowering_date" not in bud
    assert "bud_opening_date" not in pistillate


def test_an_unvalidated_classifier_refuses_naming_each_dimension(tmp_path: Path):
    """With the count dimension cleared and the classifier left unvalidated, this door takes no
    acknowledgment, so the delivery refuses, naming each dimension's own reconciled state."""
    pistillate = _deliver(tmp_path, PISTILLATE_SPEC, validated=False)

    assert "error" in pistillate
    assert pistillate["positive_state_classifier_validated"] == "false"
    assert pistillate["operating_point_validated"] == "held_out_annotations"

"""A second registered trait, driven
through both delivery doors (the JSON curve/milestone doors and export_csv), asserting its own
schema and that an unvalidated row is refused. ``registered_traits()`` returning only
``bud_opening`` was the standing gap this closes.

``currant_bloom`` is authored here, in this test file's own pinned platform state root, honestly
tentative: no domain expert has confirmed it, and it exists to prove the delivery mechanism
generalizes to a real *second* trait, not to describe a validated measurement. It deliberately
leaves ``majority_milestone``/``majority_label`` empty rather than copied from bud_opening, since
crops.yml names no "majority" bloom date for currant, proving ``_milestone_columns`` produces the
smaller, no-majority column set for a real second trait, not only for local ``TraitSpec`` shapes
constructed to prove the structural invariant.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tcip_store as ts
from tcip_mcp import traits
from tcip_web.app import app
from tcip_web.state import store

_ID_MAP = {"closed": 0, "open": 1}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _seed_currant_bloom_trait(tmp_path: Path) -> None:
    specs_dir = tmp_path / ".tcip" / "state" / "trait_specs"
    spec = {
        "name": "currant_bloom",
        "count_objective": "count_unbiased",
        "localization": "center_match",
        "localization_tolerance": "half_class_avg_size",
        "localization_tolerance_frac": 0.5,
        "positive_class_name": "open",
        "milestone_fractions": [0.05, 0.50, 0.95],
        "milestone_on": "positive_fraction",
        # No majority alias: crops.yml names no single "most blooms open" date for currant, unlike
        # bud_opening's bud_majority_date. Left empty rather than copied from bud_opening.
        "majority_milestone": "",
        "majority_provisional": False,
        "phenology_prefix": "bloom",
        "majority_label": "",
        "sliver_policy": "class_avg_size",
        "sliver_frac": 0.5,
        "count_bias_tolerance_frac": 0.01,
        "delivers": ["bloom_05per_date", "bloom_50per_date", "bloom_95per_date"],
        "notes": "Test-only, provisional: proves the delivery mechanism "
                 "generalizes to a second trait. Not a domain-expert-confirmed measurement.",
    }
    ts.replace(traits.trait_spec_key(specs_dir, "currant_bloom"), spec, expect=ts.Version.ABSENT)
    # A second trait needs its own confirmed meaning too: nothing about the record is bud_opening-shaped.
    from tests._operationalization_fixtures import seed_confirmed_crossing

    seed_confirmed_crossing(tmp_path, "currant_bloom")


def _currant_bloom_fixture(
    tmp_path: Path, *, validated: bool, fractions: tuple[float, ...] = (0.0, 0.10, 0.60, 1.0),
    detections: int = 10,
) -> dict:
    """Mirrors test_tcip_web_results_routes.py's ``_phenology_fixture`` shape, for currant_bloom's own
    classes/trait name: the same real writers, the same real doors, a different registered trait.

    A validated bucket earns a genuine validation record (:mod:`tests._binding_fixtures`) rather
    than asserting one, since a stamp that claims validated is refused unless a record outside the
    bucket answers for it.
    """
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    from tests._binding_fixtures import write_bound_sidecar

    _seed_currant_bloom_trait(tmp_path)
    dates = ["2026-02-11", "2026-02-25", "2026-03-10", "2026-03-24"][: len(fractions)]
    # A covered-bucket key is relative to a dataset root, recognised by its annotations/predictions segment.
    root = tmp_path / "ds"
    mapping, preds = {}, {}
    for date_str, frac in zip(dates, fractions):
        bucket = root / "predictions" / "live" / date_str
        bucket.mkdir(parents=True, exist_ok=True)
        assigns = []
        for plant in ("BUSH_A", "BUSH_B"):
            stem = f"{plant}_{date_str}_0"
            n_pos = int(round(frac * detections))
            subjects = ["open"] * n_pos + ["closed"] * (detections - n_pos)
            write_predictions_json(
                bucket / f"{stem}.json",
                {"boxes": [[j, 0, j + 4, 4] for j in range(detections)],
                 "labels": [_ID_MAP[s] + 1 for s in subjects],
                 "scores": [0.9] * detections, "width": 100, "height": 100},
                subject="flower", attribute="bloom_state", id_map=_ID_MAP)
            assigns.append({"image_path": f"{stem}.tif", "stem": stem, "plot_name": plant,
                            "accession_name": f"Acc{plant[-1]}", "distance_m": 1.0})
        sidecar: dict = {"id_map": _ID_MAP, "subject": "flower", "attribute": "bloom_state"}
        if validated:
            sidecar.update({
                "validated": True,
                "trait": "currant_bloom",
                "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
                "experiment_id": "exp-1",
                "checkpoint_sha256": "abc123",
            })
            write_bound_sidecar(bucket, sidecar, dataset_root=root,
                                experiment_id=f"exp-op-{date_str}", producing_experiment_id="exp-1",
                                trait="currant_bloom")
            classifier_stamp = {
                "validated": True, "trait": "currant_bloom", "experiment_id": "exp-1",
                "operating_point": {"classifier": {"value": "open",
                                                   "validated_against": "held_out_annotations"}},
            }
            write_bound_sidecar(bucket, classifier_stamp, document="classifier_operating_point",
                                dataset_root=root, experiment_id=f"exp-cls-{date_str}",
                                producing_experiment_id="exp-1", trait="currant_bloom")
        else:
            from tcip_mcp.pipelines.resolution import write_sidecar

            write_sidecar(bucket, sidecar, "operating_point")
        mapping[date_str] = assigns
        preds[date_str] = str(bucket)
    # The Results doors resolve the registry from the delivered buckets' own dataset root, not from
    # the project root the spec and the confirmed record live under.
    from tcip_mcp.class_registry import copy_registry
    from tcip_mcp.dataset_layout import classes_path

    copy_registry(classes_path(tmp_path), classes_path(root))
    from tests._binding_fixtures import write_plant_mapping

    mapping_name = "valley"
    write_plant_mapping(tmp_path, mapping_name, mapping, dataset_root=root)
    # The Results doors serve the project the GUI has open, the one this evidence belongs to.
    store.open_project(tmp_path.resolve())
    return {"project_root": str(tmp_path), "mapping_name": mapping_name,
            "predictions_by_date": preds, "trait": "currant_bloom"}


def _export(client: TestClient, body: dict, payload: str = "milestones", **extra):
    return client.post("/api/results/export_csv",
                       json={**body, "payload": payload, "filename": "x.csv", **extra})


def test_currant_bloom_is_registered_and_distinct_from_bud_opening(tmp_path: Path):
    # $TCIP_STATE_ROOT is already pinned to tmp_path by conftest.py's autouse _pin_platform_root.
    from tcip_mcp.traits import get_trait, registered_traits

    _seed_currant_bloom_trait(tmp_path)
    assert "currant_bloom" in registered_traits()
    t = get_trait("currant_bloom")
    assert t.delivers == ("bloom_05per_date", "bloom_50per_date", "bloom_95per_date")
    assert t.majority_milestone == ""  # no majority alias, unlike bud_opening


def test_currant_bloom_json_doors_deliver_its_own_schema_when_validated(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _currant_bloom_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    resp = client.post("/api/results/phenology_measurement", json=body)
    assert resp.status_code == 200
    out = resp.json()
    assert out["has_unvalidated_dimensions"] is False
    assert out["curves"]["rows"] and out["milestones"]["rows"]
    onset = next(r for r in out["milestones"]["rows"] if r["plant_id"] == "BUSH_A")
    # currant_bloom's own column names, not bud_opening's, and no majority alias column at all.
    assert "bloom_05per_date" in onset and "bloom_50per_date" in onset and "bloom_95per_date" in onset
    assert "bud_majority_date" not in onset
    assert "bloom_opening_date" not in onset  # no phantom majority column either


def test_currant_bloom_export_csv_delivers_its_own_schema(client: TestClient, tmp_path: Path) -> None:
    body = _currant_bloom_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    resp = _export(client, body, "milestones")
    assert resp.status_code == 200
    header = resp.text.splitlines()[0].split(",")
    assert "bloom_05per_date" in header and "bloom_95per_date" in header
    assert not any(c.startswith("bud_") for c in header)


def test_currant_bloom_refuses_unvalidated_evidence_on_every_door(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _currant_bloom_fixture(tmp_path, validated=False)
    resp = client.post("/api/results/phenology_measurement", json=body)
    assert resp.status_code == 400
    assert "operating_point" in resp.json()["detail"]
    for payload in ("curves", "milestones"):
        assert _export(client, body, payload).status_code == 400, payload

"""Slice 3 tests: plant mapping + per-plant curves + onset dates + CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

import tcip_store
from tcip_mcp.audit import audit_log_key
from tcip_mcp.pipelines.resolution import read_operating_point_sidecar, write_sidecar
from tcip_web.app import app
from tcip_web.state import store

from tests._binding_fixtures import producer_checkpoint_sha256

# seed_catkin_operationalization writes the spec plus the confirmed crossing record this root needs.
pytestmark = pytest.mark.usefixtures("seed_catkin_operationalization")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _write_preds(path: Path, subjects: list[str]) -> None:
    """Per-image JSON prediction file with the given classified subjects (the classifier's
    decoded call, straight into ``.subject``)."""
    anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9) for s in subjects]
    write_annotations(str(path), anns, 8, 8)


def _write_id_map_sidecar(dir_path: Path, id_map: dict) -> None:
    (dir_path / "operating_point.json").write_text(json.dumps({"id_map": id_map}), encoding="utf-8")


_ID_MAP = {"dormant": 0, "elongated": 1}


def test_list_traits_names_a_broken_spec_alongside_the_valid_one(
    client: TestClient, tmp_path: Path
) -> None:
    """catkin is seeded valid by the fixture; a second, broken spec must still be visible by
    name and reason, not silently absent the way a dropped spec looks identical to none at all."""
    from tcip_mcp import traits

    specs_dir = tmp_path / ".tcip" / "state" / "trait_specs"
    tcip_store.replace(traits.trait_spec_key(specs_dir, "unicorn"),
                       {"name": "unicorn", "delivers": ["unicorn_horn_length"]},
                       expect=tcip_store.Version.ABSENT)

    resp = client.get("/api/results/traits", params={"project_root": str(tmp_path)})
    body = resp.json()
    assert body["traits"] == ["catkin"]
    assert len(body["invalid_specs"]) == 1
    assert body["invalid_specs"][0]["file"] == "unicorn.json"
    assert "unicorn_horn_length" in body["invalid_specs"][0]["reason"]


def test_plant_mapping_build_requires_a_registered_dataset(
    client: TestClient, tmp_path: Path,
) -> None:
    """A directory that is not a registered dataset's own images/ root refuses, naming that,
    rather than silently building an empty mapping over nothing."""
    store.open_project(tmp_path.resolve())
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "images_root": str(tmp_path / "nope"),
            "plant_csv_paths": [],
        },
    )
    assert resp.status_code == 400
    assert "is not a dataset" in resp.json()["detail"]


def test_plant_mapping_load_missing_returns_empty(client: TestClient, tmp_path: Path) -> None:
    store.open_project(tmp_path.resolve())
    resp = client.post(
        "/api/results/plant_mapping/load",
        json={"name": "missing"},
    )
    body = resp.json()
    assert body["mapping"] == {}


def _phenology_fixture(
    tmp_path: Path, *, validated: bool, images_per_plant: int = 1,
    fractions: tuple[float, ...] = (0.0, 0.10, 0.60, 1.0), id_map: dict | None = None,
    detections: int = 10, producing_experiment_id: str | None = "exp-1",
    count_trait: str = "catkin",
) -> dict:
    """A mapping + per-date prediction buckets, written through the platform's own writers.

    ``validated`` controls only the sidecars' recorded validity, so the same numbers can be driven
    through every door with and without the evidence that qualifies them. A validated bucket earns a
    genuine validation record (:mod:`tests._binding_fixtures`) rather than asserting one, since a
    stamp that claims validated is refused unless a record outside the bucket answers for it. It
    also files the producing run the stamp names, for the same reason one step further out: a
    delivery repeats a producer identity only where an experiment outside the bucket corroborates
    it, so a fixture that only asserted one would deliver blank producer columns.

    The trait and its confirmed operationalization are registered in the very root this body names,
    because that is the root the doors resolve both from. A caller passing a subdirectory gets a
    project that is whole, rather than one whose spec lives somewhere else.

    ``count_trait`` is the trait the count operating point's own sidecar and validation record are
    earned for, ``"catkin"`` by default (matching the delivered trait); a caller wanting a
    stamp/record earned for a different trait than the one the body delivers under passes it, the
    classifier stamp stays earned for ``"catkin"`` regardless, so only the count dimension mismatches.
    """
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    from tests._binding_fixtures import record_producing_run, write_bound_sidecar
    from tests._operationalization_fixtures import seed_confirmed_crossing, write_spec
    from tests._trait_fixtures import CATKIN

    write_spec(tmp_path, CATKIN)
    seed_confirmed_crossing(tmp_path, CATKIN.name)

    # A stamp naming a producing run is only repeated in a delivery when that run really exists.
    checkpoint_sha256 = (record_producing_run(tmp_path, producing_experiment_id)
                         if validated and producing_experiment_id else "abc123")
    id_map = id_map or _ID_MAP
    dates = ["2026-02-11", "2026-02-25", "2026-03-10", "2026-03-24"][: len(fractions)]
    positive = "elongated" if "elongated" in id_map else None
    # A covered-bucket key is relative to a dataset root, recognised by its annotations/predictions segment.
    root = tmp_path / "ds"
    mapping, preds = {}, {}
    for date_str, frac in zip(dates, fractions):
        bucket = root / "predictions" / "live" / date_str
        bucket.mkdir(parents=True, exist_ok=True)
        assigns = []
        for plant in ("PLANT_A", "PLANT_B"):
            for i in range(images_per_plant):
                stem = f"{plant}_{date_str}_{i}"
                n_pos = int(round(frac * detections))
                if positive:
                    subjects = [positive] * n_pos + ["dormant"] * (detections - n_pos)
                else:
                    subjects = [next(iter(id_map))] * detections
                write_predictions_json(
                    bucket / f"{stem}.json",
                    {"boxes": [[j, 0, j + 4, 4] for j in range(detections)],
                     "labels": [id_map[s] + 1 for s in subjects],
                     "scores": [0.9] * detections, "width": 100, "height": 100},
                    id_map=id_map)
                assigns.append({"image_path": f"{stem}.tif", "stem": stem, "plot_name": plant,
                                "accession_name": f"Acc{plant[-1]}", "distance_m": 1.0})
        sidecar: dict = {"id_map": id_map}
        if validated:
            sidecar.update({
                "validated": True,
                "trait": count_trait,
                "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
                "experiment_id": producing_experiment_id,
                "checkpoint_sha256": checkpoint_sha256,
            })
            write_bound_sidecar(bucket, sidecar, dataset_root=root,
                                experiment_id=f"exp-op-{date_str}",
                                producing_experiment_id=producing_experiment_id, trait=count_trait)
            classifier_stamp = {
                "validated": True, "trait": "catkin", "experiment_id": producing_experiment_id,
                "operating_point": {"classifier": {"value": "elongated",
                                                   "validated_against": "held_out_annotations"}},
            }
            write_bound_sidecar(bucket, classifier_stamp, document="classifier_operating_point",
                                dataset_root=root, experiment_id=f"exp-cls-{date_str}",
                                producing_experiment_id=producing_experiment_id, trait="catkin")
        else:
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
            "predictions_by_date": preds, "trait": "catkin"}


def _expected_validation_record(body: dict) -> str:
    """The record cell a delivery from these buckets must carry, read off the buckets' own pointers.

    Derived from each stamp's ``validated_by`` rather than from the delivery being checked, so the
    assertion compares the delivered cell against the records the stamps actually name.
    """
    pointers = set()
    for pred_dir in body["predictions_by_date"].values():
        stamp = read_operating_point_sidecar(pred_dir)
        by = stamp["validated_by"]
        pointers.add(f"{by['experiment_id']}:{by['record_digest']}")
    return "; ".join(sorted(pointers))


def test_phenology_measurement_uses_mapping_and_counts(client: TestClient, tmp_path: Path) -> None:
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    resp = client.post("/api/results/phenology_measurement", json=body)
    assert resp.status_code == 200
    out = resp.json()
    assert out["curves"]["n_plants"] == 2
    assert out["positive_class_assessed"] is True
    by_key = {(r["plant_id"], r["date"]): r for r in out["curves"]["rows"]}
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.75
    assert by_key[("PLANT_B", "2026-02-25")]["ratio"] == 1.0


def test_phenology_measurement_reports_the_real_image_count_not_a_hardcoded_one(
    client: TestClient, tmp_path: Path,
) -> None:
    # The row used to assert `n_images: 1` unconditionally while per_plant_series aggregates every
    # image the mapping names for that (plant, date): the one field in a delivered row derived from
    # nothing at all. Three images per plant per date must read as 3, with counts aggregated over
    # all of them.
    body = _phenology_fixture(tmp_path, validated=True, images_per_plant=3, fractions=(0.5,),
                          detections=4)
    rows = client.post(
        "/api/results/phenology_measurement", json=body).json()["curves"]["rows"]
    row = next(r for r in rows if r["plant_id"] == "PLANT_A")
    assert row["n_images"] == 3
    assert row["n_total"] == 12


def test_phenology_measurement_flags_unclassified_predictions(
    client: TestClient, tmp_path: Path,
) -> None:
    # Bare single-class detector output: the bucket's id_map has no attribute axis, so the run
    # never assessed the positive-state class. Must disclose that rather than pass the counts off
    # as a phenology measurement, and must never report a fabricated ratio.
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0,), id_map={"catkin": 0},
                          detections=2)
    out = client.post("/api/results/phenology_measurement", json=body).json()
    assert out["positive_class_assessed"] is False
    row = out["curves"]["rows"][0]
    assert row["n_total"] == 2
    assert row["n_unclassified"] == 2
    assert row["ratio"] is None


def test_phenology_measurement_finds_crossings(client: TestClient, tmp_path: Path) -> None:
    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    rows = client.post(
        "/api/results/phenology_measurement", json=body).json()["milestones"]["rows"]
    onset = next(r for r in rows if r["plant_id"] == "PLANT_A")
    assert onset["catkin_05per_date"] is not None
    assert onset["catkin_50per_date"] is not None
    assert onset["catkin_95per_date"] is not None
    # catkin_elongation_date = "most catkins elongated" (crops.yml) = the 95% majority crossing.
    assert onset["catkin_elongation_date"] == onset["catkin_95per_date"]


def test_phenology_measurement_ignores_undated_bucket(client: TestClient, tmp_path: Path) -> None:
    # The ingest 'undated/' bucket (and any non-ISO folder) sorts to the (0,0,0) sentinel. It must
    # not crash interpolation or leak '0000-00-00' into a delivered date.
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0, 1.0), detections=10)
    from tcip_mcp.pipelines.postprocessing import plant_mapping as pm

    project_root = Path(body["project_root"])
    build = pm.load_mapping(project_root, body["mapping_name"])
    assert build is not None
    build.assignments["undated"] = build.assignments["2026-02-11"]
    build.dates = sorted([*build.dates, "undated"])
    build.capture_identity["undated"] = build.capture_identity.get("2026-02-11", "0" * 16)
    build.capture_digests["undated"] = build.capture_digests.get("2026-02-11", {})
    build.unreadable["undated"] = []
    pm.persist_mapping(build, project_root, body["mapping_name"])
    body["predictions_by_date"]["undated"] = body["predictions_by_date"]["2026-02-11"]
    rows = client.post(
        "/api/results/phenology_measurement", json=body).json()["milestones"]["rows"]
    for row in rows:
        for key in ("catkin_05per_date", "catkin_50per_date", "catkin_95per_date"):
            assert row[key] != "0000-00-00"
            if row[key] is not None:
                assert row[key].startswith("2026-")


def test_phenology_measurement_discloses_zero_observations_distinctly_from_valid(
    client: TestClient, tmp_path: Path,
) -> None:
    # A plant fully classified and fully observed can still have zero real detections on every date
    # (e.g. before emergence): n_observed_dates lets the GUI distinguish "no observations" from real
    # detection data, which otherwise read identically as "valid" next to blank milestone cells.
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0, 0.0), detections=0)
    row = client.post(
        "/api/results/phenology_measurement", json=body).json()["milestones"]["rows"][0]
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert row["n_observed_dates"] == 0
    assert row["catkin_95per_date"] is None


def test_phenology_doors_reject_a_malformed_payload_with_422_not_500(client: TestClient) -> None:
    # A payload missing required inputs is a structured 422, never an unhandled KeyError/500.
    for route in ("phenology_measurement", "export_csv"):
        resp = client.post(f"/api/results/{route}", json={"trait": "catkin"})
        assert resp.status_code == 422, route


GATE_DOORS = ("phenology_measurement",)


def _export(client: TestClient, body: dict, payload: str = "milestones", **extra):
    return client.post("/api/results/export_csv",
                       json={**body, "payload": payload, "filename": "x.csv", **extra})


def test_every_phenology_door_refuses_unvalidated_evidence(client: TestClient, tmp_path: Path) -> None:
    # Only the CSV door used to reconcile anything; curve/milestone returned unvalidated
    # phenotype until a Download refusal. Both doors now read the same evidence.
    body = _phenology_fixture(tmp_path, validated=False)
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json=body)
        assert resp.status_code == 400, route
        assert "operating_point" in resp.json()["detail"], route
    for payload in ("curves", "milestones"):
        assert _export(client, body, payload).status_code == 400, payload


def test_every_phenology_door_delivers_on_real_bucket_evidence(client: TestClient, tmp_path: Path) -> None:
    # The rail must admit valid work: the identical numbers, with the evidence on disk, ship
    # through every door. Without this the refusals above are satisfiable by a door that always
    # refuses.
    body = _phenology_fixture(tmp_path, validated=True)
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json=body)
        assert resp.status_code == 200, route
        assert resp.json()["provisional"] is False, route
        assert resp.json()["curves"]["rows"], route
        assert resp.json()["milestones"]["rows"], route
    for payload in ("curves", "milestones"):
        resp = _export(client, body, payload)
        assert resp.status_code == 200, payload
        assert resp.headers["content-type"].startswith("text/csv")
        assert len(resp.text.strip().splitlines()) > 1, payload


def test_acknowledge_reveals_provisional_numbers_on_screen_but_never_in_a_file(
    client: TestClient, tmp_path: Path,
) -> None:
    # The non-stranding escape, mirroring deliver_phenology_milestones's acknowledge_unvalidated: a breeder
    # whose operating point is not yet calibrated can LOOK at what they have, clearly marked. A file
    # leaving the platform has no such escape, so the same flag must not open the CSV door.
    body = _phenology_fixture(tmp_path, validated=False)
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json={**body, "acknowledge_unvalidated": True})
        assert resp.status_code == 200, route
        assert resp.json()["provisional"] is True, route
        assert resp.json()["validated"]["operating_point"] == "false", route
    assert _export(client, body, "milestones", acknowledge_unvalidated=True).status_code == 400
    assert _export(client, body, "curves", acknowledge_unvalidated=True).status_code == 400


def _set_tile_provenance(body: dict, tile_size_prov: dict | None) -> dict:
    """Rewrite each bucket's sidecar tile_size entry, leaving every other dimension untouched.

    A validated bucket's claim covers the whole ``operating_point`` field, so changing tile_size
    inside it makes the existing validation record answer for a claim it was never earned against;
    the record is re-earned here, over the mutated stamp, the same way a real producer would.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    from tests._binding_fixtures import file_validation_record

    for bucket_str in body["predictions_by_date"].values():
        bucket = Path(bucket_str)
        sidecar = read_operating_point_sidecar(bucket)
        op = sidecar.setdefault("operating_point", {})
        if tile_size_prov is None:
            op.pop("tile_size", None)
        else:
            op["tile_size"] = tile_size_prov
        if sidecar.get("validated"):
            root = dataset_root_of(bucket)
            sidecar = file_validation_record(
                sidecar, dataset_root=root, pred_dirs=[bucket],
                experiment_id=f"exp-tile-{bucket.name}", producing_experiment_id="exp-1",
                trait=sidecar.get("trait"))
        write_sidecar(bucket, sidecar, "operating_point")
    return body


def _tiled(ref: str, value: int = 640) -> dict:
    return {"value": value, "requires_validation": True, "validation_kind": "geometry",
            "validated_against": ref}


def test_every_phenology_door_refuses_a_fabricated_tile_size(client: TestClient, tmp_path: Path) -> None:
    """A curve is the delivered phenology measurement, built from per-image counts that the tile
    edge scales. A tiled bucket whose tile_size fell back to the fabricated default refuses at every
    Results door, the same way an uncalibrated conf does, even with the classifier and conf both
    genuinely validated on disk."""
    body = _set_tile_provenance(_phenology_fixture(tmp_path, validated=True), _tiled("false"))
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json=body)
        assert resp.status_code == 400, route
        assert "tile_size" in resp.json()["detail"], route
    for payload in ("curves", "milestones"):
        assert _export(client, body, payload).status_code == 400, payload


def test_every_phenology_door_delivers_when_the_tile_scale_has_a_real_basis(
    client: TestClient, tmp_path: Path,
) -> None:
    """The rail must admit valid work: the identical numbers, produced at a tile edge derived from
    the checkpoint's own persisted training geometry, ship through every door."""
    body = _set_tile_provenance(_phenology_fixture(tmp_path, validated=True),
                                _tiled("persisted_training_geometry", 224))
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json=body)
        assert resp.status_code == 200, route
        assert resp.json()["provisional"] is False, route
        assert resp.json()["validated"]["tile_size"] == "persisted_training_geometry", route
    for payload in ("curves", "milestones"):
        assert _export(client, body, payload).status_code == 200, payload


def test_an_untiled_delivery_is_never_gated_on_tile_size(client: TestClient, tmp_path: Path) -> None:
    """Buckets from untiled runs carry a non-gating tile_size entry; the Results doors must not
    acquire a tile-geometry dimension from it and refuse work that was always fine."""
    body = _set_tile_provenance(_phenology_fixture(tmp_path, validated=True),
                                {"value": None, "requires_validation": False,
                                 "validation_kind": None, "validated_against": None})
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json=body)
        assert resp.status_code == 200, route
        assert "tile_size" not in resp.json()["validated"], route


def test_acknowledge_shows_a_fabricated_tile_scale_on_screen_but_never_in_a_file(
    client: TestClient, tmp_path: Path,
) -> None:
    """The same non-stranding escape the other dimensions get: a breeder can look at numbers whose
    tile scale has no basis, clearly marked provisional, and still cannot download them."""
    body = _set_tile_provenance(_phenology_fixture(tmp_path, validated=True), _tiled("false"))
    for route in GATE_DOORS:
        resp = client.post(f"/api/results/{route}", json={**body, "acknowledge_unvalidated": True})
        assert resp.status_code == 200, route
        assert resp.json()["provisional"] is True, route
        assert resp.json()["validated"]["tile_size"] == "false", route
    assert _export(client, body, "milestones", acknowledge_unvalidated=True).status_code == 400


def test_export_refuses_when_nothing_was_ever_classified(client: TestClient, tmp_path: Path) -> None:
    # The same refusal deliver_phenology_milestones makes: with no positive-class axis anywhere, the fraction
    # is not a measurement. Previously only the frontend guarded this on the web side.
    body = _phenology_fixture(tmp_path, validated=True, id_map={"catkin": 0})
    resp = _export(client, body, "milestones")
    assert resp.status_code == 400
    assert "elongated" in resp.json()["detail"]


def test_the_old_declaration_bypass_no_longer_reaches_the_door(
    client: TestClient, tmp_path: Path,
) -> None:
    # Declaring export_kind="diagnostic" and handing over a table shipped a real per-plant curve
    # with 200 and zero validity evidence, because a curve row carries no registry milestone column
    # for the retained floor to catch. None of these caller-composed shapes (a curve, a per-plant
    # count table, milestone dates under renamed columns) can be handed to this door at all: it
    # computes what it exports.
    for rows in (
        [{"plant_id": "P1", "date": "2026-03-01", "n_total": 20, "n_positive": 1, "ratio": 0.05,
          "n_unclassified": 0, "n_missing": 0}],
        [{"plant_id": "P1", "date": "2026-03-15", "positive_count": 42}],
        [{"plant_id": "P1", "start_date": "2026-03-01", "end_date": "2026-04-02"}],
    ):
        resp = client.post("/api/results/export_csv", json={
            "rows": rows, "filename": "x.csv", "export_kind": "diagnostic"})
        assert resp.status_code == 422
        assert not resp.text.startswith("plant_id")


def test_no_caller_field_can_raise_the_reconciled_validity(client: TestClient, tmp_path: Path) -> None:
    # Validity is read from the buckets' own sidecars, so an optimistic caller assertion (under the
    # stamp column names the delivered CSV itself uses) must be inert. Extra fields are ignored by
    # the payload model, so the refusal is byte-identical to the one with no assertion at all.
    body = _phenology_fixture(tmp_path, validated=False)
    bare = client.post("/api/results/phenology_measurement", json=body)
    optimistic = client.post("/api/results/phenology_measurement", json={
        **body,
        "operating_point_validated": "held_out_annotations",
        "positive_state_classifier_validated": "held_out_annotations",
        "validated": {"operating_point": "held_out_annotations", "classifier": "held_out_annotations"},
        "provisional": False,
    })
    assert bare.status_code == optimistic.status_code == 400
    assert bare.json()["detail"] == optimistic.json()["detail"]


def test_a_genuinely_unvalidated_classifier_refuses_even_when_the_count_is_validated(
    client: TestClient, tmp_path: Path,
) -> None:
    # The two dimensions are reconciled separately: a validated count operating point must not
    # carry an unvalidated classifier through. Names the failing dimension so the breeder knows
    # which one to fix.
    body = _phenology_fixture(tmp_path, validated=True)
    for bucket in body["predictions_by_date"].values():
        write_sidecar(Path(bucket), {
            "validated": False, "trait": "catkin",
            "operating_point": {"classifier": {"value": "elongated", "validated_against": "false"}},
        }, "classifier_operating_point")
    resp = client.post("/api/results/phenology_measurement", json=body)
    assert resp.status_code == 400
    assert "['classifier']" in resp.json()["detail"]


def test_exported_milestone_csv_carries_the_canonical_schema_and_its_provenance(
    client: TestClient, tmp_path: Path,
) -> None:
    # The web CSV used to write whatever keys the caller's rows carried, so it lacked the MCP door's
    # provenance columns entirely. It now writes phenology_csv_columns and stamps it from the same
    # reconciliation the gate used, and every column the schema declares is filled, or it would be
    # the same phantom already removed from the schema itself.
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns
    from tcip_mcp.traits import get_trait

    body = _phenology_fixture(tmp_path, validated=True)
    resp = _export(client, body, "milestones")
    assert resp.status_code == 200
    header, first = resp.text.splitlines()[0].split(","), resp.text.splitlines()[1].split(",")
    assert header == phenology_csv_columns(get_trait("catkin"))
    cells = dict(zip(header, first))
    assert cells["operating_point_validated"] == "held_out_annotations"
    assert cells["positive_state_classifier_validated"] == "held_out_annotations"
    assert cells["operating_point_conf"] == "0.4"
    assert cells["producing_experiment_id"] == "exp-1"
    assert cells["producer_model_sha256"] == producer_checkpoint_sha256("exp-1")
    assert cells["validation_record"] == _expected_validation_record(body)
    # plant_csvs_unverified and unvalidated_dimensions are legitimately empty here (no plant CSV
    # to check, nothing floored); captures_unverified is not: no images/ tree at all in this fixture.
    exempt = {"plant_csvs_unverified", "unvalidated_dimensions"}
    assert [c for c, v in cells.items() if v == "" and c not in exempt] == []
    assert cells["plant_csvs_unverified"] == ""
    assert cells["unvalidated_dimensions"] == ""
    assert cells["captures_unverified"] != ""


def test_curve_and_milestone_projections_share_one_measurement(
    client: TestClient, tmp_path: Path,
) -> None:
    # One request computes both projections off one per_plant_phenology result, so a milestone
    # date and the curve it was read off cannot come from different numbers.
    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    out = client.post("/api/results/phenology_measurement", json=body).json()
    curves, milestones = out["curves"]["rows"], out["milestones"]["rows"]
    a_curve = sorted((r for r in curves if r["plant_id"] == "PLANT_A"), key=lambda r: r["date"])
    a_row = next(r for r in milestones if r["plant_id"] == "PLANT_A")
    assert a_row["n_dates"] == len(a_curve)
    assert a_row["n_observed_dates"] == sum(1 for r in a_curve if r["ratio"] is not None)
    # The 95% crossing must fall inside the dates whose ratios actually bracket it.
    assert a_curve[-1]["ratio"] == 1.0
    assert a_row["catkin_95per_date"] <= a_curve[-1]["date"]


def test_phenology_measurement_response_carries_every_field_the_two_deleted_doors_did(
    client: TestClient, tmp_path: Path,
) -> None:
    """Field-by-field parity: the merged door's ``curves`` projection against per_plant_curves'
    old ``{rows, n_plants, positive_class_id, **disclosure}`` shape, ``milestones`` against
    onset_dates' old ``{rows, **disclosure}`` shape (disclosure now sits once at the top level,
    identical for both since one measurement produced it). The two routes are deleted, so this
    reconstructs what each one used to assemble from the surviving producer
    (_PhenologyMeasurement.curve_rows/milestone_rows, _disclosure) it always delegated to, rather
    than calling a route that no longer exists.
    """
    from tcip_web.routes.results import PhenologyPayload, _disclosure, _measure_phenology

    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    response = client.post("/api/results/phenology_measurement", json=body).json()

    measurement = _measure_phenology(PhenologyPayload(**body))
    assert response["curves"] == {
        "rows": measurement.curve_rows(),
        "n_plants": len(measurement.plants["rows"]),
        "positive_class_id": measurement.positive_class_id,
    }
    assert response["milestones"] == {"rows": measurement.milestone_rows()}
    for key, value in _disclosure(measurement).items():
        assert response[key] == value, key


def test_web_and_mcp_phenology_doors_agree_on_validity(client: TestClient, tmp_path: Path) -> None:
    # The web route's phenology_measurement and the MCP tool's deliver_phenology_milestones read the same
    # on-disk evidence through the identical tcip_mcp.pipelines.resolution reconciliation.
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    web_validated = client.post(
        "/api/results/phenology_measurement", json=body).json()["validated"]

    mcp_result = deliver_phenology_milestones(
        trait=body["trait"], mapping_name=body["mapping_name"],
        predictions_by_date=body["predictions_by_date"], output_csv_path=str(tmp_path / "out.csv"),
        classifier_pred_dirs=list(body["predictions_by_date"].values()),
    )
    assert "error" not in mcp_result, mcp_result
    assert mcp_result["operating_point_validated"] == web_validated["operating_point"]
    assert mcp_result["positive_state_classifier_validated"] == web_validated["classifier"]


def _rewrite_classifier_sidecars(body: dict, **overrides) -> None:
    """A genuine classifier record, so a wrong-trait/wrong-experiment refusal comes from the
    disagreement a real record surfaces rather than from an absent one. ``experiment_id`` here is
    the producing run the record is checked against (``bind_classifier_validity``'s
    ``producing_experiment_id``), not the log the record itself is filed in.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    from tests._binding_fixtures import write_bound_sidecar

    for bucket in body["predictions_by_date"].values():
        bucket_path = Path(bucket)
        sidecar = {"validated": True, "trait": "catkin", "experiment_id": "exp-1",
                   "operating_point": {"classifier": {"value": "elongated",
                                                      "validated_against": "held_out_annotations"}}}
        sidecar.update(overrides)
        root = dataset_root_of(bucket_path)
        write_bound_sidecar(
            bucket_path, sidecar, document="classifier_operating_point", dataset_root=root,
            experiment_id=f"exp-cls-record-{bucket_path.name}",
            producing_experiment_id=sidecar.get("experiment_id"), trait=sidecar.get("trait"))


def test_a_classifier_calibrated_for_another_trait_does_not_validate_this_delivery(
    client: TestClient, tmp_path: Path,
) -> None:
    # deliver_phenology_milestones binds the classifier stamp to the delivery: a sidecar recorded against a
    # different trait, or against a run that did not produce these predictions, is not trusted. The
    # web door reconciled without that binding, so it accepted a stamp the MCP door rejects, and
    # then wrote it into the delivered CSV as positive_state_classifier_validated. Both doors now
    # call the one shared binding.
    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, trait="chestnut_bur")
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400
    assert "was earned for trait" in resp.json()["detail"]
    assert "chestnut_bur" in resp.json()["detail"]
    assert client.post(
        "/api/results/phenology_measurement", json=body).status_code == 400


def test_a_classifier_calibrated_against_another_experiment_does_not_validate_this_delivery(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, experiment_id="exp-OTHER")
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400
    assert "records producing run 'exp-OTHER'" in resp.json()["detail"]


def test_a_correctly_bound_classifier_still_delivers(client: TestClient, tmp_path: Path) -> None:
    # The refusals above must not be satisfiable by a door that refuses every classifier stamp.
    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, trait="catkin", experiment_id="exp-1")
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 200
    # A foreign checkpoint names no producing run on either document; two absences agree, so nothing refuses.
    foreign = _phenology_fixture(tmp_path / "foreign", validated=True, producing_experiment_id=None)
    _rewrite_classifier_sidecars(foreign, trait="catkin", experiment_id=None)
    assert client.post("/api/results/export_csv",
                       json={**foreign, "payload": "milestones", "filename": "x.csv"}
                       ).status_code == 200


def test_the_web_export_records_what_verification_found_in_the_datasets_own_log(
    client: TestClient, tmp_path: Path,
) -> None:
    """The audited arguments say what was asked for; this says which buckets stood behind the
    numbers and which records answered for them, in the log that travels with the data. The same
    event the MCP door emits, from the door that writes the same schema."""
    body = _phenology_fixture(tmp_path, validated=True)
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 200, resp.text[:300]

    emitted = tcip_store.read_log(audit_log_key(tmp_path / "ds")).records
    assert emitted, "the delivery wrote nothing to the log of the dataset its buckets sit in"
    events = [e for e in emitted if e["tool"] == "results.export_csv" and "verified_buckets" in e]
    assert len(events) == 1, emitted
    verified = events[0]["verified_buckets"]
    assert set(verified) == set(body["predictions_by_date"].values())
    assert all(v["verified"] for v in verified.values())
    assert events[0]["record_digests"], events[0]


def test_export_csv_also_saves_the_delivery_into_the_projects_exports_dir(
    client: TestClient, tmp_path: Path,
) -> None:
    """The browser download is the breeder's copy; the delivery itself belongs to the project, so
    the identical bytes land in <project>/results_export/, and the write is audited. The response
    is the saved file's bytes: the route reads ``saved_path`` back and returns it unchanged, so
    what is worth pinning is what that file carries, its header held to the schema's own column
    owner and each row's provenance cells held to what the writer composed, not a second read of
    the same bytes this route already returned."""
    from tcip_mcp.pipelines.postprocessing import phenology

    from tests._trait_fixtures import CATKIN

    body = _phenology_fixture(tmp_path, validated=True)
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "catkin_delivery.csv"})
    assert resp.status_code == 200
    saved = tmp_path / "results_export" / "catkin_delivery.csv"
    assert resp.headers["X-TCIP-Saved-To"] == str(saved)

    lines = resp.content.decode("utf-8").splitlines()
    header = lines[0].split(",")
    assert header == phenology.phenology_csv_columns(CATKIN)
    for line in lines[1:]:
        cells = dict(zip(header, line.split(",")))
        assert cells["operating_point_conf"] == "0.4"
        assert cells["operating_point_validated"] == "held_out_annotations"
        assert cells["positive_state_classifier_validated"] == "held_out_annotations"
        assert cells["producing_experiment_id"] == "exp-1"
        assert cells["validation_record"] == _expected_validation_record(body)

    audit = tcip_store.read_log(audit_log_key(tmp_path)).records
    assert any(e.get("tool") == "results.export_csv" for e in audit)
    assert any("catkin_delivery.csv" in json.dumps(e) for e in audit)


def test_the_curves_csv_carries_the_same_provenance_as_the_milestone_csv(
    client: TestClient, tmp_path: Path,
) -> None:
    # A curve is the same delivered phenology measurement, un-summarised, which is why it takes the
    # identical gate. It must therefore carry the identical evidence: the milestone branch was
    # stamped while the curve branch wrote the bare aggregation, so half of what this door delivers
    # was unauditable.
    # The expected names are spelled out rather than imported from the module under test, so this
    # fails on the delivered BYTES when the stamp is absent, not on a missing symbol.
    provenance = ["operating_point_conf", "operating_point_validated",
                  "positive_state_classifier_validated", "unvalidated_dimensions",
                  "producer_model_sha256", "producing_experiment_id", "produced_at",
                  "validation_record", "plant_mapping_sha256", "captures_unverified",
                  "plant_csvs_unverified", "dates_delivered", "images_unattributed",
                  "plant_attribution"]
    body = _phenology_fixture(tmp_path, validated=True)
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "curves", "filename": "c.csv"})
    assert resp.status_code == 200
    header = resp.text.splitlines()[0].split(",")
    assert header[-len(provenance):] == provenance
    cells = dict(zip(header, resp.text.splitlines()[1].split(",")))
    for col in provenance:
        if col in ("plant_csvs_unverified", "unvalidated_dimensions"):
            continue  # legitimately empty: no plant CSV to check, and nothing floored
        assert cells[col] != "", col
    assert cells["operating_point_validated"] == "held_out_annotations"
    assert cells["producing_experiment_id"] == "exp-1"
    assert cells["validation_record"] == _expected_validation_record(body)


def test_export_refuses_a_bucket_whose_id_map_never_carried_the_positive_class(
    client: TestClient, tmp_path: Path,
) -> None:
    # deliver_phenology_milestones's first guard is that the positive class id resolves from some bucket's own
    # recorded id_map. per_plant_phenology's positive_class_assessed flag is not a substitute: a date
    # with ZERO detections is trivially "fully classified", so an axis-less bucket with no
    # detections read as classified and the export door had nothing left to refuse on.
    body = _phenology_fixture(tmp_path, validated=True, id_map={"catkin": 0}, detections=0)
    measured = client.post("/api/results/phenology_measurement", json=body)
    assert measured.status_code == 200
    assert measured.json()["positive_class_assessed"] is False
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400
    assert "elongated" in resp.json()["detail"]


def test_registered_models_confines_project_path_to_allowed_roots(
    client: TestClient, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path_factory.mktemp("outside")
    resp = client.get("/api/results/models/registered", params={"project_path": str(outside)})
    assert resp.status_code == 403


def test_registered_models_admits_a_project_path_with_no_extra_roots_set(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """The rail must admit valid work: a project's own directory answers fine with
    TCIP_IMAGE_ROOTS unset, since the workspace root alone already admits it."""
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    resp = client.get("/api/results/models/registered", params={"project_path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["models"] == []


def test_registered_models_answers_a_resolved_absolute_checkpoint_path(
    client: TestClient, tmp_path: Path,
) -> None:
    """A relative stored checkpoint_path (the registry's own internal spelling) still answers
    absolute over this route, the surface the Inference tab feeds straight back into a launch."""
    from tcip_mcp.model_registry import ModelRegistry

    ckpt_dir = tmp_path / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "m.pt"
    ckpt.write_bytes(b"route fixture weights")
    ModelRegistry(str(tmp_path)).register_model("m", str(ckpt), {}, metrics_source=None)

    resp = client.get("/api/results/models/registered", params={"project_path": str(tmp_path)})

    assert resp.status_code == 200
    body = resp.json()
    assert Path(body["models"][0]["checkpoint_path"]) == ckpt.resolve()


def test_inference_launch_missing_checkpoint(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/inference/launch",
        json={
            "checkpoint_path": str(tmp_path / "no.pt"),
            "dataset_root": str(tmp_path),
            "model_name": "baseline",
            "date": "2026-02-11",
        },
    )
    assert resp.status_code == 404


def test_inference_list_jobs_endpoint(client: TestClient) -> None:
    resp = client.get("/api/inference/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()


def test_inference_list_jobs_carries_each_jobs_warning(client: TestClient) -> None:
    """Each row of the list route is ``_summary``'s, the one producer the stream and the persisted
    registry use, so a job's warning reaches the poll as it reaches the stream."""
    from tcip_web.routes import inference as inference_routes

    job = inference_routes.InferenceJob(
        job_id="inf-warn-test", checkpoint_path="", images_dir="", output_dir="",
        conf=0.25, iou=0.5, slice_hw=(0, 0), overlap=0.0,
        warning="3 images carried no readable capture date",
    )
    inference_routes._register(job)
    try:
        row = next(r for r in client.get("/api/inference/jobs").json()["jobs"]
                   if r["job_id"] == "inf-warn-test")
        assert row["warning"] == "3 images carried no readable capture date"
    finally:
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop("inf-warn-test", None)


def test_inference_stream_to_a_missing_job_sends_a_typed_terminal_frame(
    client: TestClient,
) -> None:
    """The not-found frame is typed the same as a run's own terminal frame, so the client stops
    reconnecting against a job that will never exist instead of retrying forever."""
    with client.websocket_connect("ws://127.0.0.1/api/inference/jobs/does-not-exist/stream") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "final"
    assert frame["error"] == "job not found"


def test_inference_by_id_job_route_is_retired(client: TestClient) -> None:
    """``GET /api/inference/jobs/{job_id}`` duplicated the list route over the same registry and
    had no caller; registering a job first proves this refuses a real job, not only an absent
    one (an unregistered id already answers 404 without this route existing at all)."""
    from tcip_web.routes import inference as inference_routes

    job = inference_routes.InferenceJob(
        job_id="inf-retired-test", checkpoint_path="", images_dir="", output_dir="",
        conf=0.25, iou=0.5, slice_hw=(0, 0), overlap=0.0,
    )
    inference_routes._register(job)
    try:
        assert client.get("/api/inference/jobs/inf-retired-test").status_code == 404
    finally:
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop("inf-retired-test", None)


def test_phenology_measurement_refuses_when_the_delivered_dataset_carries_no_registry(
    client: TestClient, tmp_path: Path,
) -> None:
    """The confirmed record and its registry live at the project root; the delivered buckets
    resolve to a different dataset root that carries none. The door refuses by name rather than
    silently checking the project root's own, unrelated registry."""
    bucket = tmp_path / "ds" / "predictions" / "live" / "2026-02-11"
    bucket.mkdir(parents=True)
    store.open_project(tmp_path.resolve())

    resp = client.post("/api/results/phenology_measurement", json={
        "project_root": str(tmp_path), "mapping_name": "valley",
        "predictions_by_date": {"2026-02-11": str(bucket)}, "trait": "catkin",
    })

    assert resp.status_code == 400
    assert "no class registry is reachable" in resp.json()["detail"]

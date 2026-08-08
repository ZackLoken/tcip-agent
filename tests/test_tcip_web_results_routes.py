"""Slice 3 tests: plant mapping + per-plant curves + onset dates + CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

from tcip_web.app import app

# No built-in traits: seed_catkin_trait_spec (conftest.py) writes a real
# catkin.yml into this test's pinned project root so get_trait("catkin") keeps resolving by default.
pytestmark = pytest.mark.usefixtures("seed_catkin_trait_spec")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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
    specs_dir = tmp_path / ".tcip" / "state" / "trait_specs"
    (specs_dir / "unicorn.yml").write_text(
        "name: unicorn\ndelivers: [unicorn_horn_length]\n", encoding="utf-8")

    resp = client.get("/api/results/traits", params={"project_root": str(tmp_path)})
    body = resp.json()
    assert body["traits"] == ["catkin"]
    assert len(body["invalid_specs"]) == 1
    assert body["invalid_specs"][0]["file"] == "unicorn.yml"
    assert "unicorn_horn_length" in body["invalid_specs"][0]["reason"]


def test_plant_mapping_build_with_empty_images(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "images_root": str(tmp_path / "nope"),
            "plant_csv_paths": [],
            "persist_path": str(tmp_path / "mapping.json"),
        },
    )
    body = resp.json()
    assert body["mapping"] == {}
    assert body["summary"] == {}


def test_plant_mapping_load_missing_returns_empty(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/results/plant_mapping/load",
        json={"persist_path": str(tmp_path / "missing.json")},
    )
    body = resp.json()
    assert body["mapping"] == {}


def _phenology_fixture(
    tmp_path: Path, *, validated: bool, images_per_plant: int = 1,
    fractions: tuple[float, ...] = (0.0, 0.10, 0.60, 1.0), id_map: dict | None = None,
    detections: int = 10,
) -> dict:
    """A mapping + per-date prediction buckets, written through the platform's own writers.

    ``validated`` controls only the sidecars' recorded validity, so the same numbers can be driven
    through every door with and without the evidence that qualifies them.
    """
    from tcip_mcp.pipelines.postprocessing.export import write_predictions_json

    id_map = id_map or _ID_MAP
    dates = ["2026-02-11", "2026-02-25", "2026-03-10", "2026-03-24"][: len(fractions)]
    positive = "elongated" if "elongated" in id_map else None
    mapping, preds = {}, {}
    for date_str, frac in zip(dates, fractions):
        bucket = tmp_path / "preds" / date_str
        bucket.mkdir(parents=True, exist_ok=True)
        sidecar: dict = {"id_map": id_map}
        if validated:
            sidecar.update({
                "validated": True,
                "operating_point": {"conf": {"value": 0.4, "validated_against": "held_out_annotations"}},
                "experiment_id": "exp-1",
                "checkpoint_sha256": "abc123",
            })
            (bucket / "classifier_operating_point.json").write_text(json.dumps({
                "validated": True, "trait": "catkin", "experiment_id": "exp-1",
                "operating_point": {"classifier": {"value": "elongated",
                                                   "validated_against": "held_out_annotations"}},
            }), encoding="utf-8")
        (bucket / "operating_point.json").write_text(json.dumps(sidecar), encoding="utf-8")
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
        mapping[date_str] = assigns
        preds[date_str] = str(bucket)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    return {"project_root": str(tmp_path), "mapping_path": str(mapping_path),
            "predictions_by_date": preds, "trait": "catkin"}


def test_per_plant_curves_uses_mapping_and_counts(client: TestClient, tmp_path: Path) -> None:
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.75, 1.0), detections=4)
    resp = client.post("/api/results/per_plant_curves", json=body)
    assert resp.status_code == 200
    out = resp.json()
    assert out["n_plants"] == 2
    assert out["positive_class_assessed"] is True
    by_key = {(r["plant_id"], r["date"]): r for r in out["rows"]}
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.75
    assert by_key[("PLANT_B", "2026-02-25")]["ratio"] == 1.0


def test_per_plant_curves_reports_the_real_image_count_not_a_hardcoded_one(
    client: TestClient, tmp_path: Path,
) -> None:
    # The row used to assert `n_images: 1` unconditionally while per_plant_series aggregates every
    # image the mapping names for that (plant, date): the one field in a delivered row derived from
    # nothing at all. Three images per plant per date must read as 3, with counts aggregated over
    # all of them.
    body = _phenology_fixture(tmp_path, validated=True, images_per_plant=3, fractions=(0.5,),
                          detections=4)
    rows = client.post("/api/results/per_plant_curves", json=body).json()["rows"]
    row = next(r for r in rows if r["plant_id"] == "PLANT_A")
    assert row["n_images"] == 3
    assert row["n_total"] == 12


def test_per_plant_curves_flags_unclassified_predictions(client: TestClient, tmp_path: Path) -> None:
    # Bare single-class detector output: the bucket's id_map has no attribute axis, so the run
    # never assessed the positive-state class. Must disclose that rather than pass the counts off
    # as a phenology measurement, and must never report a fabricated ratio.
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0,), id_map={"catkin": 0},
                          detections=2)
    out = client.post("/api/results/per_plant_curves", json=body).json()
    assert out["positive_class_assessed"] is False
    row = out["rows"][0]
    assert row["n_total"] == 2
    assert row["n_unclassified"] == 2
    assert row["ratio"] is None


def test_onset_dates_finds_crossings(client: TestClient, tmp_path: Path) -> None:
    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    rows = client.post("/api/results/onset_dates", json=body).json()["rows"]
    onset = next(r for r in rows if r["plant_id"] == "PLANT_A")
    assert onset["catkin_05per_date"] is not None
    assert onset["catkin_50per_date"] is not None
    assert onset["catkin_95per_date"] is not None
    # catkin_elongation_date = "most catkins elongated" (crops.yml) = the 95% majority crossing.
    assert onset["catkin_elongation_date"] == onset["catkin_95per_date"]


def test_onset_dates_ignores_undated_bucket(client: TestClient, tmp_path: Path) -> None:
    # The ingest 'undated/' bucket (and any non-ISO folder) sorts to the (0,0,0) sentinel. It must
    # not crash interpolation or leak '0000-00-00' into a delivered date.
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0, 1.0), detections=10)
    mapping_path = Path(body["mapping_path"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["undated"] = mapping["2026-02-11"]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    body["predictions_by_date"]["undated"] = body["predictions_by_date"]["2026-02-11"]
    rows = client.post("/api/results/onset_dates", json=body).json()["rows"]
    for row in rows:
        for key in ("catkin_05per_date", "catkin_50per_date", "catkin_95per_date"):
            assert row[key] != "0000-00-00"
            if row[key] is not None:
                assert row[key].startswith("2026-")


def test_onset_dates_discloses_zero_observations_distinctly_from_valid(
    client: TestClient, tmp_path: Path,
) -> None:
    # A plant fully classified and fully observed can still have zero real detections on every date
    # (e.g. before emergence): n_observed_dates lets the GUI distinguish "no observations" from real
    # detection data, which otherwise read identically as "valid" next to blank milestone cells.
    body = _phenology_fixture(tmp_path, validated=True, fractions=(0.0, 0.0), detections=0)
    row = client.post("/api/results/onset_dates", json=body).json()["rows"][0]
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert row["n_observed_dates"] == 0
    assert row["catkin_95per_date"] is None


def test_phenology_doors_reject_a_malformed_payload_with_422_not_500(client: TestClient) -> None:
    # A payload missing required inputs is a structured 422, never an unhandled KeyError/500.
    for route in ("per_plant_curves", "onset_dates", "export_csv"):
        resp = client.post(f"/api/results/{route}", json={"trait": "catkin"})
        assert resp.status_code == 422, route


GATE_DOORS = ("per_plant_curves", "onset_dates")


def _export(client: TestClient, body: dict, payload: str = "milestones", **extra):
    return client.post("/api/results/export_csv",
                       json={**body, "payload": payload, "filename": "x.csv", **extra})


def test_every_phenology_door_refuses_unvalidated_evidence(client: TestClient, tmp_path: Path) -> None:
    # Only the CSV door used to reconcile anything, so the curve and milestone doors returned the
    # same phenotype with no gate at all: the breeder read unvalidated phenology dates on screen and
    # met the refusal only on clicking Download. All three now read the same evidence.
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
        assert resp.json()["rows"], route
    for payload in ("curves", "milestones"):
        resp = _export(client, body, payload)
        assert resp.status_code == 200, payload
        assert resp.headers["content-type"].startswith("text/csv")
        assert len(resp.text.strip().splitlines()) > 1, payload


def test_acknowledge_reveals_provisional_numbers_on_screen_but_never_in_a_file(
    client: TestClient, tmp_path: Path,
) -> None:
    # The non-stranding escape, mirroring compute_phenology's acknowledge_unvalidated: a breeder
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
    """Rewrite each bucket's sidecar tile_size entry, leaving every other dimension untouched."""
    for bucket in body["predictions_by_date"].values():
        path = Path(bucket) / "operating_point.json"
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        op = sidecar.setdefault("operating_point", {})
        if tile_size_prov is None:
            op.pop("tile_size", None)
        else:
            op["tile_size"] = tile_size_prov
        path.write_text(json.dumps(sidecar), encoding="utf-8")
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
    # The same refusal compute_phenology makes: with no positive-class axis anywhere, the fraction
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
    bare = client.post("/api/results/per_plant_curves", json=body)
    optimistic = client.post("/api/results/per_plant_curves", json={
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
        (Path(bucket) / "classifier_operating_point.json").write_text(json.dumps({
            "validated": False, "trait": "catkin",
            "operating_point": {"classifier": {"value": "elongated", "validated_against": "false"}},
        }), encoding="utf-8")
    resp = client.post("/api/results/onset_dates", json=body)
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
    assert cells["producer_experiment_id"] == "exp-1"
    assert cells["producer_model_sha256"] == "abc123"
    assert [c for c, v in cells.items() if v == ""] == []


def test_curve_and_milestone_doors_report_the_same_measurement(
    client: TestClient, tmp_path: Path,
) -> None:
    # Both doors project one per_plant_phenology result, so a milestone date and the curve it was
    # read off cannot come from different numbers. The milestone door used to recompute from rows
    # the client handed back.
    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    curves = client.post("/api/results/per_plant_curves", json=body).json()["rows"]
    milestones = client.post("/api/results/onset_dates", json=body).json()["rows"]
    a_curve = sorted((r for r in curves if r["plant_id"] == "PLANT_A"), key=lambda r: r["date"])
    a_row = next(r for r in milestones if r["plant_id"] == "PLANT_A")
    assert a_row["n_dates"] == len(a_curve)
    assert a_row["n_observed_dates"] == sum(1 for r in a_curve if r["ratio"] is not None)
    # The 95% crossing must fall inside the dates whose ratios actually bracket it.
    assert a_curve[-1]["ratio"] == 1.0
    assert a_row["catkin_95per_date"] <= a_curve[-1]["date"]


def test_web_and_mcp_phenology_doors_agree_on_validity(client: TestClient, tmp_path: Path) -> None:
    # Two delivery doors read the same on-disk evidence for the same trait: the web route's
    # onset_dates and the MCP tool's compute_phenology. Both route through the identical
    # tcip_mcp.pipelines.resolution reconciliation functions, so identical inputs must stamp
    # identical validity, never two independently-derived answers that happen to usually agree.
    from tcip_mcp.tools.phenology_tools import compute_phenology

    body = _phenology_fixture(tmp_path, validated=True, detections=100)
    web_validated = client.post("/api/results/onset_dates", json=body).json()["validated"]

    mcp_result = compute_phenology(
        trait=body["trait"], mapping_path=body["mapping_path"],
        predictions_by_date=body["predictions_by_date"], output_csv_path=str(tmp_path / "out.csv"),
        classifier_pred_dirs=list(body["predictions_by_date"].values()),
    )
    assert "error" not in mcp_result, mcp_result
    assert mcp_result["operating_point_validated"] == web_validated["operating_point"]
    assert mcp_result["positive_state_classifier_validated"] == web_validated["classifier"]


def _rewrite_classifier_sidecars(body: dict, **overrides) -> None:
    for bucket in body["predictions_by_date"].values():
        sidecar = {"validated": True, "trait": "catkin", "experiment_id": "exp-1",
                   "operating_point": {"classifier": {"value": "elongated",
                                                      "validated_against": "held_out_annotations"}}}
        sidecar.update(overrides)
        (Path(bucket) / "classifier_operating_point.json").write_text(
            json.dumps(sidecar), encoding="utf-8")


def test_a_classifier_calibrated_for_another_trait_does_not_validate_this_delivery(
    client: TestClient, tmp_path: Path,
) -> None:
    # compute_phenology binds the classifier stamp to the delivery: a sidecar recorded against a
    # different trait, or against a run that did not produce these predictions, is not trusted. The
    # web door reconciled without that binding, so it accepted a stamp the MCP door rejects, and
    # then wrote it into the delivered CSV as positive_state_classifier_validated. Both doors now
    # call the one shared binding.
    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, trait="chestnut_bur")
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400
    assert "calibrated for trait" in resp.json()["detail"]
    assert client.post("/api/results/onset_dates", json=body).status_code == 400


def test_a_classifier_calibrated_against_another_experiment_does_not_validate_this_delivery(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, experiment_id="exp-OTHER")
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400
    assert "calibrated against experiment" in resp.json()["detail"]


def test_a_correctly_bound_classifier_still_delivers(client: TestClient, tmp_path: Path) -> None:
    # The refusals above must not be satisfiable by a door that refuses every classifier stamp.
    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, trait="catkin", experiment_id="exp-1")
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 200
    # A foreign checkpoint with no recorded experiment_id is not rejected for lacking one to
    # compare against; only a real mismatch is.
    _rewrite_classifier_sidecars(body, trait="catkin", experiment_id=None)
    assert client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"}).status_code == 200


def test_export_csv_also_saves_the_delivery_into_the_projects_exports_dir(
    client: TestClient, tmp_path: Path,
) -> None:
    """The browser download is the breeder's copy; the delivery itself belongs to the project,
    so the identical bytes land in <project>/results_export/ and the write is audited."""
    body = _phenology_fixture(tmp_path, validated=True)
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "catkin_delivery.csv"})
    assert resp.status_code == 200
    saved = tmp_path / "results_export" / "catkin_delivery.csv"
    assert resp.headers["X-TCIP-Saved-To"] == str(saved)
    assert saved.read_bytes() == resp.content
    audit = (tmp_path / ".tcip" / "audit.jsonl").read_text(encoding="utf-8")
    assert "results.export_csv" in audit
    assert "catkin_delivery.csv" in audit


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
                  "positive_state_classifier_validated", "producer_model_sha256",
                  "producer_experiment_id"]
    body = _phenology_fixture(tmp_path, validated=True)
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "curves", "filename": "c.csv"})
    assert resp.status_code == 200
    header = resp.text.splitlines()[0].split(",")
    assert header[-len(provenance):] == provenance
    cells = dict(zip(header, resp.text.splitlines()[1].split(",")))
    for col in provenance:
        assert cells[col] != "", col
    assert cells["operating_point_validated"] == "held_out_annotations"
    assert cells["producer_experiment_id"] == "exp-1"


def test_export_refuses_a_bucket_whose_id_map_never_carried_the_positive_class(
    client: TestClient, tmp_path: Path,
) -> None:
    # compute_phenology's first guard is that the positive class id resolves from some bucket's own
    # recorded id_map. per_plant_phenology's positive_class_assessed flag is not a substitute: a date
    # with ZERO detections is trivially "fully classified", so an axis-less bucket with no
    # detections read as classified and the export door had nothing left to refuse on.
    body = _phenology_fixture(tmp_path, validated=True, id_map={"catkin": 0}, detections=0)
    curves = client.post("/api/results/per_plant_curves", json=body)
    assert curves.status_code == 200
    assert curves.json()["positive_class_assessed"] is False
    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})
    assert resp.status_code == 400
    assert "elongated" in resp.json()["detail"]


def test_registered_models_confines_project_path_to_allowed_roots(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("TCIP_IMAGE_ROOTS", str(allowed))
    outside = tmp_path / "outside"
    outside.mkdir()
    resp = client.get("/api/results/models/registered", params={"project_path": str(outside)})
    assert resp.status_code == 403


def test_registered_models_unconfined_when_no_image_roots(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    resp = client.get("/api/results/models/registered", params={"project_path": str(tmp_path)})
    assert resp.status_code == 200
    assert resp.json()["models"] == []


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

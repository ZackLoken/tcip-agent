"""Slice 3 tests: plant mapping + per-plant curves + onset dates + CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _write_preds(path: Path, subjects: list[str]) -> None:
    """Per-image JSON prediction file with the given classified subjects (K4/K5: the classifier's
    decoded call, straight into ``.subject``)."""
    anns = [Annotation(subject=s, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.9) for s in subjects]
    write_annotations(str(path), anns, 8, 8)


def _write_id_map_sidecar(dir_path: Path, id_map: dict) -> None:
    (dir_path / "operating_point.json").write_text(json.dumps({"id_map": id_map}), encoding="utf-8")


_ID_MAP = {"dormant": 0, "elongated": 1}


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


def test_per_plant_curves_uses_mapping_and_counts(client: TestClient, tmp_path: Path) -> None:
    # Fabricate a mapping + predictions for two dates, two plants
    mapping_path = tmp_path / "mapping.json"
    mapping_data = {
        "2026-02-11": [
            {
                "image_path": "/x/IMG_A.JPG",
                "stem": "IMG_A",
                "date_folder": "2026-02-11",
                "plot_name": "PLANT_A",
                "accession_name": "AccA",
                "confidence": 0.9,
                "source": "sequence",
                "distance_m": 1.0,
            },
            {
                "image_path": "/x/IMG_B.JPG",
                "stem": "IMG_B",
                "date_folder": "2026-02-11",
                "plot_name": "PLANT_B",
                "accession_name": "AccB",
                "confidence": 0.9,
                "source": "sequence",
                "distance_m": 1.0,
            },
        ],
        "2026-03-24": [
            {
                "image_path": "/x/IMG_A2.JPG",
                "stem": "IMG_A2",
                "date_folder": "2026-03-24",
                "plot_name": "PLANT_A",
                "accession_name": "AccA",
                "confidence": 0.9,
                "source": "sequence",
                "distance_m": 1.0,
            },
        ],
    }
    mapping_path.write_text(json.dumps(mapping_data), encoding="utf-8")

    preds_211 = tmp_path / "preds_2-11-26"
    preds_324 = tmp_path / "preds_3-24-26"
    preds_211.mkdir()
    preds_324.mkdir()
    # K4/K5: classified predictions (real elongated/dormant subjects), each bucket carrying its own
    # recorded id_map so the coverage rule admits them.
    _write_preds(preds_211 / "IMG_A.json", ["elongated"] * 3 + ["dormant"])   # PLANT_A: 3/4
    _write_preds(preds_211 / "IMG_B.json", ["dormant", "dormant"])            # PLANT_B: 0/2
    _write_preds(preds_324 / "IMG_A2.json", ["elongated"] * 3)                # PLANT_A: 3/3
    _write_id_map_sidecar(preds_211, _ID_MAP)
    _write_id_map_sidecar(preds_324, _ID_MAP)

    resp = client.post(
        "/api/results/per_plant_curves",
        json={
            "project_root": str(tmp_path),
            "mapping_path": str(mapping_path),
            "predictions_by_date": {
                "2026-02-11": str(preds_211),
                "2026-03-24": str(preds_324),
            },
            "trait": "catkin",
        },
    )
    body = resp.json()
    assert body["n_plants"] == 2
    assert body["elongation_classified"] is True

    by_key = {(r["plant_id"], r["date"]): r for r in body["rows"]}
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.75
    assert by_key[("PLANT_B", "2026-02-11")]["n_total"] == 2
    assert by_key[("PLANT_B", "2026-02-11")]["ratio"] == 0.0
    assert by_key[("PLANT_A", "2026-03-24")]["n_total"] == 3
    assert by_key[("PLANT_A", "2026-03-24")]["ratio"] == 1.0


def test_per_plant_curves_flags_unclassified_predictions(client: TestClient, tmp_path: Path) -> None:
    # Single-class detector output (no elongation classification) must not be passed off as
    # a bloom measurement — the endpoint reports elongation_classified=False.
    mapping_path = tmp_path / "m.json"
    mapping_path.write_text(
        json.dumps(
            {
                "2026-02-11": [
                    {
                        "image_path": "/x/IMG_A.JPG",
                        "stem": "IMG_A",
                        "date_folder": "2026-02-11",
                        "plot_name": "PLANT_A",
                        "accession_name": "AccA",
                        "confidence": 0.9,
                        "source": "sequence",
                        "distance_m": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    preds = tmp_path / "preds"
    preds.mkdir()
    # Bare single-class detector output — id_map has no attribute axis at all (the round-3/4/5
    # canonical case). Must refuse, never report full coverage.
    _write_preds(preds / "IMG_A.json", ["catkin", "catkin"])
    _write_id_map_sidecar(preds, {"catkin": 0})

    body = client.post(
        "/api/results/per_plant_curves",
        json={
            "project_root": str(tmp_path),
            "mapping_path": str(mapping_path),
            "predictions_by_date": {"2026-02-11": str(preds)},
            "trait": "catkin",
        },
    ).json()
    assert body["elongation_classified"] is False
    # Isolate the real mechanism, not just the top-level flag (stage-6 review: an unwired stub
    # that always returns elongation_classified=False regardless of input would pass the assertion
    # above too). The row must show the 2 real detections counted, entirely unclassified — not a
    # stub's always-empty (0 total, 0 positive) shape.
    row = body["rows"][0]
    assert row["n_total"] == 2
    assert row["n_unclassified"] == 2
    assert row["ratio"] is None


def test_onset_dates_finds_crossings(client: TestClient) -> None:
    curves = [
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-02-11",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 0,
            "ratio": 0.0,
            "n_unclassified": 0,
            "n_missing": 0,
        },
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-03-02",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 10,
            "ratio": 0.10,
            "n_unclassified": 0,
            "n_missing": 0,
        },
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-03-09",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 60,
            "ratio": 0.60,
            "n_unclassified": 0,
            "n_missing": 0,
        },
        {
            "plant_id": "PLANT_A",
            "accession": "A",
            "date": "2026-03-18",
            "n_images": 1,
            "n_total": 100,
            "n_elongated": 100,
            "ratio": 1.0,
            "n_unclassified": 0,
            "n_missing": 0,
        },
    ]
    resp = client.post("/api/results/onset_dates", json={"curves": curves, "trait": "catkin"})
    rows = resp.json()["rows"]
    assert len(rows) == 1
    onset = rows[0]
    assert onset["catkin_05per_date"] is not None  # reached between 2-11 and 3-2
    assert onset["catkin_50per_date"] is not None  # reached between 3-2 and 3-9
    assert onset["catkin_95per_date"] is not None  # reached between 3-9 and 3-18
    # catkin_elongation_date = "most catkins elongated" (crops.yml) = the 95% majority crossing,
    # synonymous with catkin_95per_date.
    assert onset["catkin_elongation_date"] == onset["catkin_95per_date"]


def test_onset_dates_ignores_undated_bucket(client: TestClient) -> None:
    # The ingest 'undated/' bucket (and any non-ISO folder) sorts to the (0,0,0) sentinel.
    # It must not crash interpolation (date(0,0,0)) or leak '0000-00-00' into the CSV.
    curves = [
        {"plant_id": "P", "accession": "A", "date": "undated", "ratio": 0.9, "n_unclassified": 0, "n_missing": 0},
        {"plant_id": "P", "accession": "A", "date": "2026-02-11", "ratio": 0.0, "n_unclassified": 0, "n_missing": 0},
        {"plant_id": "P", "accession": "A", "date": "2026-03-09", "ratio": 1.0, "n_unclassified": 0, "n_missing": 0},
    ]
    resp = client.post("/api/results/onset_dates", json={"curves": curves, "trait": "catkin"})
    assert resp.status_code == 200
    onset = resp.json()["rows"][0]
    for key in ("catkin_05per_date", "catkin_50per_date", "catkin_95per_date"):
        assert onset[key] != "0000-00-00"
        if onset[key] is not None:
            assert onset[key].startswith("2026-")


def test_onset_dates_rejects_row_missing_required_field(client: TestClient) -> None:
    # A malformed row (missing plant_id) must be a structured 422, not an unhandled
    # KeyError/500 — onset_dates dereferences plant_id/date/ratio directly.
    curves = [{"accession": "A", "date": "2026-02-11", "ratio": 0.0}]
    resp = client.post("/api/results/onset_dates", json={"curves": curves, "trait": "catkin"})
    assert resp.status_code == 422


def test_onset_dates_rejects_row_missing_coverage_disclosure_fields(client: TestClient) -> None:
    # Stage-6 review Finding E: n_unclassified/n_missing are required, not `= 0` defaults — a
    # payload that omits them must be a structured 422, never silently read as "fully classified"
    # (the plant_fully_classified predicate dereferences them directly).
    curves = [{"plant_id": "P", "date": "2026-02-11", "ratio": 0.0}]
    resp = client.post("/api/results/onset_dates", json={"curves": curves, "trait": "catkin"})
    assert resp.status_code == 422


def test_onset_dates_discloses_zero_observations_distinctly_from_valid(client: TestClient) -> None:
    # Stage-6 review N6: a plant fully classified AND fully observed (0 unclassified, 0 missing)
    # can still have zero real detections on every date (e.g. before emergence) — n_observed_dates
    # lets the GUI distinguish "no observations" from real bloom data, both of which otherwise read
    # identically as "valid" next to blank milestone cells.
    curves = [
        {"plant_id": "P", "date": "2026-02-11", "ratio": None, "n_total": 0,
         "n_unclassified": 0, "n_missing": 0},
        {"plant_id": "P", "date": "2026-02-18", "ratio": None, "n_total": 0,
         "n_unclassified": 0, "n_missing": 0},
    ]
    resp = client.post("/api/results/onset_dates", json={"curves": curves, "trait": "catkin"})
    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert row["n_observed_dates"] == 0
    assert row["catkin_95per_date"] is None


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


def test_export_csv_row_stamp_alone_is_never_trusted(client: TestClient) -> None:
    # K3: both dimensions reconcile from the buckets' own sidecars, never a row-asserted string
    # alone. The original version of this test only asserted 400 with no predictions_by_date given
    # — vacuous, since test_export_csv_refuses_unvalidated_phenology already covers "no evidence ->
    # refuse" for a row with NO stamp at all, and a row's stamp claim was never read on this path
    # either way. What actually needs proving is that the stamp is INERT: two rows differing only in
    # what they optimistically claim must produce the identical refusal (same status, same reason),
    # since the code never looks at the claim to decide.
    optimistic = {"positive_state_classifier_validated": "validated_held_out",
                 "operating_point_validated": "validated_held_out"}
    pessimistic = {"positive_state_classifier_validated": "false", "operating_point_validated": "false"}
    resp_optimistic = client.post("/api/results/export_csv", json={
        "rows": [{"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15", **optimistic}],
        "filename": "test.csv", "export_kind": "phenology"})
    resp_pessimistic = client.post("/api/results/export_csv", json={
        "rows": [{"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15", **pessimistic}],
        "filename": "test.csv", "export_kind": "phenology"})
    assert resp_optimistic.status_code == resp_pessimistic.status_code == 400
    assert resp_optimistic.json()["detail"] == resp_pessimistic.json()["detail"]
    assert "predictions_by_date" in resp_optimistic.json()["detail"]


def test_export_csv_row_stamp_cannot_override_a_genuinely_unvalidated_classifier(
    client: TestClient, tmp_path: Path,
) -> None:
    # The actual pre-K3 vulnerability, reproduced directly: the row claims a validated classifier
    # via the OLD, pre-rename field name (elongation_classifier_validated) that the code used to
    # read straight off the row with no reconciliation at all — while the real, on-disk classifier
    # sidecar is genuinely unvalidated, and the count dimension is genuinely validated (so THAT
    # can't be why this refuses). Before K3, this combination shipped a 200 (the row's optimistic
    # old-field claim was trusted outright); after K3, the classifier dimension is reconciled from
    # classifier_operating_point.json regardless of what the row claims, under any field name.
    d = tmp_path / "preds"
    d.mkdir()
    (d / "operating_point.json").write_text(json.dumps({
        "validated": True,
        "operating_point": {"conf": {"value": 0.4, "validated_vs_gt": "validated_held_out"}},
        "id_map": _ID_MAP,
    }), encoding="utf-8")
    (d / "classifier_operating_point.json").write_text(json.dumps({
        "validated": False,
        "operating_point": {"classifier": {"value": "elongated", "validated_vs_gt": "false"}},
    }), encoding="utf-8")
    stamp = {"elongation_classifier_validated": "validated_held_out",
             "operating_point_validated": "validated_held_out"}
    rows = [{"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15", **stamp}]
    resp = client.post(
        "/api/results/export_csv",
        json={"rows": rows, "filename": "test.csv", "export_kind": "phenology",
              "predictions_by_date": {"2026-02-11": str(d)}},
    )
    assert resp.status_code == 400
    # Specifically the classifier dimension refused, not the (genuinely validated) count one.
    assert "Unvalidated: ['classifier']" in resp.json()["detail"]


def test_export_csv_delivers_with_real_bucket_evidence(client: TestClient, tmp_path: Path) -> None:
    d = tmp_path / "preds"
    d.mkdir()
    (d / "operating_point.json").write_text(json.dumps({
        "validated": True,
        "operating_point": {"conf": {"value": 0.4, "validated_vs_gt": "validated_held_out"}},
        "id_map": _ID_MAP,
    }), encoding="utf-8")
    (d / "classifier_operating_point.json").write_text(json.dumps({
        "validated": True,
        "operating_point": {"classifier": {"value": "elongated", "validated_vs_gt": "validated_held_out"}},
    }), encoding="utf-8")
    rows = [
        {"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15"},
        {"plant_id": "PLANT_B", "catkin_05per_date": "2026-03-05"},
    ]
    resp = client.post(
        "/api/results/export_csv",
        json={"rows": rows, "filename": "test.csv", "export_kind": "phenology",
              "predictions_by_date": {"2026-02-11": str(d)}},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    lines = body.strip().split("\n")
    assert len(lines) == 3
    assert "plant_id" in lines[0]
    assert "PLANT_A" in lines[1]
    assert "PLANT_B" in lines[2]


def test_export_csv_refuses_unvalidated_phenology(client: TestClient) -> None:
    # Milestone rows without the held-out-validated stamp must not ship (the second-door hole).
    rows = [{"plant_id": "PLANT_A", "catkin_05per_date": "2026-02-15"}]
    resp = client.post("/api/results/export_csv",
                       json={"rows": rows, "filename": "x.csv", "export_kind": "phenology"})
    assert resp.status_code == 400


def test_export_csv_non_phenology_unaffected(client: TestClient) -> None:
    # A diagnostic / inventory export (no trait_name + value) is not a phenotype delivery — not gated.
    rows = [{"plant_id": "PLANT_A", "some_metric": 42}]
    resp = client.post("/api/results/export_csv",
                       json={"rows": rows, "filename": "x.csv", "export_kind": "diagnostic"})
    assert resp.status_code == 200


def test_export_csv_requires_an_explicit_export_kind(client: TestClient) -> None:
    # Round-4 review: three successive attempts to INFER whether an export was a phenology delivery
    # from its row keys were each defeated, in both directions. The caller now declares it, and an
    # omitted declaration fails closed (422) rather than defaulting to the ungated branch.
    rows = [{"plant_id": "PLANT_A", "date": "2026-02-11", "ratio": 0.7}]
    resp = client.post("/api/results/export_csv", json={"rows": rows, "filename": "x.csv"})
    assert resp.status_code == 422


def test_export_csv_declared_phenology_is_gated_whatever_the_row_shape(client: TestClient) -> None:
    """Round-4 review: every real curve shape that defeated the previous key-sniffing rules must now
    be refused without evidence. Each of these is a genuine per-plant-per-date bloom curve; under the
    round-4 identity-pair rule the last three shipped 200 merely by naming their columns differently
    — `plot_name` being the platform's OWN name for the plant (per_plant_series reads it to build
    plant_id), so this was not an exotic payload."""
    shapes = [
        {"plant_id": "PLANT_A", "date": "2026-02-11", "ratio": 0.7, "n_total": 10, "n_positive": 7},
        {"accession": "acc-9", "date": "2026-02-11", "ratio": 0.7, "n_total": 10},
        {"plot_name": "PLANT_A", "date": "2026-02-11", "ratio": 0.7, "n_total": 10},
        {"plant_id": "PLANT_A", "observation_date": "2026-02-11", "ratio": 0.7},
    ]
    for row in shapes:
        resp = client.post("/api/results/export_csv",
                           json={"rows": [row], "filename": "x.csv", "export_kind": "phenology"})
        assert resp.status_code == 400, row


def test_export_csv_declared_diagnostic_still_cannot_ship_milestone_rows(client: TestClient) -> None:
    # The declaration is not a bypass: milestone column names come from the trait registry, not the
    # payload, so they remain a structural floor beneath whatever the caller declares.
    rows = [{"plant_id": "PLANT_A", "catkin_95per_date": "2026-03-10"}]
    resp = client.post("/api/results/export_csv",
                       json={"rows": rows, "filename": "x.csv", "export_kind": "diagnostic"})
    assert resp.status_code == 400


def test_export_csv_declared_diagnostic_per_plant_date_tables_still_ship(client: TestClient) -> None:
    """CLAUDE.md 'a rail must admit valid work'. Round 4's identity-pair rule 400'd every one of
    these legitimate non-phenology tables — any table keyed by plant and date carrying one generic
    numeric column collided with it. They are diagnostics, and they ship."""
    tables = [
        {"plant_id": "PLANT_A", "date": "2026-02-11", "n_total": 42},          # per-date inventory
        {"plant_id": "PLANT_A", "date": "2026-02-11", "ratio": 0.7},           # a QC ratio
        {"plant_id": "PLANT_A", "date": "2026-02-11", "n_missing": 3},         # coverage audit
        {"class": "catkin", "n_positive": 7, "n_negative": 3},                 # class balance
        {"plant_id": "PLANT_A", "split": "train", "ratio": 0.7},               # split table
    ]
    for row in tables:
        resp = client.post("/api/results/export_csv",
                           json={"rows": [row], "filename": "x.csv", "export_kind": "diagnostic"})
        assert resp.status_code == 200, row


def test_export_csv_refuses_unvalidated_per_plant_phenotype(client: TestClient) -> None:
    # A per-plant phenotype row (trait_name + value) with no validation stamp must not ship (R4:
    # the non-phenology delivery door is gated too).
    rows = [{"plant_id": "PLANT_A", "trait_name": "catkin_count", "value": 7}]
    resp = client.post("/api/results/export_csv",
                       json={"rows": rows, "filename": "x.csv", "export_kind": "diagnostic"})
    assert resp.status_code == 400


def test_export_csv_ships_validated_per_plant_phenotype(client: TestClient) -> None:
    # The same per-plant phenotype row ships once it carries a shippable measurement reference.
    rows = [{"plant_id": "PLANT_A", "trait_name": "catkin_count", "value": 7,
             "measurement_validated": "validated_held_out"}]
    resp = client.post("/api/results/export_csv",
                       json={"rows": rows, "filename": "x.csv", "export_kind": "diagnostic"})
    assert resp.status_code == 200


def test_export_csv_empty_rejected(client: TestClient) -> None:
    resp = client.post("/api/results/export_csv",
                       json={"rows": [], "filename": "x.csv", "export_kind": "diagnostic"})
    assert resp.status_code == 400


def test_inference_launch_missing_checkpoint(client: TestClient, tmp_path: Path) -> None:
    resp = client.post(
        "/api/inference/launch",
        json={
            "checkpoint_path": str(tmp_path / "no.pt"),
            "images_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
        },
    )
    assert resp.status_code == 404


def test_inference_list_jobs_endpoint(client: TestClient) -> None:
    resp = client.get("/api/inference/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()

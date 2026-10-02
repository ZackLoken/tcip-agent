"""The Results routes: plant mapping, the phenology measurement, the CSV exports and the
delivery-event listing."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.audit import audit_log_key
from tcip_web.app import app

from tests import _trait_fixtures as fx
from tests._audit_fixtures import refuse_audit_appends
from tests._web_fixtures import BROWSER, acknowledged_post, open_new_project


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(StringIO(text)))


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_list_traits_names_an_unreadable_record_alongside_the_valid_one(
    client: TestClient, opened_project: Path
) -> None:
    """A record its schema refuses is visible by name and reason, never silently absent the way a
    dropped trait looks like none."""
    from tcip_mcp import traits

    valid = traits.read_trait("bud_opening", opened_project).model_dump(mode="json")
    valid["revisions"][0]["entry"]["localization"] = "unicorn_match"
    tcip_store.replace(traits.trait_key(opened_project, "unicorn"), valid,
                       expect=tcip_store.Version.ABSENT)

    body = client.get("/api/results/traits").json()

    assert [record["trait"] for record in body["traits"]] == ["bud_opening"]
    assert [entry["trait"] for entry in body["unreadable"]] == ["unicorn"]
    assert "localization" in body["unreadable"][0]["reason"]


def test_plant_mapping_build_requires_a_registered_dataset(
    client: TestClient, opened_project: Path,
) -> None:
    """A directory that is not a registered dataset's own images/ root refuses, naming that,
    rather than silently building an empty mapping over nothing."""
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={"name": "valley", "images_root": str(opened_project / "nope"),
              "plant_registry": "unregistered"},
    )
    assert resp.status_code == 400
    assert "is not a dataset" in resp.json()["detail"]


def test_plant_mapping_load_of_an_unstored_name_answers_404_naming_it(
    client: TestClient, opened_project: Path,
) -> None:
    resp = client.post("/api/results/plant_mapping/load", json={"name": "missing"})
    assert resp.status_code == 404
    assert "'missing'" in resp.json()["detail"]


# ── The phenology measurement and its export ────────────────────────────


def _series(tmp_path: Path, **kwargs):
    pytest.importorskip("torch")
    from tests._chain_fixtures import classified_series

    return classified_series(tmp_path, **kwargs)


def _export(client: TestClient, body: dict, payload: str = "milestones", headers=None, **extra):
    return client.post("/api/results/export_csv", headers=headers,
                       json={**body, "payload": payload, "filename": "x.csv", **extra})


def test_an_assessed_series_measures_its_curves_and_milestones_validated(
    client: TestClient, tmp_path: Path,
) -> None:
    """Each (plant, date) curve row counts that date's detections, and the milestone columns the
    response names are each a date inside the series."""
    series = _series(tmp_path)

    resp = client.post("/api/results/phenology_measurement", json=series.body())

    assert resp.status_code == 200, resp.text[:300]
    out = resp.json()
    assert out["validated"] is True and out["unvalidated_reason"] is None
    assert out["positive_class_assessed"] is True
    assert out["curves"]["n_plants"] == 2
    by_key = {(r["plant_id"], r["date"]): r for r in out["curves"]["rows"]}
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.0
    assert by_key[("PLANT_B", "2026-03-10")]["ratio"] == 0.75
    plant_a = next(r for r in out["milestones"]["rows"] if r["plant_id"] == "PLANT_A")
    assert out["milestones"]["columns"]
    for column in out["milestones"]["columns"]:
        assert plant_a[column["date"]].startswith("2026-"), column
        assert plant_a[column["bound"]], column


def test_the_curve_counts_every_image_of_a_plant_on_a_date(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _series(tmp_path, fractions=(0.5,), images_per_plant=3).body()

    rows = client.post("/api/results/phenology_measurement", json=body).json()["curves"]["rows"]

    row = next(r for r in rows if r["plant_id"] == "PLANT_A")
    assert row["n_images"] == 3
    assert row["n_total"] == 12


def test_an_unassessed_series_refuses_until_shown_unvalidated_and_never_exports_on_its_own(
    client: TestClient, tmp_path: Path,
) -> None:
    """A breeder can look at provisional numbers (``show_unvalidated``, a display choice); only
    an acknowledgment opens the export."""
    body = _series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    refused = client.post("/api/results/phenology_measurement", json=body)
    shown = client.post("/api/results/phenology_measurement",
                        json={**body, "show_unvalidated": True})

    assert refused.status_code == 400
    assert refused.json()["detail"]["message"] == shown.json()["unvalidated_reason"]
    assert shown.status_code == 200
    assert shown.json()["validated"] is False
    assert "no assessment answers" in shown.json()["unvalidated_reason"]
    assert shown.json()["curves"]["rows"]
    shown_digests = shown.json()["result_sha256"]
    for payload in ("curves", "milestones"):
        export = _export(client, body, payload)
        assert export.status_code == 400, payload
        # The screen and the export name one prepared result: the digest an acknowledgment of
        # what was shown binds to is the one the export would write.
        assert export.json()["detail"]["result_sha256"] == shown_digests[payload], payload
    assert shown_digests["curves"] != shown_digests["milestones"]


def test_an_acknowledged_export_ships_unvalidated_and_its_event_names_the_act(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    resp = acknowledged_post(client, "/api/results/export_csv",
                             {**body, "payload": "milestones", "filename": "x.csv"},
                             reason="a look before assessment")

    assert resp.status_code == 200, resp.text[:300]
    assert {row["validated"] for row in _rows(resp.text)} == {"False"}
    events = client.get("/api/results/delivery-events").json()["records"]
    (event,) = [r for r in events if r["door"] == "results.export_csv"]
    assert (event["acknowledgment"]["acknowledged_by"], event["acknowledgment"]["reason"]) == (
        "user:breeder", "a look before assessment")
    assert event["validated"] is False


def test_an_acknowledgment_is_recorded_only_from_the_browser_and_with_a_reason(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request from no browser, or one declaring an agent identity, cannot originate the
    breeder's act, and a reason of spaces says nothing, nor a name of spaces who; each refuses
    before anything runs, and the same acknowledgment from the browser ships, signed by the
    person it names."""
    from tcip_mcp import agent_identity

    monkeypatch.setenv("TCIP_USER", "osuser")
    body = _series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    digest = _export(client, body).json()["detail"]["result_sha256"]
    acknowledgment = {"user": "breeder", "reason": "needs a look", "result_sha256": digest}
    bare = _export(client, body, acknowledgment=acknowledgment)
    agent = _export(client, body, acknowledgment=acknowledgment, headers={
        **BROWSER, agent_identity.HEADERS["agent_session"]: "mcp_0123"})
    blank = _export(client, body, acknowledgment={**acknowledgment, "reason": "   "},
                    headers=BROWSER)
    nameless = _export(client, body, acknowledgment={**acknowledgment, "user": "  "},
                       headers=BROWSER)

    assert (bare.status_code, agent.status_code) == (403, 403)
    assert "cannot record one" in bare.json()["detail"]
    assert blank.status_code == 400
    assert "an acknowledgment states why" in blank.json()["detail"]
    assert nameless.status_code == 400 and "names no one" in nameless.json()["detail"]
    assert client.get("/api/results/delivery-events").json()["records"] == []
    assert _export(client, body, acknowledgment=acknowledgment,
                   headers=BROWSER).status_code == 200


def test_an_export_saves_the_delivery_under_the_project_and_logs_it_with_its_dataset(
    client: TestClient, tmp_path: Path,
) -> None:
    """The browser download is the breeder's copy; the delivery belongs to the project, so the
    identical bytes land in ``results_export/``, in the trait's own schema, and the delivery's
    line lands in the log of the dataset its buckets sit in."""
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.postprocessing import phenology

    series = _series(tmp_path, fractions=(0.0, 1.0))

    resp = client.post("/api/results/export_csv", json={
        **series.body(), "payload": "milestones", "filename": "../bud_delivery.csv"})

    assert resp.status_code == 200, resp.text[:300]
    saved = tmp_path / "results_export" / "bud_delivery.csv"
    assert resp.headers["X-TCIP-Saved-To"] == str(saved)
    assert saved.read_bytes() == resp.content
    header = resp.text.splitlines()[0].split(",")
    assert header == phenology.phenology_csv_columns(
        latest_confirmed("bud_opening", tmp_path).entry)
    (record,) = read_delivery_events(tmp_path)
    assert {row["delivery_event_id"] for row in _rows(resp.text)} == {record.event_id}
    assert {row["validated"] for row in _rows(resp.text)} == {"True"}
    logged = [e for e in tcip_store.read_log(audit_log_key(series.root)).records
              if e["tool"] == "delivery_event"]
    assert [e["arguments"] for e in logged] == [{"event_id": record.event_id}]


def test_the_curves_export_writes_the_curve_schema(client: TestClient, tmp_path: Path) -> None:
    from tcip_mcp.pipelines.postprocessing import phenology

    body = _series(tmp_path, fractions=(0.0, 1.0)).body()

    resp = _export(client, body, "curves")

    assert resp.status_code == 200, resp.text[:300]
    assert resp.text.splitlines()[0].split(",") == phenology.curve_csv_columns()
    assert len(_rows(resp.text)) == 4


def test_a_walked_mapping_event_names_the_mapping_key_it_resolves_to(
    client: TestClient, tmp_path: Path,
) -> None:
    body = _series(tmp_path, fractions=(0.0, 1.0)).body()
    assert _export(client, body).status_code == 200

    (event,) = client.get("/api/results/delivery-events").json()["records"]

    assert event["plant_mapping"]["name"] == "valley"
    assert event["plant_mapping_resolved_key"] == "valley"


def test_export_csv_answers_409_when_the_delivery_event_audit_line_cannot_be_appended(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CSV is already on disk by the time the delivery's audit line is appended; a failed
    append answers 409 naming the file written, which is left in place."""
    body = _series(tmp_path, fractions=(0.0, 1.0)).body()
    refuse_audit_appends(monkeypatch)

    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "unaudited.csv"})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    saved = tmp_path / "results_export" / "unaudited.csv"
    assert detail["error"] == "audit_entry_not_written"
    assert detail["committed"] == {"saved_path": str(saved)}
    assert saved.exists()


def test_phenology_doors_reject_a_malformed_payload_with_422_not_500(client: TestClient) -> None:
    for route in ("phenology_measurement", "export_csv"):
        resp = client.post(f"/api/results/{route}", json={"trait": "bud_opening"})
        assert resp.status_code == 422, route


def test_caller_composed_rows_and_validity_claims_are_refused_at_the_payload(
    client: TestClient,
) -> None:
    """These doors compute what they deliver, so neither caller-composed rows nor a caller's
    claim about validity is a field either takes."""
    inputs = {"mapping_name": "valley", "trait": "bud_opening",
              "buckets": ["x"], "plants": ["PLANT_A"]}
    for rows in (
        [{"plant_id": "P1", "date": "2026-03-01", "n_total": 20, "ratio": 0.05}],
        [{"plant_id": "P1", "start_date": "2026-03-01", "end_date": "2026-04-02"}],
    ):
        resp = client.post("/api/results/export_csv", json={
            "rows": rows, "filename": "x.csv", "export_kind": "diagnostic"})
        assert resp.status_code == 422
    for field, value in (("validated", True), ("unvalidated_reason", None)):
        resp = client.post("/api/results/phenology_measurement", json={**inputs, field: value})
        assert resp.status_code == 422, field


def test_phenology_measurement_refuses_when_the_delivered_dataset_carries_no_registry(
    client: TestClient, tmp_path: Path,
) -> None:
    """The delivered buckets resolve to a dataset root that carries no registry; the door refuses
    by name rather than checking the project root's own."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    open_new_project(tmp_path)
    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "live" / "2026-02-11",
                       [predicted("PLANT_A", ["bud"], {"bud": 0})],
                       scope={"subject": "bud", "attribute": None, "id_map": {"bud": 0}}).path

    resp = client.post("/api/results/phenology_measurement", json={
        "mapping_name": "valley",
        "buckets": [str(bucket)], "trait": "bud_opening",
        "plants": ["PLANT_A"],
    })

    assert resp.status_code == 400
    assert "no subject registry is reachable" in resp.json()["detail"]


# ── Count CSV export: per-image kind ──────────────────────────────────────

COUNT_SCOPE = {"subject": fx.COUNT_SUBJECT, "attribute": None, "id_map": {fx.COUNT_SUBJECT: 0}}


def _unassessed_count_bucket(project: Path, *, scope: dict = COUNT_SCOPE) -> Path:
    """Two frames of three detections each, published under no assessment, in a project whose
    count trait is confirmed."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_count(project, measured_subject=fx.COUNT_SUBJECT)
    subject = scope["subject"]
    bucket = published(project, project / "ds" / "predictions" / "live" / "counts",
                       [predicted(f"img{i}", [subject] * 3, scope["id_map"]) for i in range(2)],
                       scope=scope).path
    open_new_project(project)
    return bucket


def _per_image(bucket: Path, trait: str = fx.COUNT_TRAIT, **extra) -> dict:
    return {"delivery": {"kind": "per_image_count", "predictions_dir": str(bucket),
                         "trait": trait}, "filename": "counts.csv", **extra}


COUNT_ROUTE = "/api/results/export_count_csv"


def _export_count(client: TestClient, body: dict, headers=None):
    return client.post(COUNT_ROUTE, json=body, headers=headers)


def _digest(client: TestClient, body: dict) -> str:
    """The result digest the count export's refusal of ``body`` names."""
    return _export_count(client, body).json()["detail"]["result_sha256"]


def test_an_unassessed_count_bucket_refuses_with_no_acknowledgment(
    client: TestClient, tmp_path: Path,
) -> None:
    resp = _export_count(client, _per_image(_unassessed_count_bucket(tmp_path)))

    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "delivery"
    assert "no assessment answers" in resp.json()["detail"]["message"]
    assert not (tmp_path / "results_export" / "counts.csv").exists()


def test_an_unassessed_count_bucket_delivers_under_acknowledgment(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    resp = acknowledged_post(client, COUNT_ROUTE, _per_image(bucket),
                             reason="a look before assessment")

    assert resp.status_code == 200, resp.text[:300]
    assert resp.headers["X-TCIP-Validated"] == "false"
    assert unquote(resp.headers["X-TCIP-Acknowledged-By"]) == "user:breeder"
    rows = _rows(resp.text)
    assert [r["detection_count"] for r in rows] == ["3", "3"]
    assert {r["validated"] for r in rows} == {"False"}
    (event,) = client.get("/api/results/delivery-events").json()["records"]
    assert event["door"] == "results.export_count_csv"
    assert event["acknowledgment"]["acknowledged_by"] == "user:breeder"


def test_an_assessed_count_bucket_delivers_validated_and_discards_an_acknowledgment(
    client: TestClient, tmp_path: Path,
) -> None:
    """A validated delivery's event carries no acknowledgment, even when one was posted beside
    it: it cleared nothing."""
    pytest.importorskip("torch")
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-web-count")
    open_new_project(tmp_path)

    bare = _export_count(client, _per_image(chain.bucket))
    posted = _export_count(client, _per_image(
        chain.bucket, acknowledgment={"user": "breeder", "reason": "just in case",
                                      "result_sha256": "0" * 64}),
        headers=BROWSER)

    assert bare.status_code == 200, bare.text[:300]
    assert bare.headers["X-TCIP-Validated"] == "true"
    assert bare.headers["X-TCIP-Acknowledged-By"] == ""
    assert {r["validated"] for r in _rows(bare.text)} == {"True"}
    assert posted.status_code == 200
    assert posted.headers["X-TCIP-Acknowledged-By"] == ""
    events = client.get("/api/results/delivery-events").json()["records"]
    assert [e["acknowledgment"] for e in events] == [None, None]


def test_export_count_csv_answers_409_when_the_delivery_event_audit_line_cannot_be_appended(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)
    digest = _digest(client, _per_image(bucket))
    refuse_audit_appends(monkeypatch, landing=1)  # the acknowledgment's own line lands

    resp = _export_count(client, _per_image(
        bucket, acknowledgment={"user": "breeder", "reason": "a look", "result_sha256": digest}),
        headers=BROWSER)

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    saved = tmp_path / "results_export" / "counts.csv"
    assert detail["error"] == "audit_entry_not_written"
    assert detail["committed"] == {"saved_path": str(saved)}
    assert saved.exists()


def test_export_count_csv_refuses_an_acknowledgment_from_no_browser_or_with_a_blank_reason(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    digest = _digest(client, _per_image(bucket))
    bare = _export_count(client, _per_image(
        bucket, acknowledgment={"user": "breeder", "reason": "no browser",
                                "result_sha256": digest}))
    blank = _export_count(client, _per_image(
        bucket, acknowledgment={"user": "breeder", "reason": "   ", "result_sha256": digest}),
        headers=BROWSER)

    assert bare.status_code == 403 and "cannot record one" in bare.json()["detail"]
    assert blank.status_code == 400 and "reason" in blank.json()["detail"]
    assert "an acknowledgment states why" in blank.json()["detail"]


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_export_count_csv_refuses_a_trait_that_states_no_per_image_count(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    resp = _export_count(client, _per_image(bucket, trait="bud_opening"))

    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "operationalization"


def test_export_count_csv_refuses_a_bucket_whose_scope_omits_the_confirmed_subject(
    client: TestClient, tmp_path: Path,
) -> None:
    other = {"subject": "other", "attribute": None, "id_map": {"other": 0}}
    bucket = _unassessed_count_bucket(tmp_path, scope=other)

    resp = _export_count(client, _per_image(bucket))

    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "operationalization"


def test_export_count_csv_refuses_an_unknown_trait_with_400_not_500(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    resp = _export_count(client, _per_image(bucket, trait="no-such-trait"))

    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "delivery"


def test_export_count_csv_refuses_a_bucket_outside_the_project(
    client: TestClient, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    bucket = _unassessed_count_bucket(outside)
    open_new_project(tmp_path)

    assert _export_count(client, _per_image(bucket)).status_code == 403


@pytest.mark.parametrize("delivery", [
    {"kind": "not_a_real_kind"},
    {"kind": "per_image_count", "predictions_dir": "", "trait": "stem"},
    {"kind": "per_image_count", "predictions_dir": "preds", "trait": ""},
    {"kind": "orthomosaic_plant_counts", "predictions_dir": "", "plant_registry": "reg",
     "delivered_phenotype": "stem_count", "plants": ["plot0"]},
], ids=["unknown-kind", "blank-bucket", "blank-trait", "blank-raster-bucket"])
def test_export_count_csv_refuses_a_malformed_payload_with_422_not_500(
    client: TestClient, opened_project: Path, delivery: dict,
) -> None:
    resp = _export_count(client, {"delivery": delivery, "filename": "counts.csv"})
    assert resp.status_code == 422


# ── Count CSV export: orthomosaic kind ───────────────────────────────────


def _orthomosaic(project: Path) -> tuple[Path, str]:
    """An unassessed whole-raster bucket and a registered plant registry over its 2x2 grid, in a
    project whose per-plant count is confirmed; ``(bucket, registry name)``."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tests.test_orthomosaic_tools import (
        _PLANT_PIXELS, _plant_grid_csv, _plant_registry, _raster_bucket, _write_geo_raster,
    )

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_aggregate(project, "stem_count", value_keys=["count"])
    raster_path = project / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(project, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(project, _plant_grid_csv(project, raster_path, _PLANT_PIXELS))
    open_new_project(project)
    return bucket, registry


def _per_plant(bucket: Path, registry: str, *, filename: str = "plant_counts.csv",
               **extra) -> dict:
    return {"delivery": {"kind": "orthomosaic_plant_counts", "predictions_dir": str(bucket),
                         "plant_registry": registry, "delivered_phenotype": "stem_count",
                         "plants": ["plot0", "plot1", "plot2", "plot3"]},
            "filename": filename, **extra}



def test_an_unassessed_raster_bucket_refuses_with_no_acknowledgment(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket, registry = _orthomosaic(tmp_path)

    resp = _export_count(client, _per_plant(bucket, registry))

    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "delivery"
    assert "no assessment answers" in resp.json()["detail"]["message"]


def test_an_unassessed_raster_bucket_delivers_under_acknowledgment_with_its_registry_disclosed(
    client: TestClient, tmp_path: Path,
) -> None:
    """The event discloses the registry it matched against and, naming no walked mapping, carries
    no resolved mapping key."""
    bucket, registry = _orthomosaic(tmp_path)

    resp = acknowledged_post(client, COUNT_ROUTE, _per_plant(bucket, registry),
                             reason="a look before assessment")

    assert resp.status_code == 200, resp.text[:300]
    assert resp.headers["X-TCIP-Validated"] == "false"
    assert unquote(resp.headers["X-TCIP-Acknowledged-By"]) == "user:breeder"
    rows = {r["plant_id"]: r for r in _rows(resp.text)}
    assert rows["plot0"]["value"] == "1" and rows["plot3"]["value"] == "0"
    assert {r["validated"] for r in rows.values()} == {"False"}
    (event,) = client.get("/api/results/delivery-events").json()["records"]
    assert event["door"] == "results.export_count_csv"
    assert event["plant_mapping"]["plant_registry"]["name"] == registry
    assert "plant_mapping_resolved_key" not in event


def test_a_filename_with_a_directory_saves_by_its_basename(
    client: TestClient, tmp_path: Path,
) -> None:
    bucket, registry = _orthomosaic(tmp_path)

    resp = acknowledged_post(client, COUNT_ROUTE, _per_plant(
        bucket, registry, filename="../../escape/plant_counts.csv"), reason="a look")

    assert resp.status_code == 200, resp.text[:300]
    assert resp.headers["X-TCIP-Saved-To"] == str(tmp_path / "results_export" / "plant_counts.csv")


# ── Registered models and the inference job routes ────────────────────────


def test_registered_models_refuses_while_no_project_is_open(client: TestClient) -> None:
    assert client.get("/api/results/models/registered").status_code == 409


def test_registered_models_answers_the_open_project_with_none_registered(
    client: TestClient, opened_project: Path,
) -> None:
    resp = client.get("/api/results/models/registered")
    assert resp.status_code == 200
    assert resp.json()["models"] == []


def test_registered_models_answers_a_resolved_absolute_checkpoint_path(
    client: TestClient, opened_project: Path,
) -> None:
    """A relative stored checkpoint_path (the registry's own internal spelling) still answers
    absolute over this route, the surface the Inference tab feeds straight back into a launch."""
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import checkpoint_file

    ckpt_dir = opened_project / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_file(ckpt_dir / "m.pt", "route fixture weights")
    ModelRegistry(str(opened_project)).register_model("m", str(ckpt), {})

    resp = client.get("/api/results/models/registered")

    assert resp.status_code == 200
    assert Path(resp.json()["models"][0]["checkpoint_path"]) == ckpt.resolve()


def test_inference_launch_missing_checkpoint(client: TestClient, opened_project: Path) -> None:
    resp = client.post("/api/inference/launch", json={
        "checkpoint_path": str(opened_project / "no.pt"), "dataset_root": str(opened_project),
        "date": "2026-02-11",
        "output_dir": str(opened_project / "predictions" / "baseline" / "2026-02-11")})
    assert resp.status_code == 404


def test_inference_list_jobs_endpoint(client: TestClient, opened_project: Path) -> None:
    resp = client.get("/api/inference/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()


def test_inference_list_row_and_stream_frame_are_one_projection(
    client: TestClient, opened_project: Path,
) -> None:
    """The list route's row and the stream's final frame are ``_summary``'s, so a job's audit
    warning and dropped-box count reach the poll as they reach the stream, under one name."""
    from tcip_web.routes import inference as inference_routes

    job = inference_routes.InferenceJob(
        job_id="inf-warn-test", project=str(opened_project), checkpoint_path="", images_dir="",
        output_dir="", status="completed", audit_warning="the line did not land",
        dropped_boxes=3,
    )
    inference_routes._register(job)
    try:
        row = next(r for r in client.get("/api/inference/jobs").json()["jobs"]
                   if r["job_id"] == "inf-warn-test")
        with client.websocket_connect(
                "ws://127.0.0.1/api/inference/jobs/inf-warn-test/stream") as ws:
            frames = [ws.receive_json(), ws.receive_json()]
        assert [f.pop("type") for f in frames] == ["progress", "final"]
        assert frames[1] == row
        assert (row["audit_warning"], row["dropped_boxes"]) == (
            "the line did not land", 3)
        assert "warning" not in row
    finally:
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop("inf-warn-test", None)


def test_inference_stream_to_a_missing_job_sends_a_typed_terminal_frame(
    client: TestClient,
) -> None:
    """The not-found frame is typed the same as a run's own terminal frame, so the client stops
    reconnecting against a job that will never exist."""
    with client.websocket_connect("ws://127.0.0.1/api/inference/jobs/does-not-exist/stream") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "final"
    assert frame["error"] == "job not found"


def test_inference_by_id_job_route_is_retired(client: TestClient) -> None:
    """Registering a job first proves this refuses a real job, not only an absent one."""
    from tcip_web.routes import inference as inference_routes

    job = inference_routes.InferenceJob(
        job_id="inf-retired-test", project="", checkpoint_path="", images_dir="", output_dir="",
    )
    inference_routes._register(job)
    try:
        assert client.get("/api/inference/jobs/inf-retired-test").status_code == 404
    finally:
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop("inf-retired-test", None)

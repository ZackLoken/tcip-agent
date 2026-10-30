"""The Results routes.

The tests cover plant mapping, the phenology measurement, the CSV exports and the
delivery-event listing.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import pytest
from fastapi.testclient import TestClient

import tcip_store
from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, image_root
from tcip_mcp.delivery import DeliveryRefusal
from tcip_mcp.operationalization import OperationalizationRefusal
from tcip_mcp.pipelines.execution import Stated
from tcip_mcp.pipelines.postprocessing import phenology
from tcip_web.routes.audit_gap import AuditGapRefusal
from tcip_web.routes.results import EXPORT_HEADER, CountExport, SavedExport
from tests import _trait_fixtures as fx
from tests._audit_fixtures import audit_rows, refuse_audit_appends
from tests._web_fixtures import BROWSER, acknowledged_post, open_new_project

if TYPE_CHECKING:
    from starlette.testclient import httpx

    from tcip_mcp.buckets import Bucket
    from tests._chain_fixtures import Series


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(StringIO(text)))


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_list_traits_names_an_unreadable_record_alongside_the_valid_one(
    client: TestClient, opened_project: Path
) -> None:
    """The trait listing names a record its schema refuses beside the valid one.

    The refused record is listed under ``unreadable`` by name and reason.
    """
    from tcip_mcp import traits

    valid = traits.read_trait("bud_opening", opened_project).model_dump(
        mode="json"
    )
    valid["revisions"][0]["entry"]["localization"] = "unicorn_match"
    tcip_store.replace(
        traits.trait_key(opened_project, "unicorn"),
        valid,
        expect=tcip_store.Version.ABSENT,
    )

    body = client.get("/api/results/traits").json()

    assert [record["trait"] for record in body["traits"]] == ["bud_opening"]
    assert [entry["trait"] for entry in body["unreadable"]] == ["unicorn"]
    assert "localization" in body["unreadable"][0]["reason"]


def test_plant_mapping_build_requires_a_registered_dataset(
    client: TestClient,
    opened_project: Path,
) -> None:
    """A plant mapping build over a directory that is no registered dataset is refused by name.

    The refusal answers 400 naming ``register_dataset``.
    """
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "dataset_root": str(opened_project / "nope"),
            "plant_registry": "unregistered",
            "user": "tester",
        },
    )
    assert resp.status_code == 400
    assert "register_dataset" in resp.json()["detail"]


def test_plant_mapping_build_refuses_an_empty_dataset_root_as_malformed(
    client: TestClient,
    opened_project: Path,
) -> None:
    """A plant mapping build with an empty ``dataset_root`` is refused as malformed."""
    resp = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "dataset_root": "",
            "plant_registry": "unregistered",
            "user": "tester",
        },
    )
    assert resp.status_code == 422, resp.text[:300]


def test_plant_mapping_load_of_an_unstored_name_answers_404_naming_it(
    client: TestClient,
    opened_project: Path,
) -> None:
    resp = client.post(
        "/api/results/plant_mapping/load", json={"name": "missing"}
    )
    assert resp.status_code == 404
    assert "'missing'" in resp.json()["detail"]


# ── The phenology measurement and its export ────────────────────────────


def _series(
    tmp_path: Path, **kwargs: tuple | int | bool | str | dict | Stated | None
) -> Series:
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    return attributed_series(tmp_path, **kwargs)


def _export(
    client: TestClient,
    body: dict,
    payload: str = "milestones",
    headers: dict[str, str] | None = None,
    **extra: object,
) -> httpx.Response:
    return client.post(
        "/api/results/export_csv",
        headers=headers,
        json={
            **body,
            "payload": payload,
            "user": "breeder",
            **extra,
        },
    )


def test_an_assessed_series_measures_its_curves_and_milestones_validated(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An assessed series measures validated curves and milestones.

    The answer carries two plants' curves, PLANT_A's 2026-02-11 row counting four detections,
    and each milestone crossing of PLANT_A's row names a 2026 date and a bound.
    """
    series = _series(tmp_path)

    resp = client.post(
        "/api/results/phenology_measurement", json=series.body()
    )

    assert resp.status_code == 200, resp.text[:300]
    out = resp.json()
    assert out["unvalidated"] is None
    assert out["positive_class_assessed"] is True
    assert len(out["milestones"]) == 2
    by_key = {(r["plant_id"], r["capture"]): r for r in out["curves"]}
    assert by_key[("PLANT_A", "2026-02-11")]["n_total"] == 4
    assert by_key[("PLANT_A", "2026-02-11")]["ratio"] == 0.0
    assert by_key[("PLANT_B", "2026-03-10")]["ratio"] == 0.75
    plant_a = next(r for r in out["milestones"] if r["plant_id"] == "PLANT_A")
    assert plant_a["milestones"]
    for key, crossing in plant_a["milestones"].items():
        assert crossing["date"].startswith("2026-"), key
        assert crossing["bound"], key


def test_no_door_takes_a_switch_computing_milestones_over_incomplete_dates(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The web route and the tool each refuse ``require_all_dates_complete`` by name.

    The route answers 422 naming the key, the tool raises TypeError naming it, and the tool's
    output CSV is not written.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    switched = _series(tmp_path, fractions=(0.0, 1.0)).body(
        require_all_dates_complete=False
    )

    refused = client.post("/api/results/phenology_measurement", json=switched)
    assert refused.status_code == 422
    assert "require_all_dates_complete" in refused.text
    with pytest.raises(TypeError, match="require_all_dates_complete"):
        deliver_phenology_milestones(
            tmp_path,
            **switched,
            output_csv_path=str(tmp_path / "out.csv"),
            actor=None,
        )
    assert not (tmp_path / "out.csv").exists()


def test_the_curve_counts_every_image_of_a_plant_on_a_date(
    client: TestClient,
    tmp_path: Path,
) -> None:
    body = _series(tmp_path, fractions=(0.5,), images_per_plant=3).body()

    rows = client.post("/api/results/phenology_measurement", json=body).json()[
        "curves"
    ]

    row = next(r for r in rows if r["plant_id"] == "PLANT_A")
    assert row["n_images"] == 3
    assert row["n_total"] == 12


def test_an_unassessed_series_refuses_until_shown_unvalidated_and_never_exports_on_its_own(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An unassessed series is refused until shown unvalidated, and its export stays refused.

    A breeder can look at provisional numbers (``show_unvalidated``, a display choice); each
    export refuses, naming the digest of the result shown.
    """
    body = _series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    refused = client.post("/api/results/phenology_measurement", json=body)
    shown = client.post(
        "/api/results/phenology_measurement",
        json={**body, "show_unvalidated": True},
    )

    assert refused.status_code == 400
    unvalidated = shown.json()["unvalidated"]
    assert refused.json()["detail"]["message"] == unvalidated["message"]
    assert shown.status_code == 200
    assert "no assessment answers" in unvalidated["message"]
    assert shown.json()["curves"]
    shown_digests = unvalidated["digests"]
    for payload in get_args(phenology.Projection):
        export = _export(client, body, payload)
        assert export.status_code == 400, payload
        # The screen and the export name one prepared result: the digest an acknowledgment of
        # what was shown binds to is the one the export would write.
        assert (
            export.json()["detail"]["result_sha256"] == shown_digests[payload]
        ), payload
    assert shown_digests["curves"] != shown_digests["milestones"]


def test_an_acknowledged_export_ships_unvalidated_and_its_event_names_the_act(
    client: TestClient,
    tmp_path: Path,
) -> None:
    body = _series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    resp = acknowledged_post(
        client,
        "/api/results/export_csv",
        {**body, "payload": "milestones"},
        reason="a look before assessment",
    )

    assert resp.status_code == 200, resp.text[:300]
    assert {row["validated"] for row in _rows(resp.text)} == {"False"}
    events = client.get("/api/results/delivery-events").json()["records"]
    (event,) = [
        r["event"]
        for r in events
        if r["event"]["door"] == "results.export_csv"
    ]
    assert (
        event["acknowledgment"]["acknowledged_by"],
        event["acknowledgment"]["reason"],
    ) == ("user:breeder", "a look before assessment")
    assert event["validated"] is False
    lines = [
        e
        for root in {tmp_path, Path(event["buckets"][0]["dataset_root"])}
        for tool in ("delivery_acknowledged", "delivery_event")
        for e in audit_rows(root, tool)
    ]
    assert sorted(
        tool for e in lines if isinstance(tool := e["tool"], str)
    ) == ["delivery_acknowledged", "delivery_event"]
    assert {e["actor"] for e in lines} == {"user:breeder"}
    assert all("acknowledged_by" not in e["arguments"] for e in lines)


def test_an_acknowledgment_is_recorded_only_from_the_browser_and_with_a_reason(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An acknowledgment is recorded only from the browser and with a reason.

    A request from no browser, or one declaring an agent identity, is refused 403, and a
    reason of spaces 400; none records a delivery event, and the acknowledgment from the
    browser ships.
    """
    from tcip_mcp import agent_identity

    body = _series(tmp_path, fractions=(0.0, 1.0), assessed=False).body()

    digest = _export(client, body).json()["detail"]["result_sha256"]
    acknowledgment = {"reason": "needs a look", "result_sha256": digest}
    bare = _export(client, body, acknowledgment=acknowledgment)
    agent = _export(
        client,
        body,
        acknowledgment=acknowledgment,
        headers={
            **BROWSER,
            agent_identity.HEADERS["agent_session"]: "mcp_0123",
        },
    )
    blank = _export(
        client,
        body,
        acknowledgment={**acknowledgment, "reason": "   "},
        headers=BROWSER,
    )

    assert (bare.status_code, agent.status_code) == (403, 403)
    assert "cannot record one" in bare.json()["detail"]
    assert blank.status_code == 400
    assert "an acknowledgment states why" in blank.json()["detail"]
    assert client.get("/api/results/delivery-events").json()["records"] == []
    assert (
        _export(
            client, body, acknowledgment=acknowledgment, headers=BROWSER
        ).status_code
        == 200
    )


def test_an_export_saves_the_delivery_under_the_project_and_logs_it_with_its_dataset(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An export saves the delivery under the project and logs it with its dataset.

    It saves under ``results_export/`` as ``phenology.export_filename`` names it, its bytes the
    response's, under the trait's own columns, the answer reporting the file and naming its
    attachment by it; each row names the one delivery event and reads validated, and the event's
    line lands in the log of the dataset its buckets sit in.
    """
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.postprocessing import phenology

    series = _series(tmp_path, fractions=(0.0, 1.0))
    body = series.body()

    resp = client.post(
        "/api/results/export_csv",
        json={**body, "payload": "milestones", "user": "breeder"},
    )

    assert resp.status_code == 200, resp.text[:300]
    name = phenology.export_filename(body["trait"], "milestones")
    saved = tmp_path / "results_export" / name
    assert SavedExport.model_validate_json(
        resp.headers[EXPORT_HEADER]
    ) == SavedExport(saved_path=str(saved))
    assert f'filename="{name}"' in resp.headers["content-disposition"]
    assert saved.read_bytes() == resp.content
    header = resp.text.splitlines()[0].split(",")
    assert header == phenology.phenology_csv_columns(
        latest_confirmed("bud_opening", tmp_path).entry
    )
    (record,) = read_delivery_events(tmp_path)
    assert {row["delivery_event_id"] for row in _rows(resp.text)} == {
        record.event_id
    }
    assert {row["validated"] for row in _rows(resp.text)} == {"True"}
    logged = audit_rows(series.root, "delivery_event")
    assert [e["arguments"] for e in logged] == [{"event_id": record.event_id}]


def test_the_curves_export_writes_the_curve_schema(
    client: TestClient, tmp_path: Path
) -> None:
    from tcip_mcp.pipelines.postprocessing import phenology

    body = _series(tmp_path, fractions=(0.0, 1.0)).body()

    resp = _export(client, body, "curves")

    assert resp.status_code == 200, resp.text[:300]
    assert (
        resp.text.splitlines()[0].split(",") == phenology.curve_csv_columns()
    )
    assert len(_rows(resp.text)) == 4


def test_a_walked_mapping_event_names_the_mapping_key_it_resolves_to(
    client: TestClient,
    tmp_path: Path,
) -> None:
    body = _series(tmp_path, fractions=(0.0, 1.0)).body()
    assert _export(client, body).status_code == 200

    (served,) = client.get("/api/results/delivery-events").json()["records"]

    assert served["event"]["plant_mapping"]["name"] == "valley"
    assert served["resolved_key"] == "valley"


def test_export_csv_answers_409_when_the_delivery_event_audit_line_cannot_be_appended(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed delivery audit append answers 409 naming the CSV written, left in place.

    The CSV is already on disk by the time the delivery's audit line is appended.
    """
    from tcip_mcp.pipelines.postprocessing import phenology

    body = _series(tmp_path, fractions=(0.0, 1.0)).body()
    refuse_audit_appends(monkeypatch)

    resp = client.post(
        "/api/results/export_csv",
        json={
            **body,
            "payload": "milestones",
            "user": "breeder",
        },
    )

    assert resp.status_code == 409
    refusal = AuditGapRefusal.model_validate(resp.json()["detail"])
    saved = (
        tmp_path
        / "results_export"
        / phenology.export_filename(body["trait"], "milestones")
    )
    assert SavedExport.model_validate(refusal.committed) == SavedExport(
        saved_path=str(saved)
    )
    assert saved.exists()


def test_phenology_doors_reject_a_malformed_payload_with_422_not_500(
    client: TestClient,
) -> None:
    for route in ("phenology_measurement", "export_csv"):
        resp = client.post(
            f"/api/results/{route}", json={"trait": "bud_opening"}
        )
        assert resp.status_code == 422, route


def test_caller_composed_rows_and_validity_claims_are_refused_at_the_payload(
    client: TestClient,
) -> None:
    """Caller-composed rows and validity claims are refused at the payload.

    An export carrying rows, and a phenology measurement carrying ``validated`` or
    ``unvalidated``, each answer 422.
    """
    inputs = {
        "mapping_name": "valley",
        "trait": "bud_opening",
        "buckets": ["x"],
        "plants": ["PLANT_A"],
    }
    for rows in (
        [
            {
                "plant_id": "P1",
                "date": "2026-03-01",
                "n_total": 20,
                "ratio": 0.05,
            }
        ],
        [
            {
                "plant_id": "P1",
                "start_date": "2026-03-01",
                "end_date": "2026-04-02",
            }
        ],
    ):
        resp = client.post(
            "/api/results/export_csv",
            json={
                "rows": rows,
                "export_kind": "diagnostic",
            },
        )
        assert resp.status_code == 422
    for field, value in (("validated", True), ("unvalidated", None)):
        resp = client.post(
            "/api/results/phenology_measurement", json={**inputs, field: value}
        )
        assert resp.status_code == 422, field


def test_phenology_measurement_refuses_when_the_delivered_dataset_carries_no_registry(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """The phenology measurement refuses a delivered dataset that carries no registry.

    The delivered buckets resolve to a dataset root that carries no registry; the door answers
    400 saying it holds no subject registry.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    open_new_project(tmp_path)
    bucket = published(
        tmp_path,
        "live/2026-02-11",
        [
            predicted(
                image_dir(tmp_path, "2026-02-11") / "PLANT_A.png", ["bud"]
            )
        ],
        scope={"subject": "bud"},
    )

    resp = client.post(
        "/api/results/phenology_measurement",
        json={
            "mapping_name": "valley",
            "dataset_root": str(bucket.root),
            "buckets": [bucket.name],
            "trait": "bud_opening",
            "plants": ["PLANT_A"],
        },
    )

    assert resp.status_code == 400
    assert "holds no subject registry" in resp.json()["detail"]


# ── Count CSV export: per-image kind ──────────────────────────────────────

COUNT_SCOPE = {"subject": fx.COUNT_SUBJECT}


def _unassessed_count_bucket(
    project: Path, *, scope: dict = COUNT_SCOPE
) -> Bucket:
    """Publish and return an unassessed bucket of two frames with three detections each.

    The bucket is ``live/counts`` of the project's own images, in a project whose count trait is
    confirmed.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import predicted, published

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_count(project, measured_subject=fx.COUNT_SUBJECT)
    subject = scope["subject"]
    bucket = published(
        project,
        "live/counts",
        [
            predicted(image_root(project) / f"img{i}.png", [subject] * 3)
            for i in range(2)
        ],
        scope=scope,
    )
    open_new_project(project)
    return bucket


def _per_image(
    bucket: Bucket, trait: str = fx.COUNT_TRAIT, **extra: object
) -> dict:
    """Return an export body for ``bucket`` with ``extra`` merged over it last.

    Before the merge, the body's ``delivery`` is a ``per_image_count`` of ``bucket``'s root and
    name under ``trait``, its filename ``counts.csv`` and its user ``breeder``.
    """
    return {
        "delivery": {
            "kind": "per_image_count",
            "dataset_root": str(bucket.root),
            "bucket": bucket.name,
            "trait": trait,
        },
        "filename": "counts.csv",
        "user": "breeder",
        **extra,
    }


COUNT_ROUTE = "/api/results/export_count_csv"


def _export_count(
    client: TestClient, body: dict, headers: dict[str, str] | None = None
) -> httpx.Response:
    return client.post(COUNT_ROUTE, json=body, headers=headers)


def _digest(client: TestClient, body: dict) -> str:
    """Return the result digest the count export's refusal of ``body`` names."""
    return _export_count(client, body).json()["detail"]["result_sha256"]


def test_an_unassessed_count_bucket_refuses_with_no_acknowledgment(
    client: TestClient,
    tmp_path: Path,
) -> None:
    resp = _export_count(
        client, _per_image(_unassessed_count_bucket(tmp_path))
    )

    assert resp.status_code == 400
    refusal = DeliveryRefusal.model_validate(resp.json()["detail"])
    assert "no assessment answers" in refusal.message
    assert not (tmp_path / "results_export" / "counts.csv").exists()


def test_an_unassessed_count_bucket_delivers_under_acknowledgment(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    resp = acknowledged_post(
        client,
        COUNT_ROUTE,
        _per_image(bucket),
        reason="a look before assessment",
    )

    assert resp.status_code == 200, resp.text[:300]
    reported = CountExport.model_validate_json(resp.headers[EXPORT_HEADER])
    assert (reported.validated, reported.acknowledged_by) == (
        False,
        "user:breeder",
    )
    rows = _rows(resp.text)
    assert [r["detection_count"] for r in rows] == ["3", "3"]
    assert {r["validated"] for r in rows} == {"False"}
    (served,) = client.get("/api/results/delivery-events").json()["records"]
    event = served["event"]
    assert event["door"] == "results.export_count_csv"
    assert event["acknowledgment"]["acknowledged_by"] == "user:breeder"


def test_an_assessed_count_bucket_delivers_validated_and_discards_an_acknowledgment(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An assessed count bucket delivers validated and discards a posted acknowledgment.

    A validated delivery's event carries no acknowledgment, even when one was posted beside
    it. An export whose buckets' root is not a dataset registered to the project is refused.
    """
    pytest.importorskip("torch")
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-web-count")
    open_new_project(tmp_path)
    bucket = read_bucket(chain.root, chain.bucket)
    unregistered = _export_count(client, _per_image(bucket))
    registered = register_dataset(
        tmp_path,
        str(chain.root),
        crop=sorted(registered_crops())[0],
        actor=None,
    )
    assert "error" not in registered, registered

    bare = _export_count(client, _per_image(bucket))
    posted = _export_count(
        client,
        _per_image(
            bucket,
            acknowledgment={
                "reason": "just in case",
                "result_sha256": "0" * 64,
            },
        ),
        headers=BROWSER,
    )

    assert unregistered.status_code == 403, unregistered.text[:300]
    assert bare.status_code == 200, bare.text[:300]
    reported = CountExport.model_validate_json(bare.headers[EXPORT_HEADER])
    assert (reported.validated, reported.acknowledged_by) == (True, None)
    assert {r["validated"] for r in _rows(bare.text)} == {"True"}
    assert posted.status_code == 200
    assert (
        CountExport.model_validate_json(
            posted.headers[EXPORT_HEADER]
        ).acknowledged_by
        is None
    )
    events = client.get("/api/results/delivery-events").json()["records"]
    assert [e["event"]["acknowledgment"] for e in events] == [None, None]
    lines = [
        e
        for root in {tmp_path, Path(chain.root)}
        for e in audit_rows(root, "delivery_event")
    ]
    assert [e["actor"] for e in lines] == ["user:breeder"] * 2


def test_export_count_csv_answers_409_when_the_delivery_event_audit_line_cannot_be_appended(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)
    digest = _digest(client, _per_image(bucket))
    refuse_audit_appends(
        monkeypatch, landing=1
    )  # the acknowledgment's own line lands

    resp = _export_count(
        client,
        _per_image(
            bucket,
            acknowledgment={"reason": "a look", "result_sha256": digest},
        ),
        headers=BROWSER,
    )

    assert resp.status_code == 409
    refusal = AuditGapRefusal.model_validate(resp.json()["detail"])
    saved = tmp_path / "results_export" / "counts.csv"
    assert SavedExport.model_validate(refusal.committed) == SavedExport(
        saved_path=str(saved)
    )
    assert saved.exists()


def test_export_count_csv_refuses_an_acknowledgment_from_no_browser_or_with_a_blank_reason(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    digest = _digest(client, _per_image(bucket))
    bare = _export_count(
        client,
        _per_image(
            bucket,
            acknowledgment={"reason": "no browser", "result_sha256": digest},
        ),
    )
    blank = _export_count(
        client,
        _per_image(
            bucket, acknowledgment={"reason": "   ", "result_sha256": digest}
        ),
        headers=BROWSER,
    )

    assert (
        bare.status_code == 403
        and "cannot record one" in bare.json()["detail"]
    )
    assert blank.status_code == 400 and "reason" in blank.json()["detail"]
    assert "an acknowledgment states why" in blank.json()["detail"]


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_export_count_csv_refuses_a_trait_that_states_no_per_image_count(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    resp = _export_count(client, _per_image(bucket, trait="bud_opening"))

    assert resp.status_code == 400
    OperationalizationRefusal.model_validate(resp.json()["detail"])


def test_export_count_csv_refuses_a_bucket_whose_scope_omits_the_confirmed_subject(
    client: TestClient,
    tmp_path: Path,
) -> None:
    other = {"subject": "other"}
    bucket = _unassessed_count_bucket(tmp_path, scope=other)

    resp = _export_count(client, _per_image(bucket))

    assert resp.status_code == 400
    OperationalizationRefusal.model_validate(resp.json()["detail"])


def test_export_count_csv_refuses_an_unknown_trait_with_400_not_500(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bucket = _unassessed_count_bucket(tmp_path)

    resp = _export_count(client, _per_image(bucket, trait="no-such-trait"))

    assert resp.status_code == 400
    DeliveryRefusal.model_validate(resp.json()["detail"])


def test_export_count_csv_refuses_a_bucket_outside_the_project(
    client: TestClient,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    bucket = _unassessed_count_bucket(outside)
    open_new_project(tmp_path)

    assert _export_count(client, _per_image(bucket)).status_code == 403


@pytest.mark.parametrize(
    "delivery",
    [
        {"kind": "not_a_real_kind"},
        {
            "kind": "per_image_count",
            "dataset_root": "ds",
            "bucket": "",
            "trait": "stem",
        },
        {
            "kind": "per_image_count",
            "dataset_root": "ds",
            "bucket": "preds",
            "trait": "",
        },
        {
            "kind": "orthomosaic_plant_counts",
            "dataset_root": "ds",
            "bucket": "",
            "plant_registry": "reg",
            "delivered_phenotype": "stem_count",
            "plants": ["plot0"],
        },
    ],
    ids=["unknown-kind", "blank-bucket", "blank-trait", "blank-raster-bucket"],
)
def test_export_count_csv_refuses_a_malformed_payload_with_422_not_500(
    client: TestClient,
    opened_project: Path,
    delivery: dict,
) -> None:
    resp = _export_count(
        client, {"delivery": delivery, "filename": "counts.csv"}
    )
    assert resp.status_code == 422


# ── Count CSV export: orthomosaic kind ───────────────────────────────────


def _orthomosaic(project: Path) -> tuple[Any, str]:
    """Return an unassessed whole-raster bucket and a registered plant registry over its grid.

    The bucket covers the project's own image tree and the registry its 2x2 grid, in a project
    whose per-plant count is confirmed; the return is ``(bucket, registry name)``.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tests.test_orthomosaic_tools import (
        _PLANT_PIXELS,
        _plant_grid_csv,
        _plant_registry,
        _raster_bucket,
        _write_geo_raster,
    )

    fx.seed_delivery_traits(project)
    fx.seed_confirmed_aggregate(project, "stem_count", value_keys=["count"])
    raster_path = image_dir(project, UNDATED_BUCKET) / "mosaic.tif"
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(project, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(
        project, _plant_grid_csv(project, raster_path, _PLANT_PIXELS)
    )
    open_new_project(project)
    return bucket, registry


def _per_plant(
    bucket: Bucket,
    registry: str,
    *,
    filename: str = "plant_counts.csv",
    **extra: object,
) -> dict:
    return {
        "delivery": {
            "kind": "orthomosaic_plant_counts",
            "dataset_root": str(bucket.root),
            "bucket": bucket.name,
            "plant_registry": registry,
            "delivered_phenotype": "stem_count",
            "plants": ["plot0", "plot1", "plot2", "plot3"],
        },
        "filename": filename,
        "user": "breeder",
        **extra,
    }


def test_an_unassessed_raster_bucket_refuses_with_no_acknowledgment(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bucket, registry = _orthomosaic(tmp_path)

    resp = _export_count(client, _per_plant(bucket, registry))

    assert resp.status_code == 400
    refusal = DeliveryRefusal.model_validate(resp.json()["detail"])
    assert "no assessment answers" in refusal.message


def test_an_unassessed_raster_bucket_delivers_under_acknowledgment_with_its_registry_disclosed(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An unassessed raster bucket delivers under acknowledgment with its registry disclosed.

    The event discloses the registry it matched against and, naming no walked mapping, carries
    no resolved mapping key.
    """
    bucket, registry = _orthomosaic(tmp_path)

    resp = acknowledged_post(
        client,
        COUNT_ROUTE,
        _per_plant(bucket, registry),
        reason="a look before assessment",
    )

    assert resp.status_code == 200, resp.text[:300]
    reported = CountExport.model_validate_json(resp.headers[EXPORT_HEADER])
    assert (reported.validated, reported.acknowledged_by) == (
        False,
        "user:breeder",
    )
    rows = {r["plant_id"]: r for r in _rows(resp.text)}
    assert rows["plot0"]["value"] == "1" and rows["plot3"]["value"] == "0"
    assert {r["validated"] for r in rows.values()} == {"False"}
    (served,) = client.get("/api/results/delivery-events").json()["records"]
    assert served["event"]["door"] == "results.export_count_csv"
    assert (
        served["event"]["plant_mapping"]["plant_registry"]["name"] == registry
    )
    assert served["resolved_key"] is None


def test_a_filename_with_a_directory_saves_by_its_basename(
    client: TestClient,
    tmp_path: Path,
) -> None:
    bucket, registry = _orthomosaic(tmp_path)

    resp = acknowledged_post(
        client,
        COUNT_ROUTE,
        _per_plant(bucket, registry, filename="../../escape/plant_counts.csv"),
        reason="a look",
    )

    assert resp.status_code == 200, resp.text[:300]
    assert CountExport.model_validate_json(
        resp.headers[EXPORT_HEADER]
    ).saved_path == str(tmp_path / "results_export" / "plant_counts.csv")


# ── Registered models and the inference job routes ────────────────────────


def test_registered_models_refuses_while_no_project_is_open(
    client: TestClient,
) -> None:
    assert client.get("/api/results/models/registered").status_code == 409


def test_registered_models_answers_the_open_project_with_none_registered(
    client: TestClient,
    opened_project: Path,
) -> None:
    resp = client.get("/api/results/models/registered")
    assert resp.status_code == 200
    assert resp.json()["models"] == []


def test_assessments_refuses_while_no_project_is_open(
    client: TestClient,
) -> None:
    assert client.get("/api/results/assessments").status_code == 409


def test_assessments_answers_the_open_project_with_none_recorded(
    client: TestClient,
    opened_project: Path,
) -> None:
    resp = client.get("/api/results/assessments")
    assert resp.status_code == 200
    assert resp.json()["assessments"] == []


def test_assessments_lists_each_recorded_assessment_with_the_checkpoint_it_judged(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """An assessment the platform recorded is listed with its producer's digest and run.

    The digest is the one the registered-models listing answers for the same checkpoint, so
    the two listings pair without a path comparison.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-web-assessed")
    open_new_project(tmp_path)

    listed = client.get("/api/results/assessments")
    models = client.get("/api/results/models/registered")

    assert listed.status_code == 200, listed.text
    (row,) = listed.json()["assessments"]
    (model,) = [
        m["entry"]
        for m in models.json()["models"]
        if m["entry"]["experiment_id"] == "exp-web-assessed"
    ]
    from tcip_mcp.assessment import ASSESSMENT_RECORD, read_assessment

    recorded = read_assessment(tmp_path, chain.assessment["assessment_id"])
    assert row == {
        "assessment": ASSESSMENT_RECORD.dump_python(recorded, mode="json"),
        "passed": True,
    }
    producer = row["assessment"]["producer"]
    assert (producer["checkpoint_sha256"], producer["experiment_id"]) == (
        model["sha256"],
        "exp-web-assessed",
    )


def test_registered_models_answers_a_resolved_absolute_checkpoint_path(
    client: TestClient,
    opened_project: Path,
) -> None:
    """The registered-models route answers a resolved absolute checkpoint path.

    A relative stored checkpoint_path (the registry's own internal spelling) answers absolute
    over this route.
    """
    pytest.importorskip("torch")
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    ckpt_dir = opened_project / ".tcip" / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = produced_checkpoint(ckpt_dir / "m.pt", "route fixture weights")
    ModelRegistry(str(opened_project)).register_model(
        "m", str(ckpt), actor=None
    )

    resp = client.get("/api/results/models/registered")

    assert resp.status_code == 200
    assert (
        Path(resp.json()["models"][0]["entry"]["checkpoint_path"])
        == ckpt.resolve()
    )


def test_the_plant_mapping_list_names_each_mapping_the_project_built(
    client: TestClient,
    opened_project: Path,
) -> None:
    """A mapping ``build_plant_mapping`` built under the open project is listed by its name."""
    from tests._mapping_fixtures import map_captures

    map_captures(
        opened_project, opened_project / "ds", ["2026-02-11"], name="valley"
    )

    listed = client.get("/api/results/plant_mapping/list")
    assert listed.status_code == 200, listed.text
    assert listed.json() == {"names": ["valley"]}


def test_the_plant_mapping_list_refuses_while_no_project_is_open(
    client: TestClient,
) -> None:
    assert client.get("/api/results/plant_mapping/list").status_code == 409


def test_inference_launch_missing_checkpoint(
    client: TestClient, opened_project: Path
) -> None:
    resp = client.post(
        "/api/inference/launch",
        json={
            "checkpoint_path": str(opened_project / "no.pt"),
            "dataset_root": str(opened_project),
            "date": "2026-02-11",
            "user": "tester",
            "bucket_name": "baseline",
            "stated": {},
        },
    )
    assert resp.status_code == 404


def test_a_launched_inference_job_is_canceled_by_its_id_naming_the_person(
    client: TestClient,
    opened_project: Path,
) -> None:
    """A launched inference job is canceled by its id, naming the person.

    The answer names the job and one ``inference_canceled`` line names the person; an id naming
    no job refuses by name.
    """
    pytest.importorskip("torch")
    from datetime import datetime

    from PIL import Image

    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tcip_web.routes import inference as inference_routes
    from tests._image_fixtures import capture_exif
    from tests._verified_checkpoint_fixtures import produced_checkpoint

    source = opened_project / "raw"
    source.mkdir()
    Image.new("RGB", (16, 16)).save(
        source / "tree_01.jpg", exif=capture_exif(datetime(2026, 2, 11, 9))
    )
    assert "error" not in ingest_images(
        opened_project, str(source), actor=None
    )
    ckpt = produced_checkpoint(
        opened_project / "m.pt", "cancel fixture weights"
    )
    ModelRegistry(str(opened_project)).register_model(
        "m", str(ckpt), actor=None
    )
    launched = client.post(
        "/api/inference/launch",
        json={
            "checkpoint_path": str(ckpt),
            "dataset_root": str(opened_project),
            "date": "2026-02-11",
            "bucket_name": "baseline",
            "user": "tester",
            "stated": {},
        },
    )
    assert launched.status_code == 200, launched.text
    job_id = launched.json()["job_id"]
    try:
        before = len(audit_rows(opened_project, "inference_canceled"))
        canceled = client.post(
            f"/api/inference/jobs/{job_id}/cancel", json={"user": "tester"}
        )
        assert canceled.status_code == 200, canceled.text
        assert canceled.json()["job_id"] == job_id
        lines = audit_rows(opened_project, "inference_canceled")[before:]
        assert [
            (arguments["job_id"], r["actor"])
            for r in lines
            if isinstance(arguments := r["arguments"], dict)
        ] == [(job_id, "user:tester")]
    finally:
        job = inference_routes._get(job_id)
        if job is not None and job.thread is not None:
            job.thread.join(timeout=60)
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop(job_id, None)

    missing = client.post(
        "/api/inference/jobs/inf-absent/cancel", json={"user": "tester"}
    )
    assert missing.status_code == 404
    assert "inf-absent" in missing.json()["detail"]


def test_inference_list_jobs_endpoint(
    client: TestClient, opened_project: Path
) -> None:
    resp = client.get("/api/inference/jobs")
    assert resp.status_code == 200
    assert "jobs" in resp.json()


def test_inference_list_row_and_stream_frame_are_one_projection(
    client: TestClient,
    opened_project: Path,
) -> None:
    """The inference list row and the stream's final frame carry one record of the job."""
    from tcip_web.routes import inference as inference_routes
    from tests._web_fixtures import inference_job

    job = inference_job(
        image_dir(opened_project / "ds", "2026-01-01"),
        opened_project,
        job_id="inf-warn-test",
        status="completed",
    )
    inference_routes._registry.launch(job, lambda _job: None)
    try:
        row = next(
            r
            for r in client.get("/api/inference/jobs").json()["jobs"]
            if r["job_id"] == "inf-warn-test"
        )
        with client.websocket_connect(
            "ws://127.0.0.1/api/inference/jobs/inf-warn-test/stream"
        ) as ws:
            frames = [ws.receive_json(), ws.receive_json()]
        assert [f["type"] for f in frames] == ["progress", "final"]
        assert frames[1]["job"] == row
    finally:
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop("inf-warn-test", None)


def test_inference_stream_to_a_missing_job_sends_a_typed_terminal_frame(
    client: TestClient,
) -> None:
    """A stream to a missing job sends a ``final`` frame naming the job as not found."""
    with client.websocket_connect(
        "ws://127.0.0.1/api/inference/jobs/does-not-exist/stream"
    ) as ws:
        frame = ws.receive_json()
    assert frame["type"] == "final"
    assert frame["error"] == "job not found"


def test_inference_by_id_job_route_is_retired(client: TestClient) -> None:
    """A GET of one inference job by id answers 404, even for a registered job."""
    from tcip_web.routes import inference as inference_routes
    from tests._web_fixtures import inference_job

    job = inference_job(
        image_dir("ds", "2026-01-01"), "", job_id="inf-retired-test"
    )
    inference_routes._registry.launch(job, lambda _job: None)
    try:
        assert (
            client.get("/api/inference/jobs/inf-retired-test").status_code
            == 404
        )
    finally:
        with inference_routes._registry.lock:
            inference_routes._registry.jobs.pop("inf-retired-test", None)

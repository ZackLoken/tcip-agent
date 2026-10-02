"""The check every delivery door runs: the trait's latest confirmed revision, and whether its
operationalization binds what the door is about to deliver.

A delivered phenotype without a breeder-confirmed meaning is a number nobody defined. These cases
pin each refusal, the order in which they report, and the calls that must still succeed, because
a rail that only rejects is a rail that has not been shown to admit valid work.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp import traits
from tcip_mcp.operationalization import OperationalizationRefused, bind, confirmed_revision
from tcip_mcp.traits import (
    PER_IMAGE_COUNT,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    STATE_CROSSING_DATES,
    TraitUnknownError,
)
from tcip_web.app import app
from tests import _trait_fixtures as fx


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project whose crossing and count traits are proposed and confirmed with no
    operationalization, and whose registry declares the crossing trait's positive class."""
    root = tmp_path / "project"
    fx.propose_and_confirm(root, fx.CROSSING_SPEC)
    fx.propose_and_confirm(root, fx.COUNT_SPEC)
    fx.seed_positive_class(root, "flower", fx.CROSSING_SPEC.positive_value)
    return root


def _confirmed_crossing(project: Path) -> traits.TraitRevision:
    return fx.seed_confirmed_crossing(project, fx.CROSSING_TRAIT, measured_subject="flower")


def _confirmed_aggregate(project: Path, kind: str) -> traits.TraitRevision:
    return fx.propose_and_confirm(project, fx.with_operationalization(
        fx.COUNT_SPEC, kind, delivered_phenotypes=("stem_count",),
        delivered_value_keys=("stem_count",)))


# ── each refusal, one at a time ──────────────────────────────────────────────


def test_a_kind_the_confirmed_revision_does_not_state_refuses_and_names_the_proposing_tool(
    project: Path,
):
    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT)

    message = str(excinfo.value)
    assert "states no operationalization for a state_crossing_dates delivery" in message
    assert "propose_trait" in message and "Setup tab" in message
    assert "bloom_05per_date" in message and "Date when 5%" in message


def test_a_trait_with_no_confirmed_revision_refuses_and_names_who_confirms(tmp_path: Path):
    fx.propose(tmp_path, fx.with_operationalization(
        fx.COUNT_SPEC, PER_IMAGE_COUNT, measured_subject=fx.COUNT_SUBJECT))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_IMAGE_COUNT, project=tmp_path, trait=fx.COUNT_TRAIT)

    assert "none of its 1 revision(s) is confirmed" in str(excinfo.value)
    assert excinfo.value.as_detail() == {"kind": "operationalization", "message": str(excinfo.value)}


def test_a_value_key_outside_the_confirmed_set_refuses(project: Path):
    _confirmed_aggregate(project, PER_PLANT_COUNT_AGGREGATE)

    with pytest.raises(OperationalizationRefused, match="leaf_length"):
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project=project, trait=fx.COUNT_TRAIT,
                           value_keys=["stem_count", "leaf_length"])


def test_a_row_carrying_no_value_key_refuses_and_counts_them(project: Path):
    _confirmed_aggregate(project, PER_PLANT_COUNT_AGGREGATE)

    with pytest.raises(OperationalizationRefused, match="2 row"):
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project=project, trait=fx.COUNT_TRAIT,
                           value_keys=["stem_count", "", None])


def test_a_delivered_phenotype_outside_the_confirmed_set_refuses(project: Path):
    _confirmed_crossing(project)

    with pytest.raises(OperationalizationRefused, match="bloom_95per_date"):
        confirmed_revision(STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT,
                           delivered_phenotype="bloom_95per_date")


def test_a_bucket_not_counting_the_measured_subject_refuses(project: Path):
    revision = fx.seed_confirmed_count(project)

    with pytest.raises(OperationalizationRefused) as excinfo:
        bind(revision, PER_IMAGE_COUNT, buckets={"predictions/live/2026-03-04": {"leaf"}})

    assert "stem" in str(excinfo.value) and "predictions/live/2026-03-04" in str(excinfo.value)


def test_a_registry_no_longer_declaring_the_positive_class_refuses_with_its_problem(
    project: Path,
):
    from tcip_mcp import subject_registry as cr

    _confirmed_crossing(project)
    registry = cr.SubjectRegistry(subjects=(cr.Subject(name="flower", attributes=(
        cr.Attribute(name="state", type="categorical", values=("shed",)),)),))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT,
                           registry=registry)

    problem = cr.positive_value_problem(registry, "flower", "open")
    assert problem is not None and problem in excinfo.value.as_detail()["message"]


# ── ordering ─────────────────────────────────────────────────────────────────


def test_an_unconfirmed_trait_reports_that_rather_than_a_binding(tmp_path: Path):
    fx.propose(tmp_path, fx.with_operationalization(
        fx.COUNT_SPEC, PER_PLANT_COUNT_AGGREGATE, delivered_phenotypes=("stem_count",),
        delivered_value_keys=("stem_count",)))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project=tmp_path, trait=fx.COUNT_TRAIT,
                           value_keys=["not_covered"])

    assert "is confirmed" in str(excinfo.value) and "not_covered" not in str(excinfo.value)


# ── what the rail admits ─────────────────────────────────────────────────────


def test_a_confirmed_revision_passes_every_binding_it_covers(project: Path):
    crossing = _confirmed_crossing(project)

    assert confirmed_revision(
        STATE_CROSSING_DATES, project=project, trait=fx.CROSSING_TRAIT,
        delivered_phenotype="bloom_50per_date") == crossing


@pytest.mark.parametrize(
    "delivery_kind",
    [PER_PLANT_COUNT_AGGREGATE, PER_PLANT_ORDINAL_AGGREGATE, traits.PER_PLANT_REGRESSION_AGGREGATE],
)
def test_an_aggregate_revision_admits_its_named_value_keys(project: Path, delivery_kind: str):
    revision = _confirmed_aggregate(project, delivery_kind)

    assert confirmed_revision(
        delivery_kind, project=project, trait=fx.COUNT_TRAIT,
        delivered_phenotype="stem_count", value_keys=["stem_count"]) == revision


def test_a_trait_the_named_project_does_not_hold_refuses(project: Path, tmp_path: Path):
    with pytest.raises(TraitUnknownError):
        confirmed_revision(PER_IMAGE_COUNT, project=project, trait="not_here")
    with pytest.raises(TraitUnknownError):
        confirmed_revision(PER_IMAGE_COUNT, project=tmp_path / "empty", trait=fx.COUNT_TRAIT)


# ── the crossing delivery doors ──────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _series(tmp_path: Path, *, assessed: bool = True) -> dict:
    pytest.importorskip("torch")
    from tests._chain_fixtures import classified_series

    return classified_series(tmp_path, fractions=(0.0, 1.0), assessed=assessed).body()


def _compute(project: Path, body: dict, out_csv: Path) -> dict:
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones(
        project, trait=body["trait"], mapping_name=body["mapping_name"], plants=body["plants"],
        buckets=body["buckets"], output_csv_path=str(out_csv))


def _withdraw(project_root: Path, trait: str) -> None:
    """Withdraw the breeder's confirmation of the trait's latest confirmed revision."""
    revision = traits.read_trait(trait, project_root).latest_confirmed
    assert revision is not None
    traits.confirm_revision(project_root, trait, revision.number, revision.entry_sha256,
                            user="rosalind", confirmed=False)


def _web_refusal(client: TestClient, body: dict, route: str, headers=None, **extra) -> dict:
    sent = ({**body, "payload": "milestones", "filename": "x.csv", **extra}
            if route == "export_csv" else {**body, **extra})
    resp = client.post(f"/api/results/{route}", json=sent, headers=headers)
    assert resp.status_code == 400, (route, resp.status_code, resp.text)
    return resp.json()["detail"]


def _rows(text: str) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "delivery_event_id"}
            for row in csv.DictReader(StringIO(text))]


def test_unconfirmed_crossing_door_refuses(tmp_path: Path):
    """An entry nobody confirmed is the agent's own definition, so the tool door refuses on it and
    writes nothing."""
    body = _series(tmp_path)
    _withdraw(tmp_path, "bud_opening")
    out_csv = tmp_path / "delivered.csv"

    res = _compute(tmp_path, body, out_csv)

    assert "is confirmed by the breeder" in res["error"]
    assert "Setup tab" in res["error"]
    assert not out_csv.exists()


def test_both_web_doors_refuse_identically_and_an_acknowledgment_does_not_clear_it(
    client: TestClient, tmp_path: Path,
):
    """One check, one refusal body: a curve the breeder sees is never one Download refuses, and
    acknowledging an unvalidated measurement says nothing about whether one was defined."""
    body = _series(tmp_path)
    _withdraw(tmp_path, "bud_opening")

    details = [_web_refusal(client, body, route)
               for route in ("phenology_measurement", "export_csv")]
    from tests._web_fixtures import BROWSER

    acknowledged = _web_refusal(client, body, "export_csv", headers=BROWSER,
                                acknowledgment={"user": "breeder", "reason": "a look now",
                                                "result_sha256": "0" * 64})

    assert all(d == details[0] for d in details), details
    assert acknowledged == details[0]
    assert details[0]["kind"] == "operationalization"
    assert "bud_opening" in details[0]["message"]
    assert not (tmp_path / "results_export").exists()


def test_an_acknowledgment_clears_the_gate_once_the_meaning_is_confirmed(
    client: TestClient, tmp_path: Path,
):
    body = _series(tmp_path, assessed=False)

    from tests._web_fixtures import acknowledged_post

    resp = acknowledged_post(client, "/api/results/export_csv",
                             {**body, "payload": "milestones", "filename": "x.csv"},
                             reason="test acknowledgment")

    assert resp.status_code == 200, resp.text
    assert {row["validated"] for row in _rows(resp.text)} == {"False"}


def test_the_tool_and_the_web_export_write_the_same_rows(client: TestClient, tmp_path: Path):
    """The two doors onto the one delivery write the same rows, the web door's saved file
    included, apart from each delivery's own event id."""
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.postprocessing.phenology import phenology_csv_columns

    body = _series(tmp_path)
    out_csv = tmp_path / "delivered.csv"
    assert "error" not in _compute(tmp_path, body, out_csv)

    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})

    assert resp.status_code == 200, resp.text
    tool_text = out_csv.read_text(encoding="utf-8")
    assert _rows(resp.text) == _rows(tool_text)
    assert (tmp_path / "results_export" / "x.csv").read_bytes() == resp.content
    header = next(csv.reader(StringIO(tool_text)))
    assert header == phenology_csv_columns(latest_confirmed("bud_opening", tmp_path).entry)


def test_a_changed_majority_crossing_ships_only_once_confirmed_and_then_needs_assessing_again(
    tmp_path: Path,
):
    """The majority crossing is part of the entry: changing it proposes a revision, and the
    delivery keeps shipping the confirmed crossing until the breeder confirms the change; the
    assessment behind the buckets was judged under the old revision, so the confirmed change
    refuses until the checkpoint is assessed under it."""
    body = _series(tmp_path)
    shipped = traits.read_trait("bud_opening", tmp_path).latest.entry
    changed = fx.propose(tmp_path, fx.with_fields(shipped, majority_milestone="50per"))

    out = tmp_path / "before.csv"
    assert "error" not in _compute(tmp_path, body, out)
    row = next(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert row["bud_majority_date"] == row["bud_95per_date"]

    fx.confirm(tmp_path, changed)
    res = _compute(tmp_path, body, tmp_path / "after.csv")
    assert "assess again under the revision delivered" in res["error"]


def test_an_assessment_answers_for_its_own_revision_never_a_later_one_of_equal_content(
    tmp_path: Path,
):
    """A revision is named by its number and its content together: re-proposing the identical
    entry and confirming it makes a second revision whose content hash equals the first's, and
    the assessment judged under the first still answers for no delivery under the second."""
    body = _series(tmp_path)
    first = traits.read_trait("bud_opening", tmp_path).latest_confirmed
    assert first is not None
    again = fx.propose(tmp_path, first.entry)
    assert (again.number, again.entry_sha256) == (first.number + 1, first.entry_sha256)
    fx.confirm(tmp_path, again)

    res = _compute(tmp_path, body, tmp_path / "out.csv")

    assert "assess again under the revision delivered" in res["error"]
    assert not (tmp_path / "out.csv").exists()


def test_the_screen_door_still_honors_show_unvalidated_for_the_evidence_gate(
    client: TestClient, tmp_path: Path,
):
    """A confirmed meaning plus unassessed buckets still reaches the screen, marked provisional."""
    body = _series(tmp_path, assessed=False)

    resp = client.post("/api/results/phenology_measurement", json={**body, "show_unvalidated": True})

    assert resp.status_code == 200, resp.text
    assert resp.json()["validated"] is False
    assert "no assessment answers" in resp.json()["unvalidated_reason"]
    assert client.post("/api/results/phenology_measurement", json=body).status_code == 400


def test_an_unconfirmed_and_unassessed_delivery_reports_the_meaning_alone(
    client: TestClient, tmp_path: Path,
):
    """A number with no defined meaning has nothing for a reference to validate."""
    body = _series(tmp_path, assessed=False)
    _withdraw(tmp_path, "bud_opening")

    detail = _web_refusal(client, body, "phenology_measurement")

    assert detail["kind"] == "operationalization"
    assert "assessment" not in detail["message"]


# ── the count and aggregate delivery doors ───────────────────────────────────


@pytest.fixture
def delivery_root(tmp_path: Path) -> Path:
    """The project, carrying every trait the count and aggregate doors deliver under."""
    return fx.seed_delivery_traits(tmp_path)


def test_a_count_under_a_revision_stating_no_count_refuses(delivery_root: Path):
    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_IMAGE_COUNT, project=delivery_root, trait=fx.COUNT_TRAIT)

    assert "states no operationalization" in str(excinfo.value)
    assert PER_IMAGE_COUNT in str(excinfo.value)


def test_the_count_tool_hands_back_no_counts_when_it_refuses(delivery_root: Path, tmp_path: Path):
    """A count delivery whose trait states no count refuses before reading the bucket's
    documents, and returns the refusal alone, never numbers."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tests._chain_fixtures import predicted, published

    scope = {"subject": fx.COUNT_SUBJECT, "attribute": None, "id_map": {fx.COUNT_SUBJECT: 0}}
    bucket = published(tmp_path, tmp_path / "ds" / "predictions" / "m" / "d",
                       [predicted("a", [fx.COUNT_SUBJECT], scope["id_map"])], scope=scope).path

    res = deliver_per_image_counts(tmp_path, str(bucket), str(tmp_path / "o.csv"),
                                   trait=fx.COUNT_TRAIT)

    assert "states no operationalization" in res["error"]
    assert "image_count" not in res and "total_detections" not in res
    assert not (tmp_path / "o.csv").exists()


def test_a_phenotype_no_confirmed_trait_delivers_refuses_and_names_the_proposing_tool(
    delivery_root: Path, tmp_path: Path,
):
    """A trait that delivers the phenotype only in a revision nobody confirmed is named in the
    refusal."""
    fx.propose(tmp_path, fx.entry("nut", ("cluster_nut_count",)))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project=tmp_path,
                           delivered_phenotype="cluster_nut_count", value_keys=["count"])

    assert "0 traits with a confirmed revision deliver" in str(excinfo.value)
    assert "'nut'" in str(excinfo.value) and "cluster_nut_count" in str(excinfo.value)
    assert "propose_trait" in str(excinfo.value)


def test_a_phenotype_two_confirmed_traits_deliver_refuses_as_ambiguous(
    delivery_root: Path, tmp_path: Path,
):
    fx.propose_and_confirm(tmp_path, fx.with_fields(fx.COUNT_SPEC, name="second_deliverer"))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project=tmp_path,
                           delivered_phenotype="stem_count", value_keys=["count"])

    assert "2 traits with a confirmed revision deliver" in str(excinfo.value)
    assert "second_deliverer" in str(excinfo.value)


def test_a_revision_states_each_aggregate_kind_it_delivers(delivery_root: Path, tmp_path: Path):
    """One trait, two aggregate shapes: a revision stating only the ordinal one admits ordinal and
    refuses count; the revision that adds count, once confirmed, admits both."""
    fx.seed_confirmed_aggregate(tmp_path, "astringency", value_keys=["astringency"],
                                delivery_kind=PER_PLANT_ORDINAL_AGGREGATE)

    def admits(kind: str) -> traits.TraitRevision:
        return confirmed_revision(kind, project=tmp_path, delivered_phenotype="astringency",
                                  value_keys=["astringency"])

    admits(PER_PLANT_ORDINAL_AGGREGATE)
    with pytest.raises(OperationalizationRefused, match=PER_PLANT_COUNT_AGGREGATE):
        admits(PER_PLANT_COUNT_AGGREGATE)

    fx.seed_confirmed_aggregate(tmp_path, "astringency", value_keys=["astringency"])
    confirmed = admits(PER_PLANT_COUNT_AGGREGATE)
    assert admits(PER_PLANT_ORDINAL_AGGREGATE) == confirmed
    assert set(confirmed.entry.operationalizations) == {
        PER_PLANT_ORDINAL_AGGREGATE, PER_PLANT_COUNT_AGGREGATE}


def test_deliver_per_image_counts_refuses_with_no_trait_argument(tmp_path: Path):
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    with pytest.raises(TypeError, match="'trait'"):
        deliver_per_image_counts(tmp_path, str(tmp_path), str(tmp_path / "o.csv"))  # type: ignore[call-arg]  # the omission is the subject

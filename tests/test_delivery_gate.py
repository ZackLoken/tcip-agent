"""The one delivery gate and the acknowledgment that ships past it.

A bucket is validated only by an assessment that answers for the delivery; an unvalidated
delivery refuses, carrying the digest of the result it computed, and ships only under a breeder's
recorded acknowledgment of exactly that result, stamped unvalidated with the acknowledgment on its
record. The per-image count CSV carries exactly the columns the delivery skill documents, and
counts what the published document holds.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcip_mcp.delivery import DeliveryRefused, record_acknowledgment
from tests import _trait_fixtures as fx

SCOPE = {"subject": fx.COUNT_SUBJECT, "attribute": None, "id_map": {fx.COUNT_SUBJECT: 0}}


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    fx.seed_confirmed_count(tmp_path, measured_subject=fx.COUNT_SUBJECT)


def _bucket(project: Path, results: list[dict], name: str = "m") -> Path:
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    return published(project, project / "ds" / "predictions" / name / "2026-01-01", results,
                     scope=SCOPE).path


def _deliver(project: Path, bucket: Path, acknowledgment_id: str | None = None) -> dict:
    from tcip_mcp.pipelines.postprocessing.export import deliver_per_image_counts_csv

    return deliver_per_image_counts_csv(project, bucket, str(project / "out" / "counts.csv"),
                                        trait=fx.COUNT_TRAIT, acknowledgment_id=acknowledgment_id,
                                        door="test_door")


def _acknowledged(project: Path, bucket: Path) -> dict:
    from tests._chain_fixtures import acknowledged

    return acknowledged(project, lambda ack: _deliver(project, bucket, ack))


def _rows(result: dict) -> list[dict]:
    with open(result["csv_path"], newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("who, why", [("", "known limitation"), ("   ", "known limitation"),
                                      ("user:breeder", ""), ("user:breeder", "\t\n")])
def test_an_acknowledgment_names_who_and_why_both_non_empty(tmp_path, who, why):
    field = "acknowledged_by" if not who.strip() else "reason"
    with pytest.raises(ValueError, match=field):
        record_acknowledgment(tmp_path, acknowledged_by=who, reason=why, result_sha256="digest")


def test_an_unassessed_bucket_refuses_with_no_acknowledgment_and_writes_nothing(tmp_path):
    from tests._chain_fixtures import predicted

    bucket = _bucket(tmp_path, [predicted("a", [fx.COUNT_SUBJECT], SCOPE["id_map"])])

    with pytest.raises(DeliveryRefused, match="no assessment answers") as refused:
        _deliver(tmp_path, bucket)

    assert refused.value.result_sha256 is not None
    assert not (tmp_path / "out" / "counts.csv").exists()


def test_an_acknowledged_delivery_ships_stamped_unvalidated_with_the_act_on_its_record(tmp_path):
    from tcip_mcp.delivery import read_delivery_events
    from tests._chain_fixtures import acknowledged, predicted

    bucket = _bucket(tmp_path, [predicted("a", [fx.COUNT_SUBJECT], SCOPE["id_map"])])

    result = acknowledged(tmp_path, lambda ack: _deliver(tmp_path, bucket, ack),
                          reason="a look before assessing")

    assert result["validated"] is False
    assert result["acknowledged_by"] == "user:breeder"
    assert {r["validated"] for r in _rows(result)} == {"False"}
    (event,) = read_delivery_events(tmp_path)
    assert event.acknowledgment is not None
    assert (event.acknowledgment.acknowledged_by, event.acknowledgment.reason) == (
        "user:breeder", "a look before assessing")
    assert event.buckets[0].validated is False
    assert "no assessment answers" in str(event.buckets[0].reason)


def test_an_acknowledgment_given_for_another_result_refuses(tmp_path):
    """An acknowledgment binds the one result it was given for: an act recorded against another
    result's digest ships nothing, and the refusal names the result this delivery computed."""
    from tests._chain_fixtures import predicted

    bucket = _bucket(tmp_path, [predicted("a", [fx.COUNT_SUBJECT], SCOPE["id_map"])])
    elsewhere = _bucket(tmp_path, [predicted("b", [fx.COUNT_SUBJECT], SCOPE["id_map"])], "n")
    with pytest.raises(DeliveryRefused) as other:
        _deliver(tmp_path, elsewhere)
    act = record_acknowledgment(tmp_path, acknowledged_by="user:breeder", reason="the other one",
                                result_sha256=str(other.value.result_sha256))

    with pytest.raises(DeliveryRefused, match="another result") as refused:
        _deliver(tmp_path, bucket, act.acknowledgment_id)

    assert refused.value.result_sha256 not in (None, act.result_sha256)
    assert not (tmp_path / "out" / "counts.csv").exists()


def test_an_acknowledgment_binds_the_rows_and_refuses_once_a_document_changes(tmp_path):
    """An act recorded for six detections never ships twelve: the bucket, its producer and the
    gate's finding are the same after the document changes, and the rows it would write are not."""
    from tcip_annotation import json_io

    from tcip_mcp.audit import audit_log_key
    from tests._chain_fixtures import predicted

    import tcip_store

    bucket = _bucket(tmp_path, [predicted("a", [fx.COUNT_SUBJECT] * 6, SCOPE["id_map"])])
    with pytest.raises(DeliveryRefused) as six:
        _deliver(tmp_path, bucket)
    act = record_acknowledgment(tmp_path, acknowledged_by="user:breeder", reason="six of them",
                                result_sha256=str(six.value.result_sha256))
    document = bucket / "a.json"
    annotations = json_io.read_annotations(document)
    # Twelve where six were published, as an edit in place would leave it: no head re-emits one.
    json_io.write_annotations(document, annotations * 2, 64, 64)

    with pytest.raises(DeliveryRefused, match="another result"):
        _deliver(tmp_path, bucket, act.acknowledgment_id)

    assert not (tmp_path / "out" / "counts.csv").exists()
    assert [(e["tool"], e["arguments"]["acknowledgment_id"])
            for e in tcip_store.read_log(audit_log_key(tmp_path)).records
            if e["tool"] == "delivery_acknowledged"] == [("delivery_acknowledged",
                                                          act.acknowledgment_id)]


def test_an_mcp_door_executes_a_recorded_acknowledgment_and_never_originates_one(tmp_path):
    """The agent's door takes a recorded act by its id and nothing a caller could compose an act
    from: a name and a reason are no parameters of it, and an id nothing recorded refuses."""
    import inspect

    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tests._chain_fixtures import predicted

    assert not {"acknowledged_by", "reason", "acknowledgment"} & set(
        inspect.signature(deliver_per_image_counts).parameters)
    bucket = _bucket(tmp_path, [predicted("a", [fx.COUNT_SUBJECT], SCOPE["id_map"])])

    refused = deliver_per_image_counts(tmp_path, str(bucket), "out/counts.csv",
                                       trait=fx.COUNT_TRAIT, acknowledgment_id="invented")

    assert "no acknowledgment 'invented' is recorded" in refused["error"]
    with pytest.raises(DeliveryRefused) as computed:
        _deliver(tmp_path, bucket)
    act = record_acknowledgment(tmp_path, acknowledged_by="user:breeder", reason="a look",
                                result_sha256=str(computed.value.result_sha256))
    shipped = deliver_per_image_counts(tmp_path, str(bucket), "out/counts.csv",
                                       trait=fx.COUNT_TRAIT, acknowledgment_id=act.acknowledgment_id)
    assert shipped["validated"] is False and shipped["acknowledged_by"] == "user:breeder"


def test_a_validated_delivery_carries_no_acknowledgment(tmp_path):
    from tcip_mcp.delivery import read_delivery_events
    from tests._chain_fixtures import run_the_chain

    chain = run_the_chain(tmp_path, experiment_id="exp-ack")

    result = _deliver(tmp_path, chain.bucket)

    assert result["validated"] is True
    assert result["acknowledged_by"] is None
    assert read_delivery_events(tmp_path)[0].acknowledgment is None


def test_the_delivery_skill_documents_the_per_image_csv_the_door_writes(tmp_path):
    """The delivery skill's Per-Image CSV Schema table is the schema the door actually writes."""
    from tcip_mcp.knowledge import document_path
    from tests._chain_fixtures import predicted

    bucket = _bucket(tmp_path, [predicted("a", [fx.COUNT_SUBJECT], SCOPE["id_map"])])
    result = _acknowledged(tmp_path, bucket)
    with open(result["csv_path"], newline="") as f:
        written = next(csv.reader(f))

    lines = document_path("delivery").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("### Per-Image CSV Schema"))
    documented = []
    for ln in lines[start + 1:]:
        if ln.startswith("#"):
            break
        if ln.startswith("|") and not ln.startswith("|---") and not ln.startswith("| Column"):
            documented.append(ln.split("|")[1].strip())

    assert documented == written


def test_a_zero_extent_box_is_counted_by_neither_the_document_nor_the_delivery(tmp_path):
    """A box that collapses to zero width is never a detection: the published document drops it,
    the bucket records the drop, and the delivered row counts and averages the one survivor."""
    bucket_dir = _bucket(tmp_path, [{
        "image": "a.png", "width": 200, "height": 150,
        "boxes": [[10, 10, 20, 20], [30, 30, 30, 40]], "scores": [0.9, 0.5], "labels": [1, 1]}])
    from tcip_mcp.buckets import read_bucket

    assert read_bucket(bucket_dir).dropped_boxes == 1
    (row,) = _rows(_acknowledged(tmp_path, bucket_dir))

    assert int(row["detection_count"]) == 1
    assert float(row["avg_confidence"]) == pytest.approx(0.9)


def test_a_non_finite_score_refuses_the_publication_so_no_average_ever_reads_it(tmp_path):
    """No stored number stands in for a score the model did not give: the publication refuses
    naming it, and the directory never becomes a bucket a delivery could average over."""
    from tcip_annotation.json_io import BUCKET_RECORD

    with pytest.raises(ValueError, match="nan"):
        _bucket(tmp_path, [{
            "image": "a.png", "width": 64, "height": 64,
            "boxes": [[1, 1, 5, 5], [10, 10, 15, 15]], "scores": [float("nan"), 0.5],
            "labels": [1, 1]}])

    assert not (tmp_path / "ds" / "predictions" / "m" / "2026-01-01" / BUCKET_RECORD).exists()


def test_a_delivery_over_no_bucket_refuses_and_carries_no_result_to_acknowledge(tmp_path):
    """Nothing states what a bucketless delivery's numbers were measured from, so no
    acknowledgment can ship one; every other case here delivers over one bucket and clears."""
    from tcip_mcp.delivery import Result, gate
    from tcip_mcp.traits import PER_IMAGE_COUNT

    with pytest.raises(DeliveryRefused, match="names no prediction bucket") as refused:
        gate(tmp_path, [], delivery_kind=PER_IMAGE_COUNT, revision=fx.count_revision(tmp_path),
             result=Result((), (), population=()))
    assert refused.value.result_sha256 is None


def test_a_per_plant_count_over_a_bucket_counting_another_subject_refuses(tmp_path):
    """The bucket-against-measured-subject check is the gate's, so a per-plant count meets it as
    a per-image count does: a bucket of another subject refuses, one of the measured subject
    ships."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.traits import PER_PLANT_COUNT_AGGREGATE
    from tests._chain_fixtures import deliver_acknowledged, predicted, published

    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    rows = [{"plant_id": "p1", "value": 1.0, "observations": 1, "value_key": "count",
             "plant_attribution": "image"}]

    def deliver(subject: str, name: str) -> dict:
        pytest.importorskip("torch")
        bucket = published(tmp_path, tmp_path / "ds" / "predictions" / name / "2026-01-01",
                           [predicted("a", [subject], {subject: 0})],
                           scope={"subject": subject, "attribute": None, "id_map": {subject: 0}})
        return deliver_acknowledged(
            tmp_path, rows, tmp_path / f"{name}.csv", "stem_count",
            delivery_kind=PER_PLANT_COUNT_AGGREGATE, buckets=[read_bucket(bucket.path)])

    with pytest.raises(OperationalizationRefused, match=f"measures '{fx.COUNT_SUBJECT}'"):
        deliver("leaf", "other")
    assert not (tmp_path / "other.csv").exists()
    assert deliver(fx.COUNT_SUBJECT, "measured")["validated"] is False

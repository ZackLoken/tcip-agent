"""The check every delivery door runs: the trait's latest confirmed revision, and whether its
operationalization binds what the door is about to deliver.

A delivered phenotype without a breeder-confirmed meaning is a number nobody defined. These cases
pin each refusal, the order in which they report, and the calls that must still succeed, because
a rail that only rejects is a rail that has not been shown to admit valid work.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp import traits
from tcip_mcp.operationalization import OperationalizationRefused, bind, confirmed_revision
from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild
from tcip_mcp.traits import (
    PER_IMAGE_COUNT,
    PER_PLANT_COUNT_AGGREGATE,
    PER_PLANT_ORDINAL_AGGREGATE,
    STATE_CROSSING_DATES,
    TraitUnknownError,
)
from tcip_web.app import app
from tests import _trait_fixtures as fx
from tests._binding_fixtures import complete_stamp, producer_checkpoint_sha256
from tests._population import mapped_plants
from tests.test_tcip_web_results_routes import _expected_validation_record, _phenology_fixture

# A writer-level unit test's own placeholder disclosure, built through delivery_disclosure itself
# so it carries every key the writer's cells read even as that shape grows.
_NO_MAPPING = MappingBuild(
    name="none", project_root="", dataset_root="", dataset_id="", built_by="test", built_at="",
    dates_requested=None, dates=[], nn_tolerance_m={"value": 0.0, "source": "stated"},
    plant_registry={"name": "unregistered", "digest": "0" * 64},
    capture_identity={}, capture_digests={}, unreadable={}, assignments={},
    record_sha256="0" * 16,
).delivery_disclosure({"captures_unverified": [], "plant_csvs_unverified": []}, [])


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
        confirmed_revision(STATE_CROSSING_DATES, project_root=project, trait=fx.CROSSING_TRAIT)

    message = str(excinfo.value)
    assert "states no operationalization for a state_crossing_dates delivery" in message
    assert "propose_trait" in message and "Setup tab" in message
    assert "bloom_05per_date" in message and "Date when 5%" in message


def test_a_trait_with_no_confirmed_revision_refuses_and_names_who_confirms(tmp_path: Path):
    fx.propose(tmp_path, fx.with_operationalization(
        fx.COUNT_SPEC, PER_IMAGE_COUNT, measured_subject=fx.COUNT_SUBJECT))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_IMAGE_COUNT, project_root=tmp_path, trait=fx.COUNT_TRAIT)

    assert "none of its 1 revision(s) is confirmed" in str(excinfo.value)
    assert excinfo.value.as_detail() == {"kind": "operationalization", "message": str(excinfo.value)}


def test_a_value_key_outside_the_confirmed_set_refuses(project: Path):
    _confirmed_aggregate(project, PER_PLANT_COUNT_AGGREGATE)

    with pytest.raises(OperationalizationRefused, match="leaf_length"):
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project_root=project, trait=fx.COUNT_TRAIT,
                           value_keys=["stem_count", "leaf_length"])


def test_a_row_carrying_no_value_key_refuses_and_counts_them(project: Path):
    _confirmed_aggregate(project, PER_PLANT_COUNT_AGGREGATE)

    with pytest.raises(OperationalizationRefused, match="2 row"):
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project_root=project, trait=fx.COUNT_TRAIT,
                           value_keys=["stem_count", "", None])


def test_a_delivered_phenotype_outside_the_confirmed_set_refuses(project: Path):
    _confirmed_crossing(project)

    with pytest.raises(OperationalizationRefused, match="bloom_95per_date"):
        confirmed_revision(STATE_CROSSING_DATES, project_root=project, trait=fx.CROSSING_TRAIT,
                           delivered_phenotype="bloom_95per_date")


def test_a_bucket_not_counting_the_measured_subject_refuses(project: Path):
    revision = fx.seed_confirmed_count(project)

    with pytest.raises(OperationalizationRefused) as excinfo:
        bind(revision, PER_IMAGE_COUNT,
             buckets={"predictions/live/2026-03-04": (fx.COUNT_TRAIT, {"leaf"})})

    assert "stem" in str(excinfo.value) and "predictions/live/2026-03-04" in str(excinfo.value)


def test_a_registry_no_longer_declaring_the_positive_class_refuses_with_its_problem(
    project: Path,
):
    from tcip_mcp import subject_registry as cr

    _confirmed_crossing(project)
    registry = cr.SubjectRegistry(subjects=(cr.Subject(name="flower", attributes=(
        cr.Attribute(name="state", type="categorical", values=("shed",)),)),))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(STATE_CROSSING_DATES, project_root=project, trait=fx.CROSSING_TRAIT,
                           registry=registry)

    problem = cr.positive_value_problem(registry, "flower", "open")
    assert problem is not None and problem in excinfo.value.as_detail()["message"]


# ── ordering ─────────────────────────────────────────────────────────────────


def test_an_unconfirmed_trait_reports_that_rather_than_a_binding(tmp_path: Path):
    fx.propose(tmp_path, fx.with_operationalization(
        fx.COUNT_SPEC, PER_PLANT_COUNT_AGGREGATE, delivered_phenotypes=("stem_count",),
        delivered_value_keys=("stem_count",)))

    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_PLANT_COUNT_AGGREGATE, project_root=tmp_path, trait=fx.COUNT_TRAIT,
                           value_keys=["not_covered"])

    assert "is confirmed" in str(excinfo.value) and "not_covered" not in str(excinfo.value)


# ── what the rail admits ─────────────────────────────────────────────────────


def test_a_confirmed_revision_passes_every_binding_it_covers(project: Path):
    crossing = _confirmed_crossing(project)

    assert confirmed_revision(
        STATE_CROSSING_DATES, project_root=project, trait=fx.CROSSING_TRAIT,
        delivered_phenotype="bloom_50per_date") == crossing


@pytest.mark.parametrize(
    "delivery_kind",
    [PER_PLANT_COUNT_AGGREGATE, PER_PLANT_ORDINAL_AGGREGATE, traits.PER_PLANT_REGRESSION_AGGREGATE],
)
def test_an_aggregate_revision_admits_its_named_value_keys(project: Path, delivery_kind: str):
    revision = _confirmed_aggregate(project, delivery_kind)

    assert confirmed_revision(
        delivery_kind, project_root=project, trait=fx.COUNT_TRAIT,
        delivered_phenotype="stem_count", value_keys=["stem_count"]) == revision


def test_the_revision_is_read_from_the_project_the_caller_names(
    project: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The pinned platform root is somewhere else entirely, and the revision still resolves."""
    crossing = _confirmed_crossing(project)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(project.parent / "unrelated"))

    assert confirmed_revision(
        STATE_CROSSING_DATES, project_root=project, trait=fx.CROSSING_TRAIT) == crossing


def test_a_trait_the_named_project_does_not_hold_refuses(project: Path, tmp_path: Path):
    with pytest.raises(TraitUnknownError):
        confirmed_revision(PER_IMAGE_COUNT, project_root=project, trait="not_here")
    with pytest.raises(TraitUnknownError):
        confirmed_revision(PER_IMAGE_COUNT, project_root=tmp_path / "empty", trait=fx.COUNT_TRAIT)


# ── the crossing delivery doors ──────────────────────────────────────────────


def _extract_produced_at(written: bytes) -> bytes:
    """A just-written delivery's own tail write-time cell, read back rather than predicted."""
    import csv as _csv

    rows = list(_csv.DictReader(written.decode().splitlines()))
    return rows[0]["produced_at"].encode()


def delivered_golden(body: dict, produced_at: bytes) -> bytes:
    """What a confirmed crossing delivery writes, byte for byte, for the golden inputs below.

    ``produced_at`` is the tail composition's own write-time stamp, read back from the delivery
    just written; ``validation_record`` and ``plant_mapping_sha256`` are read from the buckets'
    stamps and the mapping itself, since both digest this run's own temporary paths.
    """
    from tcip_mcp.pipelines.postprocessing import plant_mapping as pm

    build = pm.load_mapping(Path(body["project_root"]), body["mapping_name"])
    assert build is not None
    mapping_sha = build.record_sha256.encode()
    dates_delivered = ";".join(build.dates).encode()

    record = _expected_validation_record(body).encode()
    sha = producer_checkpoint_sha256("exp-1").encode()
    row = (b",2,2,0,0,2026-02-24,2026-02-12,2026-02-18,2026-02-24,interpolated,interpolated,"
           b"interpolated,interpolated,0.4,held_out_annotations,held_out_annotations,,"
           + sha + b",exp-1," + produced_at + b"," + record + b"," + mapping_sha + b","
           + dates_delivered + b",0,image,,\r\n")
    return (
        b"plant_id,accession,n_dates,n_observed_dates,n_dates_unclassified,n_dates_missing_images,"
        b"bud_majority_date,bud_05per_date,bud_50per_date,bud_95per_date,"
        b"bud_majority_date_bound,bud_05per_date_bound,bud_50per_date_bound,"
        b"bud_95per_date_bound,operating_point_conf,"
        b"operating_point_validated,positive_state_classifier_validated,unvalidated_dimensions,"
        b"producer_model_sha256,"
        b"producing_experiment_id,produced_at,validation_record,plant_mapping_sha256,"
        b"dates_delivered,images_unattributed,"
        b"plant_attribution,acknowledged_by,acknowledgment_reason\r\n"
        + b"PLANT_A,AccA" + row
        + b"PLANT_B,AccB" + row
    )


GOLDEN_INPUTS = {"fractions": (0.0, 1.0), "detections": 2}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _compute(body: dict, out_csv: Path, **kwargs) -> dict:
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    return deliver_phenology_milestones(
        trait=body["trait"],
        mapping_name=body["mapping_name"], plants=mapped_plants(body["mapping_name"]),
        predictions_by_date=body["predictions_by_date"],
        output_csv_path=str(out_csv),
        classifier_pred_dirs=list(body["predictions_by_date"].values()),
        **kwargs,
    )


def _withdraw(project_root: Path, trait: str) -> None:
    """Withdraw the breeder's confirmation of the trait's latest confirmed revision."""
    revision = traits.read_trait(trait, project_root).latest_confirmed
    assert revision is not None
    traits.confirm_revision(project_root, trait, revision.number, revision.entry_sha256,
                            user="rosalind", identity_from_request=True, confirmed=False)


def _web_refusal(client: TestClient, body: dict, route: str) -> dict:
    sent = {**body, "payload": "milestones", "filename": "x.csv"} if route == "export_csv" else body
    resp = client.post(f"/api/results/{route}", json=sent)
    assert resp.status_code == 400, (route, resp.status_code, resp.text)
    return resp.json()["detail"]


def test_unconfirmed_crossing_door_refuses(tmp_path: Path):
    """An entry nobody confirmed is the agent's own definition, so the tool door refuses on it and
    writes nothing."""
    body = _phenology_fixture(tmp_path, validated=True)
    _withdraw(tmp_path, "bud_opening")
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv)

    assert "is confirmed by the breeder" in res["error"]
    assert "Setup tab" in res["error"]
    assert not out_csv.exists()


def test_both_web_doors_refuse_identically(client: TestClient, tmp_path: Path):
    """One check, one refusal body: a curve the breeder sees is never one Download refuses."""
    body = _phenology_fixture(tmp_path, validated=True)
    _withdraw(tmp_path, "bud_opening")

    details = [_web_refusal(client, body, route)
               for route in ("phenology_measurement", "export_csv")]

    assert all(d == details[0] for d in details), details
    assert details[0]["kind"] == "operationalization"
    assert "bud_opening" in details[0]["message"]
    assert not (tmp_path / "results_export").exists()


def test_acknowledge_does_not_clear_the_meaning_check_at_every_door(
    client: TestClient, tmp_path: Path,
):
    """Acknowledging an unvalidated measurement says nothing about whether one was defined."""
    body = _phenology_fixture(tmp_path, validated=True)
    _withdraw(tmp_path, "bud_opening")
    out_csv = tmp_path / "delivered.csv"

    with pytest.raises(TypeError, match="acknowledge_unvalidated"):
        _compute(body, out_csv, acknowledge_unvalidated=True)
    assert not out_csv.exists()

    for route in ("phenology_measurement", "export_csv"):
        assert _web_refusal(client, body, route)["kind"] == "operationalization", route


def test_acknowledge_still_clears_the_gate_dimensions_in_the_same_call(
    client: TestClient, tmp_path: Path,
):
    """With the meaning confirmed, the unvalidated evidence ships stamped false through the web
    export route, the one surface that builds a real acknowledgment."""
    body = _phenology_fixture(tmp_path, validated=False)

    resp = client.post("/api/results/export_csv", json={
        **body, "payload": "milestones", "filename": "x.csv", "user": "user:tester",
        "acknowledgment": {"reason": "test acknowledgment"},
    })

    assert resp.status_code == 200, resp.text
    header = resp.text.splitlines()[0].split(",")
    cells = dict(zip(header, resp.text.splitlines()[1].split(",")))
    assert cells["positive_state_classifier_validated"] == "false"


def test_a_confirmed_delivery_writes_the_golden_bytes(tmp_path: Path):
    """The tool door's delivered CSV, asserted as bytes rather than as the absence of an error."""
    body = _phenology_fixture(tmp_path, validated=True, **GOLDEN_INPUTS)
    out_csv = tmp_path / "delivered.csv"

    res = _compute(body, out_csv)

    assert "error" not in res, res
    written = out_csv.read_bytes()
    assert written == delivered_golden(body, _extract_produced_at(written))


def test_the_web_export_door_writes_the_golden_bytes(client: TestClient, tmp_path: Path):
    """The other door onto the same writer, byte for byte, reading its own saved file back."""
    body = _phenology_fixture(tmp_path, validated=True, **GOLDEN_INPUTS)

    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})

    assert resp.status_code == 200, resp.text
    golden = delivered_golden(body, _extract_produced_at(resp.content))
    assert resp.content == golden
    assert (tmp_path / "results_export" / "x.csv").read_bytes() == golden


def test_write_phenology_csv_writes_the_confirmed_revisions_schema(tmp_path: Path):
    """The writer admits the call it was built for: a confirmed revision and cleared flags from a
    real reconciliation over the fixture's own validated buckets."""
    from tcip_mcp.pipelines.postprocessing import phenology
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity, reconcile_classifier_validity, reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )

    body = _phenology_fixture(tmp_path, validated=True)
    revision = confirmed_revision(STATE_CROSSING_DATES, project_root=tmp_path, trait="bud_opening")
    pred_dirs = list(body["predictions_by_date"].values())
    recon = reconcile_operating_point_validity(pred_dirs, trait="bud_opening")
    classifier_recon = reconcile_classifier_validity(pred_dirs)
    classifier_state, note = bind_classifier_validity(
        classifier_recon["validated"], pred_dirs, pred_dirs, trait="bud_opening")
    tile_recon = reconcile_tile_size_validity(pred_dirs)
    flags = phenology.phenology_delivery_flags(classifier_state, recon["validated"], tile_recon)
    row = {"plant_id": "P1", "accession": "acc-9", "n_dates": 2, "n_observed_dates": 2}

    phenology.write_phenology_csv(
        "test", [row], tmp_path / "out.csv", revision, flags=flags, acknowledgment=None,
        document_reconciliations={
            "operating_point": recon,
            "classifier_operating_point": {
                **classifier_recon, "bound_validated": classifier_state, "delivery_note": note,
            },
        },
        producer={}, dimension_reconciliations={"tile_size": tile_recon},
        predictions_by_date=body["predictions_by_date"], project_root=tmp_path,
        plant_mapping=_NO_MAPPING)

    header = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == phenology.phenology_csv_columns(revision.entry)


def test_a_changed_majority_crossing_ships_only_once_its_revision_is_confirmed(tmp_path: Path):
    """The majority crossing is part of the entry: changing it proposes a revision, and the
    delivery keeps shipping the confirmed crossing until the breeder confirms the change."""
    import csv as _csv

    body = _phenology_fixture(tmp_path, validated=True, **GOLDEN_INPUTS)
    shipped = traits.read_trait("bud_opening", tmp_path).latest.entry
    changed = fx.propose(tmp_path, fx.with_fields(shipped, majority_milestone="50per"))

    def majority(out: Path) -> tuple[str, str]:
        assert "error" not in _compute(body, out)
        row = next(_csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        return row["bud_majority_date"], row["bud_95per_date"]

    before_majority, before_95 = majority(tmp_path / "before.csv")
    assert before_majority == before_95

    fx.confirm(tmp_path, changed)
    after_majority, after_95 = majority(tmp_path / "after.csv")
    assert after_majority != after_95


def test_the_screen_door_still_honors_show_unvalidated_for_the_evidence_gate(
    client: TestClient, tmp_path: Path,
):
    """A confirmed meaning plus unvalidated evidence still reaches the screen, marked provisional."""
    body = _phenology_fixture(tmp_path, validated=False)

    resp = client.post(
        "/api/results/phenology_measurement", json={**body, "show_unvalidated": True})
    assert resp.status_code == 200, resp.text
    assert resp.json()["has_unvalidated_dimensions"] is True
    assert client.post(
        "/api/results/phenology_measurement", json=body).status_code == 400


def test_an_unconfirmed_and_gate_unvalidated_delivery_reports_the_meaning_alone(
    client: TestClient, tmp_path: Path,
):
    """A number with no defined meaning has nothing for a reference to validate."""
    body = _phenology_fixture(tmp_path, validated=False)
    _withdraw(tmp_path, "bud_opening")

    detail = _web_refusal(client, body, "phenology_measurement")

    assert detail["kind"] == "operationalization"
    assert "validated classifier and count operating point" not in detail["message"]


def test_a_confirmed_delivery_with_an_unbound_classifier_stamp_reports_that_refusal(
    client: TestClient, tmp_path: Path,
):
    """With the meaning confirmed, the refusal families behind the check report unchanged."""
    from tests.test_tcip_web_results_routes import _rewrite_classifier_sidecars

    body = _phenology_fixture(tmp_path, validated=True)
    _rewrite_classifier_sidecars(body, trait="chestnut_bur")

    resp = client.post("/api/results/export_csv",
                       json={**body, "payload": "milestones", "filename": "x.csv"})

    assert resp.status_code == 400
    assert "was earned for trait" in resp.json()["detail"]
    assert "chestnut_bur" in resp.json()["detail"]


# ── the count and aggregate delivery doors ───────────────────────────────────


@pytest.fixture
def delivery_root(tmp_path: Path) -> Path:
    """The pinned platform root, carrying every trait the count and aggregate doors deliver under."""
    return fx.seed_delivery_traits(tmp_path)


def _count_rows() -> list[dict]:
    return [{"image": "a.jpg", "count": 3, "scores": [0.9]}]


def _aggregate_rows(
    value_key: str | None = "count", *, measurement_document: str = "operating_point"
) -> list[dict]:
    row = {"plant_id": "p1", "value": 5, "observations": 2,
           "measurement_document": measurement_document, "plant_attribution": "image"}
    return [row if value_key is None else {**row, "value_key": value_key}]


def _bucket_recording(tmp_path: Path, id_map: dict) -> str:
    """A prediction bucket whose sidecar records which names its labels decoded to."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    bucket = tmp_path / "ds" / "predictions" / "run"
    bucket.mkdir(parents=True)
    subject = next(iter(id_map)) if len(id_map) == 1 else None
    write_sidecar(bucket, complete_stamp({"validated": False, "scope": {
        "subject": subject, "attribute": None, "id_map": id_map}}), "operating_point")
    return str(bucket)


def _validated_bucket(
    tmp_path: Path, name: str, *, trait: str, document: str = "operating_point",
    param_key: str = "conf", id_map: dict | None = None,
) -> str:
    """A prediction bucket genuinely bound to a validation record."""
    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT
    from tests._binding_fixtures import write_bound_sidecar, write_prediction

    root = tmp_path / "ds"
    bucket = root / "predictions" / name
    write_prediction(bucket, "img_a")
    subject = next(iter(id_map)) if id_map else trait
    stamp: dict = {
        "validated": True, "trait": trait,
        "operating_point": {param_key: {"value": 0.4, "requires_validation": True,
                                        "validation_kind": "annotations",
                                        "validated_against": VALIDATED_HELD_OUT}},
        "scope": {"subject": subject, "attribute": None, "id_map": id_map},
    }
    write_bound_sidecar(bucket, stamp, document=document, dataset_root=root)
    return str(bucket)


def test_a_count_under_a_revision_stating_no_count_refuses(delivery_root: Path):
    with pytest.raises(OperationalizationRefused) as excinfo:
        confirmed_revision(PER_IMAGE_COUNT, project_root=delivery_root, trait=fx.COUNT_TRAIT)

    assert "states no operationalization" in str(excinfo.value)
    assert PER_IMAGE_COUNT in str(excinfo.value)


def test_the_count_tool_hands_back_no_counts_when_it_refuses(
    delivery_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A count delivery whose trait states no count refuses before its inference pass runs, and
    returns the refusal alone, never numbers."""
    import tcip_mcp.tools.inference_tools as itools
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    ckpt = foreign_checkpoint(tmp_path)
    ran: list[bool] = []

    def inference(*args, **kwargs) -> dict:
        ran.append(True)
        return {"results": [{"image": "a.png", "count": 3, "scores": [0.9]}], "image_count": 1,
                "total_detections": 3, "operating_point": {"conf": {"value": 0.5}},
                "validated": False, "conf_source": "default", "checkpoint_sha256": "0" * 64,
                "experiment_id": None}

    monkeypatch.setattr(itools, "_run_inference_verified", inference)

    res = itools.deliver_per_image_counts(ckpt, str(tmp_path), str(tmp_path / "o.csv"),
                                          trait=fx.COUNT_TRAIT)

    assert ran == []
    assert "states no operationalization" in res["error"]
    assert "image_count" not in res and "total_detections" not in res
    assert not (tmp_path / "o.csv").exists()


def test_measured_subject_absent_from_id_maps_refuses(delivery_root: Path, tmp_path: Path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    revision = fx.seed_confirmed_count(tmp_path)
    bucket = _bucket_recording(tmp_path, {"leaf": 0})
    out_csv = tmp_path / "counts.csv"

    with pytest.raises(OperationalizationRefused) as excinfo:
        export_detection_csv(_count_rows(), str(out_csv), revision=revision, pred_dirs=[bucket])

    assert fx.COUNT_SUBJECT in str(excinfo.value)
    assert bucket in str(excinfo.value)
    assert not out_csv.exists()


def test_every_bucket_must_count_the_measured_subject(delivery_root: Path, tmp_path: Path):
    """A bucket recording the subject delivers; one recording another subject, or none, refuses
    alone and beside a matching one."""
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    revision = fx.seed_confirmed_count(tmp_path)
    matching = _validated_bucket(tmp_path, "recorded", trait=fx.COUNT_TRAIT,
                                 id_map={fx.COUNT_SUBJECT: 0})
    other = _validated_bucket(tmp_path, "other", trait=fx.COUNT_TRAIT, id_map={"leaf": 0})
    unscoped = tmp_path / "ds" / "predictions" / "unscoped"
    unscoped.mkdir(parents=True)

    export_detection_csv(_count_rows(), str(tmp_path / "recorded.csv"), revision=revision,
                         pred_dirs=[matching])
    assert (tmp_path / "recorded.csv").exists()
    for name, buckets in (("mixed.csv", [matching, other]), ("unscoped.csv", [str(unscoped)])):
        with pytest.raises(OperationalizationRefused):
            export_detection_csv(_count_rows(), str(tmp_path / name), revision=revision,
                                 pred_dirs=buckets)
        assert not (tmp_path / name).exists(), name


def test_row_without_value_key_refuses(delivery_root: Path, tmp_path: Path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    out_csv = tmp_path / "agg.csv"

    with pytest.raises(OperationalizationRefused, match="1 row"):
        export_aggregated_csv(_aggregate_rows(None), str(out_csv), delivered_phenotype="stem_count")

    assert not out_csv.exists()


def test_value_key_outside_confirmed_set_refuses(delivery_root: Path, tmp_path: Path):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])
    out_csv = tmp_path / "agg.csv"

    with pytest.raises(OperationalizationRefused, match="leaf_length"):
        export_aggregated_csv(_aggregate_rows("leaf_length"), str(out_csv),
                              delivered_phenotype="stem_count")

    assert not out_csv.exists()


def test_a_phenotype_no_confirmed_trait_delivers_refuses_and_names_the_proposing_tool(
    delivery_root: Path, tmp_path: Path,
):
    """A trait that delivers the phenotype only in a revision nobody confirmed is named in the
    refusal."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.propose(tmp_path, fx.entry("nut", ("cluster_nut_count",)))
    out_csv = tmp_path / "agg.csv"
    with pytest.raises(OperationalizationRefused) as excinfo:
        export_aggregated_csv(_aggregate_rows(), str(out_csv), delivered_phenotype="cluster_nut_count")

    assert "0 traits with a confirmed revision deliver" in str(excinfo.value)
    assert "'nut'" in str(excinfo.value) and "cluster_nut_count" in str(excinfo.value)
    assert "propose_trait" in str(excinfo.value)
    assert not out_csv.exists()


def test_a_phenotype_two_confirmed_traits_deliver_refuses_as_ambiguous(
    delivery_root: Path, tmp_path: Path,
):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.propose_and_confirm(tmp_path, fx.with_fields(fx.COUNT_SPEC, name="second_deliverer"))
    out_csv = tmp_path / "agg.csv"

    with pytest.raises(OperationalizationRefused) as excinfo:
        export_aggregated_csv(_aggregate_rows(), str(out_csv), delivered_phenotype="stem_count")

    assert "2 traits with a confirmed revision deliver" in str(excinfo.value)
    assert "second_deliverer" in str(excinfo.value)
    assert not out_csv.exists()


def test_a_revision_states_each_aggregate_kind_it_delivers(delivery_root: Path, tmp_path: Path):
    """One trait, two aggregate shapes: a revision stating only the ordinal one delivers ordinal
    and refuses count; the revision that adds count, once confirmed, delivers both."""
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    fx.seed_confirmed_aggregate(tmp_path, "astringency", value_keys=["astringency"],
                                measurement_document="ordinal_operating_point")
    ordinal_bucket = _validated_bucket(tmp_path, "ordinal", trait="astringency",
                                       document="ordinal_operating_point", param_key="ordinal")
    ordinal_rows = _aggregate_rows("astringency", measurement_document="ordinal_operating_point")
    ordinal_csv = tmp_path / "ordinal.csv"
    export_aggregated_csv(ordinal_rows, str(ordinal_csv), delivered_phenotype="astringency",
                          pred_dirs=[ordinal_bucket])
    assert ordinal_csv.exists()

    count_rows = _aggregate_rows("astringency")
    count_csv = tmp_path / "count.csv"
    with pytest.raises(OperationalizationRefused, match=PER_PLANT_COUNT_AGGREGATE):
        export_aggregated_csv(count_rows, str(count_csv), delivered_phenotype="astringency")
    assert not count_csv.exists()

    fx.seed_confirmed_aggregate(tmp_path, "astringency", value_keys=["astringency"])
    count_bucket = _validated_bucket(tmp_path, "count", trait="astringency")
    export_aggregated_csv(count_rows, str(count_csv), delivered_phenotype="astringency",
                          pred_dirs=[count_bucket])

    assert count_csv.exists()
    confirmed = traits.read_trait("astringency", tmp_path).latest_confirmed
    assert confirmed is not None
    assert set(confirmed.entry.operationalizations) == {
        PER_PLANT_ORDINAL_AGGREGATE, PER_PLANT_COUNT_AGGREGATE}


def test_export_detection_csv_refuses_with_no_revision_argument(delivery_root: Path, tmp_path: Path):
    from tcip_mcp.pipelines.postprocessing.export import export_detection_csv

    out_csv = tmp_path / "counts.csv"
    with pytest.raises(TypeError, match="'revision'"):
        export_detection_csv(_count_rows(), str(out_csv))  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to revision

    assert not out_csv.exists()


def test_deliver_per_image_counts_refuses_with_no_trait_argument(
    delivery_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import tcip_mcp.tools.inference_tools as itools

    monkeypatch.setattr(itools, "_run_inference_verified", lambda *args, **kwargs: {
        "results": [{"image": "a.png", "count": 3}], "image_count": 1, "total_detections": 3,
        "operating_point": {"conf": {"value": 0.5}}, "validated": False, "conf_source": "default"})
    out_csv = tmp_path / "o.csv"

    with pytest.raises(TypeError, match="'trait'"):
        itools.deliver_per_image_counts("m.pt", str(tmp_path), str(out_csv))  # type: ignore[call-arg]

    assert not out_csv.exists()


def test_export_aggregated_csv_refuses_with_no_delivered_phenotype_argument(
    delivery_root: Path, tmp_path: Path,
):
    from tcip_mcp.pipelines.postprocessing.aggregation import export_aggregated_csv

    out_csv = tmp_path / "agg.csv"
    with pytest.raises(TypeError, match="'delivered_phenotype'"):
        export_aggregated_csv(_aggregate_rows(), str(out_csv))  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to delivered_phenotype

    assert not out_csv.exists()

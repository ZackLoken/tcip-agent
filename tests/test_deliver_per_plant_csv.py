"""Tests for the general per-plant CSV door: ``aggregate_per_plant``'s own output plus the
existing writer, over a mapping built by ``build_plant_mapping`` and buckets written by
``run_inference``, the same producer path the orthomosaic tests use.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest

from tests import _operationalization_fixtures as fx

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _stub_checkpoint_verification(monkeypatch):
    """This module drives a stubbed predictor, not a real registered checkpoint;
    ``load_registered_checkpoint`` is stubbed to admit whatever path it is given."""
    import tcip_mcp.model_registry as model_registry_mod

    from tests._verified_checkpoint_fixtures import stub_verified_checkpoint

    def _stub(path, *a, **kw):
        p = Path(path)
        sha = model_registry_mod._sha256_of_bytes(p.read_bytes()) if p.is_file() else "stub-sha256"
        return stub_verified_checkpoint(str(path), sha256=sha)

    monkeypatch.setattr(model_registry_mod, "load_registered_checkpoint", _stub)


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """The delivery below ships under a trait whose meaning is confirmed for the aggregate kind
    the writer records under an ``operating_point`` measurement document."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])


class _FakePredictor:
    """A deterministic stand-in for a real detector: one box per detection, ``n`` set per image
    by its stem, so the test's own arithmetic (counts per plant) is known ahead of time rather
    than depending on a randomly-initialized model's actual output."""

    def __init__(self, checkpoint_path=None, **kwargs):
        pass

    def predict_batch(self, paths, **kw):
        counts = {"P1": 2, "P2": 1}
        results = []
        for p in paths:
            n = counts.get(Path(p).stem, 0)
            results.append({
                "image": p, "width": 8, "height": 8,
                "boxes": [[1.0, 1.0, 3.0, 3.0]] * n, "scores": [0.9] * n, "labels": [1] * n,
                "count": n,
            })
        return results


def _ckpt(tmp_path) -> str:
    p = tmp_path / "m.pt"
    if not p.exists():
        p.write_bytes(b"stub")
    return str(p)


def _image_counts(pred_dir: Path, stems: list[str]) -> dict[str, int]:
    from tcip_annotation import json_io

    return {stem: len(json_io.read_annotations(str(pred_dir / f"{stem}.json"))) for stem in stems}


def _write_one_plant_scene(tmp_path: Path) -> tuple[Path, Path, str]:
    """A registered dataset carrying one geolocated image on one date, plus a matching plant
    CSV: the minimal scene a mapping-refusal test needs, with no bucket or validation record
    behind it (each such test refuses before either would matter)."""
    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops

    from tests.test_plant_mapping_binding import _write_geo_image

    assert "error" not in initialize_project(str(tmp_path), site="orchard block")

    dataset_root = tmp_path / "ds"
    images_root = dataset_root / "images"
    date = "2026-02-11"
    _write_geo_image(images_root / date / "P1.jpg", 43.19670, -90.058000,
                     datetime(2026, 2, 11, 9, 30))
    result = register_dataset(
        str(dataset_root), crop=sorted(registered_crops())[0], project_root=str(tmp_path))
    assert "error" not in result, result

    plant_csv = tmp_path / "plants.csv"
    plant_csv.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-1,-90.058000,43.19670\n",
        encoding="utf-8",
    )
    return dataset_root, plant_csv, date


def test_deliver_per_plant_csv_refuses_unvalidated_then_delivers_once_validated(
    tmp_path, monkeypatch,
):
    """The whole producer chain: a mapping built by ``build_plant_mapping`` over a registered
    registry, a bucket written by ``run_inference``, and records from ``aggregate_per_plant``. The
    door takes no acknowledgement, so a bare unvalidated delivery refuses; the same delivery ships
    once the bucket earns a real reference, and records the delivery event under this door's own
    name, once in the dataset's own log and once in the platform's.
    """
    from tcip_mcp.tools.project_tools import initialize_project, register_dataset
    from tcip_mcp.traits import registered_crops

    from tests.test_plant_mapping_binding import _write_geo_image

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    assert "error" not in initialize_project(str(tmp_path), site="orchard block")

    dataset_root = tmp_path / "ds"
    images_root = dataset_root / "images"
    date = "2026-02-11"
    _write_geo_image(images_root / date / "P1.jpg", 43.19670, -90.058000,
                     datetime(2026, 2, 11, 9, 30))
    _write_geo_image(images_root / date / "P2.jpg", 43.19680, -90.057000,
                     datetime(2026, 2, 11, 9, 35))
    register_dataset(str(dataset_root), crop=sorted(registered_crops())[0], project_root=str(tmp_path))

    plant_csv = tmp_path / "plants.csv"
    plant_csv.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-1,-90.058000,43.19670\n"
        "P2,acc-2,-90.057000,43.19680\n",
        encoding="utf-8",
    )

    from tests._binding_fixtures import register_plant_registry_for
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    registry = register_plant_registry_for([plant_csv])
    mapping_name = "valley"
    mapped = build_plant_mapping(
        name=mapping_name, images_root=str(images_root), plant_registry=registry)
    assert "error" not in mapped, mapped
    assert mapped["n_mapped"] == 2

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", _FakePredictor)
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = _ckpt(tmp_path)
    pred_dir = dataset_root / "predictions" / "run" / date
    ran = run_inference(ckpt, str(images_root / date), output_dir=str(pred_dir),
                        conf_threshold=0.0, tile=False)
    assert "error" not in ran, ran

    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping

    mapping_build = load_mapping(tmp_path, mapping_name)
    rows = mapping_build.rows()[date]
    counts = _image_counts(pred_dir, [r["stem"] for r in rows])
    records = [
        {"plant_id": r["plot_name"], "count": counts[r["stem"]],
         "measurement_document": "operating_point", "plant_attribution": r["plant_attribution"]}
        for r in rows
    ]

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    results = aggregate_per_plant(records, strategy="count", plant_id_key="plant_id",
                                  value_key="count")
    out_csv = tmp_path / "out" / "counts.csv"

    predictions_by_date = {date: str(pred_dir)}

    refused = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping=mapping_name,
        predictions_by_date=predictions_by_date)
    assert "error" in refused
    assert refused["unvalidated_dimensions"] == "operating_point"
    assert not out_csv.exists()

    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT, read_operating_point_sidecar
    from tests._binding_fixtures import write_bound_sidecar

    sidecar = read_operating_point_sidecar(pred_dir) or {}
    op = dict(sidecar.get("operating_point") or {})
    op["conf"] = {**op.get("conf", {}), "validated_against": VALIDATED_HELD_OUT}
    stamp = {**sidecar, "validated": True, "trait": fx.COUNT_TRAIT, "operating_point": op}
    write_bound_sidecar(pred_dir, stamp, dataset_root=dataset_root, experiment_id="exp-promoted")

    delivered = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping=mapping_name,
        predictions_by_date=predictions_by_date)
    assert "error" not in delivered, delivered
    assert delivered["operating_point_validated"] == VALIDATED_HELD_OUT
    assert delivered["unvalidated_dimensions"] == ""
    assert delivered["n_plants"] == 2
    assert delivered["plant_mapping"] == mapping_name
    assert delivered["plant_mapping_record_sha256"] == mapping_build.record_sha256
    assert out_csv.exists()

    rows_out = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows_out["P1"]["value"] == "2"
    assert rows_out["P2"]["value"] == "1"
    assert rows_out["P1"]["crop"] == "currant"
    assert rows_out["P1"]["pipeline_version"] == "v1"

    import tcip_store

    from tcip_mcp.audit import audit_log_key

    # The dataset's own log, distinct from the @audited decorator's own platform-root entry.
    page = tcip_store.read_log(audit_log_key(dataset_root))
    door_rows = [r for r in page.records if r["tool"] == "deliver_per_plant_csv"]
    assert len(door_rows) == 1, page.records
    assert door_rows[0]["verified_buckets"][str(pred_dir)]["verified"] is True

    platform_page = tcip_store.read_log(audit_log_key())
    platform_rows = [
        r for r in platform_page.records
        if r["tool"] == "deliver_per_plant_csv" and r["status"] == "ok"
    ]
    assert len(platform_rows) == 1, platform_page.records

    from tcip_mcp.pipelines.resolution import DELIVERY_EVENTS_STORE, delivery_events_scope

    scope = delivery_events_scope(tmp_path)
    events = [
        tcip_store.read(key, default=None)
        for key in tcip_store.keys(DELIVERY_EVENTS_STORE, str(scope))
    ]
    event = next(e for e in events if e and e["door"] == "deliver_per_plant_csv")
    assert event["plant_mapping"]["name"] == mapping_name
    assert event["plant_mapping"]["record_sha256"] == mapping_build.record_sha256


def test_deliver_per_plant_csv_refuses_an_unknown_plant_mapping_by_name(tmp_path):
    """A stated ``plant_mapping`` is a claim the data must positively carry: naming one with no
    stored record refuses by name, before the writer ever runs, the way
    ``deliver_phenology_milestones`` refuses a mapping it cannot load."""
    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "PLANT_A", "count": 3, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="no-such-mapping",
        predictions_by_date={"2026-02-11": str(tmp_path)})
    assert "error" in res
    assert "no-such-mapping" in res["error"]
    assert "build_plant_mapping" in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_a_named_mapping_with_no_verification_inputs(tmp_path):
    """A delivery either fully verifies the mapping it names or names none: naming one without
    ``predictions_by_date`` refuses stating the missing argument, before the name is even
    resolved, rather than shipping a mapping nothing checked."""
    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "PLANT_A", "count": 3, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="valley")
    assert "error" in res
    assert "predictions_by_date" in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_a_malformed_record_through_the_writer(tmp_path):
    """The writer's own refusal (a plant with no value at all) reaches the caller unchanged
    through the door: this door adds no second validation of its own."""
    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"image": "a1", "plant_id": "PLANT_A", "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="")
    assert "error" in res
    assert "no" in res["error"] and "observation" in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_a_predictions_by_date_the_mapping_does_not_cover(
    tmp_path, monkeypatch,
):
    """The shared preamble (``plant_mapping.resolve_delivery_mapping``) refuses a delivered date
    the mapping was never built to cover, before any bucket or dataset identity is read."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dataset_root, plant_csv, date = _write_one_plant_scene(tmp_path)

    from tests._binding_fixtures import register_plant_registry_for
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    registry = register_plant_registry_for([plant_csv])
    mapped = build_plant_mapping(
        name="valley", images_root=str(dataset_root / "images"), plant_registry=registry)
    assert "error" not in mapped, mapped

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "P1", "count": 1, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    uncovered_date = "2026-03-01"
    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="valley",
        predictions_by_date={
            date: str(dataset_root / "predictions" / "run" / date),
            uncovered_date: str(dataset_root / "predictions" / "run" / uncovered_date),
        })
    assert "error" in res
    assert "does not cover" in res["error"]
    assert uncovered_date in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_predictions_under_a_different_dataset_than_the_mapping(
    tmp_path, monkeypatch,
):
    """A predictions bucket resolving to a dataset root the mapping was not built over refuses
    naming both roots, through the shared preamble."""
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dataset_root, plant_csv, date = _write_one_plant_scene(tmp_path)

    other_dataset_root = tmp_path / "ds2"
    other_dataset_root.mkdir()
    other_reg = register_dataset(
        str(other_dataset_root), crop=sorted(registered_crops())[0], project_root=str(tmp_path))
    assert "error" not in other_reg, other_reg

    from tests._binding_fixtures import register_plant_registry_for
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    registry = register_plant_registry_for([plant_csv])
    mapped = build_plant_mapping(
        name="valley", images_root=str(dataset_root / "images"), plant_registry=registry)
    assert "error" not in mapped, mapped

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "P1", "count": 1, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="valley",
        predictions_by_date={date: str(other_dataset_root / "predictions" / "run" / date)})
    assert "error" in res
    assert "different dataset" in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_a_mapping_with_no_capture_at_all_for_a_delivered_date(
    tmp_path, monkeypatch,
):
    """A mapping recorded with no capture at all for a delivered date refuses on the record's
    own evidence, before ``verify_mapping_inputs`` re-reads anything to disclose about."""
    from tests.test_plant_mapping_binding import PLANTS
    from tests.test_ungeoreferenced_capture_refusal import (
        DATE, _delivery_scene, _persist_synthetic_mapping, _write_plant_csv,
    )

    dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, PLANTS)
    plant_csvs = [{"path": str(plant_csv), "sha256": "0" * 64, "n_plants": len(PLANTS)}]
    _persist_synthetic_mapping(
        tmp_path, dataset_root, "valley", plant_csvs=plant_csvs, assignments={DATE: []})

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "PLANT_A", "count": 3, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="valley", predictions_by_date=preds_by_date)
    assert "error" in res
    assert "recorded no capture at all" in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_a_delivered_plant_id_outside_the_mapping(
    tmp_path, monkeypatch,
):
    """This door's one added claim over the writer: every delivered ``plant_id`` must appear
    among a plot the named mapping actually assigned, or the delivery refuses by name."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dataset_root, plant_csv, date = _write_one_plant_scene(tmp_path)

    from tests._binding_fixtures import register_plant_registry_for
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    registry = register_plant_registry_for([plant_csv])
    mapped = build_plant_mapping(
        name="valley", images_root=str(dataset_root / "images"), plant_registry=registry)
    assert "error" not in mapped, mapped

    pred_dir = dataset_root / "predictions" / "manual" / date
    pred_dir.mkdir(parents=True)

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "P1", "count": 1, "plant_attribution": "image",
         "measurement_document": "operating_point"},
        {"plant_id": "GHOST", "count": 2, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="valley",
        predictions_by_date={date: str(pred_dir)})
    assert "error" in res
    assert "GHOST" in res["error"]
    assert "P1" not in res["error"]
    assert not out_csv.exists()


def test_deliver_per_plant_csv_refuses_when_a_capture_added_since_the_mapping_was_built(
    tmp_path, monkeypatch,
):
    """A verification refusal from the shared preamble itself, distinct from its three
    resolution-time refusals: a capture the mapping's own assignments do not name means the
    mapping does not cover what is on disk now."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "unused_workspace"))
    dataset_root, plant_csv, date = _write_one_plant_scene(tmp_path)

    from tests._binding_fixtures import register_plant_registry_for
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    registry = register_plant_registry_for([plant_csv])
    mapped = build_plant_mapping(
        name="valley", images_root=str(dataset_root / "images"), plant_registry=registry)
    assert "error" not in mapped, mapped
    assert mapped["n_mapped"] == 1

    from tests.test_plant_mapping_binding import _write_geo_image

    # Added after the mapping was built: the mapping's own assignments do not name it.
    _write_geo_image(dataset_root / "images" / date / "P2.jpg", 43.19680, -90.057000,
                     datetime(2026, 2, 11, 9, 35))

    pred_dir = dataset_root / "predictions" / "manual" / date
    pred_dir.mkdir(parents=True)

    from tcip_mcp.pipelines.postprocessing.aggregation import aggregate_per_plant
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    records = [
        {"plant_id": "P1", "count": 1, "plant_attribution": "image",
         "measurement_document": "operating_point"},
    ]
    results = aggregate_per_plant(records, strategy="count", value_key="count")
    out_csv = tmp_path / "o.csv"

    res = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="currant",
        pipeline_version="v1", plant_mapping="valley",
        predictions_by_date={date: str(pred_dir)})
    assert "error" in res
    assert "does not cover what is on disk now" in res["error"]
    assert not out_csv.exists()

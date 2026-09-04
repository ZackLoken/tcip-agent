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


def test_deliver_per_plant_csv_refuses_unvalidated_then_delivers_once_validated(
    tmp_path, monkeypatch,
):
    """The whole producer chain: a mapping built by ``build_plant_mapping`` over a registered
    registry, a bucket written by ``run_inference``, and records from ``aggregate_per_plant``. The
    door takes no acknowledgement, so a bare unvalidated delivery refuses (and the retired escape
    hatch is gone outright, a ``TypeError`` rather than a quieter admission); the same delivery
    ships once the bucket earns a real reference, and records the delivery event under this
    door's own name.
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

    refused = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="hazelnut",
        pipeline_version="v1", plant_mapping=mapping_name, pred_dirs=[str(pred_dir)])
    assert "error" in refused
    assert refused["unvalidated_dimensions"] == "operating_point"
    assert not out_csv.exists()

    with pytest.raises(TypeError):
        deliver_per_plant_csv(
            results, str(out_csv), delivered_phenotype="stem_count", crop="hazelnut",
            pipeline_version="v1", plant_mapping=mapping_name, pred_dirs=[str(pred_dir)],
            acknowledge_unvalidated=True)
    assert not out_csv.exists()

    from tcip_mcp.pipelines.resolution import VALIDATED_HELD_OUT, read_operating_point_sidecar
    from tests._binding_fixtures import write_bound_sidecar

    sidecar = read_operating_point_sidecar(pred_dir) or {}
    op = dict(sidecar.get("operating_point") or {})
    op["conf"] = {**op.get("conf", {}), "validated_against": VALIDATED_HELD_OUT}
    stamp = {**sidecar, "validated": True, "trait": fx.COUNT_TRAIT, "operating_point": op}
    write_bound_sidecar(pred_dir, stamp, dataset_root=dataset_root, experiment_id="exp-promoted")

    delivered = deliver_per_plant_csv(
        results, str(out_csv), delivered_phenotype="stem_count", crop="hazelnut",
        pipeline_version="v1", plant_mapping=mapping_name, pred_dirs=[str(pred_dir)])
    assert "error" not in delivered, delivered
    assert delivered["operating_point_validated"] == VALIDATED_HELD_OUT
    assert delivered["unvalidated_dimensions"] == ""
    assert delivered["n_plants"] == 2
    assert delivered["plant_mapping"] == mapping_name
    assert out_csv.exists()

    rows_out = {r["plant_id"]: r for r in csv.DictReader(out_csv.open(newline=""))}
    assert rows_out["P1"]["value"] == "2"
    assert rows_out["P2"]["value"] == "1"
    assert rows_out["P1"]["crop"] == "hazelnut"
    assert rows_out["P1"]["pipeline_version"] == "v1"

    import tcip_store

    from tcip_mcp.audit import audit_log_key

    # The dataset's own log, distinct from the @audited decorator's platform-root entries.
    page = tcip_store.read_log(audit_log_key(dataset_root))
    door_rows = [r for r in page.records if r["tool"] == "deliver_per_plant_csv"]
    assert len(door_rows) == 1, page.records
    assert door_rows[0]["verified_buckets"][str(pred_dir)]["verified"] is True


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
        results, str(out_csv), delivered_phenotype="stem_count", crop="hazelnut",
        pipeline_version="v1", plant_mapping="")
    assert "error" in res
    assert "no" in res["error"] and "observation" in res["error"]
    assert not out_csv.exists()

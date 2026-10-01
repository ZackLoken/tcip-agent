"""The orthomosaic tools: tiled inference over a whole raster published as one bucket, then
per-plant delivery from that bucket plus a registered plant registry."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tcip_mcp.pipelines.execution import Stated
from tests import _trait_fixtures as fx

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# UTM zone 15N: the same real projected CRS test_orthomosaic_mapping.py uses.
UTM_15N_EPSG = 32615
TIEPOINT_NATIVE_X = 500_000.0
TIEPOINT_NATIVE_Y = 4_800_000.0
PIXEL_SCALE = 0.5  # native-CRS units (m) per pixel

TILE = 32
RASTER_PASS = Stated(conf=0.0, tile_size=TILE, overlap=0.2)
"""The execution values every raster pass here states."""
SCOPE = {"subject": fx.COUNT_SUBJECT, "attribute": None, "id_map": {fx.COUNT_SUBJECT: 0}}


@pytest.fixture(autouse=True)
def _recorded_meaning(tmp_path):
    """Every per-plant delivery below ships under a trait whose meaning is confirmed in the
    project these tests act on."""
    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])


def _geokeys() -> tuple[int, ...]:
    entries = [1024, 0, 1, 1, 3072, 0, 1, UTM_15N_EPSG]  # GTModelType=Projected, ProjectedCSType
    return (1, 1, 0, len(entries) // 4, *entries)


def _write_geo_raster(path: Path, *, height: int = 64, width: int = 64, channels: int = 3,
                      rowsperstrip: int = 8, tiepoint_x: float = TIEPOINT_NATIVE_X,
                      seed: int = 0) -> np.ndarray:
    """A raster carrying both real georeferencing tags and real (random, decodable) pixel
    content, so it works for both :func:`read_geotransform` and a tiling inference pass.

    ``tiepoint_x``/``seed`` vary the two halves independently, so a caller can write a
    pixel-identical copy at a moved tiepoint, or different content at the same one."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8)
    extratags = [
        (33550, "d", 3, (PIXEL_SCALE, PIXEL_SCALE, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, tiepoint_x, TIEPOINT_NATIVE_Y, 0.0), False),
        (34735, "H", len(_geokeys()), _geokeys(), False),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr, rowsperstrip=rowsperstrip, extratags=extratags)
    return arr


def _bespoke_detection_checkpoint(tmp_path: Path, *, in_chans: int = 3, tile_size: int = TILE) -> str:
    """Write a bespoke detection checkpoint under ``tmp_path`` and register it in that project's
    model registry, so a caller can hand its bare path to a door that resolves the registry
    itself."""
    from tcip_mcp.pipelines.model_build import build_model, recorded_model_dims
    from tcip_mcp.tools.model_tools import register_model

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"min_size": tile_size, "max_size": tile_size * 2},
                    "task": "detection"}
    config = {"model_source": model_source,
              "data": {"num_channels": in_chans, "scope": dict(SCOPE)}}
    model = build_model(config, recorded_model_dims(config))
    ckpt = tmp_path / "model_best.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, str(ckpt))
    result = register_model(tmp_path, name="test-model", checkpoint_path=str(ckpt), config={})
    assert "error" not in result, result
    return str(ckpt)


def _pixel_to_wgs84(raster_path: Path, px: float, py: float) -> tuple[float, float]:
    from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import OrthomosaicGeoreference

    return OrthomosaicGeoreference.from_file(raster_path).pixel_to_wgs84(px, py)


def _plant_registry(project: Path, plant_csv: Path, *, name: str = "reg") -> str:
    from tests._mapping_fixtures import register_plant_registry_for

    return register_plant_registry_for(project, [plant_csv], name=name)


def _write_plant_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["plot_name", "accession_name", "plot_number", "row_number", "col_number",
                  "WGS84_centroid_y", "WGS84_centroid_x"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plants_csv_at(tmp_path: Path, raster_path: Path, rows: list[tuple[str, float, float]]) -> Path:
    """A registry CSV naming ``rows`` (plot_name, pixel_x, pixel_y) each projected to WGS84
    through this raster's own georeferencing."""
    entries = []
    for i, (name, px, py) in enumerate(rows):
        lat, lon = _pixel_to_wgs84(raster_path, px, py)
        entries.append({
            "plot_name": name, "accession_name": f"acc-{name}", "plot_number": i,
            "row_number": 0, "col_number": i, "WGS84_centroid_y": lat, "WGS84_centroid_x": lon,
        })
    csv_path = tmp_path / f"plants_{len(rows)}.csv"
    _write_plant_csv(csv_path, entries)
    return csv_path


def _plant_grid_csv(tmp_path: Path, raster_path: Path,
                    plant_pixels: list[tuple[float, float]]) -> Path:
    return _plants_csv_at(tmp_path, raster_path, [
        (f"plot{i}", px, py) for i, (px, py) in enumerate(plant_pixels)])


# A 2x2 plant grid, 40px apart, mirroring test_orthomosaic_mapping.py's own layout convention.
_PLANT_PIXELS = [(10.0, 10.0), (10.0, 50.0), (50.0, 10.0), (50.0, 50.0)]
_GRID = ["plot0", "plot1", "plot2", "plot3"]


def _raster_bucket(project: Path, raster_path: Path,
                   boxes: list[tuple[float, float, float, float]], *, name: str = "preds") -> Path:
    """A whole-raster bucket published over ``raster_path`` holding one detection per box, through
    the platform's own publication (``_chain_fixtures.published``)."""
    from tests._chain_fixtures import published

    height, width = tifffile.imread(str(raster_path)).shape[:2]
    result = {"image": str(raster_path), "width": width, "height": height,
              "boxes": [list(b) for b in boxes], "scores": [0.9] * len(boxes),
              "labels": [1] * len(boxes)}
    return published(project, project / "ds" / "predictions" / name, [result], scope=SCOPE,
                     raster_path=raster_path).path


def _deliver(project: Path, bucket: Path, registry: str, plants: list[str], **kwargs) -> dict:
    """``orthomosaic_plant_counts`` shipped under a breeder's acknowledgment (these buckets are
    unassessed), a refusal answering ``{"error": ...}``."""
    from tcip_mcp.delivery import DeliveryRefused
    from tcip_mcp.operationalization import OperationalizationRefused
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_registry
    from tcip_mcp.tools.orthomosaic_tools import orthomosaic_plant_counts
    from tests._chain_fixtures import acknowledged

    try:
        return acknowledged(project, lambda ack: orthomosaic_plant_counts(
            project, str(bucket), load_registry(project, registry), str(project / "counts.csv"),
            "stem_count", plants, acknowledgment_id=ack, door="test_orthomosaic", **kwargs),
            reason="an unassessed bucket")
    except (DeliveryRefused, OperationalizationRefused, ValueError) as exc:
        return {"error": str(exc)}


def _rows(project: Path) -> dict[str, dict]:
    return {r["plant_id"]: r for r in csv.DictReader((project / "counts.csv").open(newline=""))}


# ── run_inference (raster_path regime) ──────────────────────────────


def test_run_inference_over_a_raster_publishes_one_document_and_the_raster_it_ran_on(tmp_path):
    """An explicit tile_size runs the pass; the bucket holds one document for the whole raster and
    a record naming the raster, its content identity and the execution record."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)
    out_dir = tmp_path / "preds"

    result = run_inference(tmp_path, ckpt, output_dir=str(out_dir), raster_path=str(raster_path),
                           stated=RASTER_PASS)

    assert "error" not in result, result
    assert Path(result["output_dir"]) == out_dir
    bucket = read_bucket(out_dir)
    assert bucket.documents == {"mosaic": "mosaic.tif"}
    assert bucket.raster(tmp_path) == str(raster_path)
    assert bucket.raster_identity is not None and bucket.raster_identity["width"] == 64
    assert bucket.execution.tile_size == TILE
    assert bucket.execution.sources["tile_size"] == "explicit"
    assert bucket.assessment_id is None


def test_a_second_raster_export_into_a_published_bucket_refuses_and_leaves_it_whole(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    ckpt = _bespoke_detection_checkpoint(tmp_path)
    out = tmp_path / "preds"
    first = run_inference(tmp_path, ckpt, output_dir=str(out), raster_path=str(raster_path),
                          stated=RASTER_PASS)
    assert "error" not in first, first
    document = (out / "mosaic.json").read_bytes()

    second = run_inference(tmp_path, ckpt, output_dir=str(out), raster_path=str(raster_path),
                           stated=RASTER_PASS)

    assert "already exists" in second["error"]
    assert (out / "mosaic.json").read_bytes() == document


def test_a_missing_checkpoint_or_raster_refuses_cleanly_with_nothing_written(tmp_path):
    from tcip_mcp.tools.inference_tools import run_inference

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    out_dir = tmp_path / "preds"

    no_checkpoint = run_inference(tmp_path, str(tmp_path / "missing.pt"), output_dir=str(out_dir),
                                  raster_path=str(raster_path))
    no_raster = run_inference(tmp_path, _bespoke_detection_checkpoint(tmp_path),
                              output_dir=str(out_dir), raster_path=str(tmp_path / "missing.tif"))

    assert "error" in no_checkpoint
    assert "raster_path not found" in no_raster["error"]
    assert not out_dir.exists()


def test_a_raster_pass_with_no_basis_for_its_tile_edge_refuses_before_writing(tmp_path):
    """No persisted training geometry and no explicit tile_size: an always-tiled pass has no edge
    to tile at, so it refuses unconditionally, never crashing mid-pass."""
    from tcip_mcp.tools.inference_tools import run_inference

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    out_dir = tmp_path / "preds"

    refused = run_inference(tmp_path, _bespoke_detection_checkpoint(tmp_path),
                            output_dir=str(out_dir), raster_path=str(raster_path))

    assert "tile_size could not be resolved" in refused["error"]
    assert not out_dir.exists()


# ── deliver_orthomosaic_plant_counts ─────────────────────────────────────


def test_an_unassessed_raster_bucket_refuses_at_the_door_that_takes_no_acknowledgment(tmp_path):
    from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(tmp_path, _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS))

    refused = deliver_orthomosaic_plant_counts(
        tmp_path, str(bucket), registry, str(tmp_path / "counts.csv"), "stem_count", _GRID)

    assert "no assessment answers" in refused["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_detections_are_counted_to_their_nearest_plant_at_detection_granularity(tmp_path):
    """Two detections near plot0, one near plot2, none near plot1 or plot3: every population plant
    gets a row, an explicit zero included, attributed per detection."""
    from tcip_mcp.delivery import read_delivery_events

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [
        (8.0, 8.0, 12.0, 12.0), (9.0, 9.0, 11.0, 11.0), (48.0, 8.0, 52.0, 12.0)])
    registry = _plant_registry(tmp_path, _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS))

    delivered = _deliver(tmp_path, bucket, registry, _GRID)

    assert "error" not in delivered, delivered
    assert delivered["n_detections"] == 3 and delivered["detections_unattributed"] == 0
    assert delivered["n_plants"] == 4
    rows = _rows(tmp_path)
    assert {p: rows[p]["value"] for p in _GRID} == {
        "plot0": "2", "plot1": "0", "plot2": "1", "plot3": "0"}
    assert {r["plant_attribution"] for r in rows.values()} == {"detection"}
    assert {r["delivered_phenotype"] for r in rows.values()} == {"stem_count"}
    (event,) = read_delivery_events(tmp_path)
    assert event.door == "test_orthomosaic"


def test_a_far_detection_is_counted_to_no_plant(tmp_path):
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [
        (8.0, 8.0, 12.0, 12.0), (3990.0, 3990.0, 4010.0, 4010.0)])
    registry = _plant_registry(tmp_path, _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS))

    result = _deliver(tmp_path, bucket, registry, _GRID)

    assert "error" not in result, result
    assert result["n_detections"] == 2 and result["detections_unattributed"] == 1
    rows = _rows(tmp_path)
    assert rows["plot0"]["value"] == "1"
    assert sum(int(r["value"]) for r in rows.values()) == 1


def test_a_raster_whose_georeferencing_cannot_be_read_refuses_cleanly(tmp_path):
    """A ModelTransformationTag raster refuses through the georeference reader, not as an
    uncaught exception."""
    raster_path = tmp_path / "mosaic.tif"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    transform = (PIXEL_SCALE, 0.0, 0.0, TIEPOINT_NATIVE_X, 0.0, -PIXEL_SCALE, 0.0,
                 TIEPOINT_NATIVE_Y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    extratags = [
        (33550, "d", 3, (PIXEL_SCALE, PIXEL_SCALE, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, TIEPOINT_NATIVE_X, TIEPOINT_NATIVE_Y, 0.0), False),
        (34735, "H", len(_geokeys()), _geokeys(), False),
        (34264, "d", 16, transform, False),
    ]
    tifffile.imwrite(str(raster_path), arr, rowsperstrip=8, extratags=extratags)
    bucket = _raster_bucket(tmp_path, raster_path, [(1.0, 1.0, 5.0, 5.0)])
    plant_csv = tmp_path / "plants.csv"
    _write_plant_csv(plant_csv, [{
        "plot_name": "plot0", "accession_name": "acc0", "plot_number": 0, "row_number": 0,
        "col_number": 0, "WGS84_centroid_y": 42.0, "WGS84_centroid_x": -93.0}])

    result = _deliver(tmp_path, bucket, _plant_registry(tmp_path, plant_csv), ["plot0"])

    assert "ModelTransformationTag" in result["error"]


def test_the_delivery_event_discloses_the_registry_raster_and_tolerance_it_matched_under(
    tmp_path,
):
    """A whole-raster frame carries no walked mapping, so the event names the registry it read,
    the raster identity, the tolerance and its source, and this delivery's unattributed count; a
    stated tolerance is recorded as stated rather than as the grid-pitch derivation."""
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.postprocessing.plant_mapping import grid_pitch_m, read_plant_csvs

    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [
        (8.0, 8.0, 12.0, 12.0), (3990.0, 3990.0, 4010.0, 4010.0)])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)
    registry = _plant_registry(tmp_path, plant_csv)

    assert "error" not in _deliver(tmp_path, bucket, registry, _GRID)
    (event,) = read_delivery_events(tmp_path)
    pm = event.model_dump(mode="json")["plant_mapping"]
    plants = read_plant_csvs([plant_csv])
    assert pm["plant_registry"]["name"] == registry
    assert pm["detections_unattributed"] == 1
    assert pm["detections_unattributed_scope"] == "delivered_raster"
    assert pm["plant_attribution"] == "detection"
    assert pm["nn_tolerance_m"] == {"value": grid_pitch_m(plants) / 6, "source": "grid_pitch"}
    assert pm["raster_identity"] == read_bucket(bucket).raster_identity
    assert "dates_delivered" not in pm and "record_sha256" not in pm

    stated = grid_pitch_m(plants) / 12
    assert "error" not in _deliver(tmp_path, bucket, registry, _GRID, nn_tolerance_m=stated)
    latest = [e for e in read_delivery_events(tmp_path) if e.event_id != event.event_id]
    assert latest[0].model_dump(mode="json")["plant_mapping"]["nn_tolerance_m"] == {
        "value": stated, "source": "stated"}


@pytest.mark.parametrize("change", ["rewritten", "deleted"])
def test_a_registered_plant_csv_that_changed_since_registration_refuses_by_name(tmp_path, change):
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    plant_csv = _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS)
    registry = _plant_registry(tmp_path, plant_csv)
    if change == "rewritten":
        plant_csv.write_text("plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
                             "plot0,acc0,-93.0,42.0\n", encoding="utf-8")
    else:
        plant_csv.unlink()

    refused = _deliver(tmp_path, bucket, registry, _GRID)

    assert plant_csv.name in refused["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_an_unchanged_registry_csv_delivers(tmp_path):
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(tmp_path, _plant_grid_csv(tmp_path, raster_path, _PLANT_PIXELS))

    assert "error" not in _deliver(tmp_path, bucket, registry, _GRID)
    assert (tmp_path / "counts.csv").exists()


def test_a_plant_outside_the_raster_is_named_and_an_edge_detection_is_not_given_to_it(tmp_path):
    """A registry plant outside the raster's own frame is named on the delivery, never counted,
    and a detection at the raster's edge nearer to it than to any in-frame plant stays
    unattributed; naming the outside plant in the population refuses, naming why."""
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)  # 64x64
    bucket = _raster_bucket(tmp_path, raster_path, [(61.0, 8.0, 65.0, 12.0)])  # centroid (63, 10)
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, [
        ("plot_in", 10.0, 10.0), ("plot_out", 70.0, 10.0)]))  # column 70 >= width 64: outside

    result = _deliver(tmp_path, bucket, registry, ["plot_in"], nn_tolerance_m=10.0)
    refused = _deliver(tmp_path, bucket, registry, ["plot_in", "plot_out"], nn_tolerance_m=10.0)

    assert "error" not in result, result
    assert result["detections_unattributed"] == 1
    assert result["plants_outside_raster"] == ["plot_out"]
    assert _rows(tmp_path)["plot_in"]["value"] == "0"
    assert "plot_out" in refused["error"] and "outside the raster's frame" in refused["error"]


@pytest.mark.parametrize("rows, message", [
    ([("plotA", 10.0, 10.0), ("plotA", 50.0, 50.0)], "duplicate plot_name"),
    ([("", 10.0, 10.0)], "blank plot_name"),
], ids=["duplicate", "blank"])
def test_a_registry_naming_a_plant_twice_or_not_at_all_refuses(tmp_path, rows, message):
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, rows))

    result = _deliver(tmp_path, bucket, registry, ["plotA"])

    assert message in result["error"]
    assert not (tmp_path / "counts.csv").exists()


# ── canopy_subject: attribution by segment containment ────────────────────


def _canopy_setup(tmp_path, boxes) -> tuple[Path, Path, Path]:
    """A raster in a registered dataset and its bucket, for the canopy regime's own
    dataset-binding check."""
    from tests._geotiff_fixtures import write_canonical_dataset_raster

    dataset_root = tmp_path / "ds"
    raster_path = write_canonical_dataset_raster(dataset_root, width=64, height=64)
    return dataset_root, raster_path, _raster_bucket(tmp_path, raster_path, boxes)


def _write_canopy_document(raster_path: Path, boxes: list[tuple[float, float, float, float]],
                          *, subject: str = "canopy") -> None:
    """A hand-traced canopy boundary document at ``raster_path``'s own canonical label position,
    one rectangle per ``boxes`` entry, under a person's identity."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, Polygon
    from tcip_mcp.dataset_layout import annotation_path_for_image

    anns = [
        Annotation(
            subject=subject,
            geometry=Polygon(rings=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]),
            created_by="user:breeder", created_at="2024-01-01T00:00:00+00:00",
        )
        for (x0, y0, x1, y1) in boxes
    ]
    doc_path = annotation_path_for_image(raster_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(str(doc_path), anns, 64, 64, keep_empty=True)


def test_canopy_segments_attribute_by_containment_and_name_every_gap(tmp_path):
    from tcip_mcp.delivery import read_delivery_events

    _root, raster_path, bucket = _canopy_setup(tmp_path, [
        (8.0, 8.0, 12.0, 12.0),    # inside segment 0: attributed to plot0
        (48.0, 8.0, 52.0, 12.0),   # inside segment 1: attributed to plot2
        (1.0, 56.0, 3.0, 58.0),    # inside segment 2: segment_without_plant
        (30.0, 30.0, 32.0, 32.0),  # inside no segment: outside_segments
    ])
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0), ("plot1", 10.0, 50.0), ("plot2", 50.0, 10.0), ("plot3", 50.0, 50.0)]))
    _write_canopy_document(raster_path, [
        (5.0, 5.0, 15.0, 15.0), (45.0, 5.0, 55.0, 15.0), (0.0, 55.0, 5.0, 60.0)])

    result = _deliver(tmp_path, bucket, registry, ["plot0", "plot2"], canopy_subject="canopy")

    assert "error" not in result, result
    assert result["detections_unattributed"] == 2
    rows = _rows(tmp_path)
    assert {p: r["value"] for p, r in rows.items()} == {"plot0": "1", "plot2": "1"}
    assert {r["plant_attribution"] for r in rows.values()} == {"segment"}
    (event,) = read_delivery_events(tmp_path)
    pm = event.model_dump(mode="json")["plant_mapping"]
    assert pm["canopy_segments"]["n_segments"] == 3 and pm["canopy_segments"]["subject"] == "canopy"
    assert pm["segments_without_plant"] == 1
    assert sorted(pm["plants_without_segment"]) == ["plot1", "plot3"]
    assert pm["plants_with_ambiguous_detections"] == []
    assert pm["detections_unattributed_by_source"] == {
        "outside_segments": 1, "overlapping_segments": 0, "segment_without_plant": 1}
    assert {t["plot_name"] for t in pm["segment_ties"]} == {"plot0", "plot2"}
    assert all(t["clearance_m"] > 0 for t in pm["segment_ties"])

    refused = _deliver(tmp_path, bucket, registry, ["plot0", "plot1"], canopy_subject="canopy")
    assert "plot1" in refused["error"] and "inside no canopy segment" in refused["error"]


def test_an_ambiguous_overlap_drops_both_implicated_plants_and_keeps_the_third(tmp_path):
    _root, raster_path, bucket = _canopy_setup(tmp_path, [
        (15.0, 8.0, 19.0, 12.0),   # centroid (17, 10): inside both segment 0 and segment 1
        (48.0, 48.0, 52.0, 52.0),  # inside segment 2 alone: attributed to plot2
    ])
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0), ("plot1", 25.0, 10.0), ("plot2", 50.0, 50.0)]))
    _write_canopy_document(raster_path, [
        (0.0, 0.0, 20.0, 20.0), (15.0, 0.0, 35.0, 20.0), (45.0, 45.0, 55.0, 55.0)])

    result = _deliver(tmp_path, bucket, registry, ["plot2"], canopy_subject="canopy")
    refused = _deliver(tmp_path, bucket, registry, ["plot0", "plot2"], canopy_subject="canopy")

    assert "error" not in result, result
    assert _rows(tmp_path) == {"plot2": _rows(tmp_path)["plot2"]}
    assert _rows(tmp_path)["plot2"]["value"] == "1"
    assert "plot0" in refused["error"] and "ambiguous" in refused["error"]


def test_a_stated_tolerance_beside_canopy_subject_refuses(tmp_path):
    _root, raster_path, bucket = _canopy_setup(tmp_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0)]))
    _write_canopy_document(raster_path, [(5.0, 5.0, 15.0, 15.0)])

    result = _deliver(tmp_path, bucket, registry, ["plot0"], canopy_subject="canopy",
                      nn_tolerance_m=5.0)

    assert "nn_tolerance_m" in result["error"]
    assert not (tmp_path / "counts.csv").exists()


def test_canopy_subject_refuses_a_raster_outside_a_registered_dataset(tmp_path):
    raster_path = tmp_path / "mosaic.tif"
    _write_geo_raster(raster_path)
    bucket = _raster_bucket(tmp_path, raster_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0)]))

    result = _deliver(tmp_path, bucket, registry, ["plot0"], canopy_subject="canopy")

    assert "registered dataset" in result["error"]


def test_canopy_subject_refuses_a_missing_canopy_document(tmp_path):
    _root, raster_path, bucket = _canopy_setup(tmp_path, [(8.0, 8.0, 12.0, 12.0)])
    registry = _plant_registry(tmp_path, _plants_csv_at(tmp_path, raster_path, [
        ("plot0", 10.0, 10.0)]))

    result = _deliver(tmp_path, bucket, registry, ["plot0"], canopy_subject="canopy")

    assert "no label document" in result["error"]

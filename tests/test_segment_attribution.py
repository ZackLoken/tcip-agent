"""Per-plant attribution by canopy segment: loading a canopy boundary document, tying its
segments to registry plants, and assigning detections to a tie's own segments.

Coverage tests for a new module: each function's own admitting and refusing shapes, on
synthetic segments over a real registered-dataset raster (:mod:`tests._geotiff_fixtures`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox, Point, Polygon

from tcip_mcp.pipelines.postprocessing.orthomosaic_mapping import OrthomosaicGeoreference
from tcip_mcp.pipelines.postprocessing.plant_mapping import PlantRecord
from tcip_mcp.pipelines.postprocessing.segment_attribution import (
    CanopySegmentRefusal,
    assign_detections_to_segments,
    load_canopy_segments,
    tie_segments_to_plants,
)

from tests._geotiff_fixtures import write_canonical_dataset_raster

WIDTH = HEIGHT = 64


def _setup(tmp_path: Path) -> tuple[Path, Path, OrthomosaicGeoreference, dict]:
    dataset_root = tmp_path / "ds"
    raster_path = write_canonical_dataset_raster(dataset_root, width=WIDTH, height=HEIGHT)
    georef = OrthomosaicGeoreference.from_file(raster_path)
    raster_identity = {"width": WIDTH, "height": HEIGHT}
    return dataset_root, raster_path, georef, raster_identity


def _doc_path(raster_path: Path) -> Path:
    from tcip_mcp.dataset_layout import annotation_path_for_image

    return annotation_path_for_image(raster_path)


def _write_document(raster_path: Path, annotations: list[Annotation]) -> bytes:
    doc_path = _doc_path(raster_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    write_annotations(str(doc_path), annotations, WIDTH, HEIGHT, keep_empty=True)
    return doc_path.read_bytes()


def _square(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon(rings=[[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]])


def _plant(georef: OrthomosaicGeoreference, plot_name: str, px: float, py: float) -> PlantRecord:
    lat, lon = georef.pixel_to_wgs84(px, py)
    return PlantRecord(plot_name=plot_name, accession_name=f"acc-{plot_name}", plot_number=None,
                       row_number=None, col_number=None, lat=lat, lon=lon)


# ── load_canopy_segments ──────────────────────────────────────────────────


def test_load_canopy_segments_admits_a_hand_traced_polygon_and_a_box(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    anns = [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20),
                  created_by="user:breeder", created_at="t"),
        Annotation(subject="canopy", geometry=BBox(30, 30, 45, 45),
                  created_by="user:breeder", created_at="t"),
    ]
    data = _write_document(raster_path, anns)

    segments = load_canopy_segments(
        data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)

    assert [s.segment_index for s in segments] == [0, 1]
    assert segments[0].polygon.rings[0][0] == (5.0, 5.0)
    # A box is admitted as the rectangle it is: four corners, same order _to_shapely builds.
    assert segments[1].polygon.rings[0] == [(30.0, 30.0), (45.0, 30.0), (45.0, 45.0), (30.0, 45.0)]


def test_load_canopy_segments_excludes_annotations_of_another_subject(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    anns = [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20), created_by="user:breeder"),
        Annotation(subject="other", geometry=_square(30, 30, 40, 40), created_by="user:breeder"),
    ]
    data = _write_document(raster_path, anns)

    segments = load_canopy_segments(
        data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)

    assert len(segments) == 1


def test_load_canopy_segments_refuses_a_document_naming_another_image_stem(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20), created_by="user:breeder"),
    ])
    raw = json.loads(data)
    raw["image"] = "some-other-stem"
    edited = json.dumps(raw).encode("utf-8")

    with pytest.raises(CanopySegmentRefusal, match="some-other-stem"):
        load_canopy_segments(
            edited, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_refuses_a_document_whose_size_differs(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20), created_by="user:breeder"),
    ])
    raw = json.loads(data)
    raw["width"] = 999
    edited = json.dumps(raw).encode("utf-8")

    with pytest.raises(CanopySegmentRefusal, match="999"):
        load_canopy_segments(
            edited, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_refuses_when_no_annotation_of_subject_exists(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="other", geometry=_square(5, 5, 20, 20), created_by="user:breeder"),
    ])

    with pytest.raises(CanopySegmentRefusal, match="canopy"):
        load_canopy_segments(
            data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_refuses_a_point_naming_the_record(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=Point(10.0, 10.0), created_by="user:breeder"),
    ])

    with pytest.raises(CanopySegmentRefusal, match="Point"):
        load_canopy_segments(
            data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_refuses_a_scored_polygon(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20), score=0.9, created_by="sam"),
    ])

    with pytest.raises(CanopySegmentRefusal, match="prediction score"):
        load_canopy_segments(
            data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_refuses_a_polygon_with_no_created_by(tmp_path: Path) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20)),
    ])

    with pytest.raises(CanopySegmentRefusal, match="no created_by"):
        load_canopy_segments(
            data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_refuses_a_machine_authored_polygon_with_no_persons_acceptance(
    tmp_path: Path,
) -> None:
    _, raster_path, _georef, identity = _setup(tmp_path)
    data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(5, 5, 20, 20), created_by="sam"),
    ])

    with pytest.raises(CanopySegmentRefusal, match="no person"):
        load_canopy_segments(
            data, subject="canopy", raster_stem=raster_path.stem, raster_identity=identity)


def test_load_canopy_segments_admits_a_sam_authored_polygon_accepted_through_review(
    tmp_path: Path,
) -> None:
    """A SAM-authored proposal accepted through the review route's own accept path delivers: the
    platform's own producer of a person's ``accepted_by``, never a hand-written provenance pair.
    """
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    dataset_root, raster_path, _georef, identity = _setup(tmp_path)
    pred_path = dataset_root / "predictions" / "sam" / "2024-06-01" / f"{raster_path.stem}.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    write_annotations(
        str(pred_path),
        [Annotation(subject="canopy", geometry=_square(5, 5, 20, 20), score=0.9, created_by="sam")],
        WIDTH, HEIGHT,
    )
    gt_path = _doc_path(raster_path)

    client = TestClient(app, base_url="http://127.0.0.1")
    resp = client.post("/api/review/action", json={
        "dataset_root": str(dataset_root),
        "image_name": raster_path.name,
        "image_path": str(raster_path),
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "det_type": "fp",
        "class_name": "canopy",
        "conf": 0.9,
        "gt_idx": None,
        "pred_idx": 0,
        "bbox": (5.0, 5.0, 20.0, 20.0),
        "action": "accepted",
        "user": "breeder",
    })
    assert resp.status_code == 200, resp.text

    segments = load_canopy_segments(
        gt_path.read_bytes(), subject="canopy", raster_stem=raster_path.stem,
        raster_identity=identity)
    assert len(segments) == 1
    stored = json.loads(gt_path.read_text())
    assert stored["annotations"][0]["created_by"] == "sam"
    assert stored["annotations"][0]["accepted_by"] == "user:breeder"
    assert "score" not in stored["annotations"][0]


# ── tie_segments_to_plants ────────────────────────────────────────────────


def test_tie_segments_to_plants_ties_two_disjoint_segments_each_to_one_plant(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
        Annotation(subject="canopy", geometry=_square(30, 30, 50, 50), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10), _plant(georef, "plot1", 40, 40)]

    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    assert {t.plot_name for t in tie.tied} == {"plot0", "plot1"}
    assert tie.untied == []
    assert tie.plants_without_segment == []
    assert tie.plants_outside_raster == []
    assert all(t.clearance_m > 0 for t in tie.tied)


def test_tie_segments_to_plants_leaves_a_plantless_segment_untied(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
        Annotation(subject="canopy", geometry=_square(30, 30, 50, 50), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10)]

    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    assert [t.plot_name for t in tie.tied] == ["plot0"]
    assert [s.segment_index for s in tie.untied] == [1]


def test_tie_segments_to_plants_refuses_a_segment_containing_two_plants(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 30, 30), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10), _plant(georef, "plot1", 15, 15)]

    with pytest.raises(ValueError, match="more than one plant"):
        tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)


def test_tie_segments_to_plants_refuses_a_plant_inside_two_segments(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 30, 30), created_by="user:breeder"),
        Annotation(subject="canopy", geometry=_square(5, 5, 25, 25), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 15, 15)]

    with pytest.raises(ValueError, match="more than one canopy segment"):
        tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)


def test_tie_segments_to_plants_refuses_a_blank_named_plant(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "", 10, 10)]

    with pytest.raises(ValueError, match="blank plot_name"):
        tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)


def test_tie_segments_to_plants_refuses_a_duplicate_plot_name(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
        Annotation(subject="canopy", geometry=_square(30, 30, 50, 50), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10), _plant(georef, "plot0", 40, 40)]

    with pytest.raises(ValueError, match="duplicate plot_name"):
        tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)


def test_tie_segments_to_plants_refuses_when_no_plant_is_in_frame(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", -50, -50)]

    with pytest.raises(ValueError, match="no registry plant"):
        tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)


def test_tie_segments_to_plants_names_a_plant_outside_the_raster(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10), _plant(georef, "plot1", -50, -50)]

    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    assert tie.plants_outside_raster == ["plot1"]
    assert [t.plot_name for t in tie.tied] == ["plot0"]


def test_tie_segments_to_plants_names_an_in_frame_plant_with_no_segment(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10), _plant(georef, "plot1", 45, 45)]

    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    assert tie.plants_without_segment == ["plot1"]
    assert [t.plot_name for t in tie.tied] == ["plot0"]


# ── assign_detections_to_segments ─────────────────────────────────────────


def test_assign_detections_to_segments_attributes_a_containment_and_names_the_rest(
    tmp_path: Path,
) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
        Annotation(subject="canopy", geometry=_square(30, 30, 50, 50), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10)]  # segment 1 (30..50) stays untied
    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    detections = {"boxes": [
        [8.0, 8.0, 12.0, 12.0],       # centroid (10, 10): inside the tied segment
        [58.0, 58.0, 62.0, 62.0],     # centroid (60, 60): inside no segment
        [38.0, 38.0, 42.0, 42.0],     # centroid (40, 40): inside the untied segment
    ]}

    assignments = assign_detections_to_segments(detections, tie)

    assert assignments[0].source == "segment_containment"
    assert assignments[0].plot_name == "plot0"
    assert assignments[0].segment_index == 0
    assert assignments[0].distance_m is None
    assert assignments[0].plant_attribution == "segment"

    assert assignments[1].source == "outside_segments"
    assert assignments[1].segment_index is None
    assert assignments[1].plot_name is None

    assert assignments[2].source == "segment_without_plant"
    assert assignments[2].segment_index == 1
    assert assignments[2].plot_name is None


def test_assign_detections_to_segments_attributes_an_overlap_to_neither(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 30, 30), created_by="user:breeder"),
        Annotation(subject="canopy", geometry=_square(20, 0, 50, 30), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10), _plant(georef, "plot1", 40, 10)]
    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    # centroid (25, 15) lies in both segments' overlap (20..30 on x).
    detections = {"boxes": [[23.0, 13.0, 27.0, 17.0]]}

    assignments = assign_detections_to_segments(detections, tie)

    assert assignments[0].source == "overlapping_segments"
    assert assignments[0].segment_index is None
    assert assignments[0].plot_name is None
    assert set(assignments[0].overlapping_segment_indices) == {0, 1}


def test_assign_detections_to_segments_no_boxes_returns_empty(tmp_path: Path) -> None:
    _, raster_path, georef, _identity = _setup(tmp_path)
    segments_data = _write_document(raster_path, [
        Annotation(subject="canopy", geometry=_square(0, 0, 20, 20), created_by="user:breeder"),
    ])
    segments = load_canopy_segments(
        segments_data, subject="canopy", raster_stem=raster_path.stem,
        raster_identity={"width": WIDTH, "height": HEIGHT})
    plants = [_plant(georef, "plot0", 10, 10)]
    tie = tie_segments_to_plants(segments, plants, georef, width=WIDTH, height=HEIGHT)

    assert assign_detections_to_segments({"boxes": []}, tie) == []

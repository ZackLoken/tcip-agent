"""``calibrate_physical_scale``: a per-pixel scale derived on a selection's calibration side from
reference objects of known physical length, checked against its holdout side, and recorded as an
assessment a dimensional delivery names.

A reference's pixel extent is the principal-axis extent of its own polygon, so orientation does
not move it; a box, two references in one image, a row-less reference, an unauthored tolerance and
a unit that is no length each refuse; too few or disagreeing references fail.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tcip_annotation.state import Annotation, BBox, Polygon
from tests import _trait_fixtures as fx, csv_rows
from tests._producer_fixtures import label_image

DATE = "2026-01-01"
PX_PER_MM = 10.0  # a fixed 0.1 mm/px reference scale, chosen for round test numbers
SUBJECT = "cal_bar"


@pytest.fixture(autouse=True)
def _traits(tmp_path):
    fx.seed_delivery_traits(tmp_path)


def _rect(length_px: float, width_px: float = 10.0, angle_deg: float = 0.0,
          center: tuple[float, float] = (60.0, 60.0)) -> list[tuple[float, float]]:
    """Four corners of a length x width rectangle, rotated ``angle_deg`` about its own center."""
    hl, hw = length_px / 2.0, width_px / 2.0
    a = math.radians(angle_deg)
    cx, cy = center
    return [(cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a))
            for x, y in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw))]


def _reference(project: Path, *, n: int = 8, extents_mm: list[float] | None = None,
               unit: str = "mm", angles: dict[int, float] | None = None) -> tuple[Path, Path]:
    """``n`` reference frames of one 100 px bar each in a registered dataset, their breeder CSV
    (each physical extent ``extents_mm[i]``, else the fixed scale's), and a selection drawn over
    them; ``(selection_dir, reference_csv)``."""
    from PIL import Image

    from tcip_mcp import subject_registry as cr
    from tests._producer_fixtures import registry_over
    from tcip_mcp.tools.data_tools import draw_splits
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops

    root = project / "ds"
    images = root / "images" / DATE
    images.mkdir(parents=True)
    registry_over(root, cr.SubjectRegistry(subjects=(cr.Subject(name=SUBJECT),)))
    assert "error" not in register_dataset(project, str(root), crop=sorted(registered_crops())[0])
    rows = []
    for i in range(n):
        stem = f"r{i}"
        Image.new("RGB", (120, 120), color=(20 + i, 20, 20)).save(images / f"{stem}.png")
        label_image(images / f"{stem}.png", [Annotation(
            subject=SUBJECT, geometry=Polygon(rings=[_rect(100.0, angle_deg=(angles or {}).get(i, 0.0))]))],
            120, 120)
        extent = extents_mm[i] if extents_mm else 100.0 / PX_PER_MM
        rows.append((stem, extent, unit))
    csv_path = project / "reference.csv"
    _write_csv(csv_path, rows)
    selection = project / "selection"
    drawn = draw_splits(project, str(root), output_path=str(selection), subject=SUBJECT,
                        group_by="stem", seed=1, val_ratio=0.25,
                        calibration_ratio=0.25, holdout_ratio=0.25)
    assert "error" not in drawn, drawn
    return selection, csv_path


def _write_csv(path: Path, rows, header=("image_stem", "physical_extent", "unit")) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)


def _author_tolerance(project: Path, tolerance_frac: float = 0.1) -> None:
    fx.propose_and_confirm(project, fx.with_fields(
        fx.latest("plant_surface_area", project), scale_tolerance_frac=tolerance_frac))


def _calibrate(project: Path, selection: Path, csv_path: Path, *, unit: str = "mm") -> dict:
    from tcip_mcp.tools.calibration_tools import calibrate_physical_scale

    return calibrate_physical_scale(project, trait="plant_surface_area",
                                    selection_dir=str(selection), reference_csv=str(csv_path),
                                    unit=unit, reference_subject=SUBJECT)


def test_the_whole_chain_delivers_a_dimensional_area_resting_on_the_scale(tmp_path):
    """A passing scale assessment covering the delivered capture answers for an mm2 area: the
    delivery names it, and the same delivery with no scale named refuses."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate
    from tests._chain_fixtures import acknowledged, predicted, published

    _author_tolerance(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "plant_surface_area", value_keys=["area_mm2"],
                                measured_subject=SUBJECT)
    selection, csv_path = _reference(tmp_path)
    scale = _calibrate(tmp_path, selection, csv_path)
    assert scale["passed"] is True, scale
    assert scale["scale"]["value"] == pytest.approx(0.1)  # 10 mm over a 100 px reference length

    image = tmp_path / "ds" / "images" / DATE / "r0.png"
    published_bucket = published(tmp_path, f"m/{DATE}", [predicted(image, [SUBJECT])],
                                 scope={"subject": SUBJECT})
    bucket = read_bucket(published_bucket.root, published_bucket.name)
    rows = [{"plant_id": "p1", "value": 12.5, "observations": 1, "value_key": "area_mm2",
             "plant_attribution": "image"}]

    def deliver(out: Path, **kw):
        return acknowledged(tmp_path, lambda ack: deliver_per_plant_aggregate(
            tmp_path, rows, str(out), delivered_phenotype="plant_surface_area",
            delivery_kind="per_plant_count_aggregate", buckets=[bucket], plants=["p1"],
            door="test_scale", acknowledgment_id=ack, actor=None, **kw),
            reason="the count is not yet assessed")

    with pytest.raises(ValueError, match="calibrate_physical_scale"):
        deliver(tmp_path / "unscaled.csv")
    deliver(tmp_path / "scaled.csv", scale_assessment_id=scale["assessment_id"])

    (row,) = csv_rows(tmp_path / "scaled.csv")
    assert row["units"] == "mm2"
    (event,) = read_delivery_events(tmp_path)
    assert event.scale_assessment_id == scale["assessment_id"]


def test_a_bar_annotated_at_45_degrees_implies_the_same_scale_as_an_axis_aligned_one(tmp_path):
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path, angles={i: 45.0 for i in range(0, 8, 2)})

    result = _calibrate(tmp_path, selection, csv_path)

    assert result["passed"] is True, result
    assert result["scale"]["value"] == pytest.approx(0.1)


def test_a_box_reference_refuses_rather_than_reading_a_projected_extent(tmp_path):
    """A box's long side is the object's projected extent, orientation-dependent: the
    assessment refuses it outright."""
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    for image in (tmp_path / "ds" / "images" / DATE).glob("*.png"):
        label_image(image, [Annotation(subject=SUBJECT, geometry=BBox(10, 55, 110, 65))],
                    120, 120)

    result = _calibrate(tmp_path, selection, csv_path)

    assert "polygon" in result["error"]


def test_an_image_with_two_reference_annotations_refuses_rather_than_picking_one(tmp_path):
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    for image in (tmp_path / "ds" / "images" / DATE).glob("*.png"):
        label_image(image, [
            Annotation(subject=SUBJECT, geometry=Polygon(rings=[_rect(100.0)])),
            Annotation(subject=SUBJECT, geometry=Polygon(rings=[_rect(40.0, center=(30, 30))]))],
            120, 120)

    result = _calibrate(tmp_path, selection, csv_path)

    assert "exactly one" in result["error"]


def test_a_reference_image_the_csv_names_no_extent_for_refuses_naming_it(tmp_path):
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    _write_csv(csv_path, [])

    result = _calibrate(tmp_path, selection, csv_path)

    assert "names no physical extent for reference image" in result["error"]


def test_an_unauthored_tolerance_refuses_asking_the_breeders_question(tmp_path):
    fx.propose_and_confirm(tmp_path, fx.with_fields(
        fx.latest("plant_surface_area", tmp_path), scale_tolerance_frac=None))
    selection, csv_path = _reference(tmp_path)

    result = _calibrate(tmp_path, selection, csv_path)

    assert "scale_tolerance_frac" in result["error"]


def test_a_unit_that_is_no_length_refuses_naming_it(tmp_path):
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path, unit="g")

    result = _calibrate(tmp_path, selection, csv_path, unit="g")

    assert "'g'" in result["error"]


def test_too_few_references_on_a_side_fail_as_unreplicated(tmp_path):
    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path, n=4)

    result = _calibrate(tmp_path, selection, csv_path)

    assert result["passed"] is False
    assert any(f.startswith("insufficient_") for f in result["failures"]), result["failures"]


def test_a_reference_set_disagreeing_with_itself_beyond_the_tolerance_fails(tmp_path):
    """Implied scales of 0.05 to 0.5 mm/px disagree across any split by far more than 5%."""
    _author_tolerance(tmp_path, tolerance_frac=0.05)
    selection, csv_path = _reference(
        tmp_path, extents_mm=[5.0, 10.0, 15.0, 50.0, 5.0, 10.0, 15.0, 50.0])

    result = _calibrate(tmp_path, selection, csv_path)

    assert result["passed"] is False, result


def test_a_holdout_image_identical_to_a_calibration_image_fails_disjointness(tmp_path):
    """The scale is checked on images it was not derived from: a holdout frame byte for byte a
    calibration frame shares its source digest, and the assessment fails naming it."""
    from tcip_mcp.pipelines.data.selection import read_selection

    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    drawn = read_selection(selection, project=tmp_path)
    calibration, holdout = drawn.on("calibration")[0], drawn.on("holdout")[0]
    Path(holdout.source).write_bytes(Path(calibration.source).read_bytes())

    result = _calibrate(tmp_path, selection, csv_path)

    assert result["passed"] is False, result
    assert "holdout_shares_calibration_image" in result["failures"]


# ── the reference CSV reader: read by name, refuse rather than guess ──────


def test_the_reference_csv_is_read_by_column_name(tmp_path):
    from tcip_mcp.assessment import _read_reference_csv

    path = tmp_path / "reference.csv"
    _write_csv(path, [("mm", "r1", 10.0), ("mm", "r2", 12.5)],
               header=("unit", "image_stem", "physical_extent"))

    assert _read_reference_csv(path.read_bytes(), str(path)) == {
        "r1": {"physical_extent": 10.0, "unit": "mm"}, "r2": {"physical_extent": 12.5, "unit": "mm"}}


@pytest.mark.parametrize(("rows", "match"), [
    ([("r1", "10.0", "mm"), ("r2", "not-a-number", "mm")], r":3\b.*non-numeric"),
    ([("r1", "10.0", "mm"), ("r1", "11.0", "mm")], "repeats stem 'r1'"),
    ([("r1", "10.0")], r":2\b"),
    ([("r1", "-10.0", "mm")], r":2\b.*positive"),
    ([("r1", "0", "mm")], r":2\b.*positive"),
], ids=["non-numeric", "duplicate-stem", "short-row", "negative", "zero"])
def test_a_malformed_reference_csv_refuses_naming_its_line(tmp_path, rows, match):
    from tcip_mcp.assessment import AssessmentRefused, _read_reference_csv

    path = tmp_path / "reference.csv"
    _write_csv(path, rows)
    with pytest.raises(AssessmentRefused, match=match):
        _read_reference_csv(path.read_bytes(), str(path))


def test_the_scale_is_measured_from_the_retained_table(tmp_path, monkeypatch):
    """The breeder's table is read once, that read is retained, and the scale is measured from
    the retained copy: a table that changes as the assessment opens is measured as it was read
    and retained, and a change after the assessment is a moved reference."""
    from tcip_mcp import assessment as module
    from tcip_mcp.assessment import assessment_dir, read_assessment

    _author_tolerance(tmp_path)
    selection, csv_path = _reference(tmp_path)
    read = module._reference_reads

    def changed_as_it_is_read(samples, extra=()):
        _write_csv(csv_path, [(f"r{i}", 20.0, "mm") for i in range(8)])
        return read(samples, extra)

    monkeypatch.setattr(module, "_reference_reads", changed_as_it_is_read)
    scale = _calibrate(tmp_path, selection, csv_path)
    monkeypatch.undo()

    assert scale["scale"]["value"] == pytest.approx(0.2), scale
    recorded = read_assessment(tmp_path, scale["assessment_id"])
    run_dir = assessment_dir(tmp_path, scale["assessment_id"])
    (table,) = [f for f in recorded.reference.ground_truth if f.ground_truth == str(csv_path)]
    assert "20.0" in (run_dir / table.copy).read_text()
    assert recorded.reference.moved(run_dir) == []
    _write_csv(csv_path, [(f"r{i}", 30.0, "mm") for i in range(8)])
    assert recorded.reference.moved(run_dir) == [str(csv_path)]

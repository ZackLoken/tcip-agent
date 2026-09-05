"""Phenology milestones on curves that are not tidy: noisy series, exact-on-target first
observations, sparse class ids, and a bucket written by the real prediction writer.

A positive-fraction curve measured from real imagery moves up and down: weather, occlusion and
sampling noise all push a plant's fraction back down between captures. The milestone a breeder
reads off such a curve is only defensible if the crossing was computed on the series as it happened
in time, and if a date the observations only bound is delivered as a bound rather than as a
measurement. These pin both, at the module's own boundary and through the delivery tool, plus the
agreement between the writer that produces a prediction bucket and the readers here that count it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.postprocessing import phenology
from tests._binding_fixtures import write_bound_sidecar
from tests._trait_fixtures import BUD_OPENING

# A registry whose ids are not consecutive: the positive class sits at id 2 with nothing at id 1.
SPARSE_ID_MAP = {"closed": 0, "open": 2}


class _Assignment:
    def __init__(self, stem: str, plot_name: str, accession_name: str) -> None:
        self.stem = stem
        self.plot_name = plot_name
        self.accession_name = accession_name


def _write_preds(dir_path: Path, stem: str, subjects: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    anns = [Annotation(subject="bud", geometry=BBox(1.0 + i, 2.0, 4.0 + i, 9.0), score=0.9,
                       attributes={"opening": s})
            for i, s in enumerate(subjects)]
    json_io.write_annotations(dir_path / f"{stem}.json", anns, 40, 24)


def _states(n_positive: int, n_negative: int) -> list[str]:
    return ["open"] * n_positive + ["closed"] * n_negative


def _write_sidecar(dir_path: Path, id_map: dict, *, dataset_root: Path, validated: bool = True,
                   conf: float = 0.37) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    ref = "held_out_annotations" if validated else "false"
    stamp = {
        "validated": validated,
        "trait": "bud_opening",
        "operating_point": {"conf": {"value": conf, "validated_against": ref}},
        "id_map": id_map,
        "experiment_id": "exp-77",
        "subject": "bud", "attribute": "opening",
    }
    if validated:
        write_bound_sidecar(dir_path, stamp, dataset_root=dataset_root,
                            experiment_id=f"exp-record-{dir_path.name}",
                            producing_experiment_id="exp-77")
    else:
        (dir_path / "operating_point.json").write_text(json.dumps(stamp), encoding="utf-8")


def _write_classifier_sidecar(dir_path: Path, *, dataset_root: Path, trait: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    stamp = {
        "validated": True,
        "operating_point": {"classifier": {"value": "open",
                                           "validated_against": "held_out_annotations"}},
        "trait": trait,
        "experiment_id": "exp-77",
    }
    write_bound_sidecar(dir_path, stamp, document="classifier_operating_point",
                        dataset_root=dataset_root, experiment_id=f"exp-classifier-{dir_path.name}",
                        producing_experiment_id="exp-77", trait=trait)


# -- crossings on a curve that rises and falls ----------------------------


def test_crossing_uses_the_bracket_that_exists_in_time():
    """The crossing is read off the two neighbouring captures that straddle the target in calendar
    order, so its date lies inside that bracket and the gap it reports is the real number of days
    between them. A series whose fraction order differs from its capture order (the ordinary case
    for a noisy curve) must not be re-ordered into a bracket that never happened.
    """
    series = [("2026-03-09", 0.30), ("2026-03-01", 0.10), ("2026-03-13", 0.90), ("2026-03-05", 0.70)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "interpolated"
    assert c.date == "2026-03-04"
    assert "2026-03-01" < c.date < "2026-03-05"
    assert c.gap_days == 4


def test_crossing_above_target_at_the_first_capture_stays_left_censored_when_the_curve_dips():
    """A curve whose first capture is already past the target and which later falls back below it
    still crosses only once, before the watching began. The later dip and recovery is not a second,
    later crossing to deliver in its place.
    """
    series = [("2026-03-01", 0.60), ("2026-03-05", 0.20), ("2026-03-09", 0.80)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"


def test_positive_onset_is_the_earliest_capture_with_any_positive_observation():
    """Onset is the first date in calendar order carrying a positive observation, not the date
    carrying the smallest positive fraction.
    """
    series = [("2026-03-01", 0.0), ("2026-03-05", 0.40), ("2026-03-09", 0.10)]

    assert phenology.positive_onset_date(series) == "2026-03-05"


def test_first_capture_exactly_on_the_target_is_an_upper_bound():
    """A first observation sitting exactly on the target met it before anyone looked, so the date is
    an upper bound on the true crossing and carries no bracket to report a gap for. Delivering it as
    an interpolated or exact crossing claims a precision the observations do not support.
    """
    series = [("2026-03-01", 0.50), ("2026-03-05", 0.90)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"
    assert c.gap_days is None


def test_first_capture_exactly_on_the_target_keeps_its_date_when_the_curve_falls_back():
    """The upper-bound reading holds when the fraction drops after that first capture: the crossing
    is still bounded by the first observation, never moved forward to a later re-crossing.
    """
    series = [("2026-03-01", 0.50), ("2026-03-05", 0.20), ("2026-03-09", 0.80)]

    c = phenology.crossing_date(series, 0.50)

    assert c.bound == "left_censored"
    assert c.date == "2026-03-01"


def test_milestones_of_a_noisy_plant_and_a_steady_plant_are_each_read_in_capture_order(tmp_path):
    """Two plants observed on the same four dates, one whose fraction dips mid-season and one that
    rises steadily, get their milestones from their own series in capture order. The mapping's dates
    are supplied out of order, as an assembled mapping file has them, which changes no result.
    """
    dates = ["2026-03-01", "2026-03-05", "2026-03-09", "2026-03-13"]
    counts = {
        "2026-03-01": {"P1": (1, 9), "P2": (0, 4)},
        "2026-03-05": {"P1": (7, 3), "P2": (1, 3)},
        "2026-03-09": {"P1": (3, 7), "P2": (3, 1)},
        "2026-03-13": {"P1": (9, 1), "P2": (4, 0)},
    }
    for d in dates:
        bucket = tmp_path / d
        _write_sidecar(bucket, SPARSE_ID_MAP, dataset_root=tmp_path)
        for plant, (pos, neg) in counts[d].items():
            _write_preds(bucket, f"{plant}_{d}", _states(pos, neg))
    mapping = {d: [_Assignment(f"P1_{d}", "P1", "acc-noisy"),
                   _Assignment(f"P2_{d}", "P2", "acc-steady")]
               for d in ["2026-03-09", "2026-03-01", "2026-03-13", "2026-03-05"]}
    preds = {d: str(tmp_path / d) for d in dates}

    out = phenology.per_plant_phenology(mapping, preds, positive_class_name="open",
                                        spec=BUD_OPENING)

    rows = {r["plant_id"]: r for r in out["rows"]}
    assert set(rows) == {"P1", "P2"}
    assert rows["P1"]["n_dates"] == 4
    assert rows["P1"]["n_dates_unclassified"] == 0
    # The noisy plant reaches half its buds between the first two captures, before the dip, and
    # never reaches 95 per cent inside the observed window.
    assert rows["P1"]["bud_50per_date"] == "2026-03-04"
    assert rows["P1"]["bud_50per_date_bound"] == "interpolated"
    assert rows["P1"]["bud_95per_date_bound"] == "right_censored"
    # The steady plant crosses later than the noisy one despite ending higher.
    assert rows["P2"]["bud_50per_date"] == "2026-03-07"
    assert rows["P2"]["bud_95per_date"] == "2026-03-12"
    assert rows["P1"]["bud_50per_date"] < rows["P2"]["bud_50per_date"]


# -- the counts the fraction is built from --------------------------------


def test_positive_detections_are_the_named_class_not_a_position_in_the_id_map(tmp_path):
    """The positive state is whichever class the trait names, wherever it sits in the registry's id
    space. A registry with a gap in its ids (the positive class at id 2, nothing at id 1) counts the
    same as a consecutive one.
    """
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p,
        [Annotation(subject="bud", geometry=BBox(1, 1, 4, 9), score=0.9,
                   attributes={"opening": "open"}),
         Annotation(subject="bud", geometry=BBox(6, 1, 9, 4), score=0.8,
                   attributes={"opening": "closed"}),
         Annotation(subject="bud", geometry=BBox(11, 2, 14, 12), score=0.7,
                   attributes={"opening": "open"})],
        40, 24,
    )

    scope = resolution.BucketScope(subject="bud", attribute="opening")
    total, positive, unclassified = phenology.count_by_class(
        p, SPARSE_ID_MAP, "open", scope=scope)

    assert (total, positive, unclassified) == (3, 2, 0)


def test_a_bucket_the_prediction_writer_produced_reads_back_with_its_own_classes(
    tmp_path, monkeypatch,
):
    """A prediction bucket written by the real export door reads back through the readers here with
    the classes the run actually decoded through: the sidecar's recorded map and each detection's own
    decoded name have to line up, or a fully classified bucket counts as unclassified.
    """
    pytest.importorskip("torch")
    from PIL import Image

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (120, 80), (110, 130, 90)).save(images_dir / "P1_2026-03-05.png")

    class FakePredictor:
        config = {"data": {"id_map": dict(SPARSE_ID_MAP), "subject": "bud",
                           "attribute": "state"}}

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, **kw):
            # torchvision labels are 1-indexed over the run's own 0-indexed class ids.
            return [{"image": str(p), "width": 120, "height": 80,
                     "boxes": [[4.0, 6.0, 18.0, 40.0], [30.0, 6.0, 44.0, 44.0],
                               [60.0, 10.0, 74.0, 52.0]],
                     "scores": [0.91, 0.84, 0.77], "labels": [3, 1, 3], "count": 3}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)
    from tcip_mcp.tools.inference_tools import run_inference

    bucket = tmp_path / "preds"
    res = run_inference(str(ckpt), str(images_dir), output_dir=str(bucket), tile=False)
    assert "error" not in res, res

    id_map = phenology.bucket_id_map(bucket)
    assert id_map == SPARSE_ID_MAP
    scope = resolution.bucket_scope(bucket)
    counts = phenology.count_by_class(
        bucket / "P1_2026-03-05.json", id_map, "open", scope=scope)
    assert counts == (3, 2, 0)


# -- the delivered CSV ----------------------------------------------------


@pytest.mark.usefixtures("seed_bud_operationalization")
def test_delivered_csv_marks_a_milestone_the_first_capture_only_bounds(tmp_path):
    """A plant already at half its buds on the first capture ships that date with its bound, so a
    breeder reading the CSV can tell an upper bound from a date the observations measured. The later
    dip below the target does not move the delivered date to the re-crossing.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    root = tmp_path / "ds"
    counts = {"2026-03-01": (1, 1), "2026-03-05": (1, 4), "2026-03-09": (4, 1)}
    buckets = {d: root / "predictions" / "run" / d for d in counts}
    for d, (pos, neg) in counts.items():
        bucket = buckets[d]
        # Predictions land before the record is filed, so the covered digest matches what delivery recomputes.
        _write_preds(bucket, f"P1_{d}", _states(pos, neg))
        _write_sidecar(bucket, SPARSE_ID_MAP, dataset_root=root)
    _write_classifier_sidecar(buckets["2026-03-01"], dataset_root=root, trait="bud_opening")
    from tests._binding_fixtures import write_plant_mapping

    mapping_name = "valley"
    write_plant_mapping(tmp_path, mapping_name, {
        d: [{"stem": f"P1_{d}", "plot_name": "P1", "accession_name": "acc-noisy"}]
        for d in counts
    }, dataset_root=root)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_phenology_milestones(
        trait="bud_opening",
        mapping_name=mapping_name,
        predictions_by_date={d: str(buckets[d]) for d in counts},
        output_csv_path=str(out_csv),
        classifier_pred_dirs=[str(buckets["2026-03-01"])],
        operating_point_validated="held_out_annotations",
    )

    assert "error" not in res, res
    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["plant_id"] == "P1"
    assert rows[0]["bud_50per_date"] == "2026-03-01"
    assert rows[0]["bud_50per_date_bound"] == "left_censored"

"""Tests for the canonical phenology measurement module.

The positive-state fraction is the share of a plant's detected objects that are in the trait's
positive/measured state, where that state is a classifier call (never a geometric proxy). These
tests pin the authoritative trait definitions:

    bud_05/50/95per_date  = dates the open fraction crosses 5/50/95%
    bud_majority_date     = date most buds are open (the majority-label alias) = the 95% crossing

and the id_map-key-membership coverage rule that decides whether a prediction
bucket ever assessed the trait's positive class at all, and the per-detection membership check
within it.
"""

from __future__ import annotations

import sys

import pytest
from pathlib import Path

_MCP_SRC = Path(__file__).resolve().parents[1] / "packages" / "tcip-mcp" / "src"
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC))

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines import resolution  # noqa: E402
from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.pipelines.postprocessing import phenology  # noqa: E402
from tcip_mcp.pipelines.postprocessing.plant_mapping import MappingBuild  # noqa: E402
from tcip_mcp.pipelines.resolution import Acknowledgment  # noqa: E402
from tcip_mcp.project_paths import project_state_dir  # noqa: E402
from tcip_mcp.traits import TraitRevision  # noqa: E402
from tests._trait_fixtures import BUD_OPENING, entry, propose_and_confirm  # noqa: E402
from tests.test_phenology_tools import _write_preds  # noqa: E402


def _revision(project_root: Path) -> TraitRevision:
    """BUD_OPENING as a confirmed revision at ``project_root``, the way a delivery door holds it."""
    return propose_and_confirm(project_root, BUD_OPENING)

# A writer-level unit test's own placeholder disclosure, built through delivery_disclosure itself
# so it carries every key the writer's cells read even as that shape grows.
_NO_MAPPING = MappingBuild(
    name="none", dataset_root="", dataset_id="", built_by="test", built_at="",
    dates_requested=None, dates=[], nn_tolerance_m={"value": 0.0, "source": "stated"},
    plant_registry={"name": "unregistered", "digest": "0" * 64},
    capture_identity={}, capture_digests={}, unreadable={}, assignments={},
    record_sha256="0" * 16,
).delivery_disclosure({"captures_unverified": [], "plant_csvs_unverified": []}, [])


def _sidecar(dir_path: Path, id_map: dict | None, *, subject: str | None = "bud",
            attribute: str | None = None) -> None:
    """Write a bucket's operating_point.json, filed under the project holding ``dir_path``,
    exactly the way run_inference does: the only fact count_by_class reads to decide whether/how
    a bucket was classified."""
    from tcip_mcp.pipelines.resolution import write_sidecar

    write_sidecar(dir_path, {"scope": {"subject": subject, "attribute": attribute, "id_map": id_map}},
                  project=dir_path.parent)


# ── date helpers (unchanged) ──────────────────────────────────────────────


def test_date_key_orders_chronologically():
    assert phenology.date_key("2024-05-01") < phenology.date_key("2024-05-15")
    assert phenology.date_key("2024-05-15") < phenology.date_key("2024-06-01")


def test_date_key_malformed_sorts_first():
    assert phenology.date_key("undated") == (0, 0, 0)
    assert phenology.date_key("2024-13") == (0, 0, 0)
    assert phenology.date_key("not-a-date-x") == (0, 0, 0)


def test_date_key_rejects_out_of_range_dates():
    assert phenology.date_key("2026-13-01") == (0, 0, 0)
    assert phenology.date_key("2026-02-30") == (0, 0, 0)


def test_crossing_date_does_not_crash_on_malformed_date():
    series = [
        ("2026-02-01", 0.0),
        ("2026-13-01", 0.9),  # out-of-range month, excluded
        ("2026-02-15", 1.0),
    ]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2026-02-08"
    assert c.bound == "interpolated"
    assert phenology.positive_onset_date(series) == "2026-02-15"


def test_real_points_drops_undated_and_sorts():
    series = [("2024-05-15", 0.5), ("undated", 0.9), ("2024-05-01", 0.1)]
    pts = phenology._real_points(series)
    assert [d for d, _ in pts] == ["2024-05-01", "2024-05-15"]


# ── crossings (censoring bound disclosed) ─────────────────────────────


def test_crossing_interpolates_between_dates():
    series = [("2024-05-01", 0.0), ("2024-05-11", 1.0)]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2024-05-06"
    assert c.bound == "interpolated"
    assert c.gap_days == 10


def test_crossing_first_point_already_at_target_is_left_censored():
    # A left-censored crossing must be flagged, not returned as a bare tuple indistinguishable
    # from a real single-date measurement.
    series = [("2024-05-01", 0.2), ("2024-05-05", 0.8)]
    c = phenology.crossing_date(series, 0.05)
    assert c.date == "2024-05-01"
    assert c.bound == "left_censored"


def test_crossing_exact_match_is_not_censored():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.50)]
    c = phenology.crossing_date(series, 0.50)
    assert c.date == "2024-05-05"
    assert c.bound == "exact"


def test_crossing_never_reached_is_right_censored():
    # The last observed point still hasn't met the target -> the true
    # crossing, if it happens at all, is after this date. Distinguishable from "no observations at
    # all" (which stays None, see below).
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.3)]
    c = phenology.crossing_date(series, 0.95)
    assert c.date == "2024-05-05"
    assert c.bound == "right_censored"
    assert c.gap_days is None


def test_crossing_no_observations_is_none():
    assert phenology.crossing_date([], 0.95) is None


def test_opening_onset_is_first_nonzero_date():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0), ("2024-05-09", 0.10), ("2024-05-13", 0.60)]
    assert phenology.positive_onset_date(series) == "2024-05-09"


def test_opening_onset_none_when_all_zero():
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0)]
    assert phenology.positive_onset_date(series) is None


def test_plant_milestones_returns_four_dates_and_bounds():
    series = [("2024-05-01", 0.0), ("2024-05-06", 0.04), ("2024-05-11", 0.50), ("2024-05-21", 1.0)]
    m = phenology.plant_milestones(series, BUD_OPENING)
    assert {"bud_05per_date", "bud_50per_date", "bud_95per_date", "bud_majority_date"} <= set(m)
    assert m["bud_majority_date"] == m["bud_95per_date"]
    assert m["bud_50per_date"] == "2024-05-11"
    assert m["bud_50per_date_bound"] == "exact"


def test_plant_milestones_requires_spec_no_bud_opening_fallback():
    # No silent default: a caller that forgets to pass spec must fail loudly.
    import pytest

    with pytest.raises(TypeError, match="'spec'"):
        phenology.plant_milestones([("2024-05-01", 0.5)])  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to spec


def test_milestone_date_columns_is_proper_subset_of_full_columns():
    cols = phenology.phenology_csv_columns(BUD_OPENING)
    milestone_cols = phenology.milestone_date_columns(BUD_OPENING)
    assert set(milestone_cols) <= set(cols)
    assert "plant_id" not in milestone_cols
    assert "positive_state_classifier_validated" not in milestone_cols


# ── count_by_class: the coverage mechanism ─────────────────────


def test_count_by_class_bare_detector_bucket_refuses_never_full_coverage(tmp_path):
    # A single-class detector's id_map ({"bud": 0}) has no attribute axis at all: "bud" is not
    # the trait's positive value, so this must not be scored classified.
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.9),
            Annotation(subject="bud", geometry=BBox(4, 4, 6, 6), score=0.8)],
        8, 8,
    )
    scope = ClassScope(subject="bud", id_map={"bud": 0})
    total, positive, unclassified = phenology.count_by_class(p, "open", scope=scope)
    assert (total, positive, unclassified) == (2, 0, 2)  # whole bucket unclassified, not full coverage


def test_count_by_class_wrong_axis_bucket_refuses(tmp_path):
    # A run classified on a different attribute of the same subject (e.g. damage severity); its
    # value set doesn't include the trait's positive value, so it must refuse, not be miscounted.
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.9,
                       attributes={"damage": "mild"})], 8, 8,
    )
    scope = ClassScope(subject="bud", attribute="damage",
                       id_map={"none": 0, "mild": 1, "severe": 2})
    total, positive, unclassified = phenology.count_by_class(p, "open", scope=scope)
    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_absent_id_map_refuses(tmp_path):
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.9,
                       attributes={"opening": "open"})], 8, 8,
    )
    scope = ClassScope(subject="bud", attribute="opening")
    total, positive, unclassified = phenology.count_by_class(p, "open", scope=scope)
    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_classified_bucket_splits_positive_negative(tmp_path):
    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.9,
                       attributes={"opening": "open"}),
            Annotation(subject="bud", geometry=BBox(4, 4, 6, 6), score=0.8,
                       attributes={"opening": "closed"}),
            Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.7,
                       attributes={"opening": "open"})],
        8, 8,
    )
    scope = ClassScope(subject="bud", attribute="opening", id_map={"closed": 0, "open": 1})
    total, positive, unclassified = phenology.count_by_class(p, "open", scope=scope)
    assert (total, positive, unclassified) == (3, 2, 0)


def test_count_by_class_foreign_record_within_classified_bucket_refuses(tmp_path):
    # A record carrying no value under the classified attribute (a stale bare-detector document)
    # refuses by name rather than reading as a classified negative.
    from tcip_annotation.json_io import ClassifiedRecordRefused

    p = tmp_path / "img.json"
    json_io.write_annotations(
        p, [Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.9,
                       attributes={"opening": "open"}),
            Annotation(subject="bud", geometry=BBox(4, 4, 6, 6), score=0.8)],
        8, 8,
    )
    scope = ClassScope(subject="bud", attribute="opening", id_map={"closed": 0, "open": 1})
    with pytest.raises(ClassifiedRecordRefused, match="carries no value under attribute"):
        phenology.count_by_class(p, "open", scope=scope)


def test_count_by_class_missing_file_reads_as_empty():
    # count_by_class degrades gracefully on a missing path (json_io.read_annotations answers []);
    # per_plant_series never calls it on one, checking is_file() and tracking n_missing itself.
    total, positive, unclassified = phenology.count_by_class(
        Path("does-not-exist.json"), "open", scope=None)
    assert (total, positive, unclassified) == (0, 0, 0)


# ── per_plant_series / per_plant_phenology: bucket-level + expected-coverage ────────────────────


class _Assignment:
    def __init__(self, stem, plot_name, accession_name):
        self.stem = stem
        self.plot_name = plot_name
        self.accession_name = accession_name


def test_per_plant_phenology_builds_fraction_series_when_classified(tmp_path):
    d1 = tmp_path / "2024-05-01"
    d2 = tmp_path / "2024-05-15"
    _write_preds(d1, "P1_a", ["closed", "closed"], attribute="opening")
    _sidecar(d1, {"closed": 0, "open": 1}, attribute="opening")
    _write_preds(d2, "P1_b", ["open", "open"], attribute="opening")
    _sidecar(d2, {"closed": 0, "open": 1}, attribute="opening")
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }
    preds = {"2024-05-01": str(d1), "2024-05-15": str(d2)}

    out = phenology.per_plant_phenology(mapping, preds, spec=BUD_OPENING,
                                        plants=["P1"])

    assert out["positive_class_assessed"] is True
    row = out["rows"][0]
    assert row["plant_id"] == "P1"
    assert row["accession"] == "acc-9"
    assert row["n_dates"] == 2
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert [s["n_positive"] for s in row["series"]] == [0, 2]
    assert row["bud_95per_date"] is not None


def test_per_plant_phenology_bare_detector_bucket_refuses_whole_delivery(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _write_preds(d1, "P1_a", ["bud"], attribute=None)
    _sidecar(d1, {"bud": 0})
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}
    preds = {"2024-05-01": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, spec=BUD_OPENING,
                                        plants=["P1"])

    assert out["positive_class_assessed"] is False
    row = out["rows"][0]
    assert row["n_dates_unclassified"] == 1
    assert row["bud_50per_date"] is None


def test_per_plant_phenology_missing_image_is_disclosed_not_a_zero(tmp_path):
    # A stem the mapping names with no prediction file must not read as an
    # observed zero (which would count as "classified, 0/0" and silently pass coverage).
    d1 = tmp_path / "2024-05-01"
    d1.mkdir(parents=True, exist_ok=True)
    _sidecar(d1, {"closed": 0, "open": 1}, attribute="opening")
    # no P1_a.json written: the mapping names it but nothing was ever inferred for it
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}
    preds = {"2024-05-01": str(d1)}

    out = phenology.per_plant_phenology(mapping, preds, spec=BUD_OPENING,
                                        plants=["P1"])

    row = out["rows"][0]
    assert row["series"][0]["n_missing"] == 1
    assert row["series"][0]["ratio"] is None
    assert row["n_dates_missing_images"] == 1
    assert out["positive_class_assessed"] is False  # nothing anywhere was actually classified
    assert row["bud_50per_date"] is None


def test_per_plant_phenology_multi_date_and_excludes_plant_with_one_bad_date(tmp_path):
    # A plant's milestones require every date to be classified; one
    # unclassified or missing date excludes the whole plant's milestones, disclosed, not silently
    # computed from the subset that happened to be usable.
    d1 = tmp_path / "2024-05-01"
    d2 = tmp_path / "2024-05-15"
    _write_preds(d1, "P1_a", ["open"], attribute="opening")
    _sidecar(d1, {"closed": 0, "open": 1}, attribute="opening")
    _write_preds(d2, "P1_b", ["bud"], attribute=None)  # bare-detector date, unclassified
    _sidecar(d2, {"bud": 0})
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }
    preds = {"2024-05-01": str(d1), "2024-05-15": str(d2)}

    out = phenology.per_plant_phenology(mapping, preds, spec=BUD_OPENING,
                                        plants=["P1"])

    row = out["rows"][0]
    assert row["n_dates"] == 2
    assert row["n_dates_unclassified"] == 1
    # The whole plant's milestones are None: one bad date excludes it, not a partial computation.
    assert row["bud_05per_date"] is None
    assert row["bud_95per_date"] is None
    # But at least one date elsewhere was classified, so the delivery-level flag is still True,
    # distinguishing "wired, some gaps" from "never wired at all".
    assert out["positive_class_assessed"] is True


def test_per_plant_series_accepts_dict_assignments(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _write_preds(d1, "P1_a", ["open"], attribute="opening")
    _sidecar(d1, {"closed": 0, "open": 1}, attribute="opening")
    mapping = {"2024-05-01": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}]}
    preds = {"2024-05-01": str(d1)}
    per_plant = phenology.per_plant_series(mapping, preds, positive_value="open", plants=["P1"])
    assert "P1" in per_plant
    assert per_plant["P1"]["accession"] == "acc-9"
    assert per_plant["P1"]["series"][0][:3] == ("2024-05-01", 1, 1)  # total=1, positive=1


# ── resolve_positive_class_id ─────────────────────────────────────────────


def test_resolve_positive_class_id_from_bucket_id_map(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _sidecar(d1, {"closed": 0, "open": 1}, attribute="opening")
    cid, msg = phenology.resolve_positive_class_id(BUD_OPENING, {"2024-05-01": str(d1)})
    assert cid == 1
    assert "resolved" in msg


def test_resolve_positive_class_id_no_bucket_has_it_refuses(tmp_path):
    d1 = tmp_path / "2024-05-01"
    _sidecar(d1, {"bud": 0})
    cid, msg = phenology.resolve_positive_class_id(BUD_OPENING, {"2024-05-01": str(d1)})
    assert cid is None
    assert "never assessed" in msg


# ── write_phenology_csv: the gate it runs, the cells it composes, the event it records ────


def _real_delivery_flags(tmp_path: Path):
    """Two validated buckets, plus the flags and reconciliations a real reconciliation produces
    over them, the way both phenology delivery doors build their own before calling this writer."""
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity, reconcile_classifier_validity, reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )
    from tests.test_phenology_tools import _delivery_setup

    _mapping_name, d1, d2 = _delivery_setup(
        tmp_path, experiment_id="exp-producer", checkpoint_sha256="a" * 64)
    predictions_by_date = {"2026-02-11": str(d1), "2026-03-09": str(d2)}
    pred_dirs = list(predictions_by_date.values())
    recon = reconcile_operating_point_validity(pred_dirs, trait="bud_opening", project=tmp_path)
    classifier_recon = reconcile_classifier_validity([str(d1)], project=tmp_path)
    classifier_state, note = bind_classifier_validity(
        classifier_recon["validated"], [str(d1)], pred_dirs, trait="bud_opening", project=tmp_path)
    tile_recon = reconcile_tile_size_validity(pred_dirs, project=tmp_path)
    flags = phenology.phenology_delivery_flags(classifier_state, recon["validated"], tile_recon)
    document_reconciliations = {
        "operating_point": recon,
        "classifier_operating_point": {
            **classifier_recon, "bound_validated": classifier_state, "delivery_note": note,
        },
    }
    dimension_reconciliations = {"tile_size": tile_recon}
    return flags, document_reconciliations, dimension_reconciliations, predictions_by_date


def test_write_phenology_csv_refuses_and_writes_nothing_when_a_dimension_is_unvalidated(tmp_path):
    """The writer runs its own delivery gate before opening the file, the way its sibling writers
    (``export_aggregated_csv``, ``export_detection_csv``) do; a call whose flags do not clear
    refuses rather than delivering a silent bare number."""
    with pytest.raises(ValueError, match="unvalidated dimension"):
        phenology.write_phenology_csv(
            "test", [], tmp_path / "out.csv", _revision(tmp_path),
            flags={"classifier": None, "operating_point": None}, acknowledgment=None,
            document_reconciliations={}, producer={},
            dimension_reconciliations={}, predictions_by_date={},
            project=tmp_path, plant_mapping=_NO_MAPPING)
    assert not (tmp_path / "out.csv").exists()


def test_write_phenology_csv_refuses_a_count_only_reconciliation_with_nothing_on_disk(tmp_path):
    """A phenology delivery declares both operating_point and classifier_operating_point to the
    event writer; a caller reconciling the count alone is refused here, before the gate runs and
    before anything is written, rather than reaching the event writer after the file exists."""
    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    count_only = {"operating_point": document_reconciliations["operating_point"]}

    with pytest.raises(ValueError, match="classifier_operating_point"):
        phenology.write_phenology_csv(
            "test", [], tmp_path / "out.csv", _revision(tmp_path), flags=flags, acknowledgment=None,
            document_reconciliations=count_only, producer={},
            dimension_reconciliations=dimension_reconciliations,
            predictions_by_date=predictions_by_date, project=tmp_path,
            plant_mapping=_NO_MAPPING)
    assert not (tmp_path / "out.csv").exists()


def test_write_phenology_csv_needs_no_declared_document_when_predictions_by_date_is_empty(
    tmp_path,
):
    """The rail must admit valid work: with nothing to reconcile, the new missing-entry check
    does not fire, so a legitimate acknowledged, bucket-less call still writes rather than being
    refused for a document it never had a chance to reconcile."""
    phenology.write_phenology_csv(
        "test", [], tmp_path / "out.csv", _revision(tmp_path),
        flags={"classifier": None, "operating_point": None},
        acknowledgment=Acknowledgment(acknowledged_by="user:tester", reason="nothing to reconcile"),
        document_reconciliations={}, producer={},
        dimension_reconciliations={}, predictions_by_date={},
        project=tmp_path, plant_mapping=_NO_MAPPING)
    assert (tmp_path / "out.csv").exists()
    assert [r["door"] for r in resolution.read_delivery_events(tmp_path)] == ["test"]


def test_write_phenology_csv_records_a_none_conf_for_a_bucket_with_no_operating_point_sidecar(
    tmp_path,
):
    """confs maps every bucket in pred_dirs to its own stamp's numeric conf, or None for a bucket
    whose operating_point.json is missing entirely, whatever the delivery's own validated state:
    a fact about what the predictions were produced at, not about whether that value cleared."""
    from tcip_mcp.pipelines.resolution import (
        bind_classifier_validity, reconcile_classifier_validity, reconcile_operating_point_validity,
        reconcile_tile_size_validity,
    )
    from tests.test_phenology_tools import (
        ID_MAP, _bucket, _ds_root, _write_classifier_sidecar, _write_op_sidecar, _write_preds,
    )

    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    _write_op_sidecar(d1, dataset_root=root, validated=True, id_map=ID_MAP, conf=0.7)
    # d2 carries predictions but no operating_point.json at all: a missing stamp.
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    predictions_by_date = {"2026-02-11": str(d1), "2026-03-09": str(d2)}
    pred_dirs = list(predictions_by_date.values())

    recon = reconcile_operating_point_validity(pred_dirs, trait="bud_opening", project=tmp_path)
    assert recon["missing_sidecars"] == [str(d2)]
    assert recon["confs"] == {str(d1): 0.7, str(d2): None}

    classifier_recon = reconcile_classifier_validity([str(d1)], project=tmp_path)
    classifier_state, note = bind_classifier_validity(
        classifier_recon["validated"], [str(d1)], pred_dirs, trait="bud_opening", project=tmp_path)
    tile_recon = reconcile_tile_size_validity(pred_dirs, project=tmp_path)
    flags = phenology.phenology_delivery_flags(classifier_state, recon["validated"], tile_recon)

    phenology.write_phenology_csv(
        "test.missing_stamp", [], tmp_path / "out.csv", _revision(tmp_path), flags=flags,
        acknowledgment=Acknowledgment(acknowledged_by="user:tester", reason="missing sidecar"),
        document_reconciliations={
            "operating_point": recon,
            "classifier_operating_point": {
                **classifier_recon, "bound_validated": classifier_state, "delivery_note": note,
            },
        },
        producer={}, dimension_reconciliations={"tile_size": tile_recon},
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)

    records = [
        r for r in resolution.read_delivery_events(tmp_path) if r["door"] == "test.missing_stamp"
    ]
    assert len(records) == 1, records
    assert records[0]["document_reconciliations"]["operating_point"]["confs"] == {
        str(d1): 0.7, str(d2): None}


def test_write_phenology_csv_refuses_when_flags_carry_no_classifier_dimension(tmp_path):
    """``gate.stamp['classifier']`` is a public entry point's own index into the caller's flags: a
    caller composing them some other way, one that leaves the key out, refuses naming what's
    missing rather than raising a bare ``KeyError``. The same flags, classifier included, still
    deliver, so the guard costs nothing on the call it was built to admit."""
    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    incomplete = {k: v for k, v in flags.items() if k != "classifier"}

    with pytest.raises(ValueError, match="classifier"):
        phenology.write_phenology_csv(
            "test", [], tmp_path / "out.csv", _revision(tmp_path), flags=incomplete, acknowledgment=None,
            document_reconciliations=document_reconciliations, producer={},
            dimension_reconciliations=dimension_reconciliations,
            predictions_by_date=predictions_by_date, project=tmp_path,
            plant_mapping=_NO_MAPPING)
    assert not (tmp_path / "out.csv").exists()

    cells = phenology.write_phenology_csv(
        "test", [], tmp_path / "out.csv", _revision(tmp_path), flags=flags, acknowledgment=None,
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)
    assert cells["positive_state_classifier_validated"]


def test_write_phenology_csv_floors_operating_point_when_tile_size_is_operative_and_unvalidated(
    tmp_path,
):
    """A tiled bucket with no persisted training geometry floors the delivery's whole
    operating-point column even though the count operating point itself cleared:
    ``column_stamp`` floors any gated dimension outside its own column, and the tile dimension has
    no column of its own. Built from a real reconciliation over real sidecars, through
    ``phenology_delivery_flags``, the way both delivery doors build their own flags."""
    from tcip_mcp.pipelines.resolution import (
        VALIDATED_FALSE, bind_classifier_validity, reconcile_classifier_validity,
        reconcile_operating_point_validity, reconcile_tile_size_validity,
    )
    from tests.test_phenology_tools import (
        ID_MAP, _bucket, _ds_root, _tiled, _write_classifier_sidecar, _write_op_sidecar, _write_preds,
    )

    root = _ds_root(tmp_path)
    d1, d2 = _bucket(tmp_path, "2026-02-11"), _bucket(tmp_path, "2026-03-09")
    _write_preds(d1, "P1_a", ["closed"])
    _write_preds(d2, "P1_b", ["open"])
    for d in (d1, d2):
        _write_op_sidecar(d, dataset_root=root, validated=True, id_map=ID_MAP,
                          tile_size_prov=_tiled(VALIDATED_FALSE))
    _write_classifier_sidecar(d1, dataset_root=root, validated=True, trait="bud_opening")
    predictions_by_date = {"2026-02-11": str(d1), "2026-03-09": str(d2)}
    pred_dirs = list(predictions_by_date.values())

    recon = reconcile_operating_point_validity(pred_dirs, trait="bud_opening", project=tmp_path)
    classifier_recon = reconcile_classifier_validity([str(d1)], project=tmp_path)
    classifier_state, note = bind_classifier_validity(
        classifier_recon["validated"], [str(d1)], pred_dirs, trait="bud_opening", project=tmp_path)
    tile_recon = reconcile_tile_size_validity(pred_dirs, project=tmp_path)
    assert tile_recon["operative"] and tile_recon["validated"] == VALIDATED_FALSE
    assert recon["validated"] != VALIDATED_FALSE  # the count operating point itself cleared

    flags = phenology.phenology_delivery_flags(classifier_state, recon["validated"], tile_recon)

    cells = phenology.write_phenology_csv(
        "test", [], tmp_path / "out.csv", _revision(tmp_path), flags=flags,
        acknowledgment=Acknowledgment(acknowledged_by="user:tester", reason="test acknowledgment"),
        document_reconciliations={
            "operating_point": recon,
            "classifier_operating_point": {
                **classifier_recon, "bound_validated": classifier_state, "delivery_note": note,
            },
        },
        producer={}, dimension_reconciliations={"tile_size": tile_recon},
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)

    assert cells["operating_point_validated"] == VALIDATED_FALSE


def test_write_phenology_csv_records_the_delivery_event_without_a_door_calling_it(tmp_path):
    """The delivery event is recorded inside the writer itself: a caller that calls the writer
    directly, never through ``deliver_phenology_milestones``, still leaves the record behind."""
    from tcip_mcp.pipelines import resolution

    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    phenology.write_phenology_csv(
        "test.direct_writer_call", [], out_csv, _revision(tmp_path), flags=flags, acknowledgment=None,
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)

    records = [r for r in resolution.read_delivery_events(tmp_path)
               if r["door"] == "test.direct_writer_call"]
    assert len(records) == 1, records
    assert records[0]["output_path"] == str(out_csv)
    assert records[0]["trait"] == "bud_opening"


def test_write_phenology_csv_fully_validated_acknowledgment_leaves_the_tail_and_event_agreeing(
    tmp_path,
):
    """A caller that passes an ``Acknowledgment`` on a delivery every dimension actually clears
    gets a gate that discards it (nothing needed acknowledging); the writer records that discarded
    outcome on the event too, rather than the caller's original object verbatim, so the CSV tail's
    blank ``acknowledged_by``/``acknowledgment_reason`` and the event's own fields can never
    disagree about whether this delivery rested on one."""
    import tcip_store as ts
    from tcip_mcp.pipelines import resolution

    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    cells = phenology.write_phenology_csv(
        "test.fully_validated_ack", [], out_csv, _revision(tmp_path), flags=flags,
        acknowledgment=Acknowledgment(acknowledged_by="user:tester", reason="just in case"),
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)

    assert cells["acknowledged_by"] is None
    assert cells["acknowledgment_reason"] is None

    scope = project_state_dir(tmp_path)
    keys = ts.keys(resolution.DELIVERY_EVENTS_STORE, str(scope))
    records = [ts.read(k) for k in keys if ts.read(k)["door"] == "test.fully_validated_ack"]
    assert len(records) == 1, records
    assert records[0]["acknowledged_by"] is None
    assert records[0]["acknowledgment_reason"] is None


def test_write_phenology_csv_cells_are_exactly_the_schemas_provenance_columns(tmp_path):
    """There is no ``stamp`` parameter: the writer composes its own provenance cells and
    returns them, so this pins that the set it returns is exactly the schema's provenance columns."""
    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))

    cells = phenology.write_phenology_csv(
        "test", [], tmp_path / "out.csv", _revision(tmp_path), flags=flags, acknowledgment=None,
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)

    assert set(cells) == set(phenology.PROVENANCE_COLUMNS)


def test_write_phenology_curve_csv_writes_the_curve_schema(tmp_path):
    """The curve table gets its own writer, sharing the same gate/cells/event machinery as the
    milestone table."""
    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    row = {"plant_id": "P1", "accession": "acc-9", "date": "2026-02-11", "n_images": 1,
          "n_total": 2, "n_positive": 1, "n_unclassified": 0, "n_missing": 0, "ratio": 0.5}

    phenology.write_phenology_curve_csv(
        "test", [row], tmp_path / "curve.csv", _revision(tmp_path), flags=flags, acknowledgment=None,
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)

    header = (tmp_path / "curve.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == phenology.curve_csv_columns()


def test_write_phenology_csv_carries_every_milestone_bound(tmp_path):
    """plant_milestones emits a `_bound` beside each milestone date
    (exact / interpolated / left_censored); those must be in phenology_csv_columns, or the writer's
    extrasaction="ignore" drops them all. A left-censored crossing (the first observation already
    met the target, so the date is only an upper bound) must not ship indistinguishable from a
    measured one, which would be a precision claim the data does not support."""
    # First observed point already at 100% -> every crossing is left-censored.
    series = [("2026-02-11", 1.0), ("2026-02-20", 1.0)]
    milestones = phenology.plant_milestones(series, BUD_OPENING)
    assert milestones["bud_05per_date_bound"] == "left_censored"

    row = {"plant_id": "P1", "n_dates": 2, "n_observed_dates": 2, **milestones}
    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    phenology.write_phenology_csv(
        "test", [row], tmp_path / "out.csv", _revision(tmp_path), flags=flags, acknowledgment=None,
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)
    written = (tmp_path / "out.csv").read_text(encoding="utf-8")
    header = written.splitlines()[0].split(",")
    values = written.splitlines()[1].split(",")

    for col in phenology.milestone_date_columns(BUD_OPENING):
        assert f"{col}_bound" in header, col
        assert values[header.index(f"{col}_bound")] == "left_censored", col


_SPEC_SHAPES = [
    # An entry stating both majority fields empty, which once made the phantom names doubly
    # malformed (`b__date`).
    entry("b", BUD_OPENING.delivers, milestone_fractions=(0.1, 0.9), phenology_prefix="b"),
    # A majority label with no majority milestone: a column must not be built from the label
    # alone without a milestone to source it.
    entry("b", BUD_OPENING.delivers, milestone_fractions=(0.1, 0.9), phenology_prefix="b",
          majority_label="peak"),
    # A majority milestone naming a crossing the trait does not compute: the column is real (the
    # spec names it) and its value is honestly None, which is not the same as a phantom.
    entry("b", BUD_OPENING.delivers, milestone_fractions=(0.1, 0.9), phenology_prefix="b",
          majority_milestone="95per", majority_label="peak"),
    BUD_OPENING,
]


@pytest.mark.parametrize("spec", _SPEC_SHAPES, ids=lambda s: f"{s.name}-{s.majority_milestone or 'nomajority'}")
def test_phenology_csv_columns_name_no_column_without_a_producer(spec):
    """Every trait-prefixed column the schema names must be filled by a producer, or the delivered
    CSV carries a permanently-blank column. The schema and the producer share ``_milestone_columns``,
    so this holds by construction for any spec shape, including one whose ``majority_milestone``
    is unset (unlike BUD_OPENING's, which is always set).
    """
    series = [("2026-02-01", 0.0), ("2026-02-10", 0.5), ("2026-02-20", 1.0)]
    produced = set(phenology.plant_milestones(series, spec))
    schema = set(phenology.phenology_csv_columns(spec))
    prefixed = {c for c in schema if c.startswith(spec.phenology_prefix + "_")}
    assert prefixed - produced == set()
    assert produced - schema == set()  # and nothing computed is silently dropped
    assert not any(c.startswith(f"{spec.phenology_prefix}__") for c in schema)


def test_excluded_plant_carries_the_same_milestone_keys_as_an_included_one(tmp_path):
    """A plant excluded from milestone computation must carry the same row shape as an included one,
    including each milestone date's ``*_date_bound`` companion, not just ``milestone_date_columns``'s
    bare dates.
    """
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    id_map = {"closed": 0, "open": 1}
    for d in (d1, d2):
        _sidecar(d, id_map, attribute="opening")
    _write_preds(d1, "GOOD", ["open", "closed"], attribute="opening")
    _write_preds(d2, "GOOD", ["open", "open"], attribute="opening")
    _write_preds(d1, "BAD", ["open", "open"], attribute="opening")
    # BAD's second date is never predicted on: the missing image excludes its milestones.
    mapping = {
        "2026-02-11": [_Assignment("GOOD", "GOOD", "a"), _Assignment("BAD", "BAD", "b")],
        "2026-03-09": [_Assignment("GOOD", "GOOD", "a"), _Assignment("BAD", "BAD", "b")],
    }
    res = phenology.per_plant_phenology(
        mapping, {"2026-02-11": str(d1), "2026-03-09": str(d2)},
        spec=BUD_OPENING, plants=["GOOD", "BAD"])
    by_plant = {r["plant_id"]: r for r in res["rows"]}
    assert by_plant["BAD"]["n_dates_missing_images"] == 1  # genuinely excluded
    assert set(by_plant["GOOD"]) == set(by_plant["BAD"])
    assert "bud_95per_date_bound" in by_plant["BAD"]


def test_per_plant_series_counts_the_images_the_mapping_names(tmp_path):
    """``n_images`` is derived from every image the mapping names for a (plant, date), not asserted
    by a consumer, so a breeder auditing coverage can tell a well-sampled plant from a single-photo
    one.
    """
    d = tmp_path / "2026-02-11"
    _sidecar(d, {"closed": 0, "open": 1}, attribute="opening")
    for i in range(3):
        _write_preds(d, f"IMG{i}", ["open", "closed"], attribute="opening")
    mapping = {"2026-02-11": [_Assignment(f"IMG{i}", "P1", "a") for i in range(3)]
               + [_Assignment("GONE", "P1", "a")]}  # named, no prediction file
    per_plant = phenology.per_plant_series(mapping, {"2026-02-11": str(d)},
                                            positive_value="open", plants=["P1"])
    series = per_plant["P1"]["series"]
    (_date, total, positive, unclassified, missing, n_images) = series[0]
    assert (total, positive, unclassified, missing) == (6, 3, 0, 1)
    # 4 images named for this (plant, date), the coverage the entry summarizes, of which one is
    # missing, not 3 (the files that happened to exist).
    assert n_images == 4


def test_per_plant_series_excludes_unattributed_assignments_from_coverage(tmp_path):
    """An assignment with no ``plot_name`` (an image the plant-mapping step could not assign) is
    silently dropped from every plant's coverage; how often that happens is disclosed once, at
    delivery scope, by ``plant_mapping.MappingBuild.unattributed``, never recomputed here."""
    d = tmp_path / "2026-02-11"
    _sidecar(d, {"closed": 0, "open": 1}, attribute="opening")
    _write_preds(d, "P1_a", ["open"], attribute="opening")
    mapping = {"2026-02-11": [
        _Assignment("P1_a", "P1", "acc-9"),
        _Assignment("STRAY", None, None),  # no plot_name: never assigned to any plant
    ]}
    per_plant = phenology.per_plant_series(
        mapping, {"2026-02-11": str(d)}, positive_value="open", plants=["P1"])
    assert list(per_plant) == ["P1"]


def test_per_plant_phenology_excludes_unattributed_assignments_from_rows(tmp_path):
    """``per_plant_phenology`` never emits a row for an unattributed assignment, and carries no
    unattributed count of its own: that disclosure is ``plant_mapping.MappingBuild.unattributed``'s,
    at delivery scope, not a per-call return value."""
    d = tmp_path / "2026-02-11"
    _sidecar(d, {"closed": 0, "open": 1}, attribute="opening")
    _write_preds(d, "P1_a", ["open"], attribute="opening")
    mapping = {"2026-02-11": [
        _Assignment("P1_a", "P1", "acc-9"),
        _Assignment("STRAY1", None, None),
        _Assignment("STRAY2", "", None),
    ]}
    out = phenology.per_plant_phenology(
        mapping, {"2026-02-11": str(d)}, spec=BUD_OPENING, plants=["P1"])
    assert [r["plant_id"] for r in out["rows"]] == ["P1"]
    assert "n_images_unmapped" not in out


def test_write_phenology_csv_carries_n_observed_dates(tmp_path):
    # A plant fully classified/observed but with zero real detections on every date must be
    # distinguishable from one with real detection data, so this column must reach the CSV.
    row = {"plant_id": "P1", "accession": "acc-9", "n_dates": 2, "n_observed_dates": 1,
          "n_dates_unclassified": 0, "n_dates_missing_images": 0}
    flags, document_reconciliations, dimension_reconciliations, predictions_by_date = (
        _real_delivery_flags(tmp_path))
    phenology.write_phenology_csv(
        "test", [row], tmp_path / "out.csv", _revision(tmp_path), flags=flags, acknowledgment=None,
        document_reconciliations=document_reconciliations, producer={},
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project=tmp_path, plant_mapping=_NO_MAPPING)
    written = (tmp_path / "out.csv").read_text(encoding="utf-8")
    header = written.splitlines()[0].split(",")
    assert "n_observed_dates" in header
    data_row = written.splitlines()[1].split(",")
    assert data_row[header.index("n_observed_dates")] == "1"

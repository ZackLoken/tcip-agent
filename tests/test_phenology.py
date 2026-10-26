"""Tests for the canonical phenology measurement module.

The positive-state fraction is the share of a plant's detected objects that are in the trait's
positive/measured state, where that state is a classifier call. These tests pin the authoritative
trait definitions:

    bud_05/50/95per_date  = dates the open fraction crosses 5/50/95%
    bud_majority_date     = date most buds are open (the majority-label alias) = the 95% crossing

and the coverage rule that decides whether a prediction bucket ever assessed the trait's positive
state at all, read off the attribute the state names in the bucket's own scope, and the
per-detection decode within it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import subject_registry as cr
from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tcip_mcp.pipelines.data.selection import ClassScope
from tcip_mcp.pipelines.postprocessing import phenology
from tests._trait_fixtures import BUD_OPENING, entry

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.traits import TraitEntry
    from tcip_store import Key

OPENING = cr.Attribute("opening", "categorical", ("closed", "open"))
"""The attribute :data:`BUD_OPENING`'s positive state names."""
OPENED = BUD_OPENING.positive_state
RULE = BUD_OPENING.crossing_rule


def _registry(*attributes: cr.Attribute) -> cr.SubjectRegistry:
    return cr.SubjectRegistry(
        subjects=(
            cr.Subject(name="bud", description="", attributes=attributes),
        )
    )


def _scope(root: Path, *attributes: cr.Attribute) -> ClassScope:
    """Return the registry scope of ``bud`` over a dataset at ``root`` declaring ``attributes``."""
    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tests._producer_fixtures import registry_over

    registry_over(root, _registry(*attributes))
    return registry_scope(image_root(root), "bud")


def _document(project: Path, annotations: list[Annotation]) -> Key:
    """Write ``annotations`` as an image's prediction document under ``project``; return its key."""
    from tcip_mcp.dataset_layout import prediction_key

    key = prediction_key(project, "m/2024-05-01", "img")
    json_io.write_label_document(key, annotations, 8, 8)
    return key


def _bucket(
    project: Path,
    date: str,
    documents: dict[str, list[str]],
    *,
    attributed: bool = True,
) -> Bucket:
    """Publish a bucket for ``date`` holding one document per stem of ``documents``; return it.

    Each detection carries its value under :data:`OPENING`, from a checkpoint whose scope declares
    it, or none from one whose scope declares no attribute (``_chain_fixtures.published``). With
    no stem named it holds one image named ``unmapped``.
    """
    pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import image_dir
    from tests._chain_fixtures import predicted, published

    attributes = (OPENING,) if attributed else ()
    images = image_dir(project / "ds", date)
    name = f"m-{'attributed' if attributed else 'bare'}/{date}"
    return published(
        project,
        name,
        [
            predicted(images / f"{stem}.jpg", values, attributes)
            for stem, values in (documents or {"unmapped": []}).items()
        ],
        scope={"subject": "bud"},
        registry=_registry(*attributes),
    )


# ── date helpers ──────────────────────────────────────────────


def test_date_key_orders_chronologically() -> None:
    assert phenology.date_key("2024-05-01") < phenology.date_key("2024-05-15")
    assert phenology.date_key("2024-05-15") < phenology.date_key("2024-06-01")


def test_date_key_malformed_sorts_first() -> None:
    assert phenology.date_key(UNDATED_BUCKET) == (0, 0, 0)
    assert phenology.date_key("2024-13") == (0, 0, 0)
    assert phenology.date_key("not-a-date-x") == (0, 0, 0)


def test_date_key_rejects_out_of_range_dates() -> None:
    assert phenology.date_key("2026-13-01") == (0, 0, 0)
    assert phenology.date_key("2026-02-30") == (0, 0, 0)


def test_crossing_date_does_not_crash_on_malformed_date() -> None:
    series = [
        ("2026-02-01", 0.0),
        ("2026-13-01", 0.9),  # out-of-range month, excluded
        ("2026-02-15", 1.0),
    ]
    c = phenology.crossing_date(series, 0.50, RULE)
    assert c.date == "2026-02-08"
    assert c.bound == "interpolated"
    assert phenology.positive_onset_date(series) == "2026-02-15"


def test_real_points_drops_undated_and_sorts() -> None:
    series = [("2024-05-15", 0.5), (UNDATED_BUCKET, 0.9), ("2024-05-01", 0.1)]
    pts = phenology._real_points(series)
    assert [d for d, _ in pts] == ["2024-05-01", "2024-05-15"]


# ── crossings (censoring bound disclosed) ─────────────────────────────


def test_crossing_interpolates_between_dates() -> None:
    series = [("2024-05-01", 0.0), ("2024-05-11", 1.0)]
    c = phenology.crossing_date(series, 0.50, RULE)
    assert c.date == "2024-05-06"
    assert c.bound == "interpolated"
    assert c.gap_days == 10


def test_crossing_first_point_already_at_target_is_left_censored() -> None:
    # A left-censored crossing must be flagged, not returned as a bare tuple indistinguishable
    # from a real single-date measurement.
    series = [("2024-05-01", 0.2), ("2024-05-05", 0.8)]
    c = phenology.crossing_date(series, 0.05, RULE)
    assert c.date == "2024-05-01"
    assert c.bound == "left_censored"


def test_crossing_exact_match_is_not_censored() -> None:
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.50)]
    c = phenology.crossing_date(series, 0.50, RULE)
    assert c.date == "2024-05-05"
    assert c.bound == "exact"


def test_crossing_never_reached_is_right_censored() -> None:
    # The last observed point never meets the target: right-censored there, unlike no
    # observations at all, which stays None.
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.3)]
    c = phenology.crossing_date(series, 0.95, RULE)
    assert c.date == "2024-05-05"
    assert c.bound == "right_censored"
    assert c.gap_days is None


def test_crossing_no_observations_is_none() -> None:
    assert phenology.crossing_date([], 0.95, RULE) is None


def test_opening_onset_is_first_nonzero_date() -> None:
    series = [
        ("2024-05-01", 0.0),
        ("2024-05-05", 0.0),
        ("2024-05-09", 0.10),
        ("2024-05-13", 0.60),
    ]
    assert phenology.positive_onset_date(series) == "2024-05-09"


def test_opening_onset_none_when_all_zero() -> None:
    series = [("2024-05-01", 0.0), ("2024-05-05", 0.0)]
    assert phenology.positive_onset_date(series) is None


def test_plant_milestones_returns_four_dates_and_bounds() -> None:
    series = [
        ("2024-05-01", 0.0),
        ("2024-05-06", 0.04),
        ("2024-05-11", 0.50),
        ("2024-05-21", 1.0),
    ]
    m = phenology.plant_milestones(series, BUD_OPENING)
    assert {
        "bud_05per_date",
        "bud_50per_date",
        "bud_95per_date",
        "bud_majority_date",
    } <= set(m)
    assert m["bud_majority_date"] == m["bud_95per_date"]
    assert m["bud_50per_date"] == "2024-05-11"
    assert m["bud_50per_date_bound"] == "exact"


def test_the_milestones_and_their_schema_restate_no_crossing_rule() -> None:
    """Neither a plant's milestones nor the CSV schema restates the crossing rule.

    The rule a crossing is read under is the trait revision's.
    """
    ms = phenology.plant_milestones(
        [("2024-05-01", 0.0), ("2024-05-09", 1.0)], BUD_OPENING
    )
    assert "crossing_rule" not in ms
    assert "crossing_rule" not in phenology.phenology_csv_columns(BUD_OPENING)


def test_plant_milestones_requires_spec_no_bud_opening_fallback() -> None:
    with pytest.raises(TypeError, match="'spec'"):
        phenology.plant_milestones([("2024-05-01", 0.5)])  # type: ignore[call-arg]  # the omission is the subject; the raises pins it to spec


def test_milestone_date_columns_is_proper_subset_of_full_columns() -> None:
    cols = phenology.phenology_csv_columns(BUD_OPENING)
    milestone_cols = {
        c[k]
        for c in phenology.milestone_date_columns(BUD_OPENING)
        for k in ("date", "bound")
    }
    assert milestone_cols <= set(cols)
    assert "plant_id" not in milestone_cols
    assert "validated" not in milestone_cols


def test_every_milestone_bound_and_the_observed_date_count_are_delivered_columns() -> (
    None
):
    """Every milestone bound and the observed-date count are delivered columns.

    ``plant_milestones`` emits a bound beside each milestone date, and the CSV schema names each
    date, each bound and ``n_observed_dates``.
    """
    cols = phenology.phenology_csv_columns(BUD_OPENING)
    ms = phenology.plant_milestones(
        [("2024-05-01", 0.0), ("2024-05-09", 1.0)], BUD_OPENING
    )
    for col in phenology.milestone_date_columns(BUD_OPENING):
        assert col["date"] in cols and col["bound"] in cols, col
        assert col["date"] in ms and col["bound"] in ms, col
    assert "n_observed_dates" in cols


# ── count_by_class: the coverage mechanism ─────────────────────


def test_count_by_class_bare_detector_bucket_refuses_never_full_coverage(
    tmp_path: Path,
) -> None:
    # A detector whose scope declares no attribute at all has no call of the positive state, so
    # none of its detections counts as assessed for it.
    p = _document(
        tmp_path,
        [
            Annotation(subject="bud", geometry=BBox(1, 1, 3, 3), score=0.9),
            Annotation(subject="bud", geometry=BBox(4, 4, 6, 6), score=0.8),
        ],
    )
    total, positive, unclassified = phenology.count_by_class(
        p, OPENED, scope=_scope(tmp_path)
    )
    # whole bucket unclassified, not full coverage
    assert (total, positive, unclassified) == (2, 0, 2)


def test_count_by_class_wrong_axis_bucket_refuses(tmp_path: Path) -> None:
    # A run whose scope declares a different attribute of the same subject (damage severity); it
    # never called the positive state's attribute, so it must refuse, not be miscounted.
    p = _document(
        tmp_path,
        [
            Annotation(
                subject="bud",
                geometry=BBox(1, 1, 3, 3),
                score=0.9,
                attributes={"damage": "mild"},
            )
        ],
    )
    scope = _scope(
        tmp_path, cr.Attribute("damage", "ordinal", ("none", "mild", "severe"))
    )
    total, positive, unclassified = phenology.count_by_class(
        p, OPENED, scope=scope
    )
    assert (total, positive, unclassified) == (1, 0, 1)


def test_count_by_class_attributed_bucket_splits_positive_negative(
    tmp_path: Path,
) -> None:
    p = _document(
        tmp_path,
        [
            Annotation(
                subject="bud",
                geometry=BBox(1, 1, 3, 3),
                score=0.9,
                attributes={"opening": "open"},
            ),
            Annotation(
                subject="bud",
                geometry=BBox(4, 4, 6, 6),
                score=0.8,
                attributes={"opening": "closed"},
            ),
            Annotation(
                subject="bud",
                geometry=BBox(1, 1, 3, 3),
                score=0.7,
                attributes={"opening": "open"},
            ),
        ],
    )
    total, positive, unclassified = phenology.count_by_class(
        p, OPENED, scope=_scope(tmp_path, OPENING)
    )
    assert (total, positive, unclassified) == (3, 2, 0)


def test_count_by_class_foreign_record_within_attributed_bucket_refuses(
    tmp_path: Path,
) -> None:
    # A record carrying no value under the state's attribute (a stale bare-detector document)
    # refuses by name rather than reading as a negative.
    p = _document(
        tmp_path,
        [
            Annotation(
                subject="bud",
                geometry=BBox(1, 1, 3, 3),
                score=0.9,
                attributes={"opening": "open"},
            ),
            Annotation(subject="bud", geometry=BBox(4, 4, 6, 6), score=0.8),
        ],
    )
    with pytest.raises(
        json_io.UndeclaredValueError, match="with a value under 'opening'"
    ):
        phenology.count_by_class(p, OPENED, scope=_scope(tmp_path, OPENING))


# ── per_plant_series / per_plant_phenology: bucket-level + expected-coverage ────────────────────


class _Assignment:
    def __init__(
        self, stem: str, plant_id: str | None, accession_name: str | None
    ) -> None:
        self.stem = stem
        self.plant_id = plant_id
        self.accession_name = accession_name


def test_per_plant_phenology_builds_fraction_series_over_attributed_buckets(
    tmp_path: Path,
) -> None:
    buckets = {
        "2024-05-01": _bucket(
            tmp_path, "2024-05-01", {"P1_a": ["closed", "closed"]}
        ),
        "2024-05-15": _bucket(
            tmp_path, "2024-05-15", {"P1_b": ["open", "open"]}
        ),
    }
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }

    out = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["P1"],
    )

    assert out["positive_class_assessed"] is True
    row = out["rows"][0]
    assert row["plant_id"] == "P1"
    assert row["accession"] == "acc-9"
    assert row["n_dates"] == 2
    assert row["n_dates_unclassified"] == 0
    assert row["n_dates_missing_images"] == 0
    assert [s["n_positive"] for s in row["series"]] == [0, 2]
    assert row["bud_95per_date"] is not None


def test_per_plant_phenology_bare_detector_bucket_refuses_whole_delivery(
    tmp_path: Path,
) -> None:
    buckets = {
        "2024-05-01": _bucket(
            tmp_path, "2024-05-01", {"P1_a": ["bud"]}, attributed=False
        )
    }
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}

    out = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["P1"],
    )

    assert out["positive_class_assessed"] is False
    row = out["rows"][0]
    assert row["n_dates_unclassified"] == 1
    assert row["bud_50per_date"] is None


def test_per_plant_phenology_missing_image_is_disclosed_not_a_zero(
    tmp_path: Path,
) -> None:
    # A stem the mapping names with no prediction document must not read as an observed zero
    # (which would count as an assessed 0/0 and silently pass coverage).
    buckets = {"2024-05-01": _bucket(tmp_path, "2024-05-01", {})}
    mapping = {"2024-05-01": [_Assignment("P1_a", "P1", "acc-9")]}

    out = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["P1"],
    )

    row = out["rows"][0]
    assert row["series"][0]["n_missing"] == 1
    assert row["series"][0]["ratio"] is None
    assert row["n_dates_missing_images"] == 1
    assert (
        out["positive_class_assessed"] is False
    )  # no date anywhere was complete
    assert row["bud_50per_date"] is None


def test_per_plant_phenology_multi_date_and_excludes_plant_with_one_bad_date(
    tmp_path: Path,
) -> None:
    # One date whose bucket declares no attribute excludes the whole plant's milestones,
    # disclosed, rather than computing them from the dates that happened to be usable.
    buckets = {
        "2024-05-01": _bucket(tmp_path, "2024-05-01", {"P1_a": ["open"]}),
        "2024-05-15": _bucket(
            tmp_path, "2024-05-15", {"P1_b": ["bud"]}, attributed=False
        ),
    }
    mapping = {
        "2024-05-01": [_Assignment("P1_a", "P1", "acc-9")],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }

    out = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["P1"],
    )

    row = out["rows"][0]
    assert row["n_dates"] == 2
    assert row["n_dates_unclassified"] == 1
    assert row["bud_05per_date"] is None
    assert row["bud_95per_date"] is None
    # At least one date elsewhere was complete, so the delivery-level flag is still True,
    # distinguishing "wired, some gaps" from "never wired at all".
    assert out["positive_class_assessed"] is True


def test_a_plant_the_mapping_captured_on_no_image_of_a_date_is_incomplete_on_it(
    tmp_path: Path,
) -> None:
    """A plant the mapping captured on no image of a date is incomplete on that date.

    A plant captured on the first date only carries an imageless second date, counted against its
    completeness, and its milestones are withheld; a plant captured on both dates is complete.
    """
    buckets = {
        "2024-05-01": _bucket(
            tmp_path, "2024-05-01", {"P1_a": ["open"], "P2_a": ["open"]}
        ),
        "2024-05-15": _bucket(tmp_path, "2024-05-15", {"P1_b": ["open"]}),
    }
    mapping = {
        "2024-05-01": [
            _Assignment("P1_a", "P1", "acc-9"),
            _Assignment("P2_a", "P2", "acc-7"),
        ],
        "2024-05-15": [_Assignment("P1_b", "P1", "acc-9")],
    }

    out = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["P1", "P2"],
    )

    p1, p2 = out["rows"]
    assert (p1["complete"], p1["n_dates_missing_images"]) == (True, 0)
    assert [(s["date"], s["n_images"]) for s in p2["series"]] == [
        ("2024-05-01", 1),
        ("2024-05-15", 0),
    ]
    assert (p2["complete"], p2["n_dates_missing_images"], p2["n_dates"]) == (
        False,
        1,
        2,
    )
    assert p2["bud_05per_date"] is None


# ── the positive state's attribute ────────────────────────────────


def test_the_positive_states_ids_read_the_bucket_records_scope(
    tmp_path: Path,
) -> None:
    bucket = _bucket(tmp_path, "2024-05-01", {})
    assert bucket.scope.state_ids(OPENED) == (
        0,
        OPENING.values.index(OPENED.value),
    )


def test_the_positive_states_ids_are_none_for_a_bucket_declaring_no_attribute(
    tmp_path: Path,
) -> None:
    bucket = _bucket(tmp_path, "2024-05-01", {}, attributed=False)
    assert bucket.scope.state_ids(OPENED) is None


_SPEC_SHAPES = [
    # An entry stating both majority fields empty, which once made the phantom names doubly
    # malformed (`b__date`).
    entry(
        "b",
        BUD_OPENING.delivers,
        milestone_fractions=(0.1, 0.9),
        crossing_rule=RULE,
        phenology_prefix="b",
    ),
    # A majority label with no majority milestone: a column must not be built from the label
    # alone without a milestone to source it.
    entry(
        "b",
        BUD_OPENING.delivers,
        milestone_fractions=(0.1, 0.9),
        crossing_rule=RULE,
        phenology_prefix="b",
        majority_label="peak",
    ),
    # A majority milestone naming a crossing the trait does not compute: the column is real (the
    # spec names it) and its value is honestly None, which is not the same as a phantom.
    entry(
        "b",
        BUD_OPENING.delivers,
        milestone_fractions=(0.1, 0.9),
        crossing_rule=RULE,
        phenology_prefix="b",
        majority_milestone="95per",
        majority_label="peak",
    ),
    BUD_OPENING,
]


@pytest.mark.parametrize(
    "spec",
    _SPEC_SHAPES,
    ids=lambda s: f"{s.name}-{s.majority_milestone or 'nomajority'}",
)
def test_phenology_csv_columns_name_no_column_without_a_producer(
    spec: TraitEntry,
) -> None:
    """Every trait-prefixed column the CSV schema names is filled by a producer.

    The parametrized spec shapes include one whose ``majority_milestone`` is unset, and no
    computed column is missing from the schema.
    """
    series = [("2026-02-01", 0.0), ("2026-02-10", 0.5), ("2026-02-20", 1.0)]
    produced = set(phenology.plant_milestones(series, spec))
    schema = set(phenology.phenology_csv_columns(spec))
    prefixed = {c for c in schema if c.startswith(spec.phenology_prefix + "_")}
    assert prefixed - produced == set()
    assert (
        produced - schema == set()
    )  # and nothing computed is silently dropped
    assert not any(c.startswith(f"{spec.phenology_prefix}__") for c in schema)


def test_excluded_plant_carries_the_same_milestone_keys_as_an_included_one(
    tmp_path: Path,
) -> None:
    """An excluded plant carries the same milestone keys as an included one.

    The shape includes each milestone date's ``*_date_bound`` key, not just
    ``milestone_date_columns``'s bare dates.
    """
    buckets = {
        "2026-02-11": _bucket(
            tmp_path,
            "2026-02-11",
            {"GOOD": ["open", "closed"], "BAD": ["open", "open"]},
        ),
        # BAD's second date is never predicted on: the missing image excludes its milestones.
        "2026-03-09": _bucket(
            tmp_path, "2026-03-09", {"GOOD": ["open", "open"]}
        ),
    }
    mapping = {
        "2026-02-11": [
            _Assignment("GOOD", "GOOD", "a"),
            _Assignment("BAD", "BAD", "b"),
        ],
        "2026-03-09": [
            _Assignment("GOOD", "GOOD", "a"),
            _Assignment("BAD", "BAD", "b"),
        ],
    }
    res = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["GOOD", "BAD"],
    )
    by_plant = {r["plant_id"]: r for r in res["rows"]}
    assert by_plant["BAD"]["n_dates_missing_images"] == 1  # genuinely excluded
    assert set(by_plant["GOOD"]) == set(by_plant["BAD"])
    assert "bud_95per_date_bound" in by_plant["BAD"]


def test_per_plant_series_counts_the_images_the_mapping_names(
    tmp_path: Path,
) -> None:
    """``n_images`` counts every image the mapping names for a (plant, date).

    Four images are named for one plant on one date, one with no prediction document: the count
    is 4, beside six detections, three positive and one missing image.
    """
    buckets = {
        "2026-02-11": _bucket(
            tmp_path,
            "2026-02-11",
            {f"IMG{i}": ["open", "closed"] for i in range(3)},
        )
    }
    mapping = {
        "2026-02-11": [_Assignment(f"IMG{i}", "P1", "a") for i in range(3)]
        + [_Assignment("GONE", "P1", "a")]
    }  # named, no prediction document
    per_plant = phenology.per_plant_series(mapping, buckets, OPENED, ["P1"])
    (_date, total, positive, unclassified, missing, n_images) = per_plant[
        "P1"
    ]["series"][0]
    assert (total, positive, unclassified, missing) == (6, 3, 0, 1)
    # 4 images named for this (plant, date), of which one is missing, not the 3 documents that
    # happened to exist.
    assert n_images == 4


def test_per_plant_series_excludes_unattributed_assignments_from_coverage(
    tmp_path: Path,
) -> None:
    """An assignment with no ``plant_id`` adds no plant to the per-plant series.

    Such an assignment is an image the plant-mapping step could not assign.
    """
    buckets = {
        "2026-02-11": _bucket(tmp_path, "2026-02-11", {"P1_a": ["open"]})
    }
    mapping = {
        "2026-02-11": [
            _Assignment("P1_a", "P1", "acc-9"),
            _Assignment(
                "STRAY", None, None
            ),  # no plant_id: never assigned to any plant
        ]
    }
    per_plant = phenology.per_plant_series(mapping, buckets, OPENED, ["P1"])
    assert list(per_plant) == ["P1"]


def test_per_plant_phenology_excludes_unattributed_assignments_from_rows(
    tmp_path: Path,
) -> None:
    """``per_plant_phenology`` emits no row and no unmapped count for an unattributed assignment."""
    buckets = {
        "2026-02-11": _bucket(tmp_path, "2026-02-11", {"P1_a": ["open"]})
    }
    mapping = {
        "2026-02-11": [
            _Assignment("P1_a", "P1", "acc-9"),
            _Assignment("STRAY1", None, None),
            _Assignment("STRAY2", "", None),
        ]
    }
    out = phenology.per_plant_phenology(
        mapping,
        buckets,
        BUD_OPENING,
        ["P1"],
    )
    assert [r["plant_id"] for r in out["rows"]] == ["P1"]
    assert "n_images_unmapped" not in out

"""Canonical phenology measurement: a trait's positive-fraction milestones, for whichever
registered trait it's computed for.

The positive-state fraction is the fraction of a plant's detected objects a classifier calls the
trait's ``positive_value``. Milestone columns come entirely from the trait's own ``TraitEntry``
(``phenology_prefix`` plus each ``milestone_fractions`` entry):

    ``<prefix>_<NN>per_date``            = the date the positive fraction first crosses NN%,
                                            for each fraction the spec declares
    ``<prefix>_<majority_label>_date``   = the majority-crossing alias
                                            (``TraitEntry.majority_milestone``), present only
                                            when the spec names one

``positive_onset_date`` (the first date any positive-state observation appears) is a separate
helper, not the delivered trait.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

from tcip_mcp.dataset_layout import label_filename

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import Result
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.traits import TraitRevision


def _milestone_targets(spec) -> dict[str, float]:
    """Milestone crossing fractions from the trait's confirmed semantics, resolved per call. Keyed
    "NNper" to match the CSV column names.
    """
    return {f"{int(round(f * 100)):02d}per": f for f in spec.milestone_fractions}


def _milestone_columns(spec) -> list[tuple[str, str]]:
    """``(column_suffix, crossing_key)`` for every milestone this trait actually delivers; the
    majority alias enters only when the spec names a crossing for it.
    """
    cols = [(key, key) for key in _milestone_targets(spec)]
    if spec.majority_milestone:
        cols.insert(0, (spec.majority_label, spec.majority_milestone))
    return cols


REQUIRE_ALL_DATES_COMPLETE = True
"""The missingness rule a phenology measurement takes when its caller states none: a plant's
milestones are computed only when every one of its dates is complete. Every door defaults to this,
and the delivery event records the rule a delivery ran under."""


def milestone_date_columns(spec) -> list[dict[str, str]]:
    """The milestone columns a trait's phenology delivery carries, each as ``{"date", "bound"}``:
    the milestone's date column and the column holding that date's evidentiary bound."""
    return [{"date": f"{spec.phenology_prefix}_{sfx}_date",
             "bound": f"{spec.phenology_prefix}_{sfx}_date_bound"}
            for sfx, _ in _milestone_columns(spec)]


def phenology_csv_columns(spec) -> list[str]:
    """The delivered per-plant phenology CSV schema for one trait, derived from its ``TraitEntry``:
    the plant, its coverage, each milestone date and its evidentiary bound, then the delivery
    columns every delivered CSV carries."""
    from tcip_mcp.delivery import DELIVERY_COLUMNS

    columns = milestone_date_columns(spec)
    return [
        "plant_id", "accession", "n_dates", "n_observed_dates", "n_dates_unclassified",
        "n_dates_missing_images", "complete",
        *[c["date"] for c in columns], *[c["bound"] for c in columns],
        *DELIVERY_COLUMNS,
    ]


CURVE_MEASUREMENT_COLUMNS = [
    "plant_id", "accession", "date",
    "n_images", "n_total", "n_positive", "n_unclassified", "n_missing", "ratio",
]
"""The per-(plant, date) columns ``per_plant_series`` produces."""


def curve_csv_columns() -> list[str]:
    """The delivered per-(plant, date) curve CSV schema: the counts the fraction is built from,
    then the delivery columns every delivered CSV carries."""
    from tcip_mcp.delivery import DELIVERY_COLUMNS

    return [*CURVE_MEASUREMENT_COLUMNS, *DELIVERY_COLUMNS]


# ── ISO date helpers ─────────────────────────────────────────────────────


def date_key(date_str: str) -> tuple[int, int, int]:
    """ISO ``YYYY-MM-DD`` -> ``(year, month, day)`` for chronological sort.

    A value that is not a calendar-legal ISO date (the ``undated/`` bucket, a non-numeric folder,
    or an out-of-range one like ``2026-13-01``) sorts first as ``(0, 0, 0)`` and is excluded from
    milestone math.
    """
    parts = date_str.split("-")
    if len(parts) != 3:
        return (0, 0, 0)
    try:
        y, m, d = (int(x) for x in parts)
        date(y, m, d)  # reject out-of-range month/day (e.g. 2026-13-01)
    except ValueError:
        return (0, 0, 0)
    return (y, m, d)


def iso(date_str: str) -> str:
    y, m, d = date_key(date_str)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _real_points(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Series sorted chronologically, with non-ISO/undated points dropped."""
    pts = [(d, r) for d, r in series if date_key(d) != (0, 0, 0)]
    pts.sort(key=lambda p: date_key(p[0]))
    return pts


# ── milestones ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Crossing:
    """One milestone crossing: the date, its evidentiary bound, and the observation gap it spans.

    ``bound``:
      - ``"exact"``: the target fraction was actually observed on this date.
      - ``"left_censored"``: the first observed point already meets the target, the true crossing
        may have happened any time before it; this date is an upper bound, not a measured crossing.
      - ``"interpolated"``: linearly interpolated between the two bracketing observed dates.
      - ``"right_censored"``: the last observed point still hasn't met the target, the true crossing,
        if it happens at all, is after this date; this date is a lower bound, the mirror of
        ``left_censored`` at the other end of the observed window, distinguishing "not yet reached,
        but we watched through this date" from "no information at all".
    ``gap_days``: for ``interpolated``, the number of days between the two bracketing observations
    (a wide gap is weaker evidence for the same interpolated date), ``0`` for ``exact``/unknown for
    ``left_censored``/``right_censored`` (no bracket on the censored side exists).
    """

    date: str
    bound: str
    gap_days: int | None = None


def crossing_date(series: list[tuple[str, float]], target: float) -> Optional[Crossing]:
    """Earliest date the fraction curve reaches ``>= target``, with its evidentiary bound.

    Linear interpolation between neighboring observed dates when the crossing falls between two
    points; a left-censored crossing (the first observed point already meets the target) or a
    right-censored one (the last observed point still hasn't) is flagged as such. ``None`` only
    when there are no real observed points.
    """
    points = _real_points(series)
    if not points:
        return None
    if points[0][1] >= target:
        return Crossing(iso(points[0][0]), "left_censored")
    for (d1, r1), (d2, r2) in zip(points, points[1:]):
        if r2 >= target:
            if r2 == target:
                return Crossing(iso(d2), "exact")
            y1, m1, day1 = date_key(d1)
            y2, m2, day2 = date_key(d2)
            gap = (date(y2, m2, day2) - date(y1, m1, day1)).days
            t = max(0.0, min(1.0, (target - r1) / (r2 - r1))) if r2 != r1 else 1.0
            est = date(y1, m1, day1) + timedelta(days=round(t * gap))
            return Crossing(est.isoformat(), "interpolated", gap_days=gap)
    return Crossing(iso(points[-1][0]), "right_censored")


def positive_onset_date(series: list[tuple[str, float]]) -> Optional[str]:
    """First date any positive-state observation appears (fraction > 0), chronologically. ``None`` if never."""
    for d, r in _real_points(series):
        if r > 0:
            return iso(d)
    return None


def plant_milestones(series: list[tuple[str, float]], spec) -> dict:
    """The phenology dates for one plant's positive-fraction series, keyed by the trait's own
    columns.

    The column names (``phenology_prefix`` + each milestone key, plus the majority alias), the
    crossing fractions and the majority mapping come from ``spec``, which is required.
    """
    crossings = {key: crossing_date(series, frac) for key, frac in _milestone_targets(spec).items()}
    out: dict = {}
    # The majority alias is another entry in ``_milestone_columns``, so it carries its crossing's
    # date and bound under the same condition as every other milestone.
    for (_sfx, key), column in zip(_milestone_columns(spec), milestone_date_columns(spec)):
        crossing = crossings.get(key)
        out[column["date"]] = crossing.date if crossing else None
        out[column["bound"]] = crossing.bound if crossing else None
    return out


# ── positive-fraction from classified predictions ────────────────────────


def count_by_class(
    json_path: Path, positive_value: str, *, scope: ClassScope,
) -> tuple[int, int, int]:
    """``(n_total, n_positive, n_unclassified)`` for one image's predictions.

    ``scope`` is the bucket's own recorded scope. When it classifies no ``positive_value``
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.positive_id`), every detection is
    unclassified: a whole-bucket decision, so a detector bucket whose one map key happens to equal
    ``positive_value`` never counts a positive.

    Under a classified scope, every record is held to
    :func:`~tcip_annotation.json_io.require_classified_record` under the bucket's own recorded
    vocabulary (``id_map``'s keys): a value outside that vocabulary refuses by name. A record whose
    value equals ``positive_value`` counts positive; every other classified record counts toward
    neither positive nor unclassified.

    Called only for a prediction file confirmed to exist.
    """
    from tcip_annotation import json_io

    annotations = json_io.detection_annotations(json_path)
    total = len(annotations)
    if scope.positive_id(positive_value) is None:
        return total, 0, total
    positive = 0
    for i, a in enumerate(annotations):
        value = json_io.require_classified_record(
            a, subject=cast(str, scope.subject), attribute=cast(str, scope.attribute),
            vocabulary=set(cast(dict, scope.id_map)), source=f"{json_path}#{i}")
        if value == positive_value:
            positive += 1
    return total, positive, 0


class EmptyPopulation(ValueError):
    """A phenology measurement was asked for with no plants named; the population is the caller's
    explicit plant list.
    """


def measurement_refusals() -> tuple[type[Exception], ...]:
    """The exceptions :func:`per_plant_phenology` refuses a measurement with, each naming why."""
    from tcip_annotation.json_io import ClassifiedRecordRefused, UnreadableLabelDocument
    from tcip_store import StoreError

    return (UnreadableLabelDocument, ClassifiedRecordRefused, StoreError, EmptyPopulation)


def population(plants: Sequence[str]) -> list[str]:
    """The delivery population as one ordered list of distinct plant ids, refusing an empty one."""
    ordered = list(dict.fromkeys(str(p) for p in plants))
    if not ordered:
        raise EmptyPopulation(
            "a delivery needs the plants it is for: pass the plant ids to deliver "
            "(plants=[...]); a mapping or registry names every plot, never the population."
        )
    return ordered


def per_plant_series(
    mapping: dict[str, list], buckets: dict[str, Bucket], positive_value: str,
    plants: list[str],
) -> dict[str, dict]:
    """Aggregate classified predictions into a per-plant positive-fraction series, for exactly the
    plants in ``plants`` (:func:`population`'s list).

    ``mapping`` is ``{date: [assignment, ...]}`` where each assignment has ``.stem`` /
    ``.plot_name`` / ``.accession_name`` (attributes or dict keys); ``buckets`` is each delivered
    date's published bucket, by its recorded date. ``plants`` is the delivery's population: every
    plant in it gets an entry, in the order given, and a plant the mapping names that is not in it
    is never read. Every population plant has a series point on every date the mapping names, one
    the mapping assigns it no capture on carrying no image. Returns ``{plant_id: {accession,
    series: [(date, total, positive, unclassified, missing, n_images), ...]}}``. An entry naming no
    plant (``plant_mapping.assignment_is_attributed`` false) is excluded from every plant's
    coverage.

    Coverage is measured against the stems the plant mapping names for each (plant, date): a named
    stem with no corresponding prediction document is a missing observation, and a date the
    mapping names with no delivered bucket counts every stem it names as missing.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        assignment_is_attributed, attr, stems_delivery_reads,
    )

    per_plant: dict[str, dict] = {
        plant_id: {"accession": None, "series": []} for plant_id in plants
    }
    # The mapping's own dates are the coverage reference, so a date it names is never absent.
    for date_str in mapping:
        bucket = buckets.get(date_str)
        read_stems = stems_delivery_reads(mapping[date_str], bucket) if bucket else set()
        by_plant: dict[str, list[int]] = {plant_id: [0, 0, 0, 0, 0] for plant_id in plants}
        for a in mapping[date_str]:
            plant_id = attr(a, "plot_name")
            if not assignment_is_attributed(a) or plant_id not in per_plant:
                continue
            acc = by_plant[plant_id]
            acc[4] += 1
            if per_plant[plant_id]["accession"] is None:
                per_plant[plant_id]["accession"] = attr(a, "accession_name")
            stem = attr(a, "stem")
            if bucket is None or stem not in read_stems:
                acc[3] += 1
                continue
            total, positive, unclassified = count_by_class(
                bucket.path / label_filename(str(stem)), positive_value, scope=bucket.scope)
            acc[0] += total
            acc[1] += positive
            acc[2] += unclassified
        for plant_id, counts in by_plant.items():
            per_plant[plant_id]["series"].append((date_str, *counts))
    return per_plant


def per_plant_phenology(
    mapping: dict[str, list], buckets: dict[str, Bucket], spec, plants: list[str], *,
    require_all_dates_complete: bool,
) -> dict:
    """Full canonical pipeline: classified predictions + plant mapping -> per-plant milestones
    against ``spec``'s ``positive_value`` and milestones, one row per plant in ``plants`` and no
    other, in its order (see :func:`per_plant_series`).

    Returns ``{rows: [...], positive_class_assessed: bool}``. Each row carries the
    positive-fraction series, the milestone dates, coverage-disclosure fields
    (``n_dates_unclassified``, ``n_dates_missing_images``, a date the mapping captured the plant on
    no image counting as missing) and ``complete``, whether every one of its dates is complete:
    imaged, fully classified and fully observed. With ``require_all_dates_complete`` a plant's
    milestones are computed only when it is complete, and otherwise from its complete dates alone.
    ``positive_class_assessed`` is ``True`` iff at least one date, anywhere in the delivery, was
    complete.
    """
    per_plant = per_plant_series(mapping, buckets, spec.positive_value, plants)
    rows = []
    any_classified_date = False
    for plant_id, info in per_plant.items():
        usable_dates = [(d, total, positive)
                        for (d, total, positive, unclassified, missing, n_images) in info["series"]
                        if unclassified == 0 and missing == 0 and n_images > 0]
        if usable_dates:
            any_classified_date = True
        # total==0 detected no objects, so it is not an observation of the fraction; kept only when
        # something was detected, a real 0% included.
        frac_series = [(d, positive / total) for (d, total, positive) in usable_dates if total]
        complete = len(usable_dates) == len(info["series"]) and len(info["series"]) > 0
        row = {
            "plant_id": plant_id,
            "accession": info["accession"],
            "n_dates": len(info["series"]),
            "n_observed_dates": len(frac_series),
            "n_dates_unclassified": sum(1 for s in info["series"] if s[3] > 0),
            "n_dates_missing_images": sum(1 for s in info["series"] if s[4] > 0 or s[5] == 0),
            "complete": complete,
            "series": [
                {"date": d, "n_total": total, "n_positive": positive, "n_unclassified": unclassified,
                 "n_missing": missing, "n_images": n_images,
                 "ratio": (positive / total if total and unclassified == 0 and missing == 0 else None)}
                for (d, total, positive, unclassified, missing, n_images) in info["series"]
            ],
        }
        # An excluded plant goes through the producer with an empty series, so every row has one
        # key set.
        row.update(plant_milestones(
            frac_series if complete or not require_all_dates_complete else [], spec))
        rows.append(row)
    return {"rows": rows, "positive_class_assessed": any_classified_date}


@dataclass(frozen=True)
class PhenologyMeasurement:
    """One trait's per-plant phenology over a plant mapping and its delivered buckets: the
    confirmed revision it is measured under, the buckets by date, the rows, the missingness rule
    they were computed under, and the mapping's delivery disclosure."""

    revision: TraitRevision
    buckets: dict[str, Bucket]
    rows: list[dict]
    positive_class_assessed: bool
    require_all_dates_complete: bool
    plant_mapping: dict
    plants: list[str]

    def curve_rows(self) -> list[dict]:
        """Per-(plant, date) rows: the milestone rows' own series, not a second aggregation."""
        return [{"plant_id": row["plant_id"], "accession": row["accession"], **point}
                for row in self.rows for point in row["series"]]

    def milestone_rows(self) -> list[dict]:
        return [{k: v for k, v in row.items() if k != "series"} for row in self.rows]

    def result(self, *, curves: bool) -> Result:
        """The delivered result: the milestone rows, or the curve rows with ``curves``, under
        their columns, with the population, missingness rule and mapping disclosure."""
        from tcip_mcp.delivery import Result

        return Result(curve_csv_columns() if curves else phenology_csv_columns(self.revision.entry),
                      self.curve_rows() if curves else self.milestone_rows(),
                      population=self.plants,
                      require_all_dates_complete=self.require_all_dates_complete,
                      plant_mapping=self.plant_mapping)


def measure_phenology(
    project: Path, *, trait: str, mapping_name: str, buckets: Sequence[str | Path],
    plants: Sequence[str], require_all_dates_complete: bool,
) -> PhenologyMeasurement:
    """Measure ``trait``'s phenology over the published ``buckets``, each at the capture date its
    record states (:func:`~tcip_mcp.buckets.by_recorded_date`), through the plant mapping
    ``mapping_name``, for exactly ``plants``.

    Reads the trait's latest confirmed revision stating a ``state_crossing_dates``
    operationalization, bound against the delivered dataset's registry (refusing, ``ValueError``,
    buckets whose dataset has none), then the buckets and the mapping
    (``plant_mapping.resolve_delivery_mapping``); each refuses as it does, and so does
    :func:`per_plant_phenology` (:func:`measurement_refusals`).
    """
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.subject_registry import registry_for_pred_dirs
    from tcip_mcp.traits import STATE_CROSSING_DATES

    registry = registry_for_pred_dirs([str(b) for b in buckets])
    if registry is None:
        raise ValueError(
            f"no subject registry is reachable for the dataset behind {list(map(str, buckets))}: "
            "register the dataset or write its subjects.json, so the trait's positive value can "
            "be checked against it.")
    revision = confirmed_revision(STATE_CROSSING_DATES, project=project, trait=trait,
                                  registry=registry)
    wanted = population(plants)
    dated = by_recorded_date(map(read_bucket, buckets))
    mapping_build, verified = plant_mapping.resolve_delivery_mapping(project, mapping_name, dated)
    measured = per_plant_phenology(mapping_build.rows(), dated, revision.entry, wanted,
                                   require_all_dates_complete=require_all_dates_complete)
    positive = revision.entry.positive_value
    return PhenologyMeasurement(
        revision=revision, buckets=dated, rows=measured["rows"],
        positive_class_assessed=(measured["positive_class_assessed"] and any(
            b.scope.positive_id(positive) is not None for b in dated.values())),
        require_all_dates_complete=require_all_dates_complete,
        plant_mapping=mapping_build.delivery_disclosure(verified, list(dated)),
        plants=wanted)


def deliver_phenology(
    project: Path, measurement: PhenologyMeasurement, *, curves: bool, output_path: Path,
    acknowledgment_id: str | None, door: str,
) -> dict[str, Any]:
    """Deliver ``measurement``'s milestone rows, or its per-(plant, date) curve rows with
    ``curves``, as the CSV at ``output_path``.

    The buckets clear the one gate for a ``state_crossing_dates`` delivery, the recorded
    acknowledgment ``acknowledgment_id`` shipping them unvalidated when it does not validate them.
    Writes exactly :func:`phenology_csv_columns` (or :func:`curve_csv_columns`) with the
    delivery's one event under ``door``, its population, missingness rule and plant-mapping
    disclosure (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was delivered.
    """
    from tcip_mcp.delivery import deliver_csv, gate
    from tcip_mcp.traits import STATE_CROSSING_DATES

    result = measurement.result(curves=curves)
    clearance = gate(project, list(measurement.buckets.values()),
                     delivery_kind=STATE_CROSSING_DATES, revision=measurement.revision,
                     result=result, acknowledgment_id=acknowledgment_id)
    delivered = deliver_csv(project, output_path, result, clearance=clearance,
                            revision=measurement.revision, door=door,
                            delivery_kind=STATE_CROSSING_DATES)
    return {**delivered, "n_rows": len(result.rows), "columns": list(result.columns),
            "require_all_dates_complete": measurement.require_all_dates_complete}

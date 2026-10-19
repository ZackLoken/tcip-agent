"""Canonical phenology measurement: a trait's positive-fraction milestones.

The measurement serves whichever registered trait it is computed for. The positive-state fraction
is the fraction of a plant's detected objects a classifier calls the trait's ``positive_state``.
Milestone columns come entirely from the trait's own ``TraitEntry`` (``phenology_prefix`` plus
each ``milestone_fractions`` entry):

    ``<prefix>_<NN>per_date``            = the date the positive fraction first crosses NN%,
                                            for each fraction the spec declares
    ``<prefix>_<majority_label>_date``   = the majority-crossing alias
                                            (``TraitEntry.majority_milestone``), present only
                                            when the spec names one

``positive_onset_date`` (the first date any positive-state observation appears) is a separate
helper, not the delivered trait.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from tcip_mcp.dataset_layout import prediction_key
from tcip_store import Key

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import Result
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.postprocessing.plant_mapping import Assignment
    from tcip_mcp.traits import (
        CrossingRule,
        PositiveState,
        TraitEntry,
        TraitRevision,
    )


def _milestone_targets(spec: TraitEntry) -> dict[str, float]:
    """Return milestone crossing fractions from the trait's confirmed semantics, per call.

    Keyed "NNper" to match the CSV column names.
    """
    return {
        f"{int(round(f * 100)):02d}per": f for f in spec.milestone_fractions
    }


def _milestone_columns(spec: TraitEntry) -> list[tuple[str, str]]:
    """Return ``(column_suffix, crossing_key)`` for every milestone this trait delivers.

    The majority alias enters, first, only when the spec names a crossing for it.
    """
    cols = [(key, key) for key in _milestone_targets(spec)]
    if spec.majority_milestone:
        cols.insert(0, (spec.majority_label, spec.majority_milestone))
    return cols


def milestone_date_columns(spec: TraitEntry) -> list[dict[str, str]]:
    """Return the milestone columns a trait's phenology delivery carries.

    Each is ``{"date", "bound"}``: the milestone's date column and the column holding that date's
    evidentiary bound.
    """
    return [
        {
            "date": f"{spec.phenology_prefix}_{sfx}_date",
            "bound": f"{spec.phenology_prefix}_{sfx}_date_bound",
        }
        for sfx, _ in _milestone_columns(spec)
    ]


def phenology_csv_columns(spec: TraitEntry) -> list[str]:
    """Return the delivered per-plant phenology CSV schema for one trait's ``TraitEntry``.

    The columns are the plant, its coverage, each milestone date and its evidentiary bound, then
    :data:`~tcip_mcp.delivery.DELIVERY_COLUMNS`.
    """
    from tcip_mcp.delivery import DELIVERY_COLUMNS

    columns = milestone_date_columns(spec)
    return [
        "plant_id",
        "accession",
        "n_dates",
        "n_observed_dates",
        "n_dates_unclassified",
        "n_dates_missing_images",
        "complete",
        *[c["date"] for c in columns],
        *[c["bound"] for c in columns],
        *DELIVERY_COLUMNS,
    ]


CURVE_MEASUREMENT_COLUMNS = [
    "plant_id",
    "accession",
    "date",
    "n_images",
    "n_total",
    "n_positive",
    "n_unclassified",
    "n_missing",
    "ratio",
]
"""The per-(plant, date) columns of a curve row (:meth:`PhenologyMeasurement.curve_rows`)."""


def curve_csv_columns() -> list[str]:
    """Return the delivered per-(plant, date) curve CSV schema.

    The columns are the counts the fraction is built from, then
    :data:`~tcip_mcp.delivery.DELIVERY_COLUMNS`.
    """
    from tcip_mcp.delivery import DELIVERY_COLUMNS

    return [*CURVE_MEASUREMENT_COLUMNS, *DELIVERY_COLUMNS]


# ── ISO date helpers ─────────────────────────────────────────────────────


def date_key(date_str: str) -> tuple[int, int, int]:
    """Return ISO ``YYYY-MM-DD`` as ``(year, month, day)`` for chronological sort.

    A value that is not a calendar-legal ISO date (the ``undated/`` bucket, a non-numeric folder,
    or an out-of-range one like ``2026-13-01``) sorts first as ``(0, 0, 0)``.
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
    """Return ``date_str`` as zero-padded ``YYYY-MM-DD``; a non-ISO one gives ``0000-00-00``."""
    y, m, d = date_key(date_str)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _real_points(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sort the series chronologically and drop non-ISO/undated points."""
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
      - ``"interpolated"``: read between the two bracketing observed dates by the crossing rule.
      - ``"right_censored"``: the last observed point still hasn't met the target, the true
        crossing, if it happens at all, is after this date; this date is a lower bound, the mirror
        of ``left_censored`` at the other end of the observed window, distinguishing "not yet
        reached, but we watched through this date" from "no information at all".
    ``gap_days``: for ``interpolated``, the number of days between the two bracketing observations
    (a wide gap is weaker evidence for the same interpolated date), ``0`` for ``exact``/unknown for
    ``left_censored``/``right_censored`` (no bracket on the censored side exists).
    """

    date: str
    bound: str
    gap_days: int | None = None


def _linear_whole_day(
    d1: str, r1: float, d2: str, r2: float, target: float
) -> Crossing:
    """Return the day ``target`` is reached on the straight line from ``(d1, r1)`` to ``(d2, r2)``.

    The day is rounded to the nearest whole day.
    """
    y1, m1, day1 = date_key(d1)
    y2, m2, day2 = date_key(d2)
    gap = (date(y2, m2, day2) - date(y1, m1, day1)).days
    t = max(0.0, min(1.0, (target - r1) / (r2 - r1))) if r2 != r1 else 1.0
    est = date(y1, m1, day1) + timedelta(days=round(t * gap))
    return Crossing(est.isoformat(), "interpolated", gap_days=gap)


CROSSING_RULES: dict[
    CrossingRule, Callable[[str, float, str, float, float], Crossing]
] = {"linear_whole_day": _linear_whole_day}
"""How a crossing is read between its two bracketing capture dates, by the name a trait entry's
``crossing_rule`` states."""


def crossing_date(
    series: list[tuple[str, float]], target: float, rule: CrossingRule
) -> Crossing | None:
    """Return the earliest date the fraction curve reaches ``>= target``, with its bound.

    A crossing between two observed points is read by the :data:`CROSSING_RULES` entry ``rule``
    names; a left-censored crossing (the first observed point already meets the target) or a
    right-censored one (the last observed point still hasn't) is flagged as such. ``None`` only
    when there are no real observed points.
    """
    between = CROSSING_RULES[rule]
    points = _real_points(series)
    if not points:
        return None
    if points[0][1] >= target:
        return Crossing(iso(points[0][0]), "left_censored")
    for (d1, r1), (d2, r2) in zip(points, points[1:]):
        if r2 >= target:
            if r2 == target:
                return Crossing(iso(d2), "exact")
            return between(d1, r1, d2, r2, target)
    return Crossing(iso(points[-1][0]), "right_censored")


def positive_onset_date(series: list[tuple[str, float]]) -> str | None:
    """Return the first date, chronologically, any positive-state observation appears.

    An observation is positive when its fraction is above 0; ``None`` if never.
    """
    for d, r in _real_points(series):
        if r > 0:
            return iso(d)
    return None


def plant_milestones(
    series: list[tuple[str, float]], spec: TraitEntry
) -> dict:
    """Return the phenology dates for one plant's positive-fraction series, keyed by column.

    The keys are the trait's own date and bound columns. The column names (``phenology_prefix`` +
    each milestone key, plus the majority alias), the crossing fractions, the crossing rule and
    the majority mapping come from ``spec``, which is required.
    """
    crossings = {
        key: crossing_date(series, frac, spec.crossing_rule)
        for key, frac in _milestone_targets(spec).items()
    }
    out: dict = {}
    # The majority alias is another entry in ``_milestone_columns``, so it carries its crossing's
    # date and bound under the same condition as every other milestone.
    for (_sfx, key), column in zip(
        _milestone_columns(spec), milestone_date_columns(spec)
    ):
        crossing = crossings.get(key)
        out[column["date"]] = crossing.date if crossing else None
        out[column["bound"]] = crossing.bound if crossing else None
    return out


# ── positive-fraction from predictions under the positive state's attribute ──


def count_by_class(
    document: Key,
    state: PositiveState | None,
    *,
    scope: ClassScope,
) -> tuple[int, int, int]:
    """Return ``(n_total, n_positive, n_unclassified)`` for one image's prediction ``document``.

    ``scope`` is the bucket's own recorded scope. When it declares no attribute of ``state``
    listing its value (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.state_ids`), every
    detection is unclassified, a whole-bucket decision. Otherwise every record is decoded under
    the scope's attributes (:func:`~tcip_annotation.json_io.attribute_ids`): a value its attribute
    does not declare, a record of another subject and a record carrying no value under the state's
    attribute refuse by name (:class:`~tcip_annotation.json_io.UndeclaredValueError`). A record
    whose id there is the state's counts positive.
    """
    from tcip_annotation import json_io

    annotations = json_io.detection_annotations(
        json_io.read_label_document(document).annotations
    )
    total = len(annotations)
    found = scope.state_ids(state)
    if found is None:
        return total, 0, total
    column, wanted = found
    read = scope.document_attributes("this bucket's scope")
    positive = 0
    for i, a in enumerate(annotations):
        ids = json_io.attribute_ids(a, read.subject, read.attributes)
        if ids is None or ids[column] == json_io.UNASSESSED:
            raise json_io.UndeclaredValueError(
                f"{'/'.join(document.parts)}#{i}: a record of {a.subject!r} with attributes "
                f"{a.attributes}, "
                f"where this bucket's every record is of {scope.subject!r} with a value under "
                f"{cast('PositiveState', state).attribute!r}."
            )
        positive += ids[column] == wanted
    return total, positive, 0


class EmptyPopulationError(ValueError):
    """A phenology measurement was asked for with no plants named.

    The population is the caller's explicit plant list.
    """


def measurement_refusals() -> tuple[type[Exception], ...]:
    """Return the exceptions a phenology measurement is refused with, each naming why."""
    from tcip_annotation.json_io import (
        UndeclaredValueError,
        UnreadableLabelDocumentError,
    )
    from tcip_store import StoreError

    return (
        UnreadableLabelDocumentError,
        UndeclaredValueError,
        StoreError,
        EmptyPopulationError,
    )


def population(plants: Sequence[str]) -> list[str]:
    """Return the delivery population as one ordered list of distinct plant ids.

    An empty one refuses (:class:`EmptyPopulationError`).
    """
    ordered = list(dict.fromkeys(str(p) for p in plants))
    if not ordered:
        raise EmptyPopulationError(
            "a delivery needs the plants it is for: pass the plant ids to deliver "
            "(plants=[...]); a mapping or registry names every plot, never the population."
        )
    return ordered


def per_plant_series(
    mapping: dict[str, list[Assignment]],
    buckets: dict[str, Bucket],
    state: PositiveState | None,
    plants: list[str],
) -> dict[str, dict]:
    """Aggregate the buckets' predictions into a per-plant positive-fraction series.

    Only buckets whose scope declares ``state``'s attribute classify; the series covers exactly
    the plants in ``plants`` (:func:`population`'s list).

    ``mapping`` is ``{date: [assignment, ...]}``, a mapping build's own per-date
    :class:`~tcip_mcp.pipelines.postprocessing.plant_mapping.Assignment` rows; ``buckets`` is
    each delivered date's published bucket, by its recorded date. ``plants`` is the delivery's
    population: every plant in it gets an entry, in the order given, and a plant the mapping names
    that is not in it is never read. Every population plant has a series point on every date the
    mapping names, one the mapping assigns it no capture on carrying no image. Returns
    ``{plant_id: {accession, series: [(date, total, positive, unclassified, missing, n_images),
    ...]}}``. An entry naming no plant (``plant_mapping.assignment_is_attributed`` false) is
    excluded from every plant's coverage.

    Coverage is measured against the stems the plant mapping names for each (plant, date): a named
    stem with no corresponding prediction document is a missing observation, and a date the
    mapping names with no delivered bucket counts every stem it names as missing.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        assignment_is_attributed,
        stems_delivery_reads,
    )

    per_plant: dict[str, dict] = {
        plant_id: {"accession": None, "series": []} for plant_id in plants
    }
    # The mapping's own dates are the coverage reference, so a date it names is never absent.
    for date_str in mapping:
        bucket = buckets.get(date_str)
        read_stems = (
            stems_delivery_reads(mapping[date_str], bucket)
            if bucket
            else set()
        )
        by_plant: dict[str, list[int]] = {
            plant_id: [0, 0, 0, 0, 0] for plant_id in plants
        }
        for a in mapping[date_str]:
            if not assignment_is_attributed(a) or a.plant_id not in per_plant:
                continue
            acc = by_plant[a.plant_id]
            acc[4] += 1
            if per_plant[a.plant_id]["accession"] is None:
                per_plant[a.plant_id]["accession"] = a.accession_name
            if bucket is None or a.stem not in read_stems:
                acc[3] += 1
                continue
            total, positive, unclassified = count_by_class(
                prediction_key(bucket.root, bucket.name, a.stem),
                state,
                scope=bucket.scope,
            )
            acc[0] += total
            acc[1] += positive
            acc[2] += unclassified
        for plant_id, counts in by_plant.items():
            per_plant[plant_id]["series"].append((date_str, *counts))
    return per_plant


def per_plant_phenology(
    mapping: dict[str, list[Assignment]],
    buckets: dict[str, Bucket],
    spec: TraitEntry,
    plants: list[str],
) -> dict:
    """Measure per-plant milestones from predictions and a plant mapping against ``spec``.

    The milestones are measured against ``spec``'s ``positive_state`` and milestones, one row per
    plant in ``plants`` and no other, in its order (see :func:`per_plant_series`).

    Returns ``{rows: [...], positive_class_assessed: bool}``. Each row carries the
    positive-fraction series, the milestone dates, coverage-disclosure fields
    (``n_dates_unclassified``, ``n_dates_missing_images``, a date the mapping captured the plant on
    no image counting as missing) and ``complete``, whether every one of its dates is complete:
    imaged, every detection carrying the positive state's attribute, and fully observed. A plant's
    milestones are computed only when it is complete. ``positive_class_assessed`` is ``True`` iff
    at least one date, anywhere in the delivery, was complete.
    """
    per_plant = per_plant_series(mapping, buckets, spec.positive_state, plants)
    rows = []
    any_classified_date = False
    for plant_id, info in per_plant.items():
        usable_dates = [
            (d, total, positive)
            for (d, total, positive, unclassified, missing, n_images) in info[
                "series"
            ]
            if unclassified == 0 and missing == 0 and n_images > 0
        ]
        if usable_dates:
            any_classified_date = True
        # total==0 detected no objects, so it is not an observation of the fraction; kept only when
        # something was detected, a real 0% included.
        frac_series = [
            (d, positive / total)
            for (d, total, positive) in usable_dates
            if total
        ]
        complete = (
            len(usable_dates) == len(info["series"])
            and len(info["series"]) > 0
        )
        row = {
            "plant_id": plant_id,
            "accession": info["accession"],
            "n_dates": len(info["series"]),
            "n_observed_dates": len(frac_series),
            "n_dates_unclassified": sum(1 for s in info["series"] if s[3] > 0),
            "n_dates_missing_images": sum(
                1 for s in info["series"] if s[4] > 0 or s[5] == 0
            ),
            "complete": complete,
            "series": [
                {
                    "date": d,
                    "n_total": total,
                    "n_positive": positive,
                    "n_unclassified": unclassified,
                    "n_missing": missing,
                    "n_images": n_images,
                    "ratio": (
                        positive / total
                        if total and unclassified == 0 and missing == 0
                        else None
                    ),
                }
                for (
                    d,
                    total,
                    positive,
                    unclassified,
                    missing,
                    n_images,
                ) in info["series"]
            ],
        }
        # An excluded plant goes through the producer with an empty series, so every row has one
        # key set.
        row.update(plant_milestones(frac_series if complete else [], spec))
        rows.append(row)
    return {"rows": rows, "positive_class_assessed": any_classified_date}


@dataclass(frozen=True)
class PhenologyMeasurement:
    """One trait's per-plant phenology over a plant mapping and its delivered buckets.

    It holds the confirmed revision it is measured under, the buckets by date, the rows, and the
    mapping's delivery disclosure.
    """

    revision: TraitRevision
    buckets: dict[str, Bucket]
    rows: list[dict]
    positive_class_assessed: bool
    plant_mapping: dict
    plants: list[str]

    def curve_rows(self) -> list[dict]:
        """Return per-(plant, date) rows from the existing series."""
        return [
            {
                "plant_id": row["plant_id"],
                "accession": row["accession"],
                **point,
            }
            for row in self.rows
            for point in row["series"]
        ]

    def milestone_rows(self) -> list[dict]:
        """Return the per-plant rows without their ``series``."""
        return [
            {k: v for k, v in row.items() if k != "series"}
            for row in self.rows
        ]

    def result(self, *, curves: bool) -> Result:
        """Return the delivered result: the milestone rows, or the curve rows with ``curves``.

        The rows are under their columns, with the population and mapping disclosure.
        """
        from tcip_mcp.delivery import Result

        return Result(
            curve_csv_columns()
            if curves
            else phenology_csv_columns(self.revision.entry),
            self.curve_rows() if curves else self.milestone_rows(),
            population=self.plants,
            plant_mapping=self.plant_mapping,
        )


def measure_phenology(
    project: Path,
    *,
    trait: str,
    mapping_name: str,
    dataset_root: Path,
    buckets: Sequence[str],
    plants: Sequence[str],
) -> PhenologyMeasurement:
    """Measure ``trait``'s phenology over the named ``buckets`` for exactly ``plants``.

    The buckets are published under the established location ``dataset_root``, each at the
    capture date its record states (:func:`~tcip_mcp.buckets.by_recorded_date`); the plants are
    read through the plant mapping ``mapping_name``.

    Reads the trait's latest confirmed revision stating a ``state_crossing_dates``
    operationalization, bound against the dataset's registry (refusing, ``ValueError``, a dataset
    with none), then the buckets and the mapping (``plant_mapping.resolve_delivery_mapping``);
    each refuses as it does, and so does :func:`per_plant_phenology`
    (:func:`measurement_refusals`).
    """
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.operationalization import confirmed_revision
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.subject_registry import registry_for_dataset_root
    from tcip_mcp.traits import STATE_CROSSING_DATES

    registry = registry_for_dataset_root(dataset_root)
    if registry is None:
        raise ValueError(
            f"{dataset_root} holds no subject registry: register the dataset or write its "
            "subjects.json, so the trait's positive state can be checked against it."
        )
    revision = confirmed_revision(
        STATE_CROSSING_DATES, project=project, trait=trait, registry=registry
    )
    wanted = population(plants)
    dated = by_recorded_date(
        read_bucket(dataset_root, name) for name in buckets
    )
    mapping_build, verified = plant_mapping.resolve_delivery_mapping(
        project, mapping_name, dated
    )
    measured = per_plant_phenology(
        mapping_build.assignments, dated, revision.entry, wanted
    )
    return PhenologyMeasurement(
        revision=revision,
        buckets=dated,
        rows=measured["rows"],
        positive_class_assessed=measured["positive_class_assessed"]
        and any(
            b.scope.state_ids(revision.entry.positive_state) is not None
            for b in dated.values()
        ),
        plant_mapping=mapping_build.delivery_disclosure(verified, list(dated)),
        plants=wanted,
    )


def deliver_phenology(
    project: Path,
    measurement: PhenologyMeasurement,
    *,
    curves: bool,
    output_path: Path,
    acknowledgment_id: str | None,
    door: str,
    actor: str | None,
) -> dict[str, Any]:
    """Deliver ``measurement``'s milestone or curve rows as the CSV at ``output_path``.

    The rows are its milestone rows, or its per-(plant, date) curve rows with ``curves``; the
    delivery is by ``actor``.

    The buckets clear :func:`~tcip_mcp.delivery.gate` for a ``state_crossing_dates`` delivery,
    the recorded acknowledgment ``acknowledgment_id`` shipping them unvalidated when it does not
    validate them.
    Writes exactly :func:`phenology_csv_columns` (or :func:`curve_csv_columns`) with the
    delivery's one event under ``door``, its population and plant-mapping disclosure
    (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was delivered.
    """
    from tcip_mcp.delivery import deliver_csv, gate
    from tcip_mcp.traits import STATE_CROSSING_DATES

    result = measurement.result(curves=curves)
    clearance = gate(
        project,
        list(measurement.buckets.values()),
        delivery_kind=STATE_CROSSING_DATES,
        revision=measurement.revision,
        result=result,
        acknowledgment_id=acknowledgment_id,
    )
    delivered = deliver_csv(
        project,
        output_path,
        result,
        clearance=clearance,
        revision=measurement.revision,
        door=door,
        delivery_kind=STATE_CROSSING_DATES,
        actor=actor,
    )
    return {
        **delivered,
        "n_rows": len(result.rows),
        "columns": list(result.columns),
    }

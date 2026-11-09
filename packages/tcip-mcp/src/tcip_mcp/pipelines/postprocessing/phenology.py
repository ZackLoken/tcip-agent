"""Canonical phenology measurement: a trait's positive-fraction milestones.

The measurement serves whichever registered trait it is computed for. The positive-state fraction
is the fraction of a plant's detected objects a classifier calls the trait's ``positive_state``.
Milestone columns come entirely from the trait's own ``TraitEntry`` (``phenology_prefix`` plus
each ``milestone_fractions`` entry):

    ``<prefix>_<NN>per_date``            = the date the positive fraction first crosses NN%,
                                            for each fraction the spec declares
    ``<prefix>_<majority_label>_date``   = the crossing at ``TraitEntry.majority_fraction``,
                                            one of those fractions, present only when the
                                            spec names one

``positive_onset_date`` (the first date any positive-state observation appears) is a separate
helper, not the delivered trait.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, computed_field

from tcip_store import Key

if TYPE_CHECKING:
    from tcip_mcp.buckets import Bucket
    from tcip_mcp.delivery import PhenologyDelivered, Result
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        Assignment,
        PlantMappingDisclosure,
    )
    from tcip_mcp.traits import (
        CrossingRule,
        PositiveState,
        TraitEntry,
        TraitRevision,
    )


_KEY_SUFFIX = "per"
_KEY_SHAPE = re.compile(rf"\d{{2,}}{_KEY_SUFFIX}")


def _percent(fraction: float) -> int:
    """Return the whole percent nearest ``fraction``."""
    return int(round(fraction * 100))


def milestone_key(fraction: float) -> str:
    """Return the key a milestone crossing at ``fraction`` is named by, ``NNper``.

    ``NN`` is the fraction's whole percent; the key is the milestone's CSV column suffix.
    """
    return f"{_percent(fraction):02d}{_KEY_SUFFIX}"


def is_milestone_key(text: str) -> bool:
    """Return whether ``text`` is shaped like a :func:`milestone_key`: digits, then ``per``."""
    return _KEY_SHAPE.fullmatch(text) is not None


def is_whole_percent(fraction: float) -> bool:
    """Return whether ``fraction`` is exactly the fraction its :func:`milestone_key` names."""
    return _percent(fraction) / 100 == fraction


def _milestone_targets(spec: TraitEntry) -> dict[str, float]:
    """Return the trait's milestone crossing fractions, each by its :func:`milestone_key`."""
    return {milestone_key(f): f for f in spec.milestone_fractions}


def _milestone_columns(spec: TraitEntry) -> list[tuple[str, str]]:
    """Return ``(column_suffix, crossing_key)`` for every milestone this trait delivers.

    The majority alias enters, first, only when the spec names a majority fraction.
    """
    cols = [(key, key) for key in _milestone_targets(spec)]
    if spec.majority_fraction is not None:
        cols.insert(
            0, (spec.majority_label, milestone_key(spec.majority_fraction))
        )
    return cols


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
    (a wide gap is weaker evidence for the same interpolated date); ``None`` for ``exact``, which
    spans no gap, and for ``left_censored``/``right_censored``, where no bracket on the censored
    side exists.
    """

    date: date
    bound: CrossingBound
    gap_days: int | None = None


type CrossingBound = Literal[
    "exact", "left_censored", "interpolated", "right_censored"
]
"""How a milestone crossing's date was read (:class:`Crossing`)."""

type Projection = Literal["curves", "milestones"]
"""A phenology measurement's two deliverable projections: the per-(plant, capture) curve rows and
the per-plant milestone rows."""


def export_filename(trait: str, projection: Projection) -> str:
    """Return the file name ``trait``'s ``projection`` is exported under."""
    return f"{trait}_{projection}.csv"


class CurveRow(BaseModel):
    """One plant's counts on one capture, the positive fraction's ``ratio`` derived from them.

    ``capture`` is the capture's name, as the dataset's tree lists it.
    """

    plant_id: str
    accession: str | None
    capture: str
    n_images: int
    n_total: int
    n_positive: int
    n_unclassified: int
    n_missing: int

    @property
    def classified(self) -> bool:
        """Whether the date was imaged and every detection on it carries the state's attribute."""
        return (
            self.n_unclassified == 0
            and self.n_missing == 0
            and self.n_images > 0
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ratio(self) -> float | None:
        """The positive fraction, an observation only when classified and something was detected.

        ``None`` otherwise: a date detecting no objects observes no fraction, a real 0% aside.
        """
        return (
            self.n_positive / self.n_total
            if self.classified and self.n_total
            else None
        )


class MilestoneRow(BaseModel):
    """One plant's coverage and milestone crossings.

    ``milestones`` holds each milestone's crossing (:func:`plant_milestones`) by the key
    :func:`_milestone_columns` names it, in that order, ``None`` where the plant's series has no
    observed point.
    """

    plant_id: str
    accession: str | None
    n_dates: int
    n_observed_dates: int
    n_dates_unclassified: int
    n_dates_missing_images: int
    milestones: dict[str, Crossing | None]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def complete(self) -> bool:
        """Whether it has dates, each imaged and carrying the state on every detection."""
        return (
            self.n_dates > 0
            and self.n_dates_unclassified == 0
            and self.n_dates_missing_images == 0
        )

    def csv_record(self, spec: TraitEntry) -> dict[str, Any]:
        """Return this row as the delivered CSV writes it, each crossing as its two columns."""
        record = self.model_dump(exclude={"milestones"})
        for key, date_column, bound_column in _milestone_csv_columns(spec):
            crossing = self.milestones[key]
            record[date_column] = None if crossing is None else crossing.date
            record[bound_column] = None if crossing is None else crossing.bound
        return record


def _milestone_csv_columns(spec: TraitEntry) -> list[tuple[str, str, str]]:
    """Return each milestone's key with the CSV columns its date and its bound are written to."""
    return [
        (
            key,
            f"{spec.phenology_prefix}_{key}_date",
            f"{spec.phenology_prefix}_{key}_date_bound",
        )
        for key, _ in _milestone_columns(spec)
    ]


def phenology_csv_columns(spec: TraitEntry) -> list[str]:
    """Return the delivered per-plant phenology CSV schema for one trait's ``TraitEntry``.

    The columns are the plant, its coverage, each milestone date and its evidentiary bound, then
    :data:`~tcip_mcp.delivery.DELIVERY_COLUMNS`.
    """
    from tcip_mcp.delivery import DELIVERY_COLUMNS

    columns = _milestone_csv_columns(spec)
    return [
        *(name for name in MilestoneRow.model_fields if name != "milestones"),
        *MilestoneRow.model_computed_fields,
        *[date for _, date, _ in columns],
        *[bound for _, _, bound in columns],
        *DELIVERY_COLUMNS,
    ]


def curve_csv_columns() -> list[str]:
    """Return the delivered per-(plant, date) curve CSV schema.

    The columns are a :class:`CurveRow`'s, the counts the fraction is built from, then
    :data:`~tcip_mcp.delivery.DELIVERY_COLUMNS`.
    """
    from tcip_mcp.delivery import DELIVERY_COLUMNS

    return [
        *CurveRow.model_fields,
        *CurveRow.model_computed_fields,
        *DELIVERY_COLUMNS,
    ]


def _real_points(series: list[tuple[str, float]]) -> list[tuple[date, float]]:
    """Return the series' dated points, in its order; a capture no date names is dropped.

    ``series`` runs in the order the dataset lists its captures
    (:func:`~tcip_mcp.dataset_layout.capture_order`), and each capture's date is
    :func:`~tcip_mcp.dataset_layout.capture_date`'s.
    """
    from tcip_mcp.dataset_layout import capture_date

    return [
        (day, r) for d, r in series if (day := capture_date(d)) is not None
    ]


# ── milestones ───────────────────────────────────────────────────────────


def _linear_whole_day(
    d1: date, r1: float, d2: date, r2: float, target: float
) -> Crossing:
    """Return the day ``target`` is reached on the straight line from ``(d1, r1)`` to ``(d2, r2)``.

    The day is rounded to the nearest whole day.
    """
    gap = (d2 - d1).days
    t = max(0.0, min(1.0, (target - r1) / (r2 - r1))) if r2 != r1 else 1.0
    est = d1 + timedelta(days=round(t * gap))
    return Crossing(est, "interpolated", gap_days=gap)


CROSSING_RULES: dict[
    CrossingRule, Callable[[date, float, date, float, float], Crossing]
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
        return Crossing(points[0][0], "left_censored")
    for (d1, r1), (d2, r2) in pairwise(points):
        if r2 >= target:
            if r2 == target:
                return Crossing(d2, "exact")
            return between(d1, r1, d2, r2, target)
    return Crossing(points[-1][0], "right_censored")


def positive_onset_date(series: list[tuple[str, float]]) -> str | None:
    """Return the first date, chronologically, any positive-state observation appears.

    An observation is positive when its fraction is above 0; ``None`` if never.
    """
    for d, r in _real_points(series):
        if r > 0:
            return d.isoformat()
    return None


def plant_milestones(
    series: list[tuple[str, float]], spec: TraitEntry
) -> dict[str, Crossing | None]:
    """Return each milestone's crossing of one plant's positive-fraction series.

    Each is keyed as :func:`_milestone_columns` names it, the majority alias among them, and is
    ``None`` where the series has no observed point. The crossing fractions, the crossing rule and
    the majority mapping come from ``spec``, which is required.
    """
    crossings = {
        key: crossing_date(series, frac, spec.crossing_rule)
        for key, frac in _milestone_targets(spec).items()
    }
    return {sfx: crossings[key] for sfx, key in _milestone_columns(spec)}


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
) -> dict[str, list[CurveRow]]:
    """Aggregate the buckets' predictions into each plant's curve rows, one per mapping date.

    Only buckets whose scope declares ``state``'s attribute classify; the rows cover exactly the
    plants in ``plants`` (:func:`population`'s list).

    ``mapping`` is ``{date: [assignment, ...]}``, a mapping build's own per-date
    :class:`~tcip_mcp.pipelines.postprocessing.plant_mapping.Assignment` rows; ``buckets`` is
    each delivered date's published bucket, by its recorded date. ``plants`` is the delivery's
    population: every plant in it gets an entry, in the order given, and a plant the mapping names
    that is not in it is never read. Every population plant has a row on every date the mapping
    names, in capture order, one the mapping assigns it no capture on carrying no image. Returns
    ``{plant_id: [CurveRow, ...]}``, every row of a plant carrying the accession its first
    attributed assignment in capture order names. An entry naming no plant
    (``plant_mapping.assignment_is_attributed`` false) is excluded from every plant's coverage.

    Coverage is measured against the stems the plant mapping names for each (plant, date): a named
    stem with no corresponding prediction document is a missing observation, and a date the
    mapping names with no delivered bucket counts every stem it names as missing.
    """
    from tcip_mcp.dataset_layout import capture_order
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        assignment_is_attributed,
        stems_delivery_reads,
    )

    # The mapping's own dates are the coverage reference, so a date it names is never absent; they
    # are walked in the order the dataset lists its captures.
    dates = capture_order(mapping)
    named: dict[str, dict[str, list[Assignment]]] = {
        date_str: {plant_id: [] for plant_id in plants} for date_str in dates
    }
    for date_str in dates:
        for a in mapping[date_str]:
            if assignment_is_attributed(a) and a.plant_id in named[date_str]:
                named[date_str][a.plant_id].append(a)
    accessions = {
        plant_id: next(
            (
                a.accession_name
                for date_str in dates
                for a in named[date_str][plant_id]
                if a.accession_name is not None
            ),
            None,
        )
        for plant_id in plants
    }
    rows: dict[str, list[CurveRow]] = {plant_id: [] for plant_id in plants}
    for date_str in dates:
        bucket = buckets.get(date_str)
        read_stems = (
            stems_delivery_reads(mapping[date_str], bucket)
            if bucket
            else set()
        )
        for plant_id, assigned in named[date_str].items():
            counts = [
                count_by_class(
                    bucket.stem_keys[a.stem], state, scope=bucket.scope
                )
                for a in assigned
                if bucket and a.stem in read_stems
            ]
            rows[plant_id].append(
                CurveRow(
                    plant_id=plant_id,
                    accession=accessions[plant_id],
                    capture=date_str,
                    n_images=len(assigned),
                    n_total=sum(c[0] for c in counts),
                    n_positive=sum(c[1] for c in counts),
                    n_unclassified=sum(c[2] for c in counts),
                    n_missing=len(assigned) - len(counts),
                )
            )
    return rows


@dataclass(frozen=True)
class PlantPhenology:
    """A population's phenology rows: one milestone row per plant, one curve row per (plant, date).

    ``any_date_complete`` is whether any date, anywhere in the population, was complete.
    """

    milestones: list[MilestoneRow]
    curves: list[CurveRow]
    any_date_complete: bool


def per_plant_phenology(
    mapping: dict[str, list[Assignment]],
    buckets: dict[str, Bucket],
    spec: TraitEntry,
    plants: list[str],
) -> PlantPhenology:
    """Measure per-plant milestones from predictions and a plant mapping against ``spec``.

    The milestones are measured against ``spec``'s ``positive_state`` and milestones, one row per
    plant in ``plants`` and no other, in its order (see :func:`per_plant_series`).

    Each milestone row carries the milestone dates, coverage-disclosure fields
    (``n_dates_unclassified``, ``n_dates_missing_images``, a date the mapping captured the plant on
    no image counting as missing) and ``complete``, whether every one of its dates is complete:
    imaged, every detection carrying the positive state's attribute, and fully observed. A plant's
    milestones are computed only when it is complete. Each curve row is one date of a plant's
    positive-fraction series.
    """
    per_plant = per_plant_series(mapping, buckets, spec.positive_state, plants)
    milestones: list[MilestoneRow] = []
    curves: list[CurveRow] = []
    for plant_id, rows in per_plant.items():
        frac_series = [
            (r.capture, r.ratio) for r in rows if r.ratio is not None
        ]
        # An excluded plant goes through the producer with no rows, so every row has one key set.
        row = MilestoneRow(
            plant_id=plant_id,
            accession=rows[0].accession if rows else None,
            n_dates=len(rows),
            n_observed_dates=len(frac_series),
            n_dates_unclassified=sum(1 for r in rows if r.n_unclassified > 0),
            n_dates_missing_images=sum(
                1 for r in rows if r.n_missing > 0 or r.n_images == 0
            ),
            milestones={},
        )
        row.milestones = plant_milestones(
            frac_series if row.complete else [], spec
        )
        milestones.append(row)
        curves.extend(rows)
    return PlantPhenology(
        milestones=milestones,
        curves=curves,
        any_date_complete=any(r.classified for r in curves),
    )


@dataclass(frozen=True)
class PhenologyMeasurement:
    """One trait's per-plant phenology over a plant mapping and its delivered buckets.

    It holds the confirmed revision it is measured under, the buckets by date, the population's
    rows (:func:`per_plant_phenology`), and the mapping's delivery disclosure.
    """

    revision: TraitRevision
    buckets: dict[str, Bucket]
    phenology: PlantPhenology
    plant_mapping: PlantMappingDisclosure
    plants: list[str]

    @property
    def positive_class_assessed(self) -> bool:
        """Whether any date was complete and any bucket declares the positive state's attribute."""
        return self.phenology.any_date_complete and any(
            b.scope.state_ids(self.revision.entry.positive_state) is not None
            for b in self.buckets.values()
        )

    def result(self, projection: Projection) -> Result:
        """Return the delivered result of ``projection``: the milestone rows or the curve rows.

        The rows are under their columns, with the population and mapping disclosure.
        """
        from tcip_mcp.delivery import Result

        entry = self.revision.entry
        curves = projection == "curves"
        return Result(
            curve_csv_columns() if curves else phenology_csv_columns(entry),
            [row.model_dump() for row in self.phenology.curves]
            if curves
            else [row.csv_record(entry) for row in self.phenology.milestones],
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
    return PhenologyMeasurement(
        revision=revision,
        buckets=dated,
        phenology=per_plant_phenology(
            mapping_build.assignments, dated, revision.entry, wanted
        ),
        plant_mapping=mapping_build.delivery_disclosure(verified, list(dated)),
        plants=wanted,
    )


def deliver_phenology(
    project: Path,
    measurement: PhenologyMeasurement,
    *,
    projection: Projection,
    output_path: Path,
    acknowledgment_id: str | None,
    door: str,
    actor: str | None,
) -> PhenologyDelivered:
    """Deliver ``measurement``'s ``projection`` as the CSV at ``output_path``.

    The rows are its milestone rows, or its per-(plant, date) curve rows for ``curves``; the
    delivery is by ``actor``.

    The buckets clear :func:`~tcip_mcp.delivery.gate` for a ``state_crossing_dates`` delivery,
    the recorded acknowledgment ``acknowledgment_id`` shipping them unvalidated when it does not
    validate them.
    Writes exactly :func:`phenology_csv_columns` (or :func:`curve_csv_columns`) with the
    delivery's one event under ``door``, its population and plant-mapping disclosure
    (:func:`~tcip_mcp.delivery.deliver_csv`); returns what was delivered.
    """
    from tcip_mcp.delivery import PhenologyDelivered, deliver_csv, gate
    from tcip_mcp.traits import STATE_CROSSING_DATES

    result = measurement.result(projection)
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
    return PhenologyDelivered(
        **dict(delivered),
        n_plants=len(measurement.plants),
        n_rows=len(result.rows),
        columns=list(result.columns),
    )

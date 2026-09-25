"""Canonical phenology measurement: a trait's positive-fraction milestones, for whichever
registered trait it's computed for.

The positive-state fraction = the fraction of a plant's detected objects that are in the trait's
positive/measured state, an expert-defined visible morphological stage emitted by a validated
classifier (the trait's ``positive_value``), never a geometric proxy such as bbox height. Milestone
columns come entirely from the trait's own ``TraitSpec`` (``phenology_prefix`` plus each
``milestone_fractions`` entry):

    ``<prefix>_<NN>per_date``            = the date the positive fraction first crosses NN%,
                                            for each fraction the spec declares
    ``<prefix>_<majority_label>_date``   = the majority-crossing alias
                                            (``TraitSpec.majority_milestone``), present only
                                            when the spec names one

A spec's majority-crossing alias is a breeder-confirmed reading of the trait's own definition text,
flagged crossing-unconfirmed (``TraitSpec.crossing_unconfirmed``) until confirmed.
``positive_onset_date`` (the first date any positive-state observation appears) is a separate
helper, not the delivered trait.

Pure (stdlib only, plus the torch-free ``resolution.py`` and ``operationalization.py``): it
    consumes prediction buckets and never touches pixels or model machinery. A classified bucket's
    own recorded scope (``resolution.bucket_scope``) says which prediction records carry the
    classifier's decoded call: ``.subject`` names the object class and the call sits under
    ``.attributes[attribute]`` (see ``count_by_class``). A bucket that never assessed the trait's
    positive-class axis is disclosed per plant and per date by ``per_plant_phenology``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from tcip_mcp.dataset_layout import label_filename
from tcip_mcp.operationalization import OperationalizationBasis
from tcip_mcp.pipelines.resolution import Acknowledgment, bucket_scope

if TYPE_CHECKING:
    from tcip_mcp.pipelines.resolution import BucketScope


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


def milestone_date_columns(spec) -> list[str]:
    """The milestone/date column names a trait's phenology delivery carries, a proper subset of
    ``phenology_csv_columns`` (no ``plant_id``/provenance columns).
    """
    return [f"{spec.phenology_prefix}_{sfx}_date" for sfx, _ in _milestone_columns(spec)]


def majority_crossing_unconfirmed_column(spec) -> str | None:
    """The column that marks a trait's majority alias as not yet breeder-confirmed, or ``None``
    when the spec names no majority crossing for it to qualify.
    """
    if not spec.majority_milestone:
        return None
    return f"{spec.phenology_prefix}_{spec.majority_label}_crossing_unconfirmed"


def phenology_csv_columns(spec) -> list[str]:
    """The delivered per-plant phenology CSV schema for one trait, derived from its ``TraitSpec``.

    The milestone/alias column names come from the spec (``phenology_prefix`` + each milestone key,
    plus the majority alias/crossing-unconfirmed columns built from ``majority_label``). The
    surrounding provenance columns (operating point, classifier validation, producer identity,
    coverage disclosure) are trait-neutral.
    """
    crossing_unconfirmed_column = majority_crossing_unconfirmed_column(spec)
    return [
        "plant_id",
        "accession",
        "n_dates",
        # A plant can be fully classified and fully observed (0 unclassified, 0 missing) while
        # still having zero real detections on every date (before emergence, or a genuinely empty
        # scene), n_dates alone doesn't distinguish that from real detection data. per_plant_phenology
        # already computes this per row; without it here it would be silently dropped by
        # DictWriter's extrasaction="ignore" for not being in this column set.
        "n_observed_dates",
        "n_dates_unclassified",
        "n_dates_missing_images",
        *milestone_date_columns(spec),
        # Each milestone's evidentiary bound, beside the date it qualifies.
        # ``plant_milestones`` has always emitted these; without this column,
        # ``write_phenology_csv``'s DictWriter would drop them and a left-censored crossing, one
        # where the first observation already met the target, so the true date is only an upper
        # bound, would be delivered indistinguishable from a measured one. That is a precision
        # claim the data does not support, which is the failure mode this platform exists to
        # prevent.
        *[f"{c}_bound" for c in milestone_date_columns(spec)],
        *([crossing_unconfirmed_column] if crossing_unconfirmed_column else []),
        *PROVENANCE_COLUMNS,
    ]


# How the counts behind a delivered number were produced, and whether the measurement is
# trustworthy, a delivered phenotype must carry this so it can be traced. Trait-neutral, and the
# single owner of the tail, so every delivered shape (milestones, curves) carries the same chain
# rather than each door listing its own.
PROVENANCE_COLUMNS = [
    # float, or one entry per date in dates_delivered order, joined with ";" when the dates
    # recorded different confs, blank for a date with no numeric conf.
    "operating_point_conf",
    "operating_point_validated",
    "positive_state_classifier_validated",
    # Every gated dimension that did not validate, ";"-joined, blank when none.
    "unvalidated_dimensions",
    # Producing-model identity, the exact checkpoint (content hash) + run behind the counts.
    "producer_model_sha256",
    "producing_experiment_id",
    "produced_at",
    # Which experiment and row answered for the buckets' claims, empty when any read bucket is unbound.
    "validation_record",
    # The plant mapping this delivery attributed detections through. What verify_mapping_inputs
    # could not check travels once, on the delivery event, never repeated on every row.
    "plant_mapping_sha256",
    # The delivered dates and this delivery's own unattributed-capture count scoped to them
    # (never the mapping's own n_dates_missing_images span), plus the attribution granularity.
    "dates_delivered",
    "images_unattributed",
    "plant_attribution",
    # Who acknowledged this delivery unvalidated, and why; blank on either when nothing was
    # acknowledged (a fully-validated delivery, or one refused outright).
    "acknowledged_by",
    "acknowledgment_reason",
]


# The per-(plant, date) columns ``per_plant_series`` produces, before the provenance tail.
CURVE_MEASUREMENT_COLUMNS = [
    "plant_id", "accession", "date",
    "n_images", "n_total", "n_positive", "n_unclassified", "n_missing", "ratio",
]


def curve_csv_columns() -> list[str]:
    """The delivered per-(plant, date) curve CSV schema.

    A curve is the same phenology measurement as the milestone summary, un-summarized, which is why it
    takes the identical delivery gate, so it carries the identical provenance tail. Trait-neutral:
    unlike the milestone schema it names no crossings, only the counts the fraction is built from.
    """
    return [*CURVE_MEASUREMENT_COLUMNS, *PROVENANCE_COLUMNS]


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
    prefix = spec.phenology_prefix
    crossings = {key: crossing_date(series, frac) for key, frac in _milestone_targets(spec).items()}
    out: dict = {}
    # The majority alias is just another entry in ``_milestone_columns`` pointing at the crossing
    # the spec names for it, so it carries the same date + evidentiary bound as every other
    # milestone and cannot be emitted under a different condition than the schema declares it under.
    for sfx, key in _milestone_columns(spec):
        crossing = crossings.get(key)
        out[f"{prefix}_{sfx}_date"] = crossing.date if crossing else None
        out[f"{prefix}_{sfx}_date_bound"] = crossing.bound if crossing else None
    return out


# ── positive-fraction from classified predictions ────────────────────────


def resolve_positive_class_id(spec, predictions_by_date: dict[str, str]) -> tuple[int | None, str]:
    """Resolve a trait's positive class id from a prediction bucket's own recorded ``id_map``.

    Returns ``(class_id, message)``; ``class_id`` is ``None`` when no bucket's ``id_map`` contains
    the trait's positive value.
    """
    name = spec.positive_value
    if not name:
        return None, f"trait {spec.name!r} defines no positive_value"
    for pred_dir in predictions_by_date.values():
        id_map = bucket_id_map(Path(pred_dir))
        if id_map is not None and name in id_map:
            try:
                return int(id_map[name]), f"resolved {name!r} -> class {id_map[name]} from {pred_dir}"
            except (TypeError, ValueError):
                continue
    return None, (f"no prediction bucket's recorded id_map contains {name!r}, the classifier that "
                  "produced these predictions never assessed this trait's positive class")


def bucket_id_map(pred_dir: Path) -> dict | None:
    """The bucket's recorded ``id_map`` (name -> int) from ``operating_point.json``, or ``None`` if
    absent or not a dict.
    """
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(pred_dir)
    if not isinstance(sidecar, dict):
        return None
    id_map = sidecar.get("id_map")
    return id_map if isinstance(id_map, dict) else None


def count_by_class(
    json_path: Path, id_map: dict | None, positive_value: str, *,
    scope: BucketScope | None,
) -> tuple[int, int, int]:
    """``(n_total, n_positive, n_unclassified)`` for one image's predictions.

    ``scope`` is the bucket's own recorded scope
    (:func:`~tcip_mcp.pipelines.resolution.bucket_scope`). Under no scope, a detector scope, or a
    classified scope whose recorded ``id_map`` lacks ``positive_value``, every detection is
    unclassified: a whole-bucket decision. A detector bucket whose one map key happens to equal
    ``positive_value`` never counts a positive.

    Under a classified scope, every record is held to
    :func:`~tcip_annotation.json_io.require_classified_record` under the bucket's own recorded
    vocabulary (``id_map``'s keys): a value outside that vocabulary refuses by name. A record whose
    value equals ``positive_value`` counts positive; every other classified record counts toward
    neither positive nor unclassified. A classified bucket's record carries the object class in
    ``subject`` and the classifier's decoded call in ``attributes[attribute]``.

    Called only for a prediction file confirmed to exist.
    """
    from tcip_annotation import json_io

    annotations = json_io.detection_annotations(json_path)
    total = len(annotations)
    if scope is None or not scope.classified or not id_map or positive_value not in id_map:
        return total, 0, total
    assert scope.subject is not None and scope.attribute is not None  # classified implies both
    positive = 0
    for i, a in enumerate(annotations):
        value = json_io.require_classified_record(
            a, subject=scope.subject, attribute=scope.attribute, vocabulary=set(id_map),
            source=f"{json_path}#{i}")
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


def _population(plants: Sequence[str]) -> list[str]:
    """The delivery population as one ordered list of distinct plant ids, refusing an empty one."""
    ordered = list(dict.fromkeys(str(p) for p in plants))
    if not ordered:
        raise EmptyPopulation(
            "a phenology measurement needs the plants it is for: pass the plant ids to deliver "
            "(plants=[...]); a mapping names every plot in its plant CSVs, never the population."
        )
    return ordered


def per_plant_series(
    mapping: dict[str, list],
    predictions_by_date: dict[str, str],
    positive_value: str,
    plants: Sequence[str],
) -> dict[str, dict]:
    """Aggregate classified predictions into a per-plant positive-fraction series, for exactly the
    plants in ``plants``.

    ``mapping`` is ``{date: [assignment, ...]}`` where each assignment has ``.stem`` /
    ``.plot_name`` / ``.accession_name`` (attributes or dict keys). ``plants`` is the delivery's
    population: every plant in it gets an entry, in the order given, and a plant the mapping names
    that is not in it is never read. A population plant the mapping never names on a date has no
    entry for that date; one the mapping never names at all has an empty series. Returns
    ``{plant_id: {accession, series: [(date, total, positive, unclassified, missing, n_images),
    ...]}}``. An entry naming no plant (``plant_mapping.assignment_is_attributed`` false) is
    excluded from every plant's coverage; its count is
    ``plant_mapping.MappingBuild.unattributed``'s.

    Coverage is measured against the stems the plant mapping names for each (plant, date): a named
    stem with no corresponding prediction file is a missing observation, and a date the mapping
    names with no ``predictions_by_date`` entry counts every stem it names as missing. Which stems
    count as read is ``plant_mapping.stems_delivery_reads``.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        assignment_is_attributed,
        stems_delivery_reads,
    )

    def _attr(a, name):
        return getattr(a, name, None) if not isinstance(a, dict) else a.get(name)

    population = _population(plants)
    per_plant: dict[str, dict] = {
        plant_id: {"accession": None, "series": []} for plant_id in population
    }
    # Iterate the mapping's own dates, not predictions_by_date's, the mapping is the coverage
    # reference, so a date it names is never silently absent just because the caller dropped it.
    for date_str in mapping:
        pred_dir = predictions_by_date.get(date_str)
        pred_path = Path(pred_dir) if pred_dir else None
        id_map = bucket_id_map(pred_path) if pred_path is not None else None
        scope = bucket_scope(pred_path) if pred_path is not None else None
        read_stems = stems_delivery_reads(mapping[date_str], pred_dir) if pred_dir else set()
        # [total, positive, unclassified, missing, n_images] per plant: n_images is every stem
        # the mapping names for this (plant, date), missing is one this delivery doesn't read.
        by_plant: dict[str, list[int]] = {}
        accession: dict[str, Optional[str]] = {}
        for a in mapping[date_str]:
            if not assignment_is_attributed(a):
                continue
            plant_id = _attr(a, "plot_name")
            if plant_id not in per_plant:
                continue
            acc = by_plant.setdefault(plant_id, [0, 0, 0, 0, 0])
            acc[4] += 1
            accession.setdefault(plant_id, _attr(a, "accession_name"))
            stem = _attr(a, "stem")
            if pred_path is None or stem not in read_stems:
                acc[3] += 1
                continue
            total, positive, unclassified = count_by_class(
                pred_path / label_filename(stem), id_map, positive_value, scope=scope)
            acc[0] += total
            acc[1] += positive
            acc[2] += unclassified
        for plant_id, (total, positive, unclassified, missing, n_images) in by_plant.items():
            entry = per_plant[plant_id]
            if entry["accession"] is None:
                entry["accession"] = accession.get(plant_id)
            entry["series"].append((date_str, total, positive, unclassified, missing, n_images))
    return per_plant


def per_plant_phenology(
    mapping: dict[str, list],
    predictions_by_date: dict[str, str],
    positive_value: str,
    spec,
    plants: Sequence[str],
) -> dict:
    """Full canonical pipeline: classified predictions + plant mapping -> per-plant milestones, one
    row per plant in ``plants`` and no other, in its order (see :func:`per_plant_series`).

    Returns ``{rows: [...], positive_class_assessed: bool}``. Each row carries the
    positive-fraction series, the milestone dates, and coverage-disclosure fields
    (``n_dates_unclassified``, ``n_dates_missing_images``). A plant's milestones are computed only
    when every one of its dates is both fully classified (``unclassified == 0``) and fully observed
    (``missing == 0``); otherwise it earns disclosure of which dates were excluded and why.
    ``positive_class_assessed`` is ``True`` iff at least one date, anywhere in the delivery, was
    fully classified.
    """
    per_plant = per_plant_series(mapping, predictions_by_date, positive_value, plants)
    rows = []
    any_classified_date = False
    for plant_id, info in per_plant.items():
        usable_dates = [(d, total, positive)
                        for (d, total, positive, unclassified, missing, _n_images) in info["series"]
                        if unclassified == 0 and missing == 0]
        n_dates_unclassified = sum(1 for s in info["series"] if s[3] > 0)
        n_dates_missing_images = sum(1 for s in info["series"] if s[4] > 0)
        if usable_dates:
            any_classified_date = True
        # total==0 detected no objects, so it's not an observation of the positive fraction
        # (pre-emergence or a detection gap), excluded from the milestone series; total>0 with
        # positive==0 is a real 0% and kept.
        frac_series = [(d, positive / total) for (d, total, positive) in usable_dates if total]
        plant_fully_classified = len(usable_dates) == len(info["series"]) and len(info["series"]) > 0
        row = {
            "plant_id": plant_id,
            "accession": info["accession"],
            "n_dates": len(info["series"]),
            "n_observed_dates": len(frac_series),
            "n_dates_unclassified": n_dates_unclassified,
            "n_dates_missing_images": n_dates_missing_images,
            "series": [
                {"date": d, "n_total": total, "n_positive": positive, "n_unclassified": unclassified,
                 "n_missing": missing, "n_images": n_images,
                 "ratio": (positive / total if total and unclassified == 0 and missing == 0 else None)}
                for (d, total, positive, unclassified, missing, n_images) in info["series"]
            ],
        }
        # A plant with any unclassified/missing date earns no milestone dates, but must still carry
        # the same keys as one that does, so both branches go through the producer (an empty series
        # crosses nothing) rather than one of them rebuilding the key set from the column names:
        # reconstructing the keys directly from column names omits every ``*_date_bound`` key, giving
        # an excluded plant's row a different shape than an included one's within a single delivery.
        row.update(plant_milestones(frac_series if plant_fully_classified else [], spec))
        rows.append(row)
    return {"rows": rows, "positive_class_assessed": any_classified_date}


def phenology_delivery_flags(
    classifier_state: str | None, operating_point_state: str | None, tile_recon: dict,
) -> dict[str, str | None]:
    """The ``check_delivery_gate`` flags dict a phenology delivery gates on, from the caller's
    reconciled classifier and count operating point states. ``tile_size`` enters only when
    ``tile_recon["operative"]`` is true; an untiled delivery's flags carry no ``tile_size`` key.
    """
    flags: dict[str, str | None] = {
        "classifier": classifier_state, "operating_point": operating_point_state,
    }
    if tile_recon["operative"]:
        flags["tile_size"] = tile_recon["validated"]
    return flags


def _operating_point_conf_cell(
    dates_delivered: Sequence[str], predictions_by_date: Mapping[str, str],
    operating_point_confs: Mapping[str, float | None],
) -> float | str | None:
    """The delivered ``operating_point_conf`` cell: the single value every delivered bucket
    records, when they all record the same one, otherwise one entry per date in ``dates_delivered``
    order, ``;``-joined, an empty entry for a bucket with no numeric conf.
    """
    values = [operating_point_confs.get(str(predictions_by_date[d])) for d in dates_delivered]
    non_none = {v for v in values if v is not None}
    if values and len(non_none) == 1 and all(v is not None for v in values):
        return next(iter(non_none))
    return ";".join("" if v is None else str(v) for v in values)


def _write_phenology_delivery(
    door: str,
    rows: list[dict],
    out_path: Path,
    spec,
    columns: list[str],
    *,
    include_majority_marker: bool,
    flags: dict[str, str | None],
    acknowledgment: Acknowledgment | None,
    basis: OperationalizationBasis | None,
    document_reconciliations: Mapping[str, Mapping],
    producer: dict,
    dimension_reconciliations: Mapping[str, Mapping],
    predictions_by_date: Mapping[str, str],
    project_root: str | Path | None,
    plant_mapping: dict,
) -> dict:
    """Gate, compose and write one phenology delivery's provenance cells, then record the delivery,
    for ``write_phenology_csv`` and ``write_phenology_curve_csv``.

    Runs ``check_delivery_gate`` over ``flags``: a gate that does not pass raises ``ValueError``
    with the gate's own reason, and nothing is written. ``acknowledgment`` is the breeder's own
    act (or ``None``) the caller already resolved. ``basis`` is what a passing
    ``check_operationalization`` returned, and it is required.

    Composes every provenance cell the schema declares (the operating-point and classifier validity
    columns, the producer tail and ``produced_at``, the delivery's own
    ``acknowledged_by``/``acknowledgment_reason``, all through ``resolution.delivered_tail``, and,
    when ``include_majority_marker`` is set, the trait's majority-alias marker through
    ``majority_crossing_unconfirmed_column``) and returns them. Records the delivery through
    ``record_delivery_binding_event`` after the file is written, under the caller-stated ``door``
    and the explicit ``project_root``, with the gate's own ``effective_acknowledgment()``. That
    write is best-effort; the returned dict carries ``delivery_event_recorded``, the write's own
    success bool, beyond the schema's own columns.

    ``plant_mapping`` is the mapping this delivery attributed detections through, shaped as
    ``delivery_events_schema.PlantMappingDisclosure`` declares
    (``MappingBuild.delivery_disclosure``); required. Its ``dates_delivered`` fills the CSV's own
    column (``";"``-joined), ``images_unattributed`` and ``plant_attribution`` fill theirs
    directly, and the whole dict travels to the delivery event unchanged.

    ``document_reconciliations`` and ``dimension_reconciliations`` are threaded straight through to
    ``record_delivery_binding_event``; the count operating point's ``bindings`` and ``confs`` are
    read from ``document_reconciliations["operating_point"]``. When ``predictions_by_date`` is
    non-empty, both ``operating_point`` and ``classifier_operating_point`` must have an entry,
    checked before the gate runs and before anything is written.

    Raises:
        ValueError: ``basis`` is not an ``OperationalizationBasis``, ``predictions_by_date`` is
            non-empty but ``document_reconciliations`` lacks an entry this writer declares, the
            gate refused, or ``flags`` carries no ``classifier`` dimension; nothing is written in
            any of these cases.
        AuditEntryNotWritten (``tcip_mcp.audit``): the dataset-scoped delivery-event audit line
            could not be appended, raised by ``record_delivery_binding_event`` after the CSV was
            already written to ``out_path``.
    """
    if not isinstance(basis, OperationalizationBasis):
        raise ValueError(
            "a phenology delivery requires the basis a passing check_operationalization returned "
            "for this trait's state_crossing_dates delivery. deliver_phenology_milestones and export_csv "
            "produce one and are the primitives to call; this writer cannot read the record itself, "
            "because it is given a trait spec rather than a project to read from."
        )
    declared_documents = ("operating_point", "classifier_operating_point")
    if predictions_by_date:
        missing = [doc for doc in declared_documents if doc not in document_reconciliations]
        if missing:
            raise ValueError(
                f"{door}: predictions_by_date names {len(predictions_by_date)} bucket(s) but "
                f"document_reconciliations carries no entry for {missing}; a phenology delivery "
                "declares both operating_point and classifier_operating_point to the event "
                "writer, so both must be reconciled before the gate runs and before anything "
                "is written."
            )
    op_recon = document_reconciliations.get("operating_point", {})
    bindings = op_recon.get("bindings", {})
    operating_point_confs = op_recon.get("confs", {})

    from tcip_mcp.operationalization import STATE_CROSSING_DATES
    from tcip_mcp.pipelines.resolution import (
        check_delivery_gate, delivered_tail, record_delivery_binding_event,
    )

    gate = check_delivery_gate(flags, acknowledgment=acknowledgment)
    if not gate.ok:
        raise ValueError(gate.reason)
    if "classifier" not in gate.stamp:
        raise ValueError(
            "a phenology delivery's flags carry no 'classifier' dimension: this writer stamps the "
            "delivered classifier-validity column from it, so flags composed some other way than "
            "phenology_delivery_flags must still include it."
        )

    cells: dict = delivered_tail(
        {"producer_model_sha256": producer.get("sha256"),
         "producing_experiment_id": producer.get("experiment_id"),
         "operating_point_conf": _operating_point_conf_cell(
             plant_mapping["dates_delivered"], predictions_by_date, operating_point_confs),
         "plant_mapping_sha256": plant_mapping["record_sha256"],
         "dates_delivered": ";".join(plant_mapping["dates_delivered"]),
         "images_unattributed": plant_mapping["images_unattributed"],
         "plant_attribution": plant_mapping["plant_attribution"]},
        bindings, gate,
        columns=tuple(PROVENANCE_COLUMNS))
    if include_majority_marker:
        crossing_unconfirmed_column = majority_crossing_unconfirmed_column(spec)
        if crossing_unconfirmed_column:
            cells[crossing_unconfirmed_column] = "true" if spec.crossing_unconfirmed else "false"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **cells})

    # Not a schema column (the CSV above is already written with extrasaction="ignore"), but a
    # caller composing its own response from these cells needs to know whether it landed.
    cells["delivery_event_recorded"] = record_delivery_binding_event(
        door, str(out_path), list(predictions_by_date.values()),
        document_reconciliations=document_reconciliations,
        dimension_reconciliations=dimension_reconciliations,
        measurement_documents=list(declared_documents),
        acknowledgment=gate.effective_acknowledgment(),
        trait=spec.name, delivery_kind=STATE_CROSSING_DATES,
        project_root=project_root, plant_mapping=plant_mapping,
    )
    return cells


def write_phenology_csv(
    door: str,
    rows: list[dict],
    out_path: Path,
    spec,
    *,
    flags: dict[str, str | None],
    acknowledgment: Acknowledgment | None,
    basis: OperationalizationBasis | None,
    document_reconciliations: Mapping[str, Mapping],
    producer: dict,
    dimension_reconciliations: Mapping[str, Mapping],
    predictions_by_date: Mapping[str, str],
    project_root: str | Path | None,
    plant_mapping: dict,
) -> dict:
    """Write per-plant milestone rows to the canonical delivery CSV, for the given trait's spec.

    Emits exactly ``phenology_csv_columns(spec)`` through ``_write_phenology_delivery`` (including
    the trait's majority crossing-unconfirmed marker when the spec names one). ``door`` is the name
    ``record_delivery_binding_event`` records the delivery under. ``predictions_by_date`` is the
    date-to-bucket mapping the delivery reads. ``plant_mapping``, ``document_reconciliations``,
    ``dimension_reconciliations`` and ``acknowledgment`` are ``_write_phenology_delivery``'s.
    """
    return _write_phenology_delivery(
        door, rows, out_path, spec, phenology_csv_columns(spec),
        include_majority_marker=True, flags=flags,
        acknowledgment=acknowledgment, basis=basis,
        document_reconciliations=document_reconciliations, producer=producer,
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project_root=project_root,
        plant_mapping=plant_mapping)


def write_phenology_curve_csv(
    door: str,
    rows: list[dict],
    out_path: Path,
    spec,
    *,
    flags: dict[str, str | None],
    acknowledgment: Acknowledgment | None,
    basis: OperationalizationBasis | None,
    document_reconciliations: Mapping[str, Mapping],
    producer: dict,
    dimension_reconciliations: Mapping[str, Mapping],
    predictions_by_date: Mapping[str, str],
    project_root: str | Path | None,
    plant_mapping: dict,
) -> dict:
    """Write per-(plant, date) curve rows to the delivery CSV, for the given trait's spec: exactly
    ``curve_csv_columns()`` through ``_write_phenology_delivery``, without the majority
    crossing-unconfirmed marker. The arguments are :func:`write_phenology_csv`'s.
    """
    return _write_phenology_delivery(
        door, rows, out_path, spec, curve_csv_columns(),
        include_majority_marker=False, flags=flags,
        acknowledgment=acknowledgment, basis=basis,
        document_reconciliations=document_reconciliations, producer=producer,
        dimension_reconciliations=dimension_reconciliations,
        predictions_by_date=predictions_by_date, project_root=project_root,
        plant_mapping=plant_mapping)

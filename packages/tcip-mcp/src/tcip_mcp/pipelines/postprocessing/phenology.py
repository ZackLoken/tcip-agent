"""Canonical phenology measurement, the one implementation of a trait's positive-fraction milestones.
Every surface (the tcip-web Results route, the MCP ``compute_phenology`` tool) routes through here,
so a phenology date always means the same thing, for whichever registered trait it's computed for.

Trait definition (authoritative; see the ``phenology`` skill + the CLAUDE.md measurement-
integrity invariant): the positive-state fraction = the fraction of a plant's detected objects
that are in the trait's positive/measured state, an expert-defined visible morphological stage
emitted by a *validated*
classifier (the trait's ``positive_class_name``), never a geometric proxy such as bbox height.
Milestone columns come entirely from the trait's own ``TraitSpec`` (``phenology_prefix`` plus
each ``milestone_fractions`` entry), so a different registered trait yields its own prefix and
columns with no code change:

    ``<prefix>_<NN>per_date``            = the date the positive fraction first crosses NN%,
                                            for each fraction the spec declares
    ``<prefix>_<majority_label>_date``   = the majority-crossing alias
                                            (``TraitSpec.majority_milestone``), present only
                                            when the spec names one

A spec's majority-crossing alias is a breeder-confirmed reading of the trait's own definition
text, flagged provisional (``TraitSpec.majority_provisional``) until confirmed; correct the
mapping in the spec if the breeder rules otherwise. ``positive_onset_date`` (the first date
any positive-state observation appears) is a separate helper, not the delivered trait.

This module is pure (stdlib only, plus ``resolution.py`` which is itself torch-free): it consumes
prediction buckets and never touches pixels or model machinery. A prediction file's ``.subject``
carries the classifier's own decoded call (see ``count_by_class``'s docstring for the full
schema-vs-GT distinction). If the bucket
never assessed the trait's positive-class axis at all, the fraction is not a valid measurement,
``per_plant_phenology`` surfaces that via per-plant/per-date disclosure fields so callers never
deliver a curve built on unclassified or missing detections.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


def _milestone_targets(spec) -> dict[str, float]:
    """Milestone crossing fractions from the trait's confirmed semantics (Tier C, read never derived).

    Resolved per call (not a module-load snapshot) so a config-authored trait or a repinned project
    is honored. Keyed "NNper" to match the CSV column names.
    """
    return {f"{int(round(f * 100)):02d}per": f for f in spec.milestone_fractions}


def _milestone_columns(spec) -> list[tuple[str, str]]:
    """``(column_suffix, crossing_key)`` for every milestone this trait actually delivers.

    The single owner of which milestone columns exist. ``plant_milestones`` iterates it to emit each
    date and its bound; the schema functions map it to names. The majority alias enters only when the
    spec names a crossing for it, and declaration and production share this same condition, so a
    trait with no ``majority_milestone`` never declares a majority date/bound/provisional column that
    no producer fills.
    """
    cols = [(key, key) for key in _milestone_targets(spec)]
    if spec.majority_milestone:
        cols.insert(0, (spec.majority_label, spec.majority_milestone))
    return cols


def milestone_date_columns(spec) -> list[str]:
    """The milestone/date column names a trait's phenology delivery carries, a proper subset of
    ``phenology_csv_columns`` (no ``plant_id``/provenance columns).

    Consumed by ``phenology_csv_columns``, which pairs each with its ``_bound``. Not an export gate:
    the web door computes what it exports directly rather than checking for these names in a
    caller-supplied table.
    """
    return [f"{spec.phenology_prefix}_{sfx}_date" for sfx, _ in _milestone_columns(spec)]


def phenology_csv_columns(spec) -> list[str]:
    """The delivered per-plant phenology CSV schema for one trait, derived from its ``TraitSpec``.

    The milestone/alias column names come from the spec (``phenology_prefix`` + each milestone key,
    plus the majority alias/provisional columns built from ``majority_label``) so the schema carries
    no trait vocabulary of its own: every registered trait resolves through its own spec to its own
    prefix and columns with no change here. The surrounding provenance columns (operating point,
    classifier validation, producer identity, coverage disclosure) are genuinely trait-neutral.
    """
    # Only when the spec names a majority crossing, the alias's provisional marker qualifies that
    # alias, so without one there is nothing for it to mark (``phenology_tools`` gates its stamp on
    # the same condition; the two must agree or ``write_phenology_csv`` raises on an unknown key).
    provisional = ([f"{spec.phenology_prefix}_{spec.majority_label}_provisional"]
                   if spec.majority_milestone else [])
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
        *provisional,
        *PROVENANCE_COLUMNS,
    ]


# How the counts behind a delivered number were produced, and whether the measurement is
# trustworthy, a delivered phenotype must carry this so it can be traced. Trait-neutral, and the
# single owner of the tail, so every delivered shape (milestones, curves) carries the same chain
# rather than each door listing its own.
PROVENANCE_COLUMNS = [
    "operating_point_conf",
    "operating_point_validated",
    "positive_state_classifier_validated",
    # Producing-model identity, the exact checkpoint (content hash) + run behind the counts.
    "producer_model_sha256",
    "producer_experiment_id",
]

# The per-(plant, date) columns ``per_plant_series`` produces, before the provenance tail.
CURVE_MEASUREMENT_COLUMNS = [
    "plant_id", "accession", "date",
    "n_images", "n_total", "n_positive", "n_unclassified", "n_missing", "ratio",
]


def curve_csv_columns() -> list[str]:
    """The delivered per-(plant, date) curve CSV schema.

    A curve is the same phenology measurement as the milestone summary, un-summarised, which is why it
    takes the identical delivery gate, so it carries the identical provenance tail. Trait-neutral:
    unlike the milestone schema it names no crossings, only the counts the fraction is built from.
    """
    return [*CURVE_MEASUREMENT_COLUMNS, *PROVENANCE_COLUMNS]


# ── ISO date helpers ─────────────────────────────────────────────────────


def date_key(date_str: str) -> tuple[int, int, int]:
    """ISO ``YYYY-MM-DD`` → ``(year, month, day)`` for chronological sort.

    A value that is not a calendar-legal ISO date (the ``undated/`` bucket, a non-numeric
    folder, or an out-of-range one like ``2026-13-01``) sorts first as ``(0, 0, 0)`` and is
    excluded from milestone math, an image with no valid capture date can't sit on a time
    series. Validating the *whole* date here (not just "three integers") keeps ``date_key``,
    ``crossing_date`` (which builds ``datetime.date`` objects to interpolate) and
    ``positive_onset_date`` agreeing on exactly which points are real, and prevents a
    malformed folder from raising mid-interpolation.
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

    Linear interpolation between neighbouring observed dates when the crossing falls between two
    points; a left-censored crossing (the first observed point already meets the target) or a
    right-censored one (the last observed point still hasn't) is flagged as such rather than
    silently returned as if it were a real single-date crossing, or as a bare ``None`` indistinguishable
    from no observations at all. ``None`` only when there are no real observed points to begin with.
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
    """The phenology dates for one plant's positive-fraction series, keyed by the trait's own columns.

    Both the column names (``phenology_prefix`` + each milestone key, plus the majority alias) and the
    crossing fractions and majority mapping come from the trait's semantics (``TraitSpec``), so the
    milestone definition lives in one place instead of scattered literals. ``spec`` is required: no
    silent default trait, a caller that forgets to thread the trait spec must fail loudly, not emit
    one trait's columns for another.
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

    The single resolution both delivery doors' ``positive_class_id`` surfaces call, not a separate
    registry re-derivation, which could disagree with the map predictions were actually decoded
    through. Returns ``(class_id, message)``; ``class_id`` is ``None`` when no bucket's ``id_map``
    contains the trait's positive value, so the caller refuses rather than guessing with a bare
    default.
    """
    name = spec.positive_class_name
    if not name:
        return None, f"trait {spec.name!r} defines no positive_class_name"
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
    """The bucket's recorded ``id_map`` (name -> int), or ``None`` if absent/malformed.

    Read from ``operating_point.json``, the same sidecar every prediction-bucket writer
    (``export_predictions``, the GUI worker) stamps ``id_map`` into, never re-derived. A non-dict
    ``id_map`` (a malformed/foreign sidecar) is treated the same as absent, fail closed, never
    duck-typed.
    """
    from tcip_mcp.pipelines.resolution import read_operating_point_sidecar

    sidecar = read_operating_point_sidecar(pred_dir)
    if not isinstance(sidecar, dict):
        return None
    id_map = sidecar.get("id_map")
    return id_map if isinstance(id_map, dict) else None


def count_by_class(json_path: Path, id_map: dict | None, positive_value: str) -> tuple[int, int, int]:
    """``(n_total, n_positive, n_unclassified)`` for one image's predictions.

    Two checks, in order:

    1. Bucket-level precondition: ``id_map`` must be a dict containing ``positive_value`` as a
       key, the run that produced this bucket must have actually classified along the trait's own
       axis. A bare single-class detector's ``id_map`` (e.g. ``{"<object>": 0}``, no attribute axis)
       fails this, correctly, since the detector never assessed the trait's positive state at all.
       If it fails, every detection in this image is unclassified, a whole-bucket decision, not a
       per-detection one.
    2. Per-detection check: within a bucket that passes (1), a detection is positive if
       ``.subject == positive_value``, classified-negative if ``.subject`` is some other key of
       ``id_map``, and unclassified if ``.subject`` is not a key of ``id_map`` at all, a stale file
       left behind by an earlier run, or a raw-index fallback (``export.py``'s
       ``id_to_name.get(cid, str(cid))``) from a checkpoint whose class vocabulary shrank after
       training. Never silently coerced into a classified negative.

    Predictions decode differently from GT annotations, verified against the real writer,
    ``write_predictions_json``, which puts the classifier's decoded call straight into
    ``Annotation.subject`` and leaves ``.attributes`` empty, always, for every prediction-bucket
    writer this reads from (``json_io.target_class_id``, built for GT's ``subject``+``attributes``
    shape, is not reused here, applying it to a prediction file returns ``None``/raises for either
    bucket shape, unusable either way).

    A missing prediction file is the caller's concern (a missing observation, not an observed zero,
    see ``per_plant_series``), not this function's: it is only ever called for a file confirmed to
    exist.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Point

    # A Point carries no detection: counting one would inflate the denominator of the positive
    # fraction this phenology curve is built from, with an object the detector never found.
    annotations = [a for a in json_io.read_annotations(str(json_path))
                   if a.geometry is not None and not isinstance(a.geometry, Point)]
    total = len(annotations)
    if not id_map or positive_value not in id_map:
        return total, 0, total
    positive = 0
    unclassified = 0
    for a in annotations:
        if a.subject not in id_map:
            unclassified += 1
        elif a.subject == positive_value:
            positive += 1
    return total, positive, unclassified


def per_plant_series(
    mapping: dict[str, list],
    predictions_by_date: dict[str, str],
    positive_class_name: str,
) -> dict[str, dict]:
    """Aggregate classified predictions into a per-plant positive-fraction series.

    ``mapping`` is ``{date: [assignment, ...]}`` where each assignment has ``.stem`` /
    ``.plot_name`` / ``.accession_name`` (attributes or dict keys). Returns
    ``{plant_id: {accession, series: [(date, total, positive, unclassified, missing), ...]}}``.

    Coverage is measured against the stems the plant mapping actually names for each (plant, date),
    not merely against whatever prediction files happen to exist: a named stem with no
    corresponding prediction file is a missing observation, disclosed separately, never read as an
    observed zero. This applies at the date level too: a date the
    mapping names for which the caller simply omits a ``predictions_by_date`` entry is not skipped,
    every stem the mapping names for it counts as missing, the same as a named stem with no file,
    rather than the date vanishing from the series with no disclosure at all.
    """
    def _attr(a, name):
        return getattr(a, name, None) if not isinstance(a, dict) else a.get(name)

    per_plant: dict[str, dict] = {}
    # Iterate the mapping's own dates, not predictions_by_date's, the mapping is the coverage
    # reference, so a date it names is never silently absent just because the caller dropped it.
    for date_str in mapping:
        pred_dir = predictions_by_date.get(date_str)
        pred_path = Path(pred_dir) if pred_dir else None
        id_map = bucket_id_map(pred_path) if pred_path is not None else None
        # [total, positive, unclassified, missing, n_images] per plant, accumulated across that
        # plant's images on this date. ``n_images`` counts every image the mapping names for this
        # (plant, date), including ones with no prediction file, which ``missing`` counts too,
        # since it is the coverage the series entry summarises, not the files that happened to exist.
        by_plant: dict[str, list[int]] = {}
        accession: dict[str, Optional[str]] = {}
        for a in mapping[date_str]:
            plant_id = _attr(a, "plot_name")
            if not plant_id:
                continue
            acc = by_plant.setdefault(plant_id, [0, 0, 0, 0, 0])
            acc[4] += 1
            accession.setdefault(plant_id, _attr(a, "accession_name"))
            img_path = pred_path / f"{_attr(a, 'stem')}.json" if pred_path is not None else None
            if img_path is None or not img_path.is_file():
                acc[3] += 1
                continue
            total, positive, unclassified = count_by_class(img_path, id_map, positive_class_name)
            acc[0] += total
            acc[1] += positive
            acc[2] += unclassified
        for plant_id, (total, positive, unclassified, missing, n_images) in by_plant.items():
            entry = per_plant.setdefault(plant_id, {"accession": accession.get(plant_id), "series": []})
            entry["series"].append((date_str, total, positive, unclassified, missing, n_images))
    return per_plant


def per_plant_phenology(
    mapping: dict[str, list],
    predictions_by_date: dict[str, str],
    positive_class_name: str,
    spec,
) -> dict:
    """Full canonical pipeline: classified predictions + plant mapping → per-plant milestones.

    Returns ``{rows: [...], positive_class_assessed: bool}``. Each row carries the positive-fraction
    series, the milestone dates, and coverage-disclosure fields (``n_dates_unclassified``,
    ``n_dates_missing_images``). A plant's milestones are computed only over dates that are both
    fully classified (``unclassified == 0``) and fully observed (``missing == 0``) for that date,
    conjunctive across dates, not an "any date" union: a plant with even one
    partially-unclassified or partially-missing date does not earn milestone dates for that plant,
    it earns disclosure of which dates were excluded and why. The top-level ``positive_class_assessed``
    is ``True`` iff at least one date, anywhere in the delivery, was fully classified, distinguishing
    "the classifier bridge was never wired at all" (nothing here, refuse the whole call) from "wired,
    with some per-plant/per-date gaps" (deliver, with per-row disclosure).
    """
    per_plant = per_plant_series(mapping, predictions_by_date, positive_class_name)
    rows = []
    any_classified_date = False
    for plant_id, info in sorted(per_plant.items()):
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


def write_phenology_csv(rows: list[dict], out_path: Path, spec, stamp: dict | None = None) -> str:
    """Write per-plant milestone rows to the canonical delivery CSV, for the given trait's spec.

    Emits exactly ``phenology_csv_columns(spec)``, a stamp key absent from that spec-derived set
    raises rather than being silently dropped: ``extrasaction="ignore"`` alone would silently
    discard a stray/mistyped provenance key with no signal. ``stamp`` (the
    operating point + validation status) is written into every row so the phenotype is traceable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = stamp or {}
    columns = phenology_csv_columns(spec)
    unknown = set(stamp) - set(columns)
    if unknown:
        raise ValueError(f"stamp key(s) {sorted(unknown)} not in this trait's column set {columns}")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **stamp})
    return str(out_path)

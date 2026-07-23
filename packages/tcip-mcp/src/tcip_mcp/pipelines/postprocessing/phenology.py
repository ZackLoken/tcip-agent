"""Canonical catkin bloom phenology — the ONE implementation of the elongated-fraction
milestones. Every surface (the tcip-web Results route, the MCP ``compute_phenology`` tool)
routes through here, so a phenology date always means the same thing.

Trait definition (authoritative; see the ``phenology`` skill + the CLAUDE.md measurement-
integrity invariant): **bloom = the fraction of a plant's detected catkins that are
elongated.** "Elongated" is an expert-defined *visible morphological stage* emitted by a
validated 2-class classifier (class ``elongated_class_id``) — never a geometric proxy such
as bbox height/aspect. Milestones:

    catkin_05/50/95per_date  = the dates the elongated fraction first crosses 5/50/95%
    catkin_elongation_date   = the date most catkins have elongated (``crops.yml``) — the 95%
                               majority crossing, i.e. synonymous with catkin_95per_date

``catkin_elongation_date`` follows ``crops.yml`` ("Date when most catkins have elongated"),
operationalized as the majority crossing the trait spec names (``TraitSpec.majority_milestone``
= "95per"). This is the current best-guess reading of that text, flagged provisional
(``TraitSpec.majority_provisional``) pending breeder confirmation — correct the mapping in the
spec if they rule otherwise. ``elongation_onset_date`` (the first date any elongation appears)
is a separate helper, not the delivered trait.

This module is pure (stdlib only): it consumes an already-classified per-(plant, date)
count and never touches pixels. Because "elongated" is a classifier class, if the
predictions carry no elongation class the fraction is not a valid bloom measurement — the
functions surface that via ``elongation_classified`` so callers never deliver a curve built
on unclassified detections.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from tcip_mcp.traits import get_trait


def _milestone_targets(spec) -> dict[str, float]:
    """Milestone crossing fractions from the trait's confirmed semantics (Tier C, read never derived).

    Resolved per call (not a module-load snapshot) so a config-authored trait or a repinned project
    is honored. Keyed "NNper" to match the CSV column names.
    """
    return {f"{int(round(f * 100)):02d}per": f for f in spec.milestone_fractions}


def phenology_csv_columns(spec) -> list[str]:
    """The delivered per-plant phenology CSV schema for one trait, derived from its ``TraitSpec``.

    The milestone/alias column NAMES come from the spec (``phenology_prefix`` + each milestone key,
    plus the majority alias/provisional columns built from ``majority_label``) so the schema carries
    no trait vocabulary of its own — catkin resolves to ``catkin_05per_date`` … ``catkin_elongation_date``,
    while a different trait's spec yields its own prefix with no change here. The surrounding
    provenance columns (operating point, classifier validation, producer identity) are trait-neutral.
    """
    prefix = spec.phenology_prefix
    majority = f"{prefix}_{spec.majority_label}_date"
    # Whether the majority-date reading (which milestone "most elongated" maps to) is still provisional
    # pending breeder confirmation — a read-semantics marker carried with the delivery, not settled fact.
    provisional = f"{prefix}_{spec.majority_label}_provisional"
    fraction_cols = [f"{prefix}_{key}_date" for key in _milestone_targets(spec)]
    return [
        "plant_id",
        "accession",
        "n_dates",
        majority,
        provisional,
        *fraction_cols,
        # Provenance stamp: how the counts behind these milestones were produced, and whether the
        # measurement is trustworthy. A delivered phenotype must carry this so it can be traced.
        "operating_point_conf",
        "operating_point_validated",
        "elongation_classifier_validated",
        # Producing-model identity — the exact checkpoint (content hash) + run behind the counts.
        "producer_model_sha256",
        "producer_experiment_id",
    ]


# Catkin's delivered CSV schema (Phase 1) — one canonical column set/order so the
# ``compute_phenology`` MCP tool and any other exporter emit the same file. Another trait resolves
# its own schema through ``phenology_csv_columns`` from that trait's spec.
PHENOLOGY_CSV_COLUMNS = phenology_csv_columns(get_trait("catkin"))


# ── ISO date helpers ─────────────────────────────────────────────────────


def date_key(date_str: str) -> tuple[int, int, int]:
    """ISO ``YYYY-MM-DD`` → ``(year, month, day)`` for chronological sort.

    A value that is not a calendar-legal ISO date (the ``undated/`` bucket, a non-numeric
    folder, or an out-of-range one like ``2026-13-01``) sorts first as ``(0, 0, 0)`` and is
    excluded from milestone math — an image with no valid capture date can't sit on a time
    series. Validating the *whole* date here (not just "three integers") keeps ``date_key``,
    ``crossing_date`` (which builds ``datetime.date`` objects to interpolate) and
    ``elongation_onset_date`` agreeing on exactly which points are real, and prevents a
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


def crossing_date(series: list[tuple[str, float]], target: float) -> Optional[str]:
    """Earliest date the fraction curve reaches ``>= target`` (linear interpolation between
    neighbouring dates). ``None`` if it never reaches the target."""
    points = _real_points(series)
    if not points:
        return None
    if points[0][1] >= target:
        return iso(points[0][0])
    for (d1, r1), (d2, r2) in zip(points, points[1:]):
        if r2 >= target:
            if r2 == r1:
                return iso(d2)
            y1, m1, day1 = date_key(d1)
            y2, m2, day2 = date_key(d2)
            t = max(0.0, min(1.0, (target - r1) / (r2 - r1)))
            est = date(y1, m1, day1) + timedelta(days=round(t * (date(y2, m2, day2) - date(y1, m1, day1)).days))
            return est.isoformat()
    return None


def elongation_onset_date(series: list[tuple[str, float]]) -> Optional[str]:
    """First date any elongation appears (fraction > 0), chronologically. ``None`` if never."""
    for d, r in _real_points(series):
        if r > 0:
            return iso(d)
    return None


def plant_milestones(series: list[tuple[str, float]], spec=None) -> dict:
    """The phenology dates for one plant's positive-fraction series, keyed by the trait's own columns.

    Both the column names (``phenology_prefix`` + each milestone key, plus the majority alias) and the
    crossing fractions and "most elongated" majority mapping come from the trait's semantics
    (``TraitSpec``), so the milestone definition lives in one place instead of scattered literals.
    ``spec`` defaults to the Phase-1 catkin trait; passing another trait's spec yields its own columns.
    """
    spec = spec or get_trait("catkin")
    prefix = spec.phenology_prefix
    out = {f"{prefix}_{key}_date": crossing_date(series, frac)
           for key, frac in _milestone_targets(spec).items()}
    # e.g. catkin's crops.yml "most catkins elongated" maps to the majority milestone the spec names
    # (95% crossing; a provisional reading pending breeder confirmation — see spec.majority_provisional).
    if spec.majority_milestone:
        out[f"{prefix}_{spec.majority_label}_date"] = out.get(f"{prefix}_{spec.majority_milestone}_date")
    return out


# ── elongated-fraction from classified predictions ───────────────────────


def count_by_class(json_path: Path, elongated_class_id: int) -> tuple[int, int, set[int]]:
    """``(total_detections, n_elongated, classes_seen)`` from a name-based prediction file.

    Deferred to K4/K5: elongation is now an *attribute* of a name-based prediction, not an integer
    class id, and the attribute→fraction bridge is not wired in this slice. Until it is, this reports
    the raw detection total and no elongation split — the ``elongated_class_id`` int is the seam K4/K5
    replaces — so nothing here fabricates a phenology fraction from unwired predictions. A missing
    file is unannotated → ``(0, 0, set())``.
    """
    from tcip_annotation import json_io

    total = sum(1 for a in json_io.read_annotations(str(json_path)) if a.geometry is not None)
    return total, 0, set()


def per_plant_series(
    mapping: dict[str, list],
    predictions_by_date: dict[str, str],
    elongated_class_id: int,
) -> tuple[dict[str, dict], set[int]]:
    """Aggregate classified predictions into a per-plant elongated-fraction series.

    ``mapping`` is ``{date: [assignment, ...]}`` where each assignment has ``.stem`` /
    ``.plot_name`` / ``.accession_name`` (attributes or dict keys). Returns
    ``({plant_id: {accession, series: [(date, total, elongated), ...]}}, classes_seen)``.
    """
    def _attr(a, name):
        return getattr(a, name, None) if not isinstance(a, dict) else a.get(name)

    per_plant: dict[str, dict] = {}
    all_classes: set[int] = set()
    for date_str, pred_dir in predictions_by_date.items():
        if date_str not in mapping:
            continue
        pred_path = Path(pred_dir)
        # accumulate per (plant, date) across that plant's images on the date
        by_plant: dict[str, list[int]] = {}
        accession: dict[str, Optional[str]] = {}
        for a in mapping[date_str]:
            plant_id = _attr(a, "plot_name")
            if not plant_id:
                continue
            total, elongated, classes_seen = count_by_class(
                pred_path / f"{_attr(a, 'stem')}.json", elongated_class_id
            )
            all_classes |= classes_seen
            acc = by_plant.setdefault(plant_id, [0, 0])
            acc[0] += total
            acc[1] += elongated
            accession.setdefault(plant_id, _attr(a, "accession_name"))
        for plant_id, (total, elongated) in by_plant.items():
            entry = per_plant.setdefault(plant_id, {"accession": accession.get(plant_id), "series": []})
            entry["series"].append((date_str, total, elongated))
    return per_plant, all_classes


def per_plant_phenology(
    mapping: dict[str, list],
    predictions_by_date: dict[str, str],
    elongated_class_id: int,
) -> dict:
    """Full canonical pipeline: classified predictions + plant mapping → per-plant milestones.

    Returns ``{rows: [...], elongation_classified: bool, classes_seen: [...]}``. Each row has
    the elongated-fraction series and the four milestone dates. When
    ``elongation_classified`` is false the predictions carried no elongation class, so the
    milestones are not a valid bloom measurement (do not deliver them).
    """
    per_plant, all_classes = per_plant_series(mapping, predictions_by_date, elongated_class_id)
    rows = []
    for plant_id, info in sorted(per_plant.items()):
        # total==0 detected no catkins, so it's not an observation of the elongated fraction
        # (pre-emergence or a detection gap) — excluded from the milestone series; total>0 with
        # elongated==0 is a real 0% and kept. The raw series keeps it with ratio=None.
        frac_series = [(d, elong / total) for (d, total, elong) in info["series"] if total]
        row = {
            "plant_id": plant_id,
            "accession": info["accession"],
            "n_dates": len(info["series"]),
            "n_observed_dates": len(frac_series),
            "series": [
                {"date": d, "n_total": total, "n_elongated": elong,
                 "ratio": (elong / total if total else None)}
                for (d, total, elong) in info["series"]
            ],
            **plant_milestones(frac_series),
        }
        rows.append(row)
    return {
        "rows": rows,
        "classes_seen": sorted(all_classes),
        "elongation_classified": elongated_class_id in all_classes,
    }


def write_phenology_csv(rows: list[dict], out_path: Path, stamp: dict | None = None) -> str:
    """Write per-plant milestone rows to the canonical delivery CSV.

    Emits exactly ``PHENOLOGY_CSV_COLUMNS`` (extra keys such as the raw ``series`` are
    dropped) so every delivered ``catkin_phenology.csv`` has the same shape. ``stamp`` (the
    operating point + validation status) is written into every row so the phenotype is traceable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = stamp or {}
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHENOLOGY_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **stamp})
    return str(out_path)

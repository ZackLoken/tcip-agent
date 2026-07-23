"""One-time migration: the pre-K13.5 on-disk layout -> the nested-schema layout.

Run ONCE, on a COPY of a real project, after the 2c code lands. This is a migration script, not a
runtime shim — nothing in the platform ever falls back to the old layout. Idempotent: it refuses a
dataset that already has ``classes.json``.

**Two-phase, so a refusal never strands a half-converted dataset:** everything that can refuse
(registry location, every label's category_id, the status-store format, each negative's
(subject,date)) is validated and built *in memory* first; only after every check passes does the
script write anything (classes.json, then labels, then the re-keyed store). A refusal therefore
leaves the dataset exactly as it was, so the ``classes.json`` idempotency guard stays truthful and a
corrected re-run works.

Old layout it reads:
  <root>/.tcip/state/classes/<subject>.json  or  <root>/classes/<subject>.json   flat {cid:{name,color}}
  <root>/annotations/<subject>/<date>/{detect,segment}/<stem>.json   {objects:[{category_id, bbox|segmentation, prov}]}
  <root>/.tcip/state/image_status.json         flat {image_name: "partial"|"negative"|...}

New layout it writes:
  <root>/classes.json                          nested subjects -> attributes -> values (this data: plain subjects, no attributes)
  <root>/annotations/<date>/<stem>.json        one file per image, all subjects, name-based
  <root>/.tcip/state/image_status.json         {status_bucket(subject,date): {image_name: status}}

Per (date, stem) the segment polygons win (the detect boxes were derived from them), so an instance is
one annotation, not two. Predictions are regenerated via inference, not migrated.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox, Polygon
from tcip_mcp import class_registry
from tcip_mcp.class_registry import ClassRegistry, Subject
from tcip_mcp.dataset_layout import normalize_status_store, status_bucket

_PROV_KEYS = ("created_by", "created_at", "accepted_by", "accepted_at")
#: Statuses that are human confirmations (cannot be recomputed from labels); the rest
#: (partial/unannotated) are re-derived by ``derive_image_status`` and are not carried.
_CARRY_STATUSES = {"negative", "complete"}


@dataclass
class _Plan:
    """The fully-validated conversion, built in memory before any write."""

    registry: ClassRegistry
    labels: dict[tuple[str, str], tuple[int, int, list[Annotation]]]  # (date, stem) -> (w, h, anns)
    negatives: dict[str, dict[str, str]]  # status_bucket -> {image: status}; empty if none to carry
    subjects: list[str]


def _read_old_registry(root: Path) -> dict[str, list[str]]:
    """``{subject: [class names in cid order]}`` from the old per-subject registry.

    Accepts either old location: the pre-slice-1 ``<root>/.tcip/state/classes/`` or the slice-1
    ``<root>/classes/`` (per-subject files). Refuses if neither holds a registry.
    """
    for d in (root / ".tcip" / "state" / "classes", root / "classes"):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.json"))
        if not files:
            continue
        out: dict[str, list[str]] = {}
        for f in files:
            raw = json.loads(f.read_text(encoding="utf-8"))
            by_cid = {int(k): (v.get("name") or f"class_{k}") for k, v in raw.items()}
            out[f.stem] = [by_cid[c] for c in sorted(by_cid)]
        return out
    raise SystemExit(
        f"no old per-subject registry at {root / '.tcip' / 'state' / 'classes'} or {root / 'classes'}")


def _build_registry(old: dict[str, list[str]], descriptions: dict[str, str]) -> ClassRegistry:
    """One :class:`Subject` per old subject.  A single-class subject (this data) is a plain detected
    subject with no attributes; a multi-class subject needs an explicit owner attribute mapping we do
    not infer (attribute type/rank cannot be guessed) — refuse it rather than fabricate one."""
    subjects: list[Subject] = []
    for subject, names in old.items():
        if len(names) > 1:
            raise SystemExit(
                f"subject {subject!r} has {len(names)} classes {names}; a multi-class subject needs an "
                "explicit attribute mapping (name, type, ordered values) — re-run with that mapping "
                "rather than let the converter guess the attribute type/rank")
        subjects.append(Subject(name=subject, description=descriptions.get(subject, "")))
    return ClassRegistry(subjects=tuple(subjects))


def _old_annotations(path: Path, subject: str) -> tuple[int, int, list[Annotation]]:
    """Read one old per-(subject,task) file -> (width, height, [Annotation]). Refuses a category_id
    it cannot map (a single-class subject with a nonzero id would be misattributed)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    w, h = int(d.get("width", 0) or 0), int(d.get("height", 0) or 0)
    anns: list[Annotation] = []
    for o in d.get("objects") or []:
        if not isinstance(o, dict):
            continue
        cid = o.get("category_id", 0)
        if cid not in (0, "0"):
            raise SystemExit(
                f"{path}: category_id {cid!r} != 0 on single-class subject {subject!r} — supply an "
                "attribute mapping and re-run rather than let the converter misattribute it")
        geom: BBox | Polygon | None = None
        seg = o.get("segmentation")
        if isinstance(seg, list) and seg and isinstance(seg[0], list) and len(seg[0]) >= 6:
            ring = seg[0]
            geom = Polygon([(float(ring[i]), float(ring[i + 1])) for i in range(0, len(ring) - 1, 2)])
        elif isinstance(o.get("bbox"), list) and len(o["bbox"]) == 4:
            x, y, bw, bh = (float(v) for v in o["bbox"])
            geom = BBox(x, y, x + bw, y + bh)
        else:
            continue
        prov = {k: o[k] for k in _PROV_KEYS if o.get(k) is not None}
        anns.append(Annotation(subject=subject, geometry=geom, attributes={}, **prov))
    return w, h, anns


def _plan_labels(root: Path, subjects: list[str]) -> dict[tuple[str, str], tuple[int, int, list[Annotation]]]:
    """Read+validate every old label into the merged per-(date,stem) form (no writes).

    Per (subject, date, stem) the segment file wins over its derived detect box (its polygons are the
    source of truth) *when it yields at least one annotation*; a segment file that parses to nothing
    (all-degenerate/empty) falls back to the detect file so a real instance is never dropped.
    """
    merged: dict[tuple[str, str], list[Annotation]] = defaultdict(list)
    dims: dict[tuple[str, str], tuple[int, int]] = {}
    for subject in subjects:
        sdir = root / "annotations" / subject
        if not sdir.is_dir():
            continue
        for date_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            date = date_dir.name
            seg_dir, det_dir = date_dir / "segment", date_dir / "detect"
            stems = {p.stem for d in (seg_dir, det_dir) if d.is_dir() for p in d.glob("*.json")}
            for stem in sorted(stems):
                anns: list[Annotation] = []
                w = h = 0
                if (seg_dir / f"{stem}.json").is_file():
                    w, h, anns = _old_annotations(seg_dir / f"{stem}.json", subject)
                if not anns and (det_dir / f"{stem}.json").is_file():  # seg empty -> fall back to detect
                    w, h, anns = _old_annotations(det_dir / f"{stem}.json", subject)
                merged[(date, stem)].extend(anns)
                dims.setdefault((date, stem), (w, h))
    return {k: (dims[k][0], dims[k][1], v) for k, v in merged.items()}


def _image_dates(root: Path) -> dict[str, str]:
    """image file name -> its capture date, from ``images/<date>/``."""
    out: dict[str, str] = {}
    idir = root / "images"
    if idir.is_dir():
        for date_dir in idir.iterdir():
            if date_dir.is_dir():
                for img in date_dir.iterdir():
                    if img.is_file():
                        out[img.name] = date_dir.name
    return out


def _plan_negatives(root: Path, negative_subject: str | None) -> dict[str, dict[str, str]]:
    """Validate+build the re-keyed ``{status_bucket(subject,date): {image: status}}`` store (no writes).

    The old flat ``{image: status}`` store lost which subject a negative was confirmed *for* — the
    ambiguity the subject-scoped store fixes — so the subject cannot be recovered from the data and
    must be supplied. A store already in the bucketed (nested) shape is left as-is (nothing to do).
    """
    store = root / ".tcip" / "state" / "image_status.json"
    if not store.is_file():
        return {}
    raw = json.loads(store.read_text(encoding="utf-8"))
    # Already bucketed (values are dicts): normalize_status_store returns it non-empty and unchanged
    # in shape — the platform already wrote it subject-scoped, so there is nothing to migrate.
    if isinstance(raw, dict) and any(isinstance(v, dict) for v in raw.values()):
        if normalize_status_store(raw):
            return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{store}: unreadable image_status.json (expected an object)")
    confirmations = {img: st for img, st in raw.items() if isinstance(st, str) and st in _CARRY_STATUSES}
    if not confirmations:
        return {}
    if not negative_subject:
        raise SystemExit(
            f"{len(confirmations)} confirmed statuses ({sorted(set(confirmations.values()))}) but the "
            "old flat store does not record which subject they were confirmed for — re-run with "
            "--negative-subject <subject>")
    dates = _image_dates(root)
    bucketed: dict[str, dict[str, str]] = defaultdict(dict)
    for img, st in confirmations.items():
        date = dates.get(img)
        if date is None:
            raise SystemExit(f"cannot resolve a capture date for confirmed image {img!r} (not under images/<date>/)")
        bucketed[status_bucket(negative_subject, date)][img] = st
    return dict(bucketed)


def plan_conversion(root: Path, descriptions: dict[str, str], negative_subject: str | None) -> _Plan:
    """Validate everything that can refuse and build the whole conversion in memory. No writes."""
    if (root / "classes.json").exists():
        raise SystemExit(f"{root / 'classes.json'} already exists — this dataset looks already converted")
    old = _read_old_registry(root)
    registry = _build_registry(old, descriptions)
    labels = _plan_labels(root, list(old))
    negatives = _plan_negatives(root, negative_subject)
    return _Plan(registry=registry, labels=labels, negatives=negatives, subjects=list(old))


def apply_conversion(root: Path, plan: _Plan) -> None:
    """Perform the writes of a validated plan (registry, labels, re-keyed store)."""
    class_registry.write_registry(root / "classes.json", plan.registry)
    for (date, stem), (w, h, anns) in sorted(plan.labels.items()):
        json_io.write_annotations(root / "annotations" / date / f"{stem}.json", anns, w, h,
                                  keep_empty=bool(anns))
    if plan.negatives:
        (root / ".tcip" / "state" / "image_status.json").write_text(
            json.dumps(plan.negatives, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="dataset/project root (holds images/ and annotations/)")
    ap.add_argument("--description", action="append", default=[], metavar="subject=text",
                    help="human description for a subject (repeatable)")
    ap.add_argument("--negative-subject", default=None,
                    help="which subject the old flat confirmations were made for (required if any exist)")
    ap.add_argument("--dry-run", action="store_true", help="validate + report only; write nothing")
    args = ap.parse_args()

    root: Path = args.root
    descriptions = dict(d.split("=", 1) for d in args.description if "=" in d)
    plan = plan_conversion(root, descriptions, args.negative_subject)  # raises before any write
    n_neg = sum(len(v) for v in plan.negatives.values())
    if not args.dry_run:
        apply_conversion(root, plan)
    print(f"{'[dry-run] ' if args.dry_run else ''}converted {len(plan.subjects)} subjects "
          f"({', '.join(plan.subjects)}), {len(plan.labels)} images, {n_neg} carried confirmations")


if __name__ == "__main__":
    main()

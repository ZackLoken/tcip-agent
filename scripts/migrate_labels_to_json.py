"""One-time migration of legacy YOLO .txt labels/predictions to per-image JSON.

Walks ``annotations/**/<task>/*.txt`` and ``predictions/**/<task>/*.txt`` under a dataset
root (skipping ``.original/`` dirs), denormalizes each via the image's dimensions, backfills
provenance (created_by/created_at), and writes ``<stem>.json`` next to the source via
``tcip_annotation.json_io``. Non-destructive by default (the .txt stays); idempotent —
re-running rewrites the same JSON, and a missing source is a no-op.

Negative invariant preserved: a 0-byte .txt is a confirmed negative and becomes a present
``{"objects": []}`` JSON (``keep_empty=True``); a missing label stays missing.

Usage: python scripts/migrate_labels_to_json.py <dataset_root> [--dry-run] [--remove-source]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tcip_annotation.json_io import write_detect, write_segment
from tcip_annotation.label_io import (
    parse_detect_labels,
    parse_detect_predictions,
    parse_segment_labels,
    parse_segment_predictions,
)
from tcip_annotation.utils import get_image_dimensions

TASKS = ("detect", "segment")
NON_IMAGE_SUFFIXES = {".txt", ".json"}


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def _find_image(dataset_root: Path, middle: tuple[str, ...], stem: str) -> Path | None:
    """Image for a label: images/<date...>/<stem>.* first, then the flat images/<stem>.*."""
    images = dataset_root / "images"
    candidates_dirs: list[Path] = []
    if middle:
        candidates_dirs.append(images.joinpath(*middle))
        if len(middle) > 1:
            candidates_dirs.append(images / middle[-1])
    candidates_dirs.append(images)
    seen: set[Path] = set()
    for d in candidates_dirs:
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        for cand in sorted(d.glob(f"{stem}.*")):
            if cand.is_file() and cand.suffix.lower() not in NON_IMAGE_SUFFIXES:
                return cand
    return None


def _created_by(tree: str, bucket: str, middle: tuple[str, ...], task: str) -> str:
    if tree == "predictions":
        return bucket or "unknown"
    if bucket == "catkin" and "2026-02-11" in middle:
        return "user:zack"
    if bucket == "bush" and task == "segment":
        return "user:emily"
    return "unknown"


def migrate(dataset_root: str | Path, *, dry_run: bool = False, remove_source: bool = False) -> dict:
    """Convert every legacy .txt under annotations/ and predictions/ to per-image JSON.

    Returns a summary dict: ``converted`` (Counter keyed by (tree, task)), ``negatives``,
    ``skipped_missing_image``, ``skipped_unparsed``.
    """
    root = Path(dataset_root)
    summary: dict = {
        "converted": Counter(),
        "negatives": 0,
        "skipped_missing_image": 0,
        "skipped_unparsed": 0,
    }
    for tree in ("annotations", "predictions"):
        base = root / tree
        if not base.is_dir():
            continue
        for txt in sorted(base.rglob("*.txt")):
            rel = txt.relative_to(base)
            if ".original" in rel.parts:
                continue
            task = txt.parent.name
            if task not in TASKS:
                continue
            dirs = rel.parts[:-1]  # e.g. ("catkin", "2026-02-11", "detect")
            bucket = dirs[0] if len(dirs) > 1 else ""
            middle = dirs[1:-1]  # date component(s), empty in a flat layout
            image = _find_image(root, middle, txt.stem)
            if image is None:
                _warn(f"no image found for {txt} — skipped")
                summary["skipped_missing_image"] += 1
                continue
            try:
                img_w, img_h = get_image_dimensions(str(image))
            except Exception as e:
                _warn(f"could not read dimensions of {image} ({e}) — skipped {txt}")
                summary["skipped_missing_image"] += 1
                continue

            is_pred = tree == "predictions"
            if task == "detect":
                parser = parse_detect_predictions if is_pred else parse_detect_labels
            else:
                parser = parse_segment_predictions if is_pred else parse_segment_labels
            shapes, _ = parser(str(txt), img_w, img_h)

            is_negative = txt.stat().st_size == 0
            if not shapes and not is_negative:
                _warn(f"{txt} is non-empty but parsed to no shapes — skipped")
                summary["skipped_unparsed"] += 1
                continue

            created_by = _created_by(tree, bucket, middle, task)
            created_at = datetime.fromtimestamp(
                txt.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            for s in shapes:
                s.created_by = created_by
                s.created_at = created_at

            if not dry_run:
                writer = write_detect if task == "detect" else write_segment
                writer(txt.with_suffix(".json"), shapes, img_w, img_h, keep_empty=not shapes)
                if remove_source:
                    txt.unlink()
            summary["converted"][(tree, task)] += 1
            if is_negative:
                summary["negatives"] += 1
    return summary


def _print_summary(summary: dict, dry_run: bool) -> None:
    verb = "would convert" if dry_run else "converted"
    if not summary["converted"]:
        print(f"{verb}: nothing (no legacy .txt labels found)")
    for (tree, task), n in sorted(summary["converted"].items()):
        print(f"{verb} {tree}/{task}: {n}")
    print(f"negatives (kept as objects: []): {summary['negatives']}")
    print(f"skipped (missing image): {summary['skipped_missing_image']}")
    if summary["skipped_unparsed"]:
        print(f"skipped (unparseable content): {summary['skipped_unparsed']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_root", help="dataset root containing annotations/ and predictions/")
    ap.add_argument("--dry-run", action="store_true", help="report what would convert; write nothing")
    ap.add_argument(
        "--remove-source", action="store_true", help="delete each source .txt after writing its JSON"
    )
    args = ap.parse_args(argv)
    summary = migrate(args.dataset_root, dry_run=args.dry_run, remove_source=args.remove_source)
    _print_summary(summary, args.dry_run)


if __name__ == "__main__":
    main()

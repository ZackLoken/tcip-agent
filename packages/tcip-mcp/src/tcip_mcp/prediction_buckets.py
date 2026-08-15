"""Prediction-bucket immutability: never silently overwrite predictions a human reviewed.

A *bucket* is a directory (or set of task directories) holding per-image ``<stem>.json``
prediction files. Once a reviewer has recorded verdicts (accept/reject/edit) against any of
a bucket's images, re-running inference or re-staging into it would orphan those verdicts
(they reference the predictions by geometry). So the prediction writers resolve a run-scoped
bucket through here: with verdicts present the default writes are redirected to the next free
``<name>@r2`` / ``@r3`` variant, and an explicit ``overwrite=True`` is refused with a count.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from tcip_mcp.pipelines.resolution import SIDECAR_FILENAMES


def bucket_stems(*dirs: Path | str) -> set[str]:
    """Image stems that have a per-image prediction file across the given bucket dir(s)."""
    stems: set[str] = set()
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in p.glob("*.json"):
            if f.name not in SIDECAR_FILENAMES:
                stems.add(f.stem)
    return stems


def verdict_count(review_state_dir: Path | str, names: Iterable[str]) -> int:
    """Review verdicts recorded against ``names`` (image stems). 0 when no review state
    exists yet: no engine is created for a never-reviewed project."""
    from tcip_annotation.review_engine import REVIEW_SHARD_DIRNAME, ReviewEngine

    d = Path(review_state_dir)
    if not (d / REVIEW_SHARD_DIRNAME).is_dir():
        return 0
    return ReviewEngine(d).verdict_count_for_images(names)


class BucketHasVerdicts(Exception):
    """Raised on ``overwrite=True`` when the target bucket has review verdicts recorded."""

    def __init__(self, name: str, count: int, suggested: str) -> None:
        self.name = name
        self.count = count
        self.suggested = suggested
        super().__init__(
            f"prediction bucket {name!r} has {count} review verdict(s) recorded against it; "
            f"refusing to overwrite in place. Write to a new bucket (e.g. {suggested!r}) or "
            f"reconcile the verdicts first."
        )


@dataclass
class BucketResolution:
    name: str  # the bucket name to actually write to
    redirected: bool  # True when the requested bucket was frozen and a fresh one was chosen
    verdict_count: int  # verdicts on the requested bucket (0 unless redirected)
    requested: str  # the originally requested bucket name


def _bucket_verdicts(review_state_dir: Path | str, dirs: list[Path]) -> int:
    stems = bucket_stems(*dirs)
    if not stems:  # empty (or missing) bucket: nothing was ever reviewed there
        return 0
    return verdict_count(review_state_dir, stems)


def resolve_writable_bucket(
    review_state_dir: Path | str,
    requested: str,
    dirs_for: Callable[[str], list[Path]],
    *,
    overwrite: bool = False,
    max_variants: int = 99,
) -> BucketResolution:
    """Resolve which bucket to write to, honoring review-verdict immutability.

    ``dirs_for(name)`` returns the task dir(s) of the bucket variant ``name`` (one for a
    single-dir bucket, detect+segment for a prediction bucket). With no verdicts on the
    requested bucket, it is used as-is. With verdicts: ``overwrite=False`` (default) picks
    the next ``<requested>@r2`` / ``@r3`` variant that has none; ``overwrite=True`` raises
    :class:`BucketHasVerdicts`.
    """
    base = _bucket_verdicts(review_state_dir, dirs_for(requested))
    if base == 0:
        return BucketResolution(name=requested, redirected=False, verdict_count=0, requested=requested)

    suggested = f"{requested}@r{max_variants + 1}"  # fallback if every variant is somehow taken
    for n in range(2, max_variants + 1):
        cand = f"{requested}@r{n}"
        if _bucket_verdicts(review_state_dir, dirs_for(cand)) == 0:
            suggested = cand
            break

    if overwrite:
        raise BucketHasVerdicts(requested, base, suggested)
    return BucketResolution(name=suggested, redirected=True, verdict_count=base, requested=requested)


def resolve_prediction_bucket(
    dataset_root: str | Path,
    model_name: str,
    date: str | None,
    *,
    review_state_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, BucketResolution]:
    """The prediction dir a run may write for ``(dataset_root, model_name, date)``.

    The one place the platform turns that triple into a writable bucket, so every writer
    agrees on both the path convention (``dataset_layout.prediction_dir``) and which segment
    varies when the requested bucket carries review verdicts: the *model* one
    (``predictions/<model>@r2/<date>``), never the date. A model-named bucket is what
    ``list_models`` / ``models_with_predictions`` enumerate, so a redirected run stays
    discoverable; a date-named sibling would be invisible to every reader.

    Returns the dir to write and the resolution behind it (which bucket was requested,
    whether it was redirected, and the verdict count that forced the redirect).
    """
    from tcip_mcp.dataset_layout import prediction_dir

    def _bucket_dirs(name: str) -> list[Path]:
        return [Path(prediction_dir(dataset_root, name, date))]

    resolution = resolve_writable_bucket(
        review_state_dir, model_name, _bucket_dirs, overwrite=overwrite
    )
    return Path(prediction_dir(dataset_root, resolution.name, date)), resolution


def stage_prediction_shapes(
    dataset_root: str,
    model_name: str,
    date: str | None,
    stem: str,
    *,
    annotations: list,
    img_w: int,
    img_h: int,
    overwrite: bool = False,
) -> dict:
    """Write already-built prediction :class:`Annotation` records into a verdict-guarded bucket.

    The one staging path shared by ``stage_proposals`` and ``accept_proposals`` so both honor
    prediction-bucket immutability: a bucket that carries review verdicts is never overwritten (the
    default redirects to the next free ``<name>@r2`` variant; ``overwrite=True`` raises
    :class:`BucketHasVerdicts`). ``annotations`` are name-based prediction records (subject + geometry
    + score) already stamped with their producer; boxes and polygons live in the one per-image file.
    Returns the bucket actually written and the path.
    """
    from tcip_annotation import json_io

    pred_dir, resolution = resolve_prediction_bucket(
        dataset_root,
        model_name,
        date,
        review_state_dir=Path(dataset_root) / ".tcip" / "state",
        overwrite=overwrite,
    )

    path = None
    if annotations:
        pred_dir.mkdir(parents=True, exist_ok=True)
        out = pred_dir / f"{stem}.json"
        json_io.write_annotations(out, annotations, img_w, img_h)
        path = str(out)

    return {
        "bucket": resolution.name,
        "redirected": resolution.redirected,
        "verdict_count": resolution.verdict_count,
        "requested": resolution.requested,
        "path": path,
    }

"""Prediction-bucket immutability — never silently overwrite predictions a human reviewed.

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

# operating_point.json is a provenance stamp beside the predictions, not a per-image label.
_NON_LABEL_JSON = {"operating_point.json"}


def bucket_stems(*dirs: Path | str) -> set[str]:
    """Image stems that have a per-image prediction file across the given bucket dir(s)."""
    stems: set[str] = set()
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in p.glob("*.json"):
            if f.name not in _NON_LABEL_JSON:
                stems.add(f.stem)
    return stems


def verdict_count(review_state_dir: Path | str, names: Iterable[str]) -> int:
    """Review verdicts recorded against ``names`` (image stems). 0 when no review state
    exists yet — no engine is created for a never-reviewed project."""
    from tcip_annotation.review_engine import (
        REVIEW_SHARD_DIRNAME,
        REVIEW_STATE_FILENAME,
        ReviewEngine,
    )

    d = Path(review_state_dir)
    if not (d / REVIEW_SHARD_DIRNAME).is_dir() and not (d / REVIEW_STATE_FILENAME).is_file():
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
    if not stems:  # empty (or missing) bucket — nothing was ever reviewed there
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

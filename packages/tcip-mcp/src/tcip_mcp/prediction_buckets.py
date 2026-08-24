"""Prediction-bucket immutability: never silently overwrite predictions a human reviewed.

A *bucket* is a directory (or set of task directories) holding per-image ``<stem>.json``
prediction files. Once a reviewer has recorded verdicts (accept/reject/edit) against any of
a bucket's images, re-running inference or re-staging into it would orphan those verdicts
(they reference the predictions by geometry). So the prediction writers resolve a run-scoped
bucket through here: with verdicts present the default writes are redirected to the next free
``<name>@r2`` / ``@r3`` variant, and an explicit ``overwrite=True`` is refused with a count.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from tcip_mcp.pipelines.resolution import SIDECAR_FILENAMES, dataset_hash


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


def bucket_content_digest(*dirs: Path | str, memo: dict[str, str] | None = None) -> str:
    """Content identity of the prediction files held in the given bucket dir(s).

    :func:`~tcip_mcp.pipelines.resolution.dataset_hash` over each directory's enumerated stems, so a
    replaced file, an added one and a deleted one all change the digest. The sidecar exclusion is
    inherited from :func:`bucket_stems` rather than restated, so a stamp written for a new dimension
    can never make the digest vouch for itself. Several directories (a prediction bucket's detect and
    segment dirs) combine in sorted directory order, so the caller's argument order does not matter.

    ``memo`` is a caller-owned dict, keyed by resolved directory, that lives for the span of one
    delivery: a bucket read several times inside one delivery is hashed once, and a call that passes
    a fresh dict (or none) reads every prediction file again. There is deliberately no cross-call
    cache: recomputation is what detects a replacement whose size and timestamp were restored.
    """
    if not dirs:
        raise ValueError("bucket_content_digest needs at least one bucket directory to hash")

    digests: list[str] = []
    for d in sorted(Path(x).resolve() for x in dirs):
        key = str(d)
        if memo is not None and key in memo:
            digests.append(memo[key])
            continue
        digest = dataset_hash(d, stems=sorted(bucket_stems(d)))
        if memo is not None:
            memo[key] = digest
        digests.append(digest)

    if len(digests) == 1:
        return digests[0]
    h = hashlib.sha256()
    for digest in digests:
        h.update(digest.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def bucket_stems_digest(*dirs: Path | str) -> str:
    """Identity of the bucket's own image set: a digest over the sorted, combined stems
    :func:`bucket_stems` enumerates, never over the prediction files' bytes.

    Used where a claim is a fact about which images a bucket holds rather than about what was
    predicted on them (a physical-scale calibration, :mod:`tcip_mcp.pipelines.measurement.
    scale_calibration`): re-exporting predictions over the same images changes nothing this digest
    covers, so a scale claim stands across a re-export, while an image added to or removed from the
    bucket changes it, correctly floors the claim, and is a real reason to re-run the calibration.
    Several directories combine in sorted order, mirroring :func:`bucket_content_digest`'s own
    combination, so the caller's argument order does not matter.
    """
    if not dirs:
        raise ValueError("bucket_stems_digest needs at least one bucket directory to hash")
    stems = sorted(bucket_stems(*dirs))
    h = hashlib.sha256()
    for stem in stems:
        h.update(stem.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def review_state_dir_of(root: str | Path) -> Path:
    """``<root>/.tcip/state``: the review-verdict store for the dataset at ``root``.

    The one derivation of where verdicts live, so the immutability guard that counts them and the
    :class:`~tcip_annotation.ReviewEngine` that records them address the same store instead of each
    composing a state dir from whichever root it happens to hold. ``ReviewEngine`` still owns the
    shard layout inside it; this only says which store root it is opened on.
    """
    return Path(root, ".tcip", "state")


def bucket_key_of(bucket_dir: str | Path | None) -> str:
    """The verdict store's key for the prediction bucket at ``bucket_dir``.

    The one spelling of a bucket's identity in that store, so the reviewer recording a verdict and
    the immutability guard counting it name the same bucket. A bucket inside a dataset is named by
    its path relative to that dataset's root, the same relative form the count document's covered
    buckets use, so verdicts follow a dataset that moves. A directory under no dataset root has no
    such anchor and is named by its own resolved path. No directory at all is
    :data:`~tcip_annotation.review_engine.NO_BUCKET`, the key for a ground-truth-only review.
    """
    from tcip_annotation.review_engine import NO_BUCKET

    from tcip_mcp.dataset_layout import dataset_root_of

    if not bucket_dir:
        return NO_BUCKET
    d = Path(bucket_dir)
    root = dataset_root_of(d)
    if root is None:
        return d.resolve().as_posix()
    return d.relative_to(root).as_posix()


def verdict_count(review_state_dir: Path | str, bucket: str, names: Iterable[str]) -> int:
    """Review verdicts recorded against ``names`` (image stems) on ``bucket``. 0 when the store
    holds no verdicts at all: no engine is created for a never-reviewed dataset."""
    import tcip_store

    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE, ReviewEngine

    d = Path(review_state_dir)
    if not tcip_store.keys(REVIEW_VERDICTS_STORE, str(d)):
        return 0
    return ReviewEngine(d).verdict_count_for_images(bucket, names)


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
    # Counted per directory under that directory's own bucket key: a namesake image reviewed under
    # another bucket must not freeze this one.
    total = 0
    for d in dirs:
        stems = bucket_stems(d)
        if not stems:  # empty (or missing) bucket: nothing was ever reviewed there
            continue
        total += verdict_count(review_state_dir, bucket_key_of(d), stems)
    return total


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
        review_state_dir=review_state_dir_of(dataset_root),
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

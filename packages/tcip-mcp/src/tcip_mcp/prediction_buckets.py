"""Prediction-bucket immutability: never silently overwrite predictions a human reviewed, and
never silently publish a second run into a bucket a prior run already filled.

A *bucket* is a directory (or set of task directories) holding per-image ``<stem>.json``
prediction files, its identity the bucket directory's own path (relative to the dataset root
when it sits under one, its own resolved path otherwise, per :func:`bucket_key_of`) rather than
a score bin or a quota allocation. The canonical ``predictions/<model>/<date>`` layout is one
regime's convention for building that path, not the definition of a bucket's identity. Once a
reviewer has recorded verdicts (accept/reject/edit) against any of a bucket's images,
re-running inference or re-staging into it would orphan those verdicts (they reference the
predictions by geometry); the staging door's own reading is wider still, since a bucket a
reviewer has simply finished with no verdict at all (a bulk accept) is just as immutable to it
(:func:`review_state_count`'s ``names`` form, see below). So the prediction writers resolve a
run-scoped bucket through here: with verdicts (or, for the staging door, review state) present
the default writes are redirected to the next free ``<name>@r2`` / ``@r3`` variant, and an
explicit ``overwrite=True`` is refused with a count.

A second, narrower rule applies only to the callers that opt into it
(:func:`resolve_writable_bucket`'s ``refuse_documents``): a requested bucket that already holds
prediction documents, with no verdict yet recorded against it, refuses outright rather than
redirecting or overwriting, since a bucket left in that state was already published by a prior
run this call would otherwise write beside or over. Three publishers opt in (``run_inference``,
``deliver_per_image_counts``'s live path, and the web route's own launch); ``stage_prediction_shapes``
alone leaves this off, since it accumulates one stem per call into a bucket by contract.

A third keyword, ``count_review_state``, widens what the guard above counts: off by default (the
three publishers' reading, detection verdicts alone), on for the staging door alone, which reads
every image a reviewer has finished, a bulk accept included, so a bucket a reviewer merely
completed is exactly as immutable to it as one carrying a verdict. The shape is the one
``refuse_documents`` already has: a caller opts into a stricter reading of the same bucket, and
the module's one guard (:func:`_bucket_guard_count`) answers either reading.

A staged bucket carries no ``operating_point.json`` stamp and so no recorded ``(subject,
attribute)`` scope: each record's ``subject`` is whatever the caller named
(:func:`~tcip_mcp.tools.proposal_tools.stage_proposals`), the platform validates no subject name,
and every reader below reads a staged bucket's records under the caller's own statement rather
than a stamp's.

Exhaustion is refused for every caller: when every ``<name>@r<n>`` variant up to the search's
ceiling is already taken under the reading that caller asked for (a detection verdict for the
three publishers, any finished image for the staging door), the resolver raises
:class:`BucketHasVerdicts` naming no suggestion rather than falling back to an unchecked,
never-searched ``<name>@r100``. ``stage_prediction_shapes``, the one caller that never opts into
the document guard, meets this same refusal on exhaustion, and reaches it sooner than a publisher
would, since its own reading takes a variant on a bulk accept alone.

A bucket's documents leave it through exactly one audited door,
:func:`~tcip_mcp.tools.inference_tools.clear_prediction_bucket`, which moves a terminal
experiment's own recorded bucket into a dated archive so the path re-publishes; it is a different
act from anything here, since it moves the whole bucket rather than resolving a writer's target,
and this module's guards never see the archive it moves into. A bucket emptied by hand, outside
that door, is invisible to every guard here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from tcip_annotation.json_io import prediction_documents
from tcip_mcp.pipelines.resolution import dataset_hash


def bucket_stems(*dirs: Path | str) -> set[str]:
    """Image stems that have a per-image prediction file across the given bucket dir(s)."""
    stems: set[str] = set()
    for d in dirs:
        stems.update(f.stem for f in prediction_documents(d))
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


def bucket_stems_digest(*dirs: Path | str, images_dir: Path | str) -> str:
    """Identity of the bucket's own imagery: a digest over the sorted, combined stems
    :func:`bucket_stems` enumerates, each stem's own image bytes read from ``images_dir``, never
    the prediction files' bytes.

    Used where a claim is a fact about the images a bucket was earned against rather than about
    what was predicted on them (a physical-scale calibration, :mod:`tcip_mcp.pipelines.measurement.
    scale_calibration`): re-exporting predictions over the same images changes nothing this digest
    covers, so a scale claim stands across a re-export, while a stem added to or removed from the
    bucket, or an image's own bytes replaced under the same filename, changes it, correctly floors
    the claim, and is a real reason to re-run the calibration. A bucket's stems are the images it
    was predicted on, so a stem ``images_dir`` cannot resolve is a bucket and an image directory
    that do not belong together, refused by name rather than hashed as if the image were empty.
    Several directories combine in sorted order, mirroring :func:`bucket_content_digest`'s own
    combination, so the caller's argument order does not matter.
    """
    if not dirs:
        raise ValueError("bucket_stems_digest needs at least one bucket directory to hash")
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.image_utils import resolve_image_source

    stems = sorted(bucket_stems(*dirs))
    h = hashlib.sha256()
    for stem in stems:
        h.update(stem.encode("utf-8"))
        h.update(b"\0")
        try:
            source = resolve_image_source(images_dir, stem)
        except FileNotFoundError as exc:
            raise ValueError(
                f"bucket stem {stem!r} has a prediction but no image under {str(images_dir)!r}; "
                "a scale claim binds to the bucket's own imagery, so pass the images directory "
                "the bucket was predicted on"
            ) from exc
        if isinstance(source, BandGroupRef):
            for band_path in source.bands.values():
                h.update(band_path.read_bytes())
        else:
            h.update(Path(source).read_bytes())
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


def _open_review_engine(review_state_dir: Path | str):
    """The review engine over ``review_state_dir``, or ``None`` when its store holds no verdicts
    at all: no engine is created for a never-reviewed dataset. The one store-emptiness guard and
    engine construction :func:`verdict_count` and :func:`review_state_count` both open through,
    so neither carries its own copy."""
    import tcip_store

    from tcip_annotation.review_engine import REVIEW_VERDICTS_STORE, ReviewEngine

    d = Path(review_state_dir)
    if not tcip_store.keys(REVIEW_VERDICTS_STORE, str(d)):
        return None
    return ReviewEngine(d)


def verdict_count(review_state_dir: Path | str, bucket: str, names: Iterable[str]) -> int:
    """Review verdicts recorded against ``names`` (image stems) on ``bucket``. 0 when the store
    holds no verdicts at all: no engine is created for a never-reviewed dataset."""
    engine = _open_review_engine(review_state_dir)
    if engine is None:
        return 0
    return engine.verdict_count_for_images(bucket, names)


_REVIEWED_IMAGE_STATUSES = ("completed", "started")
"""The two image-review states :func:`review_state_count` counts: a reviewer has recorded
something against the image, a detection verdict or a bulk accept alike. A ``not_started`` shard
an ``unmark_image_reviewed`` call leaves behind never counts: it names no review decision."""


def review_state_count(
    review_state_dir: Path | str, bucket: str, names: Iterable[str] | None = None
) -> int:
    """Every image the review engine holds state for on ``bucket``, whether or not that state
    carries a detection verdict: a bulk-accepted image (``img_status`` ``completed`` with no
    verdict entries) counts here and not in :func:`verdict_count`, and so does a reviewed image
    whose prediction document has since been removed. A ``not_started`` shard never counts. 0
    when the store holds no verdicts at all, the same fold :func:`verdict_count` applies.

    With ``names`` omitted (``None``), bucket-wide: every image name the engine holds state for
    under ``bucket``, not only the stems a document exists for today (the resolver's own residual,
    see :func:`resolve_writable_bucket`). The door this backs (``clear_prediction_bucket``)
    refuses a bucket carrying review state, not its documents, so an image whose document is gone
    must still be counted.

    With ``names`` given (identity against ``None``, never truthiness, so an empty ``()`` or an
    exhausted iterable counts nothing rather than falling back to bucket-wide), stem-scoped: only
    the images whose ``Path(img_name).stem`` is among ``{Path(n).stem for n in names}``, the exact
    rule :meth:`~tcip_annotation.review_engine.ReviewEngine.verdict_count_for_images` applies, so a
    stem the detection count matches this count matches too, and a stem carrying a dot (e.g.
    ``plot_3.east``) misses in both counts the same way, a residual of that shared matching rule
    rather than of this function; a fix belongs to the engine's rule, not to either count alone.
    Backs the staging door's guard, scoped to the stems a document is held for.
    """
    engine = _open_review_engine(review_state_dir)
    if engine is None:
        return 0
    states = engine.image_states(bucket)
    if names is not None:
        wanted = {Path(n).stem for n in names}
        if not wanted:
            return 0
        return sum(
            1 for img_name, data in states.items()
            if Path(img_name).stem in wanted and data.get("img_status") in _REVIEWED_IMAGE_STATUSES
        )
    return sum(1 for data in states.values() if data.get("img_status") in _REVIEWED_IMAGE_STATUSES)


class BucketHasVerdicts(Exception):
    """Raised on ``overwrite=True`` when the target bucket has review verdicts recorded, or when
    the variant search that would otherwise redirect around them finds every candidate up to the
    search's ceiling taken. ``unit`` names what ``count`` counts in the message: the three
    publishers' default, ``"review verdict(s)"`` (detection entries alone), or the staging door's
    own ``"reviewed image(s)"`` under ``count_review_state`` (a detection verdict or a bulk accept
    alike), so a caller reading the message is never told a verdict was given when only a bulk
    accept was."""

    def __init__(self, name: str, count: int, suggested: str | None, *,
                 unit: str = "review verdict(s)") -> None:
        self.name = name
        self.count = count
        self.suggested = suggested
        if suggested is None:
            message = (
                f"prediction bucket {name!r} has {count} {unit} recorded against it, "
                f"and every {name}@r<n> variant up to the search's ceiling is taken too: refusing "
                f"to overwrite in place with no free variant to redirect to or suggest."
            )
        else:
            message = (
                f"prediction bucket {name!r} has {count} {unit} recorded against it; "
                f"refusing to overwrite in place. Write to a new bucket (e.g. {suggested!r}) or "
                f"reconcile what is recorded against it first."
            )
        super().__init__(message)


class BucketHoldsDocuments(Exception):
    """Raised, for the callers that opt into :func:`resolve_writable_bucket`'s
    ``refuse_documents``, when the requested bucket already holds prediction documents with no
    review verdict yet recorded against it: a prior run already published there, and no second
    publish begins into a bucket in that state, whatever ``overwrite`` says."""

    def __init__(self, name: str, document_stem_count: int, suggested: str | None) -> None:
        self.name = name
        self.document_stem_count = document_stem_count
        self.suggested = suggested
        if suggested is None:
            message = (
                f"prediction bucket {name!r} already holds {document_stem_count} prediction "
                f"document(s), and every {name}@r<n> variant up to the search's ceiling holds a "
                f"verdict or a document too: refusing to publish with no free variant to suggest."
            )
        else:
            message = (
                f"prediction bucket {name!r} already holds {document_stem_count} prediction "
                f"document(s); no publish begins into a bucket that held prediction documents "
                f"when this door resolved it (it may hold an earlier stamp with no document, "
                f"which a fresh publish still replaces). Write to a new bucket instead (e.g. "
                f"{suggested!r}, which holds neither a verdict nor a document)."
            )
        super().__init__(message)


@dataclass
class BucketResolution:
    name: str  # the bucket name to actually write to
    redirected: bool  # True when the requested bucket was frozen and a fresh one was chosen
    verdict_count: int  # count under the caller's reading (verdicts, or count_review_state's reviewed images); 0 unless redirected
    requested: str  # the originally requested bucket name


def bucket_document_stem_count(dirs: Iterable[Path]) -> int:
    """The count :class:`BucketHoldsDocuments` names: the distinct image stems holding a
    prediction document across the given bucket dir(s), the same union :func:`bucket_stems`
    enumerates."""
    return len(bucket_stems(*dirs))


def _bucket_guard_count(
    review_state_dir: Path | str | None, dirs: list[Path], *, count_review_state: bool
) -> int:
    # No store to guard against: a bucket under no dataset root never reaches Path(None) below.
    if review_state_dir is None:
        return 0
    # Counted per directory under that directory's own bucket key: a namesake image reviewed under
    # another bucket must not freeze this one.
    total = 0
    for d in dirs:
        stems = bucket_stems(d)
        if not stems:  # empty (or missing) bucket: nothing was ever reviewed there
            continue
        # count_review_state reads every finished image (a verdict or a bulk accept); off, a verdict alone.
        total += (
            review_state_count(review_state_dir, bucket_key_of(d), stems)
            if count_review_state
            else verdict_count(review_state_dir, bucket_key_of(d), stems)
        )
    return total


def _first_free_variant(
    review_state_dir: Path | str | None,
    requested: str,
    dirs_for: Callable[[str], list[Path]],
    *,
    refuse_documents: bool,
    count_review_state: bool,
    max_variants: int,
) -> str | None:
    """The first ``<requested>@r2`` .. ``@r<max_variants>`` variant free of the guard's own count
    under the caller's reading (:func:`_bucket_guard_count`, ``count_review_state``), and, with
    ``refuse_documents``, also free of a document, or ``None`` when every one up to the ceiling is
    taken. The one search behind both a redirect and either exception's suggestion, so a candidate
    cannot pass as free under one predicate and fail under the other."""
    for n in range(2, max_variants + 1):
        cand = f"{requested}@r{n}"
        cand_dirs = dirs_for(cand)
        if _bucket_guard_count(review_state_dir, cand_dirs,
                                count_review_state=count_review_state) != 0:
            continue
        if refuse_documents and bucket_document_stem_count(cand_dirs):
            continue
        return cand
    return None


def resolve_writable_bucket(
    review_state_dir: Path | str | None,
    requested: str,
    dirs_for: Callable[[str], list[Path]],
    *,
    overwrite: bool = False,
    refuse_documents: bool = False,
    count_review_state: bool = False,
    max_variants: int = 99,
) -> BucketResolution:
    """Resolve which bucket to write to, honoring the review immutability the caller asked for
    (detection verdicts alone, or every finished image under ``count_review_state``) and, for the
    callers that opt in, prediction-document immutability.

    ``dirs_for(name)`` returns the task dir(s) of the bucket variant ``name`` (one for a
    single-dir bucket, detect+segment for a prediction bucket). ``review_state_dir`` is ``None``
    for a bucket under no dataset root, whose guard is inoperative either way
    (:func:`_bucket_guard_count` answers zero without touching disk); the document guard runs
    regardless, since it consults no store.

    ``count_review_state`` (default off) widens the guard's own count from detection verdicts
    alone to every image a reviewer has finished, a bulk accept included
    (:func:`review_state_count`'s stem-scoped form); the shape is the one ``refuse_documents``
    already has, a caller opting into a stricter reading of the same bucket, and the module's one
    guard answers either reading through :func:`_bucket_guard_count`.

    With the guard's count zero on the requested bucket: used as-is, unless ``refuse_documents``
    is set and the bucket already holds a prediction document, which raises
    :class:`BucketHoldsDocuments` whatever ``overwrite`` says, naming the document count and the
    suggested first variant free of both the guard's count and a document (or ``None``, see
    below). With a non-zero count on the requested bucket: that check runs first, ahead of any
    document check, so a bucket already answered for under the caller's reading redirects (or
    refuses on ``overwrite=True``) whether or not it also holds a document. ``overwrite=False``
    (default) picks the next ``<requested>@r2`` / ``@r3`` variant free of the guard's count (and,
    with ``refuse_documents``, also free of a document); ``overwrite=True`` raises
    :class:`BucketHasVerdicts`. Either exception's suggestion is the one variant search: a
    candidate free of the guard's count and, with the document keyword on, also free of a document
    is free. When no variant up to ``max_variants`` is free, the unchecked next one is never
    returned as a target: the resolver raises the class it was resolving for with
    ``suggested=None`` and a message saying every variant is taken, since a redirect or a
    suggestion onto an unchecked directory is the overwrite this guard exists to refuse. This
    exhaustion refusal fires whether or not ``refuse_documents`` is set: ``stage_prediction_shapes``,
    the one caller that leaves ``refuse_documents`` off, is gated on review state alone through
    ``count_review_state``, but on exhaustion meets this same raise rather than an unchecked
    ``@r100`` fallback.

    A raster pass' own progress records, kept under a bucket's ``<out>/.tcip/`` subdirectory,
    never register as a document either: :func:`bucket_document_stem_count`'s
    :func:`~tcip_annotation.json_io.prediction_documents` enumeration is a non-recursive
    ``*.json`` listing with sidecars excluded, so a directory holding only a progress record and
    no top-level document reads as empty to this check, and a resumed pass survives whichever
    bucket the guard (or this one) resolved it to.

    What the door guarantees, stated at its size: no publish begins into a bucket that held
    prediction documents when the door resolved it. It does not guarantee that a bucket never
    comes to hold two runs' documents: the check runs at resolution, ahead of a pass that can take
    minutes, so two doors racing into one bucket can each resolve it clean and interleave their
    writes, the same window the guard already has. Nor does a documentless bucket answer for
    review state the review store still holds against it: ``_bucket_guard_count`` skips a
    directory with no stems without consulting the store, so a bucket whose documents were removed
    after review answers zero here regardless of what the store still holds. The one audited door
    that empties a bucket on purpose,
    :func:`~tcip_mcp.tools.inference_tools.clear_prediction_bucket`, refuses one carrying review
    state for exactly that reason; a bucket emptied by hand, outside it, reaches this same silent
    admission.
    """
    requested_dirs = dirs_for(requested)
    base = _bucket_guard_count(review_state_dir, requested_dirs,
                                count_review_state=count_review_state)
    if base == 0:
        if refuse_documents:
            doc_count = bucket_document_stem_count(requested_dirs)
            if doc_count:
                suggested = _first_free_variant(
                    review_state_dir, requested, dirs_for,
                    refuse_documents=True, count_review_state=count_review_state,
                    max_variants=max_variants)
                raise BucketHoldsDocuments(requested, doc_count, suggested)
        return BucketResolution(name=requested, redirected=False, verdict_count=0, requested=requested)

    suggested = _first_free_variant(
        review_state_dir, requested, dirs_for,
        refuse_documents=refuse_documents, count_review_state=count_review_state,
        max_variants=max_variants)
    unit = "reviewed image(s)" if count_review_state else "review verdict(s)"
    if overwrite or suggested is None:
        raise BucketHasVerdicts(requested, base, suggested, unit=unit)
    return BucketResolution(name=suggested, redirected=True, verdict_count=base, requested=requested)


def resolve_prediction_bucket(
    dataset_root: str | Path,
    model_name: str,
    date: str | None,
    *,
    review_state_dir: str | Path | None,
    overwrite: bool = False,
    refuse_documents: bool = False,
    count_review_state: bool = False,
) -> tuple[Path, BucketResolution]:
    """The prediction dir a run may write for ``(dataset_root, model_name, date)``.

    The one place the platform turns that triple into a writable bucket, so every writer
    agrees on both the path convention (``dataset_layout.prediction_dir``) and which segment
    varies when the requested bucket carries review state under the caller's reading: the
    *model* one (``predictions/<model>@r2/<date>``), never the date. A model-named bucket is what
    ``list_models`` / ``models_with_predictions`` enumerate, so a redirected run stays
    discoverable; a date-named sibling would be invisible to every reader. ``refuse_documents``
    and ``count_review_state`` both forward to :func:`resolve_writable_bucket`; the staging door
    leaves the first off and turns the second on, the three publishers the reverse.

    Returns the dir to write and the resolution behind it (which bucket was requested, whether it
    was redirected, and the count that forced the redirect, in the caller's own unit).
    """
    from tcip_mcp.dataset_layout import prediction_dir

    def _bucket_dirs(name: str) -> list[Path]:
        return [Path(prediction_dir(dataset_root, name, date))]

    resolution = resolve_writable_bucket(
        review_state_dir, model_name, _bucket_dirs,
        overwrite=overwrite, refuse_documents=refuse_documents,
        count_review_state=count_review_state,
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

    The one staging path both of ``stage_proposals``'s input regimes share so each honors
    prediction-bucket immutability: a bucket that carries review state (a detection verdict or a
    bulk accept alike) is never overwritten (the default redirects to the next free ``<name>@r2``
    variant; ``overwrite=True`` raises :class:`BucketHasVerdicts`). ``annotations`` are name-based
    prediction records (subject + geometry + score) already stamped with their producer; boxes
    and polygons live in the one per-image file.

    This door leaves ``resolve_prediction_bucket``'s ``refuse_documents`` at its default off: a
    staging bucket accumulates one stem per call by contract, so a stem this call adds beside
    another call's own document is the bucket working as designed, never the second-publish the
    document guard exists to refuse. It turns ``count_review_state`` on instead, so a bucket a
    reviewer has simply finished (a bulk accept, no detection entry at all) is exactly as immutable
    to this door as one carrying a verdict: a session that stages one image, then completes it on
    the Review canvas, before staging the next sends the next stage into a fresh ``@r2`` rather
    than beside the one just finished. ``run_inference``, ``deliver_per_image_counts``'s live path
    and the web route's own launch, the three publishers that leave ``count_review_state`` off and
    turn ``refuse_documents`` on instead, never call this function.

    This door writes no stamp of its own, so a bucket holding nothing but staged shapes carries
    none, and the promotion path refuses it as a bucket with no stamp a producer wrote. A bucket a
    producer already stamped (``run_inference``, say) keeps that producer's stamp: staging into it
    only redirects when it carries review state, so a stamped-but-unreviewed bucket can gain
    staged shapes beside its own stamp.

    Returns the bucket actually written and the path; ``verdict_count`` counts reviewed images (a
    detection verdict or a bulk accept alike), not detection entries, so a session that interleaves
    staging with completing images on the Review canvas can see one engine's proposals spread over
    ``@r2``, ``@r3`` and onward, each variant frozen by the review state it acquired. Stage every
    image of a run before reviewing any to avoid it, or read ``bucket``/``bucket_redirected`` off
    each result.
    """
    from tcip_annotation import json_io

    from tcip_mcp.dataset_layout import label_filename

    if json_io.is_sidecar_name(label_filename(stem)):
        raise ValueError(
            f"{stem} names one of a prediction bucket's own provenance stamps; an image whose "
            "stem is reserved this way can never be written as a bucket's per-image prediction "
            "document, since the stamp write would then destroy or refuse over it."
        )

    pred_dir, resolution = resolve_prediction_bucket(
        dataset_root,
        model_name,
        date,
        review_state_dir=review_state_dir_of(dataset_root),
        overwrite=overwrite,
        count_review_state=True,
    )

    path = None
    if annotations:
        pred_dir.mkdir(parents=True, exist_ok=True)
        out = pred_dir / label_filename(stem)
        json_io.write_annotations(out, annotations, img_w, img_h)
        path = str(out)

    return {
        "bucket": resolution.name,
        "redirected": resolution.redirected,
        "verdict_count": resolution.verdict_count,
        "requested": resolution.requested,
        "path": path,
    }

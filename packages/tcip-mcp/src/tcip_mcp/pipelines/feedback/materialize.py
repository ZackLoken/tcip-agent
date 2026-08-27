"""Materialize a curated detection dataset from human review verdicts.

Torch-free. Turns review verdicts (per-image shards under ``.tcip/state/review/``) into training data:
  - accepted / edited GT boxes  -> positive name-based per-image JSON labels (the canonical format)
  - rejected-only images        -> confirmed-negative JSON (``{"annotations": []}``) backgrounds
plus a ``curated_manifest.json`` for provenance. The output layout (``images/`` + ``annotations/``)
matches ``data_tools._scan_dataset`` so the loop chains straight into ``make_splits`` /
``launch_training`` with no glue.

The verdict log stores normalized center-form boxes (``[cx, cy, w, h]``) plus the class *name*
(``class_name``, an annotation's subject); positives are denormalized to pixel coordinates using the
copied image's dimensions (the canonical JSON is pixel-space), no inference re-run.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import (
    CONFIRMED_NEGATIVE, annotation_root, image_root, label_filename, status_records,
)
from tcip_mcp.identity import user_identity
from tcip_mcp.pipelines.feedback.verdicts import decode_verdict

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

_CURATED_DOC = RootedFileLocator(suffix=".json")
"""A curated dataset's own documents. The output directory is wherever the caller asked the
dataset to be materialized, so no dataset resolver owns its layout."""

CURATED_MANIFEST_STORE = "curated_manifest"
_CURATED_MANIFEST_PARTS = ("curated_manifest",)
register_store(
    StoreDescriptor(
        name=CURATED_MANIFEST_STORE,
        kind="record",
        key_fields=("document",),
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_CURATED_DOC,
    )
)


def curated_manifest_key(output_dir: str | Path) -> Key:
    """What a curated dataset was assembled from, image by image.

    ``last_writer_wins``: written once, whole, at the end of the materialization that
    produced the directory it describes.
    """
    return Key(CURATED_MANIFEST_STORE, str(Path(output_dir).absolute()), _CURATED_MANIFEST_PARTS)


def curated_manifest_path(output_dir: str | Path) -> Path:
    """Where the manifest lands, for a caller reporting the artifact it just produced."""
    key = curated_manifest_key(output_dir)
    return Path(key.root, *_CURATED_DOC.relative_path(key.root, key.parts).parts)


def partition_review_verdicts(review_state: dict, *, only_completed: bool = False) -> dict[str, dict]:
    """Partition per-image review verdicts into positives / hard-negatives / skip.

    Returns ``{img_name: {"positives": [(class_name, cx, cy, w, h)], "rejected_count": int,
    "rejected_subjects": [class_name], "subjects": [class_name], "reviewers": [name],
    "status": "positive"|"hard_negative"|"skip"}}``.
    A detection's box is ``gt_bbox_norm or pred_bbox_norm`` (the fallback handles
    accepted-FP entries that carry only a predicted box); ``class_name`` is the subject.

    ``rejected_subjects`` is what the image's rejections actually answer for, so a caller keying
    negatives can check that against the subject it is keying them under: a count of rejections
    says an image was disputed, never which object was found absent. ``subjects`` is every subject
    any verdict on the image names, rejections included, which is what a review's own subject is
    derived from. ``reviewers`` is who recorded the rejections, for attributing what they
    established. A verdict naming no subject contributes to neither set: it answers for nothing.
    """
    result: dict[str, dict] = {}
    for img_name, img_data in review_state.get("image", {}).items():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        positives: list[tuple] = []
        rejected = 0
        rejected_subjects: set[str] = set()
        subjects: set[str] = set()
        reviewers: set[str] = set()
        for entry in img_data.get("detections", []):
            verdict = decode_verdict(entry)
            if verdict.class_name:
                subjects.add(verdict.class_name)
            if verdict.is_positive:
                if verdict.affirmed_box is not None:
                    positives.append((verdict.class_name, *verdict.affirmed_box))
            elif verdict.is_rejection:
                rejected += 1
                if verdict.class_name:
                    rejected_subjects.add(verdict.class_name)
                if verdict.reviewed_by:
                    reviewers.add(verdict.reviewed_by)
        status = "positive" if positives else ("hard_negative" if rejected else "skip")
        result[img_name] = {
            "positives": positives, "rejected_count": rejected,
            "rejected_subjects": sorted(rejected_subjects), "subjects": sorted(subjects),
            "reviewers": sorted(reviewers), "status": status,
        }
    return result


def _find_source_image(source_images_dir: str, img_name: str) -> "Path | BandGroupRef | None":
    """The logical image ``img_name`` names, a plain ``Path``, or (when a ``.bandgroup``
    manifest groups sibling band files under this stem) a ``BandGroupRef``. ``None`` if unresolvable
    (missing, or a stale group whose manifest references a deleted sibling).

    Lazy-imports ``image_utils`` (which pulls in torch) so this module stays torch-free at import
    time, matching every other caller in this file.
    """
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete, resolve_image_source

    stem = Path(img_name).stem
    try:
        return resolve_image_source(source_images_dir, stem)
    except (FileNotFoundError, BandGroupIncomplete):
        return None


def _write_positive_label(path: Path, positives: list[tuple], img_w: int, img_h: int) -> str | None:
    """Write one image's positive boxes to its label file, returning the refusal message on
    failure rather than letting it propagate.

    A verdict's normalized box can denormalize to zero extent (the persistence boundary's own
    refusal, ``check_box_extent`` inside ``write_annotations``); caught here so one degenerate
    record does not abort a harvest of many images with an uncaught ``ValueError``.
    """
    # Denormalize the verdict log's [cx,cy,w,h] to pixel xyxy for the name-based per-image JSON.
    anns = [
        Annotation(subject=name,
                   geometry=BBox((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                                 (cx + w / 2) * img_w, (cy + h / 2) * img_h))
        for (name, cx, cy, w, h) in positives
    ]
    try:
        write_annotations(str(path), anns, img_w, img_h, keep_empty=True)
    except ValueError as exc:
        return str(exc)
    return None


MATERIALIZER_IDENTITY = "materialize_review_dataset"
"""The actor stamped on a confirmation this harvest wrote and no one reviewer answers for.

Bare, under the platform's identity convention (a person is ``user:<name>``), so a reader can see
that a function transcribed this negative from a review rather than a person recording it in the
GUI. It names the audited tool call the write happened under, which is where the arguments and the
verdict store it read are recorded.
"""


def _attribute_negatives(
    negative_verdicts: dict[str, dict], neg_subject: str | None, verdict_subjects: set[str],
) -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Split rejected-only images into the confirmations to write and the ones nobody may claim.

    An image is confirmed negative for ``neg_subject`` only when its own rejections name that
    subject. Every other one is returned in the second list with what its rejections did answer
    for, so the caller can say which images were left unconfirmed and why instead of keying them
    under a subject no verdict on them mentions.

    A confirmation is attributed to the reviewer whose rejections established it when the image's
    rejections name exactly one, and to :data:`MATERIALIZER_IDENTITY` when they name none or
    several, which is the honest answer where no single person answers for the image.
    """
    confirmed: dict[str, dict[str, str]] = {}
    unconfirmed: list[dict] = []
    for name, info in negative_verdicts.items():
        rejected_subjects = info["rejected_subjects"]
        if neg_subject and neg_subject in rejected_subjects:
            reviewers = info["reviewers"]
            recorded_by = (user_identity(reviewers[0]) if len(reviewers) == 1
                           else MATERIALIZER_IDENTITY)
            confirmed.update(status_records({name: CONFIRMED_NEGATIVE}, recorded_by=recorded_by))
            continue
        if neg_subject:
            reason = (f"its rejections answer for {rejected_subjects or 'no subject'}, not for "
                      f"{neg_subject!r}, the subject this materialization keys negatives under")
        else:
            reason = (f"the verdicts name {sorted(verdict_subjects)}, so no single subject the "
                      f"negatives could be keyed under; state one to attribute them")
        unconfirmed.append({"image": name, "rejected_subjects": rejected_subjects,
                            "reason": reason})
    return confirmed, sorted(unconfirmed, key=lambda e: e["image"])


def materialize_dataset(
    review_state: dict,
    source_images_dir: str,
    output_dir: str,
    *,
    subject: str | None = None,
    review_state_path: str = "",
    include_hard_negatives: bool = True,
    copy_files: bool = True,
    only_completed: bool = False,
    producer_model: dict | None = None,
) -> dict:
    """Write ``output_dir/images/`` + ``output_dir/annotations/`` + manifest.

    ``subject`` is the object the review was about (the confirmed negatives are keyed under it). When
    omitted it is derived from every subject the verdicts name, rejections included, and answers only
    when they name exactly one. ``producer_model`` (best-effort) records the model whose predictions
    the human reviewed, for traceability.

    A rejected-only image becomes a confirmed negative only when its own rejections answer for that
    subject. One whose rejections answer for another subject, or for none, is materialized as an
    unconfirmed empty label and named in ``unconfirmed_negatives`` with why: keying it under the
    subject at hand would assert the human found that object absent on an image they never gave a
    verdict about, which is how an image full of one object reaches training as a zero-box sample of
    it.
    """
    partition = partition_review_verdicts(review_state, only_completed=only_completed)
    verdict_subjects = {s for info in partition.values() for s in info["subjects"]}
    neg_subject = subject or (next(iter(verdict_subjects)) if len(verdict_subjects) == 1 else None)
    out = Path(output_dir)
    images_out = image_root(out)
    labels_out = annotation_root(out)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    place = shutil.copy2 if copy_files else os.symlink
    counts = {"positive": 0, "hard_negative": 0, "skipped": 0, "total_boxes": 0,
              "missing_images": 0, "unconfirmed_negative": 0, "boundary_refused": 0}
    subjects: set[str] = set()
    manifest_images: list[dict] = []
    negative_verdicts: dict[str, dict] = {}
    boundary_refused: list[dict] = []

    for img_name, info in partition.items():
        status = info["status"]
        if status == "skip" or (status == "hard_negative" and not include_hard_negatives):
            counts["skipped"] += 1
            continue
        src = _find_source_image(source_images_dir, img_name)
        if src is None:
            counts["missing_images"] += 1
            continue

        from tcip_mcp.pipelines.image_utils import image_dimensions, place_logical_image, stem_of

        record_name = place_logical_image(src, images_out, place)
        stem = stem_of(src)
        label_path = labels_out / label_filename(stem)
        img_w, img_h = image_dimensions(src)

        if status == "positive":
            refusal = _write_positive_label(label_path, info["positives"], img_w, img_h)
            if refusal is not None:
                counts["boundary_refused"] += 1
                boundary_refused.append({"image": record_name, "reason": refusal})
                continue
            counts["positive"] += 1
            counts["total_boxes"] += len(info["positives"])
            subjects.update(name for (name, *_rest) in info["positives"])
        else:  # hard_negative -> empty label file, confirmed below only for the subject it answers for
            write_annotations(str(label_path), [], img_w, img_h, keep_empty=True)
            counts["hard_negative"] += 1
            negative_verdicts[record_name] = info

        manifest_images.append({
            "image": record_name, "status": status, "n_boxes": len(info["positives"]),
            "rejected_count": info["rejected_count"],
            "rejected_subjects": info["rejected_subjects"], "label": str(label_path),
        })

    # Training trusts a human-confirmed negative, never a bare empty file (a label may be emptied mid-work).
    negatives, unconfirmed = _attribute_negatives(negative_verdicts, neg_subject, verdict_subjects)
    counts["unconfirmed_negative"] = len(unconfirmed)
    if negatives:
        from tcip_mcp.dataset_layout import replace_image_status_store, status_bucket

        bucket_key = status_bucket(neg_subject or "", None)
        replace_image_status_store(out, {bucket_key: negatives})

        # Give the materialized dataset its own registry copy + a fresh per-image schema stamp, so
        # quarantine can actually protect these review-harvested negatives later; without this,
        # confirmed_negative_names has no classes.json to compare against and quarantine can never
        # fire here (a permanent no-op, not the "admit until proven stale" default it should be).
        from tcip_mcp.class_registry import (
            RegistryError, attribute_schema_digest, copy_registry, read_registry,
        )
        from tcip_mcp.dataset_layout import (
            classes_path, dataset_root_of, stamp_image_status_digests,
        )

        src_root = dataset_root_of(source_images_dir)
        src_classes = classes_path(src_root) if src_root is not None else None
        if src_classes is not None and src_classes.is_file():
            try:
                digest = attribute_schema_digest(read_registry(src_classes), neg_subject)
            except (OSError, RegistryError):
                digest = None
            if digest is not None:
                copy_registry(src_classes, classes_path(out))
                stamp_image_status_digests(out, bucket_key, negatives, digest)

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "review_state": review_state_path,
        "source_images_dir": str(source_images_dir),
        "output_dir": str(out),
        "producer_model": producer_model,
        "subject": neg_subject,
        "counts": counts,
        "subjects": sorted(subjects),
        "verdict_subjects": sorted(verdict_subjects),
        "unconfirmed_negatives": unconfirmed,
        "boundary_refused": boundary_refused,
        "images": manifest_images,
    }
    tcip_store.replace(curated_manifest_key(out), manifest)

    return {
        **counts,
        "subjects": sorted(subjects),
        "verdict_subjects": sorted(verdict_subjects),
        "unconfirmed_negatives": unconfirmed,
        "boundary_refused": boundary_refused,
        "subject": neg_subject,
        "output_dir": str(out),
        "structure": f"{out}/images/ + {out}/annotations/",
        "manifest": str(curated_manifest_path(out)),
    }


def reviewed_image_names(review_state: dict) -> set[str]:
    """Image names whose ``img_status == 'completed'`` (same predicate as is_image_reviewed)."""
    return {name for name, d in review_state.get("image", {}).items()
            if d.get("img_status") == "completed"}


def select_unreviewed(image_paths: list[str], reviewed_names: set[str]) -> list[str]:
    """``image_paths`` whose basename is not in ``reviewed_names`` (order-preserving)."""
    return [p for p in image_paths if os.path.basename(p) not in reviewed_names]

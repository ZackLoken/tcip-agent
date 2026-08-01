"""Materialize a curated detection dataset from human review verdicts (W5).

Torch-free. Turns review verdicts (per-image shards under ``.tcip/state/review/``) into training data:
  - accepted / edited GT boxes  -> positive name-based per-image JSON labels (the canonical format)
  - rejected-only images        -> confirmed-negative JSON (``{"annotations": []}``) backgrounds
plus a ``curated_manifest.json`` for provenance. The output layout (``images/`` + ``annotations/``)
matches ``data_tools._scan_dataset`` so the loop chains straight into ``make_splits`` /
``launch_training`` with no glue.

The verdict log stores normalized center-form boxes (``[cx, cy, w, h]``) plus the class *name*
(``class_name`` — an annotation's subject); positives are denormalized to pixel coordinates using the
copied image's dimensions (the canonical JSON is pixel-space) — no inference re-run.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef

_POSITIVE_ACTIONS = {"accepted", "edited"}


def partition_review_verdicts(review_state: dict, *, only_completed: bool = False) -> dict[str, dict]:
    """Partition per-image review verdicts into positives / hard-negatives / skip.

    Returns ``{img_name: {"positives": [(class_name, cx, cy, w, h)],
    "rejected_count": int, "status": "positive"|"hard_negative"|"skip"}}``.
    A detection's box is ``gt_bbox_norm or pred_bbox_norm`` (the fallback handles
    accepted-FP entries that carry only a predicted box); ``class_name`` is the subject.
    """
    result: dict[str, dict] = {}
    for img_name, img_data in review_state.get("image", {}).items():
        if only_completed and img_data.get("img_status") != "completed":
            continue
        positives: list[tuple] = []
        rejected = 0
        for entry in img_data.get("detections", []):
            action = entry.get("action")
            if action in _POSITIVE_ACTIONS:
                box = entry.get("gt_bbox_norm") or entry.get("pred_bbox_norm")
                if box and len(box) == 4:
                    cx, cy, w, h = (float(v) for v in box)
                    positives.append((str(entry.get("class_name", "")), cx, cy, w, h))
            elif action == "rejected":
                rejected += 1
        status = "positive" if positives else ("hard_negative" if rejected else "skip")
        result[img_name] = {"positives": positives, "rejected_count": rejected, "status": status}
    return result


def _find_source_image(source_images_dir: str, img_name: str) -> "Path | BandGroupRef | None":
    """The logical image ``img_name`` names — a plain ``Path``, or (when a ``.bandgroup``
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


def _write_positive_label(path: Path, positives: list[tuple], img_w: int, img_h: int) -> None:
    # Denormalize the verdict log's [cx,cy,w,h] to pixel xyxy for the name-based per-image JSON.
    anns = [
        Annotation(subject=name,
                   geometry=BBox((cx - w / 2) * img_w, (cy - h / 2) * img_h,
                                 (cx + w / 2) * img_w, (cy + h / 2) * img_h))
        for (name, cx, cy, w, h) in positives
    ]
    write_annotations(str(path), anns, img_w, img_h, keep_empty=True)


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
    omitted it is derived from the verdicts' own class names (single-subject reviews). ``producer_model``
    (best-effort) records the model whose predictions the human reviewed, for traceability.
    """
    partition = partition_review_verdicts(review_state, only_completed=only_completed)
    out = Path(output_dir)
    images_out = out / "images"
    labels_out = out / "annotations"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    place = shutil.copy2 if copy_files else os.symlink
    counts = {"positive": 0, "hard_negative": 0, "skipped": 0, "total_boxes": 0, "missing_images": 0}
    subjects: set[str] = set()
    manifest_images: list[dict] = []

    for img_name, info in partition.items():
        status = info["status"]
        if status == "skip" or (status == "hard_negative" and not include_hard_negatives):
            counts["skipped"] += 1
            continue
        src = _find_source_image(source_images_dir, img_name)
        if src is None:
            counts["missing_images"] += 1
            continue

        from tcip_mcp.pipelines.image_utils import BandGroupRef, image_dimensions

        if isinstance(src, BandGroupRef):
            # A grouped capture materializes as every sibling band file PLUS the .bandgroup
            # manifest itself — the manifest is what stands in for "the image" in the output
            # (its own filename is what every by-name reader treats as this capture's name).
            for band_path in src.bands.values():
                dst_band = images_out / band_path.name
                if not dst_band.exists():
                    place(str(band_path), str(dst_band))
            dst_manifest = images_out / src.manifest_path.name
            if not dst_manifest.exists():
                place(str(src.manifest_path), str(dst_manifest))
            record_name = src.manifest_path.name
            stem = src.stem
        else:
            dst_img = images_out / src.name
            if not dst_img.exists():
                place(str(src), str(dst_img))
            record_name = src.name
            stem = src.stem
        label_path = labels_out / f"{stem}.json"
        img_w, img_h = image_dimensions(src)

        if status == "positive":
            _write_positive_label(label_path, info["positives"], img_w, img_h)
            counts["positive"] += 1
            counts["total_boxes"] += len(info["positives"])
            subjects.update(name for (name, *_rest) in info["positives"])
        else:  # hard_negative -> confirmed-negative JSON ({"annotations": []})
            write_annotations(str(label_path), [], img_w, img_h, keep_empty=True)
            counts["hard_negative"] += 1

        manifest_images.append({
            "image": record_name, "status": status, "n_boxes": len(info["positives"]),
            "rejected_count": info["rejected_count"], "label": str(label_path),
        })

    # Hard negatives here come from explicit human rejection verdicts, so mark them in the output's
    # own status store under the review's subject — training only trusts human-confirmed negatives,
    # never bare empty files (someone may have emptied a label mid-work). The subject is threaded
    # (or the single subject the verdicts name); with none, negatives can't be attributed and are
    # left as unconfirmed empties rather than mis-keyed.
    negatives = {e["image"]: "negative" for e in manifest_images if e["status"] == "hard_negative"}
    neg_subject = subject or (next(iter(subjects)) if len(subjects) == 1 else None)
    if negatives and neg_subject:
        from tcip_mcp.dataset_layout import image_status_path, status_bucket

        status_file = image_status_path(out)
        status_file.parent.mkdir(parents=True, exist_ok=True)
        bucket_key = status_bucket(neg_subject, None)
        status_file.write_text(json.dumps({bucket_key: negatives}, indent=2))

        # Give the materialized dataset its own registry copy + a fresh per-image schema stamp, so
        # quarantine can actually protect these review-harvested negatives later — without this,
        # confirmed_negative_names has no classes.json to compare against and quarantine can never
        # fire here (a permanent no-op, not the "admit until proven stale" default it should be).
        from tcip_mcp.class_registry import attribute_schema_digest, read_registry
        from tcip_mcp.dataset_layout import classes_path, dataset_root_of, image_status_digest_path

        src_root = dataset_root_of(source_images_dir)
        src_classes = classes_path(src_root) if src_root is not None else None
        if src_classes is not None and src_classes.is_file():
            try:
                digest = attribute_schema_digest(read_registry(src_classes), neg_subject)
            except (OSError, ValueError):
                digest = None
            if digest is not None:
                shutil.copy2(src_classes, out / "classes.json")
                image_status_digest_path(out).write_text(json.dumps(
                    {bucket_key: {name: digest for name in negatives}}, indent=2))

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "review_state": review_state_path,
        "source_images_dir": str(source_images_dir),
        "output_dir": str(out),
        "producer_model": producer_model,
        "subject": neg_subject,
        "counts": counts,
        "subjects": sorted(subjects),
        "images": manifest_images,
    }
    (out / "curated_manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        **counts,
        "subjects": sorted(subjects),
        "subject": neg_subject,
        "output_dir": str(out),
        "structure": f"{out}/images/ + {out}/annotations/",
        "manifest": str(out / "curated_manifest.json"),
    }


def reviewed_image_names(review_state: dict) -> set[str]:
    """Image names whose ``img_status == 'completed'`` (same predicate as is_image_reviewed)."""
    return {name for name, d in review_state.get("image", {}).items()
            if d.get("img_status") == "completed"}


def select_unreviewed(image_paths: list[str], reviewed_names: set[str]) -> list[str]:
    """``image_paths`` whose basename is not in ``reviewed_names`` (order-preserving)."""
    return [p for p in image_paths if os.path.basename(p) not in reviewed_names]

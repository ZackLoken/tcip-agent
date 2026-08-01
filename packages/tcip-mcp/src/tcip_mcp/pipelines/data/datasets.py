"""Multi-task datasets with standardized interfaces.

Each dataset type returns (image_tensor, target_dict) where the target
format is task-specific but always dict-based. A factory function
`build_dataset` dispatches to the correct class by task type, or, for a
task the known loaders don't cover, to a bespoke ``dataset_source`` builder
the agent supplies (mirrors ``model_source``; see `build_from_dataset_source`).
"""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, BandGroupRef
from tcip_mcp.pipelines.image_utils import (
    IMAGE_EXTS, crop_pad_tile, image_dimensions, list_logical_images, load_image, pil_to_tensor,
    resolve_image_source,
)

logger = logging.getLogger(__name__)


def image_name_map(images_dir) -> dict[str, str]:
    """``{stem: real on-disk filename}`` from one directory listing (``list_logical_images``).

    A ``BandGroupRef``'s "name" is its own ``.bandgroup`` manifest's filename, the file that
    stands in for the grouped capture everywhere a name is matched against a store
    (``image_status.json``, a COCO ``file_name``), never one of its sibling band files.
    """
    result: dict[str, str] = {}
    for stem, src in list_logical_images(images_dir).items():
        result[stem] = src.manifest_path.name if isinstance(src, BandGroupRef) else src.name
    return result


def _authored_frame(stem: str, labels_dir, fmt: str, coco=None,
                    file_name: str = "") -> tuple[int, int] | None:
    """``(width, height)`` the labels record, or ``None`` when they record none.

    The frame the boxes were drawn in, straight from the annotation a human produced, the only
    reference that can catch a reader disagreeing with the authoring tool.
    """
    if coco is not None:
        images = coco.get("images", []) if isinstance(coco, dict) else []
        for entry in images:
            if entry.get("file_name") in (file_name, stem) or Path(
                    str(entry.get("file_name", ""))).stem == stem:
                w, h = int(entry.get("width", 0) or 0), int(entry.get("height", 0) or 0)
                return (w, h) if w > 0 and h > 0 else None
        return None
    if fmt not in ("", "json"):
        return None  # only the canonical per-image JSON carries its own frame
    path = Path(labels_dir) / f"{stem}.json"
    if not path.exists():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001, an unreadable label file is the reader's problem, not ours
        return None
    w, h = int(data.get("width", 0) or 0), int(data.get("height", 0) or 0)
    return (w, h) if w > 0 and h > 0 else None


def resolved_classes_path(dataset_dir) -> Path | None:
    """The real ``classes.json`` path for the dataset containing ``dataset_dir``, or ``None`` if it
    doesn't exist. The one fact ``_resolve_registry_id_map``'s attribute-without-registry refusal
    and any caller wanting to precheck it (``inference_tools.run_inference`` precondition-checks
    this before attempting resolution, so a legitimately absent registry degrades to an honest
    ``id_map=None`` instead of a caught-and-swallowed exception) both need, computed once, never
    two independent implementations of the same "does a registry exist" fact.
    """
    from tcip_mcp.dataset_layout import classes_path, dataset_root_of

    root = dataset_root_of(dataset_dir)
    cp = classes_path(root) if root is not None else None
    return Path(cp) if cp is not None and Path(cp).is_file() else None


def _resolve_registry_id_map(labels_dir, subject: str | None, attribute: str | None):
    """``(registry, id_map)`` for a training scope from the dataset's ``classes.json``.

    The single name→id derivation is :func:`class_registry.assign_class_ids`; the loader below,
    ``assemble_coco``, and the contract dims all read *this* map, never a second one. A plain
    single-class detector (``attribute`` is ``None``) needs no registry file, the subject *is* the
    class, so it is derived from a synthesized single-subject registry through the same
    ``assign_class_ids``, not a local ``{subject: 0}`` literal. Attribute classification needs the
    registry to order its values, and refuses when there is none.
    """
    from tcip_mcp import class_registry

    if not subject:
        raise ValueError(
            "a detection/instance_seg run needs an explicit subject to read name-based labels; "
            "none was threaded through build_dataset.")
    cp = resolved_classes_path(labels_dir)
    if cp is not None:
        registry = class_registry.read_registry(cp)
    elif attribute is not None:
        raise ValueError(
            f"attribute {attribute!r} classification needs a classes.json to order its values, "
            f"but none was found for {labels_dir}.")
    else:
        registry = class_registry.ClassRegistry(subjects=(class_registry.Subject(name=subject),))
    return registry, class_registry.assign_class_ids(registry, subject, attribute)


def _coco_det_targets(coco, file_name):
    """Pixel-xyxy boxes + 1-indexed labels for one image from an assembled COCO.

    ``category_id`` is the run's 0-indexed id (from ``to_coco_dataset`` over the run's id_map); +1
    applies the detector's background offset. The loader owns the +1, nothing on disk.
    """
    from tcip_annotation import format_io
    anns, _, _ = format_io._coco_image_annotations(coco, file_name=file_name)
    boxes, labels = [], []
    for a in anns:
        bb = a.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x, y, bw, bh = (float(v) for v in bb)
        boxes.append([x, y, x + bw, y + bh])
        labels.append(int(a.get("category_id", 0)) + 1)
    return boxes, labels


def _json_det_targets(path, subject, attribute, id_map):
    """``(boxes, labels, n_unlabeled)`` for one image from the name-based per-image JSON.

    Filters to ``subject`` + a box-derivable geometry, then maps each kept annotation to its
    0-indexed id via ``id_map`` (the single ``assign_class_ids`` map), +1 for background. An
    annotation the registry cannot decode raises, a real label read as nothing is a measurement bug.

    ``n_unlabeled`` counts instances of ``subject`` never assessed for ``attribute`` yet (a soft,
    expected gap, not a decode bug, excluded from ``boxes``/``labels`` rather than raising).
    Returning the count, not just silently dropping it, matters because an image with any
    unlabeled instance has incomplete ground truth for this scope, and a caller scoring/training
    only the labeled subset turns its real, unlabeled objects into silent false positives or
    background noise. Callers that build per-image records fresh each call (delivery
    evaluation, operating-point calibration) must exclude the whole image when ``n_unlabeled > 0``
   , the same precedent already applied to a missing label file, rather than partially trusting
    it; a caller bound to a fixed per-image dataset length (a `Dataset.__getitem__`) cannot do that
    without a deeper stems-selection change, and currently only surfaces the count.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Point, bbox_of

    boxes, labels = [], []
    n_unlabeled = 0
    for a in json_io.read_annotations(path):
        # allow_unlabeled=True: an instance never assessed for `attribute` yet is a soft, expected
        # gap, not a decode bug, must not raise and abort the whole read.
        cid = json_io.target_class_id(a, subject, attribute, id_map, allow_unlabeled=True)
        if cid == json_io.UNLABELED:
            n_unlabeled += 1
            continue
        if cid is None or a.geometry is None or isinstance(a.geometry, Point):
            continue
        box = bbox_of(a.geometry)
        boxes.append([box.x1, box.y1, box.x2, box.y2])
        labels.append(cid + 1)
    return boxes, labels, n_unlabeled


def dir_label_format(labels_dir) -> str | None:
    """``"json"`` if this dir holds canonical per-image labels, else ``None``.

    Used to route a JSON label store onto the COCO training path. A ``.json`` that is not our
    schema is not claimed, an unrecognized store must not be read as an all-empty one.
    """
    from tcip_annotation.json_io import ANNOTATIONS_KEY

    d = Path(labels_dir)
    if not d.is_dir():
        return None
    for jp in sorted(d.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        return "json" if isinstance(data, dict) and ANNOTATIONS_KEY in data else None
    return None


def trainable_stems(
    labels_dir, images_dir, stems=None, *, subject: str | None = None, date=None,
    coco: dict | None = None,
    attribute: str | None = None, id_map: dict[str, int] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """The stems that may train, plus the partition that produced them.

    A sample is admitted only when the label store actually accounts for it *for this subject*:

    - it has ≥1 annotation of ``subject``, or
    - it has none and a human marked that image negative for ``subject``
      (``confirmed_negative_names``, the Complete in ``.tcip/state/image_status.json``).

    An image with no label file, or an empty label file nobody confirmed, is **unannotated**, not a
    negative. Enumerating samples from ``images_dir`` instead served both as zero-box samples, so a
    project where the breeder labelled 30 of 400 images trained on 370 images asserted to be empty.

    Returns ``(stems, counts)`` where counts carries ``annotated`` / ``confirmed_negative`` /
    ``skipped_unannotated`` / ``skipped_unconfirmed_empty`` / ``skipped_incomplete_attribute`` /
    ``quarantined_stale_definition`` so a run can record what it dropped.
    ``quarantined_stale_definition`` is distinct from ``skipped_unconfirmed_empty``: it means
    a human did confirm the image negative, but the subject's attribute schema has since changed and
    the confirmation can no longer be trusted as-is, a different situation from nobody ever having
    looked, and one a reproduce-a-number chain must be able to tell apart (see
    ``confirmed_negative_names``'s quarantine logic).

    ``skipped_incomplete_attribute`` is the whole-image attribute-completeness rail: with
    ``attribute`` set, an image carrying any instance never assessed for it has incomplete ground
    truth for this scope and is dropped entirely, never trained on its labelled subset (which
    would leave its real, unlabelled objects to train as background). It lives here, in the one
    partition that already decides admission, rather than as a second filter over this function's
    output: a filter downstream cannot record *why* a stem left, and applying the rail there
    instead corrupts these counts outright. Both label paths reach the same verdict from one
    implementation:
    the COCO path reads the ``excluded_incomplete_attribute`` names ``to_coco_dataset`` already
    computed during assembly (never re-deriving them, and never mistaking that absence for
    "empty label file nobody confirmed"), and the direct-JSON path applies the rail through
    ``_json_det_targets``, the same reader the loader itself uses. ``attribute``/``id_map`` unset
    (every non-attribute run) applies no such rail.
    """
    names = image_name_map(images_dir)
    candidates = list(stems) if stems is not None else sorted(names)
    quarantined: set[str] = set()
    negatives = confirmed_negative_names(labels_dir, subject=subject, date=date,
                                         quarantined_out=quarantined)
    counts = {"annotated": 0, "confirmed_negative": 0, "skipped_unannotated": 0,
              "skipped_unconfirmed_empty": 0, "skipped_incomplete_attribute": 0,
              "quarantined_stale_definition": 0}

    coco_names: set[str] | None = None
    coco_annotated: set[str] = set()
    incomplete_names: set[str] = set()
    if coco is not None:
        # assemble_coco already applied this rail to build ``images``; intersecting with it *is*
        # the rail, so the two can never disagree about which samples exist.
        by_id = {e.get("id"): str(e.get("file_name", "")) for e in coco.get("images", [])}
        coco_names = set(by_id.values())
        coco_annotated = {by_id.get(a.get("image_id"), "") for a in coco.get("annotations", [])}
        # Which images to_coco_dataset dropped for attribute-incompleteness, read from its own
        # record rather than re-derived, their absence from ``images`` is otherwise
        # indistinguishable from an unconfirmed-empty one, and reporting that reason would be a lie.
        incomplete_names = {str(n) for n in coco.get("excluded_incomplete_attribute", [])}
    elif attribute is not None and id_map is not None:
        # Direct-JSON path: the same rail, through the same reader the loader uses.
        for stem in candidates:
            image_name = names.get(stem)
            if image_name is None:
                continue
            label_path = Path(labels_dir) / f"{stem}.json"
            if not label_path.is_file():
                continue
            _boxes, _labels, n_unlabeled = _json_det_targets(
                str(label_path), subject, attribute, id_map)
            if n_unlabeled:
                incomplete_names.add(image_name)

    keep: list[str] = []
    for stem in candidates:
        image_name = names.get(stem)
        if image_name is None:
            counts["skipped_unannotated"] += 1
            continue
        if image_name in incomplete_names:
            # Checked before every other verdict: an image with incomplete attribute GT is dropped
            # for that reason, not for whichever downstream category its absence happens to resemble.
            counts["skipped_incomplete_attribute"] += 1
            continue
        if coco_names is not None:
            if image_name not in coco_names:
                # assemble_coco already dropped it, but not why. A quarantined negative is dropped
                # the same way an unconfirmed one is (assemble_coco's confirmed_negative_names call
                # never sees a quarantined name as a negative), so it must be checked here too, or a
                # human-confirmed-but-schema-stale negative reads as "nobody ever looked" instead of
                # "looked, but the schema changed since", the exact distinction this count exists
                # to preserve, and the one the direct-JSON branch below already gets right.
                if image_name in quarantined:
                    counts["quarantined_stale_definition"] += 1
                    continue
                # Read the record so the operator is told the truth: "annotate this" and "confirm
                # this empty one" are different jobs.
                has_record, _ = _label_record_state(stem, labels_dir, subject)
                counts["skipped_unconfirmed_empty" if has_record
                       else "skipped_unannotated"] += 1
            elif image_name in coco_annotated:
                keep.append(stem)
                counts["annotated"] += 1
            elif image_name in negatives:
                # Zero annotations is a negative only with a human Complete. assemble_coco already
                # enforces that, but an externally supplied coco_json never went through it, so the
                # confirmation is re-checked here rather than inferred from the file's shape.
                keep.append(stem)
                counts["confirmed_negative"] += 1
            elif image_name in quarantined:
                counts["quarantined_stale_definition"] += 1
            else:
                counts["skipped_unconfirmed_empty"] += 1
            continue
        has_record, has_objects = _label_record_state(stem, labels_dir, subject)
        if not has_record:
            counts["skipped_unannotated"] += 1
        elif has_objects:
            keep.append(stem)
            counts["annotated"] += 1
        elif image_name in negatives:
            keep.append(stem)
            counts["confirmed_negative"] += 1
        elif image_name in quarantined:
            counts["quarantined_stale_definition"] += 1
        else:
            counts["skipped_unconfirmed_empty"] += 1
    return keep, counts


def _require_samples(stems: list[str], counts: dict[str, int], labels_dir) -> None:
    """Refuse an empty sample set, naming why each image was dropped.

    Filtering to the label store can legitimately empty a dataset, an images_dir where nothing is
    annotated yet. Building it anyway would train on nothing and report success.
    """
    if stems:
        return
    quarantined = counts.get("quarantined_stale_definition", 0)
    quarantine_note = (
        f" {quarantined} more were confirmed negative but quarantined because the subject's "
        f"attribute schema changed since, re-confirm them or revert the schema edit."
        if quarantined else ""
    )
    # Read defensively: this is the refusal path, and a counts dict missing a key here would
    # replace the explanation with a bare KeyError.
    incomplete = counts.get("skipped_incomplete_attribute", 0)
    incomplete_note = (
        f" {incomplete} more carry at least one instance never assessed for this run's attribute, "
        f"so their ground truth is incomplete for this scope and the whole image is held out "
        f"rather than trained on its labelled subset, finish attributing them, or run without "
        f"an attribute scope."
        if incomplete else ""
    )
    raise ValueError(
        f"no trainable samples in {labels_dir}: {counts.get('skipped_unannotated', 0)} image(s) "
        f"have no label record and {counts.get('skipped_unconfirmed_empty', 0)} have an empty one "
        f"nobody confirmed. An empty label file is a negative only once a human marks that image "
        f"Complete; until then it reads as unannotated. Annotate some images, or mark the "
        f"genuinely-empty ones Complete.{incomplete_note}{quarantine_note}"
    )


def _label_record_state(stem: str, labels_dir, subject: str | None) -> tuple[bool, bool]:
    """``(a record exists, it has ≥1 detection/seg target of ``subject``)`` for one stem.

    ``has_objects`` is subject-scoped *and box/polygon-bearing*: the unified file holds every subject,
    so "annotated" for a given subject's run means it carries an annotation of that subject whose
    geometry is a real detection/seg target, the same membership ``to_coco_dataset``/``target_class_id``
    apply (a box/polygon is a target; a geometry-less image-level label and a ``Point`` are not). Counting an
    image whose only annotations are non-targets as annotated would keep it on the direct-json path
    and train it as a zero-object negative, diverging from the COCO path and fabricating a negative no
    human confirmed.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Point

    path = Path(labels_dir) / f"{stem}.json"
    if not path.is_file():
        return False, False
    anns = json_io.read_annotations(str(path))

    def _is_target(a) -> bool:
        return a.geometry is not None and not isinstance(a.geometry, Point)

    if subject is None:
        return True, any(_is_target(a) for a in anns)
    return True, any(a.subject == subject and _is_target(a) for a in anns)


def confirmed_negative_names(
    labels_dir, *, subject: str | None, date=None, quarantined_out: set[str] | None = None,
) -> set[str]:
    """Image names a human marked negative (empty + Complete) **for this subject**.

    Reads the dataset-native ``image_status_path``, a sibling of ``classes.json``, so confirmations
    travel with the dataset rather than living in whichever project's private ``.tcip/`` happened to
    be an ancestor, and returns only the ``status_bucket(subject, date)`` bucket. A confirmation is
    a human's statement about one subject on one image; a store keyed by image name alone re-applies
    it to subjects they never looked at, so an image full of bushes trains as "contains no bushes".

    A negative is **quarantined**, excluded from the return value, only when the dataset's
    ``image_status_digest.json`` sidecar carries an explicit stamp for that image (not merely its
    bucket, a bucket holds every image ever touched under the subject/date, so a bucket-wide stamp
    would be silently overwritten by the next unrelated write and un-quarantine a stale confirmation
    nobody re-reviewed) and it no longer matches the subject's current
    :func:`~tcip_mcp.class_registry.attribute_schema_digest`: positive, provable evidence the
    subject's classification schema changed since that confirmation was made. Absence of a stamp,
    no sidecar, no stamp for that image, or a dataset that predates this mechanism entirely, is
    **not** quarantined: a rail must admit valid work, not only reject it, and treating "nobody
    stamped this yet" as "unverifiable, therefore invalid" would silently empty
    every pre-existing project's confirmed negatives. Pass ``quarantined_out`` (a set, mutated in
    place) to also learn which names were excluded, see :func:`trainable_stems`'s
    ``quarantined_stale_definition`` count.

    ``subject`` is threaded explicitly, the per-subject label dir it used to be recovered from is
    gone. When ``subject`` is unthreaded and the dataset holds confirmed negatives, this **refuses
    loudly** rather than returning nothing: a silent empty would drop every hard negative the review
    loop harvested. With no locatable dataset root, no store, or no confirmations for this subject,
    it returns an empty set.
    """
    from tcip_mcp.class_registry import attribute_schema_digest, read_registry
    from tcip_mcp.dataset_layout import (
        annotation_date, classes_path, dataset_root_of, image_status_digest_path,
        image_status_path, status_bucket,
    )

    if date is None:
        date = annotation_date(labels_dir)
    root = dataset_root_of(labels_dir)
    if root is None:
        return set()
    status_file = image_status_path(root)
    if not status_file.is_file():
        return set()
    try:
        statuses = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(statuses, dict):
        return set()
    if not subject:
        # Refuse only when there is something to lose: a store with confirmed negatives this
        # run might be entitled to. Silently returning none would drop the human's work.
        has_negatives = any(
            s == "negative" for b in statuses.values() if isinstance(b, dict)
            for s in b.values()
        )
        if not has_negatives:
            return set()
        raise ValueError(
            f"confirmed_negative_names needs an explicit subject to read the negative bucket "
            f"for {labels_dir}, and this dataset has human-confirmed negatives that would be "
            f"silently dropped. Thread the run's subject through build_dataset / assemble_coco."
        )
    bucket_key = status_bucket(subject, date)
    bucket = statuses.get(bucket_key)
    if not isinstance(bucket, dict):
        return set()  # this subject has no confirmations yet
    negatives = {name for name, s in bucket.items() if s == "negative"}
    if not negatives:
        return negatives

    stamped_by_image: dict = {}
    digest_file = image_status_digest_path(root)
    if digest_file.is_file():
        try:
            stamps = json.loads(digest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stamps = {}
        if isinstance(stamps, dict):
            bucket_stamps = stamps.get(bucket_key)
            if isinstance(bucket_stamps, dict):
                stamped_by_image = bucket_stamps
    if not stamped_by_image:
        return negatives  # nothing stamped in this bucket -> no signal -> admit (a rail admits valid work)

    current_digest = None
    cp = classes_path(root)
    if cp.is_file():
        try:
            current_digest = attribute_schema_digest(read_registry(cp), subject)
        except (OSError, ValueError):
            current_digest = None
    if current_digest is None:
        return negatives  # nothing current to compare against -> admit

    # Per-image, not per-bucket: a bucket holds every image ever touched under this subject/date, so
    # a later, unrelated write to the same bucket must never resurrect a different image's stale,
    # never-re-reviewed confirmation just because the bucket as a whole got re-stamped.
    trusted: set[str] = set()
    for name in negatives:
        stamped = stamped_by_image.get(name)
        if isinstance(stamped, str) and stamped != current_digest:
            if quarantined_out is not None:
                quarantined_out.add(name)
        else:
            trusted.add(name)
    return trusted


def assemble_coco(
    labels_dir, images_dir, stems=None, *, subject: str, attribute: str | None = None,
    id_map: dict[str, int], date=None,
) -> dict:
    """Assemble a dataset-level COCO dict from the name-based per-image JSON, scoped to ``subject``.

    Pairs each stem's ``<labels_dir>/<stem>.json`` with its image's on-disk file name, the same
    name the dataset resolves at read time, so the COCO ``file_name`` keys line up. ``id_map`` is
    the run's ``assign_class_ids`` map; this is the single delegation to ``json_io.to_coco_dataset``,
    so the COCO categories, the loader targets, and the contract dims all rest on one name→id map.
    Stems whose image is missing are skipped. This is how per-image JSON reaches training: a COCO the
    ``label_format='coco'`` path consumes.
    """
    from tcip_annotation import json_io

    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    if stems is None:
        stems = sorted(p.stem for p in labels_dir.glob("*.json"))
    # Real on-disk names: to_coco_dataset matches these against the confirmed-negative store, and a
    # constructed name would silently match nothing for an uppercase extension.
    names = image_name_map(images_dir)
    entries: list[tuple[str, str]] = []
    for stem in stems:
        file_name = names.get(stem)
        if file_name is None:
            continue
        entries.append((str(labels_dir / f"{stem}.json"), file_name))
    return json_io.to_coco_dataset(
        entries, subject=subject, id_map=id_map, attribute=attribute,
        confirmed_negative_names=confirmed_negative_names(labels_dir, subject=subject, date=date),
    )


class BaseDataset(Dataset, ABC):
    """Abstract base for all task-specific datasets."""

    task_type: str = ""
    expected_channels: int = 3  # input channels the dataset yields (3=RGB; set by build_dataset)

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def num_samples(self) -> int: ...

    @property
    def class_distribution(self) -> dict[int, int]:
        """Class ID → count. Subclasses should override for efficiency."""
        return {}

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]: ...

    def __len__(self) -> int:
        return self.num_samples


class BaseImageDataset(BaseDataset):
    """Base for image datasets, centralizes channel-aware loading + finalization.

    Subclasses set ``self.images_dir`` and ``self.transforms`` (and inherit
    ``expected_channels`` from build_dataset), then build only the task-specific target.
    """

    images_dir: Path
    transforms: Any = None

    def _resolve_path(self, stem: str) -> Path | BandGroupRef:
        """A ``stem`` may be a literal path (classification folder mode), a stem in images_dir, or
        (when a ``.bandgroup`` manifest groups sibling band files under it) a ``BandGroupRef``."""
        p = Path(stem)
        if p.is_absolute() or p.exists():
            return p
        return resolve_image_source(self.images_dir, stem)

    def _open_image(self, stem: str):
        """Open an image honoring ``expected_channels`` (PIL for 1/3/4 ch, else ndarray)."""
        return load_image(self._resolve_path(stem), self.expected_channels)

    @staticmethod
    def _image_size(img) -> tuple[int, int]:
        """Return ``(width, height)`` for a PIL image or an ``[H, W, C]`` array."""
        if isinstance(img, Image.Image):
            return img.size
        return int(img.shape[1]), int(img.shape[0])

    _warned_ndarray_transforms = False

    def _finalize(self, img, target: dict) -> tuple[torch.Tensor, dict]:
        """Apply PIL transforms (when applicable) or convert straight to a tensor."""
        if self.transforms is not None and isinstance(img, Image.Image):
            return self.transforms(img, target)
        if self.transforms is not None and not BaseImageDataset._warned_ndarray_transforms:
            # The augmentation pipeline is PIL-only, so a multi-band raster trains unaugmented
            # while the run's config records augmentation as applied. Say so once: a silent
            # divergence between what the config claims and what the model saw is a provenance
            # break, not a detail.
            BaseImageDataset._warned_ndarray_transforms = True
            logger.warning(
                "augmentation is configured but skipped for multi-band (ndarray) images: the "
                "transform pipeline is PIL-only. This run trains those images unaugmented."
            )
        return pil_to_tensor(img), target


# ====================================================================
# Detection
# ====================================================================

class DetectionDataset(BaseImageDataset):
    """Object detection. ``label_format`` selects the on-disk label format:

    - ``json`` (default): canonical per-image ``<labels_dir>/<stem>.json`` (json_io schema)
    - ``coco``: a single COCO JSON at ``coco_json``, the assembled dataset view of the per-image
      JSON, used for training (annotations matched by file name)
    """

    task_type = "detection"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
        label_format: str = "json",
        coco_json: str | None = None,
        coco_data: dict | None = None,
        subject: str | None = None,
        attribute: str | None = None,
        id_map: dict[str, int] | None = None,
        date=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.subject = subject
        self.attribute = attribute
        self.label_format = (label_format or "json").lower()
        self._coco = None
        if coco_data is not None:  # in-memory COCO assembled from per-image JSON (train/eval)
            self._coco = coco_data
            self.label_format = "coco"
        elif self.label_format == "coco":
            if not coco_json:
                raise ValueError("label_format='coco' requires coco_json (path to the COCO JSON).")
            self._coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        # The single name→id map: resolved here for a direct-json build, else supplied by
        # build_dataset (which resolved it once for the COCO assembly). One derivation either way.
        if id_map is None and self.label_format == "json":
            _reg, id_map = _resolve_registry_id_map(self.labels_dir, subject, attribute)
            self._num_classes = len(id_map)
        self.id_map = id_map
        # The attribute-completeness rail (an image with any instance never assessed for `attribute`
        # is held out whole, never trained on its labelled subset) lives inside trainable_stems, the
        # one partition that already decides admission and records why, see its docstring. A second
        # filter over trainable_stems' *output* would both corrupt those counts and never run at all
        # on the real build_dataset path (which assembles COCO and so takes the
        # `label_format == "coco"` branch). Applying it at the partition covers both label paths
        # from one implementation and reports a truthful reason for each drop.
        self.stems, self.sample_counts = trainable_stems(
            self.labels_dir, self.images_dir, stems,
            subject=subject, date=date, coco=self._coco,
            attribute=attribute, id_map=self.id_map,
        )
        _require_samples(self.stems, self.sample_counts, self.labels_dir)
        # Real on-disk filenames, for matching a stem to the COCO's ``file_name`` (which carries the
        # true name), image_name_map reads the actual directory listing, never a constructed guess.
        self._image_names = image_name_map(self.images_dir)

    def _det_targets(self, stem: str, file_name: str) -> tuple[list, list]:
        """Pixel-xyxy boxes + 1-indexed labels for one image (coco or name-based json).

        ``self.stems`` already excludes any image with an instance unlabeled for ``attribute``
        (``trainable_stems``' ``skipped_incomplete_attribute`` rail, a fixed-length dataset can't
        act on this per-``__getitem__`` call, only once, up front), so ``n_unlabeled`` is always 0
        here by construction; the 3-tuple is unpacked for the shared ``_json_det_targets``
        signature, not because a nonzero count is expected at this point.
        """
        if self.label_format == "coco":
            return _coco_det_targets(self._coco, file_name)
        boxes, labels, _n_unlabeled = _json_det_targets(
            str(self.labels_dir / f"{stem}.json"), self.subject, self.attribute, self.id_map)
        return boxes, labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        if self.label_format == "coco" and self._coco:
            # self._coco may be shared across a full/train/val split trio (assembled once in
            # training_tools.py, threaded into all three builds rather than re-assembled per
            # split), its annotations cover the whole dataset, not just this dataset's own
            # self.stems, so every consumer must filter to its own image set or a split's
            # class_distribution reports the identical, unsplit whole for train and val alike.
            own_names = {self._image_names.get(s, "") for s in self.stems}
            image_names_by_id = {e.get("id"): str(e.get("file_name", ""))
                                 for e in self._coco.get("images", [])}
            for ann in self._coco.get("annotations", []):
                if image_names_by_id.get(ann.get("image_id")) in own_names:
                    counts[ann.get("category_id", 0)] += 1
        else:  # json: parse each image's annotation
            for stem in self.stems:
                _, labels = self._det_targets(stem, "")
                for lab in labels:
                    counts[lab - 1] += 1  # back to 0-indexed cid
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)
        file_name = self._image_names.get(stem, "") if self.label_format == "coco" else ""
        boxes, labels = self._det_targets(stem, file_name)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": idx,
        }
        return self._finalize(img, target)


# ====================================================================
# Tiled Detection (SAHI-style sliding window)
# ====================================================================

class TiledDetectionDataset(BaseImageDataset):
    """Wrap a ``DetectionDataset`` and expand each source image into native-resolution
    tiles with labels clipped/remapped to tile space.

    Tile membership is computed at ``__init__`` (header-only image sizes + the YOLO
    txt) so the dataset can return one sample per tile index. ``__getitem__`` decodes
    the source image once, crops the tile, zero-pads border crops to ``tile_size``,
    and emits the same target dict shape as ``DetectionDataset``.
    """

    task_type = "detection"

    def __init__(
        self,
        base: "DetectionDataset",
        tile_size: int = 224,
        overlap: float = 0.2,
        sliver_frac: float | None = None,
        dedup_iou: float = 0.8,
        skip_empty: bool = False,
        transforms: Any = None,
    ) -> None:
        from tcip_mcp.pipelines.data.tiling import (
            compute_stride, tile_positions, clip_boxes_to_tile, dedup_boxes,
        )
        from tcip_mcp.pipelines.derivations import derive_sliver_frac

        self.base = base
        # This wrapper does its own channel-aware reads rather than delegating to base. Inherit the
        # band count from the dataset being wrapped so every construction path carries it,
        # build_dataset stamps it afterwards, but ctx.tiled_dataset constructs this directly and
        # would otherwise fall back to the 3-channel class default.
        self.images_dir = base.images_dir
        self.expected_channels = getattr(base, "expected_channels", 3)
        self.tile_size = tile_size
        self.overlap = overlap
        self.transforms = transforms
        self.stride = compute_stride(tile_size, overlap)
        self._index: list[dict] = []
        # (w, h) this index was built against, per stem, asserted again at decode time.
        self._decoded_frame: dict[str, tuple[int, int]] = {}

        # Pass 1: read every image's upright dims + full-image-px boxes, and accumulate GT box sizes
        # so the seam-sliver cutoff is derived from this dataset's class-average object size, not a
        # fixed fraction (Q5 / derive-don't-pin). skip_empty defaults False: empty tiles are valid
        # negatives (the invariant the old skip_empty=True default violated).
        stems_data: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
        char_sizes: list[float] = []
        for stem in base.stems:
            img_source = resolve_image_source(base.images_dir, stem)
            # Measured the way __getitem__ will decode it: PIL's header read misreports a
            # multi-band raster's axes, which would clip labels in a frame the tiles never use.
            w, h = image_dimensions(img_source, self.expected_channels)
            # The frame the boxes were actually drawn in, recorded in the label file itself. The
            # annotation stack measures with PIL, which reports a 40x24x5 GeoTIFF as 5x40, so on a
            # multi-band raster the authored frame and the decoded frame genuinely disagree, and
            # every box would be cropped from somewhere it was never drawn. Comparing the two
            # decoders instead would prove nothing: they share a branch and agree by construction.
            authored = _authored_frame(stem, base.labels_dir, base.label_format,
                                       base._coco, base._image_names.get(stem, ""))
            if authored is not None and authored != (w, h):
                raise ValueError(
                    f"tiled dataset frame mismatch for stem {stem!r}: the labels record a "
                    f"{authored[0]}x{authored[1]} image but it decodes as {w}x{h} at "
                    f"{self.expected_channels} channels. Tiles would be cut from a different frame "
                    f"than the boxes were drawn in, displacing every box. Re-author the labels "
                    f"against the multi-band frame, or ingest this raster as {authored[0]}x"
                    f"{authored[1]}."
                )
            # Format-aware read via the base dataset's own targeting (json/coco share one path);
            # only coco needs the image file name to match its annotations. Use the real on-disk name
            # (img_path.name can be miscased on Windows), or the coco match silently finds nothing.
            file_name = base._image_names.get(stem, "") if base.label_format == "coco" else ""
            full_boxes, full_labels = base._det_targets(stem, file_name)
            fb = np.asarray(full_boxes, dtype=np.float32).reshape(-1, 4)
            fl = np.asarray(full_labels, dtype=np.int64)
            if len(fb):
                char_sizes.extend((((fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1])).clip(min=0) ** 0.5).tolist())
            stems_data.append((stem, fb, fl, w, h))
            self._decoded_frame[stem] = (int(w), int(h))

        self.class_avg_size = float(np.mean(char_sizes)) if char_sizes else 0.0
        # A caller-supplied fraction wins; otherwise derive it from this dataset's own size spread
        # (a class with wide natural size variation needs a lower cutoff than a tightly-sized one,
        # a fixed fraction can't tell a genuinely small-but-complete object from a real tile-seam
        # fragment). Falls back to 0.5 only when the spread itself is underivable (too few boxes to
        # measure a spread from, or none at all).
        if sliver_frac is None:
            sliver_frac = derive_sliver_frac(char_sizes)
            self.sliver_frac_source = (
                "GT characteristic-size spread (p10 / mean)" if sliver_frac is not None
                else "documented default (underivable: too few GT boxes to measure a spread)")
            if sliver_frac is None:
                sliver_frac = 0.5
        else:
            self.sliver_frac_source = "explicit"
        self.sliver_frac = sliver_frac
        self.min_box_size = sliver_frac * self.class_avg_size

        # Pass 2: tile using the derived sliver cutoff.
        for stem, fb, fl, w, h in stems_data:
            for tile_x, tile_y in tile_positions(h, w, tile_size, self.stride):
                tb, tl = clip_boxes_to_tile(fb, fl, tile_x, tile_y, tile_size, self.min_box_size)
                if len(tb) > 1:
                    tb, tl = dedup_boxes(tb, tl, dedup_iou)
                if skip_empty and len(tb) == 0:
                    continue
                self._index.append({"stem": stem, "tile_x": tile_x, "tile_y": tile_y, "boxes": tb, "labels": tl})

    @property
    def num_classes(self) -> int:
        return self.base.num_classes

    @property
    def num_samples(self) -> int:
        return len(self._index)

    @property
    def stems(self) -> list[str]:
        return [e["stem"] for e in self._index]

    @property
    def class_distribution(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for e in self._index:
            for lab in e["labels"].tolist():
                counts[int(lab) - 1] += 1  # 0-indexed cid, matching DetectionDataset
        return dict(counts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        e = self._index[idx]
        stem = e["stem"]
        # Channel-aware and EXIF-oriented (via load_image) so cropped pixels align with the tile
        # geometry and the labels clipped in __init__.
        img = self._open_image(stem)
        w, h = self._image_size(img)
        expected = self._decoded_frame.get(stem)
        if expected is not None and (w, h) != expected:
            # The index was built against this frame; if the file now decodes differently the tile
            # would be cut from somewhere the boxes were never clipped to. Refuse, don't reconcile.
            raise ValueError(
                f"tiled dataset frame changed for stem {stem!r}: indexed at "
                f"{expected[0]}x{expected[1]} but now decodes as {w}x{h} at "
                f"{self.expected_channels} channels. Cropping here would displace every box."
            )
        tile = crop_pad_tile(img, e["tile_x"], e["tile_y"], self.tile_size, w, h)
        target = {
            "boxes": torch.tensor(e["boxes"], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(e["labels"], dtype=torch.int64),
            "image_id": idx,
        }
        return self._finalize(tile, target)


# ====================================================================
# Instance Segmentation
# ====================================================================

class InstanceSegDataset(BaseImageDataset):
    """Instance masks from per-image polygons: canonical per-image JSON ``<stem>.json`` (default),
    or a COCO dict / assembled per-image JSON (``label_format='coco'`` / ``coco_data``)."""

    task_type = "instance_seg"

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 1,
        label_format: str = "json",
        coco_json: str | None = None,
        coco_data: dict | None = None,
        subject: str | None = None,
        attribute: str | None = None,
        id_map: dict[str, int] | None = None,
        date=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        self.subject = subject
        self.attribute = attribute
        self.label_format = (label_format or "json").lower()
        self._coco = None
        if coco_data is not None:
            self._coco = coco_data
            self.label_format = "coco"
        elif self.label_format == "coco":
            if not coco_json:
                raise ValueError("label_format='coco' requires coco_json (path to the COCO JSON).")
            self._coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        if id_map is None and self.label_format == "json":
            _reg, id_map = _resolve_registry_id_map(self.labels_dir, subject, attribute)
            self._num_classes = len(id_map)
        self.id_map = id_map
        # attribute/id_map must be threaded through: without them the direct-JSON instance_seg path
        # has no attribute-completeness rail at all, an image with any instance never assessed for
        # `attribute` trains on its labeled subset instead of being held out whole, the same gap
        # DetectionDataset's own call already closes.
        self.stems, self.sample_counts = trainable_stems(
            self.labels_dir, self.images_dir, stems,
            subject=subject, date=date, coco=self._coco,
            attribute=attribute, id_map=self.id_map,
        )
        _require_samples(self.stems, self.sample_counts, self.labels_dir)
        # Real on-disk filenames for the COCO ``file_name`` match (see DetectionDataset / image_name_map).
        self._image_names = image_name_map(self.images_dir)

    def _read_polys(self, stem: str, w: int, h: int) -> list[tuple[list[list[tuple[float, float]]], int]]:
        """(pixel polygon rings, 1-indexed label) per instance, from the assembled COCO or the
        name-based per-image ``<stem>.json`` (filtered to ``subject`` + polygon geometry). Both are
        already pixel-space; the +1 background offset is the loader's, nothing on disk carries it.
        An instance's rings is a list, an occlusion-split instance (a leaf crossed by a stem) is
        genuinely more than one ring; ``__getitem__`` rasterizes every ring of an instance into
        that instance's one mask."""
        out: list[tuple[list[list[tuple[float, float]]], int]] = []
        if self.label_format == "coco":
            from tcip_annotation import format_io
            file_name = self._image_names.get(stem, "")
            anns, _, _ = format_io._coco_image_annotations(self._coco, file_name=file_name)
            for a in anns:
                seg = a.get("segmentation")
                if not (isinstance(seg, list) and seg):
                    continue
                rings = []
                for coords in seg:
                    if not (isinstance(coords, list) and len(coords) >= 6):
                        continue
                    rings.append([(float(coords[i]), float(coords[i + 1]))
                                 for i in range(0, len(coords) - 1, 2)])
                if not rings:
                    continue
                out.append((rings, int(a.get("category_id", 0)) + 1))
            return out
        from tcip_annotation import json_io
        from tcip_annotation.state import Polygon
        for a in json_io.read_annotations(str(self.labels_dir / f"{stem}.json")):
            if a.subject != self.subject or not isinstance(a.geometry, Polygon):
                continue
            key = a.attributes.get(self.attribute) if self.attribute else self.subject
            if key is None or self.id_map is None or key not in self.id_map:
                raise ValueError(
                    f"annotation of subject {self.subject!r} has class key {key!r} not in the run's "
                    f"id map, the registry cannot decode its own labels")
            out.append(([list(ring) for ring in a.geometry.rings], self.id_map[key] + 1))
        return out

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)

        boxes, labels, masks = [], [], []
        for rings, lab in self._read_polys(stem, w, h):
            all_pts = [p for ring in rings for p in ring]
            if not all_pts:
                continue
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(lab)

            # Rasterize every ring into the same instance mask, a multi-ring instance is one
            # occlusion-split object, not several separate ones; ImageDraw fills union naturally
            # since a pixel already painted 1 stays 1.
            mask = np.zeros((h, w), dtype=np.uint8)
            try:
                from PIL import ImageDraw
                poly_img = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(poly_img)
                for ring in rings:
                    if len(ring) >= 3:
                        draw.polygon([(p[0], p[1]) for p in ring], fill=1)
                mask = np.array(poly_img)
            except Exception:
                pass
            masks.append(mask)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks) if masks else np.zeros((0, h, w)), dtype=torch.uint8),
            "image_id": idx,
        }
        return self._finalize(img, target)


# ====================================================================
# Semantic Segmentation
# ====================================================================

class SemanticSegDataset(BaseImageDataset):
    """PNG mask images where pixel values are class IDs."""

    task_type = "semantic_seg"

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        stems: list[str] | None = None,
        transforms: Any = None,
        num_classes: int = 2,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        # A sample needs a mask. Unlike detection there is no unconfirmed-empty case: an
        # all-background mask is an explicit annotation, so existence is the whole rail here.
        # Serving an image with no mask would train it as entirely background, a fabricated
        # negative by another route.
        mask_stems = {p.stem for p in self.masks_dir.iterdir()} if self.masks_dir.is_dir() else set()
        candidates = stems or sorted(image_name_map(self.images_dir))
        self.stems = [s for s in candidates if s in mask_stems]
        self.sample_counts = {"annotated": len(self.stems),
                              "skipped_unannotated": len(candidates) - len(self.stems)}
        if not self.stems:
            raise ValueError(
                f"no trainable samples: none of the {len(candidates)} image(s) in "
                f"{self.images_dir} have a mask in {self.masks_dir}. An image with no mask would "
                f"train as entirely background."
            )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self.stems[idx]
        img = self._open_image(stem)
        w, h = self._image_size(img)
        mask_path = self.masks_dir / f"{stem}.png"
        # load_image EXIF-orients so the mask shares the image's upright frame (no-op for a
        # plain PNG mask; matters only if a mask ever carries EXIF orientation).
        mask = np.array(load_image(mask_path, 1)) if mask_path.exists() else np.zeros((h, w), dtype=np.int64)
        # Key matches the SemanticSegHead loss contract.
        target = {"masks": torch.tensor(mask, dtype=torch.int64)}
        return self._finalize(img, target)


# ====================================================================
# Classification
# ====================================================================

class ClassificationDataset(BaseImageDataset):
    """Image classification from CSV (image_stem, label) or folder structure."""

    task_type = "classification"

    def __init__(
        self,
        images_dir: str,
        csv_path: str | None = None,
        stems: list[str] | None = None,
        labels: list[int] | None = None,
        transforms: Any = None,
        num_classes: int = 2,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._num_classes = num_classes
        if csv_path is not None:
            self._stems, self._labels = self._load_csv(csv_path)
        elif stems is not None and labels is not None:
            self._stems = stems
            self._labels = labels
        else:
            # Folder-based: images_dir/<class_name>/<image>
            self._stems, self._labels = self._load_folder_structure()

    def _load_csv(self, path: str) -> tuple[list[str], list[int]]:
        stems, labels = [], []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    stems.append(row[0].strip())
                    labels.append(int(row[1].strip()))
        return stems, labels

    def _load_folder_structure(self) -> tuple[list[str], list[int]]:
        stems, labels = [], []
        class_dirs = sorted(d for d in self.images_dir.iterdir() if d.is_dir())
        for cid, cdir in enumerate(class_dirs):
            for f in cdir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    stems.append(str(f))
                    labels.append(cid)
        return stems, labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._labels))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = self._open_image(stem)
        target = {"labels": self._labels[idx]}
        return self._finalize(img, target)


# ====================================================================
# Ordinal
# ====================================================================

class OrdinalDataset(BaseImageDataset):
    """Ordinal regression from CSV (image_stem, rank). E.g., disease severity 0-4."""

    task_type = "ordinal"

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transforms: Any = None,
        num_ranks: int = 5,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._num_ranks = num_ranks
        self._stems: list[str] = []
        self._ranks: list[int] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self._stems.append(row[0].strip())
                    self._ranks.append(int(row[1].strip()))

    @property
    def num_classes(self) -> int:
        return self._num_ranks

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    @property
    def class_distribution(self) -> dict[int, int]:
        return dict(Counter(self._ranks))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = self._open_image(stem)
        # Key matches the OrdinalHead loss contract (plural, like "labels"/"masks").
        target = {"ranks": self._ranks[idx], "num_ranks": self._num_ranks}
        return self._finalize(img, target)


# ====================================================================
# Regression
# ====================================================================

class RegressionDataset(BaseImageDataset):
    """Continuous-value regression from CSV (image_stem, value)."""

    task_type = "regression"

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transforms: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self._stems: list[str] = []
        self._values: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    self._stems.append(row[0].strip())
                    self._values.append(float(row[1].strip()))

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def num_samples(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        stem = self._stems[idx]
        img = self._open_image(stem)
        # Key matches the RegressionHead loss contract.
        target = {"values": self._values[idx]}
        return self._finalize(img, target)


# ====================================================================
# Factory
# ====================================================================

_DATASET_MAP = {
    "detection": DetectionDataset,
    "instance_seg": InstanceSegDataset,
    "semantic_seg": SemanticSegDataset,
    "classification": ClassificationDataset,
    "ordinal": OrdinalDataset,
    "regression": RegressionDataset,
}

DATASET_SOURCE_KEY = "dataset_source"


def build_from_dataset_source(dataset_source: dict, **kwargs: Any) -> Dataset:
    """Import the agent's dataset builder and call it, the bespoke-task escape (mirrors
    ``build_from_model_source``). Registry-free, no ``exec``: the builder is imported like any
    module. It receives the run's data context (``images_dir`` / ``labels_dir`` / ``stems`` /
    ``transforms`` / ``task``, whatever ``build_dataset`` was given) merged with its own
    ``builder_kwargs`` (which win on conflict), and must return a torch ``Dataset``. Declare
    ``**kwargs`` on the builder to ignore context keys it doesn't use.

    ``dataset_source`` schema (parallels ``model_source``)::

        {"builder": "my_module:build_ds",  # required, 'module:function' (or 'module.function')
         "builder_kwargs": {...},          # optional, passed to the builder (win on conflict)
         "source_files": [...],            # optional, provenance (snapshot_model_source copies these)
         "task": "..."}                    # optional, measurement/eval routing
    """
    if not isinstance(dataset_source, dict):
        raise ValueError("dataset_source must be a dict")
    from tcip_mcp.pipelines.model_build import _import_dotted

    fn = _import_dotted(dataset_source.get("builder"))
    builder_kwargs = dataset_source.get("builder_kwargs") or {}
    if not isinstance(builder_kwargs, dict):
        raise ValueError("dataset_source.builder_kwargs must be a dict")
    return fn(**{**kwargs, **builder_kwargs})


def _autoresolve_json_labels(kwargs: dict, *, subject: str, attribute: str | None,
                             id_map: dict[str, int]) -> None:
    """Route a name-based per-image-JSON label dir onto the assembled-COCO path for training/eval.

    No-op when the caller pinned a format or already supplied COCO data. The single ``id_map`` is
    threaded into ``assemble_coco`` (and thus the one ``to_coco_dataset`` call), so the assembled
    categories rest on the same name→id derivation as the loader targets and the contract dims.
    """
    if kwargs.get("coco_data") is not None or kwargs.get("coco_json") or kwargs.get("label_format"):
        return
    labels_dir = kwargs.get("labels_dir", "")
    images_dir = kwargs.get("images_dir", "")
    if not labels_dir:
        return
    if dir_label_format(labels_dir) == "json" and images_dir:
        kwargs["coco_data"] = assemble_coco(
            labels_dir, images_dir, stems=kwargs.get("stems"),
            subject=subject, attribute=attribute, id_map=id_map)
        kwargs["label_format"] = "coco"


def _probe_num_channels(images_dir: str | Path | None, stems: list[str] | None,
                        default: int = 3) -> int:
    """Band count of one sample raster from ``images_dir`` (derive-don't-pin, not a pinned 3).

    Probes a single image (guard: one sample, not every image) so a multi-band raster threads its
    real channel count through ``in_chans`` instead of silently defaulting to RGB. Falls back to
    ``default`` only when no readable raster is found at all, or a genuinely unexpected decode error
    hits it, never for a stale ``.bandgroup`` manifest (``BandGroupIncomplete`` propagates loudly
    instead), since a confidently-wrong channel count silently sizes the model wrong for every
    dataset that hits it.
    """
    if not images_dir:
        return default
    images_dir = Path(images_dir)
    sample: Path | BandGroupRef | None = None
    for stem in (stems or []):
        try:
            sample = resolve_image_source(images_dir, stem)
            break
        except FileNotFoundError:
            # Per-stem skip-and-try-the-next-one (BandGroupIncomplete included): with multiple
            # candidate stems, one stale/missing entry doesn't preclude probing a different, intact
            # one, only the single-sample fallback below has no "next stem" to fall back to.
            continue
    if sample is None:
        logical = list_logical_images(images_dir)
        if logical:
            # Through resolve_image_source (not a bare dict pick): its completeness check is what
            # turns a stale manifest into a named BandGroupIncomplete here, rather than a bare
            # decode error surfacing later inside probe_channels.
            sample = resolve_image_source(images_dir, sorted(logical)[0])
    if sample is None:
        return default
    from tcip_mcp.pipelines.derivations import probe_channels

    try:
        return int(probe_channels(sample))
    except BandGroupIncomplete:
        raise
    except Exception:
        return default


def build_dataset(task: str, dataset_source: dict | None = None, **kwargs) -> Dataset:
    """Factory: build a dataset by task type, or via a bespoke ``dataset_source`` builder.

    An optional ``tiling`` dict (``{enabled, tile_size, overlap, sliver_frac,
    dedup_iou, skip_empty}``) wraps the detection dataset in a
    :class:`TiledDetectionDataset`. Ignored for non-detection tasks.

    ``num_channels`` is derived by probing one sample raster when the caller does not pin it, so a
    multi-band input threads its real band count through ``in_chans`` instead of defaulting to RGB.

    ``dataset_source`` is the bespoke seam (mirrors ``model_source``): when given, an agent-supplied
    importable builder produces the dataset for a task the known loaders don't cover. The known
    loaders stay the default; the ``Unknown task`` error below is still raised for a bad known-task
    name (an honest typo signal), the seam is the escape for a genuinely new task.
    """
    tiling = kwargs.pop("tiling", None)
    num_channels = kwargs.pop("num_channels", None)
    if num_channels is None:
        num_channels = _probe_num_channels(kwargs.get("images_dir"), kwargs.get("stems"))

    if dataset_source is not None:
        ds = build_from_dataset_source(dataset_source, task=task, **kwargs)
        if getattr(ds, "expected_channels", None) is None:
            ds.expected_channels = num_channels
        return ds

    cls = _DATASET_MAP.get(task)
    if cls is None:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_DATASET_MAP.keys())}")

    if task in ("detection", "instance_seg"):
        subject = kwargs.get("subject")
        attribute = kwargs.get("attribute")
        has_coco = (kwargs.get("coco_data") is not None or kwargs.get("coco_json")
                    or (kwargs.get("label_format") or "").lower() == "coco")
        if not has_coco and kwargs.get("labels_dir"):
            # Name-based json: resolve the single id map once, set num_classes and assemble the COCO
            # from it, the loader, the categories, and resolve_contract_dims all read this one map.
            _registry, id_map = _resolve_registry_id_map(kwargs["labels_dir"], subject, attribute)
            kwargs["id_map"] = id_map
            kwargs["num_classes"] = len(id_map)
            _autoresolve_json_labels(kwargs, subject=subject, attribute=attribute, id_map=id_map)
        elif has_coco and kwargs.get("num_classes") is None:
            coco = kwargs.get("coco_data")
            if isinstance(coco, dict):
                kwargs["num_classes"] = len(coco.get("categories") or []) or 1

    if tiling and tiling.get("enabled", True) and task == "detection":
        transforms = kwargs.pop("transforms", None)
        base = cls(**kwargs)
        # Before constructing the tiler: its __init__ indexes every image, and that pass must
        # measure frames at the band count the tiles will be decoded at. Stamping only the wrapper
        # afterwards left the index built at 3 channels and the tiles read at N.
        base.expected_channels = num_channels
        tile_kwargs = {k: tiling[k] for k in
                       ("tile_size", "overlap", "sliver_frac", "dedup_iou", "skip_empty")
                       if k in tiling}
        ds = TiledDetectionDataset(base, transforms=transforms, **tile_kwargs)
    else:
        if tiling and tiling.get("enabled", True) and task != "detection":
            logger.warning("tiling is only supported for task='detection'; ignoring for task=%r", task)
        ds = cls(**kwargs)

    ds.expected_channels = num_channels
    return ds

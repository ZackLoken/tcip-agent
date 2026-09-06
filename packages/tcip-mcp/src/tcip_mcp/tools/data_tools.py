"""Data management tools: census a dataset, split data. Per-file quality checks live in
the ``doctor`` command's ``check_data_quality``, the retired per-file quality tool folded in there."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import tcip_store
from tcip_store import RECORD_JSON, Key, StoreDescriptor, register_store
from tcip_store.file_backend import RootedFileLocator

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited

_SPLIT_DOC = RootedFileLocator(suffix=".json")
"""A split output directory's own documents. The directory is wherever the caller asked the
partition to be written, so no dataset resolver owns its layout and the entries are addressed
by name under it, the way a label tree no resolver describes is addressed."""

SPLIT_MANIFEST_STORE = "split_manifest"
_SPLIT_MANIFEST_PARTS = ("split_manifest",)
register_store(
    StoreDescriptor(
        name=SPLIT_MANIFEST_STORE,
        kind="record",
        key_fields=("document",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_SPLIT_DOC,
    )
)


def split_manifest_key(split_dir: str | Path) -> Key:
    """How a split directory's partition was produced, so it can be reconstructed.

    ``last_writer_wins``: written once, whole, at the end of the partition that produced it.
    """
    return Key(SPLIT_MANIFEST_STORE, str(Path(split_dir).absolute()), _SPLIT_MANIFEST_PARTS)


_SPLIT_MANIFEST_REQUIRED_KEYS = (
    "seed", "group_by", "dataset_fingerprint", "subject", "attribute", "id_map",
    "members", "splits", "admission_counts", "calibration_foreground_groups_by_date",
    "realized_ratios",
)
"""Every key ``draw_splits``' manifest dict literal writes, kept as one tuple so
:func:`read_split_manifest_dir`'s required set can never drift from what the writer actually
writes: a manifest missing one of them is refused there, before any caller binds to it."""


def _read_split_manifest_dir_or_none(split_dir: str | Path) -> dict | None:
    """The validated ``split_manifest`` record under ``split_dir``, or ``None`` when nothing is
    recorded there. Raises ``ValueError`` for a record that exists but will not decode, is not a
    mapping, or fails one of the required-key/shape checks; lets
    :class:`tcip_store.SchemaVersionRefused` propagate uncaught. The one validation
    :func:`read_split_manifest_dir` and :func:`read_split_manifest_dir_checked` both build on, so
    neither restates the shape a manifest must carry.
    """
    from tcip_store import DecodeError

    from tcip_mcp.pipelines.data.splits import SPLIT_NAMES

    try:
        manifest = tcip_store.read(split_manifest_key(split_dir), default=None)
    except DecodeError as exc:
        raise ValueError(f"the split manifest at {split_dir} could not be read: {exc}") from exc
    if manifest is None:
        return None
    if not isinstance(manifest, dict):
        raise ValueError(
            f"the split manifest at {split_dir} decodes to a {type(manifest).__name__}, not the "
            "mapping draw_splits writes."
        )
    missing = [k for k in _SPLIT_MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(
            f"the split manifest at {split_dir} carries no {missing}: a split_manifest record "
            "always carries every key draw_splits writes."
        )
    members = manifest.get("members")
    dates_missing_label_digests = sorted(
        date_key for date_key, block in (members if isinstance(members, dict) else {}).items()
        if not isinstance(block, dict) or not isinstance(block.get("label_digests"), dict)
        or not block["label_digests"]
    )
    if dates_missing_label_digests:
        raise ValueError(
            f"the split manifest at {split_dir} carries members under "
            f"{dates_missing_label_digests} with no label_digests: a manifest drawn before the "
            "platform recorded per-stem digests binds nothing here; regenerate it with "
            "draw_splits over the current data."
        )
    splits = manifest.get("splits")
    missing_sides = sorted(set(SPLIT_NAMES) - set(splits if isinstance(splits, dict) else {}))
    if not isinstance(splits, dict) or missing_sides:
        raise ValueError(
            f"the split manifest at {split_dir} carries no splits.{missing_sides}: a manifest "
            "drawn before the platform held out a calibration side binds nothing here; "
            "regenerate it with draw_splits over the current data, stating all three ratios "
            "(train_ratio, val_ratio, calibration_ratio)."
        )
    overlaps: dict[tuple[str, str], list[str]] = {}
    for i, side_a in enumerate(SPLIT_NAMES):
        for side_b in SPLIT_NAMES[i + 1:]:
            shared = sorted(set(splits.get(side_a) or []) & set(splits.get(side_b) or []))
            if shared:
                overlaps[(side_a, side_b)] = shared
    if overlaps:
        detail = "; ".join(f"{a}/{b}: {ids[:10]}" for (a, b), ids in sorted(overlaps.items()))
        raise ValueError(
            f"the split manifest at {split_dir} assigns the same identity to more than one side "
            f"({detail}): a member on train and calibration would be trained on it, one on val "
            "and calibration selected on it, and the binder builds its loaders from the sides as "
            "recorded."
        )
    return manifest


def compose_split_manifest(
    out_dir: Path,
    *,
    seed: int,
    group_by: str,
    dataset_fingerprint: str | None,
    subject: str | None,
    attribute: str | None,
    id_map: dict,
    members: dict,
    splits: dict[str, list[str]],
    admission_counts: dict,
    calibration_foreground_groups_by_date: dict,
    realized_ratios: dict,
    group_key_map: dict[str, str] | None = None,
    origin: dict | None = None,
) -> dict:
    """The one place a ``split_manifest`` record is built and written under ``out_dir``.

    ``draw_splits``' own draw and ``freeze_split_manifest``'s frozen partition each compose
    their own fields into this shape and call this to write it, so the two writers can never
    disagree on what a ``split_manifest`` record carries. ``origin`` is present only for a
    frozen manifest (``{"experiment_id", "frozen_at"}``), absent on a drawn one, so
    :func:`read_split_manifest_dir` (and every reader through it) can tell the two apart.

    Returns the manifest dict as written, the same one the caller's own result payload reads
    fields off.
    """
    manifest: dict[str, Any] = {
        "seed": seed,
        "group_by": group_by,
        "dataset_fingerprint": dataset_fingerprint,
        "subject": subject,
        "attribute": attribute,
        "id_map": id_map,
        "members": members,
        "splits": splits,
        "admission_counts": admission_counts,
        "calibration_foreground_groups_by_date": calibration_foreground_groups_by_date,
        "realized_ratios": realized_ratios,
    }
    if group_key_map:
        manifest["group_key_map"] = group_key_map
    if origin:
        manifest["origin"] = origin

    out_dir.mkdir(parents=True, exist_ok=True)
    tcip_store.replace(split_manifest_key(out_dir), manifest)
    return manifest


def read_split_manifest_dir(split_dir: str | Path) -> dict:
    """The ``split_manifest`` record ``draw_splits`` wrote under ``split_dir``, the one reader a
    run names its ``data.split.manifest_dir`` through.

    Refuses with ``ValueError`` naming ``split_dir`` when the record is absent, undecodable, not
    a mapping, lacks any key of :data:`_SPLIT_MANIFEST_REQUIRED_KEYS`, holds a ``members`` block
    whose ``label_digests`` is missing or not a non-empty mapping (a manifest drawn before the
    platform recorded per-stem digests binds nothing here), lacks any name
    :data:`~tcip_mcp.pipelines.data.splits.SPLIT_NAMES` states under ``splits``, or whose sides
    are not pairwise disjoint (a member on two sides would be trained on and selected on, or
    trained on and held out for calibration, at once). One reader, one refusal: the binder, the
    calibration universe, preflight and every entry point read a manifest through here and never test
    its shape themselves. Because ``seed`` is required here, a caller that reads it off the
    returned manifest (a run's bind, resolving its own seed from the manifest's) never sees it
    absent and falls through to a ``None`` in its place.
    """
    manifest = _read_split_manifest_dir_or_none(split_dir)
    if manifest is None:
        raise ValueError(f"no split manifest recorded under {split_dir}; run draw_splits first.")
    return manifest


def read_split_manifest_dir_checked(split_dir: str | Path) -> tuple[dict | None, str | None]:
    """The checked variant of :func:`read_split_manifest_dir`, for a caller (the data picker)
    that lists a candidate directory rather than binds to it, and so must tell "nothing recorded
    here" apart from "something is recorded here and it is wrong".

    Returns ``(manifest, error)``. Absence answers ``(None, None)``: nothing to offer, and
    nothing to explain either. A record that exists but will not decode, fails the required-key
    reading, or names a ``schema_version`` this reader does not accept answers ``(None, text)``,
    catching :class:`tcip_store.SchemaVersionRefused` beside the plain-shape ``ValueError``
    :func:`_read_split_manifest_dir_or_none` raises, for this purpose only: a version refusal
    must never read as an ordinary absence, the way :func:`read_split_manifest_dir` itself still
    lets it propagate uncaught.
    """
    from tcip_store import SchemaVersionRefused

    try:
        manifest = _read_split_manifest_dir_or_none(split_dir)
    except SchemaVersionRefused as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, str(exc)
    return manifest, None


@mcp.tool()
@audited
def freeze_split_manifest(experiment_id: str, output_path: str | None = None) -> dict:
    """Freeze a finished run's own drawn train/val partition into a ``split_manifest`` record,
    so a later run can bind to the identical partition instead of drawing its own.

    Reads the run's ``split.json`` (through ``read_split_manifest_checked``: a record that will
    not decode refuses rather than reading as absent) and its durable config, and refuses,
    naming the primitive, when: no split record exists for ``experiment_id`` or it does not
    decode; the run was bound to a named manifest already (``manifest_binding``: bind to that
    manifest directly instead); the split is spatial (region identities, not stems); the
    validation came from ``data.val_images_dir`` (``resolved_group_by == "external"``: those
    stems never lived under the training roots); the run's val side is empty (it trained without
    validation, a partition no bind can use); the task is not ``detection``/``instance_seg``; the
    config's ``data`` section carries no ``subject``, ``labels_dir``, ``images_dir`` or
    ``id_map``; the record carries no ``group_by`` at all (no grouping policy recorded; never
    defaulted to ``"stem"``); the record's ``dataset_hash`` is ``None`` (the
    run recorded no labels hash, so staleness cannot be checked); the labels changed since the
    run (``dataset_hash(labels_dir)`` now differs from the one ``split.json`` recorded, both
    named; the comparison is over the whole labels directory, since a drawn run's ``split.json``
    records no per-stem digests, so any change anywhere under the labels directory refuses
    freezing, not only a change to the run's own stems: draw a fresh split over the current
    data instead); or a manifest already exists at the output directory.

    The frozen manifest's ``calibration`` side is always empty: freezing a training run's own
    train/val draw records no calibration draw, so the calibration doors' own floor refuses any
    calibration measurement against it by name, and this tool's own answer carries a ``note``
    saying so instead of quietly minting a manifest that looks whole but cannot calibrate.

    Args:
        experiment_id: The finished run to freeze the drawn partition of.
        output_path: Where to write the manifest. Defaults to
            ``<dataset_root>/splits/frozen-<experiment_id>``, resolved from ``data.images_dir``
            through ``dataset_root_of``; refused when that does not resolve (an images directory
            outside the canonical ``<dataset_root>/images/...`` layout).
    """
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.experiments import config_key, read_member, read_split_manifest_checked
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.splits import manifest_date_key, member_identity, normalize_scope
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY
    from tcip_mcp.pipelines.resolution import dataset_hash as _dataset_hash
    from tcip_mcp.pipelines.resolution import dataset_hash_and_label_digests

    split, decode_error = read_split_manifest_checked(experiment_id)
    if decode_error is not None:
        return {"error": f"the split record for {experiment_id!r} could not be read: {decode_error}"}
    if not split:
        return {"error": f"no split record for {experiment_id!r}: this run wrote no split.json "
                         "(it never reached a real dataset build), so there is no drawn "
                         "partition to freeze."}
    if split.get("manifest_binding"):
        bound_dir = split["manifest_binding"].get("manifest_dir")
        if split.get("redrawn_within_manifest"):
            return {"error": f"{experiment_id!r} redrew train and val inside the split manifest "
                             f"at {bound_dir!r} already: reproduce it by binding a later run to "
                             "that same manifest with the same seed and "
                             "data.split.redraw_within_manifest=true, with the labels this run's "
                             "own split.json recorded (label_digests.at_run) unchanged since "
                             "(the redraw reads per-stem annotation counts at run time, not "
                             "only the manifest's fixed membership), never by freezing."}
        return {"error": f"{experiment_id!r} was bound to a split manifest already "
                         f"({bound_dir!r}): bind a later run "
                         "to that manifest directly instead of freezing this one's."}
    resolved_group_by = split.get("group_by")
    if resolved_group_by == "spatial_strip":
        return {"error": f"{experiment_id!r}'s split is spatial (region identities, not stems): "
                         "freeze_split_manifest binds a stem-keyed partition, which a spatial "
                         "split never draws."}
    if resolved_group_by == "external":
        return {"error": f"{experiment_id!r}'s validation came from data.val_images_dir "
                         "(resolved_group_by='external'): those stems never lived under the "
                         "training roots, so there is no partition of this dataset to freeze."}
    train_stems, val_stems = split.get("train") or [], split.get("val") or []
    if not val_stems:
        return {"error": f"{experiment_id!r} trained without validation (an empty val side): "
                         "a partition no bind can use."}

    config = read_member(config_key(experiment_id), {})
    config = config if isinstance(config, dict) else {}
    model_source = config.get(MODEL_SOURCE_KEY) or {}
    data_cfg = config.get("data") or {}
    task = model_source.get("task") or data_cfg.get("task", "detection")
    if task not in ("detection", "instance_seg"):
        return {"error": f"{experiment_id!r} trained task={task!r}; freeze_split_manifest binds "
                         "only detection and instance_seg runs, the tasks a split manifest "
                         "admits through."}
    subject, labels_dir, images_dir, id_map = (
        data_cfg.get("subject"), data_cfg.get("labels_dir"), data_cfg.get("images_dir"),
        data_cfg.get("id_map"),
    )
    missing = [name for name, value in (
        ("subject", subject), ("labels_dir", labels_dir), ("images_dir", images_dir),
        ("id_map", id_map),
    ) if not value]
    if missing:
        return {"error": f"{experiment_id!r}'s durable config carries no {missing}: "
                         "freeze_split_manifest needs every one of them to compose a manifest."}
    assert labels_dir is not None and images_dir is not None and id_map is not None, \
        "checked non-empty above"
    if resolved_group_by is None:
        return {"error": f"{experiment_id!r}'s split record carries no group_by at all (no "
                         "grouping policy recorded): freeze_split_manifest never defaults one, "
                         "since guessing 'stem' could silently misstate the policy the run "
                         "actually drew under."}

    labels_hash_at_split = split.get("dataset_hash")
    if labels_hash_at_split is None:
        return {"error": f"{experiment_id!r}'s split record carries no dataset_hash (the run "
                         "recorded no labels hash at draw time): freeze_split_manifest cannot "
                         "check the labels have not moved since, so it refuses rather than "
                         "freezing a partition it cannot vouch for."}
    labels_hash_now = _dataset_hash(labels_dir)
    if labels_hash_now != labels_hash_at_split:
        return {"error": f"the labels under {labels_dir!r} changed since {experiment_id!r} "
                         f"trained (dataset_hash was {labels_hash_at_split!r}, is now "
                         f"{labels_hash_now!r}): the comparison is over the whole labels "
                         "directory, since a drawn run's split.json records no per-stem "
                         "digests, so any change anywhere under it refuses freezing, not only "
                         "a change to the run's own stems; freeze a run whose labels have not "
                         "moved, or draw a fresh split over the current data."}

    date = split.get("date") or None
    date_key = manifest_date_key(date)

    dataset_root = dataset_root_of(images_dir)
    if dataset_root is None:
        return {"error": f"data.images_dir={images_dir!r} does not resolve under a dataset "
                         "root (dataset_root_of)."}
    if output_path is None:
        output_path = str(dataset_root / "splits" / f"frozen-{experiment_id}")
    out_dir = Path(output_path)
    existing, existing_error = read_split_manifest_dir_checked(out_dir)
    if existing is not None or existing_error is not None:
        return {"error": f"a split manifest already exists at {output_path!r}: "
                         f"{existing_error or 'freeze_split_manifest never overwrites one.'}"}

    all_stems = sorted(set(train_stems) | set(val_stems))
    date_hash, date_label_digests = dataset_hash_and_label_digests(labels_dir, all_stems)
    members = {date_key: {
        "labels_root": str(labels_dir), "images_root": str(images_dir),
        "dataset_hash": date_hash, "label_digests": date_label_digests,
    }}
    splits = {
        "train": sorted(member_identity(date, s) for s in train_stems),
        "val": sorted(member_identity(date, s) for s in val_stems),
        "calibration": [],
    }
    total = len(train_stems) + len(val_stems)
    realized_ratios = {
        "train": len(train_stems) / total if total else 0.0,
        "val": len(val_stems) / total if total else 0.0,
        "calibration": 0.0,
    }
    raw_group_key_map = split.get("group_key_map")
    group_key_map = (
        {member_identity(date, stem): key for stem, key in raw_group_key_map.items()}
        if raw_group_key_map else None
    )
    try:
        fingerprint = dataset_fingerprint(dataset_root)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the frozen manifest: {exc}"}

    from datetime import datetime, timezone

    _, normalized_attribute = normalize_scope(subject, data_cfg.get("attribute"))
    manifest = compose_split_manifest(
        out_dir, seed=int(split.get("seed", 42)), group_by=resolved_group_by,
        dataset_fingerprint=fingerprint, subject=subject, attribute=normalized_attribute,
        id_map=id_map, members=members, splits=splits, admission_counts={},
        calibration_foreground_groups_by_date={date_key: 0}, realized_ratios=realized_ratios,
        group_key_map=group_key_map,
        origin={"experiment_id": experiment_id,
               "frozen_at": datetime.now(timezone.utc).isoformat()},
    )
    return {
        "manifest_dir": str(out_dir), "train": len(train_stems), "val": len(val_stems),
        "calibration": 0, "date": date, "origin": manifest["origin"],
        "note": "the calibration side is empty (a training run's own drawn partition records "
               "no calibration draw): the calibration doors' own floor refuses any calibration "
               "measurement against this manifest by name.",
    }


ROOT_LABEL_CANDIDATES = ("annotations.json", "labels.json", "instances.json")
"""Candidate filenames for one assembled dataset-level label document at a dataset's root,
checked in this order; the first one present on disk is the dataset's label store. Shared by
:func:`_scan_dataset` and the doctor's own ``check_data_quality`` so both name the same three
candidates."""


def _scan_dataset(root: str) -> dict:
    """Scan a directory tree for images and labels.

    Labels are the name-based per-image JSON (one file per image, all subjects) under
    ``annotations/<date>/`` (no detect/segment split), a review baseline directory's copies
    excluded, plus a single assembled dataset-level COCO at the root if one is present: the root
    candidate sits beside the per-image tree, never in place of it, so a dataset carrying both
    reports every one of them.

    An unreadable first-sorted label (undecodable, non-dict, or otherwise malformed) raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` rather than being folded into "format
    undetectable": the caller reports it as the named file it is, not a guess. ``format`` is an
    informational best guess (the per-image tree's first-sorted label's shape, or the root
    candidate's when there is no per-image tree), not a claim every label file shares it; the
    doctor's own ``check_data_quality`` decides format per file instead.

    ``labels`` is a raw ``rglob``, so it counts a file whose name is reserved for a prediction
    bucket's own provenance stamp the way :func:`~tcip_mcp.dataset_layout.subjects_on_date` and
    every bucket walk through ``prediction_documents`` would not; ``reserved_name_labels`` names
    each one, so a caller comparing this census against those walks can tell the difference is a
    known exclusion rather than a disagreement. ``reserved_name_images`` names every image whose
    own stem is reserved the same way, since such an image otherwise counts as an ordinary
    unlabelled one with no signal at all that its label can never be read through any bucket walk.

    ``images``, unlike ``labels``, is built per bucket through
    :func:`~tcip_mcp.pipelines.image_utils.list_logical_images`: a stem collision within one
    bucket raises :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem` rather than this
    census silently keeping one raw file of the pair, and a grouped capture counts once, its own
    manifest, never once per band file. Walks one level under ``images/``: the flat root itself
    plus each direct date-bucket subdirectory (the same shape the ``doctor`` command's own
    ``_image_stems`` walks), never a deeper recursive descent. Falls back to a raw walk of the
    whole dataset root only when there is no canonical ``images/`` tree to route through at all.
    """
    from tcip_annotation.json_io import is_sidecar_name, prediction_documents
    from tcip_annotation.review_engine import BASELINE_DIRNAME
    from tcip_mcp.dataset_layout import annotation_root, image_root, prediction_root
    from tcip_mcp.pipelines.image_utils import BandGroupRef, IMAGE_EXTS, list_logical_images

    root_path = Path(root)
    image_exts = IMAGE_EXTS
    images: list[str] = []
    labels: list[str] = []
    preds: list[str] = []
    reserved_name_labels: list[str] = []
    reserved_name_images: list[str] = []
    detected_format: str | None = None

    # Find images through the platform's own bucket enumeration: a stem collision refuses here
    # too, and a grouped capture counts once, its own manifest.
    images_dir = image_root(root_path)
    if images_dir.is_dir():
        buckets = [images_dir] + sorted(p for p in images_dir.iterdir() if p.is_dir())
        for bucket in buckets:
            for source in list_logical_images(bucket).values():
                f = source.manifest_path if isinstance(source, BandGroupRef) else source
                images.append(str(f))
                if is_sidecar_name(f"{f.stem}.json"):
                    reserved_name_images.append(str(f))
    else:
        # No canonical images/ tree, so no bucket contract to route through this walk.
        for f in sorted(root_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in image_exts:
                images.append(str(f))
                if is_sidecar_name(f"{f.stem}.json"):
                    reserved_name_images.append(str(f))

    # Ground-truth labels: annotations/[<date>/]<stem>.json (one file per image, every subject),
    # a review baseline copy under BASELINE_DIRNAME excluded: it is a snapshot, not a label.
    ann_dir = annotation_root(root_path)
    if ann_dir.is_dir():
        labels = [
            str(f) for f in sorted(ann_dir.rglob("*.json"))
            if f.is_file() and BASELINE_DIRNAME not in f.parts
        ]
        reserved_name_labels = [f for f in labels if is_sidecar_name(Path(f).name)]
        if labels:
            try:
                from tcip_annotation.format_io import detect_format
                detected_format = detect_format(labels[0])
            except ValueError:
                detected_format = None  # unrecognized: report nothing rather than a guess

    # A single COCO JSON at the dataset root: one more present label beside the per-image tree.
    root_candidate = _root_label_candidate(root, set(labels))
    if root_candidate is not None:
        labels.append(root_candidate)
        if detected_format is None:
            try:
                from tcip_annotation.format_io import detect_format
                detected_format = detect_format(root_candidate)
            except ValueError:
                pass

    # Predictions: predictions/<model>/[<date>/]<stem>.json; each model/date bucket is walked on
    # its own through prediction_documents, so the bucket's own stamps are excluded everywhere.
    pred_dir = prediction_root(root_path)
    if pred_dir.is_dir():
        preds = [
            str(f)
            for bucket in sorted({p.parent for p in pred_dir.rglob("*.json")})
            for f in prediction_documents(bucket)
        ]

    return {
        "images": images, "labels": labels, "predictions": preds, "format": detected_format,
        "reserved_name_labels": reserved_name_labels, "reserved_name_images": reserved_name_images,
    }


@audited
def scan_dataset(folder_path: str) -> dict:
    """Scan a folder for images, labels, and predictions.

    Not an MCP tool: run through ``tcip scan-dataset``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its in-package callers.

    Reads the name-based per-image JSON labels (one file per image, all subjects), or an assembled
    dataset-level COCO.

    Expects the canonical layout (see tcip_mcp.dataset_layout):
        images/<date>/  annotations/<date>/<stem>.json  predictions/<model>/<date>/<stem>.json

    ``reserved_name_labels`` names every label counted in ``labels_count`` whose filename is
    reserved for a prediction bucket's own provenance stamp: this census walks with ``rglob`` and
    counts it, while the bucket walks this platform reads labels through exclude it.
    ``reserved_name_images`` names every image counted in ``image_count`` whose own stem is
    reserved the same way; such an image otherwise sits in ``unlabelled_images`` with no signal
    at all that its label can never be read through any bucket walk.

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused

    try:
        scan = _scan_dataset(folder_path)
    except (UnreadableLabelDocument, AmbiguousImageStem) as exc:
        return {"error": str(exc)}
    except SchemaVersionRefused as exc:
        return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}

    image_stems = {Path(p).stem: p for p in scan["images"]}
    label_stems = {Path(p).stem for p in scan["labels"]}

    paired = sum(1 for stem in image_stems if stem in label_stems)
    unlabelled = len(image_stems) - paired

    return {
        "path": folder_path,
        "format": scan.get("format"),
        "image_count": len(scan["images"]),
        "labels_count": len(scan["labels"]),
        "predictions_count": len(scan["predictions"]),
        "paired_images": paired,
        "unlabelled_images": unlabelled,
        "image_stems_sample": sorted(image_stems.keys())[:10],
        "reserved_name_labels": scan["reserved_name_labels"],
        "reserved_name_images": scan["reserved_name_images"],
    }


def _root_label_candidate(folder_path: str, already_present: set) -> str | None:
    """The dataset root's own assembled-label candidate, if one is present and not already
    counted among ``already_present``.

    The one walk of ``ROOT_LABEL_CANDIDATES`` in their declared first-match order, called by
    :func:`_scan_dataset` before the candidate joins its ``labels`` list and by any other caller
    that has its own already-counted set to check the candidate against, so a present root
    candidate can never be walked for twice by two diverging implementations. A candidate whose
    format cannot be determined is still returned: it is a present label file, not evidence the
    dataset carries none, and detecting its format is left to the caller.
    """
    root_path = Path(folder_path)
    for candidate in ROOT_LABEL_CANDIDATES:
        cpath = root_path / candidate
        if cpath.is_file():
            return None if str(cpath) in already_present else str(cpath)
    return None



def _split_date_dirs(folder_path: str | Path) -> list[tuple[str | None, Path, Path]]:
    """Every ``(date, labels_dir, images_dir)`` a dataset's per-image label tree holds: one entry
    per ``annotations/<date>/`` beside its images (``images/<date>/`` when that bucket exists, else
    the flat ``images/`` root), plus one dateless entry for any label loose directly in
    ``annotations/`` beside a dated tree, so a mixed layout's flat labels are never dropped from
    the draw. A fully flat dataset (no date subdirectories at all) yields exactly that one
    dateless entry.

    Empty when the dataset holds no per-image label tree at all (a root-level assembled COCO
    only, which :func:`_scan_dataset` counts as a label but this never walks): the platform's own
    admission for the tasks a split manifest can bind to draws through the per-image tree, never
    that document.
    """
    from tcip_annotation.json_io import prediction_documents
    from tcip_mcp.dataset_layout import (
        annotation_dir, annotation_root, is_bucket_name, resolve_images_dir,
    )

    root = Path(folder_path)
    ann_root = annotation_root(root)
    if not ann_root.is_dir():
        return []
    subdirs = sorted(d.name for d in ann_root.iterdir() if d.is_dir() and is_bucket_name(d.name))
    entries: list[tuple[str | None, Path, Path]] = [
        (d, annotation_dir(root, d), resolve_images_dir(root, d)) for d in subdirs
    ]
    loose_labels = bool(prediction_documents(ann_root))
    if loose_labels or not subdirs:
        entries.append((None, ann_root, resolve_images_dir(root, None)))
    return entries


@mcp.tool()
@audited
def draw_splits(
    folder_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    calibration_ratio: float = 0.0,
    seed: int = 42,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    stratify_foreground: bool = True,
    output_path: str | None = None,
    materialize: bool = False,
    copy_files: bool = True,
    subject: str | None = None,
    attribute: str | None = None,
) -> dict:
    """Compute a leakage-free, annotation-stratified train/val/calibration split.

    Non-destructive by default: emits a ``split_manifest.json`` record (when writing a manifest)
    plus a stats dict; a side's membership lives only in the manifest's own ``splits`` field.
    Sibling tiles of one source image are kept in the same split (no tree-/
    canopy-level leakage), and, when ``stratify_foreground`` is set, splits are balanced by
    annotation count so dense and sparse sources are proportionally represented. Groups whole
    source images; a within-image split for a folder holding a single source is a training run's
    own automatic route (``data.tiling`` in the run config), not this tool.

    Writing a manifest (``output_path`` given, or ``materialize=True``) draws its members
    through the platform's own admission for the tasks a manifest can bind to
    (``tcip_mcp.pipelines.data.label_queries.trainable_stems``, the same function a training run's
    own draw uses): for each capture date the dataset holds, every image carrying an annotation of
    ``subject`` (with every instance assessed for ``attribute``, when one is given) or a human's
    negative confirmation for it. ``subject`` is therefore required to write a manifest; a call
    with neither ``output_path`` nor ``materialize`` answers over every image in the tree instead,
    no subject needed. Each admitted member's identity is ``<date>/<stem>`` (the bare ``<stem>``
    under a flat, dateless tree), since a stem is unique only within one capture date.
    ``stratify_foreground`` only toggles the annotation-count balancing; it does not change which
    images are eligible to enter the split.

    The third side, ``calibration``, is the universe every calibration drawn under this manifest
    draws from (the operating point is measured on it, never on ``train`` or ``val``); a manifest
    write therefore has no default for any of the three ratios and refuses a zero one, naming it.
    The draw refuses, before any write, when the tree holds fewer foreground groups of
    ``subject`` (and ``attribute``, when scoped) than the three sides need at minimum (one each
    for ``train``/``val``, two for ``calibration``, so the locked calibration/holdout draw the
    calibration door makes later can still halve it), counted for the draw's own subject
    regardless of ``stratify_foreground``: that flag only toggles the balancing pass, never
    whether the minimum pass sees real foreground. The answer's
    ``calibration_foreground_groups_by_date`` (also carried on the persisted manifest record)
    then reports, per date the draw holds members under, how many of the calibration side's own
    groups for that date actually carry a foreground annotation, ``0`` for a date that drew short,
    since the floor above is over the whole draw and a single date can still land short. Both the
    answer and the manifest record also carry ``realized_ratios``, each side's member share of
    the draw actually delivered: on a tree sized at the floor, the minimum pass can consume every
    foreground group before the balancing pass ever sees the caller's fractions, so the delivered
    shares can diverge from the ratios asked for, and this states the shape actually drawn beside
    them.

    With ``materialize=True`` it additionally lays out a
    ``{train,val,calibration}/{images,labels}/`` tree under ``output_path`` (defaulting to
    ``folder_path/splits``), copying (or symlinking, ``copy_files=False``) each stem's image and
    label, and adds ``output_dir`` / ``structure`` to the return. Refused when the drawn
    membership spans more than one capture date: the materialized tree is a flat, undated
    ``{split}/{images,labels}/`` layout keyed by file name, and its negative carry writes one
    undated status bucket, so two dates sharing a file name would collide silently.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set. Defaults to 0.8, the complement of the
            unchanged 0.2 validation default once ``calibration_ratio`` is 0.
        val_ratio: Fraction for validation set.
        calibration_ratio: Fraction held out as the calibration universe. Defaults to 0.0 for a
            stats-only call, and a non-zero value is admitted there too. A manifest write
            (``output_path`` given, or ``materialize=True``) refuses a zero ratio on any of the
            three (``train_ratio``, ``val_ratio``, ``calibration_ratio``), naming it, since a
            manifest always draws all three sides.
        seed: Random seed for reproducibility.
        group_by: Group selector: ``"tile_prefix"`` (strip a trailing
            ``_<x>_<y>`` tile offset) or ``"stem"`` (one group per member). Ignored when
            ``group_key_map`` is given.
        group_key_map: An agent-derived ``{identity: group_key}`` map overriding ``group_by``,
            keyed the same way the members are (``<date>/<stem>``, the bare ``<stem>`` under a
            flat tree); must cover every member. Recorded as ``group_by="explicit_map"`` in the
            result and manifest (the resolved policy, not the raw ``group_by`` string).
        stratify_foreground: Balance splits by foreground annotation count.
        output_path: Where to write manifests (and, when materializing, the file tree).
            Defaults to ``folder_path/splits`` when materializing, else manifests are
            written only if this is set.
        materialize: Also copy/symlink files into a {train,val,calibration}/{images,labels}/ tree.
        copy_files: Copy files (True) or create symlinks (False) when materializing.
        subject: The object class the split is drawn for. Required to write a manifest
            (``output_path`` given, or ``materialize=True``): it governs both which images the
            draw admits and which of the source dataset's confirmed negatives a materialized
            split tree carries, since that tree can't recover the subject from its own path.
        attribute: Scope the draw to instances already assessed for this attribute of
            ``subject``; an image carrying an instance never assessed for it is excluded
            entirely, the same rail a training run applies. ``None`` draws over every instance of
            ``subject`` regardless of attribute state.
    """
    if abs(train_ratio + val_ratio + calibration_ratio - 1.0) > 0.01:
        return {"error": "train_ratio, val_ratio and calibration_ratio must sum to 1.0 (got "
                         f"{train_ratio}, {val_ratio}, {calibration_ratio})."}
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.splits import (
        SPLIT_NAMES,
        count_label_lines,
        group_balanced_split,
        manifest_date_key,
        member_identity,
        member_identity_parts,
        refuse_insufficient_foreground_groups,
        resolve_group_key_fn,
    )
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused

    kept_splits = SPLIT_NAMES
    out_dir = Path(output_path) if output_path else (Path(folder_path) / "splits" if materialize else None)
    if out_dir is not None:
        zero_ratios = [name for name, ratio in (
            ("train_ratio", train_ratio), ("val_ratio", val_ratio),
            ("calibration_ratio", calibration_ratio),
        ) if ratio == 0]
        if zero_ratios:
            return {"error": f"{', '.join(zero_ratios)} must be non-zero to write a split "
                             "manifest: a manifest's three sides are all drawn from, so a "
                             "manifest write states all three ratios (train_ratio, val_ratio, "
                             "calibration_ratio) as non-zero. Omit output_path and materialize "
                             "for a stats-only call, whose ratios may include a zero."}

    if out_dir is None:
        # A stats-only call writes no manifest, so the draw is a plain image/label scan: no
        # subject required, every image eligible.
        try:
            scan = _scan_dataset(folder_path)
        except (UnreadableLabelDocument, AmbiguousImageStem) as exc:
            return {"error": str(exc)}
        except SchemaVersionRefused as exc:
            return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}
        image_map = {Path(p).stem: p for p in scan["images"]}
        label_map = {Path(p).stem: p for p in scan["labels"]}

        stratified = bool(stratify_foreground and label_map)
        stems = sorted(set(image_map) & set(label_map)) if stratified else sorted(image_map)
        if not stems:
            return {"error": "No images found to split"}

        annotation_counts = None
        if stratified:
            # count_label_lines is JSON-aware; raw count_lines would count pretty-printed JSON
            # lines as annotations (a {objects: []} negative reads as ~5 foreground objects).
            try:
                annotation_counts = {
                    s: count_label_lines(Path(label_map[s]).parent, s) for s in stems
                }
            except UnreadableLabelDocument as exc:
                return {"error": str(exc)}

        try:
            group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
        except ValueError as exc:
            return {"error": str(exc)}
        resolved_group_by = "explicit_map" if group_key_map else group_by
        parts = group_balanced_split(
            stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
            splits=(train_ratio, val_ratio, calibration_ratio), seed=seed,
        )
        counts = annotation_counts or {}
        dataset_hashes_by_date: dict[str, str] = {}
        if label_map:
            from tcip_mcp.dataset_layout import annotation_date
            from tcip_mcp.pipelines.resolution import dataset_hash as _dataset_hash

            stems_by_date_key: dict[str, list[str]] = {}
            for stem in stems:
                label_path = label_map.get(stem)
                if label_path is None:
                    continue
                date_key = manifest_date_key(annotation_date(label_path))
                stems_by_date_key.setdefault(date_key, []).append(stem)
            for date_key, date_stems in stems_by_date_key.items():
                labels_dir = Path(label_map[date_stems[0]]).parent
                dataset_hashes_by_date[date_key] = _dataset_hash(labels_dir, stems=sorted(date_stems))
        # A single dataset_hash is meaningful only over one labels directory; over more than one
        # it would be blind to every date but the first, so it is carried only then.
        dataset_hash = (next(iter(dataset_hashes_by_date.values()))
                        if len(dataset_hashes_by_date) == 1 else None)
        return {
            "splits": {k: len(parts[k]) for k in kept_splits},
            "foreground_annotations": {
                k: sum(int(counts.get(s, 0)) for s in parts[k]) for k in kept_splits
            },
            "total_stems": len(stems),
            "total_annotations": sum(int(v) for v in counts.values()),
            "groups": len({group_key_fn(s) for s in stems}),
            "seed": seed,
            "dataset_hash": dataset_hash,
            "dataset_hashes_by_date": dataset_hashes_by_date,
            "group_by": resolved_group_by,
            "stratified": stratified,
            "manifest_dir": None,
        }

    # Writing a manifest: the draw is the platform's own per-subject admission.
    if not subject:
        return {"error": "draw_splits needs subject to write a split manifest (output_path given, "
                         "or materialize=True): pass the object class the run will admit under, "
                         "or drop both output_path and materialize for a stats-only call."}

    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map, trainable_stems
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete
    from tcip_mcp.pipelines.resolution import dataset_hash_and_label_digests

    date_dirs = _split_date_dirs(folder_path)
    if not date_dirs:
        return {"error": f"{folder_path} holds no per-image label tree (annotations/<date>/ or a "
                         "flat annotations/) for draw_splits to draw a subject-scoped split from; "
                         "a dataset-level assembled COCO at the root is not walked here."}

    entries_by_images_dir: dict[Path, list[str]] = {}
    for entry_date, _, entry_images_dir in date_dirs:
        entries_by_images_dir.setdefault(entry_images_dir, []).append(
            entry_date if entry_date is not None else "annotations/ (loose labels)"
        )
    colliding = {d: names for d, names in entries_by_images_dir.items() if len(names) > 1}
    if colliding:
        detail = "; ".join(
            f"{img_dir}: {sorted(names)}" for img_dir, names in sorted(colliding.items())
        )
        return {"error": f"{folder_path} has label entries that resolve to the same images "
                         f"directory ({detail}): a manifest keyed by <date>/<stem> would admit "
                         "one image file once per entry and could place the same pixels on "
                         "both sides of the split. Give each date its own images/<date>/ "
                         "bucket, or merge the colliding label entries into one."}
    try:
        _, id_map = resolve_registry_id_map(date_dirs[0][1], subject, attribute)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot resolve the class registry for the split: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}

    members: dict[str, Any] = {}
    identity_locations: dict[str, tuple[str | None, str]] = {}  # identity -> (date, bare stem)
    admission_counts: dict[str, int] = {}
    annotation_counts = None
    negative_carry: "_NegativeCarry | None" = None
    # The admission below reads every candidate stem's label first, so neither the count nor the
    # carry can raise this error over a stem the admission already read without raising.
    try:
        for date, labels_dir, images_dir in date_dirs:
            admitted, counts = trainable_stems(
                labels_dir, images_dir, subject=subject, date=date, attribute=attribute,
                id_map=id_map,
            )
            for key, value in counts.items():
                admission_counts[key] = admission_counts.get(key, 0) + value
            for stem in admitted:
                identity_locations[member_identity(date, stem)] = (date, stem)
            if admitted:
                date_hash, date_label_digests = dataset_hash_and_label_digests(
                    labels_dir, sorted(admitted))
                members[manifest_date_key(date)] = {
                    "labels_root": str(labels_dir),
                    "images_root": str(images_dir),
                    "dataset_hash": date_hash,
                    "label_digests": date_label_digests,
                }

        stems = sorted(identity_locations)
        foreground_counts: dict[str, int] = {}
        if stems:
            labels_dir_of = {date: labels_dir for date, labels_dir, _ in date_dirs}
            foreground_counts = {
                identity: count_label_lines(
                    labels_dir_of[date], stem, subject=subject, attribute=attribute)
                for identity, (date, stem) in identity_locations.items()
            }
        if stratify_foreground:
            annotation_counts = foreground_counts

        calibration_foreground_groups_by_date: dict[str, int] = {}
        realized_ratios: dict[str, float] = {}
        if stems:
            try:
                group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
            except ValueError as exc:
                return {"error": str(exc)}
            resolved_group_by = "explicit_map" if group_key_map else group_by
            min_foreground_groups = {"train": 1, "val": 1, "calibration": 2}
            fg_groups = {group_key_fn(s) for s in stems if foreground_counts.get(s, 0) > 0}
            try:
                refuse_insufficient_foreground_groups(len(fg_groups), min_foreground_groups)
            except ValueError as exc:
                return {"error": str(exc)}
            parts = group_balanced_split(
                stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
                splits=(train_ratio, val_ratio, calibration_ratio), seed=seed,
                min_foreground_groups=min_foreground_groups, foreground_counts=foreground_counts,
            )
            total_drawn = sum(len(parts[k]) for k in kept_splits)
            realized_ratios = {
                k: (len(parts[k]) / total_drawn if total_drawn else 0.0) for k in kept_splits
            }
            cal_groups_by_date: dict[str, set[str]] = {}
            for identity in parts["calibration"]:
                if foreground_counts.get(identity, 0) > 0:
                    cal_date, _ = member_identity_parts(identity)
                    cal_groups_by_date.setdefault(
                        manifest_date_key(cal_date), set()).add(group_key_fn(identity))
            calibration_foreground_groups_by_date = {
                date_key: len(cal_groups_by_date.get(date_key, set()))
                for date_key in sorted(members)
            }

            distinct_dates = {date for date, _ in identity_locations.values()}
            if materialize and len(distinct_dates) > 1:
                named = sorted(d if d is not None else "annotations/ (loose labels)"
                               for d in distinct_dates)
                return {"error": f"materialize=True refuses a manifest spanning more than one "
                                 f"capture date ({named}): its flat {{train,val,calibration}}/"
                                 "{images,labels}/ tree is keyed by file name and its negative "
                                 "carry writes one undated bucket, so two dates sharing a name "
                                 "would collide silently. Write the manifest without "
                                 "materializing, or scope this call to one capture date."}

            if materialize:
                from tcip_mcp.pipelines.image_utils import list_logical_images, resolve_image_source

                (only_date,) = distinct_dates
                labels_dir_for_date = next(d for date, d, _ in date_dirs if date == only_date)
                images_dir_for_date = next(d for date, _, d in date_dirs if date == only_date)
                image_map = list_logical_images(images_dir_for_date)
                label_map = {
                    stem: labels_dir_for_date / f"{stem}.json" for stem in image_map
                    if (labels_dir_for_date / f"{stem}.json").is_file()
                }
                bare_parts = {
                    split_name: sorted(member_identity_parts(identity)[1] for identity in identities)
                    for split_name, identities in parts.items() if split_name in kept_splits
                }
                # Resolved before anything is written: an incomplete band group answers an
                # error naming the missing band, never a raise after the manifest is on disk.
                for split_name in kept_splits:
                    for stem in bare_parts[split_name]:
                        resolve_image_source(images_dir_for_date, stem)
                # Read every confirmed negative before anything (the manifest, the split tree)
                # is written: a refusal here must leave nothing persisted.
                negative_carry = _compute_negative_carry(
                    label_map, bare_parts, image_map, subject, only_date)
    except (UnreadableLabelDocument, AmbiguousImageStem, BandGroupIncomplete) as exc:
        return {"error": str(exc)}
    except SchemaVersionRefused as exc:
        return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}

    if not stems:
        from tcip_mcp.dataset_layout import image_dir as _image_dir, list_dates as _list_dates

        searched = ", ".join(
            f"{(entry_date or 'annotations/ (loose labels)')} -> {entry_images_dir}"
            for entry_date, _, entry_images_dir in date_dirs
        )
        used_images_dirs = {entry_images_dir for _, _, entry_images_dir in date_dirs}
        unused_date_buckets = sorted(
            d for d in _list_dates(folder_path) if _image_dir(folder_path, d) not in used_images_dirs
        )
        remedy = ""
        if unused_date_buckets:
            listed = ", ".join(str(_image_dir(folder_path, d)) for d in unused_date_buckets)
            remedy = (f" {listed} exist with no label entry resolved against them; move the "
                     "labels into a matching annotations/<date>/ bucket, or move the images "
                     "to the flat images/ root, to pair them.")
        return {"error": f"no sample of subject {subject!r} was admitted under {folder_path} "
                         f"(attribute={attribute!r}): {admission_counts}. Searched {searched}."
                         f"{remedy} Annotate an instance or confirm a negative before splitting."}

    try:
        fingerprint = dataset_fingerprint(folder_path)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the split manifest: {exc}"}
    compose_split_manifest(
        out_dir, seed=seed, group_by=resolved_group_by, dataset_fingerprint=fingerprint,
        subject=subject, attribute=attribute, id_map=id_map, members=members,
        splits={k: sorted(parts[k]) for k in kept_splits}, admission_counts=admission_counts,
        calibration_foreground_groups_by_date=calibration_foreground_groups_by_date,
        realized_ratios=realized_ratios, group_key_map=group_key_map,
    )

    counts = annotation_counts or {}
    result = {
        "splits": {k: len(parts[k]) for k in kept_splits},
        "foreground_annotations": {
            k: sum(int(counts.get(s, 0)) for s in parts[k]) for k in kept_splits
        },
        "total_stems": len(stems),
        "total_annotations": sum(int(v) for v in counts.values()),
        "groups": len({group_key_fn(s) for s in stems}),
        "seed": seed,
        "group_by": resolved_group_by,
        "stratified": bool(stratify_foreground),
        "subject": subject,
        "attribute": attribute,
        "admission_counts": admission_counts,
        "manifest_dir": str(out_dir),
        "dataset_hashes_by_date": {key: block["dataset_hash"] for key, block in members.items()},
        "calibration_foreground_groups_by_date": calibration_foreground_groups_by_date,
        "realized_ratios": realized_ratios,
    }

    if materialize:
        from tcip_mcp.pipelines.image_utils import flat_image_key, place_logical_image

        # Labels are copied or symlinked directly here, mirroring copy_files; only the image
        # placement below routes through the store.
        place_fn = shutil.copy2 if copy_files else os.symlink
        for split_name in kept_splits:
            split_stems = bare_parts[split_name]
            img_dir = out_dir / split_name / "images"
            lbl_dir = out_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            def _dest_key(filename: str, _images_dir: Path = img_dir) -> Key:
                return flat_image_key(_images_dir, filename)

            for stem in split_stems:
                place_logical_image(
                    image_map[stem], img_dir, copy_files=copy_files, dest_key=_dest_key
                )
                if stem in label_map:
                    src_lbl = Path(label_map[stem])
                    dst_lbl = lbl_dir / src_lbl.name
                    if not dst_lbl.exists():
                        place_fn(str(src_lbl), str(dst_lbl))
        _apply_negative_carry(negative_carry, out_dir, subject)
        if negative_carry is not None and negative_carry.contradicted:
            result["contradicted_negatives"] = sorted(negative_carry.contradicted)
        result["output_dir"] = str(out_dir)
        result["structure"] = f"{out_dir}/{{train,val,calibration}}/{{images,labels}}/"

    return result


class _NegativeCarry:
    """What :func:`_compute_negative_carry` found, before anything is written: each split's own
    slice of the source subject's confirmed negatives, the schema digest to stamp them with, and
    the names the label content contradicted (excluded from every slice, reported to the caller)."""

    def __init__(self, by_split: dict[str, dict[str, dict[str, str]]], contradicted: set[str],
                digest: str | None, src_classes: Path | None) -> None:
        self.by_split = by_split
        self.contradicted = contradicted
        self.digest = digest
        self.src_classes = src_classes


def _compute_negative_carry(label_map: dict, parts: dict, image_map: dict,
                            subject: str | None, date: str | None) -> "_NegativeCarry | None":
    """Reads the source subject's confirmed negatives and assigns each to the split holding its
    image, entirely before any split-tree file is written (see the call site).

    A split tree is ``{train,val,calibration}/labels`` by construction and cannot recover the subject from
    its path, so the confirmations are carried explicitly under the threaded ``subject`` (keyed by
    ``status_bucket(subject, None)``, since the split carries no date). Without this, every image a
    human confirmed negative reads as an unconfirmed empty in the split and is dropped from training.
    No subject threaded -> nothing to attribute the confirmations to, so ``None`` is returned.

    ``materialize=True`` draws exactly one capture date, so the source side reads that date's own
    ``status_bucket(subject, date)`` bucket, through :func:`confirmed_negative_records`, the same
    predicate ``trainable_stems``' admission already reads: a confirmation this split's own
    admission never saw is not this split's to carry. A name whose label file now holds the
    subject is excluded the way :func:`confirmed_negative_records` always excludes one, named in
    the returned carry's ``contradicted`` rather than silently dropped.
    """
    if not subject:
        return None
    from tcip_mcp.class_registry import (
        RegistryError, attribute_schema_digest, read_registry,
    )
    from tcip_mcp.dataset_layout import classes_path, dataset_root_of
    from tcip_mcp.pipelines.data.label_queries import confirmed_negative_records
    from tcip_mcp.pipelines.image_utils import logical_image_name

    src_dirs = {Path(p).parent for p in label_map.values()}
    if not src_dirs:
        return None
    negatives: dict[str, dict[str, str]] = {}
    contradicted: set[str] = set()
    for d in sorted(src_dirs):
        negatives.update(confirmed_negative_records(
            d, subject=subject, date=date, contradicted_out=contradicted))
    if not negatives:
        return _NegativeCarry(by_split={}, contradicted=contradicted, digest=None, src_classes=None)

    # A best-effort registry read: without a stamp, a split tree can never quarantine a stale
    # confirmation later (a permanent no-op, not "admit until proven stale").
    digest = None
    src_classes: Path | None = None
    for d in src_dirs:
        root = dataset_root_of(d)
        if root is None:
            continue
        cp = classes_path(root)
        if not cp.is_file():
            continue
        try:
            candidate = attribute_schema_digest(read_registry(cp), subject)
        except (OSError, RegistryError):
            candidate = None
        if candidate is not None:
            digest, src_classes = candidate, cp
            break

    by_split: dict[str, dict[str, dict[str, str]]] = {}
    for split_name, split_stems in parts.items():
        names = {logical_image_name(image_map[s]) for s in split_stems if s in image_map}
        carried = {n: negatives[n] for n in sorted(set(negatives) & names)}
        if carried:
            by_split[split_name] = carried
    return _NegativeCarry(by_split=by_split, contradicted=contradicted, digest=digest,
                          src_classes=src_classes)


def _apply_negative_carry(carry: "_NegativeCarry | None", out_dir: Path,
                          subject: str | None) -> None:
    """Writes what :func:`_compute_negative_carry` found into each split's own status store. Each
    confirmation is copied whole, so the split records who confirmed the image and when, the source
    dataset's own answer, rather than re-attributing the human's work to the split writer."""
    if carry is None or not carry.by_split or not subject:
        return
    from tcip_mcp.class_registry import copy_registry
    from tcip_mcp.dataset_layout import (
        classes_path, replace_image_status_store, stamp_image_status_digests, status_bucket,
    )

    bucket_key = status_bucket(subject, None)
    for split_name, carried in carry.by_split.items():
        split_root = out_dir / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        replace_image_status_store(split_root, {bucket_key: carried})
        if carry.digest is not None and carry.src_classes is not None:
            copy_registry(carry.src_classes, classes_path(split_root))
            stamp_image_status_digests(split_root, bucket_key, carried, carry.digest)

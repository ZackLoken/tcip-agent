"""Data management tools: census a dataset, split data. Per-file quality checks live in
the ``doctor`` command's ``check_data_quality``."""

from __future__ import annotations

from pathlib import Path

import tcip_store

from tcip_mcp.server import mcp
from tcip_mcp.audit import audited


@mcp.tool()
@audited
def freeze_selection(experiment_id: str, output_path: str | None = None) -> dict:
    """Freeze a finished run's own drawn train/val partition into a selection, so a later run can
    bind to the identical partition instead of drawing its own.

    Reads the run's ``split.json`` (through ``read_run_partition_checked``: a record that will
    not decode refuses rather than reading as absent) and its durable config, and refuses,
    naming the primitive, when: no split record exists for ``experiment_id`` or it does not
    decode; the run was bound to a selection already (``selection_binding``: bind to that
    selection directly instead); the split is spatial (region identities, not stems); the record
    carries no per-scope ``members`` block at all, so which directory each member's ground truth
    lives under is not on record and freezing would name every one of them under this run's own
    ``data.labels_dir``; the record names a member whose ground truth lives under a scope other
    than that directory, which a validation directory the caller named is (every member is
    re-produced under the one directory below, so a member from elsewhere cannot be composed);
    the run's val side is empty (it trained without
    validation, a partition no bind can use); the task is not ``detection``/``instance_seg``; the
    config's ``data`` section carries no ``subject``, ``labels_dir``, ``images_dir`` or
    ``id_map``; the record carries no ``group_by`` at all (no grouping policy recorded; never
    defaulted to ``"stem"``); the record's ``dataset_hash`` is ``None`` (the
    run recorded no labels hash, so staleness cannot be checked); the labels changed since the
    run (``dataset_hash(labels_dir)`` now differs from the one ``split.json`` recorded, both
    named; the comparison is over the whole labels directory, wider than the run's own members'
    per-stem digests beside it, so any change anywhere under the labels directory refuses
    freezing, not only a change to the run's own stems: a stem this freeze would compose from
    that changed is a member whose ground truth moved, and one outside it that changed is a draw
    over data that is no longer what the run saw, so draw a fresh split over the current data
    instead); or a selection already exists at the output directory.

    The frozen selection's ``calibration`` side is always empty: freezing a training run's own
    train/val draw records no calibration draw, so the calibration doors' own floor refuses any
    calibration measurement against it by name, and this tool's own answer carries a ``note``
    saying so instead of quietly minting a selection that looks whole but cannot calibrate.

    Args:
        experiment_id: The finished run to freeze the drawn partition of, by its record id (one
            run's immutable record, ``tcip_mcp.experiments``).
        output_path: Where to write the selection. Defaults to
            ``<dataset_root>/splits/frozen-<experiment_id>``, resolved from ``data.images_dir``
            through ``dataset_root_of``; refused when that does not resolve (an images directory
            outside the canonical ``<dataset_root>/images/...`` layout).
    """
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.experiments import config_key, read_member, read_run_partition_checked
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.label_queries import admission_date, directory_samples
    from tcip_mcp.pipelines.data.selection import (
        Selection, read_selection_checked, write_selection,
    )
    from tcip_mcp.pipelines.data.splits import (
        GROUP_KEY_FNS, member_identity, recorded_group_key_fn,
    )
    from tcip_mcp.pipelines.operating_point import is_the_same_labels_dir
    from tcip_mcp.pipelines.model_build import MODEL_SOURCE_KEY
    from tcip_mcp.pipelines.resolution import dataset_hash as _dataset_hash
    from tcip_mcp.pipelines.resolution import label_digests as _label_digests

    split, decode_error = read_run_partition_checked(experiment_id)
    if decode_error is not None:
        return {"error": f"the split record for {experiment_id!r} could not be read: {decode_error}"}
    if not split:
        return {"error": f"no split record for {experiment_id!r}: this run wrote no split.json "
                         "(it never reached a real dataset build), so there is no drawn "
                         "partition to freeze."}
    if split.get("selection_binding"):
        bound_dir = split["selection_binding"].get("selection_dir")
        if split.get("redrawn_within_selection"):
            return {"error": f"{experiment_id!r} redrew train and val inside the selection at "
                             f"{bound_dir!r} already: reproduce it by binding a later run to that "
                             "same selection with the same seed and "
                             "data.split.redraw_within_selection=true, with the labels this run's "
                             "own split.json recorded (label_digests.at_run) unchanged since "
                             "(the redraw reads per-stem annotation counts at run time, not "
                             "only the selection's fixed membership), never by freezing."}
        return {"error": f"{experiment_id!r} was bound to a selection already ({bound_dir!r}): "
                         "bind a later run to that selection directly instead of freezing "
                         "this one's."}
    resolved_group_by = split.get("group_by")
    if resolved_group_by == "spatial_strip":
        return {"error": f"{experiment_id!r}'s split is spatial (region identities, not stems): "
                         "freeze_selection binds a stem-keyed partition, which a spatial split "
                         "never draws."}
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
        return {"error": f"{experiment_id!r} trained task={task!r}; freeze_selection binds only "
                         "detection and instance_seg runs, the tasks whose per-image label "
                         "documents a selection's samples name."}
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
                         "freeze_selection needs every one of them to compose a selection."}
    assert labels_dir is not None and images_dir is not None and id_map is not None, \
        "checked non-empty above"
    # Every member is re-produced under this one labels directory below, so the record has to say
    # its members live there. Absence of that evidence is not evidence of one scope.
    if not isinstance(split.get("members"), dict):
        return {"error": f"{experiment_id!r}'s split records no per-scope membership, only flat "
                         "train and val lists: which directory each member's ground truth lives "
                         "under is not on record, so freezing would name every one of them under "
                         f"data.labels_dir ({labels_dir}) and a later bind could read different "
                         "ground truth than this run validated on. Draw a selection over the "
                         "current data with draw_splits, or freeze a run trained under the "
                         "current tree, which records membership per scope."}
    elsewhere = sorted(d for d in (split.get("labels_dirs") or [])
                       if not is_the_same_labels_dir(d, labels_dir))
    if elsewhere:
        return {"error": f"{experiment_id!r}'s split records members whose ground truth lives "
                         f"under {', '.join(elsewhere)}, not under this run's data.labels_dir "
                         f"({labels_dir}): a validation side the caller named a directory for, "
                         "or a bind spanning several, is not a partition of one dataset this tool "
                         "can freeze. Draw a selection over the current data instead."}
    if resolved_group_by is None:
        return {"error": f"{experiment_id!r}'s split record carries no group_by at all (no "
                         "grouping policy recorded): freeze_selection never defaults one, "
                         "since guessing 'stem' could silently misstate the policy the run "
                         "actually drew under."}

    labels_hash_at_split = split.get("dataset_hash")
    if labels_hash_at_split is None:
        return {"error": f"{experiment_id!r}'s split record carries no dataset_hash (the run "
                         "recorded no labels hash at draw time): freeze_selection cannot "
                         "check the labels have not moved since, so it refuses rather than "
                         "freezing a partition it cannot vouch for."}
    labels_hash_now = _dataset_hash(labels_dir)
    if labels_hash_now != labels_hash_at_split:
        return {"error": f"the labels under {labels_dir!r} changed since {experiment_id!r} "
                         f"trained (dataset_hash was {labels_hash_at_split!r}, is now "
                         f"{labels_hash_now!r}): the comparison is over the whole labels "
                         "directory, deliberately wider than the per-stem digests the record "
                         "carries, so any change anywhere under it refuses freezing, not only "
                         "a change to the run's own stems; freeze a run whose labels have not "
                         "moved, or draw a fresh split over the current data."}

    dataset_root = dataset_root_of(images_dir)
    if dataset_root is None:
        return {"error": f"data.images_dir={images_dir!r} does not resolve under a dataset "
                         "root (dataset_root_of)."}
    if output_path is None:
        output_path = str(dataset_root / "splits" / f"frozen-{experiment_id}")
    out_dir = Path(output_path)
    existing, existing_error = read_selection_checked(out_dir)
    if existing is not None or existing_error is not None:
        return {"error": f"a selection already exists at {output_path!r}: "
                         f"{existing_error or 'freeze_selection never overwrites one.'}"}

    recorded_map = split.get("group_key_map") or {}
    date = admission_date(labels_dir)
    named_key_fn = (recorded_group_key_fn(resolved_group_by, date=date)
                    if resolved_group_by in GROUP_KEY_FNS else None)

    def group_of(stem: str) -> str:
        """This stem's group key, spelled the way every producer of one spells it.

        Through ``member_identity`` and the run's own recorded policy, so the frozen selection
        carries the keys the run it froze drew under rather than a third spelling of them.
        """
        identity = member_identity(date, stem)
        if resolved_group_by == "explicit_map":
            return recorded_map[identity]
        return named_key_fn(stem) if named_key_fn is not None else identity
    all_stems = sorted(set(train_stems) | set(val_stems))
    digests = _label_digests(labels_dir, all_stems)
    assignment = {stem: "train" for stem in train_stems}
    assignment.update({stem: "val" for stem in val_stems})
    try:
        samples = directory_samples(
            assignment, images_dir=images_dir, labels_dir=labels_dir, subject=str(subject),
            group_of=group_of, digests=digests,
        )
    except KeyError as exc:
        return {"error": f"{experiment_id!r}'s split record groups by an explicit map that names "
                         f"no key for {exc}: freeze_selection never defaults a group key."}
    except (FileNotFoundError, ValueError) as exc:
        return {"error": f"a member of {experiment_id!r}'s partition no longer resolves to an "
                         f"image under {images_dir!r}: {exc}"}

    try:
        fingerprint = dataset_fingerprint(dataset_root)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the frozen selection: {exc}"}

    from datetime import datetime, timezone

    total = len(train_stems) + len(val_stems)
    selection = write_selection(out_dir, Selection(
        samples=tuple(samples), subject=subject,
        attribute=data_cfg.get("attribute") or None, id_map=id_map,
        seed=int(split.get("seed", 42)), group_by=resolved_group_by,
        dataset_fingerprint=fingerprint, admission_counts={},
        realized_ratios={
            "train": len(train_stems) / total if total else 0.0,
            "val": len(val_stems) / total if total else 0.0,
            "calibration": 0.0,
        },
        origin={"experiment_id": experiment_id,
               "frozen_at": datetime.now(timezone.utc).isoformat()},
    ))
    return {
        "selection_dir": str(out_dir), "train": len(train_stems), "val": len(val_stems),
        "calibration": 0, "origin": selection.origin,
        "note": "the calibration side is empty (a training run's own drawn partition records "
               "no calibration draw): the calibration doors' own floor refuses any calibration "
               "measurement against this selection by name.",
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
    ``predictions`` drops every bucket under the cleared archive
    (``tcip_mcp.dataset_layout.is_cleared_bucket``), so a document
    :func:`~tcip_mcp.tools.inference_tools.clear_prediction_bucket` has moved out from under a
    terminal experiment's own path is never counted as a live prediction here.

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
    from tcip_mcp.dataset_layout import (
        annotation_root, image_root, is_cleared_bucket, prediction_root,
    )
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
            if not is_cleared_bucket(bucket)
            for f in prediction_documents(bucket)
        ]

    return {
        "images": images, "labels": labels, "predictions": preds, "format": detected_format,
        "reserved_name_labels": reserved_name_labels, "reserved_name_images": reserved_name_images,
    }


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
    admission for the tasks a selection can bind to draws through the per-image tree, never
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
    subject: str | None = None,
    attribute: str | None = None,
) -> dict:
    """Compute a leakage-free, annotation-stratified train/val/calibration selection.

    Non-destructive: it copies nothing and moves nothing. With ``output_path`` it writes a
    selection record listing, per sample, the image source, the label document, the group key and
    the side; without one it answers statistics over the tree and writes nothing. Sibling tiles of
    one source image are kept in the same split (no tree- or canopy-level leakage), and, when
    ``stratify_foreground`` is set, splits are balanced by annotation count so dense and sparse
    sources are proportionally represented. Groups whole source images; a within-image split for a
    folder holding a single source is a training run's own automatic route (``data.tiling`` in the
    run config), not this tool.

    Writing a selection draws its samples through the platform's own admission for the tasks a
    selection can bind to (``tcip_mcp.pipelines.data.label_queries.trainable_stems``, the same
    function a training run's own draw uses): for each capture date the dataset holds, every image
    carrying an annotation of ``subject`` (with every instance assessed for ``attribute``, when one
    is given) or a human's negative confirmation for it. ``subject`` is therefore required to write
    a selection; a call with no ``output_path`` answers over every image in the tree instead, no
    subject needed. Every admitted date enters one selection: a sample names its own source and its
    own label, so a selection spanning capture dates trains in place, with no derived folder of
    copied imagery, and two dates holding a same-named image are two samples. ``stratify_foreground``
    only toggles the annotation-count balancing; it does not change which images are eligible.

    The third side, ``calibration``, is the universe every calibration drawn under this selection
    draws from (the operating point is measured on it, never on ``train`` or ``val``); writing a
    selection therefore has no default for any of the three ratios and refuses a zero one, naming
    it. The draw refuses, before any write, when the tree holds fewer foreground groups of
    ``subject`` (and ``attribute``, when scoped) than the three sides need at minimum (one each
    for ``train``/``val``, two for ``calibration``, so the locked calibration/holdout draw the
    calibration door makes later can still halve it), counted for the draw's own subject
    regardless of ``stratify_foreground``: that flag only toggles the balancing pass, never
    whether the minimum pass sees real foreground. The answer's
    ``calibration_foreground_groups`` then reports how many of the calibration side's own groups
    actually carry a foreground annotation. Both the answer and the record also carry
    ``realized_ratios``, each side's share of the draw actually delivered: on a tree sized at the
    floor, the minimum pass can consume every foreground group before the balancing pass ever
    sees the caller's fractions, so the delivered shares can diverge from the ratios asked for,
    and this states the shape actually drawn beside them.

    Args:
        folder_path: Path to the dataset root directory.
        train_ratio: Fraction for training set. Defaults to 0.8, the complement of the
            unchanged 0.2 validation default once ``calibration_ratio`` is 0.
        val_ratio: Fraction for validation set.
        calibration_ratio: Fraction held out as the calibration universe. Defaults to 0.0 for a
            stats-only call, and a non-zero value is admitted there too. Writing a selection
            (``output_path`` given) refuses a zero ratio on any of the three (``train_ratio``,
            ``val_ratio``, ``calibration_ratio``), naming it, since a selection always draws all
            three sides.
        seed: Random seed for reproducibility.
        group_by: Group selector: ``"tile_prefix"`` (strip a trailing ``_<x>_<y>`` tile offset) or
            ``"stem"`` (one group per member). Ignored when ``group_key_map`` is given. The
            resolved key is recorded on every sample, so a consumer groups by what was drawn
            rather than re-resolving a policy.
        group_key_map: An agent-derived ``{identity: group_key}`` map overriding ``group_by``,
            keyed ``<date>/<stem>`` (the bare ``<stem>`` under a flat tree); must cover every
            admitted member. Recorded as ``group_by="explicit_map"``.
        stratify_foreground: Balance splits by foreground annotation count.
        output_path: Where to write the selection. Omitted, nothing is written and the answer is
            statistics only.
        subject: The object class the selection is drawn for. Required to write one
            (``output_path`` given): it governs which images the draw admits, and a run binding
            the selection reads it from there rather than restating it.
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
    from tcip_mcp.pipelines.data.selection import SIDES, Sample, Selection, write_selection
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines,
        group_balanced_split,
        member_identity,
        refuse_insufficient_foreground_groups,
        resolve_group_key_fn,
    )
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused

    kept_splits = SIDES
    out_dir = Path(output_path) if output_path else None
    if out_dir is not None:
        zero_ratios = [name for name, ratio in (
            ("train_ratio", train_ratio), ("val_ratio", val_ratio),
            ("calibration_ratio", calibration_ratio),
        ) if ratio == 0]
        if zero_ratios:
            return {"error": f"{', '.join(zero_ratios)} must be non-zero to write a selection: a "
                             "selection's three sides are all drawn from, so writing one states "
                             "all three ratios (train_ratio, val_ratio, calibration_ratio) as "
                             "non-zero. Omit output_path for a stats-only call, whose ratios may "
                             "include a zero."}

    if out_dir is None:
        # A stats-only call writes nothing, so the draw is a plain image/label scan: no subject
        # required, every image eligible.
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

            stems_by_date: dict[str, list[str]] = {}
            for stem in stems:
                label_path = label_map.get(stem)
                if label_path is None:
                    continue
                stems_by_date.setdefault(annotation_date(label_path) or "", []).append(stem)
            for date_key, date_stems in stems_by_date.items():
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
            "selection_dir": None,
        }

    # Writing a selection: the draw is the platform's own per-subject admission.
    if not subject:
        return {"error": "draw_splits needs subject to write a selection (output_path given): "
                         "pass the object class the run will admit under, or drop output_path "
                         "for a stats-only call."}

    from tcip_mcp.pipelines.data.label_queries import (
        directory_samples, resolve_registry_id_map, trainable_stems,
    )
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, BandGroupIncomplete
    from tcip_mcp.pipelines.resolution import label_digests as _label_digests

    date_dirs = _split_date_dirs(folder_path)
    if not date_dirs:
        return {"error": f"{folder_path} holds no per-image label tree (annotations/<date>/ or a "
                         "flat annotations/) for draw_splits to draw a subject-scoped selection "
                         "from; a dataset-level assembled COCO at the root is not walked here."}

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
                         f"directory ({detail}): one image file would be admitted once per entry "
                         "and could land on both sides of the split. Give each date its own "
                         "images/<date>/ bucket, or merge the colliding label entries into one."}
    try:
        _, id_map = resolve_registry_id_map(date_dirs[0][1], subject, attribute)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot resolve the subject registry for the selection: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}

    # identity -> (date, stem, labels_dir, images_dir); the identity keys the draw and is what a
    # caller-supplied group_key_map is keyed by.
    located: dict[str, tuple[str | None, str, Path, Path]] = {}
    admission_counts: dict[str, int] = {}
    drawn: dict[str, list[str]] = {}
    try:
        for date, labels_dir, images_dir in date_dirs:
            admitted, counts = trainable_stems(
                labels_dir, images_dir, subject=subject, date=date, attribute=attribute,
                id_map=id_map,
            )
            for key, value in counts.items():
                admission_counts[key] = admission_counts.get(key, 0) + value
            for stem in admitted:
                located[member_identity(date, stem)] = (date, stem, labels_dir, images_dir)

        stems = sorted(located)
        foreground_counts = {
            identity: count_label_lines(labels_dir, stem, subject=subject, attribute=attribute)
            for identity, (_date, stem, labels_dir, _images) in located.items()
        }
        annotation_counts = foreground_counts if stratify_foreground else None

        realized_ratios: dict[str, float] = {}
        calibration_foreground_groups = 0
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
            drawn = group_balanced_split(
                stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
                splits=(train_ratio, val_ratio, calibration_ratio), seed=seed,
                min_foreground_groups=min_foreground_groups, foreground_counts=foreground_counts,
            )
            total_drawn = sum(len(drawn[k]) for k in kept_splits)
            realized_ratios = {
                k: (len(drawn[k]) / total_drawn if total_drawn else 0.0) for k in kept_splits
            }
            calibration_foreground_groups = len({
                group_key_fn(identity) for identity in drawn["calibration"]
                if foreground_counts.get(identity, 0) > 0
            })
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

    digests_by_labels_dir: dict[Path, dict[str, str]] = {}
    for _date, _stem, labels_dir, _images in located.values():
        digests_by_labels_dir.setdefault(labels_dir, {})
    for labels_dir in digests_by_labels_dir:
        here = sorted({stem for _d, stem, ld, _i in located.values() if ld == labels_dir})
        digests_by_labels_dir[labels_dir] = _label_digests(labels_dir, here)

    # One producer for both paths: a drawn run's own loaders resolve their samples through this
    # same function, so the two hold one membership shape and read the same recorded paths.
    assignment_by_dir: dict[tuple[Path, Path], dict[str, str]] = {}
    group_by_dir: dict[tuple[Path, Path], dict[str, str]] = {}
    for side in kept_splits:
        for identity in sorted(drawn[side]):
            _date, stem, member_labels, member_images = located[identity]
            pair = (member_labels, member_images)
            assignment_by_dir.setdefault(pair, {})[stem] = side
            group_by_dir.setdefault(pair, {})[stem] = group_key_fn(identity)
    samples: list[Sample] = []
    try:
        for (member_labels, member_images), assignment in sorted(assignment_by_dir.items()):
            groups = group_by_dir[(member_labels, member_images)]
            samples.extend(directory_samples(
                assignment, images_dir=member_images, labels_dir=member_labels, subject=subject,
                group_of=groups.__getitem__,
                digests=digests_by_labels_dir[member_labels],
            ))
    except (FileNotFoundError, AmbiguousImageStem, BandGroupIncomplete) as exc:
        return {"error": str(exc)}

    try:
        fingerprint = dataset_fingerprint(folder_path)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the selection: {exc}"}
    try:
        write_selection(out_dir, Selection(
            samples=tuple(samples), subject=subject, attribute=attribute or None, id_map=id_map,
            seed=seed, group_by=resolved_group_by, dataset_fingerprint=fingerprint,
            admission_counts=admission_counts, realized_ratios=realized_ratios,
        ))
    except ValueError as exc:
        return {"error": str(exc)}

    counts = annotation_counts or {}
    return {
        "splits": {k: len(drawn[k]) for k in kept_splits},
        "foreground_annotations": {
            k: sum(int(counts.get(s, 0)) for s in drawn[k]) for k in kept_splits
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
        "selection_dir": str(out_dir),
        "calibration_foreground_groups": calibration_foreground_groups,
        "realized_ratios": realized_ratios,
    }

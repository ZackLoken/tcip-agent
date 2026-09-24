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

    A run of any task freezes, whatever shape its ground truth is: the record says per scope which
    members it holds, which file answered for each, where its pixels came from and what that file
    digested to at run time, and the selection is composed scope by scope from exactly those, so
    nothing here reads the config for a place, compares one against the record, or reads two
    scopes' same-named members as one.

    Reads the run's ``split.json`` (through ``read_run_partition_checked``: a record that will
    not decode refuses rather than reading as absent) and its durable config, and refuses,
    naming the primitive, when: no split record exists for ``experiment_id`` or it does not
    decode; the run was bound to a selection already (``selection_binding``: bind to that
    selection directly instead); the split is spatial (region identities, not stems); the record
    carries no per-scope ``members`` block at all, so which scope each member's ground truth lives
    under is not on record; the run's val side is empty (it trained without validation, a
    partition no bind can use); the config's ``data`` section carries no ``subject`` or
    ``id_map`` for ground truth that is per-image label documents, whose admission is
    subject-scoped; the record carries no ``group_by`` at all (no grouping policy recorded; never
    defaulted to ``"stem"``); a scope block carries no ``label_digests.at_run``, no
    ``label_digests.ground_truth``, no ``sources`` entry or no ``group_key_map`` for a member (the
    run recorded no per-member digest, path, source or group key, so the partition cannot be
    vouched for); a member's ground truth has moved since the run (the file the record names now
    digests differently, the moved members named), since a selection composed from it would bind a
    later run to ground truth this one never saw; or a selection already exists at the output
    directory.

    The frozen selection's ``calibration`` side is always empty: freezing a training run's own
    train/val draw records no calibration draw, so the calibration doors' own floor refuses any
    calibration measurement against it by name, and this tool's own answer carries a ``note``
    saying so instead of quietly minting a selection that looks whole but cannot calibrate.

    Args:
        experiment_id: The finished run to freeze the drawn partition of, by its record id (one
            run's immutable record, ``tcip_mcp.experiments``).
        output_path: Where to write the selection. Defaults to
            ``<dataset_root>/splits/frozen-<experiment_id>``, resolved from the run's own recorded
            sources through ``dataset_root_of``; refused when that does not resolve (a source
            outside the canonical ``<dataset_root>/images/...`` layout).
    """
    from tcip_mcp.dataset_layout import dataset_root_of, status_bucket
    from tcip_mcp.experiments import config_key, read_member, read_run_partition_checked
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.label_queries import Admitted, admission_date, samples_over
    from tcip_mcp.pipelines.data.selection import (
        DOCUMENT, ClassScope, Sample, Selection, read_selection_checked, shape_of, write_selection,
    )
    from tcip_mcp.pipelines.resolution import members_moved_since

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
    if not isinstance(split.get("members"), dict):
        return {"error": f"{experiment_id!r}'s split records no per-scope membership: which scope "
                         "each member's ground truth lives under, which file answered for it and "
                         "where its pixels came from are not on record, so this partition cannot "
                         "be composed. Draw a selection over the current data with draw_splits, "
                         "or freeze a run trained under the current tree, which records "
                         "membership per scope."}
    if resolved_group_by is None:
        return {"error": f"{experiment_id!r}'s split record carries no group_by at all (no "
                         "grouping policy recorded): freeze_selection never defaults one, "
                         "since guessing 'stem' could silently misstate the policy the run "
                         "actually drew under."}

    config = read_member(config_key(experiment_id), {})
    config = config if isinstance(config, dict) else {}
    # The run's own class space, through its one reader, so what this selection records is what
    # the checkpoint records.
    scope = ClassScope.recorded_in(config.get("data") or {})
    # Composed scope by scope, the way the record is written: a bare member name means one image
    # only within its own scope, so two scopes' same-named members stay two members here.
    unrecorded: list[str] = []
    moved: list[str] = []
    shapes: set[str] = set()
    samples: list[Sample] = []
    n_train = n_val = 0
    for gt_scope, block in sorted(split["members"].items()):
        block = block if isinstance(block, dict) else {}
        label_digests = block.get("label_digests") or {}
        recorded = label_digests.get("ground_truth") or {}
        digests = label_digests.get("at_run") or {}
        groups = block.get("group_key_map") or {}
        sources = block.get("sources") or {}
        train_stems, val_stems = block.get("train") or [], block.get("val") or []
        n_train, n_val = n_train + len(train_stems), n_val + len(val_stems)
        here = sorted(set(train_stems) | set(val_stems))
        unrecorded.extend(
            m for m in here
            if m not in recorded or m not in digests or m not in groups or m not in sources)
        if unrecorded:
            continue
        # A member whose ground truth is the scope itself is one row of it, by its own row key.
        records = [Admitted(member=m, source=sources[m], ground_truth=recorded[m],
                            row_key=m if recorded[m] == gt_scope else None)
                   for m in here]
        shapes.update(shape_of(r.ground_truth, r.row_key) for r in records)
        try:
            moved.extend(members_moved_since(recorded, digests, here))
        except ValueError as exc:
            return {"error": f"{experiment_id!r}'s recorded ground truth cannot be read per "
                             f"member: {exc}"}
        assignment = {stem: "train" for stem in train_stems}
        assignment.update({stem: "val" for stem in val_stems})
        # Through the one producer, over this scope's own sources, paths and group keys.
        samples.extend(samples_over(
            records, assignment, groups.__getitem__,
            confirmation_bucket=(status_bucket(scope.subject, admission_date(gt_scope))
                                 if scope.subject else None),
            digests=digests,
        ))
    if not n_val:
        return {"error": f"{experiment_id!r} trained without validation (an empty val side): "
                         "a partition no bind can use."}
    if unrecorded:
        return {"error": f"{experiment_id!r}'s split record carries no ground-truth path, digest, "
                         f"group key or image source for {sorted(unrecorded)[:5]} "
                         f"({len(unrecorded)} member(s)): a member the record does not fully "
                         "describe cannot be composed or checked for movement, so this partition "
                         "cannot be vouched for."}
    if DOCUMENT in shapes:
        # A label document's admission is subject-scoped and reads the class map the run trained
        # in; a mask raster and a table row are admitted by existing and scope no class space.
        missing = [name for name, value in
                   (("subject", scope.subject), ("id_map", scope.id_map)) if not value]
        if missing:
            return {"error": f"{experiment_id!r}'s durable config carries no {missing}: "
                             "freeze_selection needs every one of them to compose a selection "
                             "over per-image label documents."}
    if moved:
        return {"error": f"the ground truth of {len(moved)} member(s) changed since "
                         f"{experiment_id!r} trained ({sorted(moved)[:5]}): a selection "
                         "composed from them would bind a later run to ground truth this run "
                         "never saw. Freeze a run whose members have not moved, or draw a fresh "
                         "split over the current data."}

    a_source = samples[0].source
    dataset_root = dataset_root_of(a_source)
    if dataset_root is None:
        return {"error": f"this run's own source {a_source!r} does not resolve under a dataset "
                         "root (dataset_root_of), so the frozen selection has no dataset to "
                         "record a fingerprint for."}
    if output_path is None:
        output_path = str(dataset_root / "splits" / f"frozen-{experiment_id}")
    out_dir = Path(output_path)
    existing, existing_error = read_selection_checked(out_dir)
    if existing is not None or existing_error is not None:
        return {"error": f"a selection already exists at {output_path!r}: "
                         f"{existing_error or 'freeze_selection never overwrites one.'}"}

    try:
        fingerprint = dataset_fingerprint(dataset_root)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the frozen selection: {exc}"}

    from datetime import datetime, timezone

    total = n_train + n_val
    selection = write_selection(out_dir, Selection(
        samples=tuple(samples), subject=scope.subject if DOCUMENT in shapes else None,
        attribute=scope.attribute if DOCUMENT in shapes else None,
        id_map=dict(scope.id_map or {}) if DOCUMENT in shapes else {},
        seed=int(split.get("seed", 42)), group_by=resolved_group_by,
        dataset_fingerprint=fingerprint, admission_counts={},
        realized_ratios={
            "train": n_train / total if total else 0.0,
            "val": n_val / total if total else 0.0,
            "calibration": 0.0,
        },
        origin={"experiment_id": experiment_id,
               "frozen_at": datetime.now(timezone.utc).isoformat()},
    ))
    return {
        "selection_dir": str(out_dir), "train": n_train, "val": n_val,
        "calibration": 0, "origin": selection.origin,
        "note": "the calibration side is empty (a training run's own drawn partition records "
               "no calibration draw): the calibration doors' own floor refuses any calibration "
               "measurement against this selection by name.",
    }


def _scan_dataset(root: str) -> dict:
    """Scan a directory tree for images and labels.

    Labels are the name-based per-image JSON (one file per image, all subjects) under
    ``annotations/<date>/`` (no detect/segment split), a review baseline directory's copies
    excluded. The census reads no document: what each holds is its reader's answer, asked per
    file by the doctor's own ``check_data_quality``.

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
        LABEL_SUFFIX, annotation_root, image_root, is_cleared_bucket, label_filename,
        prediction_root,
    )
    from tcip_mcp.pipelines.image_utils import BandGroupRef, IMAGE_EXTS, list_logical_images

    root_path = Path(root)
    image_exts = IMAGE_EXTS
    images: list[str] = []
    labels: list[str] = []
    preds: list[str] = []
    reserved_name_labels: list[str] = []
    reserved_name_images: list[str] = []

    # Find images through the platform's own bucket enumeration: a stem collision refuses here
    # too, and a grouped capture counts once, its own manifest.
    images_dir = image_root(root_path)
    if images_dir.is_dir():
        buckets = [images_dir] + sorted(p for p in images_dir.iterdir() if p.is_dir())
        for bucket in buckets:
            for source in list_logical_images(bucket).values():
                f = source.manifest_path if isinstance(source, BandGroupRef) else source
                images.append(str(f))
                if is_sidecar_name(label_filename(f.stem)):
                    reserved_name_images.append(str(f))
    else:
        # No canonical images/ tree, so no bucket contract to route through this walk.
        for f in sorted(root_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in image_exts:
                images.append(str(f))
                if is_sidecar_name(label_filename(f.stem)):
                    reserved_name_images.append(str(f))

    # Ground-truth labels: annotations/[<date>/]<stem>.json (one file per image, every subject),
    # a review baseline copy under BASELINE_DIRNAME excluded: it is a snapshot, not a label.
    ann_dir = annotation_root(root_path)
    if ann_dir.is_dir():
        labels = [
            str(f) for f in sorted(ann_dir.rglob(f"*{LABEL_SUFFIX}"))
            if f.is_file() and BASELINE_DIRNAME not in f.parts
        ]
        reserved_name_labels = [f for f in labels if is_sidecar_name(Path(f).name)]

    # Predictions: predictions/<model>/[<date>/]<stem>.json; each model/date bucket is walked on
    # its own through prediction_documents, so the bucket's own stamps are excluded everywhere.
    pred_dir = prediction_root(root_path)
    if pred_dir.is_dir():
        preds = [
            str(f)
            for bucket in sorted({p.parent for p in pred_dir.rglob(f"*{LABEL_SUFFIX}")})
            if not is_cleared_bucket(bucket)
            for f in prediction_documents(bucket)
        ]

    return {
        "images": images, "labels": labels, "predictions": preds,
        "reserved_name_labels": reserved_name_labels, "reserved_name_images": reserved_name_images,
    }


def scan_dataset(folder_path: str) -> dict:
    """Scan a folder for images, labels, and predictions.

    Not an MCP tool: run through ``tcip scan-dataset``, per the admission standard
    (packages/tcip-mcp/CLAUDE.md), while staying importable for its in-package callers.

    Reads the name-based per-image JSON labels (one file per image, all subjects).

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

    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem
    from tcip_store import SchemaVersionRefused

    try:
        scan = _scan_dataset(folder_path)
    except AmbiguousImageStem as exc:
        return {"error": str(exc)}
    except SchemaVersionRefused as exc:
        return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}

    image_stems = {Path(p).stem: p for p in scan["images"]}
    label_stems = {Path(p).stem for p in scan["labels"]}

    paired = sum(1 for stem in image_stems if stem in label_stems)
    unlabelled = len(image_stems) - paired

    return {
        "path": folder_path,
        "image_count": len(scan["images"]),
        "labels_count": len(scan["labels"]),
        "predictions_count": len(scan["predictions"]),
        "paired_images": paired,
        "unlabelled_images": unlabelled,
        "image_stems_sample": sorted(image_stems.keys())[:10],
        "reserved_name_labels": scan["reserved_name_labels"],
        "reserved_name_images": scan["reserved_name_images"],
    }



def _split_date_dirs(folder_path: str | Path) -> list[tuple[str | None, Path, Path]]:
    """Every ``(date, labels_dir, images_dir)`` a dataset's per-image label tree holds: one entry
    per ``annotations/<date>/`` beside its images (``images/<date>/`` when that bucket exists, else
    the flat ``images/`` root), plus one dateless entry for any label loose directly in
    ``annotations/`` beside a dated tree, so a mixed layout's flat labels are never dropped from
    the draw. A fully flat dataset (no date subdirectories at all) yields exactly that one
    dateless entry.

    Empty when the dataset holds no per-image label tree at all.
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
    ground_truth: str | None = None,
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

    Writing a selection is one draw, through the platform's own admission
    (``tcip_mcp.pipelines.data.label_queries.admit``) over whichever ground truth the place it is
    pointed at holds, the same admission a training run's own draw uses. The draw's group policy,
    its three-way group-balanced split and its per-side floor are the same whatever that shape is.

    Without ``ground_truth``, the ground truth is the dataset's per-image label tree: for each
    capture date the dataset holds, every image carrying an annotation of ``subject`` (with every
    instance assessed for ``attribute``, when one is given) or a human's negative confirmation for
    it. ``subject`` is therefore required to write such a selection; a call with no ``output_path``
    answers over every image in the tree instead, no subject needed. Every admitted date enters
    one selection: a sample names its own source and its own label, so a selection spanning
    capture dates trains in place, with no derived folder of copied imagery, and two dates holding
    a same-named image are two samples. ``stratify_foreground`` only toggles the annotation-count
    balancing; it does not change which images are eligible.

    With ``ground_truth`` naming a directory of ``<stem>.png`` rasters, the ground truth is a
    per-image mask and a sample is admitted when its mask sits there beside its image; with
    ``ground_truth`` naming a ``.csv`` file, the ground truth is that table and a sample is one
    row, admitted when the image its key names exists. Neither is subject-scoped and neither reads
    a human confirmation store, so neither takes ``subject``/``attribute``, and the class space a
    run binding such a selection trains in is derived from the ground truth it was handed, once
    for the run. Balancing by foreground count applies to label documents only: a mask or a row is
    admitted by existing, so every admitted member carries ground truth and counts as one.

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
        subject: The object class the selection is drawn for. Required to write one over the
            dataset's own per-image label tree (``output_path`` given and no ``ground_truth``),
            whose admission is subject-scoped: it governs which images the draw admits, and a run
            binding the selection reads it from there rather than restating it. A ``ground_truth``
            naming label documents takes one too; the producer reads it only where the ground
            truth it admits is subject-scoped.
        attribute: Scope the draw to instances already assessed for this attribute of
            ``subject``; an image carrying an instance never assessed for it is excluded
            entirely, the same rail a training run applies. ``None`` draws over every instance of
            ``subject`` regardless of attribute state.
        ground_truth: Where this dataset's ground truth lives, named explicitly rather than
            walked as the per-image label tree: a directory of label documents, a directory of
            ``<stem>.png`` masks, or a ``.csv`` table of one row per image. The producer decides
            which of those it is from the place itself. Only with ``output_path``; the images are
            the dataset's own ``images/`` tree either way.
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
        foreground_group_count,
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
    if ground_truth is not None:
        if out_dir is None:
            return {"error": "draw_splits takes ground_truth only with output_path: a stats-only "
                             "call scans the tree's images and labels and admits nothing."}
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
                annotation_counts = {s: count_label_lines(label_map[s]) for s in stems}
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

    # Writing a selection: one draw, over whatever ground truth the dataset carries, through the
    # producer's own admission for that shape.
    from tcip_mcp.dataset_layout import resolve_images_dir
    from tcip_mcp.pipelines.data.label_queries import (
        Admission, Admitted, admit, foreground_counts, require_admitted,
    )
    from tcip_mcp.pipelines.data.selection import DOCUMENT, with_sides
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, BandGroupIncomplete
    from tcip_mcp.pipelines.resolution import ground_truth_digests

    date_dirs: list[tuple[str | None, Path, Path]] = []
    if ground_truth is not None:
        sources: list[tuple[Path, str]] = [
            (resolve_images_dir(folder_path, None), ground_truth)]
    else:
        if not subject:
            return {"error": "draw_splits needs subject to write a selection (output_path given) "
                             "over the per-image label tree: pass the object class the run will "
                             "admit under, name a ground_truth of another shape, or drop "
                             "output_path for a stats-only call."}
        date_dirs = _split_date_dirs(folder_path)
        if not date_dirs:
            return {"error": f"{folder_path} holds no per-image label tree (annotations/<date>/ "
                             "or a flat annotations/) for draw_splits to draw a subject-scoped "
                             "selection from; an external COCO document is converted into one "
                             "by import_coco first."}
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
                             f"directory ({detail}): one image file would be admitted once per "
                             "entry and could land on both sides of the split. Give each date its "
                             "own images/<date>/ bucket, or merge the colliding label entries "
                             "into one."}
        sources = [(entry_images, str(entry_labels))
                   for _date, entry_labels, entry_images in date_dirs]

    admissions: list[Admission] = []
    admission_counts: dict[str, int] = {}
    # identity -> (source index, admitted record); the identity keys the draw and is what a
    # caller-supplied group_key_map is keyed by.
    located: dict[str, tuple[int, Admitted]] = {}
    try:
        for where, (source_images, source_ground_truth) in enumerate(sources):
            admitted = admit(source_images, source_ground_truth,
                             subject=subject, attribute=attribute)
            admissions.append(admitted)
            for key, value in admitted.counts.items():
                admission_counts[key] = admission_counts.get(key, 0) + value
            for record in admitted.records:
                located[member_identity(admitted.date, record.member)] = (where, record)
    except (UnreadableLabelDocument, AmbiguousImageStem, BandGroupIncomplete) as exc:
        return {"error": str(exc)}
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"a .bandgroup manifest under {folder_path} could not be read: {exc}"}
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {"error": str(exc)}

    stems = sorted(located)
    if not stems:
        searched = ", ".join(
            f"{entry_date or 'annotations/ (loose labels)'} -> {entry_images}"
            for entry_date, _labels, entry_images in date_dirs
        ) or ", ".join(f"{gt} -> {images}" for images, gt in sources)
        unpaired = ""
        if date_dirs:
            from tcip_mcp.dataset_layout import image_dir as _image_dir, list_dates as _list_dates

            used = {entry_images for _d, _l, entry_images in date_dirs}
            buckets = sorted(d for d in _list_dates(folder_path)
                             if _image_dir(folder_path, d) not in used)
            if buckets:
                listed = ", ".join(str(_image_dir(folder_path, d)) for d in buckets)
                unpaired = (f" {listed} exist with no label entry resolved against them; move the "
                            "labels into a matching annotations/<date>/ bucket, or move the "
                            "images to the flat images/ root, to pair them.")
        try:
            require_admitted(admissions[0])
        except ValueError as exc:
            return {"error": f"{exc} Searched {searched} ({admission_counts}).{unpaired}"}

    shape = admissions[0].shape
    try:
        group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    except ValueError as exc:
        return {"error": str(exc)}
    resolved_group_by = "explicit_map" if group_key_map else group_by

    # Through the one producer, once: every admitted member as a sample under the group key this
    # draw groups by, with its ground-truth digest as it reads now. The draw sides them below.
    digest_of = ground_truth_digests(record.ground_truth for _where, record in located.values())
    sample_of: dict[str, Sample] = {}
    try:
        for where, admitted in enumerate(admissions):
            mine = {identity: record for identity, (source, record) in located.items()
                    if source == where}
            groups = {record.member: group_key_fn(identity)
                      for identity, record in mine.items()}
            digests = {record.member: digest_of[record.ground_truth]
                       for record in mine.values()}
            built = admitted.samples({record.member: "train" for record in mine.values()},
                                     groups.__getitem__, digests=digests)
            by_member = sorted(mine.items(), key=lambda entry: entry[1].member)
            sample_of.update(
                (identity, sample) for (identity, _record), sample in zip(by_member, built))
    except (FileNotFoundError, AmbiguousImageStem, BandGroupIncomplete) as exc:
        return {"error": str(exc)}

    # The one foreground count, over the samples themselves: a label document carries its own
    # annotation count, and a mask or a row is admitted by existing, so it counts as one.
    counted = foreground_counts(sample_of, admissions[0].scope)
    annotation_counts = counted if stratify_foreground else None
    min_foreground_groups = {"train": 1, "val": 1, "calibration": 2}
    try:
        refuse_insufficient_foreground_groups(
            foreground_group_count(stems, counted, group_key_fn), min_foreground_groups,
            remedy=("add ground truth for more images: annotate or confirm more of them for this "
                    "subject, or write the masks or rows that answer for them."))
    except ValueError as exc:
        return {"error": str(exc)}
    drawn = group_balanced_split(
        stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
        splits=(train_ratio, val_ratio, calibration_ratio), seed=seed,
        min_foreground_groups=min_foreground_groups, foreground_counts=counted,
    )
    total_drawn = sum(len(drawn[k]) for k in kept_splits)
    realized_ratios = {
        k: (len(drawn[k]) / total_drawn if total_drawn else 0.0) for k in kept_splits
    }
    calibration_foreground_groups = foreground_group_count(
        drawn["calibration"], counted, group_key_fn)

    # The sides the draw assigned, onto the samples already built, by each one's own identity.
    drew = {sample_of[identity].identity: side
            for side in kept_splits for identity in drawn[side]}
    try:
        fingerprint = dataset_fingerprint(folder_path)
    except tcip_store.SchemaVersionRefused as exc:
        return {"error": f"cannot fingerprint the dataset for the selection: {exc}"}
    try:
        write_selection(out_dir, with_sides(Selection(
            samples=tuple(sample for _identity, sample in sorted(sample_of.items())
                          if sample.identity in drew),
            subject=subject if shape == DOCUMENT else None,
            attribute=(attribute or None) if shape == DOCUMENT else None,
            id_map=dict(admissions[0].id_map or {}) if shape == DOCUMENT else {},
            seed=seed, group_by=resolved_group_by, dataset_fingerprint=fingerprint,
            admission_counts=admission_counts, realized_ratios=realized_ratios,
        ), drew))
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
        "stratified": bool(stratify_foreground and shape == DOCUMENT),
        "subject": subject if shape == DOCUMENT else None,
        "attribute": attribute if shape == DOCUMENT else None,
        "admission_counts": admission_counts,
        "selection_dir": str(out_dir),
        "calibration_foreground_groups": calibration_foreground_groups,
        "realized_ratios": realized_ratios,
    }

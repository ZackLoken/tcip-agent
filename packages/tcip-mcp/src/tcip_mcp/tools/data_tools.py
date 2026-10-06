"""Data management tools: census a dataset, split data."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from tcip_mcp.server import tool
from tcip_mcp.audit import audited
from tcip_mcp.pipelines.data.splits import DEFAULT_GROUP_BY, DEFAULT_SHARES


@tool()
@audited
def freeze_selection(project: Path, experiment_id: str, output_path: str | None = None) -> dict:
    """Freeze a finished run's own drawn train/val partition into a selection, so a later run can
    bind to the identical partition instead of drawing its own.

    A run of any task freezes, whatever shape its ground truth is: the selection is the run's own
    train and val samples as its resolved partition records them, each carrying the digest its
    ground truth had when the run read it.

    Reads the run's resolution (``experiments.run_resolution``) and refuses, naming the primitive,
    when: ``experiment_id`` names no training run; the split is spatial (region identities, not
    stems); the run's val side is empty; a member's ground truth has moved since the run (it now
    digests differently from the digest the record holds, the moved members named); a
    selection already exists
    at the output directory; or the selection its resolved ``scope`` composes is one
    :func:`~tcip_mcp.pipelines.data.selection.write_selection` refuses.

    The frozen selection's ``calibration`` and ``holdout`` sides are always empty, so an assessment
    against it refuses by name; this tool's answer carries a ``note`` saying so.

    Args:
        experiment_id: The finished run to freeze the drawn partition of.
        output_path: Where to write the selection. Defaults to
            ``<dataset_root>/splits/frozen-<experiment_id>``, resolved from the run's own recorded
            sources through ``dataset_root_of``; refused when that does not resolve (a source
            outside the canonical ``<dataset_root>/images/...`` layout).
    """
    from tcip_mcp.dataset_layout import dataset_root_of
    from tcip_mcp.experiments import run_resolution
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.selection import (
        ClassScope, Selection, read_selection_checked, write_selection,
    )
    from tcip_annotation.json_io import UnreadableLabelDocument

    from tcip_mcp.pipelines.data.label_queries import acquired
    from tcip_mcp.pipelines.data.split_construction import partition_samples

    try:
        resolved = run_resolution(experiment_id, project=project)
    except ValueError as exc:
        return {"error": str(exc)}
    partition = resolved["partition"]
    if "spatial_manifest" in resolved["data"]["split"]:
        return {"error": f"{experiment_id!r}'s split is spatial (region identities, not stems): "
                         "freeze_selection binds a stem-keyed partition, which a spatial split "
                         "never draws."}
    # The run's own class space, so what this selection records is what the checkpoint records.
    scope = ClassScope.of(resolved["data"])
    samples = [s for s in partition_samples(partition) if s.side in ("train", "val")]
    n_train = sum(1 for s in samples if s.side == "train")
    n_val = len(samples) - n_train
    if not n_val:
        return {"error": f"{experiment_id!r} trained without validation (an empty val side): "
                         "a partition no bind can use."}
    try:
        samples = acquired(samples)
    except (UnreadableLabelDocument, ValueError) as exc:
        return {"error": f"a member's ground truth changed since {experiment_id!r} trained "
                         f"({exc}): a selection composed from it would bind a later run to "
                         "ground truth this run never saw. Freeze a run whose members have not "
                         "moved, or draw a fresh split over the current data."}

    a_source = samples[0].source
    dataset_root = dataset_root_of(a_source)
    if dataset_root is None:
        return {"error": f"this run's own source {a_source!r} does not resolve under a dataset "
                         "root (dataset_root_of), so the frozen selection has no dataset to "
                         "record a fingerprint for."}
    if output_path is None:
        output_path = str(dataset_root / "splits" / f"frozen-{experiment_id}")
    out_dir = Path(output_path)
    existing, existing_error = read_selection_checked(out_dir, project=project)
    if existing is not None or existing_error is not None:
        return {"error": f"a selection already exists at {output_path!r}: "
                         f"{existing_error or 'freeze_selection never overwrites one.'}"}

    try:
        write_selection(out_dir, Selection(
            samples=tuple(samples), scope=scope, seed=partition["seed"],
            group_by=partition["group_by"],
            dataset_fingerprint=dataset_fingerprint(dataset_root, samples),
        ), project=project)
    except ValueError as exc:
        return {"error": f"{experiment_id!r}'s resolved record: {exc}"}
    return {
        "selection_dir": str(out_dir), "train": n_train, "val": n_val, "calibration": 0,
        "holdout": 0,
        "note": "the reference sides are empty (a training run's own drawn partition records "
               "no reference draw): an assessment against this selection refuses by name.",
    }


def _scan_dataset(root: str) -> dict:
    """A dataset's census: ``images``, every logical image of every capture
    (:func:`~tcip_mcp.dataset_layout.list_dates`, each listed through
    :func:`~tcip_mcp.pipelines.image_utils.list_logical_images`, a grouped capture once as its
    manifest) mapped to the key of its own label document
    (:func:`~tcip_mcp.dataset_layout.label_key_of`); ``labels``, the key of every label document
    under the root; ``predictions``, the key of every document a published bucket's record names
    (:func:`~tcip_mcp.buckets.buckets_under`). Reads no document. A stem collision within one
    capture raises :class:`~tcip_mcp.pipelines.image_utils.AmbiguousImageStem`.
    """
    import tcip_store

    from tcip_mcp.buckets import buckets_under
    from tcip_mcp.dataset_layout import LABEL_DOCUMENTS, image_dir, label_key_of, list_dates
    from tcip_mcp.pipelines.image_utils import list_logical_images, source_path_of

    root_path = Path(root).resolve()
    images = [source_path_of(source) for capture in list_dates(root_path)
              for source in list_logical_images(image_dir(root_path, capture)).values()]
    return {
        "images": {image: label_key_of(image) for image in images},
        "labels": tcip_store.keys(LABEL_DOCUMENTS, str(root_path)),
        "predictions": [key for b in buckets_under(root_path) for key in b.document_keys],
    }


def scan_dataset(folder_path: str) -> dict:
    """Count a dataset's images, label documents and prediction documents
    (:func:`_scan_dataset`), and how many images their own label document pairs with.

    Args:
        folder_path: Path to the dataset root directory.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem

    try:
        scan = _scan_dataset(folder_path)
    except AmbiguousImageStem as exc:
        return {"error": str(exc)}

    labels = set(scan["labels"])
    paired = sum(1 for key in scan["images"].values() if key in labels)
    return {
        "path": folder_path,
        "image_count": len(scan["images"]),
        "labels_count": len(scan["labels"]),
        "predictions_count": len(scan["predictions"]),
        "paired_images": paired,
        "unlabeled_images": len(scan["images"]) - paired,
        "images_sample": sorted(scan["images"])[:10],
    }


@tool()
def draw_splits(
    project: Path,
    folder_path: str,
    seed: int,
    val_ratio: float = DEFAULT_SHARES["val"],
    calibration_ratio: float = DEFAULT_SHARES["calibration"],
    holdout_ratio: float = DEFAULT_SHARES["holdout"],
    group_by: str = DEFAULT_GROUP_BY,
    group_key_map: dict[str, str] | None = None,
    stratify_foreground: bool = True,
    output_path: str | None = None,
    subject: str | None = None,
    ground_truth: str | None = None,
) -> dict:
    """Compute a leakage-free, annotation-stratified train/val/calibration/holdout selection.

    Non-destructive: it copies nothing and moves nothing. With ``output_path`` it writes a
    selection record listing, per sample, the image source, the label document, the group key and
    the side, and the project's audit log records the write; without one it answers the same
    draw's statistics and writes nothing, the log included. Sibling tiles
    of one source image are kept in the same split, and, when ``stratify_foreground`` is set,
    splits are balanced by annotation count. Groups whole source images; a within-image split for
    a folder holding a single source is a training run's own route (``data.tiling`` in the run
    config).

    The draw is :func:`~tcip_mcp.pipelines.data.split_construction.admitted_membership` over
    whichever ground truth the place it is pointed at holds, then
    :func:`~tcip_mcp.pipelines.data.split_construction.draw_sides`.

    Without ``ground_truth``, the ground truth is the dataset's label documents: for each capture
    the dataset holds label documents for, every image carrying an annotation of ``subject`` or a
    human's negative confirmation for it, so ``subject`` is required; the selection's scope
    carries every attribute the dataset's registry declares for it. Every admitted capture enters
    one draw: a sample names its own source and its own label document, and two captures holding
    a same-named image are two samples.
    ``stratify_foreground`` only toggles the annotation-count balancing.

    With ``ground_truth`` naming a directory of ``<stem>.png`` rasters, the ground truth is a
    per-image mask and a sample is admitted when its mask sits there beside its image; with
    ``ground_truth`` naming a ``.csv`` file, the ground truth is that table and a sample is one
    row, admitted when the image its key names exists. Neither takes ``subject``.
    Balancing by foreground count applies to label documents only.

    ``train`` takes the remainder of the other sides' shares
    (:func:`~tcip_mcp.pipelines.data.split_construction.check_shares`), each of which defaults to
    :data:`~tcip_mcp.pipelines.data.splits.DEFAULT_SHARES`. The ``calibration`` and ``holdout``
    sides are the reference an assessment fits an operating point on and checks it against, drawn
    as one share and cut between the two at the same seed; a side whose ratio is zero is not
    drawn, and shares leaving ``train`` nothing refuse. The draw refuses, before any write, when
    the tree holds fewer foreground groups of ``subject`` than one per requested side, and when a
    side would be left empty, naming it. The answer's
    ``calibration_foreground_groups`` reports how many of the reference's groups carry a
    foreground annotation, and ``realized_ratios`` each side's share of the draw actually
    delivered, which can diverge from the ratios asked for on a tree sized at the floor.

    Args:
        folder_path: Path to the dataset root directory.
        seed: Random seed the draw is reproduced by; no default.
        val_ratio: Fraction for validation set.
        calibration_ratio: Fraction held out for an assessment to fit its operating point on.
        holdout_ratio: Fraction held out for an assessment to check its operating point against.
        group_by: Group selector: ``"tile_prefix"`` (strip a trailing ``_<x>_<y>`` tile offset) or
            ``"stem"`` (one group per member). Ignored when ``group_key_map`` is given. The
            resolved key is recorded on every sample.
        group_key_map: An agent-derived ``{identity: group_key}`` map overriding ``group_by``,
            keyed ``<capture>/<stem>``; must cover every admitted member. Recorded as
            ``group_by="explicit_map"``.
        stratify_foreground: Balance splits by foreground annotation count.
        output_path: Where to write the selection. Omitted, nothing is written and the answer is
            the draw's statistics only.
        subject: The object class the selection is drawn for. Required over label documents.
        ground_truth: A dataset's ground truth that is not its label documents, named
            explicitly: a directory of ``<stem>.png`` masks, or a ``.csv`` table of one row per
            image; the images are the dataset's own ``images/`` tree either way.
    """
    if not Path(folder_path).is_dir():
        return {"error": f"Directory not found: {folder_path}"}

    import tcip_store

    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.dataset_layout import LABEL_DOCUMENTS, image_dir, list_dates
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
    from tcip_mcp.pipelines.data.selection import (
        REFERENCE_SIDES, SIDES, ClassScope, Selection, write_selection,
    )
    from tcip_mcp.pipelines.data.split_construction import (
        admitted_membership, check_shares, draw_sides,
    )
    from tcip_mcp.pipelines.data.splits import foreground_group_count
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, BandGroupIncomplete

    # Each place holding ground truth: (its name, images directory, ground truth).
    places: list[tuple[str, Path, str | None]]
    if ground_truth is not None:
        places = [(c, image_dir(folder_path, c), ground_truth) for c in list_dates(folder_path)]
        if not places:
            return {"error": f"{folder_path} holds no capture under images/ whose images "
                             f"{ground_truth} could answer for; ingest them with ingest_images."}
    else:
        captures = sorted({key.parts[0] for key in tcip_store.keys(
            LABEL_DOCUMENTS, str(Path(folder_path).resolve()))})
        places = [(c, image_dir(folder_path, c), None) for c in captures]
        if not places:
            return {"error": f"{folder_path} holds no label document for draw_splits to draw a "
                             "subject-scoped selection from; annotate its images, or convert an "
                             "external COCO document into label documents with import_coco."}

    try:
        ratios = check_shares({"val": val_ratio, "calibration": calibration_ratio,
                               "holdout": holdout_ratio})
        membership = admitted_membership(
            places, scope=ClassScope(subject=subject), group_by=group_by,
            group_key_map=group_key_map)
        drawn, counted = draw_sides(
            membership.samples, membership.scope, seed=seed, stratify=stratify_foreground,
            ratios=ratios)
    except (UnreadableLabelDocument, AmbiguousImageStem, BandGroupIncomplete,
            FileNotFoundError, ValueError, OSError) as exc:
        return {"error": str(exc)}

    by_side = {side: [key for key, sample in drawn.items() if sample.side == side]
               for side in SIDES}
    sizes = {side: len(keys) for side, keys in by_side.items()}
    group_of = {key: sample.group for key, sample in drawn.items()}.__getitem__
    response = {
        "splits": sizes,
        "foreground_annotations": {
            side: sum(counted[key] for key in keys) for side, keys in by_side.items()},
        "total_stems": len(drawn),
        "total_annotations": sum(counted.values()),
        "groups": len({sample.group for sample in drawn.values()}),
        "stratified": stratify_foreground,
        "seed": seed, "group_by": membership.group_by, "scope": asdict(membership.scope),
        "tallies": membership.tallies,
        "selection_dir": str(output_path) if output_path else None,
        "calibration_foreground_groups": foreground_group_count(
            [key for side in REFERENCE_SIDES for key in by_side[side]], counted, group_of),
        "realized_ratios": {side: size / len(drawn) for side, size in sizes.items()},
    }
    if not output_path:
        return response
    from tcip_mcp.audit import record_event_or_raise

    try:
        write_selection(output_path, Selection(
            samples=tuple(drawn[key] for key in sorted(drawn)), scope=membership.scope, seed=seed,
            group_by=membership.group_by,
            dataset_fingerprint=dataset_fingerprint(folder_path, drawn.values()),
        ), project=project)
    except ValueError as exc:
        return {"error": str(exc)}
    record_event_or_raise("draw_splits", {
        "folder_path": folder_path, "output_path": output_path, "seed": seed, "ratios": ratios,
        "group_by": membership.group_by, "subject": subject, "ground_truth": ground_truth},
        actor=None, scope=project)
    return response

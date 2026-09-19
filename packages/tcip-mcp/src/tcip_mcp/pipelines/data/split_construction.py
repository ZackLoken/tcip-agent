"""Constructing and persisting training splits from a data config, beside ``splits.py``.

Resolves ``(train_ds, val_ds)`` for a run (explicit val dir, a bound selection, an auto
group-aware draw, or a single-source spatial-strip split) and persists the drawn/bound
membership as the run's ``split.json`` provenance record.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STEM_TASKS = frozenset({"detection", "instance_seg", "semantic_seg", "classification"})
"""Tasks ``auto_train_val``'s drawn (step 2) path covers: a run outside this set never reaches
that branch, so nothing here draws a train/val split for it. Module-level so a caller deciding
whether a task admits the drawn path (``run_hyperparameter_search``'s ``split_draws`` refusal) checks the same set
``auto_train_val`` itself walks, rather than a second copy that could drift from it."""


def dataset_identity(data_cfg: dict) -> tuple[str | None, str | None]:
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset, the content end of the
    reproduce-a-number chain. The fingerprint is recomputed here (recompute-on-read is authority); the
    id comes from the dataset's ``dataset.json`` if it was registered. ``(None, None)`` for a bespoke /
    imageless run (no dataset_root), matching ``dataset_hash=None`` rather than fabricating identity.

    A version-refused identity (``tcip_store.SchemaVersionRefused``) is a real, wrong identity a
    delivered number could rest on, never the same fact as not-registered, so it propagates rather
    than being caught here: ``launch_training`` calls this outside any wrapper and turns the
    exception into a launch refusal, naming the document and its ceiling, rather than silently
    recording ``(None, fp)`` as though the dataset were simply unregistered, or training an
    untracked run against a dataset it cannot verify.
    """
    images_dir = data_cfg.get("images_dir")
    if not images_dir:
        return None, None

    from tcip_store import SchemaVersionRefused

    from tcip_mcp.dataset_layout import dataset_root_of, require_dataset_identity
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint

    root = dataset_root_of(images_dir)
    if root is None:
        return None, None
    try:
        fp = dataset_fingerprint(root)
    except OSError as exc:
        # A fingerprint read failure must not sink the whole experiment record; degrade to None.
        logger.warning("dataset_fingerprint failed for %s: %s", root, exc)
        fp = None
    ds_id = None
    try:
        ds_id = require_dataset_identity(root).get("id")
    except SchemaVersionRefused:
        raise
    except ValueError:
        ds_id = None
    return ds_id, fp


def persist_run_partition(experiment_id: str, train_ds, val_ds, data_cfg: dict, *,
                           dataset_id: str | None = None,
                           dataset_fingerprint: str | None = None,
                           partition: dict | None = None) -> None:
    """Persist which stems (+ seed + dataset_hash + dataset identity) produced this run's metrics.

    ``partition`` is ``auto_train_val``'s own third return value, for every run whose membership
    resolved into explicit samples (a bound one and a drawn geometry one alike; ``None`` for the
    paths that still build straight from a directory): the run's recorded membership as bare
    ground-truth stems, the group key each was drawn under, the directories those stems live in,
    and their per-stem digests. It is what keeps the record's members bare stems when the loaders
    index by a per-sample identity.
    Its ``label_digests`` block is written as ``split.json``'s own top-level ``label_digests``
    key, beside ``selection_binding`` rather than inside it, so a selection-disjointness check
    can name a calibration label that moved since the draw without the durable config, a
    checkpoint or a trial's resolved config ever carrying a per-stem digest.

    The same seed yields a different split if the label set changes, so a metric is only reproducible
    with the exact train/val membership recorded beside it. The whole-dataset ``dataset_fingerprint``
    (+ id) records the content identity too, so this artifact is literally "fingerprint + split",
    content identity + membership + seed in one immutable record. Best-effort against an ordinary
    write failure, but a write refused because the experiment is already terminal
    (:class:`~tcip_mcp.experiments.ExperimentTerminal`) propagates: a run whose provenance record
    was refused is a failed run, not a silently degraded one.

    The one writer of that member; :func:`~tcip_mcp.experiments.read_run_partition` is the one
    reader every consumer of the membership goes through. Records ``labels_dirs``, every label
    directory the run's own members live under, for every run, bound or not: a later selection
    check narrows itself to the directory a calibration named rather than to a capture date, so a
    run whose members span dates is checked the same way a single-date one is. When
    ``data_cfg["split"]`` carries a ``selection_binding`` (a run bound to a
    ``data.split.selection_dir`` selection, see :func:`auto_train_val`), its counts ride into this
    record too, so a reviewer opening this one file can see that a recorded partition, not a
    drawn one, governed the run.

    A group key in this record is keyed by whatever scopes it, and the record says which: the
    top-level ``group_key_map`` is the caller's own, keyed by
    :func:`~tcip_mcp.pipelines.data.splits.member_identity` because nothing in the record scopes
    it, and a per-directory block's ``group_key_map`` is keyed by the bare stem that block's own
    directory scopes, the way the member lists beside it in that block are. Every producer resolves
    a key through :func:`~tcip_mcp.pipelines.data.splits.recorded_group_key_fn`, so a reader never
    has to work out which one wrote the record it is holding.
    """
    def _stems(ds) -> list[str]:
        # set(): a tiled dataset's ``stems`` repeats one entry per tile, and a recorded member
        # list is a set of units, never a per-example list.
        return sorted(set(getattr(ds, "stems", None) or getattr(ds, "_stems", []) or []))

    from tcip_mcp.experiments import ExperimentTerminal

    try:
        from tcip_store import store

        from tcip_mcp.experiments import experiment_exists, refuse_if_terminal, split_key, status_key
        from tcip_mcp.pipelines.resolution import dataset_hash

        labels_dir = data_cfg.get("labels_dir", "")
        dh = None
        if labels_dir and Path(labels_dir).is_dir():
            dh = dataset_hash(labels_dir)
        split = data_cfg.get("split", {})
        resolved_group_by = split.get("resolved_group_by")
        # A spatial_strip split's members are per-region identities, never the bare stem;
        # auto_train_val already computed and stashed them (the dataset only knows tile positions).
        spatial = split.get("spatial_manifest") if resolved_group_by == "spatial_strip" else None
        if spatial:
            train_members, val_members = spatial["train_identities"], spatial["val_identities"]
        elif partition:
            train_members, val_members = partition["train"], partition["val"]
        else:
            train_members = _stems(train_ds)
            val_members = _stems(val_ds) if val_ds is not None else []
        labels_dirs = (partition["labels_dirs"] if partition
                       else ([str(labels_dir)] if labels_dir else []))
        record = {
            "train": train_members,
            "val": val_members,
            "seed": int(split.get("resolved_seed", split.get("seed", 42))),
            "dataset_hash": dh,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            # The actually resolved grouping ("explicit_map"/"external"/a named strategy/
            # "spatial_strip"/None); _train_disjointness recomputes group keys from this.
            "group_by": resolved_group_by,
            # Every label directory this run's own members live under, so a selection check
            # narrows to the directory a calibration named rather than to one capture date.
            "labels_dirs": labels_dirs,
        }
        if partition:
            # Per label directory: a bare stem names one image only within one of them.
            record["members"] = partition["members"]
        group_key_map = split.get("resolved_group_key_map") or split.get("group_key_map")
        if resolved_group_by == "explicit_map" and group_key_map:
            # The map itself: without it _train_disjointness has a policy name but no way to
            # compute group keys for stems outside this run.
            record["group_key_map"] = group_key_map
        if spatial:
            record["spatial"] = spatial
        binding_block = split.get("selection_binding")
        if binding_block:
            # A run bound to a named selection: its counts ride here too.
            record["selection_binding"] = binding_block
            if binding_block.get("redraw"):
                record["redrawn_within_selection"] = True
        if partition and partition.get("label_digests"):
            record["label_digests"] = partition["label_digests"]
        if experiment_exists(experiment_id):
            key, st_key = split_key(experiment_id), status_key(experiment_id)
            try:
                with store.transaction(key, st_key) as txn:
                    state = (txn.read(st_key, default={}) or {}).get("state")
                    refuse_if_terminal(experiment_id, "persist_run_partition", state)
                    txn.write(key, record)
            except ExperimentTerminal as exc:
                from tcip_mcp.experiments import audit_refusal_reraising
                audit_refusal_reraising(experiment_id, "persist_run_partition", {}, exc)
    except ExperimentTerminal:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("run partition persist failed for %s: %s", experiment_id, exc)


def spatial_split_raster_identity(data_cfg: dict, stem: str) -> dict | None:
    """This mosaic's own :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity`, best
    effort: recorded into ``spatial_manifest`` at spatial-split time (the training source is first
    known to be a raster here), read back at export time (``inference_tools.
    _export_predictions_raster``) to gate a block-calibrated bundle's claim scope to this exact
    mosaic. A provenance write must never sink a launch: an unreadable/unsupported source (a
    bespoke ``dataset_source``, a corrupt file) logs and returns ``None`` rather than raising, the
    same posture ``persist_run_partition`` already takes for its own best-effort writes.
    """
    try:
        from tcip_mcp.pipelines.derivations import probe_channels
        from tcip_mcp.pipelines.image_utils import resolve_image_source
        from tcip_mcp.pipelines.raster_source import content_identity

        source = resolve_image_source(data_cfg.get("images_dir", ""), stem)
        nc = probe_channels(source)
        identity = content_identity(source, nc)
        import dataclasses
        return dataclasses.asdict(identity)
    except Exception as exc:  # noqa: BLE001, best-effort provenance, never sinks the split/launch
        logger.warning("raster content identity for %r could not be recorded: %s", stem, exc)
        return None


def spatial_single_source_split(
    stem: str, data_cfg: dict, tiling: dict, base, split_cfg: dict, transforms,
) -> tuple | None:
    """A train/val split over one detection source's own tile lattice, by disjoint pixel strips.

    Called only from ``auto_train_val``'s single-source branch: there is no second stem to hold
    out whole, but a tiled source has many tiles, and :func:`~tcip_mcp.pipelines.data.splits.
    spatial_strip_split` can hold out disjoint, buffered regions of them. ``base`` is the
    already-built (untiled) ``DetectionDataset`` for this one source, reused for every view (its
    construction, and the class/subject/id-map resolution inside it, cost nothing extra to
    share). A test region is derived and reserved alongside train/val (excluded from both, so it
    is genuinely held out) but no dataset is built for it: nothing downstream consumes a third
    dataset from this function, so only its geometry and kept-tile count are recorded,
    material the block-aware calibration mechanism (``pipelines.block_calibration``) consumes
    without recomputing the split.

    ``split_cfg["reserve_calibration_fraction"]`` (opt-in, default unset/0) reserves a fourth
    region, ``calibration``, alongside train/val/test, at that fraction of the axis: material for
    the same block-calibration mechanism's calibration-side bands. Unset, this function's
    behavior (fractions, split_names, every returned value) is byte-identical to the 3-way split
    it runs without one. When explicitly set, all three of :func:`spatial_strip_split`'s distinct
    silent-``None``-return reasons (no extent from the label file; the strip layout itself
    infeasible; an empty train/val/test/calibration side surviving tile filtering) instead raise
    ``ValueError`` naming which one fired: an opt-in reserved region silently degrading to no
    validation at all would be exactly the kind of measurement-integrity gap this mechanism exists
    to close, unlike the unrequested 3-way case, where that same silent degradation is correct.

    Returns ``(train_ds, val_ds)``, or ``None`` when ``reserve_calibration_fraction`` was not
    requested and the extent is unknown or no strip layout can populate both train and val, in
    which case the caller falls back to training without validation. A present, unreadable label
    document raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument` unconditionally,
    the same whether or not ``reserve_calibration_fraction`` was requested: a corrupt label is
    never the same fact as one recording no width/height, and silently falling back to no
    validation over a document nobody can read would be exactly the gap this mechanism exists to
    close.
    """
    from tcip_mcp.pipelines.data.datasets import TiledDetectionDataset, tile_kwargs_from_tiling
    from tcip_mcp.pipelines.data.splits import image_extent_from_labels, spatial_strip_split

    reserve_cal = float(split_cfg.get("reserve_calibration_fraction") or 0.0)

    extent = image_extent_from_labels(data_cfg.get("labels_dir", ""), stem)
    if extent is None:
        msg = f"its label file carries no width/height for {stem!r}"
        if reserve_cal:
            raise ValueError(
                f"reserve_calibration_fraction={reserve_cal} requires a resolvable extent: {msg}; "
                "a calibration region cannot be reserved without one.")
        logger.warning("Spatial train/val split for %r skipped: %s; training without "
                       "validation.", stem, msg)
        return None
    width, height = extent

    raw_kwargs = tile_kwargs_from_tiling(tiling)
    # keep_regions is this function's own to set from the derived split, never inherited from
    # the caller's tiling dict (which has no meaningful keep_regions in a single-source launch).
    tile_kwargs = {k: v for k, v in raw_kwargs.items() if k != "keep_regions"}
    # Matches TiledDetectionDataset.__init__'s own defaults: the geometry below must agree with
    # what the datasets built further down actually resolve to.
    tile_size = tile_kwargs.get("tile_size", 224)
    overlap = tile_kwargs.get("overlap", 0.2)
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    if reserve_cal:
        train_ratio = 1.0 - val_ratio - test_ratio - reserve_cal
        split_names: tuple[str, ...] = ("train", "val", "test", "calibration")
        fractions: tuple[float, ...] = (train_ratio, val_ratio, test_ratio, reserve_cal)
    else:
        train_ratio = 1.0 - val_ratio - test_ratio
        split_names = ("train", "val", "test")
        fractions = (train_ratio, val_ratio, test_ratio)

    try:
        spatial = spatial_strip_split(
            width, height, tile_size, overlap, fractions=fractions, split_names=split_names,
            buffer=tiling.get("buffer"),
        )
    except ValueError as exc:
        if reserve_cal:
            raise ValueError(
                f"reserve_calibration_fraction={reserve_cal}: 4-way spatial split infeasible for "
                f"{stem!r} at this mosaic size/tile size ({exc}); reduce the fraction or drop "
                "reserve_calibration_fraction."
            ) from exc
        logger.warning(
            "Spatial train/val split for %r could not be derived (%s); training without "
            "validation.", stem, exc,
        )
        return None

    train_ds = TiledDetectionDataset(
        base, transforms=transforms, keep_regions=spatial.regions["train"], **tile_kwargs)
    val_ds = TiledDetectionDataset(
        base, transforms=None, keep_regions=spatial.regions["val"], **tile_kwargs)
    empty_reserved_side = False
    if reserve_cal:
        # A tile lattice occupying the region (spatial_strip_split's own check) is not proof it
        # carries GT: an all-background region still passes that but skip_empty filters it to 0.
        test_ds = TiledDetectionDataset(
            base, transforms=None, keep_regions=spatial.regions["test"], **tile_kwargs)
        cal_ds = TiledDetectionDataset(
            base, transforms=None, keep_regions=spatial.regions["calibration"], **tile_kwargs)
        empty_reserved_side = test_ds.num_samples == 0 or cal_ds.num_samples == 0
    if train_ds.num_samples == 0 or val_ds.num_samples == 0 or empty_reserved_side:
        if reserve_cal:
            raise ValueError(
                f"reserve_calibration_fraction={reserve_cal}: the derived 4-way strip layout for "
                f"{stem!r} left a side with zero kept (or zero GT-bearing) tiles after filtering "
                f"(kept_tiles={spatial.kept_tiles}); reduce the fraction or drop "
                "reserve_calibration_fraction."
            )
        logger.warning(
            "Spatial train/val split for %r yielded an empty side after tile filtering; "
            "training without validation.", stem,
        )
        return None

    def _identities(ds) -> list[str]:
        raw = {spatial.identity_for(s, tx, ty) for s, tx, ty in ds.tile_entries}
        return sorted(name for name in raw if name is not None)

    split_cfg["resolved_group_by"] = "spatial_strip"
    split_cfg["spatial_manifest"] = {
        "stem": stem,
        "train_identities": _identities(train_ds), "val_identities": _identities(val_ds),
        "train_region": spatial.regions.get("train", []),
        "val_region": spatial.regions.get("val", []),
        "test_region": spatial.regions.get("test", []),
        "calibration_region": spatial.regions.get("calibration", []),
        "kept_test_tiles": spatial.kept_tiles.get("test", 0),
        "kept_calibration_tiles": spatial.kept_tiles.get("calibration", 0),
        "width": spatial.width, "height": spatial.height, "tile_size": spatial.tile_size,
        "overlap": spatial.overlap, "axis": spatial.axis, "buffer": spatial.buffer,
        "requested_fractions": dict(zip(spatial.split_names, spatial.requested_fractions)),
        "realized_fractions": spatial.realized_fractions,
        "realized_discard_fraction": spatial.realized_discard_fraction,
        "kept_train_tiles": spatial.kept_tiles.get("train", 0),
        "kept_val_tiles": spatial.kept_tiles.get("val", 0),
        "tiles_dropped_past_extent": spatial.tiles_dropped_past_extent,
        "tiles_dropped_outside_regions": spatial.tiles_dropped_outside_regions,
        "raster_content_identity": spatial_split_raster_identity(data_cfg, stem),
    }
    logger.info(
        "Spatial train/val split for %r: %d train / %d val tiles (axis=%s, "
        "realized_fractions=%s, realized_discard_fraction=%.3f).",
        stem, train_ds.num_samples, val_ds.num_samples, spatial.axis,
        spatial.realized_fractions, spatial.realized_discard_fraction,
    )
    return train_ds, val_ds


def checked_label_format(task: str, data_cfg: dict, src: dict) -> str | None:
    """The per-image label format this run's ``data.labels_dir`` holds (``"json"``, ``"coco"``
    never returned, see below), or ``None`` for a task/config the check does not apply to.

    Refuses a dataset-level assembled COCO document sitting in ``data.labels_dir`` rather than
    per-image label files: a caller-fixable config error, since the per-image files in that
    directory would be shadowed by the assembled export, silently training on the wrong source
    being worse than refusing. Called once per run, by each caller of
    :func:`build_full_admitted_dataset` ahead of its own handler and never from inside the
    helper, so neither the auto path's degrade handler nor a manifest bind can catch this
    refusal and fold it into "training without validation".
    """
    if task not in ("detection", "instance_seg") or data_cfg.get("label_format") or data_cfg.get("coco_json"):
        return None
    labels_dir, images_dir = src.get("labels_dir", ""), src.get("images_dir", "")
    if not (labels_dir and images_dir):
        return None
    from tcip_mcp.pipelines.data.label_queries import dir_label_format, first_labels_json

    fmt = dir_label_format(labels_dir)
    if fmt == "coco":
        offending = first_labels_json(labels_dir)
        raise ValueError(
            f"data.labels_dir={labels_dir!r} holds a dataset-level COCO file "
            f"({offending}): if the per-image label files in this directory are the ones "
            "that should train, move it out of data.labels_dir; if this COCO export is "
            "the intended label source, set data.coco_json (or data.label_format='coco') "
            "to train on it directly."
        )
    return fmt


def build_full_admitted_dataset(
    task: str, data_cfg: dict, src: dict, transforms, detected_label_format: str | None,
):
    """The full, admitted-set dataset for one run's data config, plus the ``build_dataset`` kwargs
    that produced it: one implementation, called both by the auto-split path (inside its own
    degrading handler) and by a selection bind (unwrapped, so a real build failure raises
    rather than degrading to no validation over a recorded partition), so the two can never
    disagree about what this run admits.

    ``detected_label_format`` is the caller's own :func:`checked_label_format` result: a
    dataset-level COCO document sitting in ``data.labels_dir`` is refused there, ahead of the
    caller's own handler, never re-read (and so never re-refused) here.

    Returns ``(full_ds, stems, build_src)``: ``stems`` is the task path's own admitted set
    (``full_ds.stems``/``full_ds._stems``), and ``build_src`` is ``src`` plus any assembled
    ``coco_data``/``label_format``/``num_classes`` the COCO branch added, for a caller that narrows
    it with ``stems=`` afterward.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset

    build_src = dict(src)
    labels_dir, images_dir = src.get("labels_dir", ""), src.get("images_dir", "")

    if detected_label_format == "json":
        from tcip_mcp.pipelines.data.label_queries import assemble_coco, resolve_registry_id_map
        subject, attribute = src.get("subject"), src.get("attribute")
        _reg, id_map = resolve_registry_id_map(labels_dir, subject, attribute)
        assert subject is not None, "resolve_registry_id_map already refused an empty subject"
        build_src["coco_data"] = assemble_coco(
            labels_dir, images_dir, subject=subject, attribute=attribute, id_map=id_map,
            date=src.get("date"))
        build_src["label_format"] = "coco"
        build_src["num_classes"] = len(id_map)

    full_ds = build_dataset(task, **build_src, transforms=transforms)
    stems = list(getattr(full_ds, "stems", None) or getattr(full_ds, "_stems", []))
    return full_ds, stems, build_src


def _refuse_scope_disagreement(data_cfg: dict, selection, selection_dir: str) -> None:
    """Refuse a config that states a class scope other than the one the selection was drawn under.

    The selection's ``subject``/``attribute`` govern a bound run, and are written onto the config
    so the checkpoint records them. A config that already states a different one is a real
    disagreement about what this run trains: overwriting it silently would train one vocabulary
    while the caller asked for another, so it refuses and names both.
    """
    for field, recorded in (("subject", selection.subject),
                            ("attribute", selection.attribute or None)):
        stated = data_cfg.get(field) or None
        if stated is not None and stated != (recorded or None):
            raise ValueError(
                f"data.{field}={stated!r} disagrees with the selection at {selection_dir}, which "
                f"was drawn for {field}={recorded!r}: a bound run admits under the scope its "
                f"selection recorded, so drop data.{field} or bind a selection drawn for it."
            )


def _redrawn_selection(selection, selection_dir: str, seed: int):
    """``selection`` with train and val redrawn fresh over its own train-plus-val samples at
    ``seed``, calibration untouched.

    The draw is :func:`~tcip_mcp.pipelines.data.splits.draw_train_val` over the samples' own
    recorded group keys, at the val share the selection already delivered, stratified by each
    sample's foreground count. A starved side refuses by name rather than retrying or degrading:
    a run that asked for a redraw and got one empty loader trained on a partition nobody chose.
    """
    from tcip_mcp.pipelines.data.selection import with_sides
    from tcip_mcp.pipelines.data.splits import count_label_lines, draw_train_val

    pool = selection.on("train") + selection.on("val")
    val_ratio = len(selection.on("val")) / len(pool)
    by_identity = {s.identity: s for s in pool}
    group_of = {s.identity: s.group for s in pool}
    counts = {
        s.identity: count_label_lines(
            Path(s.ground_truth).parent, Path(s.ground_truth).stem,
            subject=selection.subject, attribute=selection.attribute)
        for s in pool
    }
    train_ids, val_ids = draw_train_val(
        sorted(by_identity), annotation_counts=counts, group_key_fn=lambda i: group_of[i],
        val_ratio=val_ratio, seed=seed,
    )
    if not train_ids or not val_ids:
        from tcip_mcp.pipelines.data.splits import redraw_starved_issue

        raise ValueError(redraw_starved_issue(
            list(group_of.values()),
            [group_of[i] for i in by_identity if counts.get(i, 0) > 0],
            selection_dir=selection_dir, seed=seed,
        ) or (
            f"redrawing train and val inside the selection at {selection_dir!r} at seed {seed} "
            f"starved a side (train={len(train_ids)}, val={len(val_ids)})."
        ))
    assignment = {s.identity: s.side for s in selection.on("calibration")}
    assignment.update({i: "train" for i in train_ids})
    assignment.update({i: "val" for i in val_ids})
    return with_sides(selection, assignment)


def _recorded_partition(train_samples, val_samples, bound, labels_dirs, selection=None) -> dict:
    """The run's own recorded partition, for :func:`persist_run_partition` to write onto
    ``split.json``: its train and val members per label directory, the group key each was drawn
    under, and the digest window a later selection check reads to name a calibration label that
    moved since the draw.

    Members are recorded as bare ground-truth stems, whatever key the loaders indexed by: every
    consumer of this record (the disjointness checks, ``freeze_selection``, block calibration)
    names a member by its stem under one directory. ``selection`` is the record a bound run
    bound to, and ``None`` for a run that drew its own split, whose partition rests on no
    selection to digest.

    Membership is recorded per label directory because every calibration door names a sample by
    its bare stem under one directory, and a selection spans as many as its draw admitted: a flat
    list would read two dates' same-named images as one member and a disjointness check over it
    would report the run leaking into itself. The flat ``train``/``val`` lists beside it are the
    union, for a reader that narrows to no directory at all.
    """
    from tcip_mcp.pipelines.resolution import label_digests as compute_label_digests
    from tcip_mcp.pipelines.resolution import selection_digest

    def _stems_under(samples, labels_dir: str) -> list[str]:
        return sorted({Path(s.ground_truth).stem for s in samples
                       if str(Path(s.ground_truth).parent) == labels_dir})

    sha256 = selection_digest(selection) if selection is not None else None
    members: dict[str, dict] = {}
    at_draw: dict[str, str] = {}
    at_run: dict[str, str] = {}
    for labels_dir in labels_dirs:
        here = [s for s in bound if str(Path(s.ground_truth).parent) == labels_dir]
        here_draw = {Path(s.ground_truth).stem: s.ground_truth_digest for s in here
                     if s.ground_truth_digest is not None}
        here_run = compute_label_digests(
            labels_dir, sorted({Path(s.ground_truth).stem for s in here}))
        members[labels_dir] = {
            "train": _stems_under(train_samples, labels_dir),
            "val": _stems_under(val_samples, labels_dir),
            "group_key_map": {Path(s.ground_truth).stem: s.group for s in here},
            "label_digests": {"at_split": here_draw, "at_run": here_run,
                              "selection_sha256": sha256},
        }
        at_draw.update(here_draw)
        at_run.update(here_run)
    return {
        "train": sorted({Path(s.ground_truth).stem for s in train_samples}),
        "val": sorted({Path(s.ground_truth).stem for s in val_samples}),
        "members": members,
        "labels_dirs": labels_dirs,
        "label_digests": {
            "at_split": at_draw, "at_run": at_run, "selection_sha256": sha256,
        },
    }
def _drawn_geometry_split(task: str, data_cfg: dict, src: dict, tiling, transforms):
    """``(train_ds, val_ds, None)`` for a detection/instance_seg run that draws its own split, or
    ``None`` when this run is not one this path can draw.

    The directory a caller names is a producer interface, not a second membership representation:
    admission runs once here (:func:`~tcip_mcp.pipelines.data.label_queries.trainable_stems`, the
    platform's own, including the human confirmation an empty label needs), the draw partitions
    the admitted stems, and the result becomes explicit samples
    (:func:`~tcip_mcp.pipelines.data.label_queries.directory_samples`) before either loader is
    built. A run that drew its split and a run bound to a recorded selection then hold the same
    membership shape and read through the same recorded paths.

    Answers ``None``, leaving the caller's own fallbacks to run, when: the config names no labels
    or images directory; admission or the draw fails (the caller degrades to training without
    validation, as it always has); fewer than two stems are admitted (the caller's single-source
    spatial-strip branch owns that case); or no grouping policy can populate both sides.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import (
        admission_date, directory_samples, resolve_registry_id_map, trainable_stems,
    )
    from tcip_mcp.pipelines.data.splits import (
        count_label_lines, draw_train_val, recorded_group_key_fn,
    )

    labels_dir, images_dir = src.get("labels_dir", ""), src.get("images_dir", "")
    if not labels_dir or not images_dir:
        return None
    subject, attribute = src.get("subject"), src.get("attribute")
    date = admission_date(labels_dir)
    try:
        _registry, id_map = resolve_registry_id_map(labels_dir, subject, attribute)
        stems, _counts = trainable_stems(
            labels_dir, images_dir, subject=subject, date=date,
            attribute=attribute, id_map=id_map,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Admission for the drawn split failed (%s); falling back.", exc)
        return None
    if len(stems) < 2:
        return None

    split_cfg = data_cfg.setdefault("split", {})
    group_by = split_cfg.get("group_by", "tile_prefix")
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any handler: a malformed grouping policy is a caller-config error.
    # Through recorded_group_key_fn, so this draw spells a group key the way draw_splits does.
    group_key_fn = recorded_group_key_fn(
        group_by, date=date, stems=stems, group_key_map=group_key_map)
    split_cfg["resolved_group_by"] = "explicit_map" if group_key_map else group_by

    try:
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        seed = int(split_cfg.get("seed", 42))
        annotation_counts = None
        if split_cfg.get("stratify_foreground", True):
            annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}
        train_stems, val_stems = draw_train_val(
            stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
            val_ratio=val_ratio, seed=seed,
        )
        if (not val_stems or not train_stems) and group_by != "stem" and not group_key_map:
            # Too few groups under the requested policy starved val; retry at stem grouping.
            retry_key_fn = recorded_group_key_fn("stem", date=date, stems=stems)
            retry_train, retry_val = draw_train_val(
                stems, annotation_counts=annotation_counts, group_key_fn=retry_key_fn,
                val_ratio=val_ratio, seed=seed,
            )
            if retry_train and retry_val:
                train_stems, val_stems = retry_train, retry_val
                split_cfg["resolved_group_by"] = "stem"
                group_key_fn = retry_key_fn
        if not val_stems or not train_stems:
            return None

        assignment = {s: "train" for s in train_stems}
        assignment.update({s: "val" for s in val_stems})
        samples = directory_samples(
            assignment, images_dir=images_dir, labels_dir=labels_dir, subject=subject or "",
            group_of=group_key_fn,
        )
        by_side = {side: [s for s in samples if s.side == side] for side in ("train", "val")}
        build_kwargs: dict[str, Any] = {
            "subject": subject, "attribute": attribute, "id_map": id_map, "tiling": tiling,
        }
        train_ds = build_dataset(
            task, samples=by_side["train"], transforms=transforms, **build_kwargs)
        val_ds = build_dataset(task, samples=by_side["val"], transforms=None, **build_kwargs)
        partition = _recorded_partition(
            by_side["train"], by_side["val"], samples, [str(labels_dir)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto train/val split failed (%s); falling back.", exc)
        return None
    logger.info("Auto train/val split for %s: %d train / %d val samples.",
                task, len(by_side["train"]), len(by_side["val"]))
    return train_ds, val_ds, partition


def auto_train_val(task: str, data_cfg: dict, transforms):
    """Build ``(train_ds, val_ds, partition)`` for a run, deriving a leakage-free val split.

    ``partition`` is ``None`` on every path but the selection-bound one (1.5 below), where it
    carries the membership, group keys, label directories and per-stem digests
    :func:`persist_run_partition` writes onto ``split.json`` beside, never inside,
    ``selection_binding``: passed as its own value rather than through ``data_cfg["split"]``
    since that block is copied whole into the durable config and every checkpoint, and a
    per-sample map must not multiply through every copy.

    Resolution order:
      1. ``data.val_images_dir`` set -> build val from it explicitly (a CSV-driven task -
         classification/ordinal/regression - also requires ``data.val_csv_path``; there is no
         graceful fallback to the train CSV the way the geometry tasks fall back to the train
         labels/masks dir, see the CSV branch below for why).
      1.5. ``data.split.selection_dir`` set (detection/instance_seg only) -> train on the
         selection's own ``train`` and ``val`` samples instead of drawing a split, each sample
         reading the source and label the draw recorded for it. A recorded partition is an
         explicit split ``auto_val`` does not govern, checked ahead of its gate below; every
         conflict, task and empty-side refusal here raises to the caller, and so does a build
         failure while binding, never degrading to training on the selection's held-out side with
         no validation. Nothing is re-admitted here: the selection already decided which samples
         train, so the loaders read it rather than rediscovering membership from a directory, and
         the selection's own subject, attribute and id map govern rather than being restated on
         the config and compared. The calibration side never builds a loader; its count rides into
         ``selection_binding``. ``data.split.redraw_within_selection: true`` (beside
         ``selection_dir`` and ``seed``) redraws train and val fresh inside the selection's own
         train-plus-val samples instead of binding them as recorded, calibration still untouched;
         a starved side refuses rather than retrying or degrading.
      2. ``data.auto_val`` (default True) and a stem-capable task
         (detection / instance_seg / semantic_seg / classification) -> derive a
         group-aware train/val split (no held-out test) so the trainer receives
         a real validation loader. Train keeps augmentation; val gets none.
      3. ordinal / regression, ``auto_val`` disabled, a tiny/single-group set, or
         most failures -> ``(full_train_ds, None)``. ``resolve_group_key_fn`` (an
         unrecognized ``split.group_by`` or a ``split.group_key_map`` missing stem
         coverage) is called outside any handler here and its ``ValueError`` propagates
         to the caller, silently training without validation on a policy error the
         caller could have fixed is worse than surfacing it. A present, unreadable
         validation label (``UnreadableLabelDocument``) from the explicit ``val_images_dir``
         build propagates too, rather than degrading to a run with no validation over a
         document nobody can read. Every other failure in this function (dataset build
         errors, a malformed ``val_ratio``/``seed``, a ``group_balanced_split`` failure)
         still degrades to ``(full_train_ds, None)``.

    Reads ``auto_val`` / ``val_*`` / ``split.*`` from ``data_cfg`` (== config["data"]).
    """
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.label_queries import admission_date, targets_registry_derived
    from tcip_mcp.pipelines.data.splits import (
        draw_train_val, count_label_lines, recorded_group_key_fn,
    )
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY
    from tcip_mcp.tools.training_tools import (
        _dataset_source_kwargs, _redraw_flag_issue, _split_selection_drawn_conflicts,
    )

    src = _dataset_source_kwargs(task, data_cfg)
    tiling = data_cfg.get("tiling")  # detection tiling (None for other tasks/configs)

    split_cfg_raw = data_cfg.get("split")
    split_cfg_raw = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}
    selection_dir = split_cfg_raw.get("selection_dir")

    # A binding block an earlier bound launch left behind is cleared here; only the selection
    # branch below writes it back, and only when this run itself binds.
    for _stale_key in (
        "selection_binding", "resolved_group_by", "resolved_group_key_map", "resolved_seed",
    ):
        split_cfg_raw.pop(_stale_key, None)

    # 1. Explicit validation source.
    val_images = data_cfg.get("val_images_dir")
    if selection_dir and val_images:
        raise ValueError(
            "data.split.selection_dir conflicts with data.val_images_dir: two membership "
            "sources for one run's validation split."
        )
    if val_images:
        # No computed grouping here; record "external" explicitly so persist_run_partition
        # writes a distinct marker rather than leaving the field unset (ambiguous with never-set).
        data_cfg.setdefault("split", {})["resolved_group_by"] = "external"
        try:
            train_ds = build_dataset(task, **src, transforms=transforms, tiling=tiling)
            val_src = dict(src)
            val_src["images_dir"] = val_images
            if task in ("detection", "instance_seg"):
                val_src["labels_dir"] = data_cfg.get("val_labels_dir", data_cfg.get("labels_dir", ""))
            elif task == "semantic_seg":
                val_src["masks_dir"] = data_cfg.get("val_masks_dir", data_cfg.get("masks_dir", ""))
            elif task in ("classification", "ordinal", "regression"):
                val_csv = data_cfg.get("val_csv_path")
                if not val_csv:
                    # A CSV dataset fails per-item deep in training if a row's image isn't in
                    # val_images_dir; unlike the geometry tasks, there's no graceful fallback.
                    raise ValueError(
                        "val_images_dir set for a CSV-driven task also requires "
                        "data.val_csv_path; the train CSV's rows won't generally match a "
                        "different val_images_dir."
                    )
                val_src["csv_path"] = val_csv
            return train_ds, build_dataset(task, **val_src, transforms=None, tiling=tiling), None
        except UnreadableLabelDocument:
            raise
        except Exception as exc:
            logger.warning("Explicit val build failed (%s); training without validation.", exc)
            return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

    # 1.5. A named selection is an explicit partition auto_val does not govern; every refusal
    # here, and any build failure while binding to it, raises rather than degrading.
    if selection_dir:
        conflicts = _split_selection_drawn_conflicts(data_cfg, split_cfg_raw)
        if conflicts:
            raise ValueError(
                f"data.split.selection_dir conflicts with {sorted(conflicts)}: a recorded "
                "partition and a drawn split's own parameters/source cannot both govern one run."
            )
        flag_issue = _redraw_flag_issue(split_cfg_raw)
        if flag_issue:
            raise ValueError(flag_issue)
        if task not in ("detection", "instance_seg"):
            raise ValueError(
                f"data.split.selection_dir names a selection, and only detection and "
                f"instance_seg read the per-image label documents a selection's samples name; "
                f"task={task!r} cannot bind to one."
            )

        from tcip_mcp.pipelines.data.label_queries import refuse_inadmissible_samples
        from tcip_mcp.pipelines.data.selection import read_selection

        selection = read_selection(selection_dir)
        if not selection.subject:
            raise ValueError(
                f"the selection at {selection_dir} records no subject: a run reads the subject it "
                "admits under off the selection, and one drawn without a subject names no class "
                "space to train over. Draw it again with draw_splits, which requires a subject."
            )
        _refuse_scope_disagreement(data_cfg, selection, selection_dir)

        # A redraw repartitions the selection's own train-plus-val samples; calibration untouched.
        if split_cfg_raw.get("redraw_within_selection"):
            redraw_seed = int(split_cfg_raw["seed"])
            selection = _redrawn_selection(selection, selection_dir, redraw_seed)
            split_cfg_raw["_redraw_seed"] = redraw_seed

        train_samples, val_samples = selection.on("train"), selection.on("val")
        calibration_samples = selection.on("calibration")
        if not train_samples or not val_samples:
            raise ValueError(
                f"the selection at {selection_dir} leaves an empty side (train="
                f"{len(train_samples)}, val={len(val_samples)}); a run needs both."
            )

        # Checked before either loader is built: a selected label that has since emptied with
        # nobody confirming that image negative would otherwise train as background.
        refuse_inadmissible_samples(
            train_samples + val_samples,
            attribute=selection.attribute or None, id_map=selection.id_map or None)

        bound = train_samples + val_samples + calibration_samples
        labels_dirs = sorted({str(Path(s.ground_truth).parent) for s in bound})
        # The selection's own scope and exact map become this run's, so the checkpoint records
        # the vocabulary it trained in rather than one rediscovered from a live registry.
        data_cfg["subject"] = selection.subject
        data_cfg["attribute"] = selection.attribute or None
        data_cfg["id_map"] = dict(selection.id_map)
        split_cfg = data_cfg.setdefault("split", {})
        # The selection's own named policy, not "explicit_map": the per-stem map the partition
        # records covers this run's members, and a stem outside it is what a policy name answers.
        split_cfg["resolved_group_by"] = selection.group_by
        split_cfg["resolved_seed"] = split_cfg_raw.pop("_redraw_seed", selection.seed)
        split_cfg["selection_binding"] = {
            "selection_dir": selection_dir, "subject": selection.subject,
            "attribute": selection.attribute, "labels_dirs": labels_dirs,
            "assigned": len(selection.samples),
            "train_bound": len(train_samples), "val_bound": len(val_samples),
            "calibration_bound": len(calibration_samples),
            "dataset_fingerprint_at_draw": selection.dataset_fingerprint,
        }
        if split_cfg_raw.get("redraw_within_selection"):
            split_cfg["selection_binding"]["redraw"] = {
                "seed": split_cfg["resolved_seed"],
                "val_ratio": len(val_samples) / (len(train_samples) + len(val_samples)),
            }

        # Kept out of split_cfg/selection_binding: that block is copied whole into the durable
        # config and every checkpoint. Handed to persist_run_partition as its own parameter.
        partition = _recorded_partition(
            train_samples, val_samples, bound, labels_dirs, selection)

        build_kwargs: dict[str, Any] = {
            "subject": selection.subject, "attribute": selection.attribute,
            "id_map": selection.id_map, "tiling": tiling,
            "dataset_source": data_cfg.get(DATASET_SOURCE_KEY) or None,
        }
        train_ds = build_dataset(
            task, samples=train_samples, transforms=transforms, **build_kwargs)
        val_ds = build_dataset(task, samples=val_samples, transforms=None, **build_kwargs)
        return train_ds, val_ds, partition

    if not data_cfg.get("auto_val", True) or task not in STEM_TASKS:
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

    # 2. Auto group-aware train/val split. A dataset-level COCO here is a caller-fixable config
    # error, raised outside the handler below rather than degraded.
    detected_label_format = checked_label_format(task, data_cfg, src)
    # The drawn path reads the per-image sidecars beside a registry, so a config naming its own
    # COCO document or a bespoke source is not one it may draw (targets_registry_derived).
    if task in ("detection", "instance_seg") and targets_registry_derived(data_cfg):
        drawn = _drawn_geometry_split(task, data_cfg, src, tiling, transforms)
        if drawn is not None:
            return drawn
    try:
        full_ds, stems, build_src = build_full_admitted_dataset(
            task, data_cfg, src, transforms, detected_label_format)
    except Exception as exc:
        logger.warning("Auto train/val split failed (%s); training without validation.", exc)
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

    if len(stems) < 2:
        # A single source can't hold out a whole stem, but a tiled detection source can still
        # hold out disjoint pixel blocks of its own tile lattice, see spatial_single_source_split.
        split_cfg = data_cfg.setdefault("split", {})
        if task == "detection" and tiling and tiling.get("enabled", True):
            spatial_ds = spatial_single_source_split(
                stems[0], data_cfg, tiling, full_ds, split_cfg, transforms)
            if spatial_ds is not None:
                return (*spatial_ds, None)
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

    # setdefault (not get): the resolved grouping is written back so a later
    # persist_run_partition call can record what was actually used.
    split_cfg = data_cfg.setdefault("split", {})
    group_by = split_cfg.get("group_by", "tile_prefix")
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any try/except: a malformed grouping policy is a caller-config error
    # and must reach the caller, not degrade silently like the failures handled below.
    date = admission_date(data_cfg.get("labels_dir", ""))
    group_key_fn = recorded_group_key_fn(
        group_by, date=date, stems=stems, group_key_map=group_key_map)
    split_cfg["resolved_group_by"] = "explicit_map" if group_key_map else group_by

    try:
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        seed = int(split_cfg.get("seed", 42))
        stratify = split_cfg.get("stratify_foreground", True)

        annotation_counts = None
        if stratify and task in ("detection", "instance_seg"):
            labels_dir = data_cfg.get("labels_dir", "")
            annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}

        train_stems, val_stems = draw_train_val(
            stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
            val_ratio=val_ratio, seed=seed,
        )
        if (not val_stems or not train_stems) and group_by != "stem" and not group_key_map:
            # Too few *groups* under the requested policy starved val (e.g. two sources whose
            # tile-prefix collapses to one group); retry at stem-level grouping before giving up.
            stem_key_fn = recorded_group_key_fn("stem", date=date, stems=stems)
            retry_train, retry_val = draw_train_val(
                stems, annotation_counts=annotation_counts, group_key_fn=stem_key_fn,
                val_ratio=val_ratio, seed=seed,
            )
            if retry_train and retry_val:
                logger.info(
                    "Auto train/val split for %s: group_by=%r left val empty (too few groups); "
                    "retried at stem-level grouping.", task, group_by,
                )
                train_stems, val_stems = retry_train, retry_val
                split_cfg["resolved_group_by"] = "stem"
        if not val_stems or not train_stems:
            logger.warning(
                "Auto train/val split for %s: no grouping policy could populate both sides; "
                "training without validation.", task,
            )
            return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

        if task == "classification":
            stem_to_label = dict(zip(getattr(full_ds, "_stems", []), getattr(full_ds, "_labels", [])))
            train_ds = build_dataset(
                task, images_dir=src["images_dir"], transforms=transforms,
                stems=train_stems, labels=[stem_to_label[s] for s in train_stems])
            val_ds = build_dataset(
                task, images_dir=src["images_dir"], transforms=None,
                stems=val_stems, labels=[stem_to_label[s] for s in val_stems])
        else:
            train_ds = build_dataset(task, **build_src, transforms=transforms, stems=train_stems, tiling=tiling)
            val_ds = build_dataset(task, **build_src, transforms=None, stems=val_stems, tiling=tiling)
        logger.info("Auto train/val split for %s: %d train / %d val stems.",
                    task, len(train_stems), len(val_stems))
        return train_ds, val_ds, None
    except Exception as exc:
        logger.warning("Auto train/val split failed (%s); training without validation.", exc)
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

"""Constructing and persisting training splits from a data config, beside ``splits.py``.

Resolves ``(train_ds, val_ds)`` for a run (explicit val dir, a bound split manifest, an auto
group-aware draw, or a single-source spatial-strip split) and persists the drawn/bound
membership as the run's ``split.json`` provenance record.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STEM_TASKS = frozenset({"detection", "instance_seg", "semantic_seg", "classification"})
"""Tasks ``auto_train_val``'s drawn (step 2) path covers: a run outside this set never reaches
that branch, so nothing here draws a train/val split for it. Module-level so a caller deciding
whether a task admits the drawn path (``run_hpo``'s ``split_draws`` refusal) checks the same set
``auto_train_val`` itself walks, rather than a second copy that could drift from it."""


def dataset_identity(data_cfg: dict) -> tuple[str | None, str | None]:
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset, the content end of the
    reproduce-a-number chain. The fingerprint is recomputed here (recompute-on-read is authority); the
    id comes from the dataset's ``dataset.json`` if it was registered. ``(None, None)`` for a bespoke /
    imageless run (no dataset_root), matching ``dataset_hash=None`` rather than fabricating identity.

    A version-refused identity (``tcip_store.SchemaVersionRefused``) is a real, wrong identity a
    delivered number could rest on, never the same fact as not-registered, so it propagates rather
    than being caught here: this function's own caller already wraps the call in a best-effort
    ``except Exception`` that logs and continues the run, instead of silently recording
    ``(None, fp)`` as though the dataset were simply unregistered.
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


def persist_split_manifest(experiment_id: str, train_ds, val_ds, data_cfg: dict, *,
                           dataset_id: str | None = None,
                           dataset_fingerprint: str | None = None,
                           label_digests: dict | None = None) -> None:
    """Persist which stems (+ seed + dataset_hash + dataset identity) produced this run's metrics.

    ``label_digests`` (``auto_train_val``'s own third return value for a manifest-bound run,
    ``None`` otherwise) is written as ``split.json``'s own top-level ``label_digests`` key,
    beside ``manifest_binding`` rather than inside it, so a selection-disjointness check can
    name a calibration label that moved since the draw without the durable config, a checkpoint
    or a trial's resolved config ever carrying a per-stem digest.

    The same seed yields a different split if the label set changes, so a metric is only reproducible
    with the exact train/val membership recorded beside it. The whole-dataset ``dataset_fingerprint``
    (+ id) records the content identity too, so this artifact is literally "fingerprint + split",
    content identity + membership + seed in one immutable record. Best-effort against an ordinary
    write failure, but a write refused because the experiment is already terminal
    (:class:`~tcip_mcp.experiments.ExperimentTerminal`) propagates: a run whose provenance record
    was refused is a failed run, not a silently degraded one.

    The one writer of that member; :func:`~tcip_mcp.experiments.read_split_manifest` is the one
    reader every consumer of the membership goes through. Records ``date``, the labels
    directory's own capture date (``manifest_date_key``'s empty string for a flat tree, never
    ``None``), for every run, bound or not, so a later selection check can scope itself to one
    date without re-deriving it from the config, and can tell a flat run's own date apart from a
    caller that derived no date to compare at all.
    When ``data_cfg["split"]`` carries a ``manifest_binding`` (a run bound to a
    ``data.split.manifest_dir`` split manifest, see :func:`auto_train_val`), its counts and the
    two dataset hashes ride into this record too, so a reviewer opening this one file can see
    that a recorded partition, not a drawn one, governed the run.
    """
    def _stems(ds) -> list[str]:
        # set(): a tiled dataset's ``stems`` repeats one entry per tile, and a manifest member
        # list is a set of units, never a per-example list.
        return sorted(set(getattr(ds, "stems", None) or getattr(ds, "_stems", []) or []))

    from tcip_mcp.experiments import ExperimentTerminal

    try:
        from tcip_store import store

        from tcip_mcp.dataset_layout import annotation_date
        from tcip_mcp.experiments import experiment_exists, refuse_if_terminal, split_key, status_key
        from tcip_mcp.pipelines.data.splits import manifest_date_key
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
        train_members = spatial["train_identities"] if spatial else _stems(train_ds)
        val_members = (spatial["val_identities"] if spatial
                       else (_stems(val_ds) if val_ds is not None else []))
        manifest = {
            "train": train_members,
            "val": val_members,
            "seed": int(split.get("resolved_seed", split.get("seed", 42))),
            "dataset_hash": dh,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            # The actually resolved grouping ("explicit_map"/"external"/a named strategy/
            # "spatial_strip"/None); _train_disjointness recomputes group keys from this.
            "group_by": resolved_group_by,
            # manifest_date_key's empty string for a flat tree, never None: a selection check
            # must tell a flat run's own date apart from a caller that derived none to compare.
            "date": manifest_date_key(annotation_date(labels_dir)),
        }
        resolved_group_key_map = split.get("resolved_group_key_map") or split.get("group_key_map")
        if resolved_group_by == "explicit_map" and resolved_group_key_map:
            # The map itself: without it _train_disjointness has a policy name but no way to
            # compute group keys for stems outside this run.
            manifest["group_key_map"] = resolved_group_key_map
        if spatial:
            manifest["spatial"] = spatial
        if split.get("manifest_binding"):
            # A run bound to a named split manifest: its counts and hashes ride here too.
            manifest["manifest_binding"] = split["manifest_binding"]
        if label_digests:
            manifest["label_digests"] = label_digests
        if experiment_exists(experiment_id):
            key, st_key = split_key(experiment_id), status_key(experiment_id)
            try:
                with store.transaction(key, st_key) as txn:
                    state = (txn.read(st_key, default={}) or {}).get("state")
                    refuse_if_terminal(experiment_id, "persist_split_manifest", state)
                    txn.write(key, manifest)
            except ExperimentTerminal as exc:
                from tcip_mcp.experiments import audit_refusal_reraising
                audit_refusal_reraising(experiment_id, "persist_split_manifest", {}, exc)
    except ExperimentTerminal:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("split manifest persist failed for %s: %s", experiment_id, exc)


def spatial_split_raster_identity(data_cfg: dict, stem: str) -> dict | None:
    """This mosaic's own :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity`, best
    effort: recorded into ``spatial_manifest`` at spatial-split time (the training source is first
    known to be a raster here), read back at export time (``inference_tools.
    _export_predictions_raster``) to gate a block-calibrated bundle's claim scope to this exact
    mosaic. A provenance write must never sink a launch: an unreadable/unsupported source (a
    bespoke ``dataset_source``, a corrupt file) logs and returns ``None`` rather than raising, the
    same posture ``persist_split_manifest`` already takes for its own best-effort writes.
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
    dataset from this function today, so only its geometry and kept-tile count are recorded,
    material the block-aware calibration mechanism (``pipelines.block_calibration``) consumes
    without recomputing the split.

    ``split_cfg["reserve_calibration_fraction"]`` (opt-in, default unset/0) reserves a fourth
    region, ``calibration``, alongside train/val/test, at that fraction of the axis: material for
    the same block-calibration mechanism's calibration-side bands. Unset, this function's
    behavior (fractions, split_names, every returned value) is byte-identical to the 3-way split
    it has always run. When explicitly set, all three of :func:`spatial_strip_split`'s distinct
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
    seed = int(split_cfg.get("seed", 42))
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
            seed=seed, buffer=tiling.get("buffer"),
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
        "seed": spatial.seed, "requested_fractions": dict(zip(spatial.split_names,
                                                               spatial.requested_fractions)),
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
    degrading handler) and by a split-manifest bind (unwrapped, so a real build failure raises
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


def auto_train_val(task: str, data_cfg: dict, transforms):
    """Build ``(train_ds, val_ds, label_digests)`` for a run, deriving a leakage-free val split.

    ``label_digests`` is ``None`` on every path but the manifest-bound one (1.5 below), where it
    carries the per-stem digests :func:`persist_split_manifest` writes onto ``split.json``
    beside, never inside, ``manifest_binding``: passed as its own value rather than through
    ``data_cfg["split"]`` since that block is copied whole into the durable config and every
    checkpoint, and per-stem digests must not multiply through every copy.

    Resolution order:
      1. ``data.val_images_dir`` set -> build val from it explicitly (a CSV-driven task -
         classification/ordinal/regression - also requires ``data.val_csv_path``; there is no
         graceful fallback to the train CSV the way the geometry tasks fall back to the train
         labels/masks dir, see the CSV branch below for why).
      1.5. ``data.split.manifest_dir`` set (detection/instance_seg only) -> bind this run's own
         admission to the named ``split_manifest`` record (:func:`~tcip_mcp.pipelines.data.
         splits.bind_manifest_stems`) instead of drawing a split. A recorded partition is an
         explicit split ``auto_val`` does not govern, checked ahead of its gate below; every
         conflict, task, date, images-root and binding refusal here raises to the caller, and so
         does a build failure while binding, never degrading to training on the manifest's
         held-out side with no validation. The manifest's calibration side never builds a
         loader; its own bound count and unadmitted count ride into ``manifest_binding`` beside
         ``labels_hash_now`` (over all three bound sides) and ``labels_hash_at_split``, a pair
         nothing currently reads back, kept for a reviewer to compare by eye.
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
    from tcip_mcp.pipelines.data.splits import (
        group_balanced_split, count_label_lines, resolve_group_key_fn,
    )
    from tcip_mcp.tools.training_tools import (
        _dataset_source_kwargs, _split_manifest_drawn_conflicts,
    )

    src = _dataset_source_kwargs(task, data_cfg)
    tiling = data_cfg.get("tiling")  # detection tiling (None for other tasks/configs)

    split_cfg_raw = data_cfg.get("split")
    split_cfg_raw = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}
    manifest_dir = split_cfg_raw.get("manifest_dir")

    # A binding block an earlier bound launch left behind is cleared here; only the manifest
    # branch below writes it back, and only when this run itself binds.
    for _stale_key in (
        "manifest_binding", "resolved_group_by", "resolved_group_key_map", "resolved_seed",
    ):
        split_cfg_raw.pop(_stale_key, None)

    # 1. Explicit validation source.
    val_images = data_cfg.get("val_images_dir")
    if manifest_dir and val_images:
        raise ValueError(
            "data.split.manifest_dir conflicts with data.val_images_dir: two membership "
            "sources for one run's validation split."
        )
    if val_images:
        # No computed grouping here; record "external" explicitly so persist_split_manifest
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

    # 1.5. A named split manifest is an explicit partition auto_val does not govern; every
    # refusal here, and any build failure while binding to it, raises rather than degrading.
    if manifest_dir:
        conflicts = _split_manifest_drawn_conflicts(data_cfg, split_cfg_raw)
        if conflicts:
            raise ValueError(
                f"data.split.manifest_dir conflicts with {sorted(conflicts)}: a recorded "
                "partition and a drawn split's own parameters/source cannot both govern one run."
            )
        if task not in ("detection", "instance_seg"):
            raise ValueError(
                f"data.split.manifest_dir names a split manifest, and only detection and "
                f"instance_seg admit through the trainable_stems draw a manifest is drawn "
                f"through; task={task!r} cannot bind to one."
            )

        from tcip_mcp.dataset_layout import annotation_date
        from tcip_mcp.pipelines.data.splits import (
            bind_manifest_stems, manifest_date_key, member_identity_parts,
            require_manifest_scope,
        )
        from tcip_mcp.pipelines.resolution import dataset_hash, manifest_digest
        from tcip_mcp.pipelines.resolution import label_digests as compute_label_digests
        from tcip_mcp.tools.data_tools import read_split_manifest_dir

        manifest = read_split_manifest_dir(manifest_dir)
        labels_dir, images_dir = data_cfg.get("labels_dir", ""), data_cfg.get("images_dir", "")
        run_date = annotation_date(labels_dir)
        declared_date = data_cfg.get("date")
        if declared_date is not None and declared_date != run_date:
            raise ValueError(
                f"data.date={declared_date!r} disagrees with the date data.labels_dir={labels_dir!r} "
                f"is under ({run_date!r}); a split manifest binds under one date, so the negative "
                "confirmations and the manifest must be read under the same one."
            )
        date = run_date
        # The admission draw below reads confirmed negatives under src["date"]; it must agree.
        src["date"] = date
        # Reused below for labels_hash_at_split; bind_manifest_stems refuses an absent block.
        date_block = (manifest.get("members") or {}).get(manifest_date_key(date))

        subject, attribute = src.get("subject"), src.get("attribute")
        if not subject:
            raise ValueError(
                "data.split.manifest_dir requires data.subject: a manifest binds by subject, "
                "and this run's own admission has none to compare against it."
            )
        # Config-only, ahead of a build that would fail for an unrelated reason on a bad root.
        require_manifest_scope(
            manifest, manifest_dir=manifest_dir, subject=subject, attribute=attribute, date=date,
            images_dir=images_dir, label="data.images_dir",
        )
        detected_label_format = checked_label_format(task, data_cfg, src)
        full_ds, admitted, build_src = build_full_admitted_dataset(
            task, data_cfg, src, transforms, detected_label_format)
        binding = bind_manifest_stems(
            manifest, date, subject, attribute, admitted,
            admission_counts=getattr(full_ds, "sample_counts", None),
            images_dir=images_dir, manifest_dir=manifest_dir)
        assert date_block is not None, "bind_manifest_stems already refused an absent block"

        split_cfg = data_cfg.setdefault("split", {})
        split_cfg["resolved_group_by"] = manifest.get("group_by")
        manifest_group_key_map = manifest.get("group_key_map")
        if manifest_group_key_map:
            split_cfg["resolved_group_key_map"] = {
                member_identity_parts(identity)[1]: group_key
                for identity, group_key in manifest_group_key_map.items()
                if member_identity_parts(identity)[0] == date
            }
        split_cfg["resolved_seed"] = manifest.get("seed")
        split_cfg["manifest_binding"] = {
            "manifest_dir": manifest_dir, "subject": subject, "attribute": attribute,
            "date": date, "labels_hash_at_split": date_block.get("dataset_hash"),
            "labels_hash_now": dataset_hash(
                labels_dir, stems=binding.train + binding.val + binding.calibration),
            "dataset_fingerprint_at_split": manifest.get("dataset_fingerprint"),
            "assigned": binding.assigned, "train_bound": binding.train_bound,
            "val_bound": binding.val_bound, "calibration_bound": binding.calibration_bound,
            "calibration_unadmitted": binding.calibration_unadmitted,
            "other_dates": binding.other_dates,
        }
        # Kept out of split_cfg/manifest_binding: that block is copied whole into the durable
        # config and every checkpoint. Handed to persist_split_manifest as its own parameter.
        bound_stems = sorted(set(binding.train) | set(binding.val) | set(binding.calibration))
        label_digests_block = {
            "at_split": date_block.get("label_digests"),
            "at_run": compute_label_digests(labels_dir, bound_stems),
            "manifest_sha256": manifest_digest(manifest),
        }
        # Only detection/instance_seg reach here (checked above), so the build is the plain
        # stems=-narrowed geometry path, never the classification CSV/folder branch below.
        train_ds = build_dataset(
            task, **build_src, transforms=transforms, stems=binding.train, tiling=tiling)
        val_ds = build_dataset(
            task, **build_src, transforms=None, stems=binding.val, tiling=tiling)
        return train_ds, val_ds, label_digests_block

    if not data_cfg.get("auto_val", True) or task not in STEM_TASKS:
        return build_dataset(task, **src, transforms=transforms, tiling=tiling), None, None

    # 2. Auto group-aware train/val split. A dataset-level COCO here is a caller-fixable config
    # error, raised outside the handler below rather than degraded.
    detected_label_format = checked_label_format(task, data_cfg, src)
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
    # persist_split_manifest call can record what was actually used.
    split_cfg = data_cfg.setdefault("split", {})
    group_by = split_cfg.get("group_by", "tile_prefix")
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any try/except: a malformed grouping policy is a caller-config error
    # and must reach the caller, not degrade silently like the failures handled below.
    group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    split_cfg["resolved_group_by"] = "explicit_map" if group_key_map else group_by

    try:
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        seed = int(split_cfg.get("seed", 42))
        stratify = split_cfg.get("stratify_foreground", True)

        annotation_counts = None
        if stratify and task in ("detection", "instance_seg"):
            labels_dir = data_cfg.get("labels_dir", "")
            annotation_counts = {s: count_label_lines(labels_dir, s) for s in stems}

        parts = group_balanced_split(
            stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
            splits=(1.0 - val_ratio, val_ratio, 0.0), seed=seed,
        )
        train_stems, val_stems = parts["train"], parts["val"]
        if (not val_stems or not train_stems) and group_by != "stem" and not group_key_map:
            # Too few *groups* under the requested policy starved val (e.g. two sources whose
            # tile-prefix collapses to one group); retry at stem-level grouping before giving up.
            stem_key_fn = resolve_group_key_fn("stem", stems)
            retry_parts = group_balanced_split(
                stems, annotation_counts=annotation_counts, group_key_fn=stem_key_fn,
                splits=(1.0 - val_ratio, val_ratio, 0.0), seed=seed,
            )
            if retry_parts["train"] and retry_parts["val"]:
                logger.info(
                    "Auto train/val split for %s: group_by=%r left val empty (too few groups); "
                    "retried at stem-level grouping.", task, group_by,
                )
                parts = retry_parts
                train_stems, val_stems = parts["train"], parts["val"]
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

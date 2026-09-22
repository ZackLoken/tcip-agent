"""Constructing and persisting training splits from a data config, beside ``splits.py``.

Resolves ``(train_ds, val_ds)`` for a run (a bound selection, an auto group-aware draw, or a
single-source spatial-strip split) and persists the drawn/bound membership as the run's
``split.json`` provenance record.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from tcip_mcp.pipelines.data.selection import ClassScope, Sample

logger = logging.getLogger(__name__)


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


def persist_run_partition(experiment_id: str, data_cfg: dict, *,
                           dataset_id: str | None = None,
                           dataset_fingerprint: str | None = None,
                           partition: dict | None = None) -> None:
    """Persist which members (+ seed + dataset identity) produced this run's metrics.

    ``partition`` is ``auto_train_val``'s own third return value: the run's recorded membership
    per ground-truth scope, each block naming its own sides, the group key each member was drawn
    under, the source each member's pixels came from, the file that answered for it and what that
    file digested to. Every run whose membership the platform's own producer named supplies one,
    bound or not, whatever its task and whatever built its loaders. It is ``None`` only for the
    within-image spatial route, whose manifest below is its own record and whose members are
    region identities rather than bare stems, and for a run over a task the platform reads no
    ground truth for, which has no membership here to record; nothing is read off the built
    loaders, since a loader's own keys are not what this record names.

    The same seed yields a different split if the label set changes, so a metric is only reproducible
    with the exact train/val membership recorded beside it. The whole-dataset ``dataset_fingerprint``
    (+ id) records the content identity too, so this artifact is literally "fingerprint + split",
    content identity + membership + seed in one immutable record. Whether this run's own ground
    truth has moved since is the per-member ``label_digests.at_run`` inside each scope block, read
    by every consumer that asks; no combined hash of the scope is recorded beside it, since one
    would answer for a directory of per-image label documents alone and would read as the hash of
    nothing for a scope holding any other shape. Best-effort against an ordinary
    write failure, but a write refused because the experiment is already terminal
    (:class:`~tcip_mcp.experiments.ExperimentTerminal`) propagates: a run whose provenance record
    was refused is a failed run, not a silently degraded one.

    The one writer of that member; :func:`~tcip_mcp.experiments.read_run_partition` is the one
    reader every consumer of the membership goes through. Membership is recorded per scope and
    nowhere else: a later selection check narrows itself to the scope a calibration named rather
    than to a capture date, so a run whose members span dates is checked the same way a
    single-date one is, and a bare member name means one image only within the scope its own
    block names. When ``data_cfg["split"]`` carries a ``selection_binding`` (a run bound to a
    ``data.split.selection_dir`` selection, see :func:`auto_train_val`), its counts ride into this
    record too, so a reviewer opening this one file can see that a recorded partition, not a
    drawn one, governed the run.

    A group key in this record is keyed by the bare member name its own scope block scopes, the
    way the member lists beside it in that block are. Every producer resolves a key through
    :func:`~tcip_mcp.pipelines.data.splits.recorded_group_key_fn`, so a reader reproducing a key
    for a member the record's own map does not cover spells it the way the draw did.
    """
    from tcip_mcp.experiments import ExperimentTerminal

    try:
        from tcip_store import store

        from tcip_mcp.experiments import experiment_exists, refuse_if_terminal, split_key, status_key

        split = data_cfg.get("split", {})
        resolved_group_by = split.get("resolved_group_by")
        # A spatial_strip split's members are per-region identities, never the bare stem;
        # auto_train_val already computed and stashed them (the dataset only knows tile positions).
        spatial = split.get("spatial_manifest") if resolved_group_by == "spatial_strip" else None
        record = {
            "seed": int(split.get("resolved_seed", split.get("seed", 42))),
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            # The actually resolved grouping ("explicit_map"/a named strategy/"spatial_strip"/
            # None); _train_disjointness recomputes group keys from this.
            "group_by": resolved_group_by,
        }
        if partition:
            # Per ground-truth scope: a bare member name means one image only within one of them.
            record["members"] = partition
        group_key_map = split.get("resolved_group_key_map") or split.get("group_key_map")
        if resolved_group_by == "explicit_map" and group_key_map:
            # The map itself: without it _train_disjointness has a policy name but no way to
            # compute group keys for stems outside this run.
            record["group_key_map"] = group_key_map
        if spatial:
            # The within-image route's own record: region identities, not bare member names.
            record["train"] = spatial["train_identities"]
            record["val"] = spatial["val_identities"]
            record["spatial"] = spatial
        binding_block = split.get("selection_binding")
        if binding_block:
            # A run bound to a named selection: its counts ride here too.
            record["selection_binding"] = binding_block
            if binding_block.get("redraw"):
                record["redrawn_within_selection"] = True
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


def spatial_split_raster_identity(source: str) -> dict | None:
    """This mosaic's own :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity`, best
    effort, over the source the admitted sample already names: recorded into ``spatial_manifest``
    at spatial-split time (the training source is first known to be a raster here), read back at
    export time (``inference_tools._export_predictions_raster``) to gate a block-calibrated
    bundle's claim scope to this exact mosaic. A provenance write must never sink a launch: an
    unreadable/unsupported source (a bespoke ``dataset_source``, a corrupt file) logs and returns
    ``None`` rather than raising, the same posture ``persist_run_partition`` already takes for its
    own best-effort writes.
    """
    try:
        from tcip_mcp.pipelines.derivations import probe_channels
        from tcip_mcp.pipelines.image_utils import resolve_source_path
        from tcip_mcp.pipelines.raster_source import content_identity

        resolved = resolve_source_path(source)
        nc = probe_channels(resolved)
        identity = content_identity(resolved, nc)
        import dataclasses
        return dataclasses.asdict(identity)
    except Exception as exc:  # noqa: BLE001, best-effort provenance, never sinks the split/launch
        logger.warning("raster content identity for %r could not be recorded: %s", source, exc)
        return None


def spatial_single_source_split(
    sample: "Sample", scope: "ClassScope", tiling: dict, split_cfg: dict, transforms,
    sizes: "Mapping[str, int]",
) -> tuple | None:
    """A train/val split over one detection source's own tile lattice, by disjoint pixel strips.

    Called from ``auto_train_val``'s single-source branch, and from the preflight leg that reports
    whether a reserved calibration region is feasible, so both derive their geometry from one
    implementation over one dataset. There is no second stem to hold out whole, but a tiled source
    has many tiles, and :func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split` can hold out
    disjoint, buffered regions of them. ``sample`` is the run's own single admitted sample, which
    every view here is built over: the untiled base each tiled view wraps is built from it, so the
    loaders name their member the way the producer did rather than rediscovering it from a
    directory, and a caller that already holds the sample hands it over rather than admitting
    again to find it. ``sizes`` is what this run resolved (:func:`run_sizes`), passed to the one
    factory call here so the tile lattice is indexed and the tiles read at the band count the run
    states, the sizes every other route builds its loaders at. A test region is derived and
    reserved alongside train/val (excluded from both, so it
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
    from tcip_mcp.pipelines.data.datasets import (
        TILE_OVERLAP, TILE_SIZE, DetectionDataset, TiledDetectionDataset, build_dataset,
        tile_kwargs_from_tiling,
    )
    from tcip_mcp.pipelines.data.splits import label_document_extent, spatial_strip_split

    stem = sample.member_stem
    base = build_dataset(
        "detection", samples=[sample], transforms=None, scope=scope, sizes=sizes)
    assert isinstance(base, DetectionDataset), "a detection build over samples is one of these"
    reserve_cal = float(split_cfg.get("reserve_calibration_fraction") or 0.0)

    extent = label_document_extent(sample.ground_truth)
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
    # The tiler's own defaults, so the geometry derived here and the datasets built below resolve
    # to one lattice.
    tile_size = tile_kwargs.get("tile_size", TILE_SIZE)
    overlap = tile_kwargs.get("overlap", TILE_OVERLAP)
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
        # Through the dataset's own member stem: a tile is keyed by the sample it was cut from,
        # and a region identity names the bare stem every consumer of this manifest joins on.
        raw = {spatial.identity_for(ds.member_stem_of(key), tx, ty)
               for key, tx, ty in ds.tile_entries}
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
        "raster_content_identity": spatial_split_raster_identity(sample.source),
    }
    logger.info(
        "Spatial train/val split for %r: %d train / %d val tiles (axis=%s, "
        "realized_fractions=%s, realized_discard_fraction=%.3f).",
        stem, train_ds.num_samples, val_ds.num_samples, spatial.axis,
        spatial.realized_fractions, spatial.realized_discard_fraction,
    )
    return train_ds, val_ds


def _redrawn_selection(selection, selection_dir: str, seed: int):
    """``selection`` with train and val redrawn fresh over its own train-plus-val samples at
    ``seed``, calibration untouched.

    The draw is :func:`~tcip_mcp.pipelines.data.splits.draw_train_val` over the samples' own
    recorded group keys, at the val share the selection already delivered, stratified by each
    sample's own foreground count
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`). A starved side
    refuses by name rather than retrying or degrading: a run that asked for a redraw and got one
    empty loader trained on a partition nobody chose.
    """
    from tcip_mcp.pipelines.data.label_queries import foreground_counts
    from tcip_mcp.pipelines.data.selection import with_sides
    from tcip_mcp.pipelines.data.splits import draw_train_val

    pool = selection.on("train") + selection.on("val")
    val_ratio = len(selection.on("val")) / len(pool)
    by_identity = {s.identity: s for s in pool}
    group_of = {s.identity: s.group for s in pool}
    counts = foreground_counts(by_identity, selection.scope)
    train_ids, val_ids = draw_train_val(
        sorted(by_identity), annotation_counts=counts, group_key_fn=lambda i: group_of[i],
        val_ratio=val_ratio, seed=seed,
    )
    if not train_ids or not val_ids:
        from tcip_mcp.pipelines.data.splits import redraw_starved_issue

        raise ValueError(redraw_starved_issue(
            selection, selection_dir=selection_dir, seed=seed,
        ) or (
            f"redrawing train and val inside the selection at {selection_dir!r} at seed {seed} "
            f"starved a side (train={len(train_ids)}, val={len(val_ids)})."
        ))
    assignment = {s.identity: s.side for s in selection.on("calibration")}
    assignment.update({i: "train" for i in train_ids})
    assignment.update({i: "val" for i in val_ids})
    return with_sides(selection, assignment)


def _recorded_partition(
    train_samples: "Sequence[Sample]", val_samples: "Sequence[Sample]",
    bound: "Sequence[Sample]",
) -> dict:
    """The run's own recorded partition, for :func:`persist_run_partition` to write onto
    ``split.json``: per ground-truth scope, its train and val members, the group key each was
    drawn under, the source each one's pixels came from, and the digest window a later selection
    check reads to name a calibration label that moved since the draw.

    Members are recorded as bare stems, whatever key the loaders indexed by
    (:attr:`~tcip_mcp.pipelines.data.selection.Sample.member_stem`): every consumer of this record
    (the disjointness checks, ``freeze_selection``, block calibration) names a member by its stem
    under one scope. Which selection a bound run bound to is one fact for the whole run, so it is
    recorded once in the binding block rather than copied into every scope here.

    A scope is where a member's ground truth lives, the directory holding its own label document
    or the one document that answers for many samples, and it is read off the samples rather than
    stated beside them, so the scope a member is filed under and the path its ground truth is read
    from cannot disagree. Membership is recorded per scope and nowhere else, because every
    calibration door names a sample by its bare stem under one of them, and a draw spans as many
    as it admitted: a flat list beside these blocks would read two dates' same-named images as one
    member, and a disjointness check over it would report the run leaking into itself.

    ``sources`` names where each member's pixels were read from, and each ``label_digests`` block
    carries ``ground_truth``, the path that answered for it, so a later reader composes the run's
    own samples out of this record alone rather than resolving either from a config location. Each
    member's ``at_run`` digest is its ground truth's own
    (:func:`~tcip_mcp.pipelines.resolution.ground_truth_digests`), read once per file however many
    members that file answers for, so a member whose ground truth is named unlike its image, or
    replaced by one of another extension, is still compared against what it was.
    """
    from tcip_mcp.pipelines.resolution import ground_truth_digests

    at_run = ground_truth_digests(s.ground_truth for s in bound)

    def _stems_under(samples: "Sequence[Sample]", scope: str) -> list[str]:
        return sorted({s.member_stem for s in samples if s.ground_truth_scope == scope})

    members: dict[str, dict] = {}
    for scope in sorted({s.ground_truth_scope for s in bound}):
        here = [s for s in bound if s.ground_truth_scope == scope]
        members[scope] = {
            "train": _stems_under(train_samples, scope),
            "val": _stems_under(val_samples, scope),
            "group_key_map": {s.member_stem: s.group for s in here},
            "sources": {s.member_stem: s.source for s in here},
            "label_digests": {
                "at_split": {s.member_stem: s.ground_truth_digest for s in here
                             if s.ground_truth_digest is not None},
                "at_run": {s.member_stem: at_run[s.ground_truth] for s in here},
                "ground_truth": {s.member_stem: s.ground_truth for s in here},
            },
        }
    return members


def recorded_side(members: dict, side: str) -> list[str]:
    """Every member a recorded partition names on one side, across the scopes it holds.

    Membership lives in the per-scope blocks and nowhere else, so a reader that narrows to no
    scope unions them here rather than reading a flat list written beside them. A bare member name
    means one image only within its own scope, so this union can name a leak that is only a shared
    filename and can never miss one the record holds; a reader that must not over-name narrows to
    one scope first.
    """
    return sorted({name for block in members.values() if isinstance(block, dict)
                   for name in block.get(side) or []})


def run_sizes(
    task: str, data_cfg: dict, samples: "Sequence[Sample]", dataset_source=None,
) -> dict[str, int]:
    """This run's own sizes (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes`), recorded on
    its data config as the class space is
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.onto`): what its loaders were built at
    is a fact about every checkpoint it produces, and every later reader takes it from there."""
    from tcip_mcp.pipelines.data.datasets import SIZE_NAMES, resolve_sizes

    sizes = resolve_sizes(task, data_cfg, samples, dataset_source)
    data_cfg.update({name: sizes.get(name) for name in SIZE_NAMES})
    return sizes


def _sample_loaders(
    task: str, data_cfg: dict, scope: "ClassScope", train_samples: "Sequence[Sample]",
    val_samples: "Sequence[Sample]", recorded: "Sequence[Sample]", tiling, transforms,
    dataset_source=None,
):
    """The run's loaders over its own samples, and the partition its record is written from.

    The one factory call shape, for a drawn run and a bound one alike: both sides are built under
    the run's own class space and the sizes :func:`run_sizes` resolves once over the samples the
    loaders are built from, so one run trains in one vocabulary at one set of sizes whichever
    directory or table a sample was admitted out of. A bespoke ``dataset_source`` builder is
    handed those same samples and that same class space, so what it builds over is what the
    producer named. ``recorded`` is the membership the run's own record is written from: the two
    sides for a drawn run, and the selection's held-out samples besides for a bound one.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset

    sizes = run_sizes(task, data_cfg, [*train_samples, *val_samples], dataset_source)
    build_kwargs: dict[str, Any] = {
        "tiling": tiling, "dataset_source": dataset_source, "scope": scope, "sizes": sizes,
    }
    train_ds = build_dataset(task, samples=list(train_samples), transforms=transforms,
                             **build_kwargs)
    val_ds = (build_dataset(task, samples=list(val_samples), transforms=None, **build_kwargs)
              if val_samples else None)
    partition = _recorded_partition(train_samples, val_samples, recorded)
    return train_ds, val_ds, partition


def _drawn_split(
    task: str, data_cfg: dict, *, images_dir: str, labels_dir: str, tiling, transforms,
    dataset_source=None,
):
    """``(train_ds, val_ds, partition)`` for a run that draws its own split.

    ``images_dir`` and ``labels_dir`` are this run's validated locations, checked by the caller
    through the one missing-key refusal (:func:`~tcip_mcp.tools.training_tools._data_dir_issues`)
    before anything reads them, so nothing here defaults a location or hands an absent one to a
    path.

    Admission runs once (:func:`~tcip_mcp.pipelines.data.label_queries.admit`), becomes explicit
    samples before any loader is built, and the loaders answer for their own members through those
    samples, so a run that drew its split and a run bound to a recorded selection hold one
    membership shape and read through the same recorded paths. The caller's own resolution order
    is kept: ``auto_val`` off trains on every admitted sample; a single admitted source with
    tiling on splits its own tile lattice spatially; anything else draws a group-aware split over
    the admitted members.

    Degrades to training without validation, naming which failure did it, when the draw or a
    malformed ``val_ratio``/``seed`` fails, or when no grouping policy can populate both sides. An
    admission failure over the run's own membership is not one of those: there is nothing left to
    train on, so it raises.

    A route that draws nothing records ``stem`` as its resolved grouping, the policy it actually
    applied: every member is its own group.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.label_queries import (
        admit, foreground_counts, require_admitted,
    )
    from tcip_mcp.pipelines.data.splits import draw_train_val, recorded_group_key_fn

    split_cfg = data_cfg.setdefault("split", {})
    admitted = admit(images_dir, labels_dir, subject=data_cfg.get("subject"),
                     attribute=data_cfg.get("attribute"))
    require_admitted(admitted)
    # The scope the producer admitted under becomes this run's whether or not it holds a map, so
    # a stale one a relaunched config carried cannot survive as this run's recorded vocabulary.
    admitted.scope.onto(data_cfg)

    def each_its_own_group(date: str | None) -> "Callable[[str], str]":
        """Every member its own group, the ``stem`` policy: a route that drew no split grouped
        nothing, and recording that policy by name lets a reader reproduce the same key for a
        member the record's own map does not cover."""
        return recorded_group_key_fn("stem", date=date)

    def train_only(group_of: "Callable[[str], str]"):
        samples = admitted.samples({m: "train" for m in members}, group_of)
        return _sample_loaders(
            task, data_cfg, admitted.scope, samples, [], samples, tiling, transforms,
            dataset_source)

    members = [record.member for record in admitted.records]
    if not data_cfg.get("auto_val", True):
        logger.info("data.auto_val is off for %s: training on all %d admitted sample(s) with no "
                    "validation.", task, len(members))
        split_cfg["resolved_group_by"] = "stem"
        return train_only(each_its_own_group(admitted.date))

    if len(members) < 2:
        # A single source cannot hold out a whole stem, but a tiled detection source the platform
        # builds itself can hold out disjoint pixel blocks of its own tile lattice.
        if (task == "detection" and tiling and tiling.get("enabled", True)
                and dataset_source is None):
            one = admitted.one_sample()
            spatial = spatial_single_source_split(
                one, admitted.scope, tiling, split_cfg, transforms,
                run_sizes(task, data_cfg, [one]))
            if spatial is not None:
                return (*spatial, None)
        logger.warning("Auto train/val split for %s: %d admitted source(s) leave nothing to hold "
                       "out; training without validation.", task, len(members))
        split_cfg["resolved_group_by"] = "stem"
        return train_only(each_its_own_group(admitted.date))

    group_by = split_cfg.get("group_by", "tile_prefix")
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any handler: a malformed grouping policy is a caller-config error.
    # Through recorded_group_key_fn, so this draw spells a group key the way draw_splits does.
    group_key_fn = recorded_group_key_fn(
        group_by, date=admitted.date, stems=members, group_key_map=group_key_map)
    split_cfg["resolved_group_by"] = "explicit_map" if group_key_map else group_by

    try:
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        seed = int(split_cfg.get("seed", 42))
        annotation_counts = None
        if split_cfg.get("stratify_foreground", True):
            # Counted off the admitted records under the scope they were admitted in, so the
            # sample list is built once, below, on the sides this draw gives them.
            annotation_counts = foreground_counts(
                {record.member: record for record in admitted.records}, admitted.scope)
        train_members, val_members = draw_train_val(
            members, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
            val_ratio=val_ratio, seed=seed,
        )
        if (not val_members or not train_members) and group_by != "stem" and not group_key_map:
            # Too few groups under the requested policy starved val; retry at stem grouping.
            retry_key_fn = recorded_group_key_fn("stem", date=admitted.date, stems=members)
            retry_train, retry_val = draw_train_val(
                members, annotation_counts=annotation_counts, group_key_fn=retry_key_fn,
                val_ratio=val_ratio, seed=seed,
            )
            if retry_train and retry_val:
                logger.info(
                    "Auto train/val split for %s: group_by=%r left val empty (too few groups); "
                    "retried at stem-level grouping.", task, group_by,
                )
                train_members, val_members = retry_train, retry_val
                split_cfg["resolved_group_by"] = "stem"
                group_key_fn = retry_key_fn
        if not val_members or not train_members:
            logger.warning(
                "Auto train/val split for %s: no grouping policy could populate both sides; "
                "training without validation.", task,
            )
            return train_only(group_key_fn)

        assignment = {s: "train" for s in train_members}
        assignment.update({s: "val" for s in val_members})
        samples = admitted.samples(assignment, group_key_fn)
        by_side = {side: [s for s in samples if s.side == side] for side in ("train", "val")}
    except UnreadableLabelDocument:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto train/val split for %s failed (%s); training without validation.",
                       task, exc)
        return train_only(group_key_fn)
    logger.info("Auto train/val split for %s: %d train / %d val samples.",
                task, len(by_side["train"]), len(by_side["val"]))
    return _sample_loaders(
        task, data_cfg, admitted.scope, by_side["train"], by_side["val"], samples, tiling,
        transforms, dataset_source)


def auto_train_val(task: str, data_cfg: dict, transforms):
    """Build ``(train_ds, val_ds, partition)`` for a run, deriving a leakage-free val split.

    ``partition`` carries the membership, group keys, ground-truth scopes and per-member digests
    :func:`persist_run_partition` writes onto ``split.json`` beside, never inside,
    ``selection_binding``: passed as its own value rather than through ``data_cfg["split"]``
    since that block is copied whole into the durable config and every checkpoint, and a
    per-sample map must not multiply through every copy. Every run records one, bound or not,
    whatever its task and whatever builds its loaders; it is ``None`` only for the within-image
    spatial route, whose members are regions and whose record is its own manifest.

    Two routes, and a run of any task takes one of them:
      1. ``data.split.selection_dir`` set -> train on the selection's own ``train`` and ``val``
         samples instead of drawing a split, each sample reading the source and the ground truth
         the draw recorded for it. A recorded partition is an explicit split ``auto_val`` does not
         govern, checked ahead of its gate below; every conflict and empty-side refusal here
         raises to the caller, and so does a build failure while binding, never degrading to
         training on the selection's held-out side with no validation. Nothing is rediscovered
         here: the selection already decided which samples train, so the loaders read it rather
         than a directory, and the selection's own subject, attribute and id map govern rather
         than being restated on the config and compared. A selection binds to whatever task reads
         the ground truth its samples name, and the loader refuses by name when the task reads
         another shape. The calibration side never builds a loader; its count rides into
         ``selection_binding``.
         ``data.split.redraw_within_selection: true`` (beside ``selection_dir`` and ``seed``)
         redraws train and val fresh inside the selection's own train-plus-val samples instead of
         binding them as recorded, calibration still untouched; a starved side refuses rather than
         retrying or degrading.
      2. Otherwise -> :func:`_drawn_split`, which admits once through the producer over whatever
         ground truth ``data.labels_dir`` holds and builds every loader from the resulting
         samples: ``auto_val`` off, a single tiled source's own spatial strips, and the
         group-aware draw are all that one route's branches. A bespoke ``data.dataset_source``
         builder takes this route too and is handed those samples.

    Reads ``auto_val`` / ``split.*`` from ``data_cfg`` (== config["data"]).
    """
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY
    from tcip_mcp.tools.training_tools import (
        _data_dir_issues, _redraw_flag_issue, _selection_dependent_issues,
        _split_selection_drawn_conflicts,
    )

    # The one missing-key refusal, so a run and the preflight that offered it name a missing or
    # moved location with the same words. A no-op for a config bound to a selection.
    location_issues = _data_dir_issues(data_cfg)
    if location_issues:
        raise ValueError(
            f"{'; '.join(location_issues)}: a run reads its own samples out of the places its "
            "config names, so there is nothing here to admit."
        )

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

    # 1. A named selection is an explicit partition auto_val does not govern; every refusal
    # here, and any build failure while binding to it, raises rather than degrading.
    if selection_dir:
        conflicts = _split_selection_drawn_conflicts(split_cfg_raw)
        if conflicts:
            raise ValueError(
                f"data.split.selection_dir conflicts with {sorted(conflicts)}: a recorded "
                "partition and a drawn split's own parameters/source cannot both govern one run."
            )
        flag_issue = _redraw_flag_issue(split_cfg_raw)
        if flag_issue:
            raise ValueError(flag_issue)

        from tcip_mcp.pipelines.data.label_queries import refuse_inadmissible_samples
        from tcip_mcp.pipelines.data.selection import read_selection
        from tcip_mcp.pipelines.resolution import selection_digest

        selection = read_selection(selection_dir)
        # The bind's own refusals, stated once so the preflight that offered this selection and
        # the launch that binds it say the same thing.
        bind_issues = _selection_dependent_issues(selection, selection_dir)
        if bind_issues:
            raise ValueError(" ".join(bind_issues))

        # A redraw repartitions the selection's own train-plus-val samples; calibration untouched.
        if split_cfg_raw.get("redraw_within_selection"):
            redraw_seed = int(split_cfg_raw["seed"])
            selection = _redrawn_selection(selection, selection_dir, redraw_seed)
            split_cfg_raw["_redraw_seed"] = redraw_seed

        train_samples, val_samples = selection.on("train"), selection.on("val")
        calibration_samples = selection.on("calibration")

        # Checked before either loader is built: a selected label that has since emptied with
        # nobody confirming that image negative would otherwise train as background.
        refuse_inadmissible_samples(train_samples + val_samples, selection.scope)

        bound = train_samples + val_samples + calibration_samples
        # The selection's own scope and exact map become this run's, so the checkpoint records
        # the vocabulary it trained in rather than one rediscovered from a live registry.
        selection.scope.onto(data_cfg)
        split_cfg = data_cfg.setdefault("split", {})
        # The selection's own named policy, not "explicit_map": the per-stem map the partition
        # records covers this run's members, and a stem outside it is what a policy name answers.
        split_cfg["resolved_group_by"] = selection.group_by
        split_cfg["resolved_seed"] = split_cfg_raw.pop("_redraw_seed", selection.seed)
        # The binding's own counts and the selection it bound, including the digest a later
        # calibration compares to see whether that selection was redrawn since this run.
        split_cfg["selection_binding"] = {
            "selection_dir": selection_dir,
            "assigned": len(selection.samples),
            "train_bound": len(train_samples), "val_bound": len(val_samples),
            "calibration_bound": len(calibration_samples),
            "dataset_fingerprint_at_draw": selection.dataset_fingerprint,
            "selection_sha256": selection_digest(selection),
        }
        if split_cfg_raw.get("redraw_within_selection"):
            split_cfg["selection_binding"]["redraw"] = {
                "seed": split_cfg["resolved_seed"],
                "val_ratio": len(val_samples) / (len(train_samples) + len(val_samples)),
            }

        # The partition is kept out of split_cfg/selection_binding: that block is copied whole
        # into the durable config and every checkpoint.
        return _sample_loaders(
            task, data_cfg, selection.scope, train_samples, val_samples, bound, tiling,
            transforms, data_cfg.get(DATASET_SOURCE_KEY) or None)

    # 2. Otherwise the producer names this run's membership off the ground truth its config points
    # at, and every branch of the resolution order below builds from the samples it made.
    return _drawn_split(
        task, data_cfg, images_dir=data_cfg["images_dir"], labels_dir=data_cfg["labels_dir"],
        tiling=tiling, transforms=transforms,
        dataset_source=data_cfg.get(DATASET_SOURCE_KEY) or None)

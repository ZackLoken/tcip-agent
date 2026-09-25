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


def split_seed(split_cfg: "Mapping[str, Any]") -> int:
    """The seed a draw over ``split_cfg`` resolves: its stated ``seed``, else
    :data:`~tcip_mcp.pipelines.data.splits.DEFAULT_SEED`."""
    from tcip_mcp.pipelines.data.splits import DEFAULT_SEED

    return int(split_cfg.get("seed", DEFAULT_SEED))


def dataset_identity(data_cfg: dict) -> tuple[str | None, str | None]:
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset. The fingerprint is recomputed
    here; the id comes from the dataset's ``dataset.json`` if it was registered. ``(None, None)``
    for a bespoke / imageless run (no dataset_root). ``tcip_store.SchemaVersionRefused``
    propagates.
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

    ``partition`` is ``auto_train_val``'s own third return value: the run's recorded membership per
    ground-truth scope, each block naming its own sides, the group key each member was drawn under,
    the source each member's pixels came from, the file that answered for it and what that file
    digested to. It is ``None`` only for the within-image spatial route, whose manifest below is
    its own record, and for a run over a task the platform reads no ground truth for.

    The record holds the whole-dataset ``dataset_fingerprint`` (+ id), the membership and the seed;
    each scope block's per-member ``label_digests.at_run`` states what its ground truth was. When
    ``data_cfg["split"]`` carries a ``selection_binding``, the binding rides into this record too.
    Written through :func:`~tcip_mcp.experiments.rewrite_live_member`: a terminal record refuses
    with :class:`~tcip_mcp.experiments.ExperimentTerminal`, and every write failure raises.

    A group key in this record is keyed by the bare member name its own scope block scopes,
    resolved through :func:`~tcip_mcp.pipelines.data.splits.recorded_group_key_fn`.
    """
    from tcip_mcp.experiments import experiment_exists, rewrite_live_member, split_key

    split = data_cfg["split"]
    resolved_group_by = split["resolved_group_by"]
    record = {
        "seed": split["resolved_seed"],
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        # The actually resolved grouping ("explicit_map"/a named strategy/"spatial_strip").
        "group_by": resolved_group_by,
    }
    if partition:
        # Per ground-truth scope: a bare member name means one image only within one of them.
        record["members"] = partition
    if resolved_group_by == "spatial_strip":
        # The within-image route's own record: region identities, not bare member names.
        record["spatial"] = split["spatial_manifest"]
    if "selection_binding" in split:
        record["selection_binding"] = split["selection_binding"]
    if experiment_exists(experiment_id):
        rewrite_live_member(experiment_id, split_key(experiment_id), "persist_run_partition",
                            lambda _stored: record)


def spatial_split_raster_identity(source: str) -> dict | None:
    """This mosaic's :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity` over the
    admitted source, or ``None`` (logged) when the source is unreadable or unsupported.
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
    """A train/val split over one detection source's own tile lattice, by disjoint pixel strips
    (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`).

    ``sample`` is the run's own single admitted sample, which every view here is built over.
    ``sizes`` is what this run resolved (:func:`run_sizes`). A test region is derived and reserved
    alongside train/val (excluded from both) with only its geometry and kept-tile count recorded;
    no dataset is built for it.

    ``split_cfg["reserve_calibration_fraction"]`` (opt-in, default unset/0) reserves a fourth
    region, ``calibration``, alongside train/val/test, at that fraction of the axis. When set, each
    of :func:`spatial_strip_split`'s ``None`` reasons (no extent from the label file; the strip
    layout itself infeasible; an empty train/val/test/calibration side surviving tile filtering)
    raises ``ValueError`` naming which one fired.

    Returns ``(train_ds, val_ds)``, or ``None`` when ``reserve_calibration_fraction`` was not
    requested and the extent is unknown or no strip layout can populate both train and val. A
    present, unreadable label document raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` either way.
    """
    from tcip_mcp.pipelines.data.datasets import (
        TILE_OVERLAP, TILE_SIZE, DetectionDataset, TiledDetectionDataset, build_dataset,
        tile_kwargs_from_tiling,
    )
    from tcip_mcp.pipelines.data.splits import (
        DEFAULT_VAL_RATIO, label_document_extent, spatial_strip_split,
    )

    stem = sample.member
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
    val_ratio = float(split_cfg.get("val_ratio", DEFAULT_VAL_RATIO))
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
        raw = {spatial.identity_for(ds.member_of(key), tx, ty)
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
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`). A starved side refuses by
    name.
    """
    from tcip_mcp.pipelines.data.selection import with_sides
    from tcip_mcp.pipelines.data.splits import (
        draw_train_val, redraw_pool, redraw_starved_issue,
    )

    group_of, counts = redraw_pool(selection)
    train_ids, val_ids = draw_train_val(
        sorted(group_of), annotation_counts=counts, group_key_fn=group_of.__getitem__,
        val_ratio=len(selection.on("val")) / len(group_of), seed=seed,
    )
    if not train_ids or not val_ids:
        raise ValueError(redraw_starved_issue(
            group_of, counts, selection_dir=selection_dir, seed=seed,
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
    ``split.json``: per ground-truth scope, its train and val members, the group key each was drawn
    under, the source each one's pixels came from, and the digest window a later selection check
    reads to name a calibration label that moved since the draw.

    Members are recorded by :attr:`~tcip_mcp.pipelines.data.selection.Sample.member`, per scope and
    nowhere else. A scope is where a member's ground truth lives, the directory holding its own
    label document or the one document that answers for many samples, read off the samples. Which
    selection a bound run bound to is recorded once, in the binding block.

    ``row_keys`` names the row each tabular member reads, and ``confirmation_bucket`` the
    ``image_status.json`` key a document scope's samples were admitted under (``None`` for a mask
    or a table). ``sources`` names where each member's pixels were read from, and each
    ``label_digests`` block carries ``ground_truth``, the path that answered for it. Each member's
    ``at_run`` digest is its ground truth's own
    (:func:`~tcip_mcp.pipelines.resolution.ground_truth_digests`), read once per file however many
    members that file answers for.
    """
    from tcip_mcp.pipelines.resolution import ground_truth_digests

    at_run = ground_truth_digests(s.ground_truth for s in bound)

    def _stems_under(samples: "Sequence[Sample]", scope: str) -> list[str]:
        return sorted({s.member for s in samples if s.ground_truth_scope == scope})

    members: dict[str, dict] = {}
    for scope in sorted({s.ground_truth_scope for s in bound}):
        here = [s for s in bound if s.ground_truth_scope == scope]
        members[scope] = {
            "train": _stems_under(train_samples, scope),
            "val": _stems_under(val_samples, scope),
            "group_key_map": {s.member: s.group for s in here},
            "sources": {s.member: s.source for s in here},
            "row_keys": {s.member: s.row_key for s in here if s.row_key is not None},
            "confirmation_bucket": here[0].confirmation_bucket,
            "label_digests": {
                "at_split": {s.member: s.ground_truth_digest for s in here
                             if s.ground_truth_digest is not None},
                "at_run": {s.member: at_run[s.ground_truth] for s in here},
                "ground_truth": {s.member: s.ground_truth for s in here},
            },
        }
    return members


def bound_selection_dir(split: "Mapping[str, Any]") -> str | None:
    """The selection a run's split block (a recorded partition, or a config's ``data.split``) was
    bound to, or ``None`` for a run that drew its own."""
    binding = split.get("selection_binding")
    return binding["selection_dir"] if binding else None


def recorded_side(members: dict, side: str) -> list[str]:
    """Every member a recorded partition names on one side, across the scopes it holds. A bare
    member name means one image only within its own scope, so this union can over-name a leak that
    is only a shared filename; narrow to one scope first where that matters.
    """
    return sorted({name for block in members.values() for name in block[side]})


def run_sizes(
    task: str, data_cfg: dict, samples: "Sequence[Sample]", dataset_source=None,
) -> dict[str, int]:
    """This run's own sizes (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes`), recorded on
    its data config as the class space is
    (:meth:`~tcip_mcp.pipelines.data.selection.ClassScope.onto`).
    """
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

    Both sides are built under the run's own class space and the sizes :func:`run_sizes` resolves
    once over the samples the loaders are built from. A bespoke ``dataset_source`` builder is
    handed those same samples and that same class space. ``recorded`` is the membership the run's
    own record is written from: the two sides for a drawn run, and the selection's held-out samples
    besides for a bound one.
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
    task: str, data_cfg: dict, *, tiling, transforms, dataset_source=None,
):
    """``(train_ds, val_ds, partition)`` for a run that draws its own split over ``data_cfg``'s
    ground truth.

    Admission runs once (:func:`~tcip_mcp.pipelines.data.label_queries.admit_run`) and becomes
    explicit samples before any loader is built. ``auto_val`` off trains on every admitted sample;
    a single admitted source with tiling on splits its own tile lattice spatially; anything else
    draws a group-aware split over the admitted members.

    Degrades to training without validation, naming which failure did it, when the draw or a
    malformed ``val_ratio``/``seed`` fails, or when no grouping policy can populate both sides. An
    admission failure over the run's own membership raises.

    A route that draws nothing records ``stem`` as its resolved grouping.
    """
    from tcip_annotation.json_io import UnreadableLabelDocument
    from tcip_mcp.pipelines.data.label_queries import (
        admit_run, foreground_counts, require_admitted,
    )
    from tcip_mcp.pipelines.data.splits import (
        DEFAULT_GROUP_BY, DEFAULT_VAL_RATIO, draw_train_val, recorded_group_by,
        recorded_group_key_fn,
    )

    split_cfg = data_cfg.setdefault("split", {})
    # A route that draws nothing uses no seed; the draw below records the one it used.
    split_cfg["resolved_seed"] = None
    admitted = admit_run(data_cfg)
    require_admitted(admitted)
    # The scope the producer admitted under becomes this run's whether or not it holds a map, so
    # a stale one a relaunched config carried cannot survive as this run's recorded vocabulary.
    admitted.scope.onto(data_cfg)

    # A route that draws nothing groups each member alone, the ``stem`` policy, recorded by name.
    each_its_own_group = recorded_group_key_fn("stem", date=admitted.date)

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
        return train_only(each_its_own_group)

    if len(members) < 2:
        # A single source cannot hold out a whole stem, but a tiled detection source the platform
        # builds itself can hold out disjoint pixel blocks of its own tile lattice.
        if (task == "detection" and tiling and tiling.get("enabled", True)
                and dataset_source is None):
            one = admitted.every_sample()[0]
            spatial = spatial_single_source_split(
                one, admitted.scope, tiling, split_cfg, transforms,
                run_sizes(task, data_cfg, [one]))
            if spatial is not None:
                return (*spatial, None)
        logger.warning("Auto train/val split for %s: %d admitted source(s) leave nothing to hold "
                       "out; training without validation.", task, len(members))
        split_cfg["resolved_group_by"] = "stem"
        return train_only(each_its_own_group)

    group_by = split_cfg.get("group_by", DEFAULT_GROUP_BY)
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any handler: a malformed grouping policy is a caller-config error.
    # Through recorded_group_key_fn, so this draw spells a group key the way draw_splits does.
    group_key_fn = recorded_group_key_fn(
        group_by, date=admitted.date, stems=members, group_key_map=group_key_map)
    split_cfg["resolved_group_by"] = recorded_group_by(group_by, group_key_map)

    try:
        val_ratio = float(split_cfg.get("val_ratio", DEFAULT_VAL_RATIO))
        seed = split_seed(split_cfg)
        split_cfg["resolved_seed"] = seed
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
    ``selection_binding``. It is ``None`` only for the within-image spatial route, whose members
    are regions and whose record is its own manifest.

    Two routes, and a run of any task takes one of them:
      1. ``data.split.selection_dir`` set -> train on the selection's own ``train`` and ``val``
        samples instead of drawing a split, each sample reading the source and the ground truth the
        draw recorded for it, under the selection's own subject, attribute and id map. Checked
        ahead of ``auto_val``; every conflict, empty-side refusal and build failure raises. The
        loader refuses by name when the task reads another shape than the samples name. The
        calibration side never builds a loader.
        ``data.split.redraw_within_selection: true`` (beside ``selection_dir`` and ``seed``)
        redraws train and val fresh inside the selection's own train-plus-val samples, calibration
        still untouched; a starved side refuses.
      2. Otherwise -> :func:`_drawn_split`, which admits once through the producer over whatever
        ground truth ``data.labels_dir`` holds and builds every loader from the resulting samples.
        A bespoke ``data.dataset_source`` builder takes this route too and is handed those samples.

    Reads ``auto_val`` / ``split.*`` from ``data_cfg`` (== config["data"]).
    """
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY
    from tcip_mcp.tools.training_tools import (
        _data_dir_issues, _selection_dependent_issues, _selection_dir_conflicts,
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
        "selection_binding", "resolved_group_by", "resolved_seed",
    ):
        split_cfg_raw.pop(_stale_key, None)

    # 1. A named selection is an explicit partition auto_val does not govern; every refusal
    # here, and any build failure while binding to it, raises rather than degrading.
    if selection_dir:
        conflicts = _selection_dir_conflicts(data_cfg)
        if conflicts:
            raise ValueError(" ".join(conflicts))

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
        redraw = bool(split_cfg_raw.get("redraw_within_selection"))
        seed = selection.seed
        if redraw:
            seed = int(split_cfg_raw["seed"])
            selection = _redrawn_selection(selection, selection_dir, seed)

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
        split_cfg["resolved_seed"] = seed
        # The selection this run bound, the digest a later calibration compares to see whether
        # that selection was redrawn since, and whether this run redrew inside it.
        split_cfg["selection_binding"] = {
            "selection_dir": selection_dir,
            "selection_sha256": selection_digest(selection),
            "redraw": redraw,
        }

        # The partition is kept out of split_cfg/selection_binding: that block is copied whole
        # into the durable config and every checkpoint.
        return _sample_loaders(
            task, data_cfg, selection.scope, train_samples, val_samples, bound, tiling,
            transforms, data_cfg.get(DATASET_SOURCE_KEY) or None)

    # 2. Otherwise the producer names this run's membership off the ground truth its config points
    # at, and every branch of the resolution order below builds from the samples it made.
    return _drawn_split(task, data_cfg, tiling=tiling, transforms=transforms,
                        dataset_source=data_cfg.get(DATASET_SOURCE_KEY) or None)

"""Constructing training splits from a data config, beside ``splits.py``.

Resolves a run once at launch (:func:`resolve_run`): its data section, the partition its record
holds (a bound selection, an auto group-aware draw, or a single-source spatial-strip split) and
its objective, and builds ``(train_ds, val_ds)`` from that record (:func:`recorded_datasets`).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, NamedTuple

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
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset: the fingerprint recomputed, the
    id from the dataset's identity record, ``None`` for a dataset never registered. ``(None,
    None)`` for a run whose images sit under no dataset root. A fingerprint that cannot be read and
    an identity record that does not decode raise, as does ``tcip_store.SchemaVersionRefused``.
    """
    images_dir = data_cfg.get("images_dir")
    if not images_dir:
        return None, None

    from tcip_mcp.dataset_layout import dataset_root_of, read_dataset_identity
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint

    root = dataset_root_of(images_dir)
    if root is None:
        return None, None
    record = read_dataset_identity(root)
    return (record["id"] if record is not None else None), dataset_fingerprint(root)


def spatial_split_raster_identity(source: str) -> dict:
    """This mosaic's :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity` over the
    admitted source. A source that will not probe raises."""
    import dataclasses

    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import resolve_source_path
    from tcip_mcp.pipelines.raster_source import content_identity

    resolved = resolve_source_path(source)
    return dataclasses.asdict(content_identity(resolved, probe_channels(resolved)))


def spatial_single_source_split(
    sample: "Sample", scope: "ClassScope", tiling: dict, split_cfg: dict,
    sizes: "Mapping[str, int]",
) -> bool:
    """Derive a train/val split over one detection source's own tile lattice, by disjoint pixel
    strips (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`), and record it as
    ``split_cfg["spatial_manifest"]``, which :func:`recorded_datasets` builds the run's views from.

    ``sample`` is the run's own single admitted sample, which every view here is built over
    (:func:`_spatial_views`). ``sizes`` is what this run resolved (:func:`run_sizes`). A test
    region is derived and reserved alongside train/val (excluded from both) with only its geometry
    and kept-tile count recorded.

    ``split_cfg["reserve_calibration_fraction"]`` (opt-in, default unset/0) reserves a fourth
    region, ``calibration``, alongside train/val/test, at that fraction of the axis. When set, each
    of :func:`spatial_strip_split`'s ``None`` reasons (no extent from the label file; the strip
    layout itself infeasible; an empty train/val/test/calibration side surviving tile filtering)
    raises ``ValueError`` naming which one fired.

    Returns whether the manifest was recorded: ``False`` when ``reserve_calibration_fraction`` was
    not requested and the extent is unknown or no strip layout can populate both train and val. A
    present, unreadable label document raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` either way.
    """
    from tcip_mcp.pipelines.data.datasets import TILE_SIZE
    from tcip_mcp.pipelines.resolution import DEFAULT_OVERLAP
    from tcip_mcp.pipelines.data.splits import (
        DEFAULT_VAL_RATIO, label_document_extent, spatial_strip_split,
    )

    stem = sample.member
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
        return False
    width, height = extent

    tile_options = _tile_options(tiling)
    # The tiler's own defaults, so the geometry derived here and the views built below resolve
    # to one lattice.
    tile_size = tile_options.get("tile_size", TILE_SIZE)
    overlap = tile_options.get("overlap", DEFAULT_OVERLAP)
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
        return False

    # A tile lattice occupying a reserved region (spatial_strip_split's own check) is not proof it
    # carries GT: an all-background region still passes that but skip_empty filters it to 0.
    views = _spatial_views([sample], scope, sizes, tile_options, {
        side: spatial.regions[side]
        for side in ("train", "val", *(("test", "calibration") if reserve_cal else ()))
    }, transforms=None)
    train_ds, val_ds = views["train"], views["val"]
    if any(view.num_samples == 0 for view in views.values()):
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
        return False

    def _identities(ds) -> list[str]:
        # Through the dataset's own member stem: a tile is keyed by the sample it was cut from,
        # and a region identity names the bare stem every consumer of this manifest joins on.
        raw = {spatial.identity_for(ds.member_of(key), box) for key, box in ds.tile_entries}
        return sorted(name for name in raw if name is not None)

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
    return True


def _tile_options(tiling: dict) -> dict:
    """``tiling``'s tile options (``datasets.tile_kwargs_from_tiling``) without ``keep_regions``,
    which a spatial split sets per side."""
    from tcip_mcp.pipelines.data.datasets import tile_kwargs_from_tiling

    return {k: v for k, v in tile_kwargs_from_tiling(tiling).items() if k != "keep_regions"}


def _spatial_views(samples: "Sequence[Sample]", scope: "ClassScope", sizes: "Mapping[str, int]",
                   tile_options: dict, regions: "Mapping[str, list]", *, transforms) -> dict:
    """One view per side of ``regions``: the tile lattice under ``tile_options`` of the detection
    dataset over ``samples`` at ``scope`` and ``sizes``, kept to that side's region, the ``train``
    view alone under ``transforms``."""
    from tcip_mcp.pipelines.data.datasets import (
        DetectionDataset, TiledDetectionDataset, build_dataset,
    )

    base = build_dataset("detection", samples=list(samples), transforms=None, scope=scope,
                         sizes=sizes)
    assert isinstance(base, DetectionDataset), "a detection build over samples is one of these"
    return {side: TiledDetectionDataset(base, transforms=transforms if side == "train" else None,
                                        keep_regions=region, **tile_options)
            for side, region in regions.items()}


def _redrawn_selection(selection, selection_dir: str, seed: int):
    """``selection`` with train and val redrawn fresh over its own train-plus-val samples at
    ``seed``, calibration untouched.

    The draw is :func:`~tcip_mcp.pipelines.data.splits.draw_train_val` over the samples' own
    recorded group keys, at the val share the selection already delivered, stratified by each
    sample's own foreground count
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`). A pool that cannot give
    both sides a foreground group
    (:func:`~tcip_mcp.pipelines.data.splits.redraw_starved_issue`), and a draw that starves a
    side, refuse by name.
    """
    from tcip_mcp.pipelines.data.selection import with_sides
    from tcip_mcp.pipelines.data.splits import (
        draw_train_val, redraw_pool, redraw_starved_issue,
    )

    group_of, counts = redraw_pool(selection)
    starved = redraw_starved_issue(group_of, counts, selection_dir=selection_dir, seed=seed)
    if starved is not None:
        raise ValueError(starved)
    train_ids, val_ids = draw_train_val(
        sorted(group_of), annotation_counts=counts, group_key_fn=group_of.__getitem__,
        val_ratio=len(selection.on("val")) / len(group_of), seed=seed,
    )
    if not train_ids or not val_ids:
        raise ValueError(
            f"redrawing train and val inside the selection at {selection_dir!r} at seed {seed} "
            f"starved a side (train={len(train_ids)}, val={len(val_ids)}).")
    assignment = {s.identity: s.side for s in selection.on("calibration")}
    assignment.update({i: "train" for i in train_ids})
    assignment.update({i: "val" for i in val_ids})
    return with_sides(selection, assignment)


def _partition_record(samples: "Sequence[Sample]", *, seed: int | None, group_by: str,
                      selection: dict | None) -> dict:
    """The partition a run's resolved record holds: the draw's ``seed`` (``None`` for a route
    that drew nothing) and resolved ``group_by``, every sample the run bound on the side it
    landed on in its one recorded shape
    (:func:`~tcip_mcp.pipelines.data.selection.sample_document`), ``ground_truth_digests``, each
    ground-truth file's digest when this run read it, keyed by its path, and ``selection``, the
    selection a bound run bound (its directory, its digest and whether the run redrew inside it),
    ``None`` for a run that drew its own."""
    from tcip_mcp.pipelines.data.selection import sample_document
    from tcip_mcp.pipelines.resolution import ground_truth_digests

    return {
        "seed": seed,
        "group_by": group_by,
        "samples": [sample_document(s) for s in samples],
        "ground_truth_digests": ground_truth_digests(s.ground_truth for s in samples),
        "selection": selection,
    }


def partition_samples(partition: "Mapping[str, Any]") -> list["Sample"]:
    """A resolved partition's samples read back
    (:func:`~tcip_mcp.pipelines.data.selection.read_sample`)."""
    from tcip_mcp.pipelines.data.selection import read_sample

    return [read_sample(raw, position, "a run's resolved partition")
            for position, raw in enumerate(partition["samples"])]


def run_sizes(
    task: str, data_cfg: dict, samples: "Sequence[Sample]", dataset_source=None,
) -> dict[str, int]:
    """This run's own sizes (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes`), recorded on
    its data config beside its ``scope``, each by its own name.
    """
    from tcip_mcp.pipelines.data.datasets import SIZE_NAMES, resolve_sizes

    sizes = resolve_sizes(task, data_cfg, samples, dataset_source)
    data_cfg.update({name: sizes.get(name) for name in SIZE_NAMES})
    return sizes


def _sample_loaders(task: str, data_cfg: dict, recorded: "Sequence[Sample]", transforms, *,
                    seed: int | None, group_by: str, selection: dict | None = None):
    """``(train_ds, val_ds, partition)``: the partition (:func:`_partition_record` over
    ``recorded`` at ``seed``, ``group_by`` and the ``selection`` a bound run bound) and
    :func:`recorded_datasets` over it and ``data_cfg``, whose class space and sizes
    (:func:`run_sizes`) are recorded. ``recorded`` is the membership the partition holds: the two
    sides for a drawn run, and the selection's held-out samples besides for a bound one."""
    partition = _partition_record(recorded, seed=seed, group_by=group_by, selection=selection)
    return (*recorded_datasets(task, data_cfg, partition, transforms), partition)


def _drawn_split(
    task: str, data_cfg: dict, *, tiling, transforms, dataset_source=None,
    contradicted_out: set[str] | None, counts_out: dict[str, int] | None,
):
    """``(train_ds, val_ds, partition)`` for a run that draws its own split over ``data_cfg``'s
    ground truth.

    Admission runs once (:func:`~tcip_mcp.pipelines.data.label_queries.admit_run`, handed
    ``contradicted_out``, its counts copied into ``counts_out``) and becomes explicit samples
    before any loader is built. ``auto_val`` off trains on every admitted sample; a single admitted
    source with tiling on splits its own tile lattice spatially; anything else draws a group-aware
    split over the admitted members.

    Degrades to training without validation, naming which failure did it, when the draw or a
    malformed ``val_ratio``/``seed`` fails, or when no grouping policy can populate both sides. An
    admission failure over the run's own membership raises.

    A route that draws nothing records ``stem`` as its resolved grouping and no seed.
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
    admitted = admit_run(data_cfg, contradicted_out=contradicted_out)
    if counts_out is not None:
        counts_out.update(admitted.counts)
    require_admitted(admitted)
    data_cfg["scope"] = asdict(admitted.scope)

    # A route that draws nothing groups each member alone, the ``stem`` policy, recorded by name.
    each_its_own_group = recorded_group_key_fn("stem", date=admitted.date)

    # Every route below trains on every admitted sample, so the run's sizes resolve once here.
    sizes = run_sizes(task, data_cfg, admitted.every_sample(), dataset_source)

    def train_only(group_of: "Callable[[str], str]", group_by: str = "stem"):
        samples = admitted.samples({m: "train" for m in members}, group_of)
        return _sample_loaders(task, data_cfg, samples, transforms, seed=None, group_by=group_by)

    members = [record.member for record in admitted.records]
    if not data_cfg.get("auto_val", True):
        logger.info("data.auto_val is off for %s: training on all %d admitted sample(s) with no "
                    "validation.", task, len(members))
        return train_only(each_its_own_group)

    if len(members) < 2:
        # A single source cannot hold out a whole stem, but a tiled detection source the platform
        # builds itself can hold out disjoint pixel blocks of its own tile lattice.
        if (task == "detection" and tiling and tiling.get("enabled", True)
                and dataset_source is None):
            one = admitted.every_sample()[0]
            if spatial_single_source_split(one, admitted.scope, tiling, split_cfg, sizes):
                return _sample_loaders(task, data_cfg, [one], transforms, seed=None,
                                       group_by="stem")
        logger.warning("Auto train/val split for %s: %d admitted source(s) leave nothing to hold "
                       "out; training without validation.", task, len(members))
        return train_only(each_its_own_group)

    group_by = split_cfg.get("group_by", DEFAULT_GROUP_BY)
    group_key_map = split_cfg.get("group_key_map")
    # Deliberately outside any handler: a malformed grouping policy is a caller-config error.
    # Through recorded_group_key_fn, so this draw spells a group key the way draw_splits does.
    group_key_fn = recorded_group_key_fn(
        group_by, date=admitted.date, stems=members, group_key_map=group_key_map)
    resolved_group_by = recorded_group_by(group_by, group_key_map)

    try:
        val_ratio = float(split_cfg.get("val_ratio", DEFAULT_VAL_RATIO))
        seed = split_seed(split_cfg)
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
                resolved_group_by = "stem"
                group_key_fn = retry_key_fn
        if not val_members or not train_members:
            logger.warning(
                "Auto train/val split for %s: no grouping policy could populate both sides; "
                "training without validation.", task,
            )
            return train_only(group_key_fn, resolved_group_by)

        assignment = {s: "train" for s in train_members}
        assignment.update({s: "val" for s in val_members})
        samples = admitted.samples(assignment, group_key_fn)
        by_side = {side: [s for s in samples if s.side == side] for side in ("train", "val")}
    except UnreadableLabelDocument:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto train/val split for %s failed (%s); training without validation.",
                       task, exc)
        return train_only(group_key_fn, resolved_group_by)
    logger.info("Auto train/val split for %s: %d train / %d val samples.",
                task, len(by_side["train"]), len(by_side["val"]))
    return _sample_loaders(task, data_cfg, samples, transforms, seed=seed,
                           group_by=resolved_group_by)


def auto_train_val(task: str, data_cfg: dict, transforms, *,
                   contradicted_out: set[str] | None = None,
                   counts_out: dict[str, int] | None = None):
    """Build ``(train_ds, val_ds, partition)`` for a run, deriving a leakage-free val split, and
    resolve ``data_cfg`` in place: its ``scope``, sizes, and a within-image run's
    ``split.spatial_manifest``.

    ``partition`` is :func:`_partition_record`'s; a within-image spatial split's holds its one
    sample, its regions being the manifest's. ``contradicted_out`` and ``counts_out`` receive a
    drawn run's admission's stale confirmed negatives and counts.

    Two routes, and a run of any task takes one of them:
      1. ``data.split.selection_dir`` set -> train on the selection's own ``train`` and ``val``
        samples instead of drawing a split, each sample reading the source and the ground truth the
        draw recorded for it, under the selection's own scope. Checked
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
        bind_issues = _selection_dependent_issues(selection, selection_dir, data_cfg)
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
        data_cfg["scope"] = asdict(selection.scope)
        run_sizes(task, data_cfg, train_samples + val_samples,
                  data_cfg.get(DATASET_SOURCE_KEY) or None)
        # The selection's own named policy, not "explicit_map": the per-stem map the partition
        # records covers this run's members, and a stem outside it is what a policy name answers.
        return _sample_loaders(
            task, data_cfg, bound, transforms, seed=seed, group_by=selection.group_by,
            selection={"selection_dir": selection_dir,
                       "selection_sha256": selection_digest(selection), "redraw": redraw})

    # 2. Otherwise the producer names this run's membership off the ground truth its config points
    # at, and every branch of the resolution order below builds from the samples it made.
    return _drawn_split(task, data_cfg, tiling=tiling, transforms=transforms,
                        dataset_source=data_cfg.get(DATASET_SOURCE_KEY) or None,
                        contradicted_out=contradicted_out, counts_out=counts_out)


class ResolvedRun(NamedTuple):
    """A run resolved once, at launch: the ``record`` its ``run.json`` holds (its resolved
    ``data`` section, its ``partition`` and its ``objective``) and the datasets built at it."""

    record: dict
    train_ds: Any
    val_ds: Any


def resolve_run(config: dict, *, objective: dict | None = None,
                contradicted_out: set[str] | None = None,
                counts_out: dict[str, int] | None = None) -> ResolvedRun:
    """Resolve ``config`` once: a copy of its data section through :func:`auto_train_val`, the
    geometry its train dataset serves stamped on it
    (:func:`~tcip_mcp.pipelines.training.generic_trainer.stamp_effective_data_geometry`), and its
    objective (:func:`~tcip_mcp.pipelines.training.generic_trainer.resolve_objective` for a run
    with or without a val side), or ``objective`` as given, a sweep's own for its trials.
    ``contradicted_out`` and ``counts_out`` are :func:`auto_train_val`'s. Every refusal of the
    resolution raises."""
    from tcip_mcp.pipelines.model_build import run_task
    from tcip_mcp.pipelines.training.generic_trainer import (
        resolve_objective, run_transforms, stamp_effective_data_geometry,
    )

    data = copy.deepcopy(config.get("data") or {})
    train_ds, val_ds, partition = auto_train_val(
        run_task(config), data, run_transforms(config),
        contradicted_out=contradicted_out, counts_out=counts_out)
    stamp_effective_data_geometry(data, train_ds)
    if objective is None:
        objective = resolve_objective(config, has_val_loader=val_ds is not None)
    return ResolvedRun({"data": data, "partition": partition, "objective": objective},
                       train_ds, val_ds)


def recorded_datasets(task: str, data: dict, partition: dict, transforms) -> tuple[Any, Any]:
    """``(train_ds, val_ds)`` built from what a run resolved, resolving nothing again:
    ``partition``'s train and val samples under ``data``'s recorded class space, sizes, tiling
    and dataset source, or for a within-image spatial split (``data.split.spatial_manifest``) its
    one sample's recorded train and val regions. ``val_ds`` is ``None`` for a run whose partition
    holds no val side."""
    from tcip_mcp.pipelines.data.datasets import build_dataset, stated_sizes
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import DATASET_SOURCE_KEY

    samples = partition_samples(partition)
    scope, sizes = ClassScope.of(data), stated_sizes(data)
    spatial = data["split"].get("spatial_manifest")
    if spatial is not None:
        views = _spatial_views(samples, scope, sizes, _tile_options(data["tiling"]),
                               {"train": spatial["train_region"], "val": spatial["val_region"]},
                               transforms=transforms)
        return views["train"], views["val"]
    build_kwargs: dict[str, Any] = {
        "tiling": data.get("tiling"), "dataset_source": data.get(DATASET_SOURCE_KEY) or None,
        "scope": scope, "sizes": sizes,
    }
    train = [s for s in samples if s.side == "train"]
    val = [s for s in samples if s.side == "val"]
    return (build_dataset(task, samples=train, transforms=transforms, **build_kwargs),
            build_dataset(task, samples=val, transforms=None, **build_kwargs) if val else None)

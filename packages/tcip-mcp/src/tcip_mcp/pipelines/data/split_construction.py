"""Constructing training splits from a data config, beside ``splits.py``.

Resolves a run once at launch (:func:`resolve_run`): its data section, the partition its record
holds (a bound selection, an auto group-aware draw, or a single-source spatial-strip split) and
its objective, and builds ``(train_ds, val_ds)`` from that record (:func:`recorded_datasets`).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from tcip_mcp.pipelines.data.selection import ClassScope, Sample, Selection

logger = logging.getLogger(__name__)

_SELECTION_CONFLICT_KEYS = (
    "group_by", "group_key_map", "val_ratio", "seed", "stratify_foreground",
    "test_ratio", "reserve_calibration_fraction",
)


def data_dir_issues(data_cfg: dict) -> list[str]:
    """Every objection to the data locations this run's own producer reads: ``data.images_dir``
    or ``data.labels_dir`` missing, or naming a path that does not exist, whatever the task and
    whatever builds its loaders. Empty for a config bound to a selection.
    """
    split_cfg = data_cfg.get("split")
    if isinstance(split_cfg, dict) and split_cfg.get("selection_dir"):
        return []
    issues: list[str] = []
    for name in ("images_dir", "labels_dir"):
        path = data_cfg.get(name)
        if not path:
            issues.append(f"Missing 'data.{name}'")
        elif not Path(path).exists():
            # Named without claiming a shape: what ground truth is there is the producer's own
            # read, and a config pointing at nothing is the only fact this check has.
            issues.append(f"Not found: data.{name} = '{path}'")
    return issues


def selection_compatibility(data_cfg: dict, selection: "Selection | None",
                            selection_dir: str) -> list[str]:
    """Every objection binding a run's data section ``data_cfg`` to ``selection`` (read at
    ``selection_dir``; ``None`` when it would not read) raises: a drawn split's own key under
    ``data.split`` beside the selection (``seed`` admitted only beside
    ``redraw_within_selection``), that flag with no seed, and, with a selection in hand, a stated
    ``data.scope`` (the selection records its own class space) and an empty train or val side.
    Whether the selected loader can read the ground truth these samples name is that loader's own
    refusal.
    """
    split_cfg_raw = data_cfg.get("split")
    split_cfg: dict = split_cfg_raw if isinstance(split_cfg_raw, dict) else {}
    redraw = bool(split_cfg.get("redraw_within_selection"))
    conflicts = sorted(k for k in _SELECTION_CONFLICT_KEYS
                       if split_cfg.get(k) is not None and not (redraw and k == "seed"))
    issues: list[str] = []
    if conflicts:
        issues.append(
            f"data.split.selection_dir conflicts with {conflicts}: a recorded partition and "
            "a drawn split's own parameters/source cannot both govern one run."
        )
    if redraw and split_cfg.get("seed") is None:
        issues.append(
            "data.split.redraw_within_selection=true requires data.split.seed: the seed the "
            "redraw draws train and val at."
        )
    if selection is None:
        return issues
    if "scope" in data_cfg:
        issues.append(
            f"data.scope is stated beside data.split.selection_dir, whose selection records its "
            f"own class space ({selection.scope}); a second one would not be the one the run "
            "trains in. Drop data.scope."
        )
    counts = selection.counts()
    if not counts["train"] or not counts["val"]:
        issues.append(
            f"the selection at {selection_dir} leaves an empty side (train={counts['train']}, "
            f"val={counts['val']}); a run needs both."
        )
    return issues


class Membership(NamedTuple):
    """Every member the places of one draw admit, keyed by its identity (``<date>/<stem>``, the
    bare stem for a dateless place), each a sample on the ``train`` side under its group key and
    carrying its ground truth's digest; the class space they were admitted in, the recorded
    grouping policy and the admissions' summed tallies."""

    samples: dict[str, "Sample"]
    scope: "ClassScope"
    group_by: str
    tallies: dict[str, int]


def admitted_membership(
    places: "Sequence[tuple[str, Path | str, str]]", *, scope: "ClassScope", group_by: str,
    group_key_map: "Mapping[str, str] | None", contradicted_out: set[str] | None = None,
) -> Membership:
    """Admit each place ``(name, images_dir, ground_truth)``
    (:func:`~tcip_mcp.pipelines.data.label_queries.admit`, under ``scope``, stated against that
    place when it carries no class-id map) and group every admitted member
    (:func:`~tcip_mcp.pipelines.data.splits.resolve_group_key_fn` over the identities).

    Raises ``ValueError`` when nothing is admitted (the admission's own reason, naming every
    place searched and the tallies) and for a grouping policy that does not cover the members;
    the admissions' own refusals propagate.
    """
    from tcip_mcp.pipelines.data.label_queries import admit, require_admitted, stated_scope
    from tcip_mcp.pipelines.data.selection import ground_truth_digests
    from tcip_mcp.pipelines.data.splits import (
        member_identity, recorded_group_by, resolve_group_key_fn,
    )

    admissions = []
    tallies: dict[str, int] = {}
    for _name, images_dir, ground_truth in places:
        admitted = admit(images_dir, ground_truth, contradicted_out=contradicted_out,
                         scope=scope if scope.id_map is not None
                         else stated_scope(ground_truth, scope.subject, scope.attribute))
        admissions.append(admitted)
        for key, value in admitted.tallies.items():
            tallies[key] = tallies.get(key, 0) + value
    identities = {(where, record.member): member_identity(admitted.date, record.member)
                  for where, admitted in enumerate(admissions) for record in admitted.records}
    if not identities:
        searched = ", ".join(f"{name} -> {images_dir}" for name, images_dir, _gt in places)
        try:
            require_admitted(admissions[0])
        except ValueError as exc:
            raise ValueError(f"{exc} Searched {searched} ({tallies}).") from exc
    group_of = resolve_group_key_fn(group_by, sorted(identities.values()),
                                    group_key_map=dict(group_key_map) if group_key_map else None)
    digest_of = ground_truth_digests(record.ground_truth for admitted in admissions
                                     for record in admitted.records)
    samples: dict[str, Sample] = {}
    for where, admitted in enumerate(admissions):
        built = admitted.samples(
            {record.member: "train" for record in admitted.records},
            {r.member: group_of(identities[where, r.member]) for r in admitted.records}.__getitem__,
            digests={r.member: digest_of[r.ground_truth] for r in admitted.records})
        samples.update((identities[where, sample.member], sample) for sample in built)
    return Membership(samples, admissions[0].scope, recorded_group_by(group_by, group_key_map),
                      tallies)


def run_membership(data_cfg: "Mapping[str, Any]", *,
                   contradicted_out: set[str] | None = None) -> Membership:
    """:func:`admitted_membership` over a run's data section: its one place (``images_dir`` and
    ``labels_dir``), under the ``scope`` it states (the empty one when it states none), grouped by
    its ``split`` section's policy."""
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.data.splits import DEFAULT_GROUP_BY

    split_cfg = data_cfg.get("split") or {}
    labels_dir = data_cfg["labels_dir"]
    return admitted_membership(
        [(str(labels_dir), data_cfg["images_dir"], labels_dir)],
        scope=ClassScope.of(data_cfg) if "scope" in data_cfg else ClassScope(),
        group_by=split_cfg.get("group_by", DEFAULT_GROUP_BY),
        group_key_map=split_cfg.get("group_key_map"), contradicted_out=contradicted_out)


def check_shares(ratios: "Mapping[str, float]") -> None:
    """Refuse (``ValueError``) a partition request naming a side outside the selection's sides,
    no ``train`` share, a share outside ``(0, 1)``, or shares whose correctly rounded sum is not
    exactly one."""
    import math

    from tcip_mcp.pipelines.data.selection import SIDES

    if (set(ratios) - set(SIDES) or "train" not in ratios
            or not all(0 < share < 1 for share in ratios.values())
            or math.fsum(ratios.values()) != 1.0):
        raise ValueError(f"a draw takes a share in (0, 1) for train and for each other side of "
                         f"{list(SIDES)} it draws, summing to exactly 1; got {dict(ratios)}")


def run_shares(split_cfg: "Mapping[str, Any]") -> dict[str, float]:
    """The train and val shares a run drawing its own split requests: ``val_ratio``
    (:data:`~tcip_mcp.pipelines.data.splits.DEFAULT_VAL_RATIO` unless stated) and the rest."""
    from tcip_mcp.pipelines.data.splits import DEFAULT_VAL_RATIO

    val_ratio = float(split_cfg.get("val_ratio", DEFAULT_VAL_RATIO))
    return {"train": 1.0 - val_ratio, "val": val_ratio}


def draw_sides(
    samples: "Mapping[str, Sample]", scope: "ClassScope", *, ratios: "Mapping[str, float]",
    seed: int, stratify: bool,
) -> tuple[dict[str, "Sample"], dict[str, int]]:
    """The one draw: ``samples``, keyed by member identity, onto every side ``ratios`` names, each
    group (a sample's own ``group``) kept whole
    (:func:`~tcip_mcp.pipelines.data.splits.group_balanced_split`) and, with ``stratify``,
    balanced by each member's foreground count
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts` under ``scope``) at both
    stages: the reference share (``calibration`` plus ``holdout``) is drawn as one side, then cut
    between the two at the same seed.

    Refuses (``ValueError``), before drawing, shares :func:`check_shares` refuses and members
    holding fewer foreground groups than one per side; and a draw that leaves a side empty.
    Returns each member's sample on the side it drew and every member's foreground count.
    """
    import dataclasses

    from tcip_mcp.pipelines.data.label_queries import foreground_counts
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
    from tcip_mcp.pipelines.data.splits import (
        foreground_group_count, group_balanced_split, refuse_insufficient_foreground_groups,
    )

    check_shares(ratios)
    counted = foreground_counts(samples, scope)
    group_of = {key: sample.group for key, sample in samples.items()}.__getitem__
    refuse_insufficient_foreground_groups(
        foreground_group_count(counted, counted, group_of), {side: 1 for side in ratios},
        remedy=("add ground truth for more images: annotate or confirm more of them for this "
                "subject, or write the masks or rows that answer for them."))
    reference = [side for side in REFERENCE_SIDES if side in ratios]
    first = {side: share for side, share in ratios.items() if side not in REFERENCE_SIDES}
    minimums = dict.fromkeys(first, 1)
    if reference:
        first["reference"] = sum(ratios[side] for side in reference)
        minimums["reference"] = len(reference)

    def draw(keys, splits, need) -> dict[str, list[str]]:
        return group_balanced_split(keys, counts=counted, weighted=stratify, group_key_fn=group_of,
                                    splits=splits, seed=seed, min_foreground_groups=need)

    drawn = draw(sorted(samples), first, minimums)
    if reference:
        pool = drawn.pop("reference")
        drawn.update(draw(pool, {side: ratios[side] / first["reference"] for side in reference},
                          dict.fromkeys(reference, 1)))
    starved = {side: len(keys) for side, keys in drawn.items()}
    if not all(starved.values()):
        raise ValueError(f"the draw at seed {seed} starved a side ({starved}): its groups cannot "
                         "populate every side requested.")
    return ({key: dataclasses.replace(samples[key], side=side)
             for side, keys in drawn.items() for key in keys}, counted)


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


def raster_identity(source: str) -> dict:
    """A raster's :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity` over its own
    probed band count, the one a split records its mosaic by and a bucket its raster by. A source
    that will not probe raises."""
    import dataclasses

    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import resolve_source_path
    from tcip_mcp.pipelines.raster_source import content_identity

    resolved = resolve_source_path(source)
    return dataclasses.asdict(content_identity(resolved, probe_channels(resolved)))


def spatial_single_source_split(
    sample: "Sample", scope: "ClassScope", tiling: dict, split_cfg: dict,
    sizes: "Mapping[str, int]", shares: "Mapping[str, float]",
) -> None:
    """Derive the run's requested train/val ``shares`` (:func:`run_shares`) over one detection
    source's own tile lattice, by disjoint pixel strips
    (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`), and record it as
    ``split_cfg["spatial_manifest"]``, which :func:`recorded_datasets` builds the run's views from.

    ``sample`` is the run's own single admitted sample, which every view here is built over
    (:func:`_spatial_views`). ``sizes`` is what this run resolved (:func:`run_sizes`). A test
    region (``split_cfg["test_ratio"]``, 0.1 unless stated) is carved from the train share and
    reserved, with only its geometry and kept-tile count recorded;
    ``split_cfg["reserve_calibration_fraction"]`` (unset or 0 by default) reserves a fourth
    region, ``calibration``, the same way.

    Raises ``ValueError`` for shares :func:`check_shares` refuses, and naming which reason fired
    when the strip layout is infeasible or a side keeps no tile after filtering. The label
    document's frame is read through
    :func:`~tcip_mcp.pipelines.data.splits.label_document_extent`, whose refusals propagate.
    """
    from tcip_mcp.pipelines.data.datasets import TILE_SIZE
    from tcip_mcp.pipelines.data.splits import label_document_extent, spatial_strip_split
    from tcip_mcp.pipelines.execution import DEFAULT_OVERLAP

    check_shares(shares)
    stem = sample.member
    reserve_cal = float(split_cfg.get("reserve_calibration_fraction") or 0.0)
    remedy = ("reduce reserve_calibration_fraction or drop it" if reserve_cal
              else "set data.auto_val=False to train on it without validation")

    width, height = label_document_extent(sample.ground_truth)

    tile_options = _tile_options(tiling)
    # The tiler's own defaults, so the geometry derived here and the views built below resolve
    # to one lattice.
    tile_size = tile_options.get("tile_size", TILE_SIZE)
    overlap = tile_options.get("overlap", DEFAULT_OVERLAP)
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    train_ratio = shares["train"] - test_ratio - reserve_cal
    if reserve_cal:
        split_names: tuple[str, ...] = ("train", "val", "test", "calibration")
        fractions: tuple[float, ...] = (train_ratio, shares["val"], test_ratio, reserve_cal)
    else:
        split_names = ("train", "val", "test")
        fractions = (train_ratio, shares["val"], test_ratio)

    try:
        spatial = spatial_strip_split(
            width, height, tile_size, overlap, fractions=fractions, split_names=split_names,
            buffer=tiling.get("buffer"),
        )
    except ValueError as exc:
        raise ValueError(
            f"the spatial split of {stem!r} (reserve_calibration_fraction={reserve_cal}) is "
            f"infeasible at this mosaic size/tile size ({exc}); {remedy}."
        ) from exc

    # A tile lattice occupying a reserved region (spatial_strip_split's own check) is not proof it
    # carries GT: an all-background region still passes that but skip_empty filters it to 0.
    views = _spatial_views([sample], scope, sizes, tile_options, {
        side: spatial.regions[side]
        for side in ("train", "val", *(("test", "calibration") if reserve_cal else ()))
    }, transforms=None)
    train_ds, val_ds = views["train"], views["val"]
    if any(view.num_samples == 0 for view in views.values()):
        raise ValueError(
            f"the spatial split of {stem!r} (reserve_calibration_fraction={reserve_cal}) left a "
            f"side with zero kept (or zero GT-bearing) tiles after filtering "
            f"(kept_tiles={spatial.kept_tiles}); {remedy}."
        )

    def _identities(ds) -> list[str]:
        # Through the dataset's own member stem: a tile is keyed by the sample it was cut from,
        # and a region identity names the bare stem every consumer of this manifest joins on.
        raw = {spatial.identity_for(ds.sample_of(key).member, box) for key, box in ds.tile_entries}
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
        "raster_content_identity": raster_identity(sample.source),
    }
    logger.info(
        "Spatial train/val split for %r: %d train / %d val tiles (axis=%s, "
        "realized_fractions=%s, realized_discard_fraction=%.3f).",
        stem, train_ds.num_samples, val_ds.num_samples, spatial.axis,
        spatial.realized_fractions, spatial.realized_discard_fraction,
    )


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


def redrawn_selection(selection: "Selection", selection_dir: str, seed: int) -> "Selection":
    """``selection`` with train and val redrawn fresh over its own train-plus-val samples at
    ``seed`` (:func:`draw_sides`, over each sample's recorded group key, at the val share the
    selection already delivered, stratified), the reference sides untouched. The draw's refusals
    raise, naming ``selection_dir``.
    """
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES, with_sides

    pool = {s.location: s for s in selection.trainable()}
    val_share = len(selection.on("val")) / len(pool) if pool else 0.0
    try:
        drawn, _counted = draw_sides(
            pool, selection.scope, ratios={"train": 1.0 - val_share, "val": val_share},
            seed=seed, stratify=True)
    except ValueError as exc:
        raise ValueError(f"redrawing train and val inside the selection at {selection_dir!r}: "
                         f"{exc} Drop data.split.redraw_within_selection and data.split.seed to "
                         "bind the selection's recorded partition instead.") from exc
    assignment = {s.location: s.side for s in selection.samples if s.side in REFERENCE_SIDES}
    assignment.update({location: sample.side for location, sample in drawn.items()})
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
    from tcip_mcp.pipelines.data.selection import ground_truth_digests, sample_document

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


def moved_since_run(samples: Iterable["Sample"], at_run: Mapping[str, str]) -> list[str]:
    """The members among ``samples`` whose ground truth no longer digests to what ``at_run`` (a
    run's resolved partition's ``ground_truth_digests``, keyed by path) recorded, sorted
    (:func:`~tcip_mcp.pipelines.data.selection.moved_ground_truth`)."""
    from tcip_mcp.pipelines.data.selection import moved_ground_truth

    samples = list(samples)
    moved = set(moved_ground_truth({s.ground_truth: at_run[s.ground_truth] for s in samples}))
    return sorted({s.member for s in samples if s.ground_truth in moved})


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
    contradicted_out: set[str] | None, tallies_out: dict[str, int] | None,
):
    """``(train_ds, val_ds, partition)`` for a run that draws its own split over ``data_cfg``'s
    ground truth: its one place's membership (:func:`admitted_membership`, handed
    ``contradicted_out``, its tallies copied into ``tallies_out``), then, with ``auto_val`` on,
    train and val drawn through :func:`draw_sides`, or a single tiled detection source split over
    its own tile lattice (:func:`spatial_single_source_split`). ``auto_val`` off trains on every
    member with no validation. Raises when the validation requested cannot be drawn (naming
    ``auto_val=False``), and with the membership's and the draw's own refusals. Either geometry
    draws the one partition :func:`run_shares` resolves.
    """
    split_cfg = data_cfg.setdefault("split", {})
    membership = run_membership(data_cfg, contradicted_out=contradicted_out)
    if tallies_out is not None:
        tallies_out.update(membership.tallies)
    data_cfg["scope"] = asdict(membership.scope)
    samples = list(membership.samples.values())
    sizes = run_sizes(task, data_cfg, samples, dataset_source)

    if not data_cfg.get("auto_val", True):
        return _sample_loaders(task, data_cfg, samples, transforms, seed=None,
                               group_by=membership.group_by)
    shares = run_shares(split_cfg)
    if len(samples) < 2:
        if not (task == "detection" and tiling and tiling.get("enabled", True)
                and dataset_source is None):
            raise ValueError(
                f"{len(samples)} admitted source(s) leave nothing to hold a validation side out "
                f"of for this {task} run; set data.auto_val=False to train without validation, "
                "or admit more sources.")
        # A tiled detection source the platform builds itself holds out disjoint pixel blocks.
        spatial_single_source_split(samples[0], membership.scope, tiling, split_cfg, sizes,
                                    shares)
        return _sample_loaders(task, data_cfg, samples, transforms, seed=None, group_by="stem")

    seed = split_seed(split_cfg)
    drawn, _counted = draw_sides(
        membership.samples, membership.scope, ratios=shares, seed=seed,
        stratify=split_cfg.get("stratify_foreground", True))
    return _sample_loaders(task, data_cfg, [drawn[key] for key in sorted(drawn)], transforms,
                           seed=seed, group_by=membership.group_by)


def auto_train_val(project: Path, task: str, data_cfg: dict, transforms, *,
                   contradicted_out: set[str] | None = None,
                   tallies_out: dict[str, int] | None = None):
    """Build ``(train_ds, val_ds, partition)`` for a run of ``project``, deriving a leakage-free
    val split, and resolve ``data_cfg`` in place: its ``scope``, sizes, and a within-image run's
    ``split.spatial_manifest``.

    ``partition`` is :func:`_partition_record`'s; a within-image spatial split's holds its one
    sample, its regions being the manifest's. ``contradicted_out`` and ``tallies_out`` receive a
    drawn run's admission's stale confirmed negatives and tallies.

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

    # The one missing-key refusal, so a run and the preflight that offered it name a missing or
    # moved location with the same words. A no-op for a config bound to a selection.
    location_issues = data_dir_issues(data_cfg)
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
        from tcip_mcp.pipelines.data.label_queries import refuse_inadmissible_samples
        from tcip_mcp.pipelines.data.selection import (
            REFERENCE_SIDES, read_selection, selection_digest,
        )

        selection: Selection | None = None
        unread: ValueError | None = None
        try:
            selection = read_selection(selection_dir, project=project)
        except ValueError as exc:
            unread = exc
        # The bind's own refusals, stated once so the preflight that offered this selection and
        # the launch that binds it say the same thing, the config's own even over an unread one.
        bind_issues = selection_compatibility(data_cfg, selection, selection_dir)
        if bind_issues:
            raise ValueError(" ".join(bind_issues)) from unread
        if selection is None:
            raise unread or ValueError(f"no selection read under {selection_dir}")

        # A redraw repartitions the selection's own train-plus-val samples; the reference untouched.
        redraw = bool(split_cfg_raw.get("redraw_within_selection"))
        seed = selection.seed
        if redraw:
            seed = int(split_cfg_raw["seed"])
            selection = redrawn_selection(selection, selection_dir, seed)

        train_samples, val_samples = selection.on("train"), selection.on("val")
        reference_samples = [s for s in selection.samples if s.side in REFERENCE_SIDES]

        # Checked before either loader is built: a selected label that has since emptied with
        # nobody confirming that image negative would otherwise train as background.
        refuse_inadmissible_samples(train_samples + val_samples, selection.scope)

        bound = train_samples + val_samples + reference_samples
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
                       "selection_sha256": selection_digest(selection, project),
                       "redraw": redraw})

    # 2. Otherwise the producer names this run's membership off the ground truth its config points
    # at, and every branch of the resolution order below builds from the samples it made.
    return _drawn_split(task, data_cfg, tiling=tiling, transforms=transforms,
                        dataset_source=data_cfg.get(DATASET_SOURCE_KEY) or None,
                        contradicted_out=contradicted_out, tallies_out=tallies_out)


class ResolvedRun(NamedTuple):
    """A run resolved once, at launch: the ``record`` its ``run.json`` holds (its resolved
    ``data`` section, its ``partition`` and its ``objective``) and the datasets built at it."""

    record: dict
    train_ds: Any
    val_ds: Any


def resolve_run(config: dict, *, project: Path, objective: dict | None = None,
                contradicted_out: set[str] | None = None,
                tallies_out: dict[str, int] | None = None) -> ResolvedRun:
    """Resolve ``config`` once for a run of ``project``: a copy of its data section through
    :func:`auto_train_val`, the geometry its train dataset serves stamped on it
    (:func:`~tcip_mcp.pipelines.training.generic_trainer.stamp_effective_data_geometry`), and its
    objective (:func:`~tcip_mcp.pipelines.training.generic_trainer.resolve_objective` for a run
    with or without a val side), or ``objective`` as given, a sweep's own for its trials.
    ``contradicted_out`` and ``tallies_out`` are :func:`auto_train_val`'s. Every refusal of the
    resolution raises."""
    from tcip_mcp.pipelines.model_build import run_task
    from tcip_mcp.pipelines.training.generic_trainer import (
        resolve_objective, run_transforms, stamp_effective_data_geometry,
    )

    data = copy.deepcopy(config.get("data") or {})
    train_ds, val_ds, partition = auto_train_val(
        project, run_task(config), data, run_transforms(config),
        contradicted_out=contradicted_out, tallies_out=tallies_out)
    stamp_effective_data_geometry(data, train_ds)
    if objective is None:
        objective = resolve_objective(config, project=project, has_val_loader=val_ds is not None)
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

"""Constructing training splits from a data config, beside ``splits.py``.

Resolves a run once at launch (:func:`resolve_run`): its data section, the partition its record
holds (a bound selection, an auto group-aware draw, or a single-source spatial-strip split) and
its objective, and builds ``(train_ds, val_ds)`` from that record (:func:`recorded_datasets`).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from tcip_mcp.pipelines.data.selection import ClassScope, Sample, Selection
    from tcip_mcp.pipelines.image_utils import BandGroupRef
    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import (
        DataSpec, SpatialManifest, SplitSpec, TilingSpec, TrainConfigSchema,
    )

logger = logging.getLogger(__name__)

SPATIAL_SIDE_ORDER = ("train", "val", "holdout", "calibration")
"""The order a within-image split lays its sides out in, which breaks a tie between equal shares
(:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`)."""


def _selection_conflict_keys() -> tuple[str, ...]:
    """The ``data.split`` keys a drawn split states and a bound selection records instead, each
    side's share stated as ``<side>_ratio``."""
    from tcip_mcp.pipelines.data.selection import SIDES

    return ("group_by", "group_key_map", "seed", "stratify_foreground",
            *(f"{side}_ratio" for side in SIDES[1:]))


def data_dir_issues(data: "DataSpec") -> list[str]:
    """Every objection to the data locations an unbound run's producer reads: ``data.images_dir``
    missing or naming a path that does not exist, and a stated ``data.labels_dir`` that is not a
    mask directory or a table (:func:`~tcip_mcp.pipelines.data.label_queries.ground_truth_shape`).
    Empty for a config bound to a selection (:func:`selection_compatibility`)."""
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape

    if data.split.selection_dir:
        return []
    images_dir = data.images_dir
    issues = [] if images_dir else ["Missing 'data.images_dir'"]
    if images_dir and not Path(images_dir).exists():
        issues.append(f"Not found: data.images_dir = '{images_dir}'")
    if data.labels_dir is not None:
        try:
            ground_truth_shape(data.labels_dir)
        except ValueError as exc:
            issues.append(f"data.labels_dir: {exc}")
    return issues


def selection_compatibility(data: "DataSpec", selection: "Selection | None",
                            selection_dir: str) -> list[str]:
    """Every objection binding a run's data block ``data`` to ``selection`` (read at
    ``selection_dir``; ``None`` when it would not read) raises: a drawn split's own key under
    ``data.split`` beside the selection (``seed`` admitted only beside
    ``redraw_within_selection``), that flag with no seed, a stated ``data.labels_dir`` (each
    sample names its own ground truth), and, with a selection in hand, a stated ``data.scope``
    (the selection records its own class space) and an empty train or val side.
    """
    split = data.split
    redraw = split.redraw_within_selection
    conflicts = sorted(k for k in _selection_conflict_keys()
                       if k in split.stated and not (redraw and k == "seed"))
    issues: list[str] = []
    if conflicts:
        issues.append(
            f"data.split.selection_dir conflicts with {conflicts}: a recorded partition and "
            "a drawn split's own parameters/source cannot both govern one run."
        )
    if redraw and split.seed is None:
        issues.append(
            "data.split.redraw_within_selection=true requires data.split.seed: the seed the "
            "redraw draws train and val at."
        )
    if data.labels_dir is not None:
        issues.append(
            "data.labels_dir is stated beside data.split.selection_dir, whose samples each name "
            "their own ground truth; a second place would not be the one the run reads. Drop "
            "data.labels_dir.")
    if selection is None:
        return issues
    if data.scope is not None:
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
    """Every member the places of one draw admit, keyed by its identity (``<capture>/<stem>``),
    each a sample on the ``train`` side under its group key and carrying the digest its admission
    read; the class space they were admitted in, the recorded grouping policy and the admissions'
    summed tallies."""

    samples: dict[str, "Sample"]
    scope: "ClassScope"
    group_by: str
    tallies: dict[str, int]


def admitted_membership(
    places: "Sequence[tuple[str, Path | str, str | None]]", *, scope: "ClassScope", group_by: str,
    group_key_map: "Mapping[str, str] | None",
) -> Membership:
    """Admit each place ``(name, images_dir, ground_truth)``, ``ground_truth`` ``None`` for the
    images' own label documents
    (:func:`~tcip_mcp.pipelines.data.label_queries.admit`, under ``scope``, read from the images'
    registry when it carries no attributes yet,
    :func:`~tcip_mcp.pipelines.data.label_queries.registry_scope`) and group every admitted member
    (:func:`~tcip_mcp.pipelines.data.splits.resolve_group_key_fn` over the identities).

    Raises ``ValueError`` when nothing is admitted (the admission's own reason, naming every
    place searched and the tallies) and for a grouping policy that does not cover the members;
    the admissions' own refusals propagate.
    """
    from tcip_mcp.pipelines.data.label_queries import admit, registry_scope, require_admitted
    from tcip_mcp.pipelines.data.splits import (
        member_identity, recorded_group_by, resolve_group_key_fn,
    )

    admissions = []
    tallies: dict[str, int] = {}
    for _name, images_dir, ground_truth in places:
        admitted = admit(images_dir, ground_truth,
                         scope=scope if scope.attributes is not None
                         else registry_scope(images_dir, scope.subject))
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
    samples: dict[str, Sample] = {}
    for where, admitted in enumerate(admissions):
        built = admitted.samples(
            {record.member: "train" for record in admitted.records},
            {r.member: group_of(identities[where, r.member]) for r in admitted.records}.__getitem__)
        samples.update((identities[where, sample.member], sample) for sample in built)
    return Membership(samples, admissions[0].scope, recorded_group_by(group_by, group_key_map),
                      tallies)


def run_membership(data: "DataSpec") -> Membership:
    """:func:`admitted_membership` over a run's data block naming its locations
    (:func:`data_dir_issues`): its one place (``images_dir`` and the ``labels_dir`` a mask or
    table run names), under the ``scope`` it states (the empty one when it states none), grouped
    by its ``split`` section's policy."""
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.data.splits import DEFAULT_GROUP_BY

    images_dir = cast(str, data.images_dir)
    return admitted_membership(
        [(images_dir, images_dir, data.labels_dir)],
        scope=data.scope if data.scope is not None else ClassScope(),
        group_by=data.split.group_by or DEFAULT_GROUP_BY,
        group_key_map=data.split.group_key_map)


def check_shares(sides: "Mapping[str, float]") -> dict[str, float]:
    """``sides``, the share each side but ``train`` is drawn at, with ``train`` the remainder; a
    side stated at zero is not drawn. Refuses (``ValueError``) a side outside the selection's
    sides or ``train`` itself, a share outside ``[0, 1)``, and shares leaving ``train`` nothing."""
    import math

    from tcip_mcp.pipelines.data.selection import SIDES

    drawn = {side: float(share) for side, share in sides.items() if share != 0}
    remainder = 1.0 - math.fsum(drawn.values())
    if (set(sides) - set(SIDES) or "train" in sides
            or not all(0 < share < 1 for share in drawn.values()) or remainder <= 0):
        raise ValueError(f"a draw takes a share in [0, 1) for each side of {list(SIDES)} but "
                         f"train, which takes the remainder, and the shares must leave it some; "
                         f"got {dict(sides)}")
    return {"train": remainder, **drawn}


def run_shares(split: "SplitSpec", *, spatial: bool) -> dict[str, float]:
    """The shares a run drawing its own split states (:func:`check_shares`, ``train`` the
    remainder): ``val`` at ``val_ratio``, and on a within-image spatial split each other side of
    :data:`SPATIAL_SIDE_ORDER` whose ``<side>_ratio`` is stated. A ``val`` share unstated or zero
    refuses, since the run draws its own validation side."""
    sides = SPATIAL_SIDE_ORDER[1:] if spatial else ("val",)
    stated = {side: getattr(split, f"{side}_ratio") for side in sides}
    shares = check_shares({side: share for side, share in stated.items() if share is not None})
    if "val" not in shares:
        raise ValueError("a run drawing its own validation side states data.split.val_ratio "
                         "above zero; set data.auto_val=False to train without validation.")
    return shares


def draw_sides(
    samples: "Mapping[str, Sample]", scope: "ClassScope", *, ratios: "Mapping[str, float]",
    seed: int, stratify: bool,
) -> tuple[dict[str, "Sample"], dict[str, int]]:
    """The one draw: ``samples``, keyed by member identity, onto every side ``ratios`` names
    (:func:`check_shares`'s answer, ``train`` the remainder), each group (a sample's own
    ``group``) kept whole
    (:func:`~tcip_mcp.pipelines.data.splits.group_balanced_split`) and, with ``stratify``,
    balanced by each member's foreground count
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts` under ``scope``) at both
    stages: the reference share (``calibration`` plus ``holdout``) is drawn as one side, then cut
    between the two at the same seed.

    Refuses (``ValueError``), before drawing, members holding fewer foreground groups than one
    per side; and a draw that leaves a side empty, naming it. Returns each member's sample on the
    side it drew and every member's foreground count.
    """
    import dataclasses

    from tcip_mcp.pipelines.data.label_queries import foreground_counts
    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES
    from tcip_mcp.pipelines.data.splits import (
        foreground_group_count, group_balanced_split, refuse_insufficient_foreground_groups,
    )

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
    starved = sorted(side for side, keys in drawn.items() if not keys)
    if starved:
        raise ValueError(f"the draw at seed {seed} leaves {starved} empty: its groups cannot "
                         "populate every side requested. State a share of zero for a side not "
                         "to draw it, or add ground truth.")
    return ({key: dataclasses.replace(samples[key], side=side)
             for side, keys in drawn.items() for key in keys}, counted)


def split_seed(split: "SplitSpec") -> int:
    """The seed a draw over ``split`` states; one stating none refuses (``ValueError``)."""
    if split.seed is None:
        raise ValueError("this run draws its own split and states no seed: the partition a draw "
                         "produces depends on it, so state data.split.seed.")
    return split.seed


def dataset_identity(data: "DataSpec", samples: "Iterable[Sample]" = ()
                     ) -> tuple[str | None, str | None]:
    """``(dataset_id, dataset_fingerprint)`` for the run's dataset: the fingerprint recomputed
    over ``samples``, the run's admitted partition (:func:`dataset_fingerprint`), the id from the
    dataset's identity record, ``None`` for a dataset never registered. ``(None, None)`` for a run
    whose images sit under no dataset root. A fingerprint that cannot be read and an identity
    record that does not decode raise.
    """
    images_dir = data.images_dir
    if not images_dir:
        return None, None

    from tcip_mcp.dataset_layout import dataset_root_of, read_dataset_identity
    from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint

    root = dataset_root_of(images_dir)
    if root is None:
        return None, None
    record = read_dataset_identity(root)
    return (record["id"] if record is not None else None), dataset_fingerprint(root, samples)


def raster_identity(image: "Path | BandGroupRef") -> dict:
    """The logical ``image``'s :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity`
    over its own probed band count, the one a split records its mosaic by. A source that will not
    probe raises."""
    import dataclasses

    from tcip_mcp.pipelines.raster_source import SourceHeader, content_identity

    header = SourceHeader(image)
    with header.open(header.channels) as src:
        return dataclasses.asdict(content_identity(src))


def spatial_single_source_split(
    sample: "Sample", scope: "ClassScope", tiling: TilingSpec,
    sizes: "Mapping[str, int]", shares: "Mapping[str, float]",
) -> SpatialManifest:
    """Derive the run's requested ``shares`` (:func:`run_shares`: ``train``, ``val``, ``holdout``
    and, when stated, ``calibration``) over one detection source's own tile lattice,
    by disjoint pixel strips (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`), and
    return it as the :class:`~tcip_mcp.pipelines.schemas.SpatialManifest` a run's partition
    records; the reserved regions record only their geometry and kept-tile count.

    ``sample`` is the run's own single admitted sample, which every view here is built over
    (:func:`_spatial_views`). ``sizes`` is what this run resolved (:func:`run_sizes`).

    Raises ``ValueError`` naming which reason fired when the strip layout is infeasible or a side
    keeps no tile after filtering. The label document's frame is read through
    :func:`~tcip_mcp.pipelines.data.splits.label_document_extent`, whose refusals propagate.
    """
    from tcip_mcp.pipelines.data.datasets import TILE_SIZE
    from tcip_mcp.pipelines.data.label_queries import acquired, resolved
    from tcip_mcp.pipelines.data.splits import label_document_extent, spatial_strip_split
    from tcip_mcp.pipelines.execution import DEFAULT_OVERLAP
    from tcip_mcp.pipelines.schemas import SpatialManifest

    stem = sample.member
    remedy = ("reduce the reserved shares, or set data.auto_val=False to train on it without "
              "validation")

    (sample,) = resolved(acquired([sample]))
    width, height = label_document_extent(sample.read, f"{stem}'s label document")

    tile_options = _tile_options(tiling)
    # The tiler's own defaults, so the geometry derived here and the views built below resolve
    # to one lattice.
    tile_size = tile_options.get("tile_size", TILE_SIZE)
    overlap = tile_options.get("overlap", DEFAULT_OVERLAP)
    split_names = tuple(side for side in SPATIAL_SIDE_ORDER if side in shares)

    try:
        spatial = spatial_strip_split(
            width, height, tile_size, overlap, fractions=tuple(shares[s] for s in split_names),
            split_names=split_names, buffer=tiling.buffer,
        )
    except ValueError as exc:
        raise ValueError(
            f"the spatial split of {stem!r} at shares {dict(shares)} is infeasible at this "
            f"mosaic size/tile size ({exc}); {remedy}."
        ) from exc

    # A tile lattice occupying a reserved region (spatial_strip_split's own check) is not proof it
    # carries GT: an all-background region still passes that but skip_empty filters it to 0.
    views = _spatial_views([sample], scope, sizes, tile_options,
                           {side: spatial.regions[side] for side in split_names}, transforms=None)
    train_ds, val_ds = views["train"], views["val"]
    if any(view.num_samples == 0 for view in views.values()):
        raise ValueError(
            f"the spatial split of {stem!r} at shares {dict(shares)} left a side with zero kept "
            f"(or zero GT-bearing) tiles after filtering (kept_tiles={spatial.kept_tiles}); "
            f"{remedy}."
        )

    def _identities(ds) -> list[str]:
        # Through the dataset's own member stem: a tile is keyed by the sample it was cut from,
        # and a region identity names the bare stem every consumer of this manifest joins on.
        raw = {spatial.identity_for(ds.sample_of(key).member, box) for key, box in ds.tile_entries}
        return sorted(name for name in raw if name is not None)

    logger.info(
        "Spatial train/val split for %r: %d train / %d val tiles (axis=%s, "
        "realized_fractions=%s, realized_discard_fraction=%.3f).",
        stem, train_ds.num_samples, val_ds.num_samples, spatial.axis,
        spatial.realized_fractions, spatial.realized_discard_fraction,
    )
    return SpatialManifest(
        stem=stem,
        train_identities=_identities(train_ds), val_identities=_identities(val_ds),
        train_region=spatial.regions.get("train", []),
        val_region=spatial.regions.get("val", []),
        holdout_region=spatial.regions.get("holdout", []),
        calibration_region=spatial.regions.get("calibration", []),
        kept_holdout_tiles=spatial.kept_tiles.get("holdout", 0),
        kept_calibration_tiles=spatial.kept_tiles.get("calibration", 0),
        width=spatial.width, height=spatial.height, tile_size=spatial.tile_size,
        overlap=spatial.overlap, axis=spatial.axis, buffer=spatial.buffer,
        requested_fractions=dict(zip(spatial.split_names, spatial.requested_fractions)),
        realized_fractions=spatial.realized_fractions,
        realized_discard_fraction=spatial.realized_discard_fraction,
        kept_train_tiles=spatial.kept_tiles.get("train", 0),
        kept_val_tiles=spatial.kept_tiles.get("val", 0),
        tiles_dropped_past_extent=spatial.tiles_dropped_past_extent,
        tiles_dropped_outside_regions=spatial.tiles_dropped_outside_regions,
        raster_content_identity=raster_identity(sample.image),
    )


def _tile_options(tiling: TilingSpec | None) -> dict:
    """``tiling``'s tile options (``TilingSpec.tiler_options``) without ``keep_regions``, which a
    spatial split sets per side."""
    from tcip_mcp.pipelines.data.datasets import stated_tiling

    stated = stated_tiling(tiling)
    return {} if stated is None else stated.tiler_options(frozenset({"keep_regions"}))


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
    raise, naming ``selection_dir``. Each redrawn sample keeps the read it carries
    (:func:`~tcip_mcp.pipelines.data.label_queries.acquired`).
    """
    from tcip_mcp.pipelines.data.label_queries import acquired

    pool = {s.location: s for s in acquired(selection.trainable())}
    val_share = len(selection.on("val")) / len(pool) if pool else 0.0
    try:
        drawn, _counted = draw_sides(
            pool, selection.scope, ratios=check_shares({"val": val_share}),
            seed=seed, stratify=True)
    except ValueError as exc:
        raise ValueError(f"redrawing train and val inside the selection at {selection_dir!r}: "
                         f"{exc} Drop data.split.redraw_within_selection and data.split.seed to "
                         "bind the selection's recorded partition instead.") from exc
    from tcip_mcp.pipelines.data.selection import refuse_crossing_sides

    redrawn = tuple(drawn.get(s.location, s) for s in selection.samples)
    refuse_crossing_sides(redrawn)
    return replace(selection, samples=redrawn)


def _partition_record(samples: "Sequence[Sample]", *, seed: int | None, group_by: str,
                      selection: dict | None, spatial: SpatialManifest | None) -> dict:
    """The partition a run's resolved record holds: the draw's ``seed`` (``None`` for a route
    that drew nothing) and resolved ``group_by``, every sample the run bound on the side it
    landed on in its one recorded shape
    (:func:`~tcip_mcp.pipelines.data.selection.sample_document`), with the ground-truth digest it
    carries, ``selection``, the selection a bound run bound (its directory, its digest and
    whether the run redrew inside it), ``None`` for a run that drew its own, and ``spatial``, a
    within-image split's regions (:func:`spatial_single_source_split`), ``None`` otherwise."""
    from tcip_mcp.pipelines.data.selection import sample_document

    return {
        "seed": seed,
        "group_by": group_by,
        "samples": [sample_document(s) for s in samples],
        "selection": selection,
        "spatial": None if spatial is None else spatial.model_dump(mode="json"),
    }


def partition_samples(partition: "Mapping[str, Any]") -> list["Sample"]:
    """A resolved partition's samples read back
    (:func:`~tcip_mcp.pipelines.data.selection.read_sample`)."""
    from tcip_mcp.pipelines.data.selection import read_sample

    return [read_sample(raw, position, "a run's resolved partition")
            for position, raw in enumerate(partition["samples"])]


def partition_spatial(partition: "Mapping[str, Any]") -> SpatialManifest | None:
    """A resolved partition's within-image split read back, validated; ``None`` for a partition
    that drew none."""
    from tcip_mcp.pipelines.schemas import SpatialManifest

    spatial = partition["spatial"]
    return None if spatial is None else SpatialManifest.model_validate(spatial)


def run_sizes(task: str, data: "DataSpec", scope: "ClassScope",
              samples: "Sequence[Sample]") -> "DataSpec":
    """``data`` with what a run resolved over ``samples`` recorded on it: the class space
    ``scope`` it was admitted under, and its sizes
    (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes` over the ones ``data`` states), each
    by its own name."""
    from tcip_mcp.pipelines.data.datasets import SIZE_NAMES, resolve_sizes, stated_sizes

    sizes = resolve_sizes(task, stated_sizes(data), samples, data.dataset_source)
    return data.model_copy(update={"scope": scope,
                                   **{name: sizes.get(name) for name in SIZE_NAMES}})


def _sample_loaders(task: str, data: "DataSpec", recorded: "Sequence[Sample]", transforms,
                    layout: "SourceLayout", *, seed: int | None, group_by: str,
                    selection: dict | None = None, spatial: SpatialManifest | None = None):
    """``(train_ds, val_ds, partition, data)``: the partition (:func:`_partition_record` over
    ``recorded`` at ``seed``, ``group_by``, the ``selection`` a bound run bound and a
    within-image run's ``spatial`` split) and :func:`recorded_datasets` over it, the resolved
    block ``data`` and the run's ``layout``. ``recorded`` is the membership the partition holds:
    the two sides for a drawn run, and the selection's held-out samples besides for a bound
    one."""
    partition = _partition_record(recorded, seed=seed, group_by=group_by, selection=selection,
                                  spatial=spatial)
    return (*recorded_datasets(task, data, recorded, spatial, transforms, layout), partition,
            data)


def _drawn_split(task: str, data: "DataSpec", layout: "SourceLayout", *, transforms,
                 tallies_out: dict[str, int] | None):
    """``(train_ds, val_ds, partition, resolved)`` for a run that draws its own split over
    ``data``'s ground truth: its one place's membership (:func:`admitted_membership`, its tallies
    copied into ``tallies_out``), then, with ``auto_val`` on, train and val drawn through
    :func:`draw_sides`, or a single tiled detection source split over its own tile lattice
    (:func:`spatial_single_source_split`). ``auto_val`` off trains on every member with no
    validation. ``resolved`` is ``data`` with its class space and sizes recorded, and a
    within-image split is the partition's. Raises when the validation requested cannot be drawn
    (naming ``auto_val=False``), and with the membership's and the draw's own refusals. Either
    geometry draws the one partition :func:`run_shares` resolves.
    """
    split = data.split
    membership = run_membership(data)
    if tallies_out is not None:
        tallies_out.update(membership.tallies)
    samples = list(membership.samples.values())
    resolved = run_sizes(task, data, membership.scope, samples)

    if not data.auto_val:
        return _sample_loaders(task, resolved, samples, transforms, layout, seed=None,
                               group_by=membership.group_by)
    if len(samples) < 2:
        from tcip_mcp.pipelines.data.datasets import run_tiling, stated_sizes

        tiler = run_tiling(task, data.tiling)
        if tiler is None or data.dataset_source is not None:
            raise ValueError(
                f"{len(samples)} admitted source(s) leave nothing to hold a validation side out "
                f"of for this {task} run; set data.auto_val=False to train without validation, "
                "or admit more sources.")
        # A tiled detection source the platform builds itself holds out disjoint pixel blocks.
        manifest = spatial_single_source_split(
            samples[0], membership.scope, tiler, stated_sizes(resolved),
            run_shares(split, spatial=True))
        return _sample_loaders(task, resolved, samples, transforms, layout, seed=None,
                               group_by="stem", spatial=manifest)

    seed = split_seed(split)
    drawn, _counted = draw_sides(
        membership.samples, membership.scope, ratios=run_shares(split, spatial=False),
        seed=seed, stratify=split.stratify_foreground)
    return _sample_loaders(task, resolved, [drawn[key] for key in sorted(drawn)], transforms,
                           layout, seed=seed, group_by=membership.group_by)


def auto_train_val(project: Path, task: str, data: "DataSpec", transforms, layout: "SourceLayout",
                   *, tallies_out: dict[str, int] | None = None):
    """Build ``(train_ds, val_ds, partition, resolved)`` for a run of ``project`` over its data
    block ``data``, deriving a leakage-free val split, a bespoke dataset builder imported from
    the run's ``layout``; ``resolved`` is ``data`` with its ``scope`` and sizes recorded on it.

    ``partition`` is :func:`_partition_record`'s; a within-image spatial split's holds its one
    sample and its ``spatial`` regions. ``tallies_out`` receives a drawn run's admission's
    tallies.

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
      2. Otherwise -> :func:`_drawn_split`, which admits once through the producer over the
        images' own label documents, or the mask directory or table ``data.labels_dir`` names, and
        builds every loader from the resulting samples.
        A bespoke ``data.dataset_source`` builder takes this route too and is handed those samples.

    """
    # The one missing-key refusal, so a run and the preflight that offered it name a missing or
    # moved location with the same words. A no-op for a config bound to a selection.
    location_issues = data_dir_issues(data)
    if location_issues:
        raise ValueError(
            f"{'; '.join(location_issues)}: a run reads its own samples out of the places its "
            "config names, so there is nothing here to admit."
        )

    selection_dir = data.split.selection_dir

    # 1. A named selection is an explicit partition auto_val does not govern; every refusal
    # here, and any build failure while binding to it, raises rather than degrading.
    if selection_dir:
        from tcip_mcp.pipelines.data.label_queries import readmitted_samples
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
        bind_issues = selection_compatibility(data, selection, selection_dir)
        if bind_issues:
            raise ValueError(" ".join(bind_issues)) from unread
        if selection is None:
            raise unread or ValueError(f"no selection read under {selection_dir}")

        # Re-admitted before any draw or loader, whose reads both consume: a selected label emptied
        # with nobody confirming that image negative would otherwise train as background.
        readmitted = {s.location: s for s in readmitted_samples(selection.trainable(),
                                                                selection.scope)}
        selection = replace(selection, samples=tuple(readmitted.get(s.location, s)
                                                     for s in selection.samples))
        # A redraw repartitions the selection's own train-plus-val samples; the reference untouched.
        redraw = data.split.redraw_within_selection
        seed = selection.seed
        if redraw:
            seed = split_seed(data.split)
            selection = redrawn_selection(selection, selection_dir, seed)

        reference_samples = [s for s in selection.samples if s.side in REFERENCE_SIDES]
        trained = selection.on("train") + selection.on("val")
        train_samples = [s for s in trained if s.side == "train"]
        val_samples = [s for s in trained if s.side == "val"]

        bound = trained + reference_samples
        # The selection's own scope and exact map become this run's, so the checkpoint records
        # the vocabulary it trained in rather than one rediscovered from a live registry.
        # The selection's own named policy, not "explicit_map": the per-stem map the partition
        # records covers this run's members, and a stem outside it is what a policy name answers.
        return _sample_loaders(
            task, run_sizes(task, data, selection.scope, train_samples + val_samples), bound,
            transforms, layout, seed=seed, group_by=selection.group_by,
            selection={"selection_dir": selection_dir,
                       "selection_sha256": selection_digest(selection, project),
                       "redraw": redraw})

    # 2. Otherwise the producer names this run's membership off the ground truth its config points
    # at, and every branch of the resolution order below builds from the samples it made.
    return _drawn_split(task, data, layout, transforms=transforms, tallies_out=tallies_out)


class ResolvedRun(NamedTuple):
    """A run resolved once, at launch: the datasets built at it, ``spec``, the validated launch
    config it was resolved from, and ``data``, ``partition`` and ``objective``, what it
    resolved."""

    train_ds: Any
    val_ds: Any
    spec: TrainConfigSchema
    data: "DataSpec"
    partition: dict
    objective: dict

    @property
    def spatial(self) -> SpatialManifest | None:
        """The within-image split ``partition`` records (:func:`partition_spatial`)."""
        return partition_spatial(self.partition)

    @property
    def record(self) -> dict:
        """What the run's ``run.json`` holds of this resolution: ``data`` as it records itself,
        ``partition`` and ``objective``."""
        return {"data": self.data.record(), "partition": self.partition,
                "objective": self.objective}


def resolve_run(spec: TrainConfigSchema, layout: "SourceLayout", *, project: Path,
                objective: dict | None = None,
                tallies_out: dict[str, int] | None = None) -> ResolvedRun:
    """Resolve the validated config ``spec`` (``schemas.train_config``) once for a run of
    ``project`` whose declared files ``layout`` lays out (``model_build.SourceLayout``): its data
    block through :func:`auto_train_val`, the geometry its train dataset
    serves recorded on it
    (:func:`~tcip_mcp.pipelines.training.generic_trainer.effective_data_geometry`), and its
    objective (:func:`~tcip_mcp.pipelines.training.generic_trainer.resolve_objective` for a run
    with or without a val side), or ``objective`` as given, a sweep's own for its trials.
    ``tallies_out`` is :func:`auto_train_val`'s. Every refusal of the resolution raises."""
    from tcip_mcp.pipelines.training.generic_trainer import (
        effective_data_geometry, resolve_objective, run_transforms,
    )

    task = spec.model_source.task
    train_ds, val_ds, partition, data = auto_train_val(
        project, task, spec.data, run_transforms(spec), layout, tallies_out=tallies_out)
    data = effective_data_geometry(data, train_ds)
    if objective is None:
        objective = resolve_objective(spec, project=project, has_val_loader=val_ds is not None)
    return ResolvedRun(train_ds, val_ds, spec, data, partition, objective)


def recorded_datasets(task: str, data: "DataSpec", samples: "Sequence[Sample]",
                      spatial: SpatialManifest | None, transforms,
                      layout: "SourceLayout") -> tuple[Any, Any]:
    """``(train_ds, val_ds)`` built from what a run resolved, resolving nothing again: the train
    and val ``samples`` of its partition under the resolved block ``data``'s class space, sizes,
    tiling and dataset source (imported from the run's ``layout``), or for a within-image split
    ``spatial``
    (:func:`partition_spatial`) its one sample's recorded train and val regions. ``val_ds`` is
    ``None`` for a run whose partition holds no val side. The samples' ground truth is read
    once for both loaders (:func:`~tcip_mcp.pipelines.data.label_queries.acquired`)."""
    from tcip_mcp.pipelines.data.datasets import build_dataset, stated_sizes
    from tcip_mcp.pipelines.data.label_queries import acquired

    samples = acquired(samples)
    scope, sizes = data.recorded_scope, stated_sizes(data)
    if spatial is not None:
        views = _spatial_views(samples, scope, sizes, _tile_options(data.tiling),
                               {"train": spatial.train_region, "val": spatial.val_region},
                               transforms=transforms)
        return views["train"], views["val"]
    build_kwargs: dict[str, Any] = {
        "tiling": data.tiling, "scope": scope, "sizes": sizes,
        "dataset_source": data.dataset_source and (data.dataset_source, layout),
    }
    train = [s for s in samples if s.side == "train"]
    val = [s for s in samples if s.side == "val"]
    return (build_dataset(task, samples=train, transforms=transforms, **build_kwargs),
            build_dataset(task, samples=val, transforms=None, **build_kwargs) if val else None)

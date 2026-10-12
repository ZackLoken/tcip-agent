"""Constructing training splits from a data config, beside ``splits.py``.

Resolves a run once at launch (:func:`resolve_run`): its data section, the partition its record
holds (a bound selection, an auto group-aware draw, or a single-source spatial-strip split) and
its objective, and builds ``(train_ds, val_ds)`` from that record (:func:`recorded_datasets`).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from tcip_mcp.pipelines.data.selection import ClassScope, Sample, Selection
    from tcip_mcp.pipelines.image_utils import BandGroupRef
    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.schemas import (
        BoundSplit,
        DataSpec,
        DrawnSplit,
        SpatialManifest,
        SplitSpec,
        TilingSpec,
        TrainConfigSchema,
        TrainGeometry,
        TraitRevisionRef,
    )

logger = logging.getLogger(__name__)

SPATIAL_SIDE_ORDER = ("train", "val", "holdout", "calibration")
"""The order a within-image split lays its sides out in, which breaks a tie between equal shares
(:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`)."""


def data_dir_issues(data: DataSpec) -> list[str]:
    """Every objection to the data locations an unbound run's producer reads: ``data.images_dir``
    missing or naming a path that does not exist, and a stated ``data.labels_dir`` that is not a
    mask directory or a table (:func:`~tcip_mcp.pipelines.data.label_queries.ground_truth_shape`).
    Empty for a config bound to a selection (:func:`selection_compatibility`).
    """
    from tcip_mcp.pipelines.data.label_queries import ground_truth_shape
    from tcip_mcp.pipelines.schemas import BoundSplit

    if isinstance(data.split, BoundSplit):
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


def selection_compatibility(
    data: DataSpec,
    split: BoundSplit,
    selection: Selection | None,
) -> list[str]:
    """Every objection binding a run's data block ``data`` through its bound ``split`` to
    ``selection`` (``None`` when it would not read) raises: a stated ``data.labels_dir`` (each
    sample names its own ground truth), and, with a selection in hand, a stated ``data.scope``
    (the selection records its own class space) and an empty train or val side.
    """
    issues: list[str] = []
    if data.labels_dir is not None:
        issues.append(
            "data.labels_dir is stated beside data.split.selection_dir, whose samples each name "
            "their own ground truth; a second place would not be the one the run reads. Drop "
            "data.labels_dir."
        )
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
            f"the selection at {split.selection_dir} leaves an empty side "
            f"(train={counts['train']}, val={counts['val']}); a run needs both."
        )
    return issues


class Membership(NamedTuple):
    """Every member the places of one draw admit, keyed by its identity (``<capture>/<stem>``),
    each a sample on the ``train`` side under its group key and carrying the digest its admission
    read; the class space they were admitted in, the recorded grouping policy and the admissions'
    summed tallies.
    """

    samples: dict[str, Sample]
    scope: ClassScope
    group_by: str
    tallies: dict[str, int]


def admitted_membership(
    places: Sequence[tuple[str, Path | str, str | None]],
    *,
    scope: ClassScope,
    grouping: str | Mapping[str, str],
) -> Membership:
    """Admit each place ``(name, images_dir, ground_truth)``, ``ground_truth`` ``None`` for the
    images' own label documents
    (:func:`~tcip_mcp.pipelines.data.label_queries.admit`, under ``scope``, read from the images'
    registry when it carries no attributes yet,
    :func:`~tcip_mcp.pipelines.data.label_queries.registry_scope`) and group every admitted member
    (:func:`~tcip_mcp.pipelines.data.splits.resolve_group_key_fn` over the identities).

    Raises ``ValueError`` when nothing is admitted (the admission's own reason, naming every
    place searched and the tallies) and for a ``grouping`` that does not cover the members;
    the admissions' own refusals propagate.
    """
    from tcip_mcp.pipelines.data.label_queries import (
        admit,
        registry_scope,
        require_admitted,
    )
    from tcip_mcp.pipelines.data.splits import (
        member_identity,
        recorded_group_by,
        resolve_group_key_fn,
    )

    admissions = []
    tallies: dict[str, int] = {}
    for _name, images_dir, ground_truth in places:
        admitted = admit(
            images_dir,
            ground_truth,
            scope=scope
            if scope.attributes is not None
            else registry_scope(images_dir, scope.subject),
        )
        admissions.append(admitted)
        for key, value in admitted.tallies.items():
            tallies[key] = tallies.get(key, 0) + value
    identities = {
        (where, record.member): member_identity(admitted.date, record.member)
        for where, admitted in enumerate(admissions)
        for record in admitted.records
    }
    if not identities:
        searched = ", ".join(
            f"{name} -> {images_dir}" for name, images_dir, _gt in places
        )
        try:
            require_admitted(admissions[0])
        except ValueError as exc:
            raise ValueError(
                f"{exc} Searched {searched} ({tallies})."
            ) from exc
    group_of = resolve_group_key_fn(grouping, sorted(identities.values()))
    samples: dict[str, Sample] = {}
    for where, admitted in enumerate(admissions):
        built = admitted.samples(
            {record.member: "train" for record in admitted.records},
            {
                r.member: group_of(identities[where, r.member])
                for r in admitted.records
            }.__getitem__,
        )
        samples.update(
            (identities[where, sample.member], sample) for sample in built
        )
    return Membership(
        samples,
        admissions[0].scope,
        recorded_group_by(grouping),
        tallies,
    )


def run_membership(data: DataSpec, split: DrawnSplit) -> Membership:
    """:func:`admitted_membership` over a run's data block naming its locations
    (:func:`data_dir_issues`): its one place (``images_dir`` and the ``labels_dir`` a mask or
    table run names), under the ``scope`` it states (the empty one when it states none), grouped
    by its drawn ``split``'s grouping.
    """
    from tcip_mcp.pipelines.data.selection import ClassScope

    images_dir = cast(str, data.images_dir)
    return admitted_membership(
        [(images_dir, images_dir, data.labels_dir)],
        scope=data.scope if data.scope is not None else ClassScope(),
        grouping=split.grouping,
    )


def check_shares(sides: Mapping[str, float]) -> dict[str, float]:
    """``sides``, the share each side but ``train`` is drawn at, with ``train`` the remainder; a
    side stated at zero is not drawn. Refuses (``ValueError``) a side outside the selection's
    sides or ``train`` itself, a share outside ``[0, 1)``, and shares leaving ``train`` nothing.
    """
    import math

    from tcip_mcp.pipelines.data.selection import SIDES

    drawn = {side: float(share) for side, share in sides.items() if share != 0}
    remainder = 1.0 - math.fsum(drawn.values())
    if (
        set(sides) - set(SIDES)
        or "train" in sides
        or not all(0 < share < 1 for share in drawn.values())
        or remainder <= 0
    ):
        raise ValueError(
            f"a draw takes a share in [0, 1) for each side of {list(SIDES)} but "
            f"train, which takes the remainder, and the shares must leave it some; "
            f"got {dict(sides)}"
        )
    return {"train": remainder, **drawn}


def stated_shares(split: DrawnSplit) -> dict[str, float]:
    """The shares the drawn ``split`` states, each side's ``<side>_ratio``
    (:func:`check_shares`, ``train`` the remainder and a side at zero not drawn).
    """
    from tcip_mcp.pipelines.data.selection import SIDES

    return check_shares(
        {side: getattr(split, f"{side}_ratio") for side in SIDES[1:]}
    )


def draw_sides(
    samples: Mapping[str, Sample],
    scope: ClassScope,
    *,
    ratios: Mapping[str, float],
    seed: int,
    stratify: bool,
) -> tuple[dict[str, Sample], dict[str, int]]:
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
        foreground_group_count,
        group_balanced_split,
        refuse_insufficient_foreground_groups,
    )

    counted = foreground_counts(samples, scope)
    group_of = {
        key: sample.group for key, sample in samples.items()
    }.__getitem__
    refuse_insufficient_foreground_groups(
        foreground_group_count(counted, counted, group_of),
        {side: 1 for side in ratios},
        remedy=(
            "add ground truth for more images: annotate or confirm more of them for this "
            "subject, or write the masks or rows that answer for them."
        ),
    )
    reference = [side for side in REFERENCE_SIDES if side in ratios]
    first = {
        side: share
        for side, share in ratios.items()
        if side not in REFERENCE_SIDES
    }
    minimums = dict.fromkeys(first, 1)
    if reference:
        first["reference"] = sum(ratios[side] for side in reference)
        minimums["reference"] = len(reference)

    def draw(keys, splits, need) -> dict[str, list[str]]:
        return group_balanced_split(
            keys,
            counts=counted,
            weighted=stratify,
            group_key_fn=group_of,
            splits=splits,
            seed=seed,
            min_foreground_groups=need,
        )

    drawn = draw(sorted(samples), first, minimums)
    if reference:
        pool = drawn.pop("reference")
        drawn.update(
            draw(
                pool,
                {
                    side: ratios[side] / first["reference"]
                    for side in reference
                },
                dict.fromkeys(reference, 1),
            )
        )
    starved = sorted(side for side, keys in drawn.items() if not keys)
    if starved:
        raise ValueError(
            f"the draw at seed {seed} leaves {starved} empty: its groups cannot "
            "populate every side requested. State a share of zero for a side not "
            "to draw it, or add ground truth."
        )
    return (
        {
            key: dataclasses.replace(samples[key], side=side)
            for side, keys in drawn.items()
            for key in keys
        },
        counted,
    )


def split_seed(split: SplitSpec) -> int:
    """The seed ``split`` draws at; a bound split stating none refuses (``ValueError``)."""
    if split.seed is None:
        raise ValueError(
            "this bound data.split states no seed to redraw its selection at: the "
            "partition a redraw produces depends on it, so state data.split.seed."
        )
    return split.seed


def dataset_identity(
    data: DataSpec, samples: Iterable[Sample] = ()
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
    return (record["id"] if record is not None else None), dataset_fingerprint(
        root, samples
    )


def raster_identity(image: Path | BandGroupRef) -> dict:
    """The logical ``image``'s :func:`~tcip_mcp.pipelines.raster_source.raster_content_identity`
    over its own probed band count, the one a split records its mosaic by. A source that will not
    probe raises.
    """
    import dataclasses

    from tcip_mcp.pipelines.raster_source import SourceHeader, content_identity

    header = SourceHeader(image)
    with header.open(header.channels) as src:
        return dataclasses.asdict(content_identity(src))


def spatial_single_source_split(
    task: str,
    sample: Sample,
    data: DataSpec,
    shares: Mapping[str, float],
) -> SpatialManifest:
    """Derive the run's requested ``shares`` (:func:`stated_shares`: ``train``, ``val`` and each
    reserved side stated above zero) over one source's own tile lattice,
    by disjoint pixel strips (:func:`~tcip_mcp.pipelines.data.splits.spatial_strip_split`), and
    return it as the :class:`~tcip_mcp.pipelines.schemas.SpatialManifest` a run's partition
    records; the reserved regions record only their geometry and kept-tile count. It is drawn at
    the lattice and buffer of the resolved block ``data``'s tiling (:func:`_with_lattice`).

    ``sample`` is the ``task`` run's own single admitted sample, which every side's view here is
    built over (:func:`_block_loader`) at ``data``'s class space and sizes (:func:`run_sizes`).

    Raises ``ValueError`` naming which reason fired when the strip layout is infeasible or a side
    keeps no tile after filtering. The label document's frame is read through
    :func:`~tcip_mcp.pipelines.data.splits.label_document_extent`, whose refusals propagate.
    """
    from tcip_mcp.pipelines.data.label_queries import acquired, resolved
    from tcip_mcp.pipelines.data.splits import (
        label_document_extent,
        spatial_strip_split,
    )
    from tcip_mcp.pipelines.schemas import SpatialManifest

    stem = sample.member
    remedy = (
        "reduce the reserved shares, or state every share zero to train on it whole without "
        "validation"
    )

    (sample,) = resolved(acquired([sample]))
    width, height = label_document_extent(
        sample.read, f"{stem}'s label document"
    )
    split_names = tuple(side for side in SPATIAL_SIDE_ORDER if side in shares)

    tiling = cast("TilingSpec", data.tiling)
    try:
        spatial = spatial_strip_split(
            width,
            height,
            cast(int, tiling.tile_size),
            cast(float, tiling.overlap),
            fractions=tuple(shares[s] for s in split_names),
            split_names=split_names,
            buffer=tiling.buffer,
        )
    except ValueError as exc:
        raise ValueError(
            f"the spatial split of {stem!r} at shares {dict(shares)} is infeasible at this "
            f"mosaic size/tile size ({exc}); {remedy}."
        ) from exc

    # A tile lattice occupying a reserved region (spatial_strip_split's own check) is not proof it
    # carries GT: an all-background region still passes that but skip_empty filters it to 0.
    views = {
        side: _block_loader(
            task,
            data,
            samples=[sample],
            transforms=None,
            region=spatial.regions[side],
        )
        for side in split_names
    }
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
        raw = {
            spatial.identity_for(ds.sample_of(key).member, box)
            for key, box in ds.tile_entries
        }
        return sorted(name for name in raw if name is not None)

    logger.info(
        "Spatial train/val split for %r: %d train / %d val tiles (axis=%s, "
        "realized_fractions=%s, realized_discard_fraction=%.3f).",
        stem,
        train_ds.num_samples,
        val_ds.num_samples,
        spatial.axis,
        spatial.realized_fractions,
        spatial.realized_discard_fraction,
    )
    return SpatialManifest(
        stem=stem,
        train_identities=_identities(train_ds),
        val_identities=_identities(val_ds),
        train_region=spatial.regions.get("train", []),
        val_region=spatial.regions.get("val", []),
        holdout_region=spatial.regions.get("holdout", []),
        calibration_region=spatial.regions.get("calibration", []),
        kept_holdout_tiles=spatial.kept_tiles.get("holdout", 0),
        kept_calibration_tiles=spatial.kept_tiles.get("calibration", 0),
        width=spatial.width,
        height=spatial.height,
        axis=spatial.axis,
        buffer=spatial.buffer,
        requested_fractions=dict(
            zip(spatial.split_names, spatial.requested_fractions)
        ),
        realized_fractions=spatial.realized_fractions,
        realized_discard_fraction=spatial.realized_discard_fraction,
        kept_train_tiles=spatial.kept_tiles.get("train", 0),
        kept_val_tiles=spatial.kept_tiles.get("val", 0),
        tiles_dropped_past_extent=spatial.tiles_dropped_past_extent,
        tiles_dropped_outside_regions=spatial.tiles_dropped_outside_regions,
        raster_content_identity=raster_identity(sample.image),
    )


def _block_loader(
    task: str,
    data: DataSpec,
    *,
    samples: Sequence[Sample],
    transforms: Any,
    region: list | None = None,
) -> Any:
    """The platform's ``task`` loader over ``samples`` at ``transforms`` under the resolved block
    ``data``'s class space, sizes and tiling, its tiles kept to ``region``'s rects when one is
    given.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset, stated_sizes

    tiling = (
        data.tiling
        if region is None
        else cast("TilingSpec", data.tiling).model_copy(
            update={"keep_regions": region}
        )
    )
    return build_dataset(
        task,
        samples=samples,
        transforms=transforms,
        scope=data.recorded_scope,
        sizes=stated_sizes(data),
        tiling=tiling,
    )


def resolved_tiling(
    task: str,
    tiling: TilingSpec | None,
    samples: Sequence[Sample],
    scope: ClassScope,
    sizes: Mapping[str, int],
) -> TilingSpec | None:
    """The tiling block a ``task`` loader over ``samples`` (at ``scope`` and ``sizes``) is built
    at: ``None`` when it does not tile, else ``tiling`` with its edge and overlap, the stated
    ones or the ones ``samples``' objects derive (``derivations.derive_tile_geometry``, whose
    refusals propagate).
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.derivations import derive_tile_geometry
    from tcip_mcp.pipelines.schemas import run_tiling

    tiler = run_tiling(task, tiling)
    if tiler is None:
        return None
    regions = (
        []
        if tiler.tile_size is not None and tiler.overlap is not None
        else cast(
            Any, build_dataset(task, samples=samples, scope=scope, sizes=sizes)
        ).regions
    )
    tile_size, overlap = derive_tile_geometry(
        regions, tile_size=tiler.tile_size, overlap=tiler.overlap
    )
    return tiler.model_copy(
        update={"tile_size": tile_size, "overlap": overlap}
    )


def predictor_dataset(
    task: str,
    samples: Sequence[Sample],
    scope: ClassScope,
    predictor: Any,
    tiling: TilingSpec | None,
) -> Any:
    """The ``task`` loader over ``samples`` under ``scope`` a predictor scores: at the band count
    ``predictor`` reads its sources at (``predictor.in_chans``) and the tiling block
    :func:`resolved_tiling` resolves ``tiling`` to over them, an absent block untiled. The sizing
    and the resolution refuse as they do.
    """
    from tcip_mcp.pipelines.data.datasets import build_dataset, resolve_sizes

    sizes = resolve_sizes(task, {"num_channels": predictor.in_chans}, samples)
    return build_dataset(
        task,
        samples=samples,
        scope=scope,
        sizes=sizes,
        tiling=resolved_tiling(task, tiling, samples, scope, sizes),
    )


def _with_lattice(
    task: str, data: DataSpec, samples: Sequence[Sample]
) -> DataSpec:
    """``data`` with its tiling block resolved over ``samples`` (:func:`resolved_tiling`);
    ``data`` unchanged, its tiling ``None``, for a run whose loaders a bespoke builder makes,
    the platform stating nothing of a dataset it does not build.
    """
    from tcip_mcp.pipelines.data.datasets import stated_sizes

    if data.dataset_source is not None:
        return data
    return data.model_copy(
        update={
            "tiling": resolved_tiling(
                task,
                data.tiling,
                samples,
                data.recorded_scope,
                stated_sizes(data),
            )
        }
    )


def redrawn_selection(
    selection: Selection, selection_dir: str, seed: int
) -> Selection:
    """``selection`` with train and val redrawn fresh over its own train-plus-val samples at
    ``seed`` (:func:`draw_sides`, over each sample's recorded group key, at the val share the
    selection already delivered, stratified as the selection records its own draw was), the
    reference sides untouched. The draw's refusals
    raise, naming ``selection_dir``. Each redrawn sample keeps the read it carries
    (:func:`~tcip_mcp.pipelines.data.label_queries.acquired`).
    """
    from tcip_mcp.pipelines.data.label_queries import acquired

    pool = {s.location: s for s in acquired(selection.trainable())}
    val_share = len(selection.on("val")) / len(pool) if pool else 0.0
    try:
        drawn, _counted = draw_sides(
            pool,
            selection.scope,
            ratios=check_shares({"val": val_share}),
            seed=seed,
            stratify=selection.stratify_foreground,
        )
    except ValueError as exc:
        raise ValueError(
            f"redrawing train and val inside the selection at {selection_dir!r}: "
            f"{exc} Drop data.split.seed to bind the selection's recorded partition "
            "instead."
        ) from exc
    from tcip_mcp.pipelines.data.selection import refuse_crossing_sides

    redrawn = tuple(drawn.get(s.location, s) for s in selection.samples)
    refuse_crossing_sides(redrawn)
    return replace(selection, samples=redrawn)


def _partition_record(
    samples: Sequence[Sample],
    *,
    seed: int | None,
    group_by: str,
    selection: dict | None,
    spatial: SpatialManifest | None,
) -> dict:
    """The partition a run's resolved record holds: the draw's ``seed`` (``None`` for a route
    that drew nothing) and resolved ``group_by``, every sample the run bound on the side it
    landed on in its one recorded shape
    (:func:`~tcip_mcp.pipelines.data.selection.sample_document`), with the ground-truth digest it
    carries, ``selection``, the selection a bound run bound (its directory, its digest and
    whether the run redrew inside it), ``None`` for a run that drew its own, and ``spatial``, a
    within-image split's regions (:func:`spatial_single_source_split`), ``None`` otherwise.
    """
    from tcip_mcp.pipelines.data.selection import sample_document

    return {
        "seed": seed,
        "group_by": group_by,
        "samples": [sample_document(s) for s in samples],
        "selection": selection,
        "spatial": None
        if spatial is None
        else spatial.model_dump(mode="json"),
    }


def partition_samples(partition: Mapping[str, Any]) -> list[Sample]:
    """A resolved partition's samples read back
    (:func:`~tcip_mcp.pipelines.data.selection.read_sample`).
    """
    from tcip_mcp.pipelines.data.selection import read_sample

    return [
        read_sample(raw, position, "a run's resolved partition")
        for position, raw in enumerate(partition["samples"])
    ]


def partition_spatial(
    partition: Mapping[str, Any],
) -> SpatialManifest | None:
    """A resolved partition's within-image split read back, validated; ``None`` for a partition
    that drew none.
    """
    from tcip_mcp.pipelines.schemas import SpatialManifest

    spatial = partition["spatial"]
    return None if spatial is None else SpatialManifest.model_validate(spatial)


def run_sizes(
    task: str,
    data: DataSpec,
    scope: ClassScope,
    samples: Sequence[Sample],
) -> DataSpec:
    """``data`` with what a run resolved over ``samples`` recorded on it: the class space
    ``scope`` it was admitted under, and its sizes
    (:func:`~tcip_mcp.pipelines.data.datasets.resolve_sizes` over the ones ``data`` states), each
    by its own name.
    """
    from tcip_mcp.pipelines.data.datasets import (
        SIZE_NAMES,
        resolve_sizes,
        stated_sizes,
    )

    sizes = resolve_sizes(
        task, stated_sizes(data), samples, data.dataset_source
    )
    return data.model_copy(
        update={
            "scope": scope,
            **{name: sizes.get(name) for name in SIZE_NAMES},
        }
    )


def _sample_loaders(
    task: str,
    data: DataSpec,
    recorded: Sequence[Sample],
    transforms,
    layout: SourceLayout,
    *,
    seed: int | None,
    group_by: str,
    selection: dict | None = None,
    spatial: SpatialManifest | None = None,
):
    """``(train_ds, val_ds, partition, data)``: the partition (:func:`_partition_record` over
    ``recorded`` at ``seed``, ``group_by``, the ``selection`` a bound run bound and a
    within-image run's ``spatial`` split) and :func:`recorded_datasets` over it, the resolved
    block ``data`` (its tiling already resolved, :func:`_with_lattice`) and the run's ``layout``.
    ``recorded`` is the membership the partition holds: the two sides for a drawn run, and the
    selection's held-out samples besides for a bound one.
    """
    partition = _partition_record(
        recorded,
        seed=seed,
        group_by=group_by,
        selection=selection,
        spatial=spatial,
    )
    return (
        *recorded_datasets(task, data, recorded, spatial, transforms, layout),
        partition,
        data,
    )


def _drawn_split(
    task: str,
    data: DataSpec,
    split: DrawnSplit,
    layout: SourceLayout,
    *,
    transforms,
    tallies_out: dict[str, int] | None,
):
    """``(train_ds, val_ds, partition, resolved)`` for a run that draws its own ``split`` over
    ``data``'s ground truth: its one place's membership (:func:`admitted_membership`, its tallies
    copied into ``tallies_out``), then train and val drawn at the split's shares
    (:func:`stated_shares`) through :func:`draw_sides`, or every stated side over a single tiled
    source's own tile lattice (:func:`spatial_single_source_split`). A split stating every share
    zero trains on every member with no validation. ``resolved`` is ``data`` with its class space
    and sizes recorded, and a within-image split is the partition's. Raises when a single source
    cannot hold the sides requested out, when a run over several sources states a reserved
    share (a reference is the reserved strips of one tiled source, or a selection
    ``draw_splits`` draws), and with the membership's and the draw's own refusals.
    """
    membership = run_membership(data, split)
    if tallies_out is not None:
        tallies_out.update(membership.tallies)
    samples = list(membership.samples.values())
    resolved = run_sizes(task, data, membership.scope, samples)
    shares = stated_shares(split)

    if len(shares) == 1:
        return _sample_loaders(
            task,
            _with_lattice(task, resolved, samples),
            samples,
            transforms,
            layout,
            seed=None,
            group_by=membership.group_by,
        )
    if len(samples) < 2:
        from tcip_mcp.pipelines.schemas import run_tiling

        if run_tiling(task, data.tiling) is None or "val" not in shares:
            raise ValueError(
                f"{len(samples)} admitted source(s) leave nothing to hold {sorted(shares)} out "
                f"of for this {task} run: one source splits only within itself, tiled and "
                "with a validation strip; state data.tiling and data.split.val_ratio above "
                "zero, every share zero to train on it whole without validation, or admit "
                "more sources."
            )
        # A tiled source the platform builds itself holds out disjoint pixel blocks, drawn at the
        # lattice its own objects derive.
        resolved = _with_lattice(task, resolved, samples)
        manifest = spatial_single_source_split(
            task, samples[0], resolved, shares
        )
        return _sample_loaders(
            task,
            resolved,
            samples,
            transforms,
            layout,
            seed=None,
            group_by="stem",
            spatial=manifest,
        )

    from tcip_mcp.pipelines.data.selection import REFERENCE_SIDES

    reserved = sorted(f"{side}_ratio" for side in set(REFERENCE_SIDES) & set(shares))
    if reserved:
        raise ValueError(
            f"data.split states {reserved} over {len(samples)} admitted sources: a run holds "
            "a reference out only as the reserved strips of one tiled source; draw a "
            "selection with draw_splits for a reference, or state those shares zero."
        )
    seed = split.seed
    drawn, _counted = draw_sides(
        membership.samples,
        membership.scope,
        ratios=shares,
        seed=seed,
        stratify=split.stratify_foreground,
    )
    recorded = [drawn[key] for key in sorted(drawn)]
    return _sample_loaders(
        task,
        _with_lattice(
            task, resolved, [s for s in recorded if s.side == "train"]
        ),
        recorded,
        transforms,
        layout,
        seed=seed,
        group_by=membership.group_by,
    )


def auto_train_val(
    project: Path,
    task: str,
    data: DataSpec,
    transforms,
    layout: SourceLayout,
    *,
    tallies_out: dict[str, int] | None = None,
):
    """Build ``(train_ds, val_ds, partition, resolved)`` for a run of ``project`` over its data
    block ``data``, deriving a leakage-free val split, a bespoke dataset builder imported from
    the run's ``layout``; ``resolved`` is ``data`` with its ``scope`` and sizes recorded on it.

    ``partition`` is :func:`_partition_record`'s; a within-image spatial split's holds its one
    sample and its ``spatial`` regions. ``tallies_out`` receives a drawn run's admission's
    tallies.

    Two routes, and a run of any task takes one of them:
      1. ``data.split.selection_dir`` set -> train on the selection's own ``train`` and ``val``
        samples instead of drawing a split, each sample reading the source and the ground truth the
        draw recorded for it, under the selection's own scope. Every conflict, empty-side
        refusal and build failure raises. The
        loader refuses by name when the task reads another shape than the samples name. The
        calibration side never builds a loader.
        A ``data.split.seed`` stated beside ``selection_dir`` redraws train and val fresh at it
        inside the selection's own train-plus-val samples, calibration still untouched; a
        starved side refuses.
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

    from tcip_mcp.pipelines.schemas import BoundSplit

    split = data.split

    # 1. A named selection is an explicit partition; every refusal here, and any build failure
    # while binding to it, raises rather than degrading.
    if isinstance(split, BoundSplit):
        selection_dir = split.selection_dir
        from tcip_mcp.pipelines.data.label_queries import readmitted_samples
        from tcip_mcp.pipelines.data.selection import (
            REFERENCE_SIDES,
            read_selection,
            selection_digest,
        )

        selection: Selection | None = None
        unread: ValueError | None = None
        try:
            selection = read_selection(selection_dir, project=project)
        except ValueError as exc:
            unread = exc
        # The bind's own refusals, stated once so the preflight that offered this selection and
        # the launch that binds it say the same thing, the config's own even over an unread one.
        bind_issues = selection_compatibility(data, split, selection)
        if bind_issues:
            raise ValueError(" ".join(bind_issues)) from unread
        if selection is None:
            raise unread or ValueError(
                f"no selection read under {selection_dir}"
            )

        # Re-admitted before any draw or loader, whose reads both consume: a selected label emptied
        # with nobody confirming that image negative would otherwise train as background.
        readmitted = {
            s.location: s
            for s in readmitted_samples(selection.trainable(), selection.scope)
        }
        selection = replace(
            selection,
            samples=tuple(
                readmitted.get(s.location, s) for s in selection.samples
            ),
        )
        # A redraw repartitions the selection's own train-plus-val samples; the reference untouched.
        seed = selection.seed
        if split.seed is not None:
            seed = split.seed
            selection = redrawn_selection(selection, selection_dir, seed)

        reference_samples = [
            s for s in selection.samples if s.side in REFERENCE_SIDES
        ]
        trained = selection.on("train") + selection.on("val")
        train_samples = [s for s in trained if s.side == "train"]
        val_samples = [s for s in trained if s.side == "val"]

        bound = trained + reference_samples
        # The selection's own scope and exact map become this run's, so the checkpoint records
        # the vocabulary it trained in rather than one rediscovered from a live registry.
        # The selection's own named policy, not "explicit_map": the per-stem map the partition
        # records covers this run's members, and a stem outside it is what a policy name answers.
        return _sample_loaders(
            task,
            _with_lattice(
                task,
                run_sizes(
                    task, data, selection.scope, train_samples + val_samples
                ),
                train_samples,
            ),
            bound,
            transforms,
            layout,
            seed=seed,
            group_by=selection.group_by,
            selection={
                "selection_dir": selection_dir,
                "selection_sha256": selection_digest(selection, project),
                "redraw": split.seed is not None,
            },
        )

    # 2. Otherwise the producer names this run's membership off the ground truth its config points
    # at, and every branch of the resolution order below builds from the samples it made.
    return _drawn_split(
        task,
        data,
        split,
        layout,
        transforms=transforms,
        tallies_out=tallies_out,
    )


class ResolvedRun(NamedTuple):
    """A run resolved once, at launch: the datasets built at it, ``spec``, the validated launch
    config it was resolved from, and ``data``, ``geometry``, ``partition`` and
    ``trait_revision``, what it resolved.
    """

    train_ds: Any
    val_ds: Any
    spec: TrainConfigSchema
    data: DataSpec
    geometry: TrainGeometry
    partition: dict
    trait_revision: TraitRevisionRef | None

    @property
    def spatial(self) -> SpatialManifest | None:
        """The within-image split ``partition`` records (:func:`partition_spatial`)."""
        return partition_spatial(self.partition)

    @property
    def record(self) -> dict:
        """What the run's ``run.json`` holds of this resolution: ``data`` as it records itself,
        ``geometry``, ``partition`` and ``trait_revision``.
        """
        from tcip_mcp.pipelines.schemas import revision_record

        return {
            "data": self.data.record(),
            "geometry": self.geometry.model_dump(mode="json"),
            "partition": self.partition,
            "trait_revision": revision_record(self.trait_revision),
        }


def resolve_run(
    spec: TrainConfigSchema,
    layout: SourceLayout,
    *,
    project: Path,
    trait_revision: TraitRevisionRef | None = None,
    tallies_out: dict[str, int] | None = None,
) -> ResolvedRun:
    """Resolve the validated config ``spec`` (``schemas.train_config``) once for a run of
    ``project`` whose declared files ``layout`` lays out (``model_build.SourceLayout``): its data
    block through :func:`auto_train_val`, the geometry its train dataset serves
    (:func:`~tcip_mcp.pipelines.training.generic_trainer.effective_data_geometry`), and its
    trait revision (:func:`~tcip_mcp.pipelines.training.generic_trainer.resolve_trait_revision`
    for a run with or without a val side), or ``trait_revision`` as given, a sweep's own for its
    trials. ``tallies_out`` is :func:`auto_train_val`'s. Every refusal of the resolution raises.
    """
    from tcip_mcp.pipelines.training.generic_trainer import (
        effective_data_geometry,
        resolve_trait_revision,
        run_transforms,
    )

    task = spec.model_source.task
    train_ds, val_ds, partition, data = auto_train_val(
        project,
        task,
        spec.data,
        run_transforms(spec),
        layout,
        tallies_out=tallies_out,
    )
    geometry = effective_data_geometry(task, data, train_ds)
    if trait_revision is None:
        trait_revision = resolve_trait_revision(
            spec, project=project, has_val_loader=val_ds is not None
        )
    return ResolvedRun(
        train_ds, val_ds, spec, data, geometry, partition, trait_revision
    )


def run_loader(
    task: str,
    data: DataSpec,
    layout: SourceLayout,
    spatial: SpatialManifest | None = None,
) -> Any:
    """A callable taking ``samples`` and ``transforms`` that builds a dataset of ``task`` over
    that sample list under the resolved block ``data``'s class space, by the builder its dataset
    source names (imported from the run's ``layout``) or else by the platform's factory at the
    block's sizes and tiling.

    Over a within-image split ``spatial`` (:func:`partition_spatial`) the callable also takes
    ``side`` (``"train"`` unless named) and builds the run's one sample's tiles kept to that
    side's recorded region; a sample list that is not that one sample, by member, refuses
    (``ValueError``) naming it. Either builder runs only once ``transforms`` is admitted over the
    samples at the block's band count
    (:func:`~tcip_mcp.pipelines.data.datasets.refuse_unchained`).
    """
    from tcip_mcp.pipelines.data.datasets import (
        build_from_dataset_source,
        refuse_unchained,
        stated_sizes,
    )

    def admitted(build: Any) -> Any:
        def loader(
            *, samples: Sequence[Sample], transforms: Any, **side: str
        ) -> Any:
            refuse_unchained(
                samples, stated_sizes(data)["num_channels"], transforms
            )
            return build(samples=samples, transforms=transforms, **side)

        return loader

    if data.dataset_source is not None:
        return admitted(
            partial(
                build_from_dataset_source,
                (data.dataset_source, layout),
                task=task,
                scope=data.recorded_scope,
            )
        )
    if spatial is None:
        return admitted(partial(_block_loader, task, data))

    def side_view(
        *, samples: Sequence[Sample], transforms: Any, side: str = "train"
    ) -> Any:
        members = [s.member for s in samples]
        if members != [spatial.stem]:
            raise ValueError(
                f"this run splits the one image {spatial.stem!r} by region, so its loaders are "
                f"built over that one sample, never over {members}."
            )
        return _block_loader(
            task,
            data,
            samples=samples,
            transforms=transforms,
            region=getattr(spatial, f"{side}_region"),
        )

    return admitted(side_view)


def recorded_datasets(
    task: str,
    data: DataSpec,
    samples: Sequence[Sample],
    spatial: SpatialManifest | None,
    transforms,
    layout: SourceLayout,
) -> tuple[Any, Any]:
    """``(train_ds, val_ds)`` built from what a run resolved, resolving nothing again, by
    :func:`run_loader`: the train and val ``samples`` of its partition, or for a within-image
    split ``spatial`` (:func:`partition_spatial`) its one sample's train and val sides.
    ``val_ds`` is ``None`` for a run whose partition holds no val side. The samples' ground truth
    is read once for both loaders (:func:`~tcip_mcp.pipelines.data.label_queries.acquired`).
    """
    from tcip_mcp.pipelines.data.label_queries import acquired

    samples = acquired(samples)
    build = run_loader(task, data, layout, spatial)
    if spatial is not None:
        return (
            build(samples=samples, transforms=transforms),
            build(samples=samples, transforms=None, side="val"),
        )
    train = [s for s in samples if s.side == "train"]
    val = [s for s in samples if s.side == "val"]
    return (
        build(samples=train, transforms=transforms),
        build(samples=val, transforms=None) if val else None,
    )

"""Group-aware, annotation-stratified train/val/calibration splitting: group-coherent (sibling
tiles of one source never straddle two splits), annotation-balanced, deterministic in seed.
Torch-free.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

from tcip_store import (
    RECORD_JSON,
    BadKey,
    DecodeError,
    Key,
    StoreDescriptor,
    canonical_path,
    register_store,
    store,
)

from tcip_mcp.pipelines.data.selection import SIDES

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import Sample, Selection

logger = logging.getLogger(__name__)

# A tiled stem looks like ``<source>_<x>_<y>`` (two trailing integer fields).
# Strip that suffix so all tiles of one source share a group key. A stem with a
# single trailing ``_<int>`` (e.g. ``img_001``) does not match and falls back to
# the full stem.
_TILE_GROUP_RE = re.compile(r"^(.*)_\d+_\d+$")

# The draw defaults every split door applies when its config states none.
DEFAULT_GROUP_BY = "tile_prefix"
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.2
DEFAULT_HOLDOUT_RATIO = 0.5
DEFAULT_CAL_SEED = 0
"""The holdout share and seed a first cal/holdout lock draws at when its caller states neither."""


def default_group_key(stem: str) -> str:
    """Group key for a stem: strip a trailing ``_<x>_<y>`` tile offset.

    Falls back to the full stem when the pattern does not match.
    """
    m = _TILE_GROUP_RE.match(stem)
    return m.group(1) if m else stem


GROUP_KEY_FNS: dict[str, Callable[[str], str]] = {
    "tile_prefix": default_group_key,
    "stem": lambda s: s,
}


def count_label_lines(
    label_path: str | Path, *, subject: str | None = None, attribute: str | None = None,
) -> int:
    """Annotation count for one per-image label document, by its own path, a foreground-density
    proxy for stratified splitting.

    With ``subject`` omitted, every record in the file counts regardless of subject; given
    ``subject``, only that subject's records count, further narrowed to those already assessed for
    ``attribute`` when one is given. An empty ``subject`` or ``attribute`` reads as unset.

    A missing file scores 0 foreground; a present, unreadable one raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument`.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import instances

    subject, attribute = subject or None, attribute or None
    jp = Path(label_path)
    if not jp.is_file():
        return 0
    # A crowd region is never one object, so it is never counted as one.
    records = instances(json_io.read_annotations(str(jp)))
    if subject is None:
        return len(records)
    return sum(1 for a in records if json_io.assessed_key(a, subject, attribute) is not None)


def label_document_extent(label_path: str | Path) -> tuple[int, int] | None:
    """``(width, height)`` one per-image label JSON records (the json_io schema's top-level
    ``width``/``height``, the frame its boxes were authored against), or ``None`` when the file is
    missing or carries no positive width/height. A present, unreadable file raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument`.
    """
    from tcip_annotation.json_io import load_label_document

    p = Path(label_path)
    if not p.is_file():
        return None
    data = load_label_document(p)
    w, h = int(data.get("width", 0) or 0), int(data.get("height", 0) or 0)
    return (w, h) if w > 0 and h > 0 else None


def group_balanced_split(
    stems: Sequence[str],
    annotation_counts: dict[str, int] | None = None,
    group_key_fn: Callable[[str], str] | None = None,
    splits: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = DEFAULT_SEED,
    require_foreground: bool = False,
    min_foreground_groups: dict[str, int] | None = None,
    foreground_counts: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Partition ``stems`` into train/val/calibration, keeping each group intact.

    Parameters
    ----------
    stems:
        Image stems to partition.
    annotation_counts:
        Optional ``{stem: annotation_line_count}``. When omitted or all-zero, the no-foreground
        fallback weights groups purely by tile count.
    group_key_fn:
        Maps a stem to its group key (default: strip ``_<x>_<y>`` tile offset).
    splits:
        ``(train, val, calibration)`` fractions. A 0.0 fraction disables that split.
    seed:
        Deterministic seed.
    require_foreground:
        Raise ``ValueError`` when there is no foreground signal at all.
    min_foreground_groups:
        Per-side minimum count of foreground groups the balancing pass guarantees before it runs
        its ordinary largest-first assignment, met first with the smallest foreground groups so the
        dense ones remain for balancing. Omitted, every active side gets a minimum of one; a side
        named here with no active fraction is ignored. A tree with fewer foreground groups than a
        minimum asks for gets fewer than that side's floor met; this function never raises on it
        (:func:`refuse_insufficient_foreground_groups` is the hard floor).
    foreground_counts:
        The minimum pass's own foreground signal, independent of ``annotation_counts`` (the
        balancing pass's signal). Omitted, the minimum pass draws from ``annotation_counts``;
        given, a group counts toward a side's minimum only when its ``foreground_counts`` sum is
        positive, even when ``annotation_counts`` is ``None``.

    Returns
    -------
    ``{"train": [...], "val": [...], "calibration": [...]}``, a partition of ``stems``.
    """
    if group_key_fn is None:
        group_key_fn = GROUP_KEY_FNS[DEFAULT_GROUP_BY]
    stems = list(stems)
    fracs = dict(zip(SIDES, splits))
    active = [n for n in SIDES if fracs.get(n, 0.0) > 0]

    # Group stems and tally tiles + annotations per group.
    groups: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        groups[group_key_fn(s)].append(s)

    counts = annotation_counts or {}
    has_fg_signal = any(int(counts.get(s, 0)) > 0 for s in stems)
    if not has_fg_signal and require_foreground:
        raise ValueError("No foreground annotations and require_foreground=True.")

    group_tiles = {gk: len(gs) for gk, gs in groups.items()}
    if has_fg_signal:
        group_ann = {gk: sum(int(counts.get(s, 0)) for s in gs) for gk, gs in groups.items()}
        fg_groups = [gk for gk, a in group_ann.items() if a > 0]
        bg_groups = [gk for gk, a in group_ann.items() if a == 0]
    else:
        # No-foreground fallback: balance by tile count, every group foreground.
        group_ann = dict(group_tiles)
        fg_groups = list(groups.keys())
        bg_groups = []

    result: dict[str, list[str]] = {n: [] for n in SIDES}
    if not fg_groups or not active:
        # Nothing to stratify on; dump everything into the first active split.
        target = active[0] if active else "train"
        result[target] = sorted(stems)
        return result

    total_ann = sum(group_ann[gk] for gk in fg_groups) or 1
    total_fg_tiles = sum(group_tiles[gk] for gk in fg_groups) or 1
    targets_ann = {n: fracs[n] * total_ann for n in SIDES}
    targets_tiles = {n: fracs[n] * total_fg_tiles for n in SIDES}

    state_ann = {n: 0 for n in SIDES}
    state_tiles = {n: 0 for n in SIDES}
    assignment: dict[str, str] = {}
    used: set[str] = set()

    rng = random.Random(seed)
    rng.shuffle(fg_groups)

    # Each active side's minimum, met first with the smallest foreground groups (dense ones stay
    # for the balancing pass); a short tree just meets fewer minimums, nothing here raises on it.
    if min_foreground_groups is None:
        min_fg = {n: 1 for n in active}
    else:
        min_fg = {n: min_foreground_groups[n] for n in active if n in min_foreground_groups}

    if foreground_counts is None:
        min_pass_pool = fg_groups
    else:
        min_pass_pool = [
            gk for gk in fg_groups
            if sum(int(foreground_counts.get(s, 0)) for s in groups[gk]) > 0
        ]
    fg_smallest_first = sorted(min_pass_pool, key=lambda gk: group_ann[gk])
    for split_name, need in min_fg.items():
        taken = 0
        for gk in fg_smallest_first:
            if taken >= need:
                break
            if gk in used:
                continue
            assignment[gk] = split_name
            used.add(gk)
            state_ann[split_name] += group_ann[gk]
            state_tiles[split_name] += group_tiles[gk]
            taken += 1

    # Assign remaining foreground groups (largest first) to the active split that
    # minimizes a blended annotation/tile imbalance score.
    for gk in sorted(fg_groups, key=lambda g: group_ann[g], reverse=True):
        if gk in used:
            continue
        best_split: str | None = None
        best_score: float | None = None
        for n in active:
            ann_ratio = (state_ann[n] + group_ann[gk]) / max(1.0, targets_ann[n])
            tile_ratio = (state_tiles[n] + group_tiles[gk]) / max(1.0, targets_tiles[n])
            score = 0.7 * ann_ratio + 0.3 * tile_ratio
            if best_score is None or score < best_score:
                best_split, best_score = n, score
        assert best_split is not None, "active is non-empty, so the loop above always assigns"
        assignment[gk] = best_split
        used.add(gk)
        state_ann[best_split] += group_ann[gk]
        state_tiles[best_split] += group_tiles[gk]

    # Background groups -> active split with the largest overall tile deficit.
    total_all_tiles = sum(group_tiles.values()) or 1
    tile_target_all = {n: fracs[n] * total_all_tiles for n in SIDES}
    for gk in sorted(bg_groups, key=lambda g: group_tiles[g], reverse=True):
        best_split = max(active, key=lambda n: tile_target_all[n] - state_tiles[n])
        assignment[gk] = best_split
        state_tiles[best_split] += group_tiles[gk]

    for gk, gs in groups.items():
        result[assignment.get(gk, active[0])].extend(gs)
    return {n: sorted(result[n]) for n in SIDES}


def foreground_group_count(
    members: Iterable[str], counts: Mapping[str, int], group_of: Callable[[str], str],
) -> int:
    """How many distinct groups among ``members`` carry foreground. ``counts`` is each member's own
    foreground count under the caller's own key for it
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`); a member the map does not
    name carries none.
    """
    return len({group_of(member) for member in members if counts.get(member, 0) > 0})


def refuse_insufficient_foreground_groups(
    foreground_groups: int, minimums: dict[str, int], *, remedy: str,
) -> None:
    """Refuses, before any write, a draw whose tree holds fewer foreground groups than the sum of
    the per-side minimums a caller states, naming every requested side and its minimum, the
    foreground groups found, and ``remedy``, the caller's own sentence saying what to add.
    """
    needed = sum(minimums.values())
    if foreground_groups >= needed:
        return
    sides = ", ".join(f"{name}={n}" for name, n in sorted(minimums.items()))
    raise ValueError(
        f"the draw holds {foreground_groups} foreground group(s), fewer than the {needed} the "
        f"requested sides need ({sides}): {remedy}"
    )


def member_identity(date: str | None, stem: str) -> str:
    """A draw's member identity for one image: ``<date>/<stem>``, or the bare ``stem`` under a
    flat, dateless tree. A stem is unique only within one capture date, so a draw spanning more
    than one date keys members this way, and so must an agent-supplied ``group_key_map``.
    """
    return f"{date}/{stem}" if date else stem


def recorded_group_by(group_by: str, group_key_map: Mapping[str, str] | None) -> str:
    """The grouping policy a draw records: ``"explicit_map"`` when a group key map overrides
    ``group_by``, else ``group_by`` itself."""
    return "explicit_map" if group_key_map else group_by


def resolve_group_key_fn(
    group_by: str, stems: Sequence[str], *, group_key_map: dict[str, str] | None = None,
) -> Callable[[str], str]:
    """Resolve a grouping policy to a callable. An unrecognized ``group_by`` string, or a
    ``group_key_map`` missing coverage for some of ``stems``, raises.
    """
    if group_key_map is not None:
        missing = sorted(s for s in stems if s not in group_key_map)
        if missing:
            preview = missing[:10]
            more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            raise ValueError(f"group_key_map is missing {len(missing)} stem(s): {preview}{more}")
        return lambda s: group_key_map[s]
    if group_by not in GROUP_KEY_FNS:
        raise ValueError(
            f"Unrecognized group_by {group_by!r}; must be one of {sorted(GROUP_KEY_FNS)}, "
            "or supply group_key_map."
        )
    return GROUP_KEY_FNS[group_by]


def recorded_group_key_fn(
    group_by: str, *, date: str | None, stems: Sequence[str] = (),
    group_key_map: dict[str, str] | None = None,
) -> Callable[[str], str]:
    """The group key a draw records for a bare stem admitted out of one capture date's directory:
    the stem becomes its member identity (:func:`member_identity`) and the policy is resolved
    against those identities (:func:`resolve_group_key_fn`).

    ``stems`` and ``group_key_map`` are the producer's own: the map is checked for coverage of
    those stems' identities before any key is handed out. A reader passes neither, and the named
    policy alone answers.
    """
    identities = [member_identity(date, stem) for stem in stems]
    key_fn = resolve_group_key_fn(group_by, identities, group_key_map=group_key_map)
    return lambda stem: key_fn(member_identity(date, stem))


def draw_train_val(
    stems: Sequence[str], *, annotation_counts: dict[str, int] | None,
    group_key_fn: Callable[[str], str], val_ratio: float, seed: int,
) -> tuple[list[str], list[str]]:
    """``(train, val)`` for ``stems``: one :func:`group_balanced_split` at ``(1 - val_ratio,
    val_ratio, 0.0)``; never retries or degrades on a starved side.
    """
    parts = group_balanced_split(
        list(stems), annotation_counts=annotation_counts, group_key_fn=group_key_fn,
        splits=(1.0 - val_ratio, val_ratio, 0.0), seed=seed,
    )
    return parts["train"], parts["val"]


def redraw_pool(selection: "Selection") -> tuple[dict[str, str], dict[str, int]]:
    """A redraw's pool: each train-plus-val sample's recorded group key and foreground count, by
    sample identity."""
    from tcip_mcp.pipelines.data.label_queries import foreground_counts

    pool = selection.trainable()
    return ({s.identity: s.group for s in pool},
            foreground_counts({s.identity: s for s in pool}, selection.scope))


def redraw_starved_issue(
    group_of: dict[str, str], counts: dict[str, int], *, selection_dir: str | None,
    seed: int | None,
) -> str | None:
    """Whether a selection's train-plus-val members resolve to too few foreground groups for a
    redraw to populate both a train and a val side.

    :func:`group_balanced_split`'s own per-side minimum needs one foreground group for each active
    side; a background-only group (zero foreground signal, e.g. a confirmed negative) is placed
    afterwards by tile deficit and can concentrate entirely onto one side, so only foreground
    groups count here.

    ``group_of`` and ``counts`` are the selection's :func:`redraw_pool`.

    ``None`` when at least two foreground groups are available; the refusal otherwise, naming the
    selection, the seed and the two counts, with the two remedies: drop the redraw to bind the
    selection's recorded partition instead, or draw a selection with at least two foreground groups
    across train and val.
    """
    distinct = set(group_of.values())
    foreground = foreground_group_count(group_of, counts, group_of.__getitem__)
    if foreground >= 2:
        return None
    name = f"the selection at {selection_dir!r}" if selection_dir else "the selection"
    return (
        f"redrawing train and val inside {name}'s own members at seed {seed} would starve a "
        f"side: they resolve to only {foreground} foreground group(s) among "
        f"{len(distinct)} distinct group(s), short of the two a train side and a val side each "
        "need at least one of. Drop data.split.redraw_within_selection and data.split.seed to "
        "bind the selection's recorded partition instead, or draw a selection with at least two "
        "foreground groups across train and val."
    )


def same_directory(a: str | Path | None, b: str | Path | None) -> bool:
    """Whether ``a`` and ``b`` name one path, compared through ``tcip_store.canonical_path`` so
    a trailing separator, forward slashes, a relative spelling, a link or a case variant reads as
    the same path. ``False`` when either side is empty.
    """
    if not a or not b:
        return False
    return canonical_path(a) == canonical_path(b)


# -- spatial (within-image) strip split ---------------------------------------

_SPATIAL_IDENTITY_SEP = "::"


def spatial_strip_identity(stem: str, region_label: str) -> str:
    """A spatial split's per-region membership identity for one tile's source stem.

    ``region_label`` names the contiguous pixel-space strip a tile fell into (e.g.
    ``"strip_x_2"``). A selection that groups by this instead of the bare stem never reads a
    within-image split as the same stem appearing on more than one side:
    :func:`stem_of_spatial_identity` is the one place that identity is parsed back.
    """
    return f"{stem}{_SPATIAL_IDENTITY_SEP}{region_label}"


def stem_of_spatial_identity(identity: str) -> str:
    """The bare stem inside a :func:`spatial_strip_identity` string, for a leak check that only
    cares which source image a split member came from, not which region. An identity with no
    separator (not one this module produced) is returned unchanged."""
    idx = identity.rfind(_SPATIAL_IDENTITY_SEP)
    return identity[:idx] if idx != -1 else identity


@dataclass(frozen=True)
class SpatialStripSplit:
    """A within-image train/val(/test) split: the image partitioned into contiguous pixel-space
    strips along one axis, each strip assigned whole to one side, with a buffer band excluded at
    every boundary between differently-assigned strips.

    At ``stripes_per_split=1`` (the default) each side is exactly one contiguous region. Regions
    are ordered largest-share-first from the axis center outward (:func:`_center_out_order`), and
    each boundary is shrunk from one side only (enough on its own to guarantee ``>= buffer``
    separation), so the image-edge-facing side of the outermost two regions is never shrunk.
    Raising ``stripes_per_split`` asks for that many pieces per side, but the largest share's
    pieces sort adjacent and merge back into one region, so it is the smaller sides that end up
    split into pieces flanking it: three names of distinct shares, each cut into three pieces at
    ``stripes_per_split=3``, land in the order ``C, B, B, A, A, A, B, C, C`` (``A`` the largest
    share, ``C`` the smallest), five merged regions once adjacent same-name pieces combine.
    ``discard_ceiling`` caps how many pieces actually get used.

    ``regions`` maps each split name to its list of half-open pixel rects (already merged where two
    same-split strips landed adjacent, and buffer-shrunk on any side bordering a different-split
    neighbor): :class:`TiledDetectionDataset`'s ``keep_regions`` consumes these directly.
    ``realized_fractions`` is each side's kept tile count over the total kept across every side
    (post-buffer), not the requested fractions.
    """

    width: int
    height: int
    tile_size: int
    overlap: float
    stride: int
    axis: str
    buffer: int
    split_names: tuple[str, ...]
    requested_fractions: tuple[float, ...]
    stripes_per_split: int
    discard_ceiling: float
    regions: dict[str, list[tuple[int, int, int, int]]]
    region_bounds: list[tuple[str, int, int]]
    total_tiles: int
    tiles_dropped_past_extent: int
    tiles_dropped_outside_regions: int
    kept_tiles: dict[str, int]
    realized_fractions: dict[str, float]
    realized_discard_fraction: float

    def _region_index_for(self, tile_x: int, tile_y: int) -> int | None:
        """Index into ``region_bounds`` for a kept tile at ``(tile_x, tile_y)``, or ``None`` when
        this position falls in a dropped gap (buffer band or past-extent) rather than fully inside
        any region.
        """
        pos = tile_x if self.axis == "x" else tile_y
        starts = [start for _, start, _ in self.region_bounds]
        idx = bisect.bisect_right(starts, pos) - 1
        if idx < 0:
            return None
        _, start, end = self.region_bounds[idx]
        return idx if (start <= pos and pos + self.tile_size <= end) else None

    def split_name_for(self, tile_x: int, tile_y: int) -> str | None:
        """Which split (``"train"``/``"val"``/``"test"``/...) a kept tile at ``(tile_x,
        tile_y)`` belongs to, or ``None`` when it falls in a dropped gap."""
        idx = self._region_index_for(tile_x, tile_y)
        return None if idx is None else self.region_bounds[idx][0]

    def identity_for(self, stem: str, tile_x: int, tile_y: int) -> str | None:
        """The region identity for a kept tile at ``(tile_x, tile_y)``, or ``None`` when
        this position falls in a dropped gap (buffer band or past-extent) rather than fully
        inside any region."""
        idx = self._region_index_for(tile_x, tile_y)
        if idx is None:
            return None
        return spatial_strip_identity(stem, f"strip_{self.axis}_{idx}")


def _center_out_order(slots: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Order slots by descending share, largest first, then placed axis-center-out: each next
    (smaller) slot alternately extends the left or right end of the growing arrangement, so every
    slot but the largest faces at most one differently-assigned neighbor. A tie among equal shares
    resolves in the order the slots were given, never by a seed.
    """
    indexed = list(enumerate(slots))
    indexed.sort(key=lambda p: (-p[1][1], p[0]))
    ordered = [item for _, item in indexed]
    left: list[tuple[str, float]] = []
    right: list[tuple[str, float]] = []
    for i, item in enumerate(ordered):
        (right if i % 2 == 0 else left).append(item)
    return list(reversed(left)) + right


def _strip_regions(
    positions: list[int], tile_size: int, buffer: int,
    split_names: tuple[str, ...], fractions: tuple[float, ...],
    discard_ceiling: float, stripes_per_split: int,
) -> list[tuple[str, int, int]]:
    """Merged, buffer-shrunk ``(name, start, end)`` pixel regions along one axis, in axis order,
    cut and shrunk in the discrete tile-origin lattice rather than continuous pixel space: a region
    with positive pixel width could otherwise miss the stride-spaced lattice entirely and contain
    zero real tile origins.

    ``stripes_per_split`` sets how many pieces a side is cut into before adjacent same-name pieces
    merge (capped by ``discard_ceiling``, the maximum share of the axis a buffer band between
    differing sides may consume); the fraction each side targets sets its total share of the axis.
    The piece order is :func:`_center_out_order`'s; see :class:`SpatialStripSplit`.
    """
    n = len(positions)
    axis_span = positions[-1] + tile_size - positions[0]
    n_splits = len(split_names)
    max_stripes = max(1, int(discard_ceiling * axis_span / max(1, n_splits * buffer)))
    stripes = max(1, min(stripes_per_split, max_stripes))

    slots: list[tuple[str, float]] = []
    for name, frac in zip(split_names, fractions):
        slots.extend([(name, frac / stripes)] * stripes)
    slots = _center_out_order(slots)

    raw: list[tuple[str, int, int]] = []
    cursor = 0.0
    for i, (name, share) in enumerate(slots):
        end_f = n if i == len(slots) - 1 else cursor + share * n
        start_idx, end_idx = int(round(cursor)), max(int(round(end_f)), int(round(cursor)))
        raw.append((name, start_idx, end_idx))
        cursor = end_f

    merged: list[tuple[str, int, int]] = []
    for name, start_idx, end_idx in raw:
        if merged and merged[-1][0] == name:
            merged[-1] = (name, merged[-1][1], end_idx)
        else:
            merged.append((name, start_idx, end_idx))
    merged = [(name, s, e) for name, s, e in merged if e > s]

    # A boundary is shrunk from one side only (the higher-index region's left edge, against
    # its neighbor's raw end): that alone already guarantees >= buffer separation.
    shrunk: list[tuple[str, int, int]] = []
    for i, (name, s, e) in enumerate(merged):
        if i > 0 and merged[i - 1][0] != name:
            neighbor_end_pixel = positions[merged[i - 1][2] - 1] + tile_size
            while s < e and positions[s] < neighbor_end_pixel + buffer:
                s += 1
        if e > s:
            shrunk.append((name, s, e))

    return [(name, positions[s], positions[e - 1] + tile_size) for name, s, e in shrunk]


def spatial_strip_split(
    width: int, height: int, tile_size: int, overlap: float, *,
    fractions: tuple[float, ...],
    split_names: tuple[str, ...] = ("train", "val", "test"),
    buffer: int | None = None, discard_ceiling: float = 0.05, stripes_per_split: int = 1,
) -> SpatialStripSplit:
    """Split one image's own tile lattice into disjoint pixel-space strips, one side per name, for
    the case where there are too few source images to hold one out whole: a strip is train, val, or
    test instead of a stem.

    The tile lattice comes from :func:`~tcip_mcp.pipelines.data.tiling.tile_positions` at the
    training stride, so the regions this returns tile the same grid a
    :class:`TiledDetectionDataset` built at this ``tile_size``/``overlap`` will index. The split
    runs along whichever axis (width or height) offers more distinct tile positions. Piece count
    and order follow :class:`SpatialStripSplit` (``stripes_per_split``, capped by
    ``discard_ceiling``).

    ``buffer`` (pixels) is the minimum gap kept around every boundary between two
    differently-assigned strips: an explicit value below ``tile_size`` is refused. Omitted, it
    defaults to ``tile_size``.

    ``fractions`` must be non-negative and sum to 1.0, matching ``split_names`` in length; a zero
    fraction drops that name from the split entirely (fewer than two non-zero fractions is
    refused). Raises ``ValueError`` when no tile fits fully inside the image extent at this
    ``tile_size``, or when the derived strip layout leaves any requested, non-zero-fraction side
    with zero kept tiles.
    """
    from tcip_mcp.pipelines.data.tiling import compute_stride, tile_positions, tile_within_extent

    if len(fractions) != len(split_names):
        raise ValueError(
            f"fractions ({len(fractions)}) and split_names ({len(split_names)}) must be the "
            "same length."
        )
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {fractions}.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions} (sum={sum(fractions)}).")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}.")
    if buffer is None:
        buffer = tile_size
    elif buffer < tile_size:
        raise ValueError(
            f"buffer ({buffer}) must be at least tile_size ({tile_size}): a smaller buffer "
            "cannot guarantee a kept tile on one side never shares pixels or immediate context "
            "with a kept tile on another, including under overlap > 0."
        )

    active_names = tuple(n for n, f in zip(split_names, fractions) if f > 0)
    active_fracs = tuple(f for f in fractions if f > 0)
    if len(active_names) < 2:
        raise ValueError(
            f"at least two non-zero fractions are needed for a spatial split, got {fractions}."
        )

    stride = compute_stride(tile_size, overlap)
    lattice = tile_positions(height, width, tile_size, stride)
    total_tiles = len(lattice)
    if total_tiles == 0:
        raise ValueError(f"no tile position fits a {width}x{height} image at tile_size={tile_size}.")

    in_extent = [(tx, ty) for tx, ty in lattice
                 if tile_within_extent(tx, ty, tile_size, width, height)]
    tiles_dropped_past_extent = total_tiles - len(in_extent)
    if not in_extent:
        raise ValueError(
            f"no tile fits fully inside the {width}x{height} image extent at tile_size="
            f"{tile_size} (every tile position needs edge padding); a spatial split needs at "
            "least one fully-real tile to assign."
        )

    xs = sorted({tx for tx, _ in in_extent})
    ys = sorted({ty for _, ty in in_extent})
    axis = "x" if len(xs) >= len(ys) else "y"
    positions = xs if axis == "x" else ys

    region_bounds = _strip_regions(
        positions, tile_size, buffer, active_names, active_fracs, discard_ceiling,
        stripes_per_split,
    )
    if len({name for name, _, _ in region_bounds}) < len(active_names):
        raise ValueError(
            f"no strip layout at buffer={buffer} leaves every requested split {active_names} "
            f"with a non-empty region on a {width}x{height} image at tile_size={tile_size}; "
            "try a smaller buffer, fewer stripes_per_split, fewer splits, or a larger image."
        )

    kept: dict[str, int] = {name: 0 for name in active_names}
    dropped_outside = 0
    starts = [start for _, start, _ in region_bounds]
    for tx, ty in in_extent:
        pos = tx if axis == "x" else ty
        idx = bisect.bisect_right(starts, pos) - 1
        name, start, end = region_bounds[idx] if idx >= 0 else (None, 0, 0)
        if name is not None and start <= pos and pos + tile_size <= end:
            kept[name] += 1
        else:
            dropped_outside += 1

    if any(kept[name] == 0 for name in active_names):
        raise ValueError(
            f"the derived strip layout leaves at least one requested split with zero kept "
            f"tiles on a {width}x{height} image at tile_size={tile_size}, buffer={buffer}: "
            f"kept={kept}. Try a smaller buffer, fewer stripes_per_split, or a larger image."
        )

    regions: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in active_names}
    for name, start, end in region_bounds:
        rect = (start, 0, end, height) if axis == "x" else (0, start, width, end)
        regions[name].append(rect)

    total_kept = sum(kept.values()) or 1
    tiles_within_extent = len(in_extent)
    return SpatialStripSplit(
        width=width, height=height, tile_size=tile_size, overlap=overlap, stride=stride,
        axis=axis, buffer=buffer, split_names=split_names,
        requested_fractions=fractions, stripes_per_split=stripes_per_split,
        discard_ceiling=discard_ceiling, regions=regions, region_bounds=region_bounds,
        total_tiles=total_tiles, tiles_dropped_past_extent=tiles_dropped_past_extent,
        tiles_dropped_outside_regions=dropped_outside,
        kept_tiles=kept, realized_fractions={n: kept[n] / total_kept for n in active_names},
        realized_discard_fraction=dropped_outside / tiles_within_extent,
    )




def cal_holdout_split(
    stems: Sequence[str],
    annotation_counts: dict[str, int] | None = None,
    group_key_fn: Callable[[str], str] | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_CAL_SEED,
) -> dict[str, list[str]]:
    """A disjoint, group-coherent, annotation-balanced calibration/holdout split.

    A thin wrapper over :func:`group_balanced_split` (never reimplemented): its third split's
    fraction is always 0, and the resulting ``{"train", "val"}`` parts are the two halves of
    whatever universe ``stems`` holds, remapped to ``{"calibration", "holdout"}`` for the
    calibration callers.
    """
    parts = group_balanced_split(
        stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
        splits=(1.0 - holdout_ratio, holdout_ratio, 0.0), seed=seed,
    )
    return {"calibration": parts["train"], "holdout": parts["val"]}


_LOCK_DIR = (".tcip", "artifacts")
_LOCK_STEM = "cal_holdout_split_"


@dataclass(frozen=True)
class _CalHoldoutLockLocator:
    """One locked split per dataset identity, named for the identity it locks."""

    def relative_path(self, scope: str, parts: tuple[str, ...]) -> "PurePosixPath":
        (identity_hash,) = parts
        return PurePosixPath(*_LOCK_DIR, f"{_LOCK_STEM}{identity_hash}.json")

    def parts_from(self, relative_path: "PurePosixPath") -> tuple[str, ...] | None:
        segments = relative_path.parts
        if segments[:len(_LOCK_DIR)] != _LOCK_DIR or len(segments) != len(_LOCK_DIR) + 1:
            return None
        name = segments[-1]
        if not name.startswith(_LOCK_STEM) or not name.endswith(".json"):
            return None
        return (name[len(_LOCK_STEM):-len(".json")],)


CAL_HOLDOUT_LOCK_STORE = "cal_holdout_split_lock"
register_store(
    StoreDescriptor(
        name=CAL_HOLDOUT_LOCK_STORE,
        kind="record",
        key_fields=("identity_hash",),
        frozen=True,
        codec=RECORD_JSON,
        concurrency="last_writer_wins",
        locator=_CalHoldoutLockLocator(),
        enumerable=True,
    )
)


def cal_holdout_lock_key(identity_hash: str, *, scope_root: str | Path) -> Key:
    """A dataset identity's locked calibration/holdout split, under the root it was drawn over.

    ``scope_root`` is required: the lock travels with the dataset whose images it held back.
    ``last_writer_wins``.
    """
    if PureWindowsPath(identity_hash).name != identity_hash or identity_hash == "..":
        raise BadKey(
            f"dataset identity {identity_hash!r} is not a single name: an identity carrying a "
            "path separator would address a lock outside the artifact store"
        )
    return Key(CAL_HOLDOUT_LOCK_STORE, str(Path(scope_root).resolve()), (identity_hash,))


def cal_holdout_lock_path(identity_hash: str, *, scope_root: str | Path) -> Path:
    """Where a dataset identity's locked cal/holdout split lives on disk under ``scope_root``,
    placed by the store's own locator.
    """
    key = cal_holdout_lock_key(identity_hash, scope_root=scope_root)
    return Path(key.root, *_CalHoldoutLockLocator().relative_path(key.root, key.parts).parts)


def cal_holdout_scope_root(labels_dir: str | Path) -> Path:
    """The root a labeled directory's locked cal/holdout split is scoped to: the dataset root the
    labels live under, or, for a directory the dataset layout cannot place, the directory itself.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    root = dataset_root_of(labels_dir)
    return (root if root is not None else Path(labels_dir)).resolve()


def label_image_stems(
    labels_dir: str | Path, images_dir: str | Path | None = None,
) -> tuple[list[str], dict[str, "Path | BandGroupRef"]]:
    """Stems with a readable per-image label file, over a whole labeled directory.

    With ``images_dir`` omitted, returns every stem with a label file (``stem_to_image`` empty),
    the label-only universe. With ``images_dir`` given, only stems that also have a matching
    logical image (a plain file, or a ``.bandgroup``-grouped capture) survive. ``labels_dir`` may
    itself be a prediction bucket, so its own provenance sidecars are excluded through
    :func:`~tcip_annotation.json_io.prediction_documents`. A selection-restricted door reads
    :func:`selection_calibration_universe` instead.
    """
    from tcip_annotation.json_io import prediction_documents

    labels_p = Path(labels_dir)
    label_stems = {p.stem for p in prediction_documents(labels_p)}
    if images_dir is None:
        return sorted(label_stems), {}
    from tcip_mcp.pipelines.image_utils import list_logical_images

    stem_to_image = {stem: src for stem, src in list_logical_images(images_dir).items()
                     if stem in label_stems}
    return sorted(stem_to_image), stem_to_image


def selection_calibration_universe(
    selection: "Selection", labels_dir: str | Path,
    *, min_foreground_groups: dict[str, int] | None = None,
) -> tuple[list[str], str, dict[str, str], dict[str, list[str]], dict[str, int],
           dict[str, "Sample"]]:
    """The calibration universe a selection gives one caller restricting a read to ``labels_dir``:
    the selection's ``calibration`` samples whose own recorded ground truth lives in
    ``labels_dir``, whatever shape that ground truth is. Nothing is intersected against a directory
    listing.

    The floor counts only the groups that carry foreground, through
    :func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts` under the selection's own
    class space. ``min_foreground_groups`` is forwarded to
    :func:`refuse_insufficient_foreground_groups`; omitted, it defaults to ``{"calibration": 2}``,
    since a locked cal/holdout draw halves the universe into two non-empty parts.

    Returns ``(stems, group_by, group_key_map, excluded, counts, samples)``: ``stems`` is the calibration
    side's bare member names under ``labels_dir``; ``group_key_map`` is each one's recorded group
    key, with ``group_by="explicit_map"``; ``excluded`` names the selection's recorded train
    members (``excluded_training_stems``) and val members (``excluded_validation_stems``) under
    this scope; ``counts`` is each member's foreground count, the one the floor read;
    ``samples`` is each universe member's own recorded sample.

    Refuses a calibration sample naming a pixel rect
    (:func:`~tcip_mcp.pipelines.data.selection.refuse_unreadable_samples`), and, naming the count,
    the labels directory and the floor, a universe holding fewer foreground groups than the floor
    states.
    """
    from tcip_mcp.pipelines.data.label_queries import foreground_counts
    from tcip_mcp.pipelines.data.selection import refuse_unreadable_samples

    scope = selection.scope
    # Through the sample's own recorded scope, never the parent of its ground-truth path: a table
    # scope is the table itself, which no sample's parent directory equals.
    in_scope = [s for s in selection.samples if same_directory(s.ground_truth_scope, labels_dir)]
    by_side: dict[str, dict[str, str]] = {name: {} for name in SIDES}
    universe_samples: dict[str, "Sample"] = {}
    for sample in in_scope:
        by_side[sample.side][sample.member] = sample.group
        if sample.side == "calibration":
            universe_samples[sample.member] = sample
    refuse_unreadable_samples(universe_samples.values())

    stems = sorted(universe_samples)
    group_key_map = {stem: by_side["calibration"][stem] for stem in stems}
    excluded = {
        "excluded_training_stems": sorted(by_side["train"]),
        "excluded_validation_stems": sorted(by_side["val"]),
    }

    counts = foreground_counts(universe_samples, scope)
    n_groups = foreground_group_count(stems, counts, group_key_map.__getitem__)
    floor = min_foreground_groups if min_foreground_groups is not None else {"calibration": 2}
    try:
        refuse_insufficient_foreground_groups(
            n_groups, floor,
            remedy="annotate or confirm more foreground groups of this subject.")
    except ValueError as exc:
        raise ValueError(
            f"the selection's calibration side under {labels_dir} gives a calibration universe "
            f"of {n_groups} foreground group(s) ({len(stems)} member(s) total): {exc} Draw the "
            "selection again with a larger calibration_ratio or more foreground groups under "
            "this ground truth."
        ) from exc
    return stems, "explicit_map", group_key_map, excluded, counts, universe_samples


def _split_content_hash(parts: dict[str, list[str]] | None) -> str | None:
    """Content hash over a split's calibration+holdout membership (order-independent per side)."""
    if not parts:
        return None
    h = hashlib.sha256()
    for key in ("calibration", "holdout"):
        for s in sorted(parts[key]):
            h.update(s.encode("utf-8"))
            h.update(b"\0")
        h.update(b"\0\0")
    return h.hexdigest()[:16]


def selection_policy_conflict(selection_dir: str | Path | None, group_by: str | None,
                              group_key_map: dict[str, str] | None) -> str | None:
    """Why a grouping policy cannot be stated beside ``selection_dir`` for a locked draw, or
    ``None`` when nothing conflicts."""
    if selection_dir is None or (group_by is None and group_key_map is None):
        return None
    return (f"selection_dir={str(selection_dir)!r} conflicts with group_by/group_key_map: the "
            "group keys the selection recorded on its own samples govern the locked draw; pass "
            "neither beside it.")


def resolve_locked_cal_holdout_split(
    stems: Sequence[str] | None,
    *,
    identity_hash: str,
    scope_root: str | Path,
    annotation_counts: dict[str, int] | None = None,
    group_by: str | None = None,
    group_key_map: dict[str, str] | None = None,
    holdout_ratio: float = DEFAULT_HOLDOUT_RATIO,
    seed: int = DEFAULT_CAL_SEED,
    force_redraw: bool = False,
    timestamp: str | None = None,
    selection_dir: str | None = None,
    reason: str | None = None,
) -> dict:
    """Resolve (and lock) the calibration/holdout split for one dataset identity.

    The split locks on its first draw for a given ``identity_hash``: every later call for the same
    identity returns the identical split unless the caller passes ``force_redraw=True`` (the
    ``redraw_calibration_holdout`` MCP tool). The grouping policy is resolved via
    :func:`resolve_group_key_fn` first, so a malformed ``group_by``/``group_key_map`` raises; an
    unstated ``group_by`` is :data:`DEFAULT_GROUP_BY`.

    If a lock already exists and the caller's declared policy
    (``group_by``/``group_key_map``/``seed``/``holdout_ratio``/``selection_dir``) differs from what
    is recorded in it, the divergence is logged as a warning and returned under
    ``"policy_divergence"`` (``{"requested": ..., "locked": ...}``) and the locked split is
    returned unchanged. Stems the caller has that the lock doesn't cover are returned under
    ``"unlocked_stems"``.

    ``selection_dir`` names the selection a caller drew ``stems`` from (``None`` for a
    whole-directory draw); it is recorded in the lock and in every ``redraw_history`` entry, never
    resolved or compared here.

    ``stems`` ``None`` names the existing lock's own members as the universe; with no lock to read
    them from it raises ``ValueError``.

    A locked stem with no corresponding entry in the caller's current ``stems`` raises
    ``ValueError``. A lock file that exists but fails to parse raises when ``force_redraw=False``
    or when ``stems`` is ``None``; ``force_redraw=True`` over stems the caller holds proceeds past
    it, without the unreadable redraw history.

    ``scope_root`` is required: the root the lock is stored under (:func:`cal_holdout_scope_root`).
    ``timestamp`` is the caller's, and only meaningful when a new draw happens.

    Every draw that writes a lock, a first draw or a forced redraw, leaves its one audit line,
    ``calibration_holdout_drawn``, naming the policy, the membership before and after and the
    caller's ``reason``; returning an existing lock writes nothing and leaves none.

    Returns the full locked-split dict: ``{identity_hash, calibration, holdout, group_by,
    group_key_map, seed, holdout_ratio, selection_dir, redraw_history}``, plus the optional
    ``policy_divergence`` / ``unlocked_stems`` report fields above when a lock already existed, or,
    on a draw, the ``old_membership`` it replaced (``None`` for a first draw).
    """
    lock_key = cal_holdout_lock_key(identity_hash, scope_root=scope_root)
    try:
        existing = store.read(lock_key, default=None)
    except DecodeError as exc:
        if not force_redraw or stems is None:
            raise ValueError(
                f"the cal/holdout lock for identity_hash={identity_hash!r} exists but could not "
                f"be read/parsed ({exc}). Refusing to silently treat a corrupt lock as 'no lock "
                "exists' and redraw. Investigate the file, or redraw it with "
                "redraw_calibration_holdout over the labels its stems come from."
            ) from exc
        logger.warning(
            "the cal/holdout lock for identity_hash=%s is corrupt (%s); force_redraw=True "
            "proceeds to draw a fresh lock. Its prior "
            "redraw history could not be recovered from the unreadable record.",
            identity_hash, exc,
        )
        existing = None
    if stems is None:
        if existing is None:
            raise ValueError(f"no lock exists for identity_hash={identity_hash!r} to take the "
                             "redraw's stems from; name the labels its stems come from.")
        stems = sorted(set(existing["calibration"]) | set(existing["holdout"]))
    group_by = group_by or DEFAULT_GROUP_BY
    group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    declared_policy = {
        "group_by": group_by, "group_key_map": group_key_map,
        "seed": seed, "holdout_ratio": holdout_ratio,
        "selection_dir": selection_dir,
    }

    if existing is not None and not force_redraw:
        locked_stems = set(existing["calibration"]) | set(existing["holdout"])
        stems_set = set(stems)
        stale = sorted(locked_stems - stems_set)
        if stale:
            preview = stale[:10]
            more = f" (+{len(stale) - 10} more)" if len(stale) > 10 else ""
            raise ValueError(
                f"locked cal/holdout split for identity_hash={identity_hash!r} references "
                f"{len(stale)} stem(s) no longer present in the current data (image/label "
                f"deleted or renamed since the split was locked): {preview}{more}. Use "
                "redraw_calibration_holdout to redraw deliberately, or restore the missing "
                "file(s)."
            )
        result = dict(existing)
        unlocked_stems = sorted(stems_set - locked_stems)
        if unlocked_stems:
            result["unlocked_stems"] = unlocked_stems
        recorded_policy = {k: existing[k] for k in declared_policy}
        if recorded_policy != declared_policy:
            logger.warning(
                "cal/holdout split for identity_hash=%s is locked with a different policy than "
                "declared (locked=%s, declared=%s); returning the locked split unchanged. Use "
                "redraw_calibration_holdout to redraw deliberately.",
                identity_hash, recorded_policy, declared_policy,
            )
            result["policy_divergence"] = {"requested": declared_policy, "locked": recorded_policy}
        return result

    parts = cal_holdout_split(stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
                              holdout_ratio=holdout_ratio, seed=seed)
    redraw_history = list(existing["redraw_history"]) if existing else []
    redraw_history.append({
        "policy": declared_policy,
        "seed": seed,
        "old_content_hash": _split_content_hash(existing),
        "new_content_hash": _split_content_hash(parts),
        "timestamp": timestamp,
    })
    locked = {
        "identity_hash": identity_hash,
        "calibration": parts["calibration"],
        "holdout": parts["holdout"],
        **declared_policy,
        "redraw_history": redraw_history,
    }
    store.replace(lock_key, locked)
    from tcip_mcp.audit import dataset_scope_of, record_event_or_raise

    old_membership = ({"calibration": existing["calibration"], "holdout": existing["holdout"]}
                      if existing else None)
    # The draw's one receipt, a first draw and a redraw alike; the lock has already landed.
    record_event_or_raise(
        "calibration_holdout_drawn",
        {"identity_hash": identity_hash, **declared_policy, "reason": reason},
        scope=dataset_scope_of(str(scope_root)), old_membership=old_membership,
        new_membership={"calibration": parts["calibration"], "holdout": parts["holdout"]},
    )
    return {**locked, "old_membership": old_membership}

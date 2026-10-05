"""Group-aware, annotation-stratified train/val/calibration splitting: group-coherent (sibling
tiles of one source never straddle two splits), annotation-balanced, deterministic in seed.
Torch-free.
"""

from __future__ import annotations

import bisect
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

from tcip_store import canonical_path

if TYPE_CHECKING:
    from tcip_annotation.json_io import LabelDocument
    from tcip_mcp.pipelines.data.selection import ClassScope

# A tiled stem looks like ``<source>_<x>_<y>`` (two trailing integer fields).
# Strip that suffix so all tiles of one source share a group key. A stem with a
# single trailing ``_<int>`` (e.g. ``img_001``) does not match and falls back to
# the full stem.
_TILE_GROUP_RE = re.compile(r"^(.*)_\d+_\d+$")

# The grouping policy every split door applies when its config states none.
DEFAULT_GROUP_BY = "tile_prefix"

DEFAULT_SHARES = {"val": 0.15, "calibration": 0.10, "holdout": 0.05}
"""Provisional: the owner's documented default share of each side beside ``train``, which takes
the remainder (0.70); soft targets the group draw rounds to whole groups. A door states any of
them to replace it, a share of zero drawing no such side."""


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


def count_label_lines(document: "LabelDocument", scope: "ClassScope") -> int:
    """How many instances of ``scope``'s subject (of any subject when it names none) one per-image
    label ``document`` carries."""
    from tcip_annotation import json_io
    from tcip_annotation.state import instances

    # A crowd region is never one object, so it is never counted as one.
    return sum(1 for a in instances(document.annotations) if scope.subject is None
               or json_io.attribute_ids(a, scope.subject, ()) is not None)


def label_document_extent(document: "LabelDocument", where: str) -> tuple[int, int]:
    """``(width, height)`` one per-image label ``document`` records, the frame its geometry was
    authored against. A document that states no positive width and height refuses
    (``ValueError``) naming ``where``.
    """
    w, h = document.width, document.height
    if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
        raise ValueError(
            f"{where} states no positive width and height ({w!r}, {h!r}): the frame its "
            "geometry was drawn in is unknown, so nothing can place that geometry on the image. "
            "Write the label document through the annotation store, which records the frame.")
    return w, h


def group_balanced_split(
    stems: Sequence[str], *, counts: Mapping[str, int], weighted: bool,
    group_key_fn: Callable[[str], str], splits: Mapping[str, float], seed: int,
    min_foreground_groups: Mapping[str, int],
) -> dict[str, list[str]]:
    """Partition ``stems`` onto the sides ``splits`` names, each at its positive fraction, keeping
    each group (``group_key_fn``) whole; deterministic in ``seed`` and the groups, whatever the
    stems are named.

    ``counts`` is every stem's measured foreground count, and a stem it does not name refuses
    (``ValueError``). Each side first takes ``min_foreground_groups`` groups carrying foreground,
    smallest first, without raising on a short tree
    (:func:`refuse_insufficient_foreground_groups` is the floor). The rest is balanced by a
    blended foreground/tile imbalance when ``weighted`` and any group carries foreground, the
    groups carrying none then filling the largest tile deficit; otherwise by tile count alone.
    Returns each side's sorted stems.
    """
    stems = list(stems)
    missing = sorted(s for s in stems if s not in counts)
    if missing:
        raise ValueError(f"no foreground count for {len(missing)} member(s) {missing[:10]}: a "
                         "draw balances only on counts measured for every member")
    fracs = dict(splits)
    active = list(fracs)

    # Group stems and tally tiles + annotations per group.
    groups: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        groups[group_key_fn(s)].append(s)
    group_tiles = {gk: len(gs) for gk, gs in groups.items()}
    group_fg = {gk: sum(counts[s] for s in gs) for gk, gs in groups.items()}

    if weighted and any(group_fg.values()):
        group_ann = group_fg
        fg_groups = [gk for gk in sorted(groups) if group_ann[gk] > 0]
        bg_groups = [gk for gk in sorted(groups) if group_ann[gk] == 0]
    else:
        group_ann = dict(group_tiles)
        fg_groups = sorted(groups)
        bg_groups = []

    result: dict[str, list[str]] = {n: [] for n in fracs}
    if not fg_groups:
        return result

    total_ann = sum(group_ann[gk] for gk in fg_groups) or 1
    total_fg_tiles = sum(group_tiles[gk] for gk in fg_groups) or 1
    targets_ann = {n: fracs[n] * total_ann for n in fracs}
    targets_tiles = {n: fracs[n] * total_fg_tiles for n in fracs}

    state_ann = {n: 0 for n in fracs}
    state_tiles = {n: 0 for n in fracs}
    assignment: dict[str, str] = {}
    used: set[str] = set()

    rng = random.Random(seed)
    rng.shuffle(fg_groups)

    # Each side's minimum, met first with the smallest foreground groups (dense ones stay for the
    # balancing pass); a short tree just meets fewer minimums, nothing here raises on it.
    min_pass_pool = [gk for gk in fg_groups if group_fg[gk] > 0]
    fg_smallest_first = sorted(min_pass_pool, key=lambda gk: group_ann[gk])
    for split_name, need in min_foreground_groups.items():
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
    tile_target_all = {n: fracs[n] * total_all_tiles for n in fracs}
    for gk in sorted(bg_groups, key=lambda g: group_tiles[g], reverse=True):
        best_split = max(active, key=lambda n: tile_target_all[n] - state_tiles[n])
        assignment[gk] = best_split
        state_tiles[best_split] += group_tiles[gk]

    for gk, gs in groups.items():
        result[assignment.get(gk, active[0])].extend(gs)
    return {n: sorted(result[n]) for n in fracs}


def foreground_group_count(
    members: Iterable[str], counts: Mapping[str, int], group_of: Callable[[str], str],
) -> int:
    """How many distinct groups among ``members`` carry foreground. ``counts`` is each member's own
    foreground count under the caller's own key for it
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`), naming every member.
    """
    return len({group_of(member) for member in members if counts[member] > 0})


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


def member_identity(capture: str, stem: str) -> str:
    """A draw's member identity for one image, ``<capture>/<stem>``, the key an agent-supplied
    ``group_key_map`` names it by."""
    return f"{capture}/{stem}"


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
    from tcip_mcp.pipelines.model_build import resolve_named

    return resolve_named(group_by, GROUP_KEY_FNS, kind="group_by policy")


def recorded_group_key_fn(
    group_by: str, *, date: str, stems: Sequence[str] = (),
    group_key_map: dict[str, str] | None = None,
) -> Callable[[str], str]:
    """The group key a draw records for a bare stem admitted out of capture ``date``'s directory:
    the stem becomes its member identity (:func:`member_identity`) and the policy is resolved
    against those identities (:func:`resolve_group_key_fn`).

    With ``stems``, ``group_key_map`` is checked for coverage of their identities before any key
    is handed out; with neither, the named policy alone answers.
    """
    identities = [member_identity(date, stem) for stem in stems]
    key_fn = resolve_group_key_fn(group_by, identities, group_key_map=group_key_map)
    return lambda stem: key_fn(member_identity(date, stem))


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
    """A spatial split's per-region membership identity for one tile's source stem, ``region_label``
    naming the contiguous pixel-space strip the tile fell into (e.g. ``"strip_x_2"``);
    :func:`stem_of_spatial_identity` parses it back."""
    return f"{stem}{_SPATIAL_IDENTITY_SEP}{region_label}"


def stem_of_spatial_identity(identity: str) -> str:
    """The bare stem inside a :func:`spatial_strip_identity` string, for a leak check that only
    cares which source image a split member came from, not which region. An identity with no
    separator (not one this module produced) is returned unchanged."""
    idx = identity.rfind(_SPATIAL_IDENTITY_SEP)
    return identity[:idx] if idx != -1 else identity


@dataclass(frozen=True)
class SpatialStripSplit:
    """A within-image split over named sides: the image partitioned into contiguous pixel-space
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
    neighbor). ``realized_fractions`` is each side's kept tile count over the total kept across every side
    (post-buffer), not the requested fractions.
    """

    width: int
    height: int
    tile_size: int
    overlap: float
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

    def identity_for(self, stem: str, box: tuple[int, int, int, int]) -> str | None:
        """The region identity for the lattice slice ``box`` of ``stem``, or ``None`` when it
        falls in a dropped gap."""
        idx = _region_index(self.region_bounds, self.axis, box)
        if idx is None:
            return None
        return spatial_strip_identity(stem, f"strip_{self.axis}_{idx}")


def _region_index(region_bounds: list[tuple[str, int, int]], axis: str,
                  box: tuple[int, int, int, int]) -> int | None:
    """Index into ``region_bounds`` of the region the slice ``box`` lies wholly inside along
    ``axis``, or ``None`` when it falls in a dropped gap (a buffer band or past the extent)."""
    lo, hi = (box[0], box[2]) if axis == "x" else (box[1], box[3])
    idx = bisect.bisect_right([start for _, start, _ in region_bounds], lo) - 1
    if idx < 0:
        return None
    _, start, end = region_bounds[idx]
    return idx if start <= lo and hi <= end else None


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
    spans: list[tuple[int, int]], buffer: int,
    split_names: tuple[str, ...], fractions: tuple[float, ...],
    discard_ceiling: float, stripes_per_split: int,
) -> list[tuple[str, int, int]]:
    """Merged, buffer-shrunk ``(name, start, end)`` pixel regions along one axis, in axis order,
    cut and shrunk over the lattice's own slice ``spans`` (each slice's ``(start, end)`` on the
    axis, in order) so every region holds whole slices.

    ``stripes_per_split`` sets how many pieces a side is cut into before adjacent same-name pieces
    merge (capped by ``discard_ceiling``, the maximum share of the axis a buffer band between
    differing sides may consume); the fraction each side targets sets its total share of the axis.
    The piece order is :func:`_center_out_order`'s; see :class:`SpatialStripSplit`.
    """
    n = len(spans)
    axis_span = spans[-1][1] - spans[0][0]
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
            neighbor_end_pixel = spans[merged[i - 1][2] - 1][1]
            while s < e and spans[s][0] < neighbor_end_pixel + buffer:
                s += 1
        if e > s:
            shrunk.append((name, s, e))

    return [(name, spans[s][0], spans[e - 1][1]) for name, s, e in shrunk]


def spatial_strip_split(
    width: int, height: int, tile_size: int, overlap: float, *,
    fractions: tuple[float, ...], split_names: tuple[str, ...],
    buffer: int | None = None, discard_ceiling: float = 0.05, stripes_per_split: int = 1,
) -> SpatialStripSplit:
    """Split one image's own tile lattice into disjoint pixel-space strips, one side per name, for
    the case where there are too few source images to hold one out whole: a strip is train, val, or
    holdout instead of a stem.

    The tile lattice is :func:`~tcip_mcp.pipelines.slicing.slice_lattice`'s at this
    ``tile_size``/``overlap``. The split runs along whichever axis (width or height) offers more distinct tile positions. Piece count
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
    from tcip_mcp.pipelines.slicing import is_full_slice, slice_lattice

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

    lattice = slice_lattice(height, width, tile_size, overlap)
    total_tiles = len(lattice)
    in_extent = [box for box in lattice if is_full_slice(box, tile_size)]
    tiles_dropped_past_extent = total_tiles - len(in_extent)
    if not in_extent:
        raise ValueError(
            f"no tile fits fully inside the {width}x{height} image extent at tile_size="
            f"{tile_size} (the frame is shorter than the tile on an axis); a spatial split needs "
            "at least one full tile to assign."
        )

    x_spans = sorted({(x0, x1) for x0, _y0, x1, _y1 in in_extent})
    y_spans = sorted({(y0, y1) for _x0, y0, _x1, y1 in in_extent})
    axis = "x" if len(x_spans) >= len(y_spans) else "y"

    region_bounds = _strip_regions(
        x_spans if axis == "x" else y_spans, buffer, active_names, active_fracs, discard_ceiling,
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
    for box in in_extent:
        idx = _region_index(region_bounds, axis, box)
        if idx is None:
            dropped_outside += 1
        else:
            kept[region_bounds[idx][0]] += 1

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
        width=width, height=height, tile_size=tile_size, overlap=overlap,
        axis=axis, buffer=buffer, split_names=split_names,
        requested_fractions=fractions, stripes_per_split=stripes_per_split,
        discard_ceiling=discard_ceiling, regions=regions, region_bounds=region_bounds,
        total_tiles=total_tiles, tiles_dropped_past_extent=tiles_dropped_past_extent,
        tiles_dropped_outside_regions=dropped_outside,
        kept_tiles=kept, realized_fractions={n: kept[n] / total_kept for n in active_names},
        realized_discard_fraction=dropped_outside / tiles_within_extent,
    )

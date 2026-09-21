"""Group-aware, annotation-stratified train/val/calibration splitting.

Pure standard library, intentionally imports no torch and no ``datasets`` so it
can be used from ``tcip_mcp.tools.data_tools`` (which is in the server's
always-load group and must stay torch-free).

Ported from the chestnut-burr ``burr_detection/dataset.py`` ``_group_balanced_split``
with three deliberate deviations needed for the MCP tool / auto-val use:

1. Operates on stems + an ``annotation_counts`` dict (not image paths + a
   labels dir), so callers that already scanned the folder don't re-glob.
2. No-foreground fallback: when ``annotation_counts`` is ``None`` or all
   zero, every group is treated as foreground weighted by its tile count, so
   label-less tasks (semantic_seg / classification) can reuse the same code.
   The reference raises; we only raise when ``require_foreground=True``.
3. Fraction-gated min-foreground: the minimum-foreground guarantee is
   applied only to splits whose fraction is > 0, so ``splits=(0.8, 0.2, 0.0)``
   yields a clean two-way split (auto-val needs this).

The split is group-coherent (sibling tiles of one source never straddle two
splits), annotation-balanced, and deterministic in ``seed``.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import os
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
    register_store,
    store,
)

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import Sample, Selection

logger = logging.getLogger(__name__)

# A tiled stem looks like ``<source>_<x>_<y>`` (two trailing integer fields).
# Strip that suffix so all tiles of one source share a group key. A stem with a
# single trailing ``_<int>`` (e.g. ``img_001``) does not match and falls back to
# the full stem.
_TILE_GROUP_RE = re.compile(r"^(.*)_\d+_\d+$")

SPLIT_NAMES = ("train", "val", "calibration")


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


def count_lines(label_path: str | Path) -> int:
    """Count non-empty lines in a label file. Missing file -> 0."""
    p = Path(label_path)
    if not p.is_file():
        return 0
    try:
        return sum(1 for line in p.read_text().splitlines() if line.strip())
    except OSError:
        return 0


def count_label_lines(
    label_path: str | Path, *, subject: str | None = None, attribute: str | None = None,
) -> int:
    """Annotation count for one per-image label document, by its own path.

    Drives stratified splitting, a foreground-density proxy. With ``subject`` omitted, every
    record in the file counts regardless of subject (the whole-tree proxy a stats-only draw
    still uses); given ``subject``, only that subject's records count, further narrowed to those
    already assessed for ``attribute`` when one is given (an instance never assessed for it is
    not yet a foreground fact for this scope). A selection's draw is subject-scoped, so it always
    states its own ``subject`` here: an unscoped count would read a group's annotations of some
    other subject sharing the file as this draw's own foreground. An empty ``subject`` or
    ``attribute`` reads as unset, so a caller passing a checkpoint's stamped ``attribute=""``
    counts against every record assessed or not (its real meaning, "no attribute") rather than
    scoring zero against a key no record ever carries.

    The document is named by the path the caller holds, never by a directory and a name it would
    be recomposed from: a caller holding an admitted sample already knows which file answered for
    it, and one enumerating a directory composes the path once at its own site
    (:func:`~tcip_mcp.dataset_layout.label_filename`).

    A missing file scores 0 foreground; a present, unreadable one raises
    :class:`~tcip_annotation.json_io.UnreadableLabelDocument` rather than scoring 0, since a
    corrupt document is not the same fact as an empty one.
    """
    from tcip_annotation import json_io

    subject, attribute = subject or None, attribute or None
    jp = Path(label_path)
    if not jp.is_file():
        return 0
    records = json_io.read_annotations(str(jp))
    if subject is None:
        return len(records)
    return sum(
        1 for a in records
        if a.subject == subject and (attribute is None or a.attributes.get(attribute) is not None)
    )


def label_document_extent(label_path: str | Path) -> tuple[int, int] | None:
    """``(width, height)`` one per-image label JSON records, or ``None`` when the file is missing
    or carries no positive width/height.

    The label file already carries the frame its boxes were authored against (the json_io
    schema's top-level ``width``/``height``), so a caller that needs an image's pixel extent for
    split geometry reads it here rather than decoding the image. This is the same field
    :class:`~tcip_mcp.pipelines.data.datasets.TiledDetectionDataset` treats as authoritative for
    its own authored-vs-decoded frame check, so a split derived from this extent and a tiled
    dataset later built over the same sample agree on the frame by construction. A present,
    unreadable file raises :class:`~tcip_annotation.json_io.UnreadableLabelDocument` rather than
    reading ``None``, the same distinction between "no extent" and "unreadable" every other reader
    keeps.
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
    seed: int = 42,
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
        Optional ``{stem: annotation_line_count}``. When omitted or all-zero,
        the no-foreground fallback weights groups purely by tile count.
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
        its ordinary largest-first assignment, met first with the smallest foreground groups so
        the dense ones remain for balancing. Omitted, every active side gets a minimum of one,
        the floor :func:`cal_holdout_split` and a stats-only draw resolve to; a side
        named here with no active fraction is ignored. A tree with fewer foreground groups than a
        minimum asks for simply gets fewer than that side's floor met, this function never raises
        on it (a caller wanting a hard floor calls :func:`refuse_insufficient_foreground_groups`
        first, before any write).
    foreground_counts:
        The minimum pass's own foreground signal, independent of ``annotation_counts`` (which
        stays the balancing pass's signal, gated by whatever a caller's own stratification flag
        does before this function is ever called). Omitted, the minimum pass draws from whatever
        ``annotation_counts`` produced; given, a group counts toward a side's
        minimum only when its ``foreground_counts`` sum is positive, even when
        ``annotation_counts`` is ``None`` and the balancing pass is running its no-foreground,
        every-group-is-foreground fallback. A selection's draw passes its subject-scoped
        foreground count here on every draw, so the calibration side's floor is met with real
        foreground regardless of whether balancing itself is stratified.

    Returns
    -------
    ``{"train": [...], "val": [...], "calibration": [...]}``, a partition of ``stems``.
    """
    if group_key_fn is None:
        group_key_fn = GROUP_KEY_FNS["tile_prefix"]
    stems = list(stems)
    fracs = dict(zip(SPLIT_NAMES, splits))
    active = [n for n in SPLIT_NAMES if fracs.get(n, 0.0) > 0]

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

    result: dict[str, list[str]] = {n: [] for n in SPLIT_NAMES}
    if not fg_groups or not active:
        # Nothing to stratify on; dump everything into the first active split.
        target = active[0] if active else "train"
        result[target] = sorted(stems)
        return result

    total_ann = sum(group_ann[gk] for gk in fg_groups) or 1
    total_fg_tiles = sum(group_tiles[gk] for gk in fg_groups) or 1
    targets_ann = {n: fracs[n] * total_ann for n in SPLIT_NAMES}
    targets_tiles = {n: fracs[n] * total_fg_tiles for n in SPLIT_NAMES}

    state_ann = {n: 0 for n in SPLIT_NAMES}
    state_tiles = {n: 0 for n in SPLIT_NAMES}
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
    tile_target_all = {n: fracs[n] * total_all_tiles for n in SPLIT_NAMES}
    for gk in sorted(bg_groups, key=lambda g: group_tiles[g], reverse=True):
        best_split = max(active, key=lambda n: tile_target_all[n] - state_tiles[n])
        assignment[gk] = best_split
        state_tiles[best_split] += group_tiles[gk]

    for gk, gs in groups.items():
        result[assignment.get(gk, active[0])].extend(gs)
    return {n: sorted(result[n]) for n in SPLIT_NAMES}


def foreground_group_count(
    members: Iterable[str], counts: Mapping[str, int], group_of: Callable[[str], str],
) -> int:
    """How many distinct groups among ``members`` carry foreground.

    The one reading of "carries foreground" every floor below is measured against, so a draw over
    a whole tree, a bound run's calibration universe and a preflight all count groups the same
    way. ``counts`` is each member's own foreground count under the caller's own key for it
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`); a member the map does not
    name carries none.
    """
    return len({group_of(member) for member in members if counts.get(member, 0) > 0})


def refuse_insufficient_foreground_groups(
    foreground_groups: int, minimums: dict[str, int], *, remedy: str,
) -> None:
    """Refuses, before any write, a draw whose tree holds fewer foreground groups than the sum
    of the per-side minimums a caller states.

    :func:`~tcip_mcp.tools.data_tools.draw_splits` calls this once, over the whole tree, ahead of
    drawing a selection, over the three sides a selection always draws (a draw refuses a zero
    ratio on any of them, so every side named here is always requested);
    :func:`selection_calibration_universe` calls it for its one held-out side in place of its own
    inline count. Names every requested side and its minimum, the foreground groups actually
    found, and ``remedy``, the caller's own sentence saying what to add: what counts as a
    foreground group differs by the ground truth a draw reads, so the remedy is stated by the
    caller that knows which it read rather than assumed here.
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
    flat, dateless tree.

    A stem is unique only within one capture date (cameras reuse names across dates), so a draw
    spanning more than one date needs this to keep two same-named images from two dates apart.
    :func:`~tcip_mcp.tools.data_tools.draw_splits` and ``tcip plant-aware-group-splits`` both key
    their members this way, through this one function, so an agent-supplied ``group_key_map``
    lands on the keys the draw looks up.
    """
    return f"{date}/{stem}" if date else stem


def resolve_group_key_fn(
    group_by: str, stems: Sequence[str], *, group_key_map: dict[str, str] | None = None,
) -> Callable[[str], str]:
    """Resolve a grouping policy to a callable, raising loudly rather than silently degrading.

    The one resolution every call site goes through: an unrecognized ``group_by`` string, or a
    ``group_key_map`` missing coverage for some of ``stems``, is a policy error the caller must
    see immediately, not a silent fallback to the tile-prefix default that could mis-group a
    dataset without anyone noticing.
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
    """The group key a draw records for a bare stem admitted out of one capture date's directory.

    The one derivation of that key: the stem becomes its member identity
    (:func:`member_identity`, so two dates' same-named images are two groups rather than one) and
    the policy is resolved against those identities (:func:`resolve_group_key_fn`, so an
    agent-supplied ``group_key_map`` is keyed the way ``draw_splits`` already keys one). Every
    producer of a recorded group key goes through this, and so does the leak check that has to
    reproduce a key for a stem a recorded map does not cover; a producer and a reader spelling
    one key two ways is a leak that reads as no leak.

    ``stems`` and ``group_key_map`` are the producer's own: the map is checked for coverage of
    those stems' identities before any key is handed out. A reader passes neither, since with no
    map to check there is nothing to cover and the named policy alone answers.

    :func:`~tcip_mcp.tools.data_tools.draw_splits` does not call this: it already holds member
    identities, because it draws across several capture dates at once, so what it needs is this
    function's middle step, :func:`resolve_group_key_fn` over those identities. This is that same
    resolution plus the stem-to-identity adapter a producer working under one directory needs.
    """
    identities = [member_identity(date, stem) for stem in stems]
    key_fn = resolve_group_key_fn(group_by, identities, group_key_map=group_key_map)
    return lambda stem: key_fn(member_identity(date, stem))


def draw_train_val(
    stems: Sequence[str], *, annotation_counts: dict[str, int] | None,
    group_key_fn: Callable[[str], str], val_ratio: float, seed: int,
) -> tuple[list[str], list[str]]:
    """``(train, val)`` for ``stems``: one call to :func:`group_balanced_split` at
    ``(1 - val_ratio, val_ratio, 0.0)``, the assembly the drawn path (``auto_train_val``'s
    group-aware split) and a bound run's redraw (``data.split.redraw_within_selection``) both
    wrap, so the two agree on how a train/val split is actually drawn from a group-keyed stem
    list. Neither retries nor degrades on a starved side here; each caller decides what a
    starved side means for it (the drawn path's own stem-level retry and its degrade to no
    validation, the redraw's own refusal).
    """
    parts = group_balanced_split(
        list(stems), annotation_counts=annotation_counts, group_key_fn=group_key_fn,
        splits=(1.0 - val_ratio, val_ratio, 0.0), seed=seed,
    )
    return parts["train"], parts["val"]


def redraw_starved_issue(
    selection: "Selection", *, selection_dir: str | None, seed: int | None,
) -> str | None:
    """Whether a selection's train-plus-val members resolve to too few *foreground* groups for a
    redraw to populate both a train and a val side.

    :func:`group_balanced_split`'s own per-side minimum needs one foreground group for each active
    side, met first from the smallest foreground groups; a background-only group (zero foreground
    signal, e.g. a confirmed negative) is placed afterwards by tile deficit and can concentrate
    entirely onto the side that already met its minimum, leaving the other side genuinely empty.
    Counting every distinct group regardless of foreground signal misses exactly that case: two
    groups, one of them entirely background, reads as enough groups when only one of them can ever
    satisfy a side's minimum. Checked before any run starts
    (:func:`~tcip_mcp.tools.training_tools.preflight_config`, and ``run_hyperparameter_search``'s
    own pre-mint check for ``split_draws``), ahead of the redraw's own refusal once it has actually
    drawn a starved side for real.

    The members are the selection's own train-plus-val samples, grouped by the keys its draw
    recorded on them and counted under the scope it drew with
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`), never re-derived from a
    grouping policy here.

    ``None`` when at least two foreground groups are available; the refusal otherwise, naming the
    selection, the seed and the two counts, with the two remedies: drop the redraw to bind the
    selection's recorded partition instead, or draw a selection with at least two foreground
    groups across train and val.
    """
    from tcip_mcp.pipelines.data.label_queries import foreground_counts

    pool = selection.on("train") + selection.on("val")
    group_of = {s.identity: s.group for s in pool}
    counts = foreground_counts({s.identity: s for s in pool}, selection.scope)
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
    """Whether ``a`` and ``b`` name the identical on-disk directory, by filesystem identity
    (``os.path.samefile``) rather than a bare string comparison, so a trailing separator,
    forward slashes or a relative spelling of the same directory reads as the same path.

    ``False`` whenever either side is empty or does not exist as a directory: there is nothing
    to compare a missing path against, and a caller distinguishing "same" from "cannot tell" gets
    the honest "not the same" rather than a crash. The one comparison every caller checking a
    recorded directory against a caller-stated one shares (the bound-checkpoint selection
    comparison in ``calibration.py``, the project-root checks in ``experiments.py``, the image-root
    checks the results route and the phenology tools make), never a second re-derivation of it.
    """
    if not a or not b:
        return False
    a_path, b_path = Path(a), Path(b)
    return (a_path.is_dir() and b_path.is_dir()
            and os.path.samefile(a_path.resolve(), b_path.resolve()))


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
    """A within-image train/val(/test) split: the image partitioned into contiguous
    pixel-space strips along one axis, each strip assigned whole to one side, with a buffer
    band excluded at every boundary between differently-assigned strips.

    A square block grid couples how finely a ratio can be hit to how much boundary a buffer
    removes: fewer, larger blocks minimize discard but leave too little assignment
    granularity to land near a requested split, while enough blocks for precision multiplies
    boundary count (and therefore discard) with it. Striping along one axis decouples the
    two, and at ``stripes_per_split=1`` (the default) each side is exactly one contiguous
    region: the fewest possible boundaries, so discard is minimized and concentrates at the
    internal cuts between sides rather than scattering through the interior. Regions are
    ordered largest-share-first from the axis center outward, so a small side faces at most
    one differently-assigned neighbor rather than being sandwiched between two; each boundary
    is shrunk from one side only (enough on its own to guarantee ``>= buffer`` separation), so
    the image-edge-facing side of the outermost two regions is never shrunk at all. Raising
    ``stripes_per_split`` asks for that many separate pieces per side, but the fixed
    center-out order does not scatter every side equally: the largest share's pieces sort
    adjacent under that order and merge back into one contiguous region, so it is the smaller
    sides that actually end up split into pieces flanking it. Three names of distinct shares,
    each cut into three pieces at ``stripes_per_split=3``, land in the order ``C, B, B, A, A, A, B, C, C`` (``A`` the largest
    share, ``C`` the smallest), five merged regions once adjacent same-name pieces combine:
    ``A`` stays whole, ``B`` and ``C`` each split into two pieces on either side of it.
    ``discard_ceiling`` caps how many pieces actually get used regardless of how many were
    asked for, since boundary cost scales with piece count, not with how finely the tile
    lattice itself could be subdivided.

    ``regions`` maps each split name to its list of half-open pixel rects (already merged
    where two same-split strips landed adjacent, and buffer-shrunk on any side bordering a
    different-split neighbor): :class:`TiledDetectionDataset`'s ``keep_regions`` consumes
    these directly. ``realized_fractions`` is each side's kept tile count over the total kept
    across every side (post-buffer), not the requested fractions.
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
        """Index into ``region_bounds`` for a kept tile at ``(tile_x, tile_y)``, or ``None``
        when this position falls in a dropped gap (buffer band or past-extent) rather than
        fully inside any region. The one containment lookup every split-membership query
        (identity, split name) shares."""
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
    (smaller) slot alternately extends the left or right end of the growing arrangement.

    A share sandwiched between two differently-assigned neighbors needs buffer margin on both
    sides at once, so the slot least likely to survive that is the smallest one, exactly the
    one placed last. Center-out puts the largest share (the most likely to have enough raw
    lattice positions to absorb a two-sided margin) in the middle and tapers outward, so every
    other slot faces at most one differently-assigned neighbor. A tie among equal shares
    resolves in the order the slots were given, so which cardinal side a tied slot lands on is
    fixed by the caller's declared order and never by a seed.
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
    """Merged, buffer-shrunk ``(name, start, end)`` pixel regions along one axis, in axis
    order, cut and shrunk in the discrete tile-origin lattice rather than continuous pixel
    space: a region with positive pixel width could otherwise miss the stride-spaced lattice
    entirely and contain zero real tile origins.

    Two independent knobs: ``stripes_per_split`` sets how many separate pieces a side is cut
    into before adjacent same-name pieces merge (capped by ``discard_ceiling``, the maximum share of the axis a buffer band
    between differing sides may consume, so asking for more pieces never buys precision at
    unbounded discard cost); the fraction each side targets sets its total share of the axis
    directly. At the default of one piece per side, every side is one contiguous region, the
    fewest possible boundaries (``len(split_names) - 1``), and only the *outermost* two
    regions' image-edge-facing sides go unshrunk, so discard concentrates at the internal
    boundaries between sides rather than scattering through the interior; more pieces trade
    some of that back, but not evenly across sides: the fixed center-out order
    (:func:`_center_out_order`) sorts the largest share's own pieces adjacent, so they merge
    back into one contiguous region regardless of ``stripes_per_split``, while the smaller
    sides are the ones that actually end up scattered into pieces flanking it (see
    :class:`SpatialStripSplit`'s docstring for the worked example), a choice left to the caller.
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
    """Split one image's own tile lattice into disjoint pixel-space strips, one side per name.

    Unlike :func:`group_balanced_split` (which partitions whole source images), this
    partitions the tiles *of a single image*, for the case where there are too few source
    images to hold one out whole: a strip is train, val, or test instead of a stem.

    The tile lattice comes from :func:`~tcip_mcp.pipelines.data.tiling.tile_positions` at the
    training stride (never re-derived), so the regions this returns tile the same grid a
    :class:`TiledDetectionDataset` built at this ``tile_size``/``overlap`` will actually
    index. The split runs along whichever axis (width or height) offers more distinct tile
    positions, for the finest achievable ratio precision; at the default
    ``stripes_per_split=1`` each requested split is one contiguous region, minimizing discard
    and concentrating it at the internal cuts between sides. A higher ``stripes_per_split``
    (capped by ``discard_ceiling``, see :class:`SpatialStripSplit`) places more pieces per
    side by the same fixed center-out order, but that order sorts the largest share's own
    pieces adjacent, so they merge back into one region regardless of the count asked for; it
    is the smaller sides that actually end up scattered into pieces flanking it, trading their
    own discard for a guard against those sides correlating with a spatial gradient in the
    field.

    ``buffer`` (pixels) is the minimum gap kept around every boundary between two
    differently-assigned strips: an explicit value below ``tile_size`` is refused, since a
    smaller gap cannot guarantee a kept tile on one side never shares pixels or immediate
    context with a kept tile on another, including under ``overlap > 0``. Omitted, it
    defaults to ``tile_size``.

    ``fractions`` must be non-negative and sum to 1.0, matching ``split_names`` in length; a
    zero fraction drops that name from the split entirely (fewer than two non-zero fractions
    is refused, nothing to split). Raises ``ValueError`` when no tile fits fully inside the
    image extent at this ``tile_size``, or when the derived strip layout leaves any requested,
    non-zero-fraction side with zero kept tiles.
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
    holdout_ratio: float = 0.5,
    seed: int = 0,
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
    """One locked split per dataset identity, named for the identity it locks.

    The identity is in the filename rather than in a directory of its own, which is the
    convention the operating-point sweep artifact beside it already uses.
    """

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

    ``scope_root`` is required and has no default. The lock is evidence about one dataset (which
    of its images were held back), so it travels with that data. A root this function resolved for
    itself would be the process-wide platform root, which adopting a project repins mid-life: a
    lock drawn before the adoption and read after resolves under a different scope, reads as
    absent, and is redrawn, which is the silent re-cut this lock exists to prevent.

    ``last_writer_wins`` rather than compare-and-set: a redraw is the only write that reads
    first, and it is also the deliberate recovery for a lock whose bytes do not decode, which
    is a state no version token can be read from. The redraw is an audited admin call on one
    dataset identity, so the write it stakes is single-shot in practice.
    """
    if PureWindowsPath(identity_hash).name != identity_hash or identity_hash == "..":
        raise BadKey(
            f"dataset identity {identity_hash!r} is not a single name: an identity carrying a "
            "path separator would address a lock outside the artifact store"
        )
    return Key(CAL_HOLDOUT_LOCK_STORE, str(Path(scope_root).resolve()), (identity_hash,))


def cal_holdout_lock_path(identity_hash: str, *, scope_root: str | Path) -> Path:
    """Where a dataset identity's locked cal/holdout split lives on disk under ``scope_root``.

    Placed by the store's own locator under the key's own scope, never by a second reconstruction
    of either, so this answers the path the seam reads and writes rather than a parallel one.
    """
    key = cal_holdout_lock_key(identity_hash, scope_root=scope_root)
    return Path(key.root, *_CalHoldoutLockLocator().relative_path(key.root, key.parts).parts)


def cal_holdout_scope_root(labels_dir: str | Path) -> Path:
    """The root a labeled directory's locked cal/holdout split is scoped to.

    The dataset root the labels live under, so the lock travels with the data the split was drawn
    over. A directory the dataset layout cannot place is its own anchor, the answer
    ``prediction_buckets.bucket_key_of`` already gives a bucket under no dataset root, so a
    calibration over a loose labeled directory still gets a scope that survives a project adoption.
    The calibration door and the redraw tool both resolve the scope through this, so a redraw
    addresses the lock the calibration wrote instead of drawing one nothing reads.
    """
    from tcip_mcp.dataset_layout import dataset_root_of

    root = dataset_root_of(labels_dir)
    return (root if root is not None else Path(labels_dir)).resolve()


def label_image_stems(
    labels_dir: str | Path, images_dir: str | Path | None = None,
) -> tuple[list[str], dict[str, "Path | BandGroupRef"]]:
    """Stems with a readable per-image label file, one scan shared by every caller.

    ``calibrate_operating_point`` and ``redraw_calibration_holdout`` both call this rather
    than scanning independently, so they agree on what "the dataset's stems" are: a caller
    passing ``images_dir`` gets the stronger labels-intersect-images stem universe, one that
    only globs labels gets the weaker (and possibly stale, if an image was deleted/renamed)
    labels-only universe.

    With ``images_dir`` omitted, returns every stem with a label file (``stem_to_image`` empty),
    the label-only universe, for a caller (e.g. a redraw with no images to check) that has no
    images directory to intersect against. With ``images_dir`` given, only stems that also have a
    matching logical image (a plain file, or a ``.bandgroup``-grouped capture) survive, so a stem
    in the labels dir with no image left (deleted/renamed) never enters the split universe.
    ``labels_dir`` may itself be a prediction bucket (a calibration/holdout split of one), so its
    own provenance sidecars are excluded through :func:`~tcip_annotation.json_io.prediction_documents`
    rather than named as if they were image stems.

    This is the whole-directory universe, for a door measuring over a whole labelled directory. A
    door restricted to a selection reads :func:`selection_calibration_universe` instead, which
    answers from the selection's own recorded samples and scans nothing.
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
) -> tuple[list[str], str, dict[str, str], dict[str, list[str]], dict[str, "Sample"]]:
    """The calibration universe a selection gives one caller restricting a read to ``labels_dir``.

    The selection's ``calibration`` samples whose own ground truth lives in ``labels_dir``, so a
    read measures on exactly the side the draw held out for it, never the side the shipped
    checkpoint was chosen on (a disjointness check catches a leak onto that side separately,
    against the checkpoint's own selection). The narrowing is by the ground-truth scope each
    sample records, never by a capture date compared against a directory: a selection spans dates,
    and a door that names one date's ground truth reads that date's held-out samples out of it.

    The selection is the whole membership authority here: nothing is intersected against a
    directory listing, and whether each member's ground truth still admits is the one re-admission
    over recorded samples
    (:func:`~tcip_mcp.pipelines.data.label_queries.refuse_inadmissible_samples`), which the door
    building a loader over these samples runs. Whatever shape that ground truth is, this universe
    returns it: a door refuses only what its own measurement cannot do, never a shape on this
    function's behalf.

    The floor below counts only the groups that actually carry foreground, read through the one
    per-sample counter
    (:func:`~tcip_mcp.pipelines.data.label_queries.foreground_counts`) under the selection's own
    class space, so a universe of background-only groups cannot pass it and every door that names
    a selection counts the same way the draw did.

    ``min_foreground_groups`` is the caller's own floor, forwarded to
    :func:`refuse_insufficient_foreground_groups` verbatim; omitted, it defaults to
    ``{"calibration": 2}``, the shape a locked cal/holdout draw needs since it halves the universe
    into two non-empty parts. A caller that draws no lock and halves nothing (``evaluate_model``)
    states its own floor of one instead, so a legitimate single-foreground-group universe is not
    refused for a halving this caller never performs.

    Returns ``(stems, group_by, group_key_map, excluded, samples)``:
    ``stems`` is the calibration side's bare member names under ``labels_dir``; ``group_key_map``
    is each one's recorded group key, with ``group_by="explicit_map"``, so the locked draw groups
    by exactly the keys the selection drew with rather than re-resolving a policy; ``excluded``
    names what the selection put elsewhere under this scope, its recorded train members
    (``excluded_training_stems``) and its recorded val members (``excluded_validation_stems``);
    ``samples`` is each universe member's own recorded sample, so a caller measures on the pixels
    the draw held out rather than on whatever a directory listing of the same names happens to
    hold today. The class space the foreground was counted under is the selection's own
    (``Selection.scope``), which the caller already holds.

    Refuses a calibration sample naming a pixel rect
    (:func:`~tcip_mcp.pipelines.data.selection.refuse_unreadable_samples`), the same refusal the
    loaders apply, so no door measures a region over its whole source instead. Refuses, naming the
    count, the labels directory and the caller's own floor, when the resulting universe would hold
    fewer foreground (when known) groups than that floor states.
    """
    from tcip_mcp.pipelines.data.label_queries import foreground_counts
    from tcip_mcp.pipelines.data.selection import refuse_unreadable_samples

    scope = selection.scope
    here = Path(labels_dir).resolve()

    # Through the sample's own recorded scope, never the parent of its ground-truth path: a table
    # scope is the table itself, which no sample's parent directory equals.
    in_scope = [s for s in selection.samples
                if Path(s.ground_truth_scope).resolve() == here]
    by_side: dict[str, dict[str, str]] = {name: {} for name in SPLIT_NAMES}
    universe_samples: dict[str, "Sample"] = {}
    for sample in in_scope:
        by_side[sample.side][sample.member_stem] = sample.group
        if sample.side == "calibration":
            universe_samples[sample.member_stem] = sample
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
    return stems, "explicit_map", group_key_map, excluded, universe_samples


def _split_content_hash(parts: dict[str, list[str]] | None) -> str | None:
    """Content hash over a split's calibration+holdout membership (order-independent per side)."""
    if not parts:
        return None
    h = hashlib.sha256()
    for key in ("calibration", "holdout"):
        for s in sorted(parts.get(key) or []):
            h.update(s.encode("utf-8"))
            h.update(b"\0")
        h.update(b"\0\0")
    return h.hexdigest()[:16]


def resolve_locked_cal_holdout_split(
    stems: Sequence[str],
    *,
    identity_hash: str,
    scope_root: str | Path,
    annotation_counts: dict[str, int] | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    holdout_ratio: float = 0.5,
    seed: int = 0,
    force_redraw: bool = False,
    timestamp: str | None = None,
    selection_dir: str | None = None,
) -> dict:
    """Resolve (and lock) the calibration/holdout split for one dataset identity.

    A held-out reference that is not actually held out can happen when the split is a
    deterministic lexicographic cut, redrawn fresh on every call with no train-disjointness check
    and no record of what was drawn, so a "held-out validation" gate could pass on data that
    wasn't actually held out. This locks the split on its first draw for a given
    ``identity_hash``: every later call for the same identity returns the identical split, never a
    silent re-cut, unless the caller explicitly passes ``force_redraw=True`` (the audited admin
    path, see the ``redraw_calibration_holdout`` MCP tool; never wired to a default kwarg on a
    high-traffic tool).

    The grouping policy is resolved via :func:`resolve_group_key_fn` first, so a malformed
    ``group_by``/``group_key_map`` raises loudly here rather than silently degrading.

    If a lock already exists and the caller's declared policy (``group_by``/``group_key_map``/
    ``seed``/``holdout_ratio``/``selection_dir``) differs from what is recorded in it, the
    divergence is logged as a warning and returned under ``"policy_divergence"``
    (``{"requested": ..., "locked": ...}``), the locked split is still returned unchanged, never
    silently redrawn, so a caller can see the mismatch on the result rather than in a server
    log. Stems the caller has that the lock doesn't cover are similarly surfaced under
    ``"unlocked_stems"`` rather than silently dropped, the lock stays authoritative for what it
    already covers.

    ``selection_dir`` names the selection a caller drawing ``stems`` from one restricted it to
    (``None`` for a whole-directory draw, the record's key set stays the same either way); it is
    written into the lock and into every ``redraw_history`` entry alongside the rest of the
    declared policy, never resolved or compared here, that is the caller's own job
    (:func:`selection_calibration_universe`, which each door calls before it ever draws a universe
    to lock).

    A locked stem with no corresponding entry in the caller's current ``stems`` (its image/label
    was deleted or renamed since the split was locked) raises ``ValueError`` rather than silently
    returning stale membership for a caller to crash on later; this mirrors
    ``resolve_group_key_fn``'s already-loud policy-error convention, which this function already
    lets propagate unmodified. A lock file that exists but fails to parse (corrupt, not merely
    absent) raises for the same reason when ``force_redraw=False``, "unreadable" must never
    silently become "no lock exists yet, draw a fresh one", which would violate this function's
    own never-a-silent-re-cut guarantee. ``force_redraw=True`` (the audited admin path)
    proceeds past a corrupt lock file rather than also being blocked by it; redraw history just
    can't be recovered from what couldn't be read.

    ``scope_root`` is required and has no default: it is the root the lock is stored under, the
    dataset root of the labels or records the split was drawn over. See
    :func:`cal_holdout_lock_key` for why a root this layer resolved for itself would make the
    never-a-silent-re-cut guarantee above violable, and :func:`cal_holdout_scope_root` for the
    derivation a caller holding a labeled directory uses.

    ``timestamp`` is threaded in from the caller (this pipeline layer does not call
    ``datetime.now()`` itself, matching the rest of the codebase's tool-boundary convention) and
    is only meaningful when a new draw actually happens (first draw, or ``force_redraw=True``).

    Returns the full locked-split dict: ``{identity_hash, calibration, holdout, group_by,
    group_key_map, seed, holdout_ratio, selection_dir, redraw_history}``, plus the optional
    ``policy_divergence`` / ``unlocked_stems`` report fields above when a lock already existed.
    """
    group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    lock_key = cal_holdout_lock_key(identity_hash, scope_root=scope_root)
    try:
        existing = store.read(lock_key, default=None)
    except DecodeError as exc:
        if not force_redraw:
            raise ValueError(
                f"the cal/holdout lock for identity_hash={identity_hash!r} exists but could not "
                f"be read/parsed ({exc}). Refusing to silently treat a corrupt lock as 'no lock "
                "exists' and redraw. Investigate the file, or use redraw_calibration_holdout "
                "once you've deliberately decided to replace it."
            ) from exc
        logger.warning(
            "the cal/holdout lock for identity_hash=%s is corrupt (%s); force_redraw=True "
            "proceeds to draw a fresh lock. Its prior "
            "redraw history could not be recovered from the unreadable record.",
            identity_hash, exc,
        )
        existing = None
    declared_policy = {
        "group_by": group_by, "group_key_map": group_key_map,
        "seed": seed, "holdout_ratio": holdout_ratio,
        "selection_dir": selection_dir,
    }

    if existing is not None and not force_redraw:
        locked_stems = set(existing.get("calibration", [])) | set(existing.get("holdout", []))
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
        recorded_policy = {k: existing.get(k) for k in declared_policy}
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
    redraw_history = list(existing.get("redraw_history", [])) if existing else []
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
    return locked

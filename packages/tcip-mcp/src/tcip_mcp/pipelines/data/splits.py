"""Group-aware, annotation-stratified train/val/test splitting.

Pure standard library — intentionally imports no torch and no ``datasets`` so it
can be used from ``tcip_mcp.tools.data_tools`` (which is in the server's
always-load group and must stay torch-free).

Ported from the chestnut-burr ``burr_detection/dataset.py`` ``_group_balanced_split``
with three deliberate deviations needed for the MCP tool / auto-val use:

1. Operates on **stems + an ``annotation_counts`` dict** (not image paths + a
   labels dir), so callers that already scanned the folder don't re-glob.
2. **No-foreground fallback** — when ``annotation_counts`` is ``None`` or all
   zero, every group is treated as foreground weighted by its tile count, so
   label-less tasks (semantic_seg / classification) can reuse the same code.
   The reference raises; we only raise when ``require_foreground=True``.
3. **Fraction-gated min-foreground** — the minimum-foreground guarantee is
   applied only to splits whose fraction is > 0, so ``splits=(0.8, 0.2, 0.0)``
   yields a clean two-way split (auto-val needs this).

The split is group-coherent (sibling tiles of one source never straddle two
splits), annotation-balanced, and deterministic in ``seed``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from tcip_mcp.project_paths import project_root
from tcip_mcp.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

# A tiled stem looks like ``<source>_<x>_<y>`` (two trailing integer fields).
# Strip that suffix so all tiles of one source share a group key. A stem with a
# single trailing ``_<int>`` (e.g. ``img_001``) does not match and falls back to
# the full stem.
_TILE_GROUP_RE = re.compile(r"^(.*)_\d+_\d+$")

SPLIT_NAMES = ("train", "val", "test")


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


def count_label_lines(labels_dir: str | Path, stem: str) -> int:
    """Annotation count for ``stem`` from its name-based per-image ``<stem>.json``.

    Drives stratified splitting (a foreground-density proxy across all subjects in the file), so a
    stem whose labels cannot be read scores 0 foreground.
    """
    from tcip_annotation import json_io

    jp = Path(labels_dir) / f"{stem}.json"
    if not jp.is_file():
        return 0
    return len(json_io.read_annotations(str(jp)))


def group_balanced_split(
    stems: Sequence[str],
    annotation_counts: dict[str, int] | None = None,
    group_key_fn: Callable[[str], str] | None = None,
    splits: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    require_foreground: bool = False,
) -> dict[str, list[str]]:
    """Partition ``stems`` into train/val/test, keeping each group intact.

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
        ``(train, val, test)`` fractions. A 0.0 fraction disables that split.
    seed:
        Deterministic seed.
    require_foreground:
        Raise ``ValueError`` when there is no foreground signal at all.

    Returns
    -------
    ``{"train": [...], "val": [...], "test": [...]}`` — a partition of ``stems``.
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

    # Minimum-foreground guarantee (fraction-gated): train always; val if >=2 fg
    # groups exist; test if >=3. Met first with the smallest foreground groups so
    # the dense ones remain for the balancing pass.
    n_fg = len(fg_groups)
    min_fg: dict[str, int] = {}
    if "train" in active:
        min_fg["train"] = 1
    if "val" in active and n_fg >= 2:
        min_fg["val"] = 1
    if "test" in active and n_fg >= 3:
        min_fg["test"] = 1

    fg_smallest_first = sorted(fg_groups, key=lambda gk: group_ann[gk])
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
        best_split, best_score = None, None
        for n in active:
            ann_ratio = (state_ann[n] + group_ann[gk]) / max(1.0, targets_ann[n])
            tile_ratio = (state_tiles[n] + group_tiles[gk]) / max(1.0, targets_tiles[n])
            score = 0.7 * ann_ratio + 0.3 * tile_ratio
            if best_score is None or score < best_score:
                best_split, best_score = n, score
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


def resolve_group_key_fn(
    group_by: str, stems: Sequence[str], *, group_key_map: dict[str, str] | None = None,
) -> Callable[[str], str]:
    """Resolve a grouping policy to a callable, raising loudly rather than silently degrading.

    Replaces every ``GROUP_KEY_FNS.get(group_by, default_group_key)`` call site (K1): an
    unrecognized ``group_by`` string, or a ``group_key_map`` missing coverage for some of
    ``stems``, is a policy error the caller must see immediately, not a silent fallback to the
    tile-prefix default that could mis-group a dataset without anyone noticing.
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


def cal_holdout_split(
    stems: Sequence[str],
    annotation_counts: dict[str, int] | None = None,
    group_key_fn: Callable[[str], str] | None = None,
    holdout_ratio: float = 0.5,
    seed: int = 0,
) -> dict[str, list[str]]:
    """A disjoint, group-coherent, annotation-balanced calibration/holdout split.

    A thin wrapper over :func:`group_balanced_split` (never reimplemented): the ``test``
    fraction is always 0, and the ``{"train", "val"}`` output is remapped to
    ``{"calibration", "holdout"}`` for the calibration callers.
    """
    parts = group_balanced_split(
        stems, annotation_counts=annotation_counts, group_key_fn=group_key_fn,
        splits=(1.0 - holdout_ratio, holdout_ratio, 0.0), seed=seed,
    )
    return {"calibration": parts["train"], "holdout": parts["val"]}


def cal_holdout_lock_path(identity_hash: str) -> Path:
    """Where a dataset identity's locked cal/holdout split lives (mirrors the operating-point
    sweep artifact's ``.tcip/artifacts/`` convention — see ``inference_tools.py``)."""
    return project_root() / ".tcip" / "artifacts" / f"cal_holdout_split_{identity_hash}.json"


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def label_image_stems(
    labels_dir: str | Path, images_dir: str | Path | None = None,
) -> tuple[list[str], dict[str, Path]]:
    """Stems with a readable per-image label file, one scan shared by every caller (K1 finding 4).

    ``_calibrate_operating_point`` and ``force_redraw_cal_holdout_split`` each used to run their
    own independent labels/images scan; a caller adding an images_dir got the stronger
    labels-intersect-images stem universe, one that only globbed labels got the weaker (and
    possibly stale, if an image was deleted/renamed) labels-only universe — the two could disagree
    on what "the dataset's stems" are. This is the one implementation both call.

    With ``images_dir`` omitted, returns every stem with a label file (``stem_to_image`` empty) —
    the label-only universe, for a caller (e.g. a redraw with no images to check) that has no
    images directory to intersect against. With ``images_dir`` given, only stems that ALSO have a
    matching image file on disk survive, so a stem in the labels dir with no image left
    (deleted/renamed) never enters the split universe in the first place.
    """
    labels_p = Path(labels_dir)
    label_stems = {p.stem for p in labels_p.glob("*.json")}
    if images_dir is None:
        return sorted(label_stems), {}
    images_p = Path(images_dir)
    stem_to_image = {p.stem: p for p in images_p.iterdir()
                     if p.suffix.lower() in _IMAGE_EXTS and p.stem in label_stems}
    return sorted(stem_to_image), stem_to_image


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
    annotation_counts: dict[str, int] | None = None,
    group_by: str = "tile_prefix",
    group_key_map: dict[str, str] | None = None,
    holdout_ratio: float = 0.5,
    seed: int = 0,
    force_redraw: bool = False,
    timestamp: str | None = None,
) -> dict:
    """Resolve (and lock) the calibration/holdout split for one dataset identity (K1).

    "The held-out reference is not held out" was a deterministic lexicographic cut, redrawn
    fresh on every call with no train-disjointness check and no record of what was drawn — so a
    "held-out validation" gate could pass on data that wasn't actually held out. This locks the
    split on its FIRST draw for a given ``identity_hash``: every later call for the same
    identity returns the identical split, never a silent re-cut, unless the caller explicitly
    passes ``force_redraw=True`` (the audited admin path — see the
    ``force_redraw_cal_holdout_split`` MCP tool; never wired to a default kwarg on a
    high-traffic tool).

    The grouping policy is resolved via :func:`resolve_group_key_fn` FIRST, so a malformed
    ``group_by``/``group_key_map`` raises loudly here rather than silently degrading.

    If a lock already exists and the caller's declared policy (``group_by``/``group_key_map``/
    ``seed``/``holdout_ratio``) differs from what is recorded in it, the divergence is logged as
    a warning AND returned under ``"policy_divergence"`` (``{"requested": ..., "locked": ...}``) —
    the locked split is still returned unchanged, never silently redrawn, but a caller now has a
    way to *see* the mismatch instead of reading server logs (K1 finding 5). Stems the caller has
    that the lock doesn't cover are similarly surfaced under ``"unlocked_stems"`` rather than
    silently dropped — the lock stays authoritative for what it already covers.

    A locked stem with no corresponding entry in the caller's current ``stems`` (its image/label
    was deleted or renamed since the split was locked) raises ``ValueError`` rather than silently
    returning stale membership for a caller to crash on later (K1 finding 4) — this mirrors
    ``resolve_group_key_fn``'s already-loud policy-error convention, which this function already
    lets propagate unmodified. A lock file that exists but fails to parse (corrupt, not merely
    absent) raises for the same reason when ``force_redraw=False`` — "unreadable" must never
    silently become "no lock exists yet, draw a fresh one", which would violate this function's
    own never-a-silent-re-cut guarantee. ``force_redraw=True`` (the audited admin path) is
    ITSELF the deliberate fix for a corrupt lock, so it proceeds past a corrupt file rather than
    also being blocked by it — redraw history just can't be recovered from what couldn't be read.

    ``timestamp`` is threaded in from the caller (this pipeline layer does not call
    ``datetime.now()`` itself, matching the rest of the codebase's tool-boundary convention) and
    is only meaningful when a new draw actually happens (first draw, or ``force_redraw=True``).

    Returns the full locked-split dict: ``{identity_hash, calibration, holdout, group_by,
    group_key_map, seed, holdout_ratio, redraw_history}``, plus the optional ``policy_divergence``
    / ``unlocked_stems`` report fields above when a lock already existed.
    """
    group_key_fn = resolve_group_key_fn(group_by, stems, group_key_map=group_key_map)
    lock_path = cal_holdout_lock_path(identity_hash)
    existing = None
    if lock_path.is_file():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if not force_redraw:
                raise ValueError(
                    f"cal/holdout lock file at {lock_path} exists but could not be read/parsed "
                    f"({exc}) — refusing to silently treat a corrupt lock as 'no lock exists' and "
                    "redraw. Investigate the file, or use force_redraw_cal_holdout_split once "
                    "you've deliberately decided to replace it."
                ) from exc
            logger.warning(
                "cal/holdout lock file at %s is corrupt (%s); force_redraw=True proceeds to draw "
                "a fresh lock (this call IS the deliberate, audited fix) — its prior redraw "
                "history could not be recovered from the unreadable file.", lock_path, exc,
            )
    declared_policy = {
        "group_by": group_by, "group_key_map": group_key_map,
        "seed": seed, "holdout_ratio": holdout_ratio,
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
                "force_redraw_cal_holdout_split to redraw deliberately, or restore the missing "
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
                "declared (locked=%s, declared=%s); returning the LOCKED split unchanged. Use "
                "force_redraw_cal_holdout_split to redraw deliberately.",
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
    atomic_write_json(lock_path, locked)
    return locked

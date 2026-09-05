"""Annotation-density stratification and per-dataset calibration-lock identity.

Two standing properties of the split layer. Foreground density, not tile count, is what
balances a group-coherent split, so a group with nothing annotated on it is background and
never stands in for an annotated one. And a locked calibration/holdout membership belongs to
exactly one dataset identity, so no other identity can inherit it.
"""

from __future__ import annotations

import tcip_store as ts
from tcip_mcp.pipelines.data.splits import (
    cal_holdout_lock_key,
    cal_holdout_lock_path,
    group_balanced_split,
    resolve_locked_cal_holdout_split,
)


def _annotated_and_empty_groups() -> tuple[list[str], dict[str, int]]:
    """Two annotated groups of four tiles each, plus twelve single-tile groups carrying nothing.

    The empty groups outnumber the annotated ones and are individually the smallest, so a
    weighting that reads tile count as foreground signal would satisfy a split's minimum
    foreground with one of them.
    """
    annotated = [f"{g}_{i}_0" for g in ("budsA", "budsB") for i in range(4)]
    empty = [f"bare{g}_0_0" for g in range(12)]
    counts = {s: 2 for s in annotated}
    counts.update({s: 0 for s in empty})
    return annotated + empty, counts


def _density_skewed_dataset() -> tuple[list[str], dict[str, int]]:
    """Groups whose annotation densities span an order of magnitude, running opposite to their
    tile counts, alongside groups with no annotations at all."""
    heavy = [f"burrHeavy_{i}_0" for i in range(2)]
    mid = [f"burrMid_{i}_0" for i in range(5)]
    light = [f"burrLight_{i}_0" for i in range(9)]
    blank = [f"blank{g}_0_0" for g in range(6)]
    counts = {s: 40 for s in heavy}
    counts.update({s: 6 for s in mid})
    counts.update({s: 1 for s in light})
    counts.update({s: 0 for s in blank})
    return heavy + mid + light + blank, counts


def test_minimum_foreground_guarantee_is_met_with_an_annotated_group():
    stems, counts = _annotated_and_empty_groups()
    parts = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.8, 0.2, 0.0), seed=1
    )
    assert parts["train"] and parts["val"]
    for name in ("train", "val"):
        assert sum(counts[s] for s in parts[name]) > 0, (
            f"{name} was given only stems with no annotations on them"
        )


def test_annotation_density_rather_than_tile_count_drives_the_balance():
    stems, counts = _density_skewed_dataset()
    stratified = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.6, 0.4, 0.0), seed=1
    )
    train_ann = sum(counts[s] for s in stratified["train"])
    val_ann = sum(counts[s] for s in stratified["val"])
    assert train_ann > 0 and val_ann > 0
    assert train_ann > val_ann, (
        "the side asking for the larger fraction received the smaller share of the foreground"
    )

    tile_only = group_balanced_split(
        stems, annotation_counts=None, splits=(0.6, 0.4, 0.0), seed=1
    )
    assert stratified != tile_only, (
        "the annotation-stratified split matched the no-annotation fallback exactly"
    )


def test_empty_groups_do_not_change_which_side_holds_the_denser_foreground():
    stems, counts = _density_skewed_dataset()
    annotated_only = [s for s in stems if counts[s] > 0]
    without_blanks = group_balanced_split(
        annotated_only, annotation_counts=counts, splits=(0.6, 0.4, 0.0), seed=1
    )
    with_blanks = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.6, 0.4, 0.0), seed=1
    )
    assert annotated_only
    for name in ("train", "val"):
        assert [s for s in with_blanks[name] if counts[s] > 0] == without_blanks[name]


def test_groups_with_no_annotations_reach_every_active_split():
    stems, counts = _density_skewed_dataset()
    parts = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.6, 0.4, 0.0), seed=1
    )
    for name in ("train", "val"):
        assert parts[name]
        assert any(counts[s] == 0 for s in parts[name]), (
            f"{name} received no stem without annotations"
        )


def test_lock_path_distinguishes_identities_sharing_an_eight_character_prefix(tmp_path):
    first = "9f3c17b20a1b2c3d4e5f6071"
    second = "9f3c17b2ffeeddccbbaa9988"
    assert first[:8] == second[:8] and first != second
    assert (cal_holdout_lock_path(first, scope_root=tmp_path)
            != cal_holdout_lock_path(second, scope_root=tmp_path))


def test_a_prefix_sharing_identity_never_inherits_another_datasets_lock(tmp_path):
    first_stems = [f"plotA{g}_{i}_0" for g in range(6) for i in range(2)]
    second_stems = first_stems + [f"plotB{g}_{i}_0" for g in range(6) for i in range(2)]
    first_hash = "9f3c17b20a1b2c3d"
    second_hash = "9f3c17b2ffeeddcc"

    first = resolve_locked_cal_holdout_split(
        first_stems, identity_hash=first_hash, scope_root=tmp_path, seed=1)
    assert set(first["calibration"]) | set(first["holdout"]) == set(first_stems)

    second = resolve_locked_cal_holdout_split(
        second_stems, identity_hash=second_hash, scope_root=tmp_path, seed=1)
    assert second["identity_hash"] == second_hash
    assert "unlocked_stems" not in second, (
        "the second dataset was handed a lock drawn over a different stem universe"
    )
    assert set(second["calibration"]) | set(second["holdout"]) == set(second_stems)
    assert ts.exists(cal_holdout_lock_key(first_hash, scope_root=tmp_path))
    assert ts.exists(cal_holdout_lock_key(second_hash, scope_root=tmp_path))

    reread = resolve_locked_cal_holdout_split(
        first_stems, identity_hash=first_hash, scope_root=tmp_path, seed=1)
    assert reread["calibration"] == first["calibration"]
    assert reread["holdout"] == first["holdout"]

"""Tests for the group-aware / annotation-stratified split utility (torch-free)."""

from __future__ import annotations

from collections import defaultdict

import pytest

from tcip_mcp.pipelines.data.splits import (
    default_group_key,
    GROUP_KEY_FNS,
    cal_holdout_lock_path,
    cal_holdout_split,
    count_lines,
    group_balanced_split,
    resolve_group_key_fn,
    resolve_locked_cal_holdout_split,
)


def test_default_group_key_strips_tile_offset():
    assert default_group_key("canopyA_128_256") == "canopyA"
    assert default_group_key("plainname") == "plainname"
    # A single trailing "_<int>" field does not match the two-field tile pattern.
    assert default_group_key("img_001") == "img_001"
    assert GROUP_KEY_FNS["stem"]("a_1_2") == "a_1_2"


def test_count_lines(tmp_path):
    p = tmp_path / "lbl.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n\n0 0.2 0.2 0.1 0.1\n")
    assert count_lines(p) == 2  # blank line ignored
    assert count_lines(tmp_path / "missing.txt") == 0


def _grouped(stems):
    return [f"src{g}_{r}_0" for g in range(stems) for r in range(3)]


def test_group_split_no_group_spans_splits():
    stems = _grouped(6)  # 6 sources x 3 tiles
    parts = group_balanced_split(stems, splits=(0.6, 0.2, 0.2), seed=1)
    group_to_splits = defaultdict(set)
    for split, ss in parts.items():
        for s in ss:
            group_to_splits[default_group_key(s)].add(split)
    assert all(len(v) == 1 for v in group_to_splits.values())


def test_group_split_deterministic_and_partition():
    stems = _grouped(8)
    a = group_balanced_split(stems, seed=7)
    b = group_balanced_split(stems, seed=7)
    assert a == b
    union = a["train"] + a["val"] + a["test"]
    assert sorted(union) == sorted(stems)
    assert len(union) == len(set(union)) == len(stems)


def test_group_split_foreground_stratification():
    fg = [f"fg{g}_{r}_0" for g in range(4) for r in range(2)]
    bg = [f"bg{g}_0_0" for g in range(4)]
    stems = fg + bg
    counts = {s: (2 if s.startswith("fg") else 0) for s in stems}
    parts = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.6, 0.4, 0.0), seed=3
    )
    assert parts["test"] == []  # 0.0 fraction -> empty

    def has_fg(ss):
        return any(counts[s] > 0 for s in ss)

    assert has_fg(parts["train"]) and has_fg(parts["val"])


def test_group_split_no_foreground_fallback():
    stems = _grouped(4)
    counts = {s: 0 for s in stems}
    parts = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.7, 0.3, 0.0),
        seed=5, require_foreground=False,
    )
    assert parts["train"] and parts["val"]  # fallback splits by tile count
    with pytest.raises(ValueError):
        group_balanced_split(stems, annotation_counts=counts, seed=5, require_foreground=True)


# --- resolve_group_key_fn (K1) ---

def test_resolve_group_key_fn_unrecognized_group_by_raises():
    with pytest.raises(ValueError, match="Unrecognized"):
        resolve_group_key_fn("not_a_real_key", ["a", "b"])


def test_resolve_group_key_fn_group_key_map_missing_stems_raises():
    with pytest.raises(ValueError, match="missing"):
        resolve_group_key_fn("tile_prefix", ["a", "b"], group_key_map={"a": "g1"})


def test_resolve_group_key_fn_group_key_map_used_when_it_covers_every_stem():
    fn = resolve_group_key_fn("tile_prefix", ["a", "b"], group_key_map={"a": "g1", "b": "g1"})
    assert fn("a") == fn("b") == "g1"


def test_resolve_group_key_fn_named_policy_still_works():
    fn = resolve_group_key_fn("tile_prefix", ["a_0_0"])
    assert fn("a_0_0") == "a"
    assert resolve_group_key_fn("stem", ["a_0_0"])("a_0_0") == "a_0_0"


# --- cal_holdout_split (K1) ---

def test_cal_holdout_split_remaps_train_val_to_calibration_holdout():
    stems = _grouped(6)
    parts = cal_holdout_split(stems, holdout_ratio=0.5, seed=1)
    assert set(parts) == {"calibration", "holdout"}
    assert sorted(parts["calibration"] + parts["holdout"]) == sorted(stems)


# --- resolve_locked_cal_holdout_split (K1) ---

def test_resolve_locked_cal_holdout_split_group_straddle():
    # Group "m" has two tiles; many other stems sort alphabetically BETWEEN them, so a naive
    # lexicographic midpoint cut (the pre-K1 behavior at every calibration call site) would split
    # them across cal/holdout. The locked, group-aware split must not.
    stems = ["m_0_0"] + [f"g{i}_0_0" for i in range(8)] + ["m_9_9"]
    locked = resolve_locked_cal_holdout_split(stems, identity_hash="straddle-test", seed=1)
    cal, hold = set(locked["calibration"]), set(locked["holdout"])
    assert ("m_0_0" in cal) == ("m_9_9" in cal)
    assert ("m_0_0" in hold) == ("m_9_9" in hold)


def test_resolve_locked_cal_holdout_split_stable_across_redeclared_policy():
    stems = _grouped(6)
    first = resolve_locked_cal_holdout_split(stems, identity_hash="stable-test", seed=1)
    # A later call declaring a DIFFERENT seed/group_by, with no force_redraw, must not redraw.
    second = resolve_locked_cal_holdout_split(
        stems, identity_hash="stable-test", seed=99, group_by="stem")
    assert first["calibration"] == second["calibration"]
    assert first["holdout"] == second["holdout"]


def test_resolve_locked_cal_holdout_split_persists_lock_file():
    stems = _grouped(4)
    resolve_locked_cal_holdout_split(stems, identity_hash="persist-test", seed=3)
    assert cal_holdout_lock_path("persist-test").is_file()


def test_resolve_locked_cal_holdout_split_group_key_map_produces_working_split():
    # Rail-admits-valid-work: a valid group_key_map covering every stem still produces a usable
    # locked split (not just a raise-on-bad-input path).
    stems = ["p1", "p2", "p3", "p4"]
    group_key_map = {"p1": "gA", "p2": "gA", "p3": "gB", "p4": "gB"}
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash="map-test", group_by="ignored", group_key_map=group_key_map, seed=2)
    cal, hold = set(locked["calibration"]), set(locked["holdout"])
    assert ("p1" in cal) == ("p2" in cal)  # gA never straddles
    assert ("p3" in cal) == ("p4" in cal)  # gB never straddles
    assert cal | hold == set(stems)


def test_resolve_locked_cal_holdout_split_force_redraw_records_history():
    stems = _grouped(6)
    first = resolve_locked_cal_holdout_split(stems, identity_hash="redraw-test", seed=1)
    assert len(first["redraw_history"]) == 1
    assert first["redraw_history"][0]["old_content_hash"] is None  # nothing existed before it

    second = resolve_locked_cal_holdout_split(
        stems, identity_hash="redraw-test", seed=2, force_redraw=True,
        timestamp="2026-01-01T00:00:00Z")
    assert len(second["redraw_history"]) == 2  # the first draw + this redraw, never dropped
    entry = second["redraw_history"][-1]
    assert entry["timestamp"] == "2026-01-01T00:00:00Z"
    assert entry["old_content_hash"] is not None  # the OLD (first) split's membership, captured
    assert entry["policy"]["seed"] == 2

    # Re-running with the SAME (now-locked) policy and no force_redraw returns it unchanged.
    third = resolve_locked_cal_holdout_split(stems, identity_hash="redraw-test", seed=2)
    assert third == second

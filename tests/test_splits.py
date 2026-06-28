"""Tests for the group-aware / annotation-stratified split utility (torch-free)."""

from __future__ import annotations

from collections import defaultdict

import pytest

from tcip_mcp.pipelines.data.splits import (
    default_group_key,
    GROUP_KEY_FNS,
    count_lines,
    group_balanced_split,
)


def test_default_group_key_strips_tile_offset():
    assert default_group_key("canopyA_128_256") == "canopyA"
    assert default_group_key("plainname") == "plainname"
    # A single trailing "_<int>" field does NOT match the two-field tile pattern.
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

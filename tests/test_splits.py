"""Tests for the group-aware / annotation-stratified split utility (torch-free)."""

from __future__ import annotations

from collections import defaultdict

import pytest

from tcip_mcp.pipelines.data.splits import (
    default_group_key,
    GROUP_KEY_FNS,
    group_balanced_split,
    label_document_extent,
    resolve_group_key_fn,
    spatial_strip_identity,
    spatial_strip_split,
    stem_of_spatial_identity,
)
from tcip_mcp.pipelines.slicing import slice_lattice


def test_default_group_key_strips_tile_offset():
    assert default_group_key("canopyA_128_256") == "canopyA"
    assert default_group_key("plainname") == "plainname"
    # A single trailing "_<int>" field does not match the two-field tile pattern.
    assert default_group_key("img_001") == "img_001"
    assert GROUP_KEY_FNS["stem"]("a_1_2") == "a_1_2"


def _grouped(stems):
    return [f"src{g}_{r}_0" for g in range(stems) for r in range(3)]


_SIDES = {"train": 0.6, "val": 0.2, "calibration": 0.2}


def _split(stems, *, counts=None, splits=_SIDES, seed=1, weighted=True, minimums=None):
    """The draw over ``stems`` grouped by tile prefix, every member counted (one each unless
    ``counts`` says otherwise), at a minimum of one foreground group per side unless stated."""
    return group_balanced_split(
        stems, counts=counts if counts is not None else dict.fromkeys(stems, 1),
        weighted=weighted, group_key_fn=default_group_key, splits=splits, seed=seed,
        min_foreground_groups=minimums if minimums is not None else dict.fromkeys(splits, 1))


def test_group_split_no_group_spans_splits():
    stems = _grouped(6)  # 6 sources x 3 tiles
    parts = _split(stems)
    group_to_splits = defaultdict(set)
    for split, ss in parts.items():
        for s in ss:
            group_to_splits[default_group_key(s)].add(split)
    assert all(len(v) == 1 for v in group_to_splits.values())


def test_group_split_deterministic_and_partition():
    stems = _grouped(8)
    a = _split(stems, seed=7)
    b = _split(stems, seed=7)
    assert a == b
    union = a["train"] + a["val"] + a["calibration"]
    assert sorted(union) == sorted(stems)
    assert len(union) == len(set(union)) == len(stems)


def test_group_split_foreground_stratification():
    fg = [f"fg{g}_{r}_0" for g in range(4) for r in range(2)]
    bg = [f"bg{g}_0_0" for g in range(4)]
    stems = fg + bg
    counts = {s: (2 if s.startswith("fg") else 0) for s in stems}
    parts = _split(stems, counts=counts, splits={"train": 0.6, "val": 0.4}, seed=3)
    assert set(parts) == {"train", "val"}

    def has_fg(ss):
        return any(counts[s] > 0 for s in ss)

    assert has_fg(parts["train"]) and has_fg(parts["val"])


def test_group_split_refuses_a_member_with_no_measured_count():
    """A member the counts do not name refuses at the algorithm rather than balancing as a
    background member."""
    stems = _grouped(4)
    counts = {s: 1 for s in stems[1:]}
    with pytest.raises(ValueError, match="no foreground count"):
        _split(stems, counts=counts)


def test_group_split_calibration_side_gets_its_stated_minimum():
    """Over six foreground groups at a skewed (0.6, 0.2, 0.2) ratio, a one-per-side minimum
    lands 4/1/1 (train's floor absorbs every group the balancing pass would otherwise give the
    smaller sides); stating a per-side minimum of two for the third side raises its own floor to
    two groups, taking from train's share instead."""
    stems = _grouped(6)

    default_parts = _split(stems)
    assert len({default_group_key(s) for s in default_parts["train"]}) == 4
    assert len({default_group_key(s) for s in default_parts["val"]}) == 1
    assert len({default_group_key(s) for s in default_parts["calibration"]}) == 1

    stated_parts = _split(stems, minimums={"train": 1, "val": 1, "calibration": 2})
    assert len({default_group_key(s) for s in stated_parts["calibration"]}) == 2


def test_refuse_insufficient_foreground_groups_admits_a_sufficient_tree():
    from tcip_mcp.pipelines.data.splits import refuse_insufficient_foreground_groups

    refuse_insufficient_foreground_groups(
        4, {"train": 1, "val": 1, "calibration": 2}, remedy="annotate more.")


def test_refuse_insufficient_foreground_groups_names_the_sides_and_the_shortfall():
    from tcip_mcp.pipelines.data.splits import refuse_insufficient_foreground_groups

    with pytest.raises(ValueError) as exc_info:
        refuse_insufficient_foreground_groups(
            2, {"train": 1, "val": 1, "calibration": 2}, remedy="annotate more.")
    message = str(exc_info.value)
    assert "train=1" in message and "val=1" in message and "calibration=2" in message
    assert "2 foreground group" in message


def test_refuse_insufficient_foreground_groups_carries_the_callers_own_remedy():
    """The caller states what to add, since what counts as foreground differs by the ground
    truth a draw reads."""
    from tcip_mcp.pipelines.data.splits import refuse_insufficient_foreground_groups

    with pytest.raises(ValueError) as exc_info:
        refuse_insufficient_foreground_groups(
            2, {"train": 1, "val": 1, "calibration": 2},
            remedy="annotate or confirm more foreground groups of this subject.")
    message = str(exc_info.value)
    assert "drop a side" not in message
    assert "annotate or confirm more foreground groups" in message


def test_group_split_with_no_foreground_balances_by_tile_count():
    stems = _grouped(4)
    parts = _split(stems, counts=dict.fromkeys(stems, 0), splits={"train": 0.7, "val": 0.3},
                   seed=5)
    assert parts["train"] and parts["val"]


# --- resolve_group_key_fn ---

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


# --- spatial_strip_split ---

def _kept_tiles(split, width, height):
    """Every kept tile's rect, tagged with the side whose region contains it, recomputed
    independently of the split's own kept-tile counters."""
    from tcip_mcp.pipelines.raster_source import rect_contains_rect

    lattice = slice_lattice(height, width, split.tile_size, split.overlap)
    by_name: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in split.regions}
    for box in lattice:
        for name, rects in split.regions.items():
            if any(rect_contains_rect(r, box) for r in rects):
                by_name[name].append(box)
    return by_name


def test_spatial_strip_split_admits_valid_work():
    split = spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.7, 0.3, 0.0))
    assert split.kept_tiles["train"] > 0
    assert split.kept_tiles["val"] > 0
    assert 0.0 < split.realized_fractions["val"] < 1.0


def test_spatial_strip_split_accounts_for_every_tile():
    split = spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.7, 0.3, 0.0))
    total = (sum(split.kept_tiles.values())
             + split.tiles_dropped_past_extent + split.tiles_dropped_outside_regions)
    assert total == split.total_tiles


def test_spatial_strip_split_no_tile_shared_and_buffer_respected():
    width, height = 4000, 3000
    split = spatial_strip_split(width, height, 320, 0.2, fractions=(0.7, 0.3, 0.0))
    by_name = _kept_tiles(split, width, height)
    train, val = by_name["train"], by_name["val"]
    assert train and val
    for tx0, ty0, tx1, ty1 in train:
        for vx0, vy0, vx1, vy1 in val:
            gap_x = max(vx0 - tx1, tx0 - vx1, 0)
            gap_y = max(vy0 - ty1, ty0 - vy1, 0)
            # Never overlapping, offset on some axis; that offset is the real gap.
            assert gap_x > 0 or gap_y > 0
            assert max(gap_x, gap_y) >= split.buffer


def test_center_out_order_ties_resolve_in_declared_order():
    """Coverage of ``_center_out_order`` directly: this documents ``_center_out_order`` itself
    rather than guarding a regression; the property is proven through
    :func:`spatial_strip_split` in the test below.

    Two equal-share slots resolve center-out in the order they were given, whichever order
    that is: swapping the input order swaps which cardinal side each lands on, so the tie is
    the caller's declared order, never a coin flip."""
    from tcip_mcp.pipelines.data.splits import _center_out_order

    forward = _center_out_order([("val", 0.2), ("test", 0.2), ("train", 0.6)])
    reversed_input = _center_out_order([("test", 0.2), ("val", 0.2), ("train", 0.6)])

    assert [name for name, _ in forward] == ["val", "train", "test"]
    assert [name for name, _ in reversed_input] == ["test", "train", "val"]


def test_spatial_strip_split_tied_shares_place_by_declared_order():
    """Coverage of the fixed tie-break: equal shares resolve center-out by their position in
    the declared ``split_names`` order, so permuting which of the two tied names comes second
    swaps which cardinal side that name lands on, mirroring the swap in the declared order, and
    the returned split carries no ``seed`` field."""
    # 4160 = 320 + 15 * 256: the lattice ends flush, so no pulled-back slice crowds an edge side.
    forward = spatial_strip_split(4160, 3000, 320, 0.2, fractions=(0.6, 0.2, 0.2),
                                   split_names=("train", "val", "test"))
    swapped = spatial_strip_split(4160, 3000, 320, 0.2, fractions=(0.6, 0.2, 0.2),
                                   split_names=("train", "test", "val"))
    assert forward.regions["val"] != forward.regions["test"]
    assert forward.regions["val"] == swapped.regions["test"]
    assert forward.regions["test"] == swapped.regions["val"]
    assert not hasattr(forward, "seed")


def test_spatial_strip_split_deterministic():
    a = spatial_strip_split(8000, 6000, 320, 0.2, fractions=(0.7, 0.2, 0.1))
    b = spatial_strip_split(8000, 6000, 320, 0.2, fractions=(0.7, 0.2, 0.1))
    assert a == b


def test_spatial_strip_split_buffer_defaults_to_tile_size():
    split = spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.8, 0.2, 0.0))
    assert split.buffer == 320


def test_spatial_strip_split_explicit_buffer_below_tile_size_refuses():
    with pytest.raises(ValueError, match="buffer"):
        spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.8, 0.2, 0.0), buffer=100)


def test_spatial_strip_split_refuses_when_no_tile_fits_extent():
    with pytest.raises(ValueError, match="no tile fits"):
        spatial_strip_split(100, 100, 320, 0.2, fractions=(0.8, 0.2, 0.0))


def test_spatial_strip_split_refuses_when_geometry_is_too_tight():
    # An image barely larger than one tile: nothing survives the buffer margin on a second side.
    with pytest.raises(ValueError, match="no strip layout"):
        spatial_strip_split(340, 340, 320, 0.2, fractions=(0.8, 0.2, 0.0))


def test_spatial_strip_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.7, 0.2, 0.2))


def test_spatial_strip_split_three_way_populates_every_side():
    split = spatial_strip_split(8000, 6000, 320, 0.2, fractions=(0.7, 0.2, 0.1))
    assert split.kept_tiles["train"] > 0
    assert split.kept_tiles["val"] > 0
    assert split.kept_tiles["test"] > 0


def test_spatial_strip_split_realized_fractions_stay_near_requested():
    # A generous but real tolerance: catches a regression to arbitrary quantization, not
    # merely "nonzero on every side".
    split = spatial_strip_split(16000, 12000, 320, 0.2, fractions=(0.7, 0.2, 0.1))
    for name, requested in zip(split.split_names, split.requested_fractions):
        assert abs(split.realized_fractions[name] - requested) < 0.05


def test_spatial_strip_identity_roundtrip():
    identity = spatial_strip_identity("mosaic_north_02", "strip_x_1")
    assert identity == "mosaic_north_02::strip_x_1"
    assert stem_of_spatial_identity(identity) == "mosaic_north_02"


def test_stem_of_spatial_identity_passes_through_a_non_spatial_string():
    assert stem_of_spatial_identity("plain_stem_0_0") == "plain_stem_0_0"


def test_label_document_extent_reads_the_frame_off_the_document_it_is_handed(tmp_path):
    """The extent comes from the path a sample records, never from a name composed under a
    directory: a split derived here reads the same file the loader will."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    json_io.write_annotations(
        str(labels_dir / "mosaic1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 4000, 3000,
    )
    assert label_document_extent(labels_dir / "mosaic1.json") == (4000, 3000)
    with pytest.raises(json_io.UnreadableLabelDocument, match="missing.json"):
        label_document_extent(labels_dir / "missing.json")


# -- scope normalization ---------------------------------------------------------


def test_count_label_lines_reads_an_empty_attribute_as_unset(tmp_path):
    """A scope stated with ``attribute=""`` means "no attribute", so every record of the subject
    counts; reading it as a key no annotation carries would score every stem zero and starve the
    minimum-foreground pass."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.data.splits import count_label_lines

    labels_dir = tmp_path / "annotations"
    labels_dir.mkdir()
    json_io.write_annotations(
        labels_dir / "a.json", [Annotation(subject="leaf", geometry=BBox(1, 1, 5, 5))], 32, 32)

    document = labels_dir / "a.json"
    assert count_label_lines(document, ClassScope(subject="leaf", attribute="")) == 1
    assert count_label_lines(document, ClassScope(subject="", attribute="")) == 1
    assert count_label_lines(document, ClassScope(subject="leaf", attribute="condition")) == 0


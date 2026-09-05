"""Tests for the group-aware / annotation-stratified split utility (torch-free)."""

from __future__ import annotations

from collections import defaultdict

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.data.splits import (
    default_group_key,
    GROUP_KEY_FNS,
    cal_holdout_lock_key,
    cal_holdout_split,
    count_lines,
    group_balanced_split,
    image_extent_from_labels,
    resolve_group_key_fn,
    resolve_locked_cal_holdout_split,
    spatial_strip_identity,
    spatial_strip_split,
    stem_of_spatial_identity,
)
from tcip_mcp.pipelines.data.tiling import compute_stride, tile_positions, tile_within_extent


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
    union = a["train"] + a["val"] + a["calibration"]
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
    assert parts["calibration"] == []  # 0.0 fraction -> empty

    def has_fg(ss):
        return any(counts[s] > 0 for s in ss)

    assert has_fg(parts["train"]) and has_fg(parts["val"])


def test_group_split_calibration_side_gets_its_stated_minimum():
    """Over six foreground groups at a skewed (0.6, 0.2, 0.2) ratio, the default one-per-side
    minimum lands 4/1/1 (train's floor absorbs every group the balancing pass would otherwise
    give the smaller sides); stating a per-side minimum of two for the third side raises its own
    floor to two groups, taking from train's share instead."""
    stems = _grouped(6)
    counts = {s: 1 for s in stems}

    default_parts = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.6, 0.2, 0.2), seed=1)
    assert len({default_group_key(s) for s in default_parts["train"]}) == 4
    assert len({default_group_key(s) for s in default_parts["val"]}) == 1
    assert len({default_group_key(s) for s in default_parts["calibration"]}) == 1

    stated_parts = group_balanced_split(
        stems, annotation_counts=counts, splits=(0.6, 0.2, 0.2), seed=1,
        min_foreground_groups={"train": 1, "val": 1, "calibration": 2},
    )
    assert len({default_group_key(s) for s in stated_parts["calibration"]}) == 2


def test_refuse_insufficient_foreground_groups_admits_a_sufficient_tree():
    from tcip_mcp.pipelines.data.splits import refuse_insufficient_foreground_groups

    refuse_insufficient_foreground_groups(4, {"train": 1, "val": 1, "calibration": 2})


def test_refuse_insufficient_foreground_groups_names_the_sides_and_the_shortfall():
    from tcip_mcp.pipelines.data.splits import refuse_insufficient_foreground_groups

    with pytest.raises(ValueError) as exc_info:
        refuse_insufficient_foreground_groups(2, {"train": 1, "val": 1, "calibration": 2})
    message = str(exc_info.value)
    assert "train=1" in message and "val=1" in message and "calibration=2" in message
    assert "2 foreground group" in message


def test_refuse_insufficient_foreground_groups_remedy_names_no_ratio_escape():
    """The remedy names annotating or confirming more foreground; a manifest write requires
    every side's ratio non-zero, so no side can be dropped by zeroing its ratio."""
    from tcip_mcp.pipelines.data.splits import refuse_insufficient_foreground_groups

    with pytest.raises(ValueError) as exc_info:
        refuse_insufficient_foreground_groups(2, {"train": 1, "val": 1, "calibration": 2})
    message = str(exc_info.value)
    assert "drop a side" not in message
    assert "annotate or confirm more foreground groups" in message


def test_calibration_universe_from_manifest_floor_remedy_names_ratio_and_date():
    """The composed refusal ends with a redraw remedy naming a larger calibration ratio or more
    foreground groups on the date, never the whole directory: a manifest write refuses a zero
    calibration_ratio by name, so that fallback names an action no caller can take."""
    from tcip_mcp.pipelines.data.splits import calibration_universe_from_manifest

    manifest = {
        "splits": {"train": ["d1/a"], "val": ["d1/b"], "calibration": ["d1/c"]},
        "group_by": "stem",
    }
    with pytest.raises(ValueError) as exc_info:
        calibration_universe_from_manifest(manifest, "d1", {"c"}, foreground_stems=set())
    message = str(exc_info.value)
    assert "whole directory" not in message
    assert "calibration_ratio" in message
    assert "'d1'" in message


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


# --- cal_holdout_split ---

def test_cal_holdout_split_remaps_train_val_to_calibration_holdout():
    stems = _grouped(6)
    parts = cal_holdout_split(stems, holdout_ratio=0.5, seed=1)
    assert set(parts) == {"calibration", "holdout"}
    assert sorted(parts["calibration"] + parts["holdout"]) == sorted(stems)


# --- resolve_locked_cal_holdout_split ---

def test_resolve_locked_cal_holdout_split_group_straddle(tmp_path):
    # Group "m" has two tiles; many other stems sort alphabetically between them, so a naive
    # lexicographic midpoint cut would split them across cal/holdout. The locked, group-aware
    # split must not.
    stems = ["m_0_0"] + [f"g{i}_0_0" for i in range(8)] + ["m_9_9"]
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash="straddle-test", scope_root=tmp_path, seed=1)
    cal, hold = set(locked["calibration"]), set(locked["holdout"])
    assert ("m_0_0" in cal) == ("m_9_9" in cal)
    assert ("m_0_0" in hold) == ("m_9_9" in hold)


def test_resolve_locked_cal_holdout_split_stable_across_redeclared_policy(tmp_path):
    stems = _grouped(6)
    first = resolve_locked_cal_holdout_split(
        stems, identity_hash="stable-test", scope_root=tmp_path, seed=1)
    # A later call declaring a different seed/group_by, with no force_redraw, must not redraw.
    second = resolve_locked_cal_holdout_split(
        stems, identity_hash="stable-test", scope_root=tmp_path, seed=99, group_by="stem")
    assert first["calibration"] == second["calibration"]
    assert first["holdout"] == second["holdout"]


def test_resolve_locked_cal_holdout_split_persists_lock_file(tmp_path):
    stems = _grouped(4)
    resolve_locked_cal_holdout_split(
        stems, identity_hash="persist-test", scope_root=tmp_path, seed=3)
    assert ts.exists(cal_holdout_lock_key("persist-test", scope_root=tmp_path))


def test_resolve_locked_cal_holdout_split_group_key_map_produces_working_split(tmp_path):
    # Rail-admits-valid-work: a valid group_key_map covering every stem still produces a usable
    # locked split (not just a raise-on-bad-input path).
    stems = ["p1", "p2", "p3", "p4"]
    group_key_map = {"p1": "gA", "p2": "gA", "p3": "gB", "p4": "gB"}
    locked = resolve_locked_cal_holdout_split(
        stems, identity_hash="map-test", scope_root=tmp_path, group_by="ignored",
        group_key_map=group_key_map, seed=2)
    cal, hold = set(locked["calibration"]), set(locked["holdout"])
    assert ("p1" in cal) == ("p2" in cal)  # gA never straddles
    assert ("p3" in cal) == ("p4" in cal)  # gB never straddles
    assert cal | hold == set(stems)


def test_resolve_locked_cal_holdout_split_force_redraw_records_history(tmp_path):
    stems = _grouped(6)
    first = resolve_locked_cal_holdout_split(
        stems, identity_hash="redraw-test", scope_root=tmp_path, seed=1)
    assert len(first["redraw_history"]) == 1
    assert first["redraw_history"][0]["old_content_hash"] is None  # nothing existed before it

    second = resolve_locked_cal_holdout_split(
        stems, identity_hash="redraw-test", scope_root=tmp_path, seed=2, force_redraw=True,
        timestamp="2026-01-01T00:00:00Z")
    assert len(second["redraw_history"]) == 2  # the first draw + this redraw, never dropped
    entry = second["redraw_history"][-1]
    assert entry["timestamp"] == "2026-01-01T00:00:00Z"
    assert entry["old_content_hash"] is not None  # the old (first) split's membership, captured
    assert entry["policy"]["seed"] == 2

    # Re-running with the same (now-locked) policy and no force_redraw returns it unchanged.
    third = resolve_locked_cal_holdout_split(
        stems, identity_hash="redraw-test", scope_root=tmp_path, seed=2)
    assert third == second


def test_lock_survives_an_active_project_repin(tmp_path, monkeypatch):
    """A locked split belongs to the dataset it was drawn over, not to the adopted project.

    ``activate_project`` repins the platform state root inside a live process. A lock scoped to
    that root reads as absent once it moves, and the next call cuts a fresh split for the same
    identity, which is the silent re-cut this lock exists to prevent.
    """
    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()
    stems = _grouped(8)
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "before_adoption"))
    first = resolve_locked_cal_holdout_split(
        stems, identity_hash="repin-test", scope_root=dataset_root, seed=1)

    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path / "adopted_project"))
    after = resolve_locked_cal_holdout_split(
        stems, identity_hash="repin-test", scope_root=dataset_root, seed=2, group_by="stem")

    assert after["calibration"] == first["calibration"]
    assert after["holdout"] == first["holdout"]
    # The lock was read, not redrawn: the declared policy is reported as diverging from it.
    assert after["policy_divergence"]["locked"]["seed"] == 1
    assert ts.exists(cal_holdout_lock_key("repin-test", scope_root=dataset_root))


# --- spatial_strip_split ---

def _kept_tiles(split, width, height):
    """Every kept tile's rect, tagged with the side it was assigned to via ``split_name_for``,
    recomputed independently of the split's own kept-tile counters."""
    stride = compute_stride(split.tile_size, split.overlap)
    lattice = tile_positions(height, width, split.tile_size, stride)
    in_extent = [(tx, ty) for tx, ty in lattice
                if tile_within_extent(tx, ty, split.tile_size, width, height)]
    by_name: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in split.regions}
    for tx, ty in in_extent:
        name = split.split_name_for(tx, ty)
        if name is not None:
            by_name[name].append((tx, ty, tx + split.tile_size, ty + split.tile_size))
    return by_name


def test_spatial_strip_split_admits_valid_work():
    split = spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.7, 0.3, 0.0), seed=3)
    assert split.kept_tiles["train"] > 0
    assert split.kept_tiles["val"] > 0
    assert 0.0 < split.realized_fractions["val"] < 1.0


def test_spatial_strip_split_accounts_for_every_tile():
    split = spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.7, 0.3, 0.0), seed=3)
    total = (sum(split.kept_tiles.values())
             + split.tiles_dropped_past_extent + split.tiles_dropped_outside_regions)
    assert total == split.total_tiles


def test_spatial_strip_split_no_tile_shared_and_buffer_respected():
    width, height = 4000, 3000
    split = spatial_strip_split(width, height, 320, 0.2, fractions=(0.7, 0.3, 0.0), seed=5)
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


def test_spatial_strip_split_deterministic():
    a = spatial_strip_split(8000, 6000, 320, 0.2, fractions=(0.7, 0.2, 0.1), seed=11)
    b = spatial_strip_split(8000, 6000, 320, 0.2, fractions=(0.7, 0.2, 0.1), seed=11)
    assert a == b


def test_spatial_strip_split_buffer_defaults_to_tile_size():
    split = spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.8, 0.2, 0.0), seed=1)
    assert split.buffer == 320


def test_spatial_strip_split_explicit_buffer_below_tile_size_refuses():
    with pytest.raises(ValueError, match="buffer"):
        spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.8, 0.2, 0.0), seed=1, buffer=100)


def test_spatial_strip_split_refuses_when_no_tile_fits_extent():
    with pytest.raises(ValueError, match="no tile fits"):
        spatial_strip_split(100, 100, 320, 0.2, fractions=(0.8, 0.2, 0.0), seed=1)


def test_spatial_strip_split_refuses_when_geometry_is_too_tight():
    # An image barely larger than one tile: nothing survives the buffer margin on a second side.
    with pytest.raises(ValueError, match="no strip layout"):
        spatial_strip_split(340, 340, 320, 0.2, fractions=(0.8, 0.2, 0.0), seed=1)


def test_spatial_strip_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        spatial_strip_split(4000, 3000, 320, 0.2, fractions=(0.7, 0.2, 0.2), seed=1)


def test_spatial_strip_split_three_way_populates_every_side():
    split = spatial_strip_split(8000, 6000, 320, 0.2, fractions=(0.7, 0.2, 0.1), seed=3)
    assert split.kept_tiles["train"] > 0
    assert split.kept_tiles["val"] > 0
    assert split.kept_tiles["test"] > 0


def test_spatial_strip_split_realized_fractions_stay_near_requested():
    # A generous but real tolerance: catches a regression to arbitrary quantization, not
    # merely "nonzero on every side".
    split = spatial_strip_split(16000, 12000, 320, 0.2, fractions=(0.7, 0.2, 0.1), seed=3)
    for name, requested in zip(split.split_names, split.requested_fractions):
        assert abs(split.realized_fractions[name] - requested) < 0.05


def test_spatial_strip_identity_roundtrip():
    identity = spatial_strip_identity("mosaic_north_02", "strip_x_1")
    assert identity == "mosaic_north_02::strip_x_1"
    assert stem_of_spatial_identity(identity) == "mosaic_north_02"


def test_stem_of_spatial_identity_passes_through_a_non_spatial_string():
    assert stem_of_spatial_identity("plain_stem_0_0") == "plain_stem_0_0"


def test_image_extent_from_labels(tmp_path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    json_io.write_annotations(
        str(labels_dir / "mosaic1.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 4000, 3000,
    )
    assert image_extent_from_labels(labels_dir, "mosaic1") == (4000, 3000)
    assert image_extent_from_labels(labels_dir, "missing") is None


# -- manifest_scope_issues / normalize_scope -------------------------------------


def _leaf_dataset(root, *, date: str | None):
    """A minimal ``leaf``-labeled tree ``draw_splits`` can draw from: a dated
    ``images/<date>/`` + ``annotations/<date>/`` pair when ``date`` is given, a flat
    ``images/`` + ``annotations/`` pair otherwise."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry

    write_registry(root / "classes.json", ClassRegistry(subjects=(Subject(name="leaf"),)))
    images_dir = root / "images" / date if date else root / "images"
    labels_dir = root / "annotations" / date if date else root / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for stem in ("a", "b", "c", "d"):
        Image.new("RGB", (64, 64), (100, 120, 90)).save(images_dir / f"{stem}.jpg")
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="leaf", geometry=BBox(4, 4, 20, 20))], 64, 64, keep_empty=True,
        )
    return images_dir, labels_dir


def _draw_leaf_manifest(root, out, *, date: str | None):
    from tcip_mcp.tools.data_tools import draw_splits, split_manifest_key

    result = draw_splits(str(root), output_path=str(out), subject="leaf", seed=1,
                         train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25)
    assert "error" not in result, result
    return ts.read(split_manifest_key(out))


def test_normalize_scope_maps_empty_string_to_none():
    from tcip_mcp.pipelines.data.splits import normalize_scope

    assert normalize_scope("", "") == (None, None)
    assert normalize_scope("leaf", "") == ("leaf", None)
    assert normalize_scope(None, "condition") == (None, "condition")


def test_manifest_scope_issues_reports_every_objection_together(tmp_path):
    from tcip_mcp.pipelines.data.splits import manifest_scope_issues

    root = tmp_path / "ds"
    date = "2-11-26"
    _leaf_dataset(root, date=date)
    manifest = _draw_leaf_manifest(root, tmp_path / "m", date=date)

    issues, narrowing = manifest_scope_issues(
        manifest, subject="other", attribute=None, date=date, images_dir=None,
        label="data.images_dir",
    )

    assert any("subject" in i for i in issues)
    assert any("images_dir" in i for i in issues)
    assert len(issues) >= 2
    assert narrowing is not None


def test_manifest_scope_issues_names_the_manifest_directory_when_given(tmp_path):
    from tcip_mcp.pipelines.data.splits import manifest_scope_issues

    root = tmp_path / "ds"
    date = "2-11-26"
    _leaf_dataset(root, date=date)
    out = tmp_path / "m"
    manifest = _draw_leaf_manifest(root, out, date=date)

    unnamed, _ = manifest_scope_issues(
        manifest, subject="leaf", attribute=None, date="2099-01-01", images_dir=None,
        label="data.images_dir",
    )
    assert any(i.startswith("the split manifest holds no members") for i in unnamed)
    assert any("Regenerate the split over this date" in i for i in unnamed)

    named, _ = manifest_scope_issues(
        manifest, subject="leaf", attribute=None, date="2099-01-01", images_dir=None,
        label="data.images_dir", manifest_dir=str(out),
    )
    assert any(i.startswith(f"split manifest at {str(out)!r} holds no members") for i in named)


def test_manifest_scope_issues_admits_a_draw_splits_manifest_dated_and_flat(tmp_path):
    from tcip_mcp.pipelines.data.splits import manifest_scope_issues

    dated_root = tmp_path / "dated"
    date = "2-11-26"
    dated_images, _ = _leaf_dataset(dated_root, date=date)
    dated_manifest = _draw_leaf_manifest(dated_root, tmp_path / "m-dated", date=date)
    issues, narrowing = manifest_scope_issues(
        dated_manifest, subject="leaf", attribute=None, date=date,
        images_dir=str(dated_images), label="data.images_dir",
    )
    assert issues == []
    assert narrowing is not None

    flat_root = tmp_path / "flat"
    flat_images, _ = _leaf_dataset(flat_root, date=None)
    flat_manifest = _draw_leaf_manifest(flat_root, tmp_path / "m-flat", date=None)
    issues, narrowing = manifest_scope_issues(
        flat_manifest, subject="leaf", attribute=None, date=None,
        images_dir=str(flat_images), label="data.images_dir",
    )
    assert issues == []
    assert narrowing is not None

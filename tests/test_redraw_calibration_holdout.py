"""redraw_calibration_holdout: the audited admin path to redraw a locked cal/holdout split.
A locked split can only be redrawn deliberately, with a recorded reason and an old->new
membership diff, never silently, never automatically.
"""

from __future__ import annotations

from pathlib import Path


def _audit_events(root: Path, tool: str) -> list[dict]:
    import tcip_store

    from tcip_mcp.audit import audit_log_key

    page = tcip_store.read_log(audit_log_key(root))
    return [entry for entry in page.records if entry.get("tool") == tool]


def test_force_redraw_requires_nonempty_reason(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    assert "error" in redraw_calibration_holdout(
        dataset_root=str(tmp_path), identity_hash="abc123", reason="")
    assert "error" in redraw_calibration_holdout(
        dataset_root=str(tmp_path), identity_hash="abc123", reason="   ")


def test_force_redraw_requires_labels_dir_or_identity_hash(tmp_path: Path):
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    r = redraw_calibration_holdout(dataset_root=str(tmp_path), reason="need to redraw")
    assert "error" in r


def test_force_redraw_records_old_to_new_membership_diff(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    stems = [f"src{g}_{r}_0" for g in range(6) for r in range(3)]
    first = resolve_locked_cal_holdout_split(
        stems, identity_hash="redraw-tool-test", scope_root=tmp_path, seed=1)

    result = redraw_calibration_holdout(
        dataset_root=str(tmp_path), identity_hash="redraw-tool-test", seed=2,
        reason="original holdout coincided with the demo set")
    assert "error" not in result
    assert result["old_membership"] == {"calibration": first["calibration"],
                                        "holdout": first["holdout"]}
    assert result["new_membership"]["calibration"] or result["new_membership"]["holdout"]

    # A later, unrelated call keeps the redraw a deliberate, one-off action, not automatic.
    from tcip_store import read

    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_key
    locked_after = read(cal_holdout_lock_key("redraw-tool-test", scope_root=tmp_path))
    assert locked_after["calibration"] == result["new_membership"]["calibration"]
    assert locked_after["seed"] == 2

    # The @audited call-args line and this tool's own explicit result line are both recorded, on
    # distinct tool names: the audit record captures what the redraw actually produced, not just
    # that one was requested (@audited alone only logs kwargs, never the return value).
    call_events = _audit_events(tmp_path, "redraw_calibration_holdout")
    assert len(call_events) == 1
    assert call_events[0]["arguments"]["reason"] == "original holdout coincided with the demo set"

    result_events = _audit_events(tmp_path, "redraw_calibration_holdout_result")
    assert len(result_events) == 1
    ev = result_events[0]
    assert ev["arguments"]["reason"] == "original holdout coincided with the demo set"
    assert ev["old_membership"] == result["old_membership"]
    assert ev["new_membership"] == result["new_membership"]


def test_force_redraw_with_labels_dir_rescans_stems(tmp_path: Path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for i in range(4):
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 8, 8)

    result = redraw_calibration_holdout(
        dataset_root=str(tmp_path), labels_dir=str(labels_dir), seed=1,
        reason="first deliberate draw for this labels dir")
    assert "error" not in result
    assert sorted(result["new_membership"]["calibration"] + result["new_membership"]["holdout"]) == \
        ["img0", "img1", "img2", "img3"]


def test_force_redraw_answers_an_error_dict_over_an_unreadable_label(tmp_path: Path):
    """An unreadable label under ``labels_dir`` is an error dict naming the file, never a raw
    raise through this audited door."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    json_io.write_annotations(
        str(labels_dir / "img0.json"),
        [Annotation(subject="bud", geometry=BBox(1, 1, 5, 5))], 8, 8)
    bad = labels_dir / "img1.json"
    bad.write_bytes(b"{not json")

    result = redraw_calibration_holdout(
        dataset_root=str(tmp_path), labels_dir=str(labels_dir), seed=1,
        reason="an unreadable label must answer an error, not raise")
    assert "error" in result
    assert str(bad) in result["error"]


def test_force_redraw_refuses_a_root_the_labels_own_lock_does_not_live_under(tmp_path: Path):
    """A redraw under a root the labels do not lock against replaces a lock nothing reads.

    The calibration door scopes the lock to the labeled dir's own root, so a redraw stating a
    different root would leave the split that door keeps reading untouched while reporting success.
    """
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    labels_dir = tmp_path / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    result = redraw_calibration_holdout(
        dataset_root=str(tmp_path / "elsewhere"), labels_dir=str(labels_dir), seed=1,
        reason="stating a root these labels do not lock against")
    assert repr(str(tmp_path / "dataset")) in result["error"]
    assert repr(str(tmp_path / "elsewhere")) in result["error"]


def test_force_redraw_replaces_the_lock_the_calibration_door_drew(tmp_path: Path):
    """The redraw and the calibration door address one lock, so a redraw truly replaces it."""
    from tcip_mcp.pipelines.data.splits import (
        cal_holdout_scope_root,
        resolve_locked_cal_holdout_split,
    )
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    labels_dir = tmp_path / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    stems = [f"src{g}_{r}_0" for g in range(6) for r in range(3)]
    drawn = resolve_locked_cal_holdout_split(
        stems, identity_hash="door-agreement", scope_root=cal_holdout_scope_root(labels_dir),
        seed=1)

    result = redraw_calibration_holdout(
        dataset_root=str(tmp_path / "dataset"), identity_hash="door-agreement", seed=2,
        reason="the redraw addresses the lock the calibration drew")
    assert "error" not in result
    assert result["old_membership"] == {"calibration": drawn["calibration"],
                                        "holdout": drawn["holdout"]}


def test_a_redraw_records_in_the_log_of_the_dataset_whose_split_it_replaced(tmp_path, monkeypatch):
    """Both of the redraw entries file against the dataset the split was drawn over.

    A locked split is evidence about one dataset, so the record of replacing it travels with that
    data rather than with whatever project this process happens to be pinned to.
    """
    import tcip_mcp.audit as audit_module
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split
    from tcip_mcp.tools.calibration_tools import redraw_calibration_holdout

    platform_root = tmp_path / "platform"
    platform_root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", platform_root)
    dataset_root = tmp_path / "orchard_dataset"
    dataset_root.mkdir()
    stems = [f"src{g}_{r}_0" for g in range(6) for r in range(3)]
    resolve_locked_cal_holdout_split(stems, identity_hash="scoped", scope_root=dataset_root, seed=1)

    result = redraw_calibration_holdout(
        dataset_root=str(dataset_root), identity_hash="scoped", seed=2,
        reason="the redraw travels with the data its split was drawn over")

    assert "error" not in result
    for tool in ("redraw_calibration_holdout", "redraw_calibration_holdout_result"):
        assert len(_audit_events(dataset_root, tool)) == 1, tool
        assert _audit_events(platform_root, tool) == [], tool
    call_entry = _audit_events(dataset_root, "redraw_calibration_holdout")[0]
    assert call_entry["scope"] == str(dataset_root.resolve())

"""force_redraw_cal_holdout_split: the audited admin path to redraw a locked cal/holdout split.
A locked split can only be redrawn deliberately, with a recorded reason and an old->new
membership diff, never silently, never automatically.
"""

from __future__ import annotations

import json
from pathlib import Path


def _audit_events(root: Path, tool: str) -> list[dict]:
    path = root / ".tcip" / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if json.loads(x).get("tool") == tool]


def test_force_redraw_requires_nonempty_reason():
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    assert "error" in force_redraw_cal_holdout_split(identity_hash="abc123", reason="")
    assert "error" in force_redraw_cal_holdout_split(identity_hash="abc123", reason="   ")


def test_force_redraw_requires_labels_dir_or_identity_hash():
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    r = force_redraw_cal_holdout_split(reason="need to redraw")
    assert "error" in r


def test_force_redraw_records_old_to_new_membership_diff(tmp_path: Path):
    from tcip_mcp.pipelines.data.splits import resolve_locked_cal_holdout_split
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    stems = [f"src{g}_{r}_0" for g in range(6) for r in range(3)]
    first = resolve_locked_cal_holdout_split(stems, identity_hash="redraw-tool-test", seed=1)

    result = force_redraw_cal_holdout_split(
        identity_hash="redraw-tool-test", seed=2,
        reason="original holdout coincided with the demo set")
    assert "error" not in result
    assert result["old_membership"] == {"calibration": first["calibration"],
                                        "holdout": first["holdout"]}
    assert result["new_membership"]["calibration"] or result["new_membership"]["holdout"]

    # A later, unrelated call keeps the redraw a deliberate, one-off action, not automatic.
    from tcip_store import read

    from tcip_mcp.pipelines.data.splits import cal_holdout_lock_key
    locked_after = read(cal_holdout_lock_key("redraw-tool-test"))
    assert locked_after["calibration"] == result["new_membership"]["calibration"]
    assert locked_after["seed"] == 2

    # The @audited call-args line and this tool's own explicit result line are both recorded, on
    # distinct tool names: the audit record captures what the redraw actually produced, not just
    # that one was requested (@audited alone only logs kwargs, never the return value).
    call_events = _audit_events(tmp_path, "force_redraw_cal_holdout_split")
    assert len(call_events) == 1
    assert call_events[0]["arguments"]["reason"] == "original holdout coincided with the demo set"

    result_events = _audit_events(tmp_path, "force_redraw_cal_holdout_split_result")
    assert len(result_events) == 1
    ev = result_events[0]
    assert ev["arguments"]["reason"] == "original holdout coincided with the demo set"
    assert ev["old_membership"] == result["old_membership"]
    assert ev["new_membership"] == result["new_membership"]


def test_force_redraw_with_labels_dir_rescans_stems(tmp_path: Path):
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.inference_tools import force_redraw_cal_holdout_split

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for i in range(4):
        json_io.write_annotations(
            str(labels_dir / f"img{i}.json"),
            [Annotation(subject="catkin", geometry=BBox(1, 1, 5, 5))], 8, 8)

    result = force_redraw_cal_holdout_split(
        labels_dir=str(labels_dir), seed=1, reason="first deliberate draw for this labels dir")
    assert "error" not in result
    assert sorted(result["new_membership"]["calibration"] + result["new_membership"]["holdout"]) == \
        ["img0", "img1", "img2", "img3"]

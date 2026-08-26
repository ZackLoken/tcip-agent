"""``scripts/conform_cal_holdout_locks.py``: the one-off fix for a lock written before this
family declared ``split_manifest_dir``. The two fixture locks are authored directly, the
pre-family shape no live writer produces any more, since conforming exactly that shape is
this script's whole job."""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts

from scripts.conform_cal_holdout_locks import conform_cal_holdout_locks
from tcip_mcp.pipelines.data.splits import cal_holdout_lock_key


def _pre_family_lock(root: Path, identity_hash: str, *, with_history: bool) -> None:
    record = {
        "identity_hash": identity_hash, "calibration": ["a"], "holdout": ["b"],
        "group_by": "tile_prefix", "group_key_map": None, "seed": 0, "holdout_ratio": 0.5,
        "redraw_history": (
            [{"policy": {"group_by": "tile_prefix", "group_key_map": None, "seed": 0,
                        "holdout_ratio": 0.5},
              "seed": 0, "old_content_hash": None, "new_content_hash": "abc", "timestamp": None}]
            if with_history else []
        ),
    }
    ts.replace(cal_holdout_lock_key(identity_hash, scope_root=root), record)


def test_conforms_every_lock_and_its_redraw_history_under_the_root(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    _pre_family_lock(root, "hash_one", with_history=True)
    _pre_family_lock(root, "hash_two", with_history=False)

    count = conform_cal_holdout_locks(str(root))

    assert count == 2
    for identity_hash in ("hash_one", "hash_two"):
        record = ts.read(cal_holdout_lock_key(identity_hash, scope_root=root))
        assert record["split_manifest_dir"] is None
        for entry in record["redraw_history"]:
            assert entry["policy"]["split_manifest_dir"] is None


def test_a_record_already_conformed_is_left_untouched(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()
    _pre_family_lock(root, "hash_one", with_history=True)
    conform_cal_holdout_locks(str(root))

    second_pass = conform_cal_holdout_locks(str(root))

    assert second_pass == 0


def test_reports_zero_for_a_root_with_no_locks(tmp_path: Path):
    root = tmp_path / "ds"
    root.mkdir()

    assert conform_cal_holdout_locks(str(root)) == 0

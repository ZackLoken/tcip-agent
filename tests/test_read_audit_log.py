"""``read_audit_log``: the one tool reading a scope's own audit log back, on the record itself.

Coverage, not GUARDS: the tool is wholly new, so there is no prior behavior to prove absent.
Every entry read here comes from a real ``@audited`` call, never a hand-built dict, and the
corrupt-page case plants bytes no writer through the seam could have produced, the same shape
``tests/test_experiment_log_version_refused.py`` uses for its own log reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_mcp.audit as audit_module
import tcip_store as ts
from tcip_mcp.tools.meta_tools import read_audit_log


@pytest.fixture
def platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "platform"
    root.mkdir()
    monkeypatch.setattr(audit_module, "AUDIT_ROOT", root)
    return root


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "orchard_dataset"
    root.mkdir()
    return root


def _subjects() -> dict:
    return {"catkin": {"description": "a hazelnut catkin"}}


def test_read_audit_log_filters_by_tool_and_status_newest_first(
    platform_root: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_class_map

    ok = write_class_map(str(dataset_root), subjects=_subjects())
    assert "error" not in ok, ok
    failed = write_class_map(str(dataset_root), subjects={})
    assert "error" in failed, failed
    ok2 = write_class_map(str(dataset_root), subjects=_subjects(), allow_removals=True)
    assert "error" not in ok2, ok2

    result = read_audit_log(scope=str(dataset_root), tool="write_class_map", status="ok")
    assert "error" not in result, result
    assert result["count"] == 2
    assert [e["status"] for e in result["entries"]] == ["ok", "ok"]
    # Newest first: the second successful call's own timestamp sorts ahead of the first's.
    assert result["entries"][0]["timestamp"] >= result["entries"][1]["timestamp"]
    assert result["scope_resolved"] == str(dataset_root.resolve())
    assert result["skipped"] == 1  # the one status="error" entry, filtered out


def test_read_audit_log_limit_states_what_it_truncated(
    platform_root: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.annotation_tools import write_class_map

    for _ in range(3):
        res = write_class_map(str(dataset_root), subjects=_subjects(), allow_removals=True)
        assert "error" not in res, res

    result = read_audit_log(scope=str(dataset_root), tool="write_class_map", limit=1)
    assert result["count"] == 1
    assert result["skipped"] == 2


def test_read_audit_log_platform_default_scope_excludes_dataset_scoped_entries(
    platform_root: Path, dataset_root: Path,
) -> None:
    from tcip_mcp.tools.data_tools import scan_dataset
    from tcip_mcp.tools.annotation_tools import write_class_map

    write_class_map(str(dataset_root), subjects=_subjects())
    scan_dataset(str(dataset_root))

    platform_result = read_audit_log(tool="scan_dataset")
    assert platform_result["count"] == 1
    assert platform_result["scope_resolved"] == str(platform_root.resolve())

    dataset_result = read_audit_log(scope=str(dataset_root), tool="scan_dataset")
    assert dataset_result["count"] == 0


def test_read_audit_log_refuses_on_a_page_carrying_an_undecodable_entry(
    platform_root: Path, dataset_root: Path,
) -> None:
    from tcip_store.file_backend import FileBackend
    from tcip_mcp.tools.annotation_tools import write_class_map

    ts.bind(FileBackend())
    try:
        assert "error" not in write_class_map(str(dataset_root), subjects=_subjects())
        key = audit_module.audit_log_key(dataset_root)
        with open(FileBackend().path_for(key), "ab") as handle:
            handle.write(b'{"tool": "write_class_map", bro\n')

        result = read_audit_log(scope=str(dataset_root))
    finally:
        ts.unbind()

    assert "error" in result
    assert "undecodable" in result["error"]
    assert "entries" not in result

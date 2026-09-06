"""Baseline-resident guards for the version-field family's reader branch.

Every import and assertion here names only symbols that predate the family, so
tools/prove_test_fails_before.py can observe each test failing at the pre-family
baseline. The family's own test files import the refusal class the family introduces
and cannot be collected at that baseline; these tests pin the same behaviors through
message text and returned content instead.
"""

from __future__ import annotations

import pytest
from PIL import Image

import tcip_store as ts
from tcip_mcp.audit import audit_log_key
from tcip_mcp.dataset_layout import region_completeness_key
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tcip_store.adoption import adopt_root
from tcip_store.file_backend import FileBackend
from tcip_store.layout_claims import ROOT


def test_a_record_above_its_stores_ceiling_refuses_at_the_seam(tmp_path):
    key = region_completeness_key(tmp_path)
    with pytest.raises(Exception, match="above the 1 this reader knows"):
        ts.replace(key, {"schema_version": 99}, expect=ts.Version.ABSENT)
        ts.read(key)


def test_a_version_that_is_not_a_plain_integer_refuses_at_the_seam(tmp_path):
    key = region_completeness_key(tmp_path)
    with pytest.raises(Exception, match="not a version number"):
        ts.replace(key, {"schema_version": "high"}, expect=ts.Version.ABSENT)
        ts.read(key)


def test_a_log_line_above_the_ceiling_is_never_served_as_content(tmp_path):
    key = audit_log_key(str(tmp_path))
    ts.bind(FileBackend())
    try:
        ts.append(key, {"event": "before"})
        poisoned = ts.get_descriptor(key.store).codec.encode(
            {"event": "guard_probe", "schema_version": 99})
        with open(FileBackend().path_for(key), "ab") as handle:
            handle.write(poisoned + b"\n")
        ts.append(key, {"event": "after"})
        page = ts.read_log(key)
    finally:
        ts.unbind()
    assert all(entry.get("schema_version") != 99 for entry in page.records)
    assert [entry["event"] for entry in page.records] == ["before", "after"]


def test_the_dataset_fingerprint_carries_its_formula_version(tmp_path):
    images = tmp_path / "images" / "2024-01-01"
    labels = tmp_path / "annotations" / "2024-01-01"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    (labels / "a.json").write_text(
        '{"image": "a", "width": 10, "height": 10, "annotations": []}', encoding="utf-8"
    )
    fingerprint = dataset_fingerprint(tmp_path)
    assert fingerprint is not None
    assert fingerprint.startswith("v1:")


def test_adoptions_preflight_refuses_a_document_above_the_ceiling_naming_the_version(tmp_path):
    # Planted directly on disk: the seam's own writer now refuses this same document.
    key = region_completeness_key(tmp_path)
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.get_descriptor(key.store).codec.encode({"schema_version": 99}))

    with pytest.raises(Exception, match="schema_version 99"):
        adopt_root(str(tmp_path), ROOT, report=lambda line: None)

"""The schema-version check at the blob parse sites the storage seam does not reach on its own:
each decodes its own JSON shape outside file_backend._decode, so each wires the check itself and
raises in its own vocabulary. No writer stamps a version yet (the field is lazy), so a refusal
case here writes the version-bearing document directly through the store rather than through a
writer that does not exist.
"""

from __future__ import annotations

import json

import pytest

import tcip_store as ts
from tcip_mcp import class_registry
from tcip_mcp.class_registry import Attribute, ClassRegistry, RegistryError, Subject
from tcip_mcp.dataset_layout import (
    class_registry_key,
    dataset_identity_key,
    require_dataset_identity,
)
from tcip_mcp.pipelines.data.band_groups import (
    band_group_manifest_key,
    read_band_group_manifest,
    write_band_group_manifest,
)
from tcip_mcp.tools.project_tools import register_dataset
from tcip_annotation.json_io import (
    UnreadableLabelDocument,
    annotation_record_key,
    annotations_from_bytes,
    read_annotations,
    write_annotations,
)
from tcip_annotation.state import Annotation


# ── the label parse ──────────────────────────────────────────────────────────

def test_a_version_one_label_document_reads_back_through_the_platforms_own_writer(tmp_path):
    labels_dir = tmp_path / "annotations" / "2024-01-01"
    labels_dir.mkdir(parents=True)
    write_annotations(labels_dir / "a.json", [Annotation(subject="leaf")], 10, 10)
    assert [a.subject for a in read_annotations(labels_dir / "a.json")] == ["leaf"]


def test_a_label_document_above_the_ceiling_refuses_as_unreadable(tmp_path):
    key = annotation_record_key(tmp_path, "a")
    payload = {"image": "a", "width": 10, "height": 10, "annotations": [], "schema_version": 2}
    ts.put_blob(key, json.dumps(payload).encode("utf-8"))

    with pytest.raises(UnreadableLabelDocument):
        annotations_from_bytes(ts.read_blob_versioned(key).value, source="a")


# ── the class registry ───────────────────────────────────────────────────────

def test_a_version_one_registry_reads_back_through_the_platforms_own_writer(tmp_path):
    registry = ClassRegistry(subjects=(
        Subject(name="bur", description="one chestnut bur", defined_by="user:breeder"),
        Subject(name="leaf", attributes=(
            Attribute(name="condition", type="categorical", values=("healthy", "diseased")),
        )),
    ))
    path = tmp_path / "classes.json"
    class_registry.write_registry(path, registry)
    assert class_registry.read_registry(path) == registry


def test_a_registry_above_the_ceiling_refuses_as_a_registry_error(tmp_path):
    key = class_registry_key(tmp_path)
    document = {"bur": {"description": "", "defined_by": "", "defined_at": ""},
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(RegistryError):
        class_registry.read_registry(tmp_path / "classes.json")


# ── the band-group manifest ──────────────────────────────────────────────────

def test_a_version_one_manifest_reads_back_through_the_platforms_own_writer(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a_red.tif").write_bytes(b"x")
    (images_dir / "a_green.tif").write_bytes(b"y")
    write_band_group_manifest(
        images_dir, "a", {"red": images_dir / "a_red.tif", "green": images_dir / "a_green.tif"},
    )
    ref = read_band_group_manifest(images_dir / "a.bandgroup")
    assert set(ref.bands) == {"red", "green"}


def test_a_manifest_above_the_ceiling_refuses_as_a_value_error(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a_red.tif").write_bytes(b"x")
    key = band_group_manifest_key(images_dir, "a")
    document = {"bands": {"red": "a_red.tif"}, "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(ValueError):
        read_band_group_manifest(images_dir / "a.bandgroup")


# ── dataset identity ──────────────────────────────────────────────────────────

def test_a_version_one_identity_reads_back_through_register_dataset(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    result = register_dataset(str(tmp_path), "chestnut", str(tmp_path))
    assert "error" not in result, result
    identity = require_dataset_identity(tmp_path)
    assert identity["crop"] == "chestnut"


def test_an_identity_above_the_ceiling_refuses_as_a_value_error(tmp_path):
    key = dataset_identity_key(tmp_path)
    document = {"crop": "chestnut", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(ValueError):
        require_dataset_identity(tmp_path)

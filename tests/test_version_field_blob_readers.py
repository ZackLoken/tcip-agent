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
from tcip_store import SchemaVersionRefused
from tcip_mcp import class_registry
from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject
from tcip_mcp.dataset_layout import (
    class_registry_key,
    dataset_identity_key,
    require_dataset_identity,
)
from tcip_mcp.pipelines.data.band_groups import (
    band_group_manifest_key,
    detect_and_write_band_groups,
    read_band_group_manifest,
    write_band_group_manifest,
)
from tcip_mcp.pipelines.image_utils import list_logical_images
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tcip_mcp.tools.project_tools import register_dataset
from tcip_annotation.format_io import _parse_coco_json, detect_format
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


def test_a_coco_documents_own_schema_version_is_never_checked_against_annotation_records(tmp_path):
    """COCO is interop (frozen=False by its own row): a legitimate external COCO document
    naming a schema_version this platform's own annotation_records store does not know must
    read, never refuse against a store it never claimed to be."""
    coco_path = tmp_path / "dataset.json"
    coco_path.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 0, "name": "leaf"}],
        "schema_version": 999,
    }), encoding="utf-8")

    assert detect_format(str(coco_path)) == "coco"
    coco = _parse_coco_json(str(coco_path))
    assert coco["schema_version"] == 999


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


def test_a_registry_above_the_ceiling_refuses_as_a_schema_version_refusal(tmp_path):
    """Every other top-level key is a well-formed subject, so ``registry_from_dict`` would parse
    it fine on its own: the refusal below must come from the version check, not from
    ``schema_version: 2`` incidentally failing to parse as a subject body (the wrong reason this
    test used to pass for, when ``2`` is not a dict and trips the shape parser before the version
    check ever runs)."""
    key = class_registry_key(tmp_path)
    document = {"bur": {"description": "", "defined_by": "", "defined_at": ""},
                "leaf": {"description": "", "defined_by": "", "defined_at": ""},
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(SchemaVersionRefused, match="schema_version"):
        class_registry.read_registry(tmp_path / "classes.json")


def test_a_registry_above_the_ceiling_refuses_replace_registry_rather_than_repairing_it(tmp_path):
    """replace_registry's allow_removals repair path is for a registry that will not decode, never
    for one that decodes fine but names a newer schema_version: that document must not be treated
    as a broken registry a repair write is entitled to overwrite."""
    key = class_registry_key(tmp_path)
    document = {"bur": {"description": "", "defined_by": "", "defined_at": ""},
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    incoming = ClassRegistry(subjects=(Subject(name="leaf"),))
    with pytest.raises(SchemaVersionRefused):
        class_registry.replace_registry(tmp_path / "classes.json", incoming, expect=None,
                                        allow_removals=True)
    # Nothing was overwritten: the newer document is still exactly what was stored.
    assert ts.RECORD_JSON.decode(ts.read_blob_versioned(key).value) == document


def test_a_registry_above_the_ceiling_refuses_the_dataset_fingerprint(tmp_path):
    """dataset_fingerprint._registry_term folds a genuinely absent/undecodable registry into "", the
    fingerprint's own no-registry answer, but a version refusal is a wrong content identity a
    delivered number could rest on, so it must refuse the whole fingerprint computation instead."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a.png").write_bytes(b"fake-image-bytes")
    labels_dir = tmp_path / "annotations" / "2024-01-01"
    labels_dir.mkdir(parents=True)
    write_annotations(labels_dir / "a.json", [Annotation(subject="leaf")], 10, 10)

    key = class_registry_key(tmp_path)
    document = {"bur": {"description": "", "defined_by": "", "defined_at": ""},
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(SchemaVersionRefused):
        dataset_fingerprint(tmp_path)


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


def test_a_manifest_above_the_ceiling_refuses_as_a_schema_version_refusal(tmp_path):
    """Deliberately not a ValueError: list_logical_images and detect_and_write_band_groups both
    soften (OSError, ValueError) for a corrupt manifest, and must not absorb this fact too."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a_red.tif").write_bytes(b"x")
    key = band_group_manifest_key(images_dir, "a")
    document = {"bands": {"red": "a_red.tif"}, "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(SchemaVersionRefused):
        read_band_group_manifest(images_dir / "a.bandgroup")


def test_a_version_one_manifest_still_groups_through_list_logical_images(tmp_path):
    """The admitting direction: a manifest write_band_group_manifest actually produced still
    enumerates as one grouped logical image, not dissolved into its individual band files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a_red.tif").write_bytes(b"x")
    (images_dir / "a_green.tif").write_bytes(b"y")
    write_band_group_manifest(
        images_dir, "a", {"red": images_dir / "a_red.tif", "green": images_dir / "a_green.tif"},
    )

    result = list_logical_images(images_dir)
    assert set(result) == {"a"}
    assert set(result["a"].bands) == {"red", "green"}


def test_a_v99_manifest_refuses_the_enumeration_rather_than_dissolving_the_group(tmp_path):
    """A version-refused manifest must not silently fold into 'no manifest here', which would
    change the image list from one grouped capture to its individual band files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "a_red.tif").write_bytes(b"x")
    (images_dir / "a_green.tif").write_bytes(b"y")
    key = band_group_manifest_key(images_dir, "a")
    document = {"bands": {"red": "a_red.tif", "green": "a_green.tif"}, "schema_version": 99}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(SchemaVersionRefused):
        list_logical_images(images_dir)
    with pytest.raises(SchemaVersionRefused):
        detect_and_write_band_groups(images_dir)


# ── dataset identity ──────────────────────────────────────────────────────────

def test_a_version_one_identity_reads_back_through_register_dataset(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    result = register_dataset(str(tmp_path), "chestnut", str(tmp_path))
    assert "error" not in result, result
    identity = require_dataset_identity(tmp_path)
    assert identity["crop"] == "chestnut"


def test_an_identity_above_the_ceiling_refuses_as_a_schema_version_refusal(tmp_path):
    """Deliberately not a plain ValueError: a caller that tolerates the plain ValueError
    require_dataset_identity raises for genuine absence (register_dataset's re-register read,
    training_tools' lineage identity read) must not absorb this fact as though nothing were
    registered yet."""
    key = dataset_identity_key(tmp_path)
    document = {"crop": "chestnut", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    with pytest.raises(SchemaVersionRefused):
        require_dataset_identity(tmp_path)


def test_a_version_refused_identity_refuses_the_re_register_rather_than_overwriting_it(tmp_path):
    """register_dataset must route its re-register read through the same checked reader
    require_dataset_identity uses, so a newer writer's identity document refuses the call
    instead of being silently overwritten with a v1-shaped one."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    key = dataset_identity_key(tmp_path)
    document = {"crop": "chestnut", "id": "abc123", "fingerprint": "v1:deadbeef",
                "schema_version": 2}
    ts.put_blob(key, ts.RECORD_JSON.encode(document))

    result = register_dataset(str(tmp_path), "chestnut", str(tmp_path))
    assert "error" in result

    # Nothing was overwritten: the newer document is still exactly what was stored.
    assert ts.RECORD_JSON.decode(ts.read_blob_versioned(key).value) == document

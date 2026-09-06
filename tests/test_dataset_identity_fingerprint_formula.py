"""A dataset identity carrying a bare, pre-prefix fingerprint (from before ``dataset_fingerprint``
started stamping a ``v<n>:`` formula-version prefix) is refused rather than admitted as the
dataset's current identity, both by the identity decoder and by the project registry reader:
re-registering through ``register_dataset`` is the remedy either names. A fixture registers a
dataset for real and hand-edits its identity to a bare value afterwards, standing in for a
dataset registered before this family.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.dataset_layout import (
    dataset_identity_key,
    decode_dataset_identity,
    require_dataset_identity,
)
from tcip_mcp.tools.project_tools import read_datasets, register_dataset, upsert_dataset


def _real_dataset(root: Path) -> None:
    from PIL import Image

    images = root / "images" / "2024-01-01"
    labels = root / "annotations" / "2024-01-01"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(images / "a.png")
    (labels / "a.json").write_text(
        '{"image": "a", "width": 10, "height": 10, "annotations": []}', encoding="utf-8")


def _bare_the_identity(root: Path) -> str:
    """Overwrite a real registration's prefixed fingerprint with its bare hex."""
    identity = require_dataset_identity(root)
    prefixed = identity["fingerprint"]
    bare = prefixed.split(":", 1)[1]
    document = {**identity, "fingerprint": bare}
    ts.put_blob(dataset_identity_key(root), ts.RECORD_JSON.encode(document))
    return bare


def test_decode_dataset_identity_refuses_a_bare_fingerprint_naming_register_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    _bare_the_identity(root)

    with pytest.raises(ValueError, match="register_dataset"):
        require_dataset_identity(root)


def test_decode_dataset_identity_admits_a_prefixed_fingerprint(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    identity = require_dataset_identity(root)
    assert identity["fingerprint"] == result["fingerprint"]
    assert identity["fingerprint"].startswith("v1:")


def test_decode_dataset_identity_admits_a_null_fingerprint(tmp_path: Path):
    """A dataset with no images or labels registers with a null fingerprint; that is not the
    bare-legacy-value case and must not refuse."""
    root = tmp_path / "dataset"
    root.mkdir()
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    assert result["fingerprint"] is None

    identity = require_dataset_identity(root)
    assert identity["fingerprint"] is None


def test_decode_dataset_identity_direct_call_refuses_and_admits(tmp_path: Path):
    """decode_dataset_identity itself, not only through require_dataset_identity."""
    bare_bytes = ts.RECORD_JSON.encode({"crop": "chestnut", "id": "abc", "fingerprint": "f" * 16})
    with pytest.raises(ValueError, match="register_dataset"):
        decode_dataset_identity(bare_bytes, dataset_root=tmp_path)

    prefixed_bytes = ts.RECORD_JSON.encode(
        {"crop": "chestnut", "id": "abc", "fingerprint": "v1:" + "f" * 16})
    decoded = decode_dataset_identity(prefixed_bytes, dataset_root=tmp_path)
    assert decoded["fingerprint"] == "v1:" + "f" * 16


def test_read_datasets_refuses_an_entry_carrying_a_bare_fingerprint_naming_register_dataset(
    tmp_path: Path,
):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result
    upsert_dataset(root, {"id": result["id"], "path": ".", "crop": "chestnut",
                          "fingerprint": "f" * 16})

    with pytest.raises(ValueError, match="register_dataset"):
        read_datasets(root)


def test_read_datasets_admits_prefixed_and_null_fingerprint_entries(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    _real_dataset(root)
    result = register_dataset(str(root), "chestnut", str(root))
    assert "error" not in result, result

    entries = read_datasets(root)
    assert entries[0]["id"] == result["id"]
    assert entries[0]["fingerprint"] == result["fingerprint"]

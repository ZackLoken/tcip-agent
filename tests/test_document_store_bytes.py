"""The exact bytes of the documents the platform lands as files.

A file's bytes are those its producer writes, and a record's those the store dump writes for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tcip_store as ts
from tcip_mcp.cli.dump_store import dump_store

if TYPE_CHECKING:
    import pytest

SUBJECT = "subject_under_test"
REGISTRY_VALUE = {
    SUBJECT: {
        "description": "a description with an ümlaut",
        "attributes": {
            "state": {"type": "categorical", "values": ["closed", "open"]}
        },
    }
}
REGISTRY_BYTES = (
    "{\n"
    '  "subject_under_test": {\n'
    '    "description": "a description with an ümlaut",\n'
    '    "attributes": {\n'
    '      "state": {\n'
    '        "type": "categorical",\n'
    '        "values": [\n'
    '          "closed",\n'
    '          "open"\n'
    "        ]\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
).encode()

IDENTITY_BYTES = (
    b"{\n"
    b'  "crop": "crop_under_test",\n'
    b'  "id": "a1b2c3d4e5f6",\n'
    b'  "fingerprint": "9f2c1b0a4d6e8f31"\n'
    b"}\n"
)

BAND_FILENAMES = {"Grün": "cap_G.tif", "Red": "cap_R.tif"}
BAND_WAVELENGTHS = {"Grün": 560.0, "Red": 650.0}
BAND_GROUP_BYTES = (
    "{\n"
    '  "bands": {\n'
    '    "Grün": "cap_G.tif",\n'
    '    "Red": "cap_R.tif"\n'
    "  },\n"
    '  "source": "embedded-metadata",\n'
    '  "identity_tolerance": {},\n'
    '  "central_wavelength_nm": {\n'
    '    "Grün": 560.0,\n'
    '    "Red": 650.0\n'
    "  }\n"
    "}\n"
).encode()

NOW = "2026-03-04T12:00:00+00:00"
REPORT_BYTES = (
    "{\n"
    '  "timestamp": "2026-03-04T12:00:00+00:00",\n'
    '  "category": "missing_tool",\n'
    '  "detail": "ein Werkzeug für ü",\n'
    '  "context": {\n'
    '    "trait": "trait_under_test"\n'
    "  },\n"
    '  "user_disagreement": false\n'
    "}\n"
).encode()

RETROSPECTIVE_BYTES = (
    "{\n"
    '  "timestamp": "2026-03-04T12:00:00+00:00",\n'
    '  "project_id": "project_under_test",\n'
    '  "task": "t",\n'
    '  "worked": "was gut lief für ü",\n'
    '  "did_not_work": "d",\n'
    '  "assumptions_wrong": "",\n'
    '  "knowledge_for_future": "",\n'
    '  "missing_or_hard_tools": "",\n'
    '  "would_do_differently": ""\n'
    "}\n"
).encode()


def _dumped(project: Path, store: str) -> bytes:
    """Return the bytes of the file the dump writes for ``project``'s one record in ``store``."""
    (written,) = (
        path
        for path in dump_store(project, project.parent / "dump")
        if path.parent.name == store
    )
    return written.read_bytes()


def test_the_subject_registry_lands_as_the_ordered_json_document_labels_are_decoded_by(
    tmp_path: Path,
) -> None:
    """The subject registry lands as its ordered JSON document, byte for byte.

    Written through ``replace_registry``, the subject and attribute sequences keep their
    declared order.
    """
    from tcip_mcp import subject_registry
    from tcip_mcp.dataset_layout import subjects_path

    path = subjects_path(tmp_path)
    subject_registry.replace_registry(
        tmp_path,
        subject_registry.registry_from_dict(REGISTRY_VALUE),
        expect=None,
        actor=None,
    )

    assert path.read_bytes() == REGISTRY_BYTES


def test_the_dataset_identity_document_lands_as_the_json_every_citing_record_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dataset identity document lands as the exact JSON bytes the fixture pins.

    It is written through ``register_dataset``, its minted id and computed fingerprint held
    still.
    """
    from tcip_mcp import project_record
    from tcip_mcp.dataset_layout import dataset_identity_path
    from tcip_mcp.pipelines.data import dataset_fingerprint
    from tcip_mcp.tools.project_tools import register_dataset

    monkeypatch.setattr(project_record, "mint_id", lambda: "a1b2c3d4e5f6")
    monkeypatch.setattr(
        dataset_fingerprint,
        "dataset_fingerprint",
        lambda root: "9f2c1b0a4d6e8f31",
    )
    dataset = tmp_path / "ds"
    dataset.mkdir()

    assert "error" not in register_dataset(
        tmp_path, str(dataset), crop="crop_under_test", actor=None
    )

    assert dataset_identity_path(dataset).read_bytes() == IDENTITY_BYTES


def test_a_band_group_manifest_lands_as_the_json_the_image_enumerators_parse(
    tmp_path: Path,
) -> None:
    """A band-group manifest lands as its expected JSON document, byte for byte.

    The manifest is written through ``write_band_group_manifest``.
    """
    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    images = image_root(tmp_path)
    images.mkdir()
    bands = {
        name: images / filename for name, filename in BAND_FILENAMES.items()
    }

    path = write_band_group_manifest(
        images,
        "cap",
        bands,
        central_wavelength_nm=BAND_WAVELENGTHS,
        source="embedded-metadata",
        identity_tolerance={},
        expect=ts.Version.ABSENT,
    )

    assert path.read_bytes() == BAND_GROUP_BYTES


def test_a_friction_report_dumps_as_the_json_document_every_reader_of_the_corpus_parses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A friction report dumps as the exact JSON bytes the fixture pins.

    It is written through ``report_friction``, its clock held still.
    """
    from tcip_mcp.tools import meta_tools

    monkeypatch.setattr(meta_tools, "now_iso", lambda: NOW)

    meta_tools.report_friction(
        tmp_path,
        "missing_tool",
        "ein Werkzeug für ü",
        context={"trait": "trait_under_test"},
        actor=None,
    )

    assert _dumped(tmp_path, meta_tools.MemoryKind.reports) == REPORT_BYTES


def test_a_retrospective_dumps_as_the_json_document_every_reader_of_the_corpus_parses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retrospective dumps as the exact JSON bytes the fixture pins.

    It is written through ``write_retrospective``, its clock held still.
    """
    from tcip_mcp.tools import meta_tools

    monkeypatch.setattr(meta_tools, "now_iso", lambda: NOW)

    meta_tools.write_retrospective(
        tmp_path,
        "project_under_test",
        task="t",
        worked="was gut lief für ü",
        did_not_work="d",
        actor=None,
    )

    assert (
        _dumped(tmp_path, meta_tools.MemoryKind.retrospectives)
        == RETROSPECTIVE_BYTES
    )

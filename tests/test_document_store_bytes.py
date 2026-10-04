"""The exact bytes of the documents a human or a tool reads as files: a file's on the file its
producer writes, a record's on the file the read-only dump writes for it.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts

from tcip_mcp.cli.dump_store import dump_store

SUBJECT = "subject_under_test"
REGISTRY_VALUE = {
    SUBJECT: {
        "description": "a description with an ümlaut",
        "attributes": {"state": {"type": "categorical", "values": ["closed", "open"]}},
    }
}
REGISTRY_BYTES = (
    '{\n'
    '  "subject_under_test": {\n'
    '    "description": "a description with an ümlaut",\n'
    '    "attributes": {\n'
    '      "state": {\n'
    '        "type": "categorical",\n'
    '        "values": [\n'
    '          "closed",\n'
    '          "open"\n'
    '        ]\n'
    '      }\n'
    '    }\n'
    '  }\n'
    '}\n'
).encode("utf-8")

IDENTITY_BYTES = (
    '{\n'
    '  "crop": "crop_under_test",\n'
    '  "id": "a1b2c3d4e5f6",\n'
    '  "fingerprint": "9f2c1b0a4d6e8f31"\n'
    '}\n'
).encode("utf-8")

BAND_FILENAMES = {"Grün": "cap_G.tif", "Red": "cap_R.tif"}
BAND_WAVELENGTHS = {"Grün": 560.0, "Red": 650.0}
BAND_GROUP_BYTES = (
    '{\n'
    '  "bands": {\n'
    '    "Grün": "cap_G.tif",\n'
    '    "Red": "cap_R.tif"\n'
    '  },\n'
    '  "source": "embedded-metadata",\n'
    '  "central_wavelength_nm": {\n'
    '    "Grün": 560.0,\n'
    '    "Red": 650.0\n'
    '  }\n'
    '}\n'
).encode("utf-8")

NOW = "2026-03-04T12:00:00+00:00"
REPORT_BYTES = (
    '{\n'
    '  "timestamp": "2026-03-04T12:00:00+00:00",\n'
    '  "category": "missing_tool",\n'
    '  "detail": "ein Werkzeug für ü",\n'
    '  "context": {\n'
    '    "trait": "trait_under_test"\n'
    '  },\n'
    '  "user_disagreement": false\n'
    '}\n'
).encode("utf-8")

RETROSPECTIVE_BYTES = (
    '"# project_under_test\\n\\n## Retrospective: 2026-03-04T12:00:00+00:00\\n\\n'
    '### Task\\n\\nt\\n\\n### What worked\\n\\nwas gut lief für ü\\n\\n'
    '### What did not work\\n\\nd\\n\\n'
    '### Assumptions that turned out to be wrong\\n\\n_(none noted)_\\n\\n'
    '### Knowledge for future sessions\\n\\n_(none noted)_\\n\\n'
    '### Missing or hard-to-use tools\\n\\n_(none noted)_\\n\\n'
    '### What I would do differently\\n\\n_(none noted)_\\n\\n---\\n"\n'
).encode("utf-8")


def _dumped(project: Path, store: str) -> bytes:
    """The bytes of the one file the dump writes for ``project``'s one record in ``store``."""
    (written,) = (path for path in dump_store(project, project.parent / "dump")
                  if path.parent.name == store)
    return written.read_bytes()


def test_the_subject_registry_lands_as_the_ordered_json_document_labels_are_decoded_by(tmp_path):
    """Written through ``replace_registry``, which encodes with the canonical record codec: the
    subject and attribute sequences keep their declared order rather than being sorted."""
    from tcip_mcp import subject_registry
    from tcip_mcp.dataset_layout import subjects_path

    path = subjects_path(tmp_path)
    subject_registry.replace_registry(
        tmp_path, subject_registry.registry_from_dict(REGISTRY_VALUE), expect=None, actor=None)

    assert path.read_bytes() == REGISTRY_BYTES


def test_the_dataset_identity_document_lands_as_the_json_every_citing_record_reads(
    tmp_path, monkeypatch,
):
    """Written through ``register_dataset``, its minted id and computed fingerprint held still."""
    from tcip_mcp import project_record
    from tcip_mcp.dataset_layout import dataset_identity_path
    from tcip_mcp.pipelines.data import dataset_fingerprint
    from tcip_mcp.tools.project_tools import register_dataset

    monkeypatch.setattr(project_record, "mint_id", lambda: "a1b2c3d4e5f6")
    monkeypatch.setattr(dataset_fingerprint, "dataset_fingerprint", lambda root: "9f2c1b0a4d6e8f31")
    dataset = tmp_path / "ds"
    dataset.mkdir()

    assert "error" not in register_dataset(tmp_path, str(dataset), crop="crop_under_test")

    assert dataset_identity_path(dataset).read_bytes() == IDENTITY_BYTES


def test_a_band_group_manifest_lands_as_the_json_the_image_enumerators_parse(tmp_path):
    """A ``.bandgroup`` is itself an enumerated logical image, so its bytes are what the
    enumerators read; written through ``write_band_group_manifest``."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    images = tmp_path / "images"
    images.mkdir()
    bands = {name: images / filename for name, filename in BAND_FILENAMES.items()}

    path = write_band_group_manifest(
        images, "cap", bands,
        central_wavelength_nm=BAND_WAVELENGTHS, source="embedded-metadata",
        expect=ts.Version.ABSENT,
    )

    assert path.read_bytes() == BAND_GROUP_BYTES


def test_a_friction_report_dumps_as_the_json_document_every_reader_of_the_corpus_parses(
    tmp_path, monkeypatch,
):
    """Written through ``report_friction``, its clock and id suffix held still."""
    from tcip_mcp.tools import meta_tools

    monkeypatch.setattr(meta_tools, "now_iso", lambda: NOW)
    monkeypatch.setattr(meta_tools.secrets, "token_hex", lambda n: "a1b2")

    assert "error" not in meta_tools.report_friction(
        tmp_path, "missing_tool", "ein Werkzeug für ü", context={"trait": "trait_under_test"})

    assert _dumped(tmp_path, meta_tools.FRICTION_REPORT_STORE) == REPORT_BYTES


def test_a_retrospective_dumps_as_its_markdown_text_spelled_as_one_json_string(
    tmp_path, monkeypatch,
):
    """Written through ``write_retrospective``, its clock held still: the text itself, as one
    JSON string."""
    from tcip_mcp.tools import meta_tools

    monkeypatch.setattr(meta_tools, "now_iso", lambda: NOW)

    assert "error" not in meta_tools.write_retrospective(
        tmp_path, "project_under_test", task="t", worked="was gut lief für ü", did_not_work="d")

    assert _dumped(tmp_path, meta_tools.RETROSPECTIVE_STORE) == RETROSPECTIVE_BYTES

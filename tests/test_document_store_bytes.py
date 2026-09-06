"""The exact bytes of the stores whose entries are documents a human or a tool reads as files.

These stores hold their value as opaque bytes at the seam and encode it in the module that owns
them, so the encoding is no longer something the seam can be asked about: it is something this
suite has to pin. Every case compares the landed file byte for byte against the spelling recorded
here, so a codec swapped for a re-spelled serializer, a dropped trailing newline, or an
``ensure_ascii`` flip shows up as a failing byte comparison rather than as a dataset that reads
differently a season later.

Two cases drive the owning module's own writer end to end: ``write_registry`` and
``write_band_group_manifest``. A third, ``write_trait_spec_fields``, no longer needs one: a trait
spec now writes through the same ``RECORD_JSON`` codec every other record store uses, so its byte
spelling is already the one ``test_the_canonical_record_codec_writes_the_bytes_this_test_spells_out``
pins centrally in ``test_store_contract.py``, and its own placement and codec application are
already covered there by the ``trait_specs`` case of
``test_a_registered_store_lands_where_its_locator_says_with_the_bytes_its_codec_produces``. Pinning
a second literal byte sequence here would only restate that pin under a weaker, hand-picked field
set; ``test_trait_authoring.py`` is where ``write_trait_spec_fields``'s own field-level content is
asserted. The other four (dataset identity, friction report, retrospective, snapshot manifest) pin
the codec and the path only, through the seam expression their writer makes, because those writers
mint an id, stamp a timestamp, draw a random suffix or capture a live environment, none of which a
fixed byte comparison can hold still. That those writers reach the store through this very
expression is covered where each writer's own content is asserted: ``test_project_tools`` for the
identity document, ``test_meta_tools`` for reports and retrospectives, and ``test_bespoke_provenance``
plus ``test_model_build_provenance_and_dims`` for the snapshot manifest.

The bytes are the ones these documents carry on disk today; the placement of each file is pinned
separately, for every registered store at once, by ``test_store_contract``. The cases whose store
is a record bind the file backend, since only there is the file the bytes land in the store's own
answer rather than an export's.
"""

from __future__ import annotations

from pathlib import Path

import tcip_store as ts

SUBJECT = "subject_under_test"
REGISTRY_VALUE = {
    SUBJECT: {
        "description": "a description with an ümlaut",
        "defined_by": "breeder",
        "defined_at": "2026-03-04T12:00:00+00:00",
        "attributes": {"state": {"type": "categorical", "values": ["closed", "open"]}},
    }
}
REGISTRY_BYTES = (
    '{\n'
    '  "subject_under_test": {\n'
    '    "description": "a description with an ümlaut",\n'
    '    "defined_by": "breeder",\n'
    '    "defined_at": "2026-03-04T12:00:00+00:00",\n'
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

IDENTITY_VALUE = {"crop": "crop_under_test", "id": "a1b2c3d4e5f6",
                  "fingerprint": "9f2c1b0a4d6e8f31"}
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

REPORT_ID = "20260304T120000Z_missing_tool_a1b2"
REPORT_VALUE = {
    "timestamp": "2026-03-04T12:00:00+00:00",
    "category": "missing_tool",
    "detail": "ein Werkzeug für ü",
    "context": {"trait": "trait_under_test"},
    "user_disagreement": False,
}
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

RETROSPECTIVE_ID = "project_under_test"
RETROSPECTIVE_BODY = "## Retrospective: 2026-03-04T12:00:00+00:00\n\nwas gut lief für ü\n\n---\n"
RETROSPECTIVE_BYTES = (
    "# project_under_test\n"
    "\n"
    "## Retrospective: 2026-03-04T12:00:00+00:00\n"
    "\n"
    "was gut lief für ü\n"
    "\n"
    "---\n"
).encode("utf-8")

EXPERIMENT = "exp_042"
SNAPSHOT_MANIFEST_VALUE = {
    "builder": "my_module:build",
    "training_source": None,
    "dataset_builder": None,
    "declared_files": ["my_model_ü.py"],
    "files": [{"file": "ab12cd34/my_model_ü.py", "sha256": "0" * 64, "bytes": 27}],
    "missing": [],
    "snapshot_errors": [],
    "env": {"python": "3.12.13"},
    "seed": 7,
}
SNAPSHOT_MANIFEST_BYTES = (
    '{\n'
    '  "builder": "my_module:build",\n'
    '  "training_source": null,\n'
    '  "dataset_builder": null,\n'
    '  "declared_files": [\n'
    '    "my_model_ü.py"\n'
    '  ],\n'
    '  "files": [\n'
    '    {\n'
    '      "file": "ab12cd34/my_model_ü.py",\n'
    '      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",\n'
    '      "bytes": 27\n'
    '    }\n'
    '  ],\n'
    '  "missing": [],\n'
    '  "snapshot_errors": [],\n'
    '  "env": {\n'
    '    "python": "3.12.13"\n'
    '  },\n'
    '  "seed": 7\n'
    '}\n'
).encode("utf-8")


def test_the_class_registry_lands_as_the_ordered_json_document_labels_are_decoded_by(tmp_path):
    """Written through ``write_registry``, which encodes with the canonical record codec: the
    subject and attribute sequences keep their declared order rather than being sorted."""
    from tcip_mcp import class_registry
    from tcip_mcp.dataset_layout import classes_path

    path = classes_path(tmp_path)
    class_registry.write_registry(path, class_registry.registry_from_dict(REGISTRY_VALUE))

    assert path.read_bytes() == REGISTRY_BYTES


def test_the_dataset_identity_document_lands_as_the_json_every_citing_record_reads(tmp_path):
    """The identity write ``register_dataset`` makes: the canonical record codec's bytes, put
    under the version the caller read."""
    from tcip_mcp.dataset_layout import dataset_identity_key, dataset_identity_path

    ts.put_blob(
        dataset_identity_key(tmp_path),
        ts.RECORD_JSON.encode(IDENTITY_VALUE),
        expect=ts.Version.ABSENT,
    )

    assert dataset_identity_path(tmp_path).read_bytes() == IDENTITY_BYTES


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


def test_a_friction_report_lands_as_the_json_document_every_reader_of_the_corpus_parses(tmp_path):
    """The write ``report_friction`` makes: the canonical record codec's bytes, create-only."""
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.tools import meta_tools

    ts.bind(FileBackend())
    ts.replace(
        meta_tools.friction_report_key(str(tmp_path), REPORT_ID),
        REPORT_VALUE,
        expect=ts.Version.ABSENT,
    )

    assert meta_tools._report_path(str(tmp_path), REPORT_ID).read_bytes() == REPORT_BYTES


def test_a_retrospective_lands_as_the_markdown_text_and_nothing_around_it(tmp_path):
    """The first section ``write_retrospective`` writes: the text itself, no envelope."""
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.tools import meta_tools

    ts.bind(FileBackend())
    key = meta_tools.retrospective_key(str(tmp_path), RETROSPECTIVE_ID)
    stored = ts.read_versioned(key, default=None)
    ts.replace(key, f"# {RETROSPECTIVE_ID}\n\n{RETROSPECTIVE_BODY}", expect=stored.version)

    path = meta_tools._retrospective_path(str(tmp_path), RETROSPECTIVE_ID)
    assert path.read_bytes() == RETROSPECTIVE_BYTES


def test_a_snapshot_manifest_lands_in_its_experiment_directory_under_the_experiments_scope(tmp_path):
    """The manifest hangs off the experiments root, the scope its experiment's other members
    already use, and still lands at ``<experiment_id>/model_src/manifest.json`` for the raw
    reader that checks a bespoke run's provenance.

    Bound to the file backend on purpose: unlike the documents around it this manifest is a
    record, so the path its bespoke locator preserves is a fact about the file layout, and a
    database backend keeps it in the database instead.
    """
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.pipelines.model_build import snapshot_manifest_key

    ts.bind(FileBackend())
    exp_dir = tmp_path / EXPERIMENT
    exp_dir.mkdir()
    key = snapshot_manifest_key(exp_dir)
    assert key.root == str(Path(tmp_path).resolve())
    assert key.parts == (EXPERIMENT, "manifest")

    ts.replace(key, SNAPSHOT_MANIFEST_VALUE, expect=ts.Version.ABSENT)

    landed = exp_dir / "model_src" / "manifest.json"
    assert landed.read_bytes() == SNAPSHOT_MANIFEST_BYTES

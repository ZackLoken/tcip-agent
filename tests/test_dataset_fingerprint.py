"""The whole-dataset content fingerprint (dataset identity).

Pins: the fingerprint reuses ``ground_truth_digest`` for its label term; it is pixel-aware (a
re-encode under an unchanged filename changes it); registry order matters but whitespace
doesn't; a moved dataset keeps its fingerprint; and it is ``None`` for a bespoke dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

import tcip_store as ts
from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key
from tcip_mcp.pipelines.data import (
    dataset_fingerprint as fingerprint_mod,
    selection,
)
from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint
from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry
from tests._producer_fixtures import label_image, registry_over


def _make_dataset(
    root: Path, *, pixel=(120, 120, 120), bud_box=(10, 10, 40, 40), ext="jpg"
) -> None:
    """Write a minimal nested-schema dataset: one dated image + its bud label + a registry."""
    date = "2026-02-11"
    (root / "images" / date).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=pixel).save(
        root / "images" / date / f"IMG_1.{ext}"
    )
    label_image(
        root / "images" / date / f"IMG_1.{ext}",
        [Annotation(subject="bud", geometry=BBox(*bud_box))],
        64,
        64,
    )
    registry_over(
        root,
        SubjectRegistry(subjects=(Subject(name="bud", description="a bud"),)),
    )


def test_fingerprint_reuses_ground_truth_digest_for_labels(
    tmp_path, monkeypatch
):
    _make_dataset(tmp_path)
    calls = []
    real = selection.ground_truth_digest

    def recording_digest(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(selection, "ground_truth_digest", recording_digest)
    fp = dataset_fingerprint(tmp_path)
    assert fp is not None
    assert calls, (
        "dataset_fingerprint must digest each label document through ground_truth_digest"
    )


def test_a_label_edit_changes_the_fingerprint(tmp_path):
    _make_dataset(tmp_path)
    before = dataset_fingerprint(tmp_path)
    # move the box -> label bytes change
    json_io.write_label_document(
        label_key(tmp_path, "2026-02-11", "IMG_1"),
        [Annotation(subject="bud", geometry=BBox(11, 11, 41, 41))],
        64,
        64,
    )
    assert dataset_fingerprint(tmp_path) != before


def test_pixel_reencode_under_same_filename_changes_the_fingerprint(tmp_path):
    """A pixel re-encode under the same file name changes the fingerprint.

    A BMP re-encoded with other pixels at the same file name and byte size, its labels
    untouched, changes the fingerprint and not the labels term.
    """
    _make_dataset(tmp_path, pixel=(120, 120, 120), ext="bmp")
    before_fp = dataset_fingerprint(tmp_path)
    before_labels = fingerprint_mod._labels_term(tmp_path, {})
    img_path = tmp_path / "images" / "2026-02-11" / "IMG_1.bmp"
    size_before = img_path.stat().st_size
    # re-encode the image with different pixels, same filename, labels untouched
    Image.new("RGB", (64, 64), color=(0, 200, 0)).save(img_path)
    # confirms the size channel is closed, not just JPEG luck
    assert img_path.stat().st_size == size_before
    assert (
        fingerprint_mod._labels_term(tmp_path, {}) == before_labels
    )  # labels-only: blind
    # fingerprint: pixel-aware, catches it even though size didn't
    assert dataset_fingerprint(tmp_path) != before_fp


def test_an_image_rewritten_at_its_size_and_mtime_changes_the_fingerprint(
    tmp_path,
):
    """An image rewritten at its size and modification time changes the fingerprint.

    A rewrite with other pixels that keeps the file's size and modification time moves the
    fingerprint.
    """
    import os

    _make_dataset(tmp_path, pixel=(120, 120, 120), ext="bmp")
    img_path = tmp_path / "images" / "2026-02-11" / "IMG_1.bmp"
    before = dataset_fingerprint(tmp_path)
    stat = img_path.stat()

    Image.new("RGB", (64, 64), color=(0, 200, 0)).save(img_path)
    os.utime(img_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert (img_path.stat().st_size, img_path.stat().st_mtime_ns) == (
        stat.st_size,
        stat.st_mtime_ns,
    )
    assert dataset_fingerprint(tmp_path) != before


def test_registry_value_order_matters_but_whitespace_does_not(tmp_path):
    _make_dataset(tmp_path)
    # add an ordered attribute -> registry (and thus fingerprint) changes
    reg2 = SubjectRegistry(
        subjects=(
            Subject(
                name="bud",
                description="a currant bud",
                attributes=(
                    Attribute(
                        name="opening",
                        type="categorical",
                        values=("closed", "open"),
                    ),
                ),
            ),
        )
    )
    registry_over(tmp_path, reg2)
    with_attr = dataset_fingerprint(tmp_path)
    _make_dataset(tmp_path)  # reset registry to no-attr
    assert dataset_fingerprint(tmp_path) != with_attr

    # a whitespace-only reformat of subjects.json must not change identity (canonical
    # re-serialization)
    reg2_again = SubjectRegistry(
        subjects=(
            Subject(
                name="bud",
                description="a currant bud",
                attributes=(
                    Attribute(
                        name="opening",
                        type="categorical",
                        values=("closed", "open"),
                    ),
                ),
            ),
        )
    )
    registry_over(tmp_path, reg2_again)
    fp_a = dataset_fingerprint(tmp_path)
    cp = tmp_path / "subjects.json"
    # reformat
    cp.write_text(
        json.dumps(json.loads(cp.read_text()), indent=4) + "\n\n",
        encoding="utf-8",
    )
    assert dataset_fingerprint(tmp_path) == fp_a


def test_an_admitted_sample_enters_at_its_admitted_version_under_the_root_as_spelled(
    tmp_path, monkeypatch
):
    """An admitted sample enters the labels term at its admitted version, under the root as spelled.

    With a dataset root spelled through a ``..`` segment, the fingerprint over the admitted
    samples is unchanged when ``ground_truth_digest`` refuses every call, so no label document
    is digested again.
    """
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.data.split_construction import admitted_membership

    (tmp_path / "elsewhere").mkdir()
    root = tmp_path / "elsewhere" / ".." / "ds"
    _make_dataset(root)
    samples = admitted_membership(
        [("2026-02-11", root / "images" / "2026-02-11", None)],
        scope=ClassScope(subject="bud"),
        grouping="tile_prefix",
    ).samples.values()
    admitted = dataset_fingerprint(root, samples)
    assert admitted is not None

    def digested_again(key):
        raise AssertionError(f"{key} was digested again")

    monkeypatch.setattr(selection, "ground_truth_digest", digested_again)
    assert dataset_fingerprint(root, samples) == admitted


def test_fingerprint_is_content_addressed_move_preserves_it(tmp_path):
    src = tmp_path / "a"
    _make_dataset(src)
    dst = tmp_path / "b"
    ts.release_root(src)
    shutil.copytree(src, dst)  # same content, different path
    assert dataset_fingerprint(dst) == dataset_fingerprint(src)


def test_the_labels_term_keys_each_document_by_its_capture(tmp_path):
    """The labels term keys each document by its capture beside its stem.

    One stem's document under two different captures is two different label sets.
    """
    undated, dated = tmp_path / "undated", tmp_path / "dated"
    box = [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))]
    json_io.write_label_document(
        label_key(undated, UNDATED_BUCKET, "A"), box, 32, 32
    )
    json_io.write_label_document(
        label_key(dated, "2026-02-11", "A"), box, 32, 32
    )

    assert fingerprint_mod._labels_term(
        undated, {}
    ) != fingerprint_mod._labels_term(dated, {})
    assert fingerprint_mod._labels_term(tmp_path / "unlabeled", {}) is None


def test_rgb_nested_dataset_fingerprint_golden_pins_the_current_implementations_own_determinism(
    tmp_path,
):
    """An RGB nested dataset fingerprints to the value one run of ``dataset_fingerprint`` recorded.

    The dataset holds fixed image bytes, one label document and a registry.
    """
    date = "2026-02-11"
    (tmp_path / "images" / date).mkdir(parents=True)
    (tmp_path / "images" / date / "IMG_1.jpg").write_bytes(
        b"fixed-jpeg-bytes-for-fingerprint-test"
    )
    label_image(
        tmp_path / "images" / date / "IMG_1.jpg",
        [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))],
        64,
        64,
    )
    registry_over(
        tmp_path,
        SubjectRegistry(subjects=(Subject(name="bud", description="a bud"),)),
    )

    assert dataset_fingerprint(tmp_path) == "107a8fbb2dddc690"


def test_bandgroup_manifest_file_itself_is_hashed_not_only_its_member_bands(
    tmp_path,
):
    """A band group's manifest file is hashed itself, not only its member bands.

    .bandgroup is hashed as its own bytes: changing the manifest alone, with its named band
    files held byte-for-byte fixed, changes the fingerprint; it fingerprints identically across
    two calls.
    """
    date = "2026-02-11"
    images = tmp_path / "images" / date
    images.mkdir(parents=True)
    Image.new("L", (16, 16), color=10).save(images / "band_r.png")
    Image.new("L", (16, 16), color=20).save(images / "band_nir.png")
    bands = {"r": images / "band_r.png", "nir": images / "band_nir.png"}
    manifest_path = write_band_group_manifest(
        images,
        "capture_1",
        bands,
        central_wavelength_nm=None,
        identity_tolerance={},
    )
    label_image(
        manifest_path,
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))],
        16,
        16,
    )
    registry_over(tmp_path, SubjectRegistry(subjects=(Subject(name="bud"),)))

    fp1 = dataset_fingerprint(tmp_path)
    assert fp1 is not None
    assert (
        dataset_fingerprint(tmp_path) == fp1
    )  # deterministic across two calls

    # Rewrite the manifest's own bytes; the band files it names are untouched.
    write_band_group_manifest(
        images,
        "capture_1",
        bands,
        central_wavelength_nm={"r": 660.0, "nir": 850.0},
        identity_tolerance={},
    )
    assert dataset_fingerprint(tmp_path) != fp1


def test_a_malformed_registry_refuses_the_fingerprint_and_an_absent_one_admits_it(
    tmp_path,
):
    """A malformed registry refuses the fingerprint, and an absent one admits it.

    A registry whose bytes will not decode raises ``RegistryError``; a dataset holding no
    registry fingerprints to a value other than its registered one.
    """
    import pytest

    from tcip_mcp.dataset_layout import subjects_path
    from tcip_mcp.subject_registry import RegistryError

    _make_dataset(tmp_path)
    with_registry = dataset_fingerprint(tmp_path)
    path = subjects_path(tmp_path)
    ts.put_blob(path, b"{not json")
    with pytest.raises(RegistryError):
        dataset_fingerprint(tmp_path)

    ts.delete_blob(path)
    without = dataset_fingerprint(tmp_path)
    assert without is not None and without != with_registry


def test_bespoke_dataset_has_no_fingerprint(tmp_path):
    # images but no labels, and labels but no images, both -> None (never a fabricated identity)
    (tmp_path / "images" / "d").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "d" / "x.jpg")
    assert dataset_fingerprint(tmp_path) is None  # no labels
    assert dataset_fingerprint(tmp_path / "nonexistent") is None

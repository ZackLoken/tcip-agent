"""The whole-dataset content fingerprint (dataset identity).

Pins: the fingerprint reuses ``dataset_hash`` for its label term (no second label-hasher); it is
pixel-aware (a re-encode under the same filename changes it, a gap ``dataset_hash`` alone leaves
open); registry order matters but whitespace doesn't; it is content-addressed (enumeration-order- and
path-independent, so a moved dataset keeps its identity); and it is ``None`` for a bespoke dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import Annotation, BBox
from tcip_mcp import class_registry
from tcip_mcp.class_registry import Attribute, ClassRegistry, Subject
from tcip_mcp.pipelines import resolution
from tcip_mcp.pipelines.data import dataset_fingerprint as fingerprint_mod
from tcip_mcp.pipelines.data.dataset_fingerprint import dataset_fingerprint


def _make_dataset(root: Path, *, pixel=(120, 120, 120), bud_box=(10, 10, 40, 40), ext="jpg") -> None:
    """A minimal nested-schema dataset: one dated image + its bud label + a registry."""
    date = "2026-02-11"
    (root / "images" / date).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=pixel).save(root / "images" / date / f"IMG_1.{ext}")
    (root / "annotations" / date).mkdir(parents=True, exist_ok=True)
    json_io.write_annotations(
        root / "annotations" / date / "IMG_1.json",
        [Annotation(subject="bud", geometry=BBox(*bud_box))], 64, 64)
    class_registry.write_registry(
        root / "classes.json",
        ClassRegistry(subjects=(Subject(name="bud", description="a currant bud"),)))


def test_fingerprint_reuses_dataset_hash_for_labels(tmp_path, monkeypatch):
    _make_dataset(tmp_path)
    calls = []
    real = resolution.dataset_hash

    def recording_dataset_hash(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(resolution, "dataset_hash", recording_dataset_hash)
    fp = dataset_fingerprint(tmp_path)
    assert fp is not None
    assert calls, "dataset_fingerprint must call dataset_hash for its label term, not re-implement it"


def test_a_label_edit_changes_the_fingerprint(tmp_path):
    _make_dataset(tmp_path)
    before = dataset_fingerprint(tmp_path)
    # move the box -> label bytes change
    json_io.write_annotations(
        tmp_path / "annotations" / "2026-02-11" / "IMG_1.json",
        [Annotation(subject="bud", geometry=BBox(11, 11, 41, 41))], 64, 64)
    assert dataset_fingerprint(tmp_path) != before


def test_pixel_reencode_under_same_filename_changes_the_fingerprint(tmp_path):
    """dataset_hash (labels-only) is unchanged, but the fingerprint must change when the pixels
    change under an untouched filename + labels.

    Uses BMP (uncompressed) rather than JPEG so the re-encode is guaranteed to preserve the file's
    byte size regardless of color: a JPEG re-encode's size varies with content too, so a test built
    on it would pass identically for a pixel-blind names+size(+mtime) image term, proving nothing
    about pixel awareness specifically (that stat-only term is asserted below to not detect this
    edit, confirming the size channel really is closed off here).
    """
    _make_dataset(tmp_path, pixel=(120, 120, 120), ext="bmp")
    before_fp = dataset_fingerprint(tmp_path)
    before_labels = resolution.dataset_hash(tmp_path / "annotations" / "2026-02-11")
    img_path = tmp_path / "images" / "2026-02-11" / "IMG_1.bmp"
    size_before = img_path.stat().st_size
    # re-encode the image with different pixels, same filename, labels untouched
    Image.new("RGB", (64, 64), color=(0, 200, 0)).save(img_path)
    assert img_path.stat().st_size == size_before  # confirms the size channel is closed, not just JPEG luck
    assert resolution.dataset_hash(tmp_path / "annotations" / "2026-02-11") == before_labels  # labels-only: blind
    assert dataset_fingerprint(tmp_path) != before_fp  # fingerprint: pixel-aware, catches it even though size didn't


def test_registry_value_order_matters_but_whitespace_does_not(tmp_path):
    _make_dataset(tmp_path)
    # add an ordered attribute -> registry (and thus fingerprint) changes
    reg2 = ClassRegistry(subjects=(Subject(
        name="bud", description="a currant bud",
        attributes=(Attribute(name="opening", type="categorical", values=("closed", "open")),)),))
    class_registry.write_registry(tmp_path / "classes.json", reg2)
    with_attr = dataset_fingerprint(tmp_path)
    _make_dataset(tmp_path)  # reset registry to no-attr
    assert dataset_fingerprint(tmp_path) != with_attr

    # a whitespace-only reformat of classes.json must not change identity (canonical re-serialization)
    reg2_again = ClassRegistry(subjects=(Subject(
        name="bud", description="a currant bud",
        attributes=(Attribute(name="opening", type="categorical", values=("closed", "open")),)),))
    class_registry.write_registry(tmp_path / "classes.json", reg2_again)
    fp_a = dataset_fingerprint(tmp_path)
    cp = tmp_path / "classes.json"
    cp.write_text(json.dumps(json.loads(cp.read_text()), indent=4) + "\n\n", encoding="utf-8")  # reformat
    assert dataset_fingerprint(tmp_path) == fp_a


def test_fingerprint_is_content_addressed_move_preserves_it(tmp_path):
    src = tmp_path / "a"
    _make_dataset(src)
    dst = tmp_path / "b"
    shutil.copytree(src, dst)  # same content, different path
    assert dataset_fingerprint(dst) == dataset_fingerprint(src)


def test_flat_layout_does_not_collide_with_a_subdir_literally_named_annotations(tmp_path):
    """A flat annotations/*.json dataset must not key its label term identically to a nested
    dataset whose one subdir happens to be literally named 'annotations'; both would otherwise
    hash to the same key bytes (the flat branch keyed by the root dir's own name)."""
    flat = tmp_path / "flat"
    (flat / "annotations").mkdir(parents=True)
    json_io.write_annotations(
        flat / "annotations" / "A.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)

    nested = tmp_path / "nested"
    (nested / "annotations" / "annotations").mkdir(parents=True)
    json_io.write_annotations(
        nested / "annotations" / "annotations" / "A.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)

    assert (fingerprint_mod._labels_term(flat / "annotations")
            != fingerprint_mod._labels_term(nested / "annotations"))


def test_labels_term_excludes_a_bucket_sidecar(tmp_path):
    """A bucket's own provenance stamp beside a real label does not change the dir's labels
    term, and a dir holding only one has no labels term at all."""
    d = tmp_path / "annotations"
    d.mkdir(parents=True)
    json_io.write_annotations(
        d / "A.json", [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)
    without_sidecar = fingerprint_mod._labels_term(d)

    (d / "operating_point.json").write_text("{}", encoding="utf-8")
    assert fingerprint_mod._labels_term(d) == without_sidecar

    sidecar_only = tmp_path / "sidecar_only"
    sidecar_only.mkdir()
    (sidecar_only / "operating_point.json").write_text("{}", encoding="utf-8")
    assert fingerprint_mod._labels_term(sidecar_only) is None


def test_rgb_nested_dataset_fingerprints_byte_identically_before_and_after_the_extension_widening(
        tmp_path):
    """The images term's extension set widened from a photographic-only set to
    ``image_utils.IMAGE_EXTS`` (adding ``.heic``/``.npy``/``.npz``/``.bandgroup``); ``.jpg`` was a
    member of both the old and the new set, so an RGB-only dataset under ``images/<date>/`` walks
    the identical file list and hashes to the identical value either side of the change.

    ``_images_term`` hashes each file's raw bytes, never decoded pixels, so the image is written
    as fixed literal bytes here rather than through Pillow's JPEG encoder: the literal below then
    depends on nothing but those fixed bytes and the label/registry writers below, not on
    Pillow/libjpeg's own encoder version or settings.
    """
    date = "2026-02-11"
    (tmp_path / "images" / date).mkdir(parents=True)
    (tmp_path / "images" / date / "IMG_1.jpg").write_bytes(
        b"fixed-jpeg-bytes-for-fingerprint-test")
    (tmp_path / "annotations" / date).mkdir(parents=True)
    json_io.write_annotations(
        tmp_path / "annotations" / date / "IMG_1.json",
        [Annotation(subject="bud", geometry=BBox(10, 10, 40, 40))], 64, 64)
    class_registry.write_registry(
        tmp_path / "classes.json",
        ClassRegistry(subjects=(Subject(name="bud", description="a currant bud"),)))

    assert dataset_fingerprint(tmp_path) == "v1:2b72f04cd064379a"


def test_bandgroup_manifest_file_itself_is_hashed_not_only_its_member_bands(tmp_path):
    """.bandgroup was not walked before this change (a mixed dataset's changed manifest content
    certified as identical). It is hashed as its own bytes, like any other file, so changing the
    manifest alone, with its named band files held byte-for-byte fixed, must change the
    fingerprint; it also fingerprints deterministically across two calls."""
    date = "2026-02-11"
    images = tmp_path / "images" / date
    images.mkdir(parents=True)
    (tmp_path / "annotations" / date).mkdir(parents=True)
    Image.new("L", (16, 16), color=10).save(images / "band_r.png")
    Image.new("L", (16, 16), color=20).save(images / "band_nir.png")
    manifest_path = images / "capture_1.bandgroup"
    json.dump({"bands": {"r": "band_r.png", "nir": "band_nir.png"}}, open(manifest_path, "w"))
    json_io.write_annotations(
        tmp_path / "annotations" / date / "capture_1.json",
        [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 16, 16)
    class_registry.write_registry(
        tmp_path / "classes.json",
        ClassRegistry(subjects=(Subject(name="bud"),)))

    fp1 = dataset_fingerprint(tmp_path)
    assert fp1 is not None
    assert dataset_fingerprint(tmp_path) == fp1  # deterministic across two calls

    # Rewrite the manifest's own bytes; the band files it names are untouched.
    json.dump({"bands": {"r": "band_r.png", "nir": "band_nir.png"},
              "central_wavelength_nm": {"r": 660.0, "nir": 850.0}}, open(manifest_path, "w"))
    assert dataset_fingerprint(tmp_path) != fp1


def test_bespoke_dataset_has_no_fingerprint(tmp_path):
    # images but no labels, and labels but no images, both -> None (never a fabricated identity)
    (tmp_path / "images" / "d").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "images" / "d" / "x.jpg")
    assert dataset_fingerprint(tmp_path) is None  # no labels
    assert dataset_fingerprint(tmp_path / "nonexistent") is None

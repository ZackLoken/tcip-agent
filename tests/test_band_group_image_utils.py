"""image_utils.list_logical_images / resolve_image_source, and the BandGroupRef-accepting
overloads of load_image / load_multiband / image_dimensions.

Uses the real DJI multispectral sample (copied into tmp_path, never mutated in place) for at least
one grouped-capture decode, plus synthetic 2-band fixtures for the cheaper/faster edge cases.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

REAL_DJI_DIR = Path(
    r"C:\Users\breeder\tcip-projects\currant_multispectral_valley-farm-2023\images\2023-05-23"
)
requires_real_dji_data = pytest.mark.skipif(
    not REAL_DJI_DIR.is_dir(),
    reason="real DJI multispectral sample project not present on this machine",
)


@pytest.fixture
def grouped_dir(tmp_path: Path) -> Path:
    """Two sibling single-band files sharing a manifest, plus one ordinary plain image, plus one
    unclaimed loose raster: the shape every enumeration call site needs to fold and pass through
    correctly at once."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    d = tmp_path / "images"
    d.mkdir()
    band_a = d / "cap_G.tif"
    band_b = d / "cap_R.tif"
    tifffile.imwrite(str(band_a), np.full((8, 8), 111, dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((8, 8), 222, dtype=np.uint16))
    write_band_group_manifest(d, "cap", {"Green": band_a, "Red": band_b})

    from PIL import Image
    Image.new("RGB", (8, 8), (10, 20, 30)).save(d / "plain.jpg")
    tifffile.imwrite(str(d / "loose.tif"), np.zeros((8, 8), dtype=np.uint16))
    return d


# ── list_logical_images / resolve_image_source ─────────────────────────────────────────


def test_list_logical_images_folds_a_group_and_keeps_the_rest(grouped_dir):
    from tcip_mcp.pipelines.image_utils import BandGroupRef, list_logical_images

    logical = list_logical_images(grouped_dir)
    assert set(logical) == {"cap", "plain", "loose"}
    assert isinstance(logical["cap"], BandGroupRef)
    assert isinstance(logical["plain"], Path)
    assert isinstance(logical["loose"], Path)
    # The group's own sibling files never also appear as their own raw-file entries.
    assert "cap_G" not in logical and "cap_R" not in logical


def test_resolve_image_source_is_the_single_lookup(grouped_dir):
    from tcip_mcp.pipelines.image_utils import list_logical_images, resolve_image_source

    logical = list_logical_images(grouped_dir)
    for stem in logical:
        assert resolve_image_source(grouped_dir, stem) == logical[stem]


def test_resolve_image_source_unknown_stem_raises_file_not_found(grouped_dir):
    from tcip_mcp.pipelines.image_utils import resolve_image_source

    with pytest.raises(FileNotFoundError):
        resolve_image_source(grouped_dir, "does_not_exist")


def test_resolve_image_source_stale_group_raises_band_group_incomplete(grouped_dir):
    from tcip_mcp.pipelines.image_utils import BandGroupIncomplete, resolve_image_source

    (grouped_dir / "cap_R.tif").unlink()
    with pytest.raises(BandGroupIncomplete):
        resolve_image_source(grouped_dir, "cap")


def test_list_logical_images_skips_a_corrupt_manifest_without_raising(tmp_path):
    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    (d / "broken.bandgroup").write_text("not json", encoding="utf-8")
    assert list_logical_images(d) == {}


def test_list_logical_images_on_missing_dir_returns_empty(tmp_path):
    from tcip_mcp.pipelines.image_utils import list_logical_images

    assert list_logical_images(tmp_path / "nope") == {}


def test_list_logical_images_refuses_a_stem_collision_between_a_group_and_a_standalone_file(
    tmp_path,
):
    """A raw standalone file whose own stem collides with a manifest's canonical stem must refuse
    loudly, not vanish silently with the manifest entry winning while the standalone file simply
    never gets added; every listing (training, splits, review, gallery) would otherwise lose it
    with no error."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    band_a = d / "cap_a_G.tif"
    band_b = d / "cap_a_R.tif"
    tifffile.imwrite(str(band_a), np.full((8, 8), 1, dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((8, 8), 2, dtype=np.uint16))
    # The manifest's own canonical stem (commonprefix of "cap_a_G"/"cap_a_R", trimmed) is "cap_a".
    write_band_group_manifest(d, "cap_a", {"Green": band_a, "Red": band_b})
    # An unrelated standalone file that happens to share that exact stem.
    tifffile.imwrite(str(d / "cap_a.tif"), np.full((8, 8), 3, dtype=np.uint16))

    with pytest.raises(AmbiguousImageStem):
        list_logical_images(d)


def test_bucket_logical_identities_over_the_grouped_fixture(grouped_dir):
    from tcip_mcp.pipelines.image_utils import bucket_logical_identities

    identities = bucket_logical_identities(grouped_dir)
    assert set(identities) == {"cap", "plain", "loose"}
    assert identities["cap"] == [grouped_dir / "cap.bandgroup"]
    assert identities["plain"] == [grouped_dir / "plain.jpg"]
    assert identities["loose"] == [grouped_dir / "loose.tif"]


def test_bucket_logical_identities_on_missing_dir_returns_empty(tmp_path):
    from tcip_mcp.pipelines.image_utils import bucket_logical_identities

    assert bucket_logical_identities(tmp_path / "nope") == {}


def test_list_logical_images_refuses_two_raw_files_sharing_one_stem(tmp_path):
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    (d / "foo.jpg").write_bytes(b"a")
    (d / "foo.png").write_bytes(b"b")

    with pytest.raises(AmbiguousImageStem) as raised:
        list_logical_images(d)
    message = str(raised.value)
    assert "foo.jpg" in message and "foo.png" in message


def test_list_logical_images_refuses_a_case_variant_stem_pair(tmp_path):
    from tcip_mcp.pipelines.image_utils import AmbiguousImageStem, list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    (d / "Foo.jpg").write_bytes(b"a")
    (d / "foo.png").write_bytes(b"b")

    with pytest.raises(AmbiguousImageStem) as raised:
        list_logical_images(d)
    message = str(raised.value)
    assert "Foo.jpg" in message and "foo.png" in message


def test_list_logical_images_with_a_corrupt_manifest_beside_a_same_stem_raw_file_lists_it(
    tmp_path,
):
    """A corrupt manifest claims nothing and is no identity: a raw file sharing its stem is not
    ambiguous, unlike a readable manifest under the same stem."""
    from PIL import Image

    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    (d / "broken.bandgroup").write_text("not json", encoding="utf-8")
    Image.new("RGB", (4, 4)).save(d / "broken.jpg")

    logical = list_logical_images(d)
    assert set(logical) == {"broken"}
    assert logical["broken"] == d / "broken.jpg"


def test_list_logical_images_propagates_schema_version_refused(tmp_path):
    import json

    import tcip_store as ts

    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    (d / "cap.bandgroup").write_text(
        json.dumps({"bands": {"Red": "a.tif"}, "schema_version": 2}), encoding="utf-8"
    )

    with pytest.raises(ts.SchemaVersionRefused):
        list_logical_images(d)


def test_list_logical_images_does_not_raise_when_stems_are_all_distinct(grouped_dir):
    """A rail must admit valid work: the ordinary (non-colliding) grouped_dir fixture used
    throughout this file must keep resolving cleanly; the new refusal is scoped to a genuine
    collision, not triggered by every group's mere presence."""
    from tcip_mcp.pipelines.image_utils import list_logical_images

    logical = list_logical_images(grouped_dir)
    assert set(logical) == {"cap", "plain", "loose"}


def test_image_exts_recognizes_npy_npz_and_bandgroup():
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    for ext in (".npy", ".npz", ".bandgroup", ".jpg", ".png", ".tif", ".tiff"):
        assert ext in IMAGE_EXTS


def test_stem_of_a_plain_path_and_a_band_group(grouped_dir):
    from tcip_mcp.pipelines.image_utils import resolve_image_source, stem_of

    assert stem_of(str(grouped_dir / "plain.jpg")) == "plain"
    assert stem_of(grouped_dir / "plain.jpg") == "plain"
    assert stem_of(resolve_image_source(grouped_dir, "cap")) == "cap"


def test_logical_image_name_agrees_with_display_source_paths_basename(grouped_dir):
    """A caller building a by-name filename map can reach for either the direct
    ``logical_image_name`` call or ``Path(display_source_path(x)).name``; pin them to the same
    value for a plain path and a band group, so a caller's choice between the two is cosmetic."""
    from tcip_mcp.pipelines.image_utils import (
        display_source_path, logical_image_name, resolve_image_source,
    )

    plain = grouped_dir / "plain.jpg"
    assert Path(display_source_path(plain)).name == logical_image_name(plain) == "plain.jpg"

    ref = resolve_image_source(grouped_dir, "cap")
    assert Path(display_source_path(ref)).name == logical_image_name(ref)


def test_probe_channels_of_a_plain_tif_does_not_decode_pixels(grouped_dir, monkeypatch):
    """probe_channels reads a TIFF's header-only series shape when it can, never paying for a
    full pixel decode just to learn the band count."""
    import tifffile as _tifffile

    from tcip_mcp.pipelines.derivations import probe_channels

    called = []
    real_imread = _tifffile.imread

    def _spy_imread(*a, **kw):
        called.append(True)
        return real_imread(*a, **kw)

    monkeypatch.setattr(_tifffile, "imread", _spy_imread)
    n = probe_channels(grouped_dir / "loose.tif")
    assert n == 1
    assert called == []  # header-only path succeeded; the full decode was never reached


# ── BandGroupRef-accepting decode overloads (synthetic) ────────────────────────────────


def test_image_dimensions_of_a_band_group(grouped_dir):
    from tcip_mcp.pipelines.image_utils import image_dimensions, resolve_image_source

    ref = resolve_image_source(grouped_dir, "cap")
    assert image_dimensions(ref) == (8, 8)


def test_load_multiband_stacks_siblings_in_declared_order(grouped_dir):
    from tcip_mcp.pipelines.image_utils import load_multiband, resolve_image_source

    ref = resolve_image_source(grouped_dir, "cap")
    arr = load_multiband(ref, 2)
    assert arr.shape == (8, 8, 2)
    # Declared order is {"Green": band_a (111), "Red": band_b (222)}.
    assert int(arr[0, 0, 0]) == 111
    assert int(arr[0, 0, 1]) == 222


def test_load_image_dispatches_a_band_group_to_load_multiband(grouped_dir):
    from tcip_mcp.pipelines.image_utils import load_image, resolve_image_source

    ref = resolve_image_source(grouped_dir, "cap")
    arr = load_image(ref, 2)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8, 2)


def test_probe_channels_of_a_band_group_sums_each_siblings_own_count(grouped_dir):
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import resolve_image_source

    ref = resolve_image_source(grouped_dir, "cap")
    assert probe_channels(ref) == 2  # 1 band each, summed (never assumed)


# ── BandGroupRef-accepting decode overloads (real DJI data) ────────────────────────────


@requires_real_dji_data
def test_real_dji_capture_decodes_as_a_4_band_stack(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import (
        image_dimensions, load_image, resolve_image_source,
    )

    d = tmp_path / "images"
    d.mkdir()
    for p in REAL_DJI_DIR.iterdir():
        shutil.copy2(p, d / p.name)
    detect_and_write_band_groups(d)

    logical_stem = sorted(p.stem for p in d.glob("*.bandgroup"))[0]
    ref = resolve_image_source(d, logical_stem)
    assert probe_channels(ref) == 4
    assert image_dimensions(ref) == (2592, 1944)  # the real DJI M3M frame size

    arr = load_image(ref, 4)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1944, 2592, 4)
    assert arr.dtype == np.uint16

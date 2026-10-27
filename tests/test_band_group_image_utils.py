"""Logical-image listing and resolution, and the band-group decode overloads.

Covers image_utils.list_logical_images / resolve_image_path, and the BandGroupRef-accepting
overloads of load_image / image_dimensions.

A grouped-capture decode reads the real DJI multispectral sample copied into tmp_path; the
other cases read synthetic 2-band fixtures.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
import tifffile

from tcip_mcp.dataset_layout import image_dir, image_root
from tests._producer_fixtures import GroupedDataset

if TYPE_CHECKING:
    from collections.abc import Callable

REAL_DJI_DIR = image_dir(
    r"C:\Users\breeder\tcip-projects\currant_multispectral_valley-farm-2023",
    "2023-05-23",
)
requires_real_dji_data = pytest.mark.skipif(
    not REAL_DJI_DIR.is_dir(),
    reason="real DJI multispectral sample project not present on this machine",
)


@pytest.fixture
def grouped_dir(tmp_path: Path) -> GroupedDataset:
    """Write a directory of a two-band group and a plain image; return what was written.

    The group is two sibling single-band files sharing a manifest, valued 111 and 222.
    """
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tests._producer_fixtures import write_image

    d = image_root(tmp_path)
    d.mkdir()
    bands = {"Green": d / "cap_G.tif", "Red": d / "cap_R.tif"}
    for band, value in zip(bands.values(), (111, 222), strict=True):
        tifffile.imwrite(str(band), np.full((8, 8), value, dtype=np.uint16))
    manifest = write_band_group_manifest(
        d, "cap", bands, central_wavelength_nm=None, identity_tolerance={}
    )
    plain = write_image(d / "plain.jpg", (8, 8), (10, 20, 30))
    return GroupedDataset(tmp_path, d, manifest, bands, plain)


@pytest.fixture
def loose_raster(grouped_dir: GroupedDataset) -> Path:
    """Write an unclaimed single-band ``loose.tif`` beside ``grouped_dir``'s; return it."""
    loose = grouped_dir.images_dir / "loose.tif"
    tifffile.imwrite(str(loose), np.zeros((8, 8), dtype=np.uint16))
    return loose


# ── list_logical_images / resolve_image_path ───────────────────────────────────────────


def test_list_logical_images_folds_a_group_and_keeps_the_rest(
    grouped_dir: GroupedDataset, loose_raster: Path
) -> None:
    from tcip_mcp.pipelines.image_utils import (
        BandGroupRef,
        list_logical_images,
    )

    logical = list_logical_images(grouped_dir.images_dir)
    assert set(logical) == {"cap", "plain", "loose"}
    assert isinstance(logical["cap"], BandGroupRef)
    assert isinstance(logical["plain"], Path)
    assert isinstance(logical["loose"], Path)
    # The group's own sibling files never also appear as their own raw-file entries.
    assert "cap_G" not in logical and "cap_R" not in logical


def test_resolve_image_path_answers_what_the_listing_names(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import (
        list_logical_images,
        resolve_image_paths,
        source_path_of,
    )

    logical = list_logical_images(grouped_dir.images_dir)
    assert resolve_image_paths(
        source_path_of(src) for src in logical.values()
    ) == list(logical.values())


def test_resolve_image_path_refuses_a_band_member_naming_its_manifest(
    grouped_dir: GroupedDataset,
) -> None:
    """``resolve_image_path`` refuses a band member naming its manifest, and admits the manifest.

    A band of a grouped capture is no logical image of its own: its path refuses naming the
    manifest that claims it, and the manifest's path admits the grouped capture.
    """
    from tcip_mcp.pipelines.image_utils import BandGroupRef, resolve_image_path

    with pytest.raises(FileNotFoundError, match=grouped_dir.manifest.name):
        resolve_image_path(grouped_dir.bands["Red"])
    assert isinstance(resolve_image_path(grouped_dir.manifest), BandGroupRef)


def test_resolve_image_path_unknown_path_raises_file_not_found(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import resolve_image_path

    with pytest.raises(FileNotFoundError):
        resolve_image_path(grouped_dir.images_dir / "does_not_exist.jpg")


def test_resolve_image_path_stale_group_raises_band_group_incomplete(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import (
        BandGroupIncompleteError,
        resolve_image_path,
    )

    grouped_dir.bands["Red"].unlink()
    with pytest.raises(BandGroupIncompleteError):
        resolve_image_path(grouped_dir.manifest)


def test_list_logical_images_skips_a_corrupt_manifest_without_raising(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = image_root(tmp_path)
    d.mkdir()
    (d / "broken.bandgroup").write_text("not json", encoding="utf-8")
    assert list_logical_images(d) == {}


def test_list_logical_images_on_missing_dir_returns_empty(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.image_utils import list_logical_images

    assert list_logical_images(tmp_path / "nope") == {}


def test_list_logical_images_refuses_a_stem_collision_between_a_group_and_a_standalone_file(
    tmp_path: Path,
) -> None:
    """``list_logical_images`` refuses a standalone file sharing a group manifest's stem.

    A raw standalone file whose own stem collides with a manifest's canonical stem is refused
    loudly, not dropped silently behind the manifest entry.
    """
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStemError,
        list_logical_images,
    )

    d = image_root(tmp_path)
    d.mkdir()
    band_a = d / "cap_a_G.tif"
    band_b = d / "cap_a_R.tif"
    tifffile.imwrite(str(band_a), np.full((8, 8), 1, dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((8, 8), 2, dtype=np.uint16))
    # The manifest's own canonical stem (commonprefix of "cap_a_G"/"cap_a_R", trimmed) is "cap_a".
    write_band_group_manifest(
        d,
        "cap_a",
        {"Green": band_a, "Red": band_b},
        central_wavelength_nm=None,
        identity_tolerance={},
    )
    # An unrelated standalone file that happens to share that exact stem.
    tifffile.imwrite(str(d / "cap_a.tif"), np.full((8, 8), 3, dtype=np.uint16))

    with pytest.raises(AmbiguousImageStemError):
        list_logical_images(d)


def test_bucket_logical_identities_over_the_grouped_fixture(
    grouped_dir: GroupedDataset, loose_raster: Path
) -> None:
    from tcip_mcp.pipelines.image_utils import bucket_logical_identities

    identities = bucket_logical_identities(grouped_dir.images_dir)
    assert identities == {
        "cap": [grouped_dir.manifest],
        "plain": [grouped_dir.plain],
        "loose": [loose_raster],
    }


def test_bucket_logical_identities_on_missing_dir_returns_empty(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.image_utils import bucket_logical_identities

    assert bucket_logical_identities(tmp_path / "nope") == {}


def test_list_logical_images_refuses_two_raw_files_sharing_one_stem(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStemError,
        list_logical_images,
    )

    d = image_root(tmp_path)
    d.mkdir()
    (d / "foo.jpg").write_bytes(b"a")
    (d / "foo.png").write_bytes(b"b")

    with pytest.raises(AmbiguousImageStemError) as raised:
        list_logical_images(d)
    message = str(raised.value)
    assert "foo.jpg" in message and "foo.png" in message


def test_list_logical_images_refuses_a_case_variant_stem_pair(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStemError,
        list_logical_images,
    )

    d = image_root(tmp_path)
    d.mkdir()
    (d / "Foo.jpg").write_bytes(b"a")
    (d / "foo.png").write_bytes(b"b")

    with pytest.raises(AmbiguousImageStemError) as raised:
        list_logical_images(d)
    message = str(raised.value)
    assert "Foo.jpg" in message and "foo.png" in message


def test_list_logical_images_with_a_corrupt_manifest_beside_a_same_stem_raw_file_lists_it(
    tmp_path: Path,
) -> None:
    """A raw file beside a corrupt manifest of the same stem is listed.

    Beside ``broken.bandgroup`` holding no JSON, the stem ``broken`` lists once, as
    ``broken.jpg``.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tests._producer_fixtures import write_image

    d = image_root(tmp_path)
    d.mkdir()
    (d / "broken.bandgroup").write_text("not json", encoding="utf-8")
    raw = write_image(d / "broken.jpg", (4, 4))

    logical = list_logical_images(d)
    assert set(logical) == {"broken"}
    assert logical["broken"] == raw


def test_list_logical_images_does_not_raise_when_stems_are_all_distinct(
    grouped_dir: GroupedDataset, loose_raster: Path
) -> None:
    """``list_logical_images`` lists every stem when the stems are all distinct.

    A directory holding a band group, a plain image and a loose raster lists the three stems.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images

    logical = list_logical_images(grouped_dir.images_dir)
    assert set(logical) == {"cap", "plain", "loose"}


def test_image_exts_recognizes_npy_npz_and_bandgroup() -> None:
    from tcip_mcp.pipelines.image_utils import IMAGE_EXTS

    for ext in (".npy", ".npz", ".bandgroup", ".jpg", ".png", ".tif", ".tiff"):
        assert ext in IMAGE_EXTS


def test_stem_of_a_plain_path_and_a_band_group(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import resolve_image_path, stem_of

    assert stem_of(str(grouped_dir.plain)) == "plain"
    assert stem_of(grouped_dir.plain) == "plain"
    assert stem_of(resolve_image_path(grouped_dir.manifest)) == "cap"


def test_logical_image_name_is_the_file_an_image_is_recorded_under(
    grouped_dir: GroupedDataset,
) -> None:
    """``logical_image_name`` is a plain image's own file name and a band group's manifest's."""
    from tcip_mcp.pipelines.image_utils import (
        logical_image_name,
        resolve_image_path,
    )

    assert logical_image_name(grouped_dir.plain) == "plain.jpg"
    assert (
        logical_image_name(resolve_image_path(grouped_dir.manifest))
        == "cap.bandgroup"
    )


def test_probe_channels_of_a_plain_tif_does_not_decode_pixels(
    loose_raster: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``probe_channels`` reads a TIFF's band count off its header without decoding a pixel.

    Neither ``TiffFile.asarray`` nor ``TiffPage.asarray`` runs.
    """
    import tifffile as _tifffile

    from tcip_mcp.pipelines.derivations import probe_channels

    def no_decode(*a: object, **kw: object) -> NoReturn:
        raise AssertionError("a TIFF was decoded to answer its band count")

    monkeypatch.setattr(_tifffile.TiffFile, "asarray", no_decode)
    monkeypatch.setattr(_tifffile.TiffPage, "asarray", no_decode)
    assert probe_channels(loose_raster) == 1


# ── BandGroupRef-accepting decode overloads (synthetic) ────────────────────────────────


def test_the_frame_of_a_band_group_at_its_own_count(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import (
        image_dimensions,
        resolve_image_path,
    )

    ref = resolve_image_path(grouped_dir.manifest)
    assert image_dimensions(ref, 2) == (8, 8)


def test_load_image_stacks_siblings_in_declared_order(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import load_image, resolve_image_path

    ref = resolve_image_path(grouped_dir.manifest)
    arr = load_image(ref, 2)
    assert arr.shape == (8, 8, 2)
    # Declared order is {"Green": band_a (111), "Red": band_b (222)}.
    assert int(arr[0, 0, 0]) == 111
    assert int(arr[0, 0, 1]) == 222


def test_load_image_of_a_band_group_is_an_array(
    grouped_dir: GroupedDataset,
) -> None:
    from tcip_mcp.pipelines.image_utils import load_image, resolve_image_path

    ref = resolve_image_path(grouped_dir.manifest)
    arr = load_image(ref, 2)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8, 2)


def test_a_band_groups_header_and_reader_state_one_band_per_member(
    grouped_dir: GroupedDataset,
) -> None:
    """A band group's header and reader state one band per member.

    A group of single-band members is as many bands as it has members, and the reader its header
    opens serves exactly those bands, each member's own values.
    """
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.raster_source import Rect, SourceHeader

    ref = resolve_image_path(grouped_dir.manifest)
    header = SourceHeader(ref)
    assert probe_channels(ref) == header.channels == 2
    with header.open(header.channels) as reader:
        assert reader.num_channels == header.channels
        pixels, _spec = reader.read_region(Rect(0, 0, 1, 1))
    assert pixels[0, 0].tolist() == [111, 222]


def test_a_band_group_of_channel_first_single_band_arrays_is_admitted_as_it_is_read(
    tmp_path: Path,
) -> None:
    """A band group of channel-first single-band arrays is admitted as it is read.

    A member stored as a ``(1, H, W)`` array is one band read the way it is opened, at one band:
    the header admits it and states the frame and count the reader serves.
    """
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.raster_source import Rect, SourceHeader

    d = image_root(tmp_path)
    d.mkdir()
    np.save(str(d / "cap_a.npy"), np.full((1, 8, 12), 7, dtype=np.uint8))
    np.savez(str(d / "cap_b.npz"), band=np.full((1, 8, 12), 9, dtype=np.uint8))
    write_band_group_manifest(
        d,
        "cap",
        {"A": d / "cap_a.npy", "B": d / "cap_b.npz"},
        central_wavelength_nm=None,
        identity_tolerance={},
    )

    header = SourceHeader(resolve_image_path(d / "cap.bandgroup"))
    assert header.channels == 2
    assert header.frame_at(2) == (12, 8)
    with header.open(header.channels) as reader:
        assert (reader.width, reader.height, reader.num_channels) == (12, 8, 2)
        pixels, _spec = reader.read_region(Rect(0, 0, 1, 1))
    assert pixels[0, 0].tolist() == [7, 9]


@pytest.mark.parametrize("save", [np.save, np.savez], ids=["npy", "npz"])
def test_a_band_group_naming_a_four_dimensional_array_is_refused_at_its_header(
    tmp_path: Path, save: Callable[[str, np.ndarray], None]
) -> None:
    """A band group naming a four-dimensional array is refused at its header.

    An array of four dimensions is no image of one band: the header refuses it when asked for its
    band count and when opened, naming the member file and its shape.
    """
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.raster_source import SourceHeader

    d = image_root(tmp_path)
    d.mkdir()
    ext = ".npy" if save is np.save else ".npz"
    for name in ("cap_a", "cap_b"):
        save(str(d / f"{name}{ext}"), np.zeros((1, 8, 12, 2), dtype=np.uint8))
    write_band_group_manifest(
        d,
        "cap",
        {"A": d / f"cap_a{ext}", "B": d / f"cap_b{ext}"},
        central_wavelength_nm=None,
        identity_tolerance={},
    )

    header = SourceHeader(resolve_image_path(d / "cap.bandgroup"))
    with pytest.raises(
        ValueError,
        match=r"cap_a\.np[yz]' decodes to an array shaped \(1, 8, 12, 2\)",
    ):
        _ = header.channels
    with pytest.raises(ValueError, match=r"shaped \(1, 8, 12, 2\)"):
        header.open(2)


def test_a_band_group_naming_a_multi_band_member_is_refused_at_its_header(
    tmp_path: Path,
) -> None:
    """A band group naming a three-band member is refused by name at its header."""
    from PIL import Image

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.pipelines.raster_source import SourceHeader
    from tests._producer_fixtures import write_image

    d = image_root(tmp_path)
    d.mkdir()
    gray = d / "cap_gray.png"
    Image.new("L", (8, 8), 123).save(gray)
    manifest = write_band_group_manifest(
        d,
        "cap",
        {
            "Red": write_image(d / "cap_rgb.png", (8, 8), (255, 0, 0)),
            "Gray": gray,
        },
        central_wavelength_nm=None,
        identity_tolerance={},
    )

    header = SourceHeader(resolve_image_path(manifest))
    with pytest.raises(ValueError, match="'Red'.*cap_rgb.png.*holds 3 bands"):
        _ = header.channels
    with pytest.raises(ValueError, match="holds 3 bands"):
        header.open(2)


# ── BandGroupRef-accepting decode overloads (real DJI data) ────────────────────────────


@requires_real_dji_data
def test_real_dji_capture_decodes_as_a_4_band_stack(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )
    from tcip_mcp.pipelines.derivations import probe_channels
    from tcip_mcp.pipelines.image_utils import (
        image_dimensions,
        load_image,
        resolve_image_path,
    )

    d = image_root(tmp_path)
    d.mkdir()
    for p in REAL_DJI_DIR.iterdir():
        shutil.copy2(p, d / p.name)
    detect_and_write_band_groups(d)

    logical_stem = sorted(p.stem for p in d.glob("*.bandgroup"))[0]
    ref = resolve_image_path(d / f"{logical_stem}.bandgroup")
    assert probe_channels(ref) == 4
    assert image_dimensions(ref, 4) == (
        2592,
        1944,
    )  # the real DJI M3M frame size

    arr = load_image(ref, 4)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1944, 2592, 4)
    assert arr.dtype == np.uint16

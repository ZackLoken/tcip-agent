"""TiledDetectionDataset: keep-region filtering, the sampler-facing properties, eager
backend-conditioned opens, windowed tile reads, and worker pickling.

Fixtures use a 200x200 frame with tile 64 / overlap 0.2 (stride 51): origins 0/51/102/153 per
axis, so tiles at 153 overhang the extent (153 + 64 > 200).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcip_mcp.pipelines import raster_source
from tcip_mcp.pipelines.data import datasets as datasets_module
from tcip_mcp.pipelines.data.datasets import DetectionDataset, TiledDetectionDataset
from tcip_mcp.pipelines.image_utils import crop_pad_tile, load_image, pil_to_tensor


@pytest.fixture(autouse=True)
def _empty_source_pool():
    """Each test starts and ends with an empty pool: it is process-global state."""
    raster_source.close_source_pool()
    yield
    raster_source.close_source_pool()


def _distinctive(height: int, width: int, channels: int = 3, dtype=np.uint8) -> np.ndarray:
    arr = np.zeros((height, width, channels), dtype=dtype)
    for c in range(channels):
        arr[:, :, c] = (np.add.outer(np.arange(height) * (c + 1), np.arange(width))) % 251
    return arr


def _write_labels(labels_dir: Path, stem: str, size_wh: tuple[int, int]) -> None:
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    w, h = size_wh
    box = BBox(0.3 * w, 0.3 * h, 0.5 * w, 0.5 * h)
    json_io.write_annotations(str(labels_dir / f"{stem}.json"),
                              [Annotation(subject="bud", geometry=box)], w, h,
                              keep_empty=True)


def _jpeg_project(tmp_path: Path, size: int = 200):
    from PIL import Image

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    Image.fromarray(_distinctive(size, size)).save(images_dir / "img0.jpg")
    _write_labels(labels_dir, "img0", (size, size))
    return images_dir, labels_dir


def _tiff_project(tmp_path: Path, height: int = 200, width: int = 200, dtype=np.uint8):
    import tifffile

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    arr = _distinctive(height, width, dtype=dtype)
    tifffile.imwrite(str(images_dir / "img0.tif"), arr, photometric="rgb", rowsperstrip=16)
    _write_labels(labels_dir, "img0", (width, height))
    return images_dir, labels_dir, arr


def _tiled(images_dir, labels_dir, **kwargs) -> TiledDetectionDataset:
    base = DetectionDataset(images_dir=str(images_dir), labels_dir=str(labels_dir),
                            subject="bud")
    return TiledDetectionDataset(base, tile_size=64, overlap=0.2, **kwargs)


# -- keep_regions ---------------------------------------------------------


def test_keep_regions_keeps_only_fully_contained_tiles(tmp_path):
    images_dir, labels_dir = _jpeg_project(tmp_path)
    full = _tiled(images_dir, labels_dir)
    assert len(full) == 16
    assert full.tiles_dropped_past_extent == 0
    assert full.tiles_dropped_outside_regions == 0

    ds = _tiled(images_dir, labels_dir, keep_regions=[(0, 0, 115, 115)])
    # Contained: origins 0/51 per axis (51 + 64 = 115, half-open). Overhanging: any origin 153.
    assert [(tx, ty) for _s, tx, ty in ds.tile_entries] == [(0, 0), (51, 0), (0, 51), (51, 51)]
    assert ds.tiles_dropped_past_extent == 7
    assert ds.tiles_dropped_outside_regions == 5
    assert len(ds) == 4

    # Kept tiles carry exactly the boxes the unfiltered dataset computed for the same tiles.
    by_key = {(e["stem"], e["tile_x"], e["tile_y"]): e for e in full._index}
    for e in ds._index:
        ref = by_key[(e["stem"], e["tile_x"], e["tile_y"])]
        assert np.array_equal(e["boxes"], ref["boxes"])
        assert np.array_equal(e["labels"], ref["labels"])


def test_keep_regions_union_across_disjoint_rects(tmp_path):
    images_dir, labels_dir = _jpeg_project(tmp_path)
    ds = _tiled(images_dir, labels_dir, keep_regions=[(0, 0, 64, 64), (102, 102, 166, 166)])
    assert [(tx, ty) for _s, tx, ty in ds.tile_entries] == [(0, 0), (102, 102)]
    assert ds.tiles_dropped_past_extent == 7
    assert ds.tiles_dropped_outside_regions == 7


def test_an_empty_keep_regions_sequence_keeps_nothing(tmp_path):
    images_dir, labels_dir = _jpeg_project(tmp_path)
    ds = _tiled(images_dir, labels_dir, keep_regions=[])
    assert len(ds) == 0
    assert ds.tiles_dropped_past_extent == 7
    assert ds.tiles_dropped_outside_regions == 9


def test_a_malformed_keep_region_refuses_by_name(tmp_path):
    images_dir, labels_dir = _jpeg_project(tmp_path)
    with pytest.raises(ValueError, match="keep region"):
        _tiled(images_dir, labels_dir, keep_regions=[(0, 0, 115)])
    with pytest.raises(ValueError, match="no extent"):
        _tiled(images_dir, labels_dir, keep_regions=[(115, 0, 0, 115)])


# -- Sampler-facing properties --------------------------------------------


def test_tile_entries_matches_index_order_and_getitem(tmp_path):
    images_dir, labels_dir = _jpeg_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    entries = ds.tile_entries
    assert len(entries) == len(ds)
    assert all(isinstance(s, str) and isinstance(tx, int) and isinstance(ty, int)
               for s, tx, ty in entries)
    assert entries[0] == ("img0", 0, 0)
    assert entries == [(e["stem"], e["tile_x"], e["tile_y"]) for e in ds._index]


def test_source_frames_for_a_photographic_source(tmp_path):
    images_dir, labels_dir = _jpeg_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    assert ds.source_frames == {
        "img0": {"width": 200, "height": 200, "channels": 3, "dtype_itemsize": None,
                 "windowed": False},
    }


def test_source_frames_for_a_windowed_raster(tmp_path):
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    assert ds.source_frames == {
        "img0": {"width": 200, "height": 200, "channels": 3, "dtype_itemsize": 1,
                 "windowed": True},
    }


# -- Eager opens, backend-conditioned -------------------------------------


def test_photographic_construction_opens_no_raster_backend(tmp_path, monkeypatch):
    """A directory of JPEGs must index from header probes alone: opening each one through the
    raster layer would decode every frame at construction."""
    images_dir, labels_dir = _jpeg_project(tmp_path)
    base = DetectionDataset(images_dir=str(images_dir), labels_dir=str(labels_dir),
                            subject="bud")

    def _refuse(*_a, **_k):
        raise AssertionError("construction must not open a raster backend for photographic sources")

    monkeypatch.setattr(raster_source, "open_raster", _refuse)
    ds = TiledDetectionDataset(base, tile_size=64, overlap=0.2)
    assert len(ds) == 16


def test_windowed_construction_registers_the_source_in_the_pool(tmp_path):
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    tiff = str(images_dir / "img0.tif")
    assert any(key[0] == tiff for key in raster_source._POOL)
    assert ds.source_frames["img0"]["windowed"] is True


def test_an_unopenable_windowed_layout_refuses_at_construction(tmp_path):
    """A raster that cannot be opened surfaces its refusal when the index is built, not at
    step N of an epoch."""
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    # Corrupt the file after writing the labels: header parse now fails.
    (images_dir / "img0.tif").write_bytes(b"II*\x00garbage")
    with pytest.raises(ValueError, match="cannot open raster"):
        _tiled(images_dir, labels_dir)


# -- Windowed reads -------------------------------------------------------


def test_windowed_tiles_match_the_whole_decode_crop(tmp_path):
    """Every tile served through the windowed path is identical to the whole-decode crop of the
    same source: interior tiles, right/bottom edge tiles, and the zero-padded corner."""
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    img = load_image(images_dir / "img0.tif", 3)
    assert len(ds.tile_entries) == 16, "the 200x200 frame tiles into 16 windows at stride 51"
    for i, (_stem, tx, ty) in enumerate(ds.tile_entries):
        expected = pil_to_tensor(crop_pad_tile(img, tx, ty, 64, 200, 200))
        got, target = ds[i]
        assert torch.equal(got, expected), f"tile at ({tx}, {ty}) differs"
        assert target["boxes"].shape[1] == 4


def test_windowed_getitem_never_whole_decodes(tmp_path, monkeypatch):
    """A windowed stem's tiles come from pooled window reads; the whole-decode loader must not
    run at all, or a huge raster would be decoded whole per tile."""
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)

    def _refuse(*_a, **_k):
        raise AssertionError("windowed __getitem__ must not decode the source whole")

    monkeypatch.setattr(datasets_module, "load_image", _refuse)
    img_tensor, _target = ds[0]
    assert tuple(img_tensor.shape) == (3, 64, 64)


def test_a_replaced_windowed_source_refuses_at_read(tmp_path):
    """The pool keys on mtime and size, so a file replaced after the index was built opens fresh
    with its own dims, and the dims disagreement refuses instead of serving displaced tiles."""
    import tifffile

    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    tifffile.imwrite(str(images_dir / "img0.tif"), _distinctive(150, 180), photometric="rgb",
                     rowsperstrip=16)
    with pytest.raises(ValueError, match="frame changed"):
        ds[0]


def test_a_decoder_disagreeing_with_its_header_refuses(tmp_path, monkeypatch):
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    real = raster_source.GdalSource.read_region

    def short(self, rect, **kwargs):
        region, spec = real(self, rect, **kwargs)
        return region[:-1], spec

    monkeypatch.setattr(raster_source.GdalSource, "read_region", short)
    with pytest.raises(ValueError, match="decoder disagrees"):
        ds[0]


def test_a_uint16_windowed_tile_stays_ndarray_and_scales_by_its_dtype(tmp_path):
    images_dir, labels_dir, arr = _tiff_project(tmp_path, dtype=np.uint16)
    ds = _tiled(images_dir, labels_dir)
    assert ds.source_frames["img0"]["dtype_itemsize"] == 2
    img_tensor, _target = ds[0]
    expected = torch.from_numpy(
        arr[0:64, 0:64].astype(np.float32) / 65535.0).permute(2, 0, 1)
    assert torch.allclose(img_tensor, expected)


# -- Worker pickling ------------------------------------------------------


def test_dataset_pickles_after_construction_and_after_a_read(tmp_path):
    """Spawned DataLoader workers receive the dataset by pickle, so it must never hold an open
    RasterSource, before or after serving a tile."""
    images_dir, labels_dir, _arr = _tiff_project(tmp_path)
    ds = _tiled(images_dir, labels_dir)
    clone = pickle.loads(pickle.dumps(ds))
    original_tile, _ = ds[0]
    clone_after = pickle.loads(pickle.dumps(ds))
    for restored in (clone, clone_after):
        tile, _ = restored[0]
        assert torch.equal(tile, original_tile)

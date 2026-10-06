"""Every enumeration/resolution call site routed onto ``list_logical_images``/
``resolve_image_source``, exercised against a grouped-capture folder: datasets.py, splits.py,
annotation_tools.py, vision_tools.py.

A minimal synthetic 2-band group (two tiny single-band TIFFs + a manifest) stands in for a real
capture in most of these: the mechanism under test is "does the call site fold the group and
route pixels through image_utils", which the real DJI sample already proves at the band_groups
layer in test_band_groups.py / test_band_group_image_utils.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from tests._producer_fixtures import dataset_over, label_image, registry_over  # noqa: E402


def _write_group(images_dir: Path, stem: str, fill=(111, 222)) -> None:
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    band_a = images_dir / f"{stem}_G.tif"
    band_b = images_dir / f"{stem}_R.tif"
    tifffile.imwrite(str(band_a), np.full((16, 16), fill[0], dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((16, 16), fill[1], dtype=np.uint16))
    write_band_group_manifest(images_dir, stem, {"Green": band_a, "Red": band_b})


@pytest.fixture
def grouped_dataset(tmp_path: Path) -> Path:
    """A minimal dataset root: one grouped capture + one plain photo, each with a detection GT
    label document, under the canonical images/ layout ``build_dataset`` reads.
    """
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject

    root = tmp_path / "proj"
    date = "2026-04-01"
    images_dir = root / "images" / date
    images_dir.mkdir(parents=True)

    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud", description="a bud"),)))

    _write_group(images_dir, "capture_001")
    Image.new("RGB", (16, 16), (5, 5, 5)).save(images_dir / "plain_002.jpg")

    for source in ("capture_001.bandgroup", "plain_002.jpg"):
        label_image(images_dir / source, [Annotation(subject="bud", geometry=BBox(2, 2, 6, 6))],
                    16, 16)
    return root


# ── datasets.py ─────────────────────────────────────────────────────────────────────────


def test_a_group_is_named_by_its_manifest_wherever_a_name_is_read(grouped_dataset):
    """The one naming primitive every by-name reader resolves a capture under: a grouped
    capture answers with its own manifest filename, the file a name store holds it by, never
    one of its sibling band files."""
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

    sources = list_logical_images(image_dir(grouped_dataset, "2026-04-01"))
    names = {stem: logical_image_name(src) for stem, src in sources.items()}
    assert names["capture_001"] == "capture_001.bandgroup"
    assert names["plain_002"] == "plain_002.jpg"


def test_detection_dataset_trains_on_a_grouped_capture(grouped_dataset):
    """The grouped capture alone: this bucket also holds a plain RGB source, and one model reads
    one band count, so a run over both refuses rather than reading either at the other's count."""
    torch = pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import image_dir

    ds = dataset_over('detection', str(image_dir(grouped_dataset, "2026-04-01")), subject="bud",
                      members=["capture_001"])
    assert ds.expected_channels == 2  # derived from the group's own bands, not defaulted to RGB
    members = [ds.sample_of(key).member for key in ds.stems]
    assert "capture_001" in members
    idx = members.index("capture_001")
    img, target = ds[idx]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 2  # Green + Red, stacked
    assert target["boxes"].shape[0] == 1


def test_the_channel_probe_derives_2_for_the_grouped_sample(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.data.datasets import _band_count
    from tests._producer_fixtures import samples_over

    samples = samples_over(str(image_dir(grouped_dataset, "2026-04-01")), subject="bud",
                           members=["capture_001"])
    assert _band_count(samples) == 2


def test_the_channel_probe_raises_on_a_stale_manifest_instead_of_silently_defaulting(tmp_path):
    """A broad ``except Exception: return default`` must not swallow ``BandGroupIncomplete``
    along with genuinely unexpected errors and silently default to 3 channels: a
    confidently-wrong value on exactly the parameter 'derive, don't pin' exists to guard
    against."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, label_key
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, write_band_group_manifest
    from tcip_mcp.pipelines.data.datasets import _band_count
    from tcip_mcp.pipelines.data.selection import Sample

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    band_a = images_dir / "cap_G.tif"
    band_b = images_dir / "cap_R.tif"
    tifffile.imwrite(str(band_a), np.full((8, 8), 1, dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((8, 8), 2, dtype=np.uint16))
    manifest = write_band_group_manifest(images_dir, "cap", {"Green": band_a, "Red": band_b})
    band_b.unlink()  # the manifest now references a sibling that no longer exists

    sample = Sample(member="cap", source=str(manifest),
                    ground_truth=label_key(tmp_path, UNDATED_BUCKET, "cap"), group="g",
                    side="train")
    with pytest.raises(BandGroupIncomplete):
        _band_count([sample])


# ── image_utils.image_path_dimensions ───────────────────────────────────────────────────


def test_image_path_dimensions_of_a_grouped_capture_path(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.image_utils import image_path_dimensions

    manifest = image_dir(grouped_dataset, "2026-04-01") / "capture_001.bandgroup"
    assert image_path_dimensions(str(manifest)) == (16, 16)


def test_focus_annotate_lands_on_the_grouped_capture_by_manifest_name(grouped_dataset):
    from tcip_mcp.tools.gui_tools import focus_human_attention

    res = focus_human_attention(grouped_dataset, grouped_dataset.parent, str(grouped_dataset),
                                "bud", "2026-04-01")
    assert "error" not in res
    assert res["n_images"] == 2
    # Sorted names: "capture_001.bandgroup" < "plain_002.jpg"
    assert res["image"] == "capture_001.bandgroup"
    assert res["image_index"] == 0


# ── vision_tools.py ─────────────────────────────────────────────────────────────────────


def test_display_read_composites_a_group_into_rgb_pixels(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import _display_for_path

    manifest = image_dir(grouped_dataset, "2026-04-01") / "capture_001.bandgroup"
    read = _display_for_path(str(manifest))
    assert read.pixels.shape == (16, 16, 3)
    assert read.pixels.dtype == np.uint8
    assert read.native_size == (16, 16)


def test_display_read_of_a_plain_photo_is_the_files_own_pixels(grouped_dataset):
    """A photographic file is read, never stretched: what the renderer draws on is what the
    file holds."""
    from PIL import Image

    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import _display_for_path

    plain = image_dir(grouped_dataset, "2026-04-01") / "plain_002.jpg"
    read = _display_for_path(str(plain))
    assert np.array_equal(read.pixels, np.asarray(Image.open(plain).convert("RGB")))


def test_display_read_of_a_plain_3band_rgb_geotiff_keeps_its_true_colors(tmp_path):
    """An ordinary 3-band RGB .tif is a real, pre-existing supported format: it must reach the
    renderer as its own pixels, not as a synthetic per-channel min-max stretch of them."""
    from tcip_mcp.tools.vision_tools import _display_for_path

    d = tmp_path / "images"
    d.mkdir()
    # Real, non-constant RGB content (never a flat fill: a flat array's min==max would make a
    # stretch indistinguishable from the original by accident).
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(10, 200, 12, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.linspace(5, 90, 12, dtype=np.uint8)[None, :]
    rgb[..., 2] = 40
    path = d / "plain_rgb.tif"
    tifffile.imwrite(str(path), rgb, photometric="rgb")

    assert np.array_equal(_display_for_path(str(path)).pixels, rgb)


def test_display_read_still_stretches_a_genuinely_multiband_geotiff(tmp_path):
    """The scoping above must not swallow the real non-standard case: a raster with more bands
    than any true-color reading covers is composited to three display bands and stretched."""
    from tcip_mcp.tools.vision_tools import _display_for_path

    d = tmp_path / "images"
    d.mkdir()
    # Per-band value levels plus a within-band gradient, so a min-max stretch has something to
    # span and its result is distinguishable from the raw values.
    arr = np.zeros((10, 12, 6), dtype=np.uint16)
    for i in range(6):
        arr[:, :, i] = (i + 1) * 100 + np.linspace(0, 300, 12, dtype=np.uint16)[None, :]
    path = d / "multiband.tif"
    tifffile.imwrite(str(path), arr)

    pixels = _display_for_path(str(path)).pixels
    assert pixels.shape == (10, 12, 3)      # three display bands out of six
    assert pixels.dtype == np.uint8
    assert (pixels.min(), pixels.max()) == (0, 255)   # each band stretched across the range


def test_visualize_annotations_on_a_grouped_capture(tmp_path, grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import visualize

    manifest = image_dir(grouped_dataset, "2026-04-01") / "capture_001.bandgroup"
    result = visualize(tmp_path, source="annotations", path=str(manifest))
    assert "error" not in result
    assert result["count"] == 1
    assert Path(result["image_path"]).is_file()


def test_visualize_annotations_on_a_band_member_answers_the_resolvers_refusal(tmp_path,
                                                                             grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import visualize

    member = image_dir(grouped_dataset, "2026-04-01") / "capture_001_G.tif"
    for source in ("annotations", "predictions", "comparison"):
        result = visualize(tmp_path, source=source, path=str(member), bucket="preds")
        assert "capture_001.bandgroup" in result["error"], (source, result)
        assert "name the manifest" in result["error"]


def test_viz_dataset_sample_folds_a_grouped_capture_into_one_entry(tmp_path, grouped_dataset):
    from tcip_mcp.tools.vision_tools import visualize

    result = visualize(tmp_path, source="dataset", path=str(grouped_dataset), n=16)
    assert "error" not in result
    assert result["total_images"] == 2  # one grouped capture + one plain photo, never 3 raw files


# ── tools/data_tools.py: draw_splits(materialize=True) ───────────────────────────────────


def test_a_selection_records_a_grouped_capture_by_its_own_manifest_path(tmp_path, grouped_dataset):
    """A drawn selection names a grouped capture by the manifest sitting beside its bands, and
    that path resolves back to the whole group in place: the group's bands are read where they
    were captured, never copied into a side's own directory."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.pipelines.image_utils import resolve_image_path
    from tcip_mcp.tools.data_tools import draw_splits

    # The fixture's own two groups (capture_001, plain_002) need two more to clear a draw's
    # foreground floor, added here rather than in the shared fixture.
    images_dir = grouped_dataset / "images" / "2026-04-01"
    for stem in ("plain_003", "plain_004"):
        Image.new("RGB", (16, 16), (5, 5, 5)).save(images_dir / f"{stem}.jpg")
        label_image(images_dir / f"{stem}.jpg",
                    [Annotation(subject="bud", geometry=BBox(2, 2, 6, 6))], 16, 16)

    out = grouped_dataset / "splits"
    result = draw_splits(tmp_path, str(grouped_dataset), output_path=str(out),
                         subject="bud", seed=1, val_ratio=0.25,
                         calibration_ratio=0.125, holdout_ratio=0.125)
    assert "error" not in result, result

    drawn = read_selection(out, project=tmp_path)
    grouped = next(s for s in drawn.samples if Path(s.source).stem == "capture_001")
    assert Path(grouped.source).parent == images_dir  # read in place, never copied
    assert Path(grouped.source).suffix == ".bandgroup"  # the admission records the manifest
    resolved = resolve_image_path(grouped.source)
    assert isinstance(resolved, BandGroupRef)
    assert all(p.is_file() for p in resolved.bands.values())


def test_scan_dataset_counts_a_grouped_capture_once_not_once_per_band_file(grouped_dataset):
    """``_scan_dataset``'s image census is built through ``list_logical_images``: a grouped
    capture is one logical image, its own manifest, never one entry per sibling band file."""
    from tcip_mcp.tools.data_tools import _scan_dataset

    scan = _scan_dataset(str(grouped_dataset))
    stems = {Path(p).stem for p in scan["images"]}
    assert stems == {"capture_001", "plain_002"}
    assert len(scan["images"]) == 2

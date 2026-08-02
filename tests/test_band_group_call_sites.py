"""Every enumeration/resolution call site routed onto ``list_logical_images``/
``resolve_image_source``, exercised against a grouped-capture folder: datasets.py, splits.py,
annotation_tools.py, vision_tools.py, feedback/materialize.py.

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
    label, the canonical images/ + annotations/ layout ``build_dataset``/``label_image_stems``
    read.
    """
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    root = tmp_path / "proj"
    date = "2026-04-01"
    images_dir = root / "images" / date
    labels_dir = root / "annotations" / date
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    class_registry.write_registry(
        root / "classes.json",
        ClassRegistry(subjects=(Subject(name="catkin", description="a hazelnut catkin"),)),
    )

    _write_group(images_dir, "capture_001")
    Image.new("RGB", (16, 16), (5, 5, 5)).save(images_dir / "plain_002.jpg")

    for stem in ("capture_001", "plain_002"):
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject="catkin", geometry=BBox(2, 2, 6, 6))],
            16, 16,
        )
    return root


# ── datasets.py ─────────────────────────────────────────────────────────────────────────


def test_image_name_map_uses_manifest_filename_for_a_group(grouped_dataset):
    from tcip_mcp.pipelines.data.datasets import image_name_map
    from tcip_mcp.dataset_layout import image_dir

    names = image_name_map(image_dir(grouped_dataset, "2026-04-01"))
    assert names["capture_001"] == "capture_001.bandgroup"
    assert names["plain_002"] == "plain_002.jpg"


def test_detection_dataset_trains_on_a_grouped_capture(grouped_dataset):
    torch = pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import image_dir, annotation_dir
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    ds = DetectionDataset(
        images_dir=str(image_dir(grouped_dataset, "2026-04-01")),
        labels_dir=str(annotation_dir(grouped_dataset, "2026-04-01")),
        subject="catkin",
    )
    ds.expected_channels = 2  # the group's own band count (probe_channels would derive this)
    assert "capture_001" in ds.stems
    idx = ds.stems.index("capture_001")
    img, target = ds[idx]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 2  # Green + Red, stacked
    assert target["boxes"].shape[0] == 1


def test_probe_num_channels_derives_2_for_the_grouped_sample(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.data.datasets import _probe_num_channels

    n = _probe_num_channels(str(image_dir(grouped_dataset, "2026-04-01")), ["capture_001"])
    assert n == 2


def test_probe_num_channels_raises_on_a_stale_manifest_instead_of_silently_defaulting(tmp_path):
    """A broad ``except Exception: return default`` must not swallow ``BandGroupIncomplete``
    along with genuinely unexpected errors and silently default to 3 channels: a
    confidently-wrong value on exactly the parameter 'derive, don't pin' exists to guard against.
    No ``stems`` given, so the single-sample fallback (not the per-stem loop) is the path under
    test."""
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, write_band_group_manifest
    from tcip_mcp.pipelines.data.datasets import _probe_num_channels

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    band_a = images_dir / "cap_G.tif"
    band_b = images_dir / "cap_R.tif"
    tifffile.imwrite(str(band_a), np.full((8, 8), 1, dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((8, 8), 2, dtype=np.uint16))
    write_band_group_manifest(images_dir, "cap", {"Green": band_a, "Red": band_b})
    band_b.unlink()  # the manifest now references a sibling that no longer exists

    with pytest.raises(BandGroupIncomplete):
        _probe_num_channels(str(images_dir), None)


# ── splits.py ───────────────────────────────────────────────────────────────────────────


def test_label_image_stems_intersects_a_grouped_capture(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir, annotation_dir
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.splits import label_image_stems

    stems, stem_to_image = label_image_stems(
        annotation_dir(grouped_dataset, "2026-04-01"), image_dir(grouped_dataset, "2026-04-01"),
    )
    assert set(stems) == {"capture_001", "plain_002"}
    assert isinstance(stem_to_image["capture_001"], BandGroupRef)
    assert isinstance(stem_to_image["plain_002"], Path)


# ── annotation_tools.py ─────────────────────────────────────────────────────────────────


def test_dims_for_a_grouped_capture_path(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.annotation_tools import _dims_for

    manifest = image_dir(grouped_dataset, "2026-04-01") / "capture_001.bandgroup"
    assert _dims_for(str(manifest)) == (16, 16)


def test_focus_annotate_lands_on_the_grouped_capture_by_manifest_name(grouped_dataset):
    from tcip_mcp.tools.annotation_tools import focus

    res = focus("annotate", str(grouped_dataset), str(grouped_dataset), "catkin", "2026-04-01")
    assert "error" not in res
    assert res["n_images"] == 2
    # Sorted names: "capture_001.bandgroup" < "plain_002.jpg"
    assert res["image"] == "capture_001.bandgroup"
    assert res["image_index"] == 0


# ── vision_tools.py ─────────────────────────────────────────────────────────────────────


def test_renderable_path_materializes_a_preview_for_a_group(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import _renderable_path

    manifest = image_dir(grouped_dataset, "2026-04-01") / "capture_001.bandgroup"
    out = _renderable_path(str(manifest))
    assert out != str(manifest)
    assert Path(out).is_file()
    assert Path(out).suffix == ".png"

    from PIL import Image
    with Image.open(out) as im:
        assert im.mode == "RGB"
        assert im.size == (16, 16)


def test_renderable_path_is_unchanged_for_a_plain_photo(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import _renderable_path

    plain = image_dir(grouped_dataset, "2026-04-01") / "plain_002.jpg"
    assert _renderable_path(str(plain)) == str(plain)


def test_materialize_if_needed_unchanged_for_a_plain_3band_rgb_geotiff(tmp_path):
    """_materialize_if_needed must not route every recognized-but-non-jpg/png extension
    (including an ordinary 3-band RGB .tif, a pre-existing supported format) through a synthetic
    per-channel min-max stretch before rendering. A plain 3-band RGB GeoTIFF must render through
    the unchanged path, with PIL decoding it to the same pixel values, not a per-channel-stretched
    reinterpretation."""
    from PIL import Image

    from tcip_mcp.tools.vision_tools import _materialize_if_needed

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

    out = _materialize_if_needed(path)
    assert out == str(path)  # unchanged path, no synthetic preview materialized

    decoded = np.asarray(Image.open(path).convert("RGB"))
    assert np.array_equal(decoded, rgb)  # true colors, no per-channel stretch applied


def test_materialize_if_needed_still_stretches_a_genuinely_multiband_geotiff(tmp_path):
    """The scoping in the fix above must not swallow the real non-standard case: a >4-band
    GeoTIFF still routes through the channel-aware materialized preview."""
    from PIL import Image

    from tcip_mcp.tools.vision_tools import _materialize_if_needed

    d = tmp_path / "images"
    d.mkdir()
    arr = np.zeros((5, 10, 6), dtype=np.uint16)
    for i in range(6):
        arr[:, :, i] = (i + 1) * 100
    path = d / "multiband.tif"
    tifffile.imwrite(str(path), arr)

    out = _materialize_if_needed(path)
    assert out != str(path)
    assert Path(out).suffix == ".png"
    with Image.open(out) as im:
        assert im.mode == "RGB"


def test_visualize_annotations_on_a_grouped_capture(grouped_dataset):
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.vision_tools import visualize

    manifest = image_dir(grouped_dataset, "2026-04-01") / "capture_001.bandgroup"
    result = visualize(source="annotations", path=str(manifest))
    assert "error" not in result
    assert result["count"] == 1
    assert Path(result["image_path"]).is_file()


def test_viz_dataset_sample_folds_a_grouped_capture_into_one_entry(grouped_dataset):
    from tcip_mcp.tools.vision_tools import visualize

    result = visualize(source="dataset", path=str(grouped_dataset), n=16)
    assert "error" not in result
    assert result["total_images"] == 2  # one grouped capture + one plain photo, never 3 raw files


# ── feedback/materialize.py ──────────────────────────────────────────────────────────────


def test_materialize_dataset_copies_every_sibling_and_the_manifest(tmp_path):
    from tcip_mcp.pipelines.feedback.materialize import materialize_dataset

    src = tmp_path / "src"
    src.mkdir()
    _write_group(src, "cap")

    state = {"image": {"cap.bandgroup": {"img_status": "completed", "detections": [
        {"action": "accepted", "class_name": "catkin",
         "gt_bbox_norm": [0.5, 0.5, 0.2, 0.2], "pred_bbox_norm": None},
    ]}}}
    out = tmp_path / "out"
    result = materialize_dataset(state, str(src), str(out))

    assert result["positive"] == 1
    assert (out / "images" / "cap.bandgroup").is_file()
    assert (out / "images" / "cap_G.tif").is_file()
    assert (out / "images" / "cap_R.tif").is_file()
    assert (out / "annotations" / "cap.json").is_file()

    import json
    manifest = json.loads((out / "curated_manifest.json").read_text())
    assert manifest["images"][0]["image"] == "cap.bandgroup"

    # The output's own manifest is independently readable (band filenames resolve alongside it).
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest
    ref = read_band_group_manifest(out / "images" / "cap.bandgroup")
    assert all(p.is_file() for p in ref.bands.values())


def test_materialize_dataset_dims_from_the_grouped_capture(tmp_path):
    from tcip_mcp.pipelines.feedback.materialize import materialize_dataset

    src = tmp_path / "src"
    src.mkdir()
    _write_group(src, "cap")  # 16x16

    state = {"image": {"cap.bandgroup": {"img_status": "completed", "detections": [
        {"action": "accepted", "class_name": "catkin",
         "gt_bbox_norm": [0.5, 0.5, 0.25, 0.25], "pred_bbox_norm": None},
    ]}}}
    out = tmp_path / "out"
    materialize_dataset(state, str(src), str(out))

    from tcip_annotation import json_io
    anns = json_io.read_annotations(str(out / "annotations" / "cap.json"))
    assert len(anns) == 1
    box = anns[0].geometry
    # cx=cy=0.5, w=h=0.25 on a 16x16 frame -> [6,6,10,10]
    assert (box.x1, box.y1, box.x2, box.y2) == (6.0, 6.0, 10.0, 10.0)

"""Per-band statistics in model_source.builder_kwargs must carry a record of which images they
came from: model_source.image_stats_sampling, beside builder_kwargs rather than inside it.
preflight_config refuses a config with statistics and no provenance, and refuses provenance whose
window paths do not resolve to one of the run's own admitted sources."""

from __future__ import annotations

from pathlib import Path

import pytest

SUBJECT = "leaf"


def _cfg(images_dir, labels_dir, *, builder_kwargs, image_stats_sampling=None):
    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": builder_kwargs, "task": "detection"}
    if image_stats_sampling is not None:
        model_source["image_stats_sampling"] = image_stats_sampling
    return {
        "model_source": model_source,
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "subject": SUBJECT},
    }


def _label(labels_dir, *stems, width=16, height=16):
    """One label document per stem, so the producer admits the images these tests point at: the
    containment check reads the run's own admitted sources, not a directory listing."""
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    for stem in stems:
        json_io.write_annotations(
            labels_dir / f"{stem}.json",
            [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))], width, height,
            keep_empty=True)


def test_preflight_refuses_statistics_with_no_provenance(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 2,
                                           "image_mean": [0.1, 0.2], "image_std": [0.1, 0.1]})

    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any("image_stats_sampling" in i for i in r["issues"])


def test_preflight_refuses_a_window_path_outside_images_dir(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    Image.new("RGB", (16, 16)).save(imgs / "a.jpg")
    _label(lbls, "a")
    outside = tmp_path / "elsewhere.jpg"
    Image.new("RGB", (16, 16)).save(outside)

    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 3,
                                           "image_mean": [0.1, 0.2, 0.3],
                                           "image_std": [0.1, 0.1, 0.1]},
              image_stats_sampling={"windows": [[str(outside), None]], "seed": None,
                                    "pixel_fraction": 1.0, "window_size": None,
                                    "max_windows_per_image": None})

    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any("outside" in i for i in r["issues"])
    assert r["image_stats_containment"] == "checked"


def test_preflight_admits_a_sampling_record_naming_images_inside_images_dir(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    a = imgs / "a.jpg"
    Image.new("RGB", (16, 16)).save(a)
    _label(lbls, "a")

    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 3,
                                           "image_mean": [0.1, 0.2, 0.3],
                                           "image_std": [0.1, 0.1, 0.1]},
              image_stats_sampling={"windows": [[str(a), None]], "seed": None,
                                    "pixel_fraction": 1.0, "window_size": None,
                                    "max_windows_per_image": None})

    r = preflight_config(cfg)
    assert not any("image_stats_sampling" in i or "outside" in i for i in r["issues"]), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_records_not_checked_when_no_membership_resolved(tmp_path):
    """A config naming no locations resolves no membership, so there are no sources to check the
    window paths against; preflight says so explicitly rather than silently skipping or passing.
    The missing locations are their own named issues, beside this."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    cfg = {
        "model_source": {
            "builder": "tests.bespoke_models:build_bespoke_detection",
            "builder_kwargs": {"num_classes": 1, "in_chans": 2,
                              "image_mean": [0.1, 0.2], "image_std": [0.1, 0.1]},
            "task": "detection",
            "image_stats_sampling": {"windows": [["a.tif", None]], "seed": None,
                                     "pixel_fraction": 1.0, "window_size": None,
                                     "max_windows_per_image": None},
        },
        "data": {"dataset_source": {"builder": "tests.bespoke_models:build_bespoke_classifier"}},
    }

    r = preflight_config(cfg)
    assert r["image_stats_containment"] == "not_checked"
    assert not any("outside" in i for i in r["issues"])


def test_preflight_admits_the_exact_derivations_own_record(tmp_path):
    """derivations.image_stats_provenance renders the exact sibling's (mean, std, paths_read)
    tuple into the record preflight checks; that record must pass containment, not just a
    hand-assembled one shaped like it."""
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_mcp.pipelines.derivations import band_normalization_stats, image_stats_provenance
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    a = imgs / "a.jpg"
    Image.new("RGB", (16, 16)).save(a)
    _label(lbls, "a")

    result = band_normalization_stats([a], 3)
    assert result is not None
    mean, std, _ = result
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 3,
                                           "image_mean": mean, "image_std": std},
              image_stats_sampling=image_stats_provenance(result))

    r = preflight_config(cfg)
    assert not any("image_stats_sampling" in i or "outside" in i for i in r["issues"]), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_admits_the_sampled_derivations_own_record(tmp_path):
    """image_stats_provenance's sampled branch renders SampledNormalizationStats, carrying the
    caller's own window_size/max_windows_per_image beside the seed WindowSampling itself does not
    keep, into the same record shape the exact sibling renders."""
    pytest.importorskip("torch")
    from PIL import Image

    from tcip_mcp.pipelines.derivations import (
        band_normalization_stats_sampled, image_stats_provenance,
    )
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    a = imgs / "a.jpg"
    Image.new("RGB", (16, 16)).save(a)
    _label(lbls, "a")

    result = band_normalization_stats_sampled(
        [a], 3, seed=1, window_size=8, max_windows_per_image=4)
    assert result is not None
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 3,
                                           "image_mean": result.mean, "image_std": result.std},
              image_stats_sampling=image_stats_provenance(
                  result, window_size=8, max_windows_per_image=4))

    r = preflight_config(cfg)
    assert not any("image_stats_sampling" in i or "outside" in i for i in r["issues"]), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_admits_the_exact_derivations_record_over_a_band_group_dataset(tmp_path):
    """The exact derivation must label a band-group raster by its manifest path (the same label
    preflight's containment check resolves data.images_dir's own band groups to), not str() of the
    BandGroupRef object itself, which would never match and would refuse every band-group run."""
    pytest.importorskip("torch")
    import numpy as np
    import tifffile

    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tcip_mcp.pipelines.derivations import band_normalization_stats, image_stats_provenance
    from tcip_mcp.pipelines.image_utils import list_logical_images
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    band_a, band_b = imgs / "cap_G.tif", imgs / "cap_R.tif"
    tifffile.imwrite(str(band_a), np.full((8, 8), 111, dtype=np.uint16))
    tifffile.imwrite(str(band_b), np.full((8, 8), 222, dtype=np.uint16))
    write_band_group_manifest(imgs, "cap", {"Green": band_a, "Red": band_b})
    _label(lbls, "cap", width=8, height=8)

    ref = list_logical_images(imgs)["cap"]
    result = band_normalization_stats([ref], 2)
    assert result is not None
    mean, std, _ = result
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 2,
                                           "image_mean": mean, "image_std": std},
              image_stats_sampling=image_stats_provenance(result))

    r = preflight_config(cfg)
    assert not any("outside" in i for i in r["issues"]), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_keeps_every_sample_of_a_two_date_selection(tmp_path):
    """A selection spanning two capture dates holds two samples of one name, and a population
    keyed by that name would report a run smaller than the one that trains and one of its own
    training sources as outside it. Every sample the bound run would train on is in the
    population, so a window over any of them is contained."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.data.selection import read_selection
    from tcip_mcp.tools.training_tools import preflight_config

    from tests.test_selection_binding import DATES, SUBJECT, _two_subject_two_date_dataset

    root = _two_subject_two_date_dataset(tmp_path / "ds")
    out = tmp_path / "m"
    from tcip_mcp.tools.data_tools import draw_splits

    assert "error" not in draw_splits(
        str(root), output_path=str(out), subject=SUBJECT, seed=7,
        train_ratio=0.5, val_ratio=0.25, calibration_ratio=0.25), "the draw is the fixture"
    drawn = read_selection(out)
    bound = [s for s in drawn.samples if s.side in ("train", "val")]
    assert len({s.member_stem for s in bound}) < len(bound), (
        "the fixture must hold one member name on two dates for this to bite")
    assert {Path(s.source).parent.name for s in bound} == set(DATES)

    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": {"num_classes": 1, "in_chans": 3,
                                       "image_mean": [0.1, 0.2, 0.3],
                                       "image_std": [0.1, 0.1, 0.1]},
                    "task": "detection",
                    "image_stats_sampling": {
                        "windows": [[str(s.source), None] for s in bound], "seed": None,
                        "pixel_fraction": 1.0, "window_size": None,
                        "max_windows_per_image": None}}
    r = preflight_config({"model_source": model_source,
                          "data": {"split": {"selection_dir": str(out)}}})

    assert r["image_stats_containment"] == "checked"
    assert not any("outside" in i for i in r["issues"]), r["issues"]
    assert r["valid"] is True, r["issues"]


def test_preflight_refuses_a_hand_written_dict_naming_the_record_keys(tmp_path):
    """A record shaped by neither derivation, carrying neither 'windows' nor 'pixel_fraction', is
    refused, and the refusal names both keys rather than a generic 'missing provenance'."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 2,
                                           "image_mean": [0.1, 0.2], "image_std": [0.1, 0.1]},
              image_stats_sampling={"note": "derived by hand, not through either derivation"})

    r = preflight_config(cfg)
    assert r["valid"] is False
    assert any("'windows'" in i and "'pixel_fraction'" in i for i in r["issues"]), r["issues"]


def test_preflight_reads_the_band_count_over_every_source_the_run_admits(tmp_path):
    """The channel firewall reads the run's own band count the way the run reads it, over every
    source it admits: a second source of another band count is named here, rather than one source
    being probed and the rest trained at its count."""
    pytest.importorskip("torch")
    import numpy as np
    import tifffile
    from PIL import Image

    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    Image.new("RGB", (16, 16)).save(imgs / "a.png")
    tifffile.imwrite(str(imgs / "b.tif"), np.zeros((16, 16, 5), dtype=np.uint8))
    _label(lbls, "a", "b")
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 3})

    r = preflight_config(cfg)

    assert r["valid"] is False
    assert any("different band counts" in i for i in r["issues"]), r["issues"]


def test_preflight_admits_a_three_channel_config_with_no_statistics(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1})

    r = preflight_config(cfg)
    assert not any("image_stats_sampling" in i for i in r["issues"])
    assert "image_stats_containment" not in r

"""Per-band statistics in model_source.builder_kwargs must carry a record of which images they
came from: model_source.image_stats_sampling, beside builder_kwargs rather than inside it.
preflight_config refuses a config with statistics and no provenance, and refuses provenance whose
window paths do not resolve to files under the run's own data.images_dir."""

from __future__ import annotations

import pytest


def _cfg(images_dir, labels_dir, *, builder_kwargs, image_stats_sampling=None):
    model_source = {"builder": "tests.bespoke_models:build_bespoke_detection",
                    "builder_kwargs": builder_kwargs, "task": "detection"}
    if image_stats_sampling is not None:
        model_source["image_stats_sampling"] = image_stats_sampling
    return {
        "model_source": model_source,
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir)},
    }


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

    cfg = _cfg(imgs, lbls, builder_kwargs={"num_classes": 1, "in_chans": 3,
                                           "image_mean": [0.1, 0.2, 0.3],
                                           "image_std": [0.1, 0.1, 0.1]},
              image_stats_sampling={"windows": [[str(a), None]], "seed": None,
                                    "pixel_fraction": 1.0, "window_size": None,
                                    "max_windows_per_image": None})

    r = preflight_config(cfg)
    assert not any("image_stats_sampling" in i or "outside" in i for i in r["issues"]), r["issues"]
    assert r["image_stats_containment"] == "checked"


def test_preflight_records_not_checked_for_a_bespoke_dataset_source(tmp_path):
    """A bespoke dataset_source run legitimately has no data.images_dir to check window paths
    against; preflight says so explicitly rather than silently skipping or passing."""
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

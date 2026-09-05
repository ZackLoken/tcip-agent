"""Several raw directory walks over one bucket of ``images/`` used to stay outside
``list_logical_images``' own stem-collision refusal, so each could pick one member of a
stem-collided pair rather than raising ``AmbiguousImageStem`` the way every other reader of a
bucket does. Each is now routed through that shared enumeration instead.

``ingest_images`` itself already refuses to create a stem-collided pair (the ingest-collision
family's own rail), so the pair a test needs here is built the only way one can actually reach
disk: one real file ingested through that door, then a second, case-differing raw file added
directly into the same bucket, standing in for a dataset an external tool or a manual copy
touched after ingestion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.pipelines.image_utils import AmbiguousImageStem


def _ingested_bucket(tmp_path: Path) -> Path:
    from PIL import Image

    from tcip_mcp.tools.ingest_tools import ingest_images

    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (16, 16)).save(source / "shoot_001.jpg")

    result = ingest_images(str(source), "proj", "test-site",
                           project_path=str(tmp_path / "proj"), date_from="none")
    assert "error" not in result, result
    return Path(result["image_root"]) / "undated"


def _collide(bucket: Path) -> None:
    from PIL import Image

    Image.new("RGB", (16, 16)).save(bucket / "Shoot_001.png")


def _ingested_bucket_of(tmp_path: Path, stems: list[str]) -> Path:
    """An uncollided bucket holding one real ingested file per ``stem``, no case-differing
    collision added: the rail's admitting counterpart to :func:`_ingested_bucket` plus
    :func:`_collide`, proving a routed site still enumerates every member of a clean bucket."""
    from PIL import Image

    from tcip_mcp.tools.ingest_tools import ingest_images

    source = tmp_path / "source"
    source.mkdir()
    for stem in stems:
        Image.new("RGB", (16, 16)).save(source / f"{stem}.jpg")

    result = ingest_images(str(source), "proj", "test-site",
                           project_path=str(tmp_path / "proj"), date_from="none")
    assert "error" not in result, result
    return Path(result["image_root"]) / "undated"


def test_preflight_channel_firewall_sample_refuses_a_stem_collision(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "in_chans": 3}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(tmp_path / "labels")},
    }
    with pytest.raises(AmbiguousImageStem):
        preflight_config(cfg)


def test_preflight_channel_firewall_sample_reports_a_newer_written_bandgroup_manifest(tmp_path):
    """A ``.bandgroup`` manifest above this reader's ceiling propagates as
    ``tcip_store.SchemaVersionRefused`` out of ``list_logical_images``, uncaught by design; the
    channel firewall must report that as an issue naming the images tree rather than raising
    through the tool boundary, the same contract it holds for an ambiguous stem."""
    pytest.importorskip("torch")
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    from tcip_mcp.pipelines.data.band_groups import band_group_manifest_key
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket(tmp_path)
    key = band_group_manifest_key(bucket, "plotA")
    path = FileBackend().path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ts.RECORD_JSON.encode(
        {"schema_version": 2, "bands": {"B1": "plotA_B1.npy", "B2": "plotA_B2.npy"}}))

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "in_chans": 3}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(tmp_path / "labels")},
    }
    r = preflight_config(cfg)
    assert any("could not be read" in i and "schema_version 2" in i for i in r["issues"]), r["issues"]


def test_preflight_split_policy_stems_refuses_a_stem_collision(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(tmp_path / "labels"),
                 "split": {"group_by": "spatial_strip"}},
    }
    with pytest.raises(AmbiguousImageStem):
        preflight_config(cfg)


def test_reserve_calibration_feasibility_stems_refuses_a_stem_collision(tmp_path):
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(labels_dir),
                 "tiling": {"enabled": True},
                 "split": {"reserve_calibration_fraction": 0.2}},
    }
    with pytest.raises(AmbiguousImageStem):
        preflight_config(cfg)


def test_doctor_image_stems_refuses_a_stem_collision(tmp_path):
    from scripts.doctor import _image_stems

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    with pytest.raises(AmbiguousImageStem):
        _image_stems(project_root)


def test_scan_dataset_image_census_refuses_a_stem_collision(tmp_path):
    from tcip_mcp.tools.data_tools import _scan_dataset

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    with pytest.raises(AmbiguousImageStem):
        _scan_dataset(str(project_root))


def test_doctor_script_reports_a_stem_collision_with_no_label_file_instead_of_crashing(tmp_path):
    """``check_negatives`` and ``check_state`` both call ``_image_stems`` directly, ahead of any
    per-label read that could catch the same ambiguity another way: a collision with no label
    file for its stem at all reaches only that direct call, and must still surface as a finding,
    not a crashed subprocess."""
    import subprocess
    import sys

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    doctor = str(Path(__file__).parent.parent / "scripts" / "doctor.py")
    res = subprocess.run([sys.executable, doctor, str(project_root)],
                        capture_output=True, text=True)

    assert "Traceback" not in res.stderr, res.stderr
    assert res.returncode == 2
    assert "shoot_001.jpg" in res.stdout and "Shoot_001.png" in res.stdout


def test_doctor_script_reports_a_stem_collision_once_not_once_per_check(tmp_path):
    """``check_negatives``, ``check_data_quality`` and ``check_state`` each independently
    enumerate the same ``images/`` tree and reach the identical collision; a breeder reads about
    it once, not once per check that happens to hit it."""
    import subprocess
    import sys

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    doctor = str(Path(__file__).parent.parent / "scripts" / "doctor.py")
    res = subprocess.run([sys.executable, doctor, str(project_root)],
                        capture_output=True, text=True)

    assert "Traceback" not in res.stderr, res.stderr
    assert res.stdout.count("name more than one logical image") == 1, res.stdout


# ── The admitting case: a clean, uncollided bucket lists every image through each routed site ──


def test_preflight_channel_firewall_sample_admits_a_clean_multi_image_bucket(tmp_path):
    """The rail's own admitting counterpart: two uncollided images reach ``list_logical_images``
    without raising, and the sampled image's real channel count (3, an RGB jpg) clears the
    firewall against a matching declared in_chans."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1, "in_chans": 3}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(tmp_path / "labels")},
    }
    r = preflight_config(cfg)
    assert not any("in_chans" in i for i in r["issues"]), r["issues"]


def test_preflight_split_policy_stems_admits_a_clean_multi_image_bucket(tmp_path):
    """Two uncollided images resolve cleanly through ``resolve_group_key_fn``, proving the
    listing reached both stems rather than raising or silently dropping one."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(tmp_path / "labels"),
                 "split": {"group_by": "stem"}},
    }
    r = preflight_config(cfg)
    assert not any(i.startswith("data.split:") for i in r["issues"]), r["issues"]


def test_reserve_calibration_feasibility_admits_a_clean_multi_image_bucket(tmp_path):
    """Both uncollided images are counted: the feasibility issue names the real count (2), the
    direct evidence the listing reached every member rather than one raw-walked file."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    cfg = {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(bucket), "labels_dir": str(labels_dir),
                 "tiling": {"enabled": True},
                 "split": {"reserve_calibration_fraction": 0.2}},
    }
    r = preflight_config(cfg)
    assert any("2 source images" in i for i in r["issues"]), r["issues"]


def test_doctor_image_stems_admits_a_clean_multi_image_bucket(tmp_path):
    from scripts.doctor import _image_stems

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    project_root = bucket.parent.parent

    stems = _image_stems(project_root)
    assert stems == {"shoot_001": "shoot_001.jpg", "shoot_002": "shoot_002.jpg"}


def test_scan_dataset_image_census_admits_a_clean_multi_image_bucket(tmp_path):
    from tcip_mcp.tools.data_tools import _scan_dataset

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    project_root = bucket.parent.parent

    scan = _scan_dataset(str(project_root))
    assert {Path(p).stem for p in scan["images"]} == {"shoot_001", "shoot_002"}

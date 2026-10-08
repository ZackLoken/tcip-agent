"""Every walk over one bucket of ``images/`` routes through ``list_logical_images``' own
stem-collision refusal, raising ``AmbiguousImageStemError`` the way every other reader of a bucket
does, never picking one member of a stem-collided pair silently.

``ingest_images`` itself refuses to create a stem-collided pair, so the pair a test needs here
is built the only way one can actually reach disk: one real file ingested through that door,
then a second, case-differing raw file added directly into the same bucket, standing in for a
dataset an external tool or a manual copy touched after ingestion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.pipelines.image_utils import AmbiguousImageStemError
from tests._chain_fixtures import BESPOKE_DETECTION
from tests._cli_fixtures import run_tcip


def _ingested_bucket(tmp_path: Path) -> Path:
    from PIL import Image

    from tcip_mcp.tools.ingest_tools import ingest_images

    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (16, 16)).save(source / "shoot_001.jpg")

    result = ingest_images(tmp_path / "proj", str(source), date_from="none")
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

    result = ingest_images(tmp_path / "proj", str(source), date_from="none")
    assert "error" not in result, result
    return Path(result["image_root"]) / "undated"


SUBJECT = "leaf"


def _label(bucket: Path, *stems: str) -> None:
    """One label document per stem's image in ``bucket``, so the producer admits this bucket: a
    preflight reads the run's own admitted membership, never a directory listing of its own."""
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image

    for stem in stems:
        label_image(bucket / f"{stem}.jpg",
                    [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))], 16, 16)


def test_preflight_refuses_a_stem_collision(tmp_path):
    """The run's own resolution is the one walk every preflight leg reads, so a collided bucket
    is refused once, where that resolution runs, as an issue naming both colliding files."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    _label(bucket, "shoot_001")

    cfg = {
        "model_source": {"builder": BESPOKE_DETECTION,
                         "task": "detection"},
        "data": {"images_dir": str(bucket), "scope": {"subject": SUBJECT}},
    }
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any("shoot_001.jpg" in i and "Shoot_001.png" in i for i in r["issues"]), r["issues"]


def test_scan_dataset_image_census_refuses_a_stem_collision(tmp_path):
    from tcip_mcp.tools.data_tools import _scan_dataset

    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    with pytest.raises(AmbiguousImageStemError):
        _scan_dataset(str(project_root))


def test_doctor_script_reports_a_stem_collision_with_no_label_file_instead_of_crashing(tmp_path):
    """A collision with no label record for its stem surfaces as a finding of the doctor's one
    census, not a crashed subprocess."""
    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    res = run_tcip("doctor", [str(project_root)])

    assert "Traceback" not in res.stderr, res.stderr
    assert res.returncode == 2
    assert "shoot_001.jpg" in res.stdout and "Shoot_001.png" in res.stdout


def test_doctor_script_reports_a_stem_collision_once_not_once_per_check(tmp_path):
    """The doctor enumerates ``images/`` once, so a breeder reads about a collision once."""
    bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    project_root = bucket.parent.parent

    res = run_tcip("doctor", [str(project_root)])

    assert "Traceback" not in res.stderr, res.stderr
    assert res.stdout.count("name more than one logical image") == 1, res.stdout


# ── The admitting case: a clean, uncollided bucket lists every image through each routed site ──


def test_preflight_admits_a_clean_multi_image_bucket(tmp_path):
    """The rail's own admitting counterpart: two uncollided images reach ``list_logical_images``
    without raising, and preflight reads the run's sizes off them with nothing to object to."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    _label(bucket, "shoot_001", "shoot_002")

    cfg = {
        "model_source": {"builder": BESPOKE_DETECTION,
                         "task": "detection"},
        "data": {"images_dir": str(bucket), "scope": {"subject": SUBJECT},
                 "split": {"seed": 0, "val_ratio": 0.15}},
    }
    r = preflight_config(tmp_path, cfg)
    assert r["issues"] == [], r["issues"]


def test_preflight_split_policy_stems_admits_a_clean_multi_image_bucket(tmp_path):
    """Two uncollided images resolve cleanly through ``resolve_group_key_fn``, proving the
    listing reached both stems rather than raising or silently dropping one."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    _label(bucket, "shoot_001", "shoot_002")

    cfg = {
        "model_source": {"builder": BESPOKE_DETECTION,
                         "task": "detection"},
        "data": {"images_dir": str(bucket), "scope": {"subject": SUBJECT},
                 "split": {"group_by": "stem", "seed": 0, "val_ratio": 0.15}},
    }
    r = preflight_config(tmp_path, cfg)
    assert not any(i.startswith("data.split:") for i in r["issues"]), r["issues"]


def test_calibration_ratio_feasibility_admits_a_clean_multi_image_bucket(tmp_path):
    """Both uncollided images are counted: the feasibility issue names the real count (2), the
    direct evidence the admission reached every member rather than one raw-walked file."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    _label(bucket, "shoot_001", "shoot_002")

    cfg = {
        "model_source": {"builder": BESPOKE_DETECTION,
                         "task": "detection"},
        "data": {"images_dir": str(bucket), "scope": {"subject": SUBJECT},
                 "tiling": {"enabled": True, "sliver_frac": 0.5},  # stated: two boxes, no spread
                 "split": {"calibration_ratio": 0.2, "val_ratio": 0.15, "seed": 1}},
    }
    r = preflight_config(tmp_path, cfg)
    assert any("2 admitted sources" in i for i in r["issues"]), r["issues"]


def test_scan_dataset_image_census_admits_a_clean_multi_image_bucket(tmp_path):
    from tcip_mcp.tools.data_tools import _scan_dataset

    bucket = _ingested_bucket_of(tmp_path, ["shoot_001", "shoot_002"])
    project_root = bucket.parent.parent

    scan = _scan_dataset(str(project_root))
    assert {Path(p).stem for p in scan["images"]} == {"shoot_001", "shoot_002"}

"""Readers of one bucket of ``images/`` refuse a stem-collided pair, naming both files.

Each collided pair is one real file ingested through ``ingest_images`` and a second,
case-differing raw file added directly into that bucket.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tcip_mcp.pipelines.image_utils import AmbiguousImageStemError
from tests._chain_fixtures import (
    DETECTION_SOURCE,
    training_config,
    weights_statistics,
)
from tests._cli_fixtures import run_tcip
from tests._producer_fixtures import write_image
from tests._training_values import drawn_split, tiling_block


def _ingested_bucket(tmp_path: Path) -> tuple[Path, Path]:
    """Ingest one 16 px JPEG ``shoot_001`` into the project ``proj``.

    Returns the project and its undated capture.
    """
    from tcip_mcp.tools.ingest_tools import ingest_images

    source = tmp_path / "source"
    write_image(source / "shoot_001.jpg", (16, 16))

    project = tmp_path / "proj"
    result = ingest_images(project, str(source), actor=None)
    assert "error" not in result, result
    return project, image_dir(project, UNDATED_BUCKET)


def _collide(bucket: Path) -> None:
    write_image(bucket / "Shoot_001.png", (16, 16))


def _ingested_bucket_of(tmp_path: Path, stems: list[str]) -> tuple[Path, Path]:
    """Ingest one 16 px JPEG per ``stem`` into the project ``proj`` through ``ingest_images``.

    Returns the project and its undated capture.
    """
    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._producer_fixtures import BRIGHT, painted_frame

    source = tmp_path / "source"
    source.mkdir()
    for stem in stems:
        painted_frame(16, 16, (0, 0, 0), [((2, 2, 8, 8), BRIGHT)]).save(
            source / f"{stem}.jpg"
        )

    project = tmp_path / "proj"
    result = ingest_images(project, str(source), actor=None)
    assert "error" not in result, result
    return project, image_dir(project, UNDATED_BUCKET)


SUBJECT = "leaf"


def _label(bucket: Path, *stems: str) -> None:
    """Write, for each stem's image in ``bucket``, a label document holding one ``leaf`` box."""
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    for stem in stems:
        label_image(
            bucket / f"{stem}.jpg",
            [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 5, 5))],
            16,
            16,
        )


def test_preflight_refuses_a_stem_collision(tmp_path: Path) -> None:
    """Preflight refuses a stem-collided bucket with an issue naming both colliding files."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    _project, bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    _label(bucket, "shoot_001")

    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {"images_dir": str(bucket), "scope": {"subject": SUBJECT}},
    )
    r = preflight_config(tmp_path, cfg)
    assert r["valid"] is False
    assert any(
        "shoot_001.jpg" in i and "Shoot_001.png" in i for i in r["issues"]
    ), r["issues"]


def test_scan_dataset_image_census_refuses_a_stem_collision(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.data_tools import _scan_dataset

    project_root, bucket = _ingested_bucket(tmp_path)
    _collide(bucket)

    with pytest.raises(AmbiguousImageStemError):
        _scan_dataset(str(project_root))


def test_the_image_count_counts_both_images_of_a_stem_the_census_refuses(
    tmp_path: Path,
) -> None:
    """The image count counts each of two images of one stem; the census over them refuses."""
    from tcip_mcp.pipelines.image_utils import logical_image_counts
    from tcip_mcp.tools.data_tools import _scan_dataset

    project_root, bucket = _ingested_bucket(tmp_path)
    write_image(bucket / "shoot_001.png", (16, 16))

    assert logical_image_counts(project_root) == {UNDATED_BUCKET: 2}
    with pytest.raises(AmbiguousImageStemError):
        _scan_dataset(str(project_root))


def _diagnosed_collision(tmp_path: Path) -> list[list[str]]:
    """Return the findings ``tcip diagnose-project`` prints over a project holding a collision.

    The project is made through the platform's creation door over an ingested bucket holding a
    stem-collided pair and no label file; the command is asserted to answer with no traceback.
    """
    import json

    from tests._web_fixtures import new_project

    project_root, bucket = _ingested_bucket(tmp_path)
    _collide(bucket)
    new_project(project_root)

    res = run_tcip(
        "diagnose-project",
        ["--project", str(project_root), "--user", "tester"],
    )

    assert "Traceback" not in res.stderr, res.stderr
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)["findings"]


def test_doctor_command_reports_a_stem_collision_with_no_label_file_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """The doctor's command reports a stem collision with no label file as an error finding.

    The finding names both colliding files.
    """
    found = _diagnosed_collision(tmp_path)

    assert [
        level
        for level, msg in found
        if "shoot_001.jpg" in msg and "Shoot_001.png" in msg
    ] == ["error"], found


def test_doctor_command_reports_a_stem_collision_once_not_once_per_check(
    tmp_path: Path,
) -> None:
    """The doctor's command reports a stem collision once."""
    found = _diagnosed_collision(tmp_path)

    collisions = [
        msg for _, msg in found if "name more than one logical image" in msg
    ]
    assert len(collisions) == 1, found


# ── The admitting case: a clean, uncollided bucket lists every image through each routed site ──


def test_preflight_admits_a_clean_multi_image_bucket(tmp_path: Path) -> None:
    """Preflight admits a clean bucket of two uncollided images with no issue."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    _project, bucket = _ingested_bucket_of(
        tmp_path, ["shoot_001", "shoot_002"]
    )
    _label(bucket, "shoot_001", "shoot_002")

    cfg = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {
            "images_dir": str(bucket),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )
    r = preflight_config(tmp_path, cfg)
    assert r["issues"] == [], r["issues"]


def test_preflight_split_policy_stems_admits_a_clean_multi_image_bucket(
    tmp_path: Path,
) -> None:
    """A split policy over a clean two-image bucket resolves with no split issue."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    _project, bucket = _ingested_bucket_of(
        tmp_path, ["shoot_001", "shoot_002"]
    )
    _label(bucket, "shoot_001", "shoot_002")

    cfg = training_config(
        tmp_path,
        DETECTION_SOURCE,
        {
            "images_dir": str(bucket),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(val_ratio=0.15),
        },
    )
    r = preflight_config(tmp_path, cfg)
    assert not any(i.startswith("data.split:") for i in r["issues"]), r[
        "issues"
    ]


def test_calibration_ratio_feasibility_admits_a_clean_multi_image_bucket(
    tmp_path: Path,
) -> None:
    """Calibration feasibility over a clean bucket names a count of two admitted sources."""
    pytest.importorskip("torch")
    from tcip_mcp.tools.training_tools import preflight_config

    _project, bucket = _ingested_bucket_of(
        tmp_path, ["shoot_001", "shoot_002"]
    )
    _label(bucket, "shoot_001", "shoot_002")

    cfg = training_config(
        tmp_path,
        weights_statistics(DETECTION_SOURCE),
        {
            "images_dir": str(bucket),
            "scope": {"subject": SUBJECT},
            "tiling": tiling_block(
                sliver_frac=0.5
            ),  # stated: two boxes, no spread
            "split": drawn_split(
                calibration_ratio=0.2,
                val_ratio=0.15,
                seed=1,
                grouping="tile_prefix",
            ),
        },
    )
    r = preflight_config(tmp_path, cfg)
    assert any("2 admitted sources" in i for i in r["issues"]), r["issues"]


def test_scan_dataset_image_census_admits_a_clean_multi_image_bucket(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.data_tools import _scan_dataset

    project_root, _bucket = _ingested_bucket_of(
        tmp_path, ["shoot_001", "shoot_002"]
    )

    scan = _scan_dataset(str(project_root))
    assert {Path(p).stem for p in scan["images"]} == {"shoot_001", "shoot_002"}

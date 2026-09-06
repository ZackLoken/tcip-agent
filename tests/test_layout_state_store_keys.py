"""The dataset-native state stores: the bucket key the writer composes and the published inverse
takes apart, and the separation between the advisory view-coverage store and the
confirmed-negatives store that decides what trains as empty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tcip_store as ts
from tcip_mcp.dataset_layout import (
    bucket_subject_date,
    image_status_key,
    normalize_status_store,
    status_bucket,
    status_records,
    view_coverage_key,
)
from tcip_web.app import app

DOCTOR_PATH = (Path(__file__).parent.parent / "packages" / "tcip-mcp" / "src" / "tcip_mcp"
               / "cli" / "doctor.py")


def _doctor():
    """The data-state doctor loaded as a module, for its own inverse of the bucket key."""
    from tcip_mcp.cli import doctor

    return doctor


@pytest.mark.parametrize(
    ("subject", "date"),
    [("bud", "2026-02-11"), ("bush", "2026-03-02"), ("leaf", None)],
)
def test_the_doctor_reads_a_bucket_key_through_the_published_inverse(
    subject: str, date: str | None
) -> None:
    """The doctor takes a bucket key apart with the inverse published beside the composer, so the
    two cannot disagree; a disagreement attributes one subject's confirmations to another."""
    assert not hasattr(_doctor(), "_bucket_subject_date"), (
        "the doctor is splitting a bucket key itself again instead of calling the published inverse"
    )
    assert "bucket_subject_date" in DOCTOR_PATH.read_text(encoding="utf-8")
    assert bucket_subject_date(status_bucket(subject, date)) == (subject, date)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def dated_dataset(tmp_path: Path) -> tuple[Path, str]:
    """A dataset with one non-square image under a date bucket."""
    img_dir = tmp_path / "ds" / "images" / "2026-03-01"
    img_dir.mkdir(parents=True)
    path = img_dir / "plot.tif"
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(path)
    return tmp_path / "ds", str(path)


def test_recording_view_coverage_leaves_the_confirmed_negatives_untouched(
    client: TestClient, dated_dataset: tuple[Path, str]
) -> None:
    """View coverage is its own store: recording it can never rewrite a human confirmation.

    Both stores are keyed by subject and date, so a coverage record landing in the negatives store
    would overwrite the confirmation for that image and silently drop it from training.
    """
    root, path = dated_dataset
    bucket = status_bucket("bush", "2026-03-01")
    ts.replace(image_status_key(root), {bucket: status_records(
        {"plot.tif": "negative", "other.tif": "complete"}, recorded_by="user:breeder")},
        expect=ts.Version.ABSENT)

    grid = client.get(
        "/api/coverage/grid", params={"path": path, "tile_size": 50}).json()["grid"]
    cell = grid["cells"][0]["name"]
    resp = client.post("/api/coverage", json={
        "image_path": path,
        "subject": "bush",
        "date": "2026-03-01",
        "grid": {k: grid[k] for k in ("width", "height", "tile_size", "overlap", "cols", "rows")},
        "cells_served_at_native": [cell],
        "cells_seen_at_scale": {},
        "viewing": {"stretch": "minmax", "stats_source": {"read": "none"},
                   "display_bounds": None, "base_served_size": None},
    })
    assert resp.status_code == 200, resp.text

    stored = normalize_status_store(ts.read(image_status_key(root)))
    assert stored.get(bucket, {}).get("plot.tif") == "negative"
    assert stored.get(bucket, {}).get("other.tif") == "complete"

    coverage = ts.read(view_coverage_key(root))
    record = coverage[bucket]["plot.tif"]
    assert record["cells_served_at_native"] == [cell]
    assert record["cells_seen_at_scale"] == {}
    assert record["grid"]["cols"] * record["grid"]["rows"] == len(grid["cells"])

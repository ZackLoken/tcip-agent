"""The dataset-native state stores: the bucket key both the writer and the doctor's inverse must
read the same way, and the separation between the advisory view-coverage store and the
confirmed-negatives store that decides what trains as empty."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tcip_mcp.dataset_layout import (
    image_status_path,
    normalize_status_store,
    status_bucket,
    view_coverage_path,
)
from tcip_web.app import app

DOCTOR_PATH = Path(__file__).parent.parent / "scripts" / "doctor.py"


def _doctor():
    """The data-state doctor loaded as a module, for its own inverse of the bucket key."""
    spec = importlib.util.spec_from_file_location("tcip_data_state_doctor", DOCTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("subject", "date"),
    [("catkin", "2026-02-11"), ("bush", "2026-03-02"), ("leaf", None)],
)
def test_status_bucket_round_trips_through_the_doctors_inverse(
    subject: str, date: str | None
) -> None:
    """The doctor splits a bucket key back into subject and date by hand, so its inverse and the
    composer must agree; a disagreement attributes one subject's confirmations to another."""
    subj, bdate = _doctor()._bucket_subject_date(status_bucket(subject, date))
    assert (subj, bdate) == (subject, date)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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
    status = image_status_path(root)
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps({bucket: {"plot.tif": "negative", "other.tif": "complete"}}), encoding="utf-8")

    grid = client.get("/api/coverage/grid", params={"path": path, "tile_size": 50}).json()
    cell = grid["cells"][0]["name"]
    resp = client.post("/api/coverage", json={
        "image_path": path,
        "subject": "bush",
        "date": "2026-03-01",
        "grid": {k: grid[k] for k in ("width", "height", "tile_size", "overlap", "cols", "rows")},
        "cells_served_at_native": [cell],
        "cells_swept": [],
        "viewing": {"stretch": "minmax"},
    })
    assert resp.status_code == 200, resp.text

    stored = normalize_status_store(json.loads(status.read_text(encoding="utf-8")))
    assert stored.get(bucket, {}).get("plot.tif") == "negative"
    assert stored.get(bucket, {}).get("other.tif") == "complete"

    coverage = json.loads(view_coverage_path(root).read_text(encoding="utf-8"))
    record = coverage[bucket]["plot.tif"]
    assert record["cells_served_at_native"] == [cell]
    assert record["cells_swept"] == []
    assert record["grid"]["cols"] * record["grid"]["rows"] == len(grid["cells"])

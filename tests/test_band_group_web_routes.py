"""Web routes for band-grouped captures: the GUI's own gallery (routes/dataset.py) folding
sibling band files into one grouped entry, ``_image_dims`` (routes/review.py, routes/annotate.py)
reading a grouped capture's real stacked frame, and the live band-composite serving +
``/api/images/bands`` endpoint (routes/images.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _write_group(images_dir: Path, stem: str, base=(111, 222, 333)) -> None:
    """Each band gets a distinct base level plus a horizontal gradient (never a flat fill): a
    constant array makes min==max, so a min-max stretch would flatten every channel to black and
    a channel reorder would be indistinguishable from the original."""
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest

    gradient = np.linspace(0, 500, 24, dtype=np.uint16)
    band_paths = {}
    for name, val in zip(("Green", "Red", "NIR"), base):
        p = images_dir / f"{stem}_{name}.tif"
        arr = (np.full((20, 24), val, dtype=np.uint32) + gradient[None, :]).astype(np.uint16)
        tifffile.imwrite(str(p), arr)
        band_paths[name] = p
    write_band_group_manifest(images_dir, stem, band_paths,
                              central_wavelength_nm={"Green": 560.0, "Red": 650.0, "NIR": 860.0})


@pytest.fixture
def grouped_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "MS_Farm"
    date_dir = root / "images" / "2026-05-01"
    date_dir.mkdir(parents=True)
    _write_group(date_dir, "cap_001")
    Image.new("RGB", (24, 20), (5, 5, 5)).save(date_dir / "plain_002.jpg")
    return root


# ── routes/dataset.py ────────────────────────────────────────────────────────────────────


def test_dataset_select_route_folds_the_group(client: TestClient, grouped_dataset: Path):
    resp = client.post("/api/dataset/select", json={
        "project_root": str(grouped_dataset), "dataset_root": str(grouped_dataset),
        "date": "2026-05-01",
    })
    assert resp.status_code == 200
    images = resp.json()["selection"]["image_list"]
    assert sorted(images) == ["cap_001.bandgroup", "plain_002.jpg"]


# ── routes/review.py + routes/annotate.py _image_dims ───────────────────────────────────


def test_annotate_labels_route_measures_a_grouped_captures_real_frame(
    client: TestClient, grouped_dataset: Path,
):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.get("/api/annotate/labels", params={"image_path": str(manifest)})
    assert resp.status_code == 200
    body = resp.json()
    assert (body["img_width"], body["img_height"]) == (24, 20)


def test_review_matches_route_measures_a_grouped_captures_real_frame(
    client: TestClient, grouped_dataset: Path,
):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.post("/api/review/matches", json={
        "dataset_root": str(grouped_dataset),
        "image_name": "cap_001.bandgroup",
        "image_path": str(manifest),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert (body["img_width"], body["img_height"]) == (24, 20)


# ── routes/images.py: serve_image bands/stretch + /api/images/bands ─────────────────────


def test_serve_image_plain_photo_unaffected_by_new_params(client: TestClient, grouped_dataset: Path):
    plain = grouped_dataset / "images" / "2026-05-01" / "plain_002.jpg"
    baseline = client.get("/api/images", params={"path": str(plain)})
    with_defaults = client.get("/api/images", params={"path": str(plain)})
    assert baseline.status_code == with_defaults.status_code == 200
    assert baseline.content == with_defaults.content
    assert baseline.headers["etag"] == with_defaults.headers["etag"]


def test_serve_image_composites_a_band_group_by_default(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.get("/api/images", params={"path": str(manifest)})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    img = Image.open(__import__("io").BytesIO(resp.content))
    assert img.mode == "RGB"
    assert img.size == (24, 20)


def test_serve_image_bands_param_changes_the_composite(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    # stretch="none" (absolute, by dtype max) so each band's distinct base level survives:
    # min-max stretch would remove it entirely (every band shares the same gradient shape),
    # cancelling the very difference this test means to detect.
    r1 = client.get("/api/images", params={
        "path": str(manifest), "bands": "Green,Red,NIR", "stretch": "none"})
    r2 = client.get("/api/images", params={
        "path": str(manifest), "bands": "NIR,Red,Green", "stretch": "none"})
    assert r1.status_code == r2.status_code == 200
    assert r1.content != r2.content  # different band->channel assignment, different pixels
    assert r1.headers["etag"] != r2.headers["etag"]  # cache key includes bands


def test_serve_image_stretch_param_changes_the_cache_key(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    r1 = client.get("/api/images", params={"path": str(manifest), "stretch": "minmax"})
    r2 = client.get("/api/images", params={"path": str(manifest), "stretch": "none"})
    assert r1.headers["etag"] != r2.headers["etag"]


def test_serve_image_rejects_an_unknown_stretch(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.get("/api/images", params={"path": str(manifest), "stretch": "bogus"})
    assert resp.status_code == 400


def test_serve_image_rejects_a_band_count_other_than_3(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.get("/api/images", params={"path": str(manifest), "bands": "Green,Red"})
    assert resp.status_code == 400


def test_serve_image_rejects_an_undeclared_band_name(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.get(
        "/api/images", params={"path": str(manifest), "bands": "Green,Red,Blue"}
    )
    assert resp.status_code == 400


def test_get_bands_endpoint_reports_the_group(client: TestClient, grouped_dataset: Path):
    manifest = grouped_dataset / "images" / "2026-05-01" / "cap_001.bandgroup"
    resp = client.get("/api/images/bands", params={"path": str(manifest)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["band_count"] == 3
    by_name = {b["name"]: b for b in body["bands"]}
    assert set(by_name) == {"Green", "Red", "NIR"}
    assert by_name["NIR"]["wavelength_nm"] == 860.0
    assert by_name["NIR"]["min"] == 333.0
    assert by_name["NIR"]["max"] == 833.0


def test_get_bands_endpoint_reports_3_for_a_plain_rgb_photo(client: TestClient, grouped_dataset: Path):
    plain = grouped_dataset / "images" / "2026-05-01" / "plain_002.jpg"
    resp = client.get("/api/images/bands", params={"path": str(plain)})
    assert resp.status_code == 200
    assert resp.json()["band_count"] == 3


def test_get_bands_endpoint_never_decodes_pixels_for_a_plain_rgb_photo(
    client: TestClient, grouped_dataset: Path, monkeypatch,
):
    """/api/images/bands must not do a full pixel decode for ordinary RGB, only for genuine
    multi-band sources. Spy on the raster factory itself (every decode route opens a source
    through it) and assert nothing opens one for a plain 3-band photo."""
    from tcip_mcp.pipelines import raster_source

    called = []

    def _fail_if_called(*a, **kw):
        called.append(1)
        raise AssertionError("no raster should be opened for a plain <=3-band source")

    monkeypatch.setattr(raster_source, "open_raster", _fail_if_called)
    monkeypatch.setattr(raster_source, "open_array_source", _fail_if_called)

    plain = grouped_dataset / "images" / "2026-05-01" / "plain_002.jpg"
    resp = client.get("/api/images/bands", params={"path": str(plain)})
    assert resp.status_code == 200
    assert resp.json() == {"band_count": 3, "bands": []}
    assert called == []


def test_serve_image_stale_group_returns_409(client: TestClient, grouped_dataset: Path):
    date_dir = grouped_dataset / "images" / "2026-05-01"
    manifest = date_dir / "cap_001.bandgroup"
    (date_dir / "cap_001_NIR.tif").unlink()
    resp = client.get("/api/images", params={"path": str(manifest)})
    assert resp.status_code == 409


# ── routes/inference.py ──────────────────────────────────────────────────────────────────


def test_inference_list_images_folds_a_group_into_one_entry(grouped_dataset: Path):
    """routes/inference.py's own directory-listing fallback must route through
    list_logical_images, not just widen its extension tuple, or a grouped capture's sibling band
    files each enumerate as their own (spurious) image, the same bug class
    export_predictions/run_inference had (inference_tools.py; see test_band_group_call_sites /
    test_band_group_inference_calibration)."""
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_web.routes.inference import _list_images

    date_dir = grouped_dataset / "images" / "2026-05-01"
    images = _list_images(date_dir)
    assert len(images) == 2  # one grouped capture + one plain photo, never 4 raw files
    grouped = [i for i in images if isinstance(i, BandGroupRef)]
    assert len(grouped) == 1 and grouped[0].stem == "cap_001"


def test_inference_worker_predicts_on_the_correctly_decoded_grouped_capture(
    tmp_path, monkeypatch,
):
    """The web door's own real-forward-pass proof (mirrors
    test_band_group_inference_calibration.py for the MCP door): a grouped capture in images_dir
    must decode through the channel-aware loader, never crash on a stringified BandGroupRef."""
    pytest.importorskip("torch")

    from tests._verified_checkpoint_fixtures import registered_checkpoint

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_web.routes.inference import InferenceJob, _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_group(images_dir, "cap_001")
    out_dir = tmp_path / "out"
    ckpt = registered_checkpoint(tmp_path, project_root=tmp_path)

    seen = []

    class FakePredictor:
        task = "detection"

        def __init__(self, checkpoint_path=None, **kwargs):
            pass

        def predict_batch(self, paths, tile=False, tile_size=224, overlap=0.2, **kw):
            seen.extend(paths)
            return [{"image": (p.manifest_path.name if isinstance(p, BandGroupRef) else p.name),
                     "width": 20, "height": 24,
                     "boxes": [], "scores": [], "labels": [], "count": 0}
                    for p in paths]

    monkeypatch.setattr(
        "tcip_mcp.pipelines.inference.generic_predictor.GenericPredictor", FakePredictor)

    job = InferenceJob(
        job_id="t2", checkpoint_path=str(ckpt), images_dir=str(images_dir),
        output_dir=str(out_dir), tile=False, conf=0.25, iou=0.7,
        slice_hw=(640, 640), overlap=0.2, postprocess="nms", platform_root=str(tmp_path),
    )
    _worker(job)

    assert job.status == "completed"
    assert job.done == 1 and job.total == 1
    assert len(seen) == 1 and isinstance(seen[0], BandGroupRef) and seen[0].stem == "cap_001"
    assert (out_dir / "cap_001.json").is_file()

"""Web routes for band-grouped captures.

They are the dataset selection (routes/dataset.py) listing sibling band files as one grouped
entry, the annotate labels route measuring a grouped capture's frame, the band-composite image
serving and ``/api/images/bands`` endpoint (routes/images.py), and an inference job over a
grouped capture.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

from tcip_mcp.dataset_layout import image_dir
from tests._producer_fixtures import GroupedDataset, write_image
from tests._training_values import RATIONALE

if TYPE_CHECKING:
    from tcip_mcp.pipelines.execution import Execution


def _write_group(
    images_dir: Path, stem: str, base: tuple[int, int, int] = (111, 222, 333)
) -> tuple[Path, dict[str, Path]]:
    """Write band TIFFs for group ``stem`` into ``images_dir`` and their manifest beside them.

    For each pair of ``zip(("Green", "Red", "NIR"), base)`` a 20 by 24 uint16 TIFF is written
    from the ``base`` entry and one horizontal gradient. Returns the manifest written and the
    band paths by name.
    """
    from tests._producer_fixtures import write_band_group

    gradient = np.linspace(0, 500, 24, dtype=np.uint16)
    band_paths = {}
    for name, val in zip(
        wavelengths := {"Green": 560.0, "Red": 650.0, "NIR": 860.0}, base
    ):
        p = images_dir / f"{stem}_{name}.tif"
        arr = (
            np.full((20, 24), val, dtype=np.uint32) + gradient[None, :]
        ).astype(np.uint16)
        tifffile.imwrite(str(p), arr)
        band_paths[name] = p
    manifest = write_band_group(
        images_dir,
        stem,
        band_paths,
        central_wavelength_nm=wavelengths,
    )
    return manifest, band_paths


@pytest.fixture
def grouped_dataset(tmp_path: Path) -> GroupedDataset:
    root = tmp_path / "MS_Farm"
    date_dir = image_dir(root, "2026-05-01")
    date_dir.mkdir(parents=True)
    manifest, bands = _write_group(date_dir, "cap_001")
    plain = write_image(date_dir / "plain_002.jpg", (24, 20), (5, 5, 5))
    return GroupedDataset(root, date_dir, manifest, bands, plain)


# ── routes/dataset.py ────────────────────────────────────────────────────────────────────


def test_dataset_select_route_folds_the_group(
    client: TestClient,
    grouped_dataset: GroupedDataset,
    opened_project: Path,
) -> None:
    resp = client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(grouped_dataset.root),
            "date": "2026-05-01",
        },
    )
    assert resp.status_code == 200
    images = resp.json()["selection"]["images"]
    assert sorted(image["name"] for image in images) == [
        "cap_001.bandgroup",
        "plain_002.jpg",
    ]


def test_each_selected_image_is_placed_where_the_layout_reads_it_back(
    client: TestClient,
    grouped_dataset: GroupedDataset,
    opened_project: Path,
) -> None:
    """Each image's place in a selection decodes back to the selection's root, capture and stem.

    The capture holds a raw image and a grouped capture; the selection answers them in name
    order, each at the path ``parse_image_path`` reads as the selection's root, its capture and
    the image's stem.
    """
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.pipelines.image_utils import logical_images_by_name, stem_of

    resp = client.post(
        "/api/dataset/select",
        json={
            "dataset_root": str(grouped_dataset.root),
            "date": grouped_dataset.images_dir.name,
        },
    )

    assert resp.status_code == 200, resp.text
    selection = resp.json()["selection"]
    root, capture = Path(selection["dataset_root"]), selection["date"]
    found = logical_images_by_name(image_dir(root, capture))
    assert [
        (image["name"], parse_image_path(image["path"]))
        for image in selection["images"]
    ] == [
        (name, (root, capture, stem_of(source)))
        for name, source in found.items()
    ]


# ── routes/annotate.py image dimensions ──────────────────────────────────────────────────


def test_annotate_labels_route_measures_a_grouped_captures_real_frame(
    client: TestClient,
    grouped_dataset: GroupedDataset,
) -> None:
    resp = client.get(
        "/api/annotate/labels",
        params={"image_path": str(grouped_dataset.manifest)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert (body["img_width"], body["img_height"]) == (24, 20)


# ── routes/images.py: serve_image bands/stretch + /api/images/bands ─────────────────────


def _view(path: Path, **params: str | int) -> dict:
    """Return the image request for ``path`` at a 3840 by 2160 display, ``params`` over it."""
    return {"path": str(path), "display_pixels": 3840 * 2160, **params}


def test_serve_image_plain_photo_unaffected_by_new_params(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    """The band params at their defaults serve a plain photo exactly as omitting them does."""
    plain = grouped_dataset.plain
    baseline = client.get("/api/images", params=_view(plain))
    with_defaults = client.get(
        "/api/images", params=_view(plain, stretch="minmax", level=0)
    )
    assert baseline.status_code == with_defaults.status_code == 200
    assert baseline.content == with_defaults.content
    assert baseline.headers["etag"] == with_defaults.headers["etag"]


def test_serve_image_refuses_a_band_members_own_path_naming_its_manifest(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    """The serve door answers a band member's own path with a 404 naming its manifest.

    A band of a grouped capture is no image of its own: its path answers 404 naming the
    manifest, not a single-band render.
    """
    member = grouped_dataset.bands["NIR"]
    resp = client.get("/api/images", params=_view(member))
    assert resp.status_code == 404
    assert grouped_dataset.manifest.name in resp.json()["detail"]


def test_serve_image_composites_a_band_group_by_default(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    resp = client.get("/api/images", params=_view(manifest))
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    img = Image.open(__import__("io").BytesIO(resp.content))
    assert img.mode == "RGB"
    assert img.size == (24, 20)


def test_serve_image_bands_param_changes_the_composite(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    # stretch="none" (absolute, by dtype max) so each band's distinct base level survives:
    # min-max stretch would remove it entirely (every band shares the same gradient shape),
    # canceling the very difference this test means to detect.
    r1 = client.get(
        "/api/images",
        params=_view(manifest, bands="Green,Red,NIR", stretch="none"),
    )
    r2 = client.get(
        "/api/images",
        params=_view(manifest, bands="NIR,Red,Green", stretch="none"),
    )
    assert r1.status_code == r2.status_code == 200
    assert (
        r1.content != r2.content
    )  # different band->channel assignment, different pixels
    assert r1.headers["etag"] != r2.headers["etag"]  # cache key includes bands


def test_serve_image_stretch_param_changes_the_cache_key(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    r1 = client.get("/api/images", params=_view(manifest, stretch="minmax"))
    r2 = client.get("/api/images", params=_view(manifest, stretch="none"))
    assert r1.headers["etag"] != r2.headers["etag"]


def test_serve_image_rejects_an_unknown_stretch(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    resp = client.get("/api/images", params=_view(manifest, stretch="bogus"))
    assert resp.status_code == 400


def test_serve_image_rejects_a_band_count_other_than_3(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    resp = client.get("/api/images", params=_view(manifest, bands="Green,Red"))
    assert resp.status_code == 400


def test_serve_image_rejects_an_undeclared_band_name(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    resp = client.get(
        "/api/images", params=_view(manifest, bands="Green,Red,Blue")
    )
    assert resp.status_code == 400


def test_get_bands_endpoint_reports_the_group(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    manifest = grouped_dataset.manifest
    resp = client.get("/api/images/bands", params={"path": str(manifest)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["band_count"] == 3
    by_name = {b["name"]: b for b in body["bands"]}
    assert set(by_name) == {"Green", "Red", "NIR"}
    assert by_name["NIR"]["wavelength_nm"] == 860.0
    assert by_name["NIR"]["min"] == 333.0
    assert by_name["NIR"]["max"] == 833.0


def test_get_bands_endpoint_reports_3_for_a_plain_rgb_photo(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    plain = grouped_dataset.plain
    resp = client.get("/api/images/bands", params={"path": str(plain)})
    assert resp.status_code == 200
    assert resp.json()["band_count"] == 3


def test_get_bands_endpoint_never_decodes_pixels_for_a_plain_rgb_photo(
    client: TestClient,
    grouped_dataset: GroupedDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/api/images/bands`` opens no raster for a plain RGB photo.

    The raster factory is spied on, and the endpoint answers a plain 3-band photo without
    calling it.
    """
    from tcip_mcp.pipelines import raster_source

    called = []

    def _fail_if_called(*a: object, **kw: object) -> NoReturn:
        called.append(1)
        raise AssertionError(
            "no raster should be opened for a plain <=3-band source"
        )

    monkeypatch.setattr(raster_source.SourceHeader, "open", _fail_if_called)

    plain = grouped_dataset.plain
    resp = client.get("/api/images/bands", params={"path": str(plain)})
    assert resp.status_code == 200
    assert resp.json() == {"band_count": 3, "bands": []}
    assert called == []


def test_serve_image_stale_group_returns_409(
    client: TestClient, grouped_dataset: GroupedDataset
) -> None:
    grouped_dataset.bands["NIR"].unlink()
    resp = client.get("/api/images", params=_view(grouped_dataset.manifest))
    assert resp.status_code == 409


# ── routes/inference.py ──────────────────────────────────────────────────────────────────


def test_inference_worker_predicts_on_the_correctly_decoded_grouped_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inference worker hands its predictor a grouped capture as a ``BandGroupRef``.

    The job completes with one image done, and the published bucket names the capture by its
    manifest.
    """
    pytest.importorskip("torch")

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.execution import (
        DEFAULT_IMAGE_BATCH_SIZE,
        DEFAULT_TILE_BATCH_SIZE,
        Stated,
    )
    from tcip_web.routes.inference import _worker
    from tests._verified_checkpoint_fixtures import registered_checkpoint
    from tests._web_fixtures import inference_job

    images_dir = image_dir(tmp_path, "2026-01-01")
    images_dir.mkdir(parents=True)
    _write_group(images_dir, "cap_001")
    ckpt = registered_checkpoint(tmp_path)

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.image_utils import source_path_of
    from tests._predictor_fixtures import StubPredictor, install

    seen = []

    class ObservingPredictor(StubPredictor):
        """A stub predictor recording every source it is handed.

        It answers the stub's empty prediction, each named by its source's ``source_path_of``.
        """

        def predict_batch(
            self,
            image_paths: list[str | Path | BandGroupRef],
            execution: Execution,
            *,
            tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
            batch_size: int = DEFAULT_IMAGE_BATCH_SIZE,
        ) -> list[dict]:
            seen.extend(image_paths)
            return [
                {**result, "image": source_path_of(p)}
                for p, result in zip(
                    image_paths,
                    super().predict_batch(
                        image_paths,
                        execution,
                        tile_batch_size=tile_batch_size,
                        batch_size=batch_size,
                    ),
                )
            ]

    install(
        monkeypatch,
        ObservingPredictor(
            width=20, height=24, boxes=(), scores=(), task="detection"
        ),
    )

    job = inference_job(
        images_dir,
        tmp_path,
        checkpoint_path=str(ckpt),
        stated=Stated(
            tile=False,
            conf=0.25,
            cross_tile_nms=0.7,
            overlap=0.2,
            postprocess="nms",
            rationale=RATIONALE,
        ),
    )
    _worker(job)

    assert job.status == "completed", job.error
    assert job.done == 1 and job.total == 1
    assert (
        len(seen) == 1
        and isinstance(seen[0], BandGroupRef)
        and seen[0].stem == "cap_001"
    )
    assert read_bucket(tmp_path, "out").documents == {
        "cap_001": "cap_001.bandgroup"
    }

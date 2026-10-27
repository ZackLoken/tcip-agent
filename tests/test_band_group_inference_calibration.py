"""A band-grouped capture decodes through the channel-aware reading layer in a predicting pass.

The passes are the assessment's evaluation records over grouped samples, a pass over a grouped
directory, and a pass handed stringified ``BandGroupRef`` sources, which it refuses. Each is
prepared over a tiny 2-channel detection checkpoint, and the first two run its forward pass.
The captures are 2-band groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir

if TYPE_CHECKING:
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.raster_source import RasterSource

torch = pytest.importorskip("torch")

from tests._chain_fixtures import BESPOKE_DETECTION  # noqa: E402
from tests._training_values import RATIONALE  # noqa: E402

pytest.importorskip("torchvision")

TILE = 32


def _write_group(
    images_dir: Path, stem: str, fill: tuple[int, int] = (111, 222)
) -> Path:
    """Write the two band files of ``stem`` in ``images_dir`` and their manifest; return it."""
    from tests._producer_fixtures import write_band_group

    band_a = images_dir / f"{stem}_G.tif"
    band_b = images_dir / f"{stem}_R.tif"
    ramp = np.arange(TILE, dtype=np.uint16)[None, :].repeat(TILE, axis=0)
    tifffile.imwrite(str(band_a), ramp + fill[0])
    tifffile.imwrite(str(band_b), ramp + fill[1])
    return write_band_group(
        images_dir,
        stem,
        {"Green": band_a, "Red": band_b},
        central_wavelength_nm=None,
    )


def _detection_checkpoint(tmp_path: Path) -> str:
    """Return the path of a 2-channel detector a run in ``tmp_path`` registered by completing.

    The run, over grouped captures of its own, saves its built weights without a training step.
    """
    from tests._chain_fixtures import BESPOKE_MODELS, FRAME_STATISTICS
    from tests._verified_checkpoint_fixtures import (
        fixture_data_dir,
        registered_checkpoint,
    )

    model_source = {
        "builder": BESPOKE_DETECTION,
        "source_files": [BESPOKE_MODELS],
        "task": "detection",
        "normalization": FRAME_STATISTICS,
    }
    images_dir = _grouped_dataset(fixture_data_dir(tmp_path, "grouped"))
    return registered_checkpoint(
        tmp_path,
        model_source=model_source,
        data={
            "images_dir": str(images_dir),
            "num_channels": 2,
            "scope": {"subject": "bud"},
        },
    )


def _grouped_dataset(root: Path) -> Path:
    """Write two 2-band grouped captures, each with a ground-truth label document.

    Returns their images directory.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image

    images_dir = image_dir(root, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)

    for manifest in (
        _write_group(images_dir, "capture_001", fill=(111, 222)),
        _write_group(images_dir, "capture_002", fill=(50, 90)),
    ):
        label_image(
            manifest,
            [Annotation(subject="bud", geometry=BBox(2, 2, 10, 10))],
            TILE,
            TILE,
            keep_empty=True,
        )
    return images_dir


def test_the_assessments_records_over_grouped_samples_decode_each_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The assessment's records over grouped samples open each capture as a ``BandGroupRef``.

    The raster layer is handed each capture as a ``BandGroupRef``, and the 2-channel predictor
    writes a record per capture.
    """
    from tcip_mcp.assessment import _records
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines import raster_source
    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.data.split_construction import predictor_dataset
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tests._producer_fixtures import samples_over

    images_dir = _grouped_dataset(tmp_path)
    checkpoint = load_registered_checkpoint(
        _detection_checkpoint(tmp_path), project=tmp_path
    )
    p = prepare(
        checkpoint,
        Stated(tile=False, conf=0.0, postprocess="nms", rationale=RATIONALE),
        device="cpu",
        tile_batch_size=8,
    ).runnable()
    samples = samples_over(images_dir, subject="bud")

    seen_sources = []
    real_open_raster = raster_source.open_raster

    def _spy_open_raster(
        source: str | Path | BandGroupRef, num_channels: int
    ) -> RasterSource:
        seen_sources.append(source)
        return real_open_raster(source, num_channels)

    monkeypatch.setattr(raster_source, "open_raster", _spy_open_raster)

    reference = predictor_dataset(
        "detection", samples, p.scope, p.predictor, None
    )
    records = _records(p, reference, source_digests(samples), p.execution)

    assert len(records) == 2
    grouped = [s for s in seen_sources if isinstance(s, BandGroupRef)]
    assert {g.stem for g in grouped} == {"capture_001", "capture_002"}


def test_run_inference_images_dir_folds_a_grouped_capture(
    tmp_path: Path,
) -> None:
    """A pass over an images directory folds a grouped capture's band files into one image."""
    from tests._verified_checkpoint_fixtures import predicted_over

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    images_dir.mkdir(parents=True)
    manifest = _write_group(images_dir, "capture_001")
    ckpt = _detection_checkpoint(tmp_path)

    _pass, results = predicted_over(
        tmp_path, ckpt, str(images_dir), device="cpu", tile=False
    )

    assert len(results) == 1
    assert results[0]["image"].endswith(manifest.name)


def test_predict_batch_rejects_stringified_band_group_refs(
    tmp_path: Path,
) -> None:
    """A pass refuses stringified ``BandGroupRef`` sources with a ``ValueError``."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tcip_mcp.pipelines.image_utils import list_logical_images

    images_dir = _grouped_dataset(tmp_path)
    checkpoint = load_registered_checkpoint(
        _detection_checkpoint(tmp_path), project=tmp_path
    )
    p = prepare(
        checkpoint,
        Stated(tile=False, conf=0.0, rationale=RATIONALE),
        device="cpu",
    ).runnable()

    logical = list_logical_images(images_dir)
    with pytest.raises(ValueError):
        p.predict([repr(logical[s]) for s in sorted(logical)])

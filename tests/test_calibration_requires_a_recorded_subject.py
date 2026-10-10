"""A detection assessment reads its reference under the run's own recorded subject, or refuses.

A run admitted by the ground-truth shape its bespoke builder reads (mask rasters here) records no
subject, so nothing says which reference records its detections are of. The run is trained one
epoch through the platform's own path and registered; asked to assess it over a drawn document
reference, ``assess_checkpoint`` refuses it by name before any assessment is recorded. The
admitting half is the measurement chain's assessed run (``test_end_to_end_measurement_chain``),
which records the subject it was trained on.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from torch.utils.data import Dataset  # noqa: E402

from tcip_mcp.assessment import ASSESSMENTS_DIR  # noqa: E402

IMG = 48
STEMS = ("m0", "m1", "m2", "m3")
BOX = (8, 10, 30, 34)


class _MaskBoxDataset(Dataset):
    """A bespoke detection dataset over the producer's mask samples: one box per mask, the
    extent of its foreground, as a builder an agent writes for ground truth no registry scopes."""

    def __init__(self, *, samples=None, transforms=None, **_ignored):
        self.samples = list(samples or [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image

        from tcip_mcp.pipelines.image_utils import load_image, pil_to_tensor

        sample = self.samples[idx]
        mask = np.array(Image.open(sample.ground_truth)) > 0
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        return pil_to_tensor(load_image(sample.image, 3)), {
            "boxes": torch.tensor([box]), "labels": torch.tensor([1]),
            "iscrowd": torch.tensor([0]), "image_id": idx}


def build_mask_box_ds(**kwargs) -> _MaskBoxDataset:
    return _MaskBoxDataset(**kwargs)


def _capture(root: Path) -> tuple[Path, Path]:
    """Frames holding one bright square, and its mask raster."""
    from tests._producer_fixtures import BRIGHT, painted_frame

    images, masks = root / "images" / UNDATED_BUCKET, root / "masks"
    for d in (images, masks):
        d.mkdir(parents=True)
    for index, stem in enumerate(STEMS):
        shade = 30 + index
        painted_frame(IMG, IMG, (shade, shade, shade), [(BOX, BRIGHT)]).save(
            images / f"{stem}.png")
        painted_frame(IMG, IMG, 0, [(BOX, 1)], mode="L").save(masks / f"{stem}.png")
    return images, masks


def _train_and_register(data_cfg: dict, project_root: Path) -> dict:
    """Train a detector over ``data_cfg`` in a run of ``project_root`` the launcher's own
    producer opened and the child's own entry ran, which registers its checkpoint by completing:
    ``{"checkpoint", "data"}``, the data block the run resolved in the form its record holds
    it."""
    from tcip_mcp.experiments import observe
    from tests._chain_fixtures import REGION_BUILDER, training_config
    from tests._training_values import evaluation_block
    from tests._verified_checkpoint_fixtures import worker_run

    observation = observe(worker_run(project_root, training_config(
        REGION_BUILDER, data_cfg, evaluation=evaluation_block(selection_metric="loss"))))
    assert observation.checkpoint is not None, observation.final
    return {"checkpoint": observation.checkpoint["path"],
            "data": observation.record["resolved"]["data"]}


def test_a_run_that_recorded_no_subject_is_refused_assessment_by_name(tmp_path: Path):
    from tests import _chain_fixtures as chain

    images, masks = _capture(tmp_path / "masks")
    data_cfg = {"images_dir": str(images), "labels_dir": str(masks), "auto_val": False,
                "dataset_source": {"builder": f"{Path(__file__).stem}:build_mask_box_ds",
                                   "source_files": [__file__]}}
    trained = _train_and_register(data_cfg, tmp_path)
    checkpoint = trained["checkpoint"]
    # The producer admitted by shape and recorded an empty scope.
    assert trained["data"]["scope"] == {"subject": None, "attributes": None}
    chain.synthetic_capture(tmp_path / "ds")
    chain.draw_reference_selection(tmp_path, tmp_path / "ds", tmp_path / "selection")
    chain.confirm_count_trait(tmp_path)

    result = chain.assess(tmp_path, checkpoint, tmp_path / "selection", device="cpu")

    assert "records no subject" in result.get("error", ""), result
    assert not (tmp_path / ASSESSMENTS_DIR).exists()

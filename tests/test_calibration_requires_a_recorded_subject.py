"""A detection calibration reads its reference under the run's own recorded subject, or refuses.

A run admitted by the ground-truth shape its bespoke builder reads (mask rasters here) records no
subject, and neither a trait nor a labels directory says which reference records its detections
are of. The run is trained one epoch through the platform's own path and registered; asked to
calibrate, ``run_inference`` refuses it by name before any reference is read or any split locked.
The admitting half is the measurement chain's calibrated run
(``test_end_to_end_measurement_chain``), which records the subject it was trained on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from torch.utils.data import Dataset  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402

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
        return pil_to_tensor(load_image(Path(sample.source), 3)), {
            "boxes": torch.tensor([box]), "labels": torch.tensor([1]), "image_id": idx}


def build_mask_box_ds(**kwargs) -> _MaskBoxDataset:
    return _MaskBoxDataset(**kwargs)


def _capture(root: Path) -> tuple[Path, Path, Path]:
    """Frames holding one bright square, its mask raster, and a reviewed per-image reference."""
    from PIL import Image, ImageDraw

    images, masks, reference = root / "images", root / "masks", root / "reference"
    for d in (images, masks, reference):
        d.mkdir(parents=True)
    for index, stem in enumerate(STEMS):
        shade = 30 + index
        frame = Image.new("RGB", (IMG, IMG), (shade, shade, shade))
        ImageDraw.Draw(frame).rectangle([BOX[0], BOX[1], BOX[2] - 1, BOX[3] - 1], fill=(230,) * 3)
        frame.save(images / f"{stem}.png")
        mask = Image.new("L", (IMG, IMG), 0)
        ImageDraw.Draw(mask).rectangle([BOX[0], BOX[1], BOX[2] - 1, BOX[3] - 1], fill=1)
        mask.save(masks / f"{stem}.png")
        json_io.write_annotations(
            reference / f"{stem}.json",
            [Annotation(subject="bur", geometry=BBox(*BOX), created_by="user:breeder")], IMG, IMG)
    return images, masks, reference


def _train_and_register(data_cfg: dict, out_dir: Path, project_root: Path) -> str:
    from torch.utils.data import DataLoader

    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.training.collation import task_collate
    from tcip_mcp.pipelines.training.generic_trainer import train
    from tcip_mcp.pipelines.training.run_registry import create_run
    from tcip_mcp.tools.model_tools import register_model

    config = {
        "model_source": {"builder": "tests.bespoke_models:build_bright_region_detector",
                         "builder_kwargs": {}, "task": "detection", "in_chans": 3},
        "data": data_cfg, "batch_size": 2, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu", "checkpoint_every_n_epochs": 1,
        "early_stopping": {"enabled": False},
        "optimizer": {"name": "sgd", "backbone_lr": 1e-3, "head_lr": 1e-2, "weight_decay": 0},
        "scheduler": {"type": "cosine"}, "gradient_accumulation_steps": 1,
    }
    train_ds, _val, _partition = auto_train_val("detection", data_cfg, None)
    collate = task_collate("detection")
    run = create_run(config, str(out_dir), id=out_dir.name)
    completed = train(run, DataLoader(train_ds, batch_size=2, collate_fn=collate),
                      val_loader=DataLoader(train_ds, batch_size=2, collate_fn=collate),
                      task="detection")
    assert completed.status == "completed", completed.status
    checkpoint = out_dir / "model_best.pt"
    registered = register_model(name=out_dir.name, checkpoint_path=str(checkpoint), config={},
                                project_path=str(project_root))
    assert "error" not in registered, registered
    return str(checkpoint)


def _calibrate(checkpoint: str, images: Path, reference: Path, out: Path) -> dict:
    from tests import _operationalization_fixtures as fx

    from tcip_mcp.tools.inference_tools import run_inference

    return run_inference(checkpoint_path=checkpoint, images_dir=str(images),
                         output_dir=str(out), trait=fx.COUNT_TRAIT,
                         calibration_labels_dir=str(reference))


def test_a_run_that_recorded_no_subject_is_refused_calibration_by_name(tmp_path: Path):
    from tests import _operationalization_fixtures as fx

    images, masks, reference = _capture(tmp_path / "ds")
    fx.write_spec(tmp_path, fx.COUNT_SPEC)
    data_cfg = {"images_dir": str(images), "labels_dir": str(masks), "auto_val": False,
                "dataset_source": {"builder": f"{__name__}:build_mask_box_ds"}}
    checkpoint = _train_and_register(data_cfg, tmp_path / "unscoped", tmp_path)
    assert data_cfg.get("subject") is None  # the producer admitted by shape and stamped none

    result = _calibrate(checkpoint, images, reference, tmp_path / "ds" / "predictions" / "a")

    assert "records no subject" in result.get("error", ""), result
    assert not (tmp_path / "ds" / "predictions" / "a").exists() or not any(
        (tmp_path / "ds" / "predictions" / "a").iterdir())


def test_the_id_map_resolver_reads_the_scopes_own_subject_decision(tmp_path: Path):
    """A training scope naming no subject is refused by the scope's one statement of that
    refusal, an empty name read as none; a named single-class scope still resolves its map."""
    from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map

    _images, _masks, reference = _capture(tmp_path / "ds")
    for subject, attribute in ((None, None), ("", "")):
        with pytest.raises(ValueError, match="records no subject"):
            resolve_registry_id_map(reference, subject, attribute)
    assert resolve_registry_id_map(reference, "bur", None)[1] == {"bur": 0}

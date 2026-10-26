"""The batch the contract synthesizes stands in for the batch the real loaders build.

``check_model_contract`` and ``overfit_check`` smoke a model against a synthetic batch whenever
none is supplied. Its targets carry the semantics the run's own loaders emit: 1-indexed
foreground detection labels (0 is background for a torchvision detector), a positive-area box
inside the frame, and the rank key an ordinal head consumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")
pytest.importorskip("PIL")
import torch  # noqa: E402

from tcip_mcp.pipelines.components.heads import OrdinalHead  # noqa: E402
from tcip_mcp.pipelines.model_contract import (
    check_model_contract,
    overfit_check,
)  # noqa: E402
from tcip_mcp.pipelines.training.collation import task_collate  # noqa: E402
from tests._producer_fixtures import dataset_over, registry_over  # noqa: E402

# What the smokes below synthesize their batch at, the shape a run resolves for itself.
DET_DIMS = {"in_chans": 3, "num_classes": 1, "img_size": (64, 64)}
ORD_DIMS = {"in_chans": 3, "num_classes": 4, "img_size": (64, 64)}


class _DetectionTargetRecorder(torch.nn.Module):
    """A detection model recording the targets of each training forward.

    In training with targets given it appends them to ``seen`` and answers ``{"loss": self.scale
    * 1.0}``; otherwise it answers one zero box scored 1 with label 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen: list = []

    def forward(
        self, images: list[torch.Tensor], targets: list[dict] | None = None
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        if self.training and targets is not None:
            self.seen.append(targets)
            return {"loss": self.scale * 1.0}
        return [
            {
                "boxes": torch.zeros((1, 4)),
                "scores": torch.ones(1),
                "labels": torch.ones(1, dtype=torch.int64),
            }
        ]


class _OrdinalProbe(torch.nn.Module):
    """A minimal ordinal model built on the real ``OrdinalHead``.

    It feeds the head each image's per-channel mean. In training with targets given it appends
    the targets to ``seen`` and answers the head's own losses, prefixed ``head0_``; otherwise it
    answers the head's decoding.
    """

    def __init__(self, num_ranks: int, in_chans: int = 3) -> None:
        super().__init__()
        self.head = OrdinalHead(in_chans, num_ranks)
        self.seen: list = []

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict[str, torch.Tensor]:
        out = self.head(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            self.seen.append(targets)
            return {
                f"head0_{k}": v
                for k, v in self.head.compute_loss(out, targets).items()
            }
        return self.head.decode(out)


def _real_detection_item(tmp_path: Path) -> tuple[torch.Tensor, dict]:
    """Return one ``(image, target)`` from a real ``DetectionDataset``.

    It writes a 96 by 48 image labeled with one ``bud`` box ``(4, 6, 40, 19)`` under a registry
    listing ``bush`` then ``bud``, and returns the first item of the detection dataset over it.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import label_image, write_image

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    registry_over(
        tmp_path, SubjectRegistry((Subject("bush", ""), Subject("bud", "")))
    )
    write_image(images_dir / "a.png", (96, 48))
    label_image(
        images_dir / "a.png",
        [Annotation(subject="bud", geometry=BBox(4, 6, 40, 19))],
        96,
        48,
    )
    return dataset_over("detection", str(images_dir), subject="bud")[0]


def _real_detection_target(tmp_path: Path) -> dict:
    return _real_detection_item(tmp_path)[1]


def _real_ordinal_items(tmp_path: Path) -> list[tuple[torch.Tensor, dict]]:
    """Return two ``(image, target)`` items from a real ``OrdinalDataset``.

    The dataset is over two blank 96 by 48 frames ranked 0 and 3.
    """
    from tests._producer_fixtures import write_image

    images_dir = image_dir(tmp_path / "ordinal", UNDATED_BUCKET)
    for name in ("a", "b"):
        write_image(images_dir / f"{name}.png", (96, 48))
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("image_stem,rank\na,0\nb,3\n")
    dataset = dataset_over("ordinal", str(images_dir), str(csv_path))
    return [dataset[0], dataset[1]]


def _real_ordinal_target(tmp_path: Path) -> dict:
    return _real_ordinal_items(tmp_path)[1][1]


def _synthetic_detection_target() -> dict:
    recorder = _DetectionTargetRecorder()
    check_model_contract(
        recorder, "detection", device=torch.device("cpu"), dims=DET_DIMS
    )
    assert len(recorder.seen) == 1, (
        "the contract never ran a train-mode forward"
    )
    targets = recorder.seen[0]
    assert len(targets) == 1
    return targets[0]


def test_synthetic_detection_target_labels_a_foreground_object(
    tmp_path: Path,
) -> None:
    """The synthetic detection target labels a foreground object, as the real loader does.

    Label 0 is background to a torchvision detector. The real loader and the synthetic stand-in
    both emit one label of the same dtype, at least 1.
    """
    synth_labels = _synthetic_detection_target()["labels"]
    real_labels = _real_detection_target(tmp_path)["labels"]

    assert synth_labels.numel() == 1
    assert real_labels.numel() == 1
    assert synth_labels.dtype == real_labels.dtype
    assert int(real_labels.min()) >= 1
    assert int(synth_labels.min()) >= 1


def test_synthetic_detection_box_covers_a_positive_area_inside_the_frame() -> (
    None
):
    """The synthetic detection box covers a positive area inside a non-square frame."""
    width, height = 96, 48
    recorder = _DetectionTargetRecorder()
    check_model_contract(
        recorder,
        "detection",
        device=torch.device("cpu"),
        dims={**DET_DIMS, "img_size": (width, height)},
    )
    boxes = recorder.seen[0][0]["boxes"]

    assert boxes.shape == (1, 4)
    x1, y1, x2, y2 = (float(v) for v in boxes[0])
    assert x2 > x1
    assert y2 > y1
    assert 0.0 <= x1 and x2 <= width
    assert 0.0 <= y1 and y2 <= height


def test_synthetic_detection_target_carries_only_keys_the_real_loader_carries(
    tmp_path: Path,
) -> None:
    """The synthetic detection target carries only keys the real loader emits.

    It carries ``boxes`` and ``labels``, and no key the real loader's target lacks.
    """
    synth = _synthetic_detection_target()
    real = _real_detection_target(tmp_path)

    assert set(synth)
    assert {"boxes", "labels"} <= set(synth)
    assert set(synth) <= set(real), (sorted(synth), sorted(real))


def test_synthetic_ordinal_target_carries_only_keys_the_real_loader_carries(
    tmp_path: Path,
) -> None:
    """The synthetic ordinal rank target carries only keys the real ordinal loader emits."""
    probe = _OrdinalProbe(num_ranks=4)
    check_model_contract(
        probe, "ordinal", device=torch.device("cpu"), dims=ORD_DIMS
    )
    assert len(probe.seen) == 1, "the contract never ran a train-mode forward"
    synth = probe.seen[0]
    real = _real_ordinal_target(tmp_path)

    assert set(synth)
    assert set(synth) <= set(real), (sorted(synth), sorted(real))


def test_an_ordinal_head_consumes_the_synthetic_ordinal_batch() -> None:
    """A model on the real ordinal head passes the contract on the synthetic batch.

    The report is ok, carries a training loss, and names a dict eval output.
    """
    report = check_model_contract(
        _OrdinalProbe(num_ranks=4),
        "ordinal",
        device=torch.device("cpu"),
        dims=ORD_DIMS,
    )

    assert report["ok"], report["issues"]
    assert report["train_loss"] is not None
    assert report["eval_output_type"] == "dict"


class _BatchRecorder(torch.nn.Module):
    """A model recording the images and targets of each training forward.

    In training with targets given it appends ``(images, targets)`` to ``seen`` and answers
    ``{"loss": self.scale * 1.0}``; otherwise it answers ``{"values": torch.zeros(1)}``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen: list = []

    def forward(
        self, images: torch.Tensor, targets: dict | None = None
    ) -> dict[str, torch.Tensor]:
        if self.training and targets is not None:
            self.seen.append((images, targets))
            return {"loss": self.scale * 1.0}
        return {"values": torch.zeros(1)}


def test_synthetic_detection_batch_is_assembled_by_the_loader_collate(
    tmp_path: Path,
) -> None:
    """The contract's detection target carries the loader-collated target's keys and types.

    Its per-image target holds exactly the keys the real loader's collated target holds, each at
    the same Python type.
    """
    recorder = _DetectionTargetRecorder()
    check_model_contract(
        recorder, "detection", device=torch.device("cpu"), dims=DET_DIMS
    )
    assert len(recorder.seen) == 1, (
        "the contract never ran a train-mode forward"
    )
    synth_target = recorder.seen[0][0]

    real_images, real_targets = task_collate("detection")(
        [_real_detection_item(tmp_path)]
    )

    assert isinstance(real_images, list)
    assert isinstance(real_targets, list)
    assert set(synth_target) == set(real_targets[0]), (
        sorted(synth_target),
        sorted(real_targets[0]),
    )
    for key, real_value in real_targets[0].items():
        assert type(synth_target[key]) is type(real_value), key


def test_synthetic_ordinal_batch_is_assembled_by_the_loader_collate(
    tmp_path: Path,
) -> None:
    """The contract's ordinal batch matches the loader-collated batch's keys, dtypes and ranks.

    Each target is batched along the same axis as the images.
    """
    recorder = _BatchRecorder()
    check_model_contract(
        recorder, "ordinal", device=torch.device("cpu"), dims=ORD_DIMS
    )
    assert len(recorder.seen) == 1, (
        "the contract never ran a train-mode forward"
    )
    synth_images, synth_targets = recorder.seen[0]

    real_images, real_targets = task_collate("ordinal")(
        _real_ordinal_items(tmp_path)
    )

    assert isinstance(synth_images, torch.Tensor)
    assert isinstance(real_targets, dict)
    assert synth_images.ndim == real_images.ndim
    assert set(synth_targets) == set(real_targets), (
        sorted(synth_targets),
        sorted(real_targets),
    )
    for key, real_value in real_targets.items():
        assert synth_targets[key].dtype == real_value.dtype, key
        assert synth_targets[key].ndim == real_value.ndim, key
        assert synth_targets[key].shape[0] == synth_images.shape[0], key


def test_real_models_still_smoke_green_on_the_collated_batch() -> None:
    """A real detector and an ordinal-head model pass on the collated batch.

    A real torchvision detector passes the contract on the collated batch, per-image key
    included, and a model on the real ordinal head passes the overfit check on it.
    """
    pytest.importorskip("torchvision")
    from tests import bespoke_models
    from tests._chain_fixtures import detector_values

    detector = bespoke_models.build_bespoke_detection(
        num_classes=1, **detector_values()
    )
    report = check_model_contract(
        detector, "detection", device=torch.device("cpu"), dims=DET_DIMS
    )
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"

    learned = overfit_check(
        _OrdinalProbe(num_ranks=4),
        "ordinal",
        steps=20,
        dims=ORD_DIMS,
        seed=0,
        deterministic=False,
        device=torch.device("cpu"),
    )
    assert learned["passed"], learned["issue"]

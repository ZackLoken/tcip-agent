"""The batch the contract synthesizes stands in for the batch the real loaders build.

``check_model_contract`` and ``overfit_check`` smoke a model against a synthetic batch whenever
the caller supplies none. That batch is what the report is earned against, so its targets carry
the same semantics the run's own loaders emit: 1-indexed foreground detection labels (0 is
background for a torchvision detector), a real positive-area box, and the rank key an ordinal
head consumes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")
import torch  # noqa: E402

from tcip_mcp.pipelines.components.heads import OrdinalHead  # noqa: E402
from tcip_mcp.pipelines.model_contract import check_model_contract, overfit_check  # noqa: E402
from tcip_mcp.pipelines.training.collation import task_collate  # noqa: E402


class _DetectionTargetRecorder(torch.nn.Module):
    """Records the training batch handed to it and answers with a well-formed detection output."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen: list = []

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            self.seen.append(targets)
            return {"loss": self.scale * 1.0}
        return [{"boxes": torch.zeros((1, 4)), "scores": torch.ones(1),
                 "labels": torch.ones(1, dtype=torch.int64)}]


class _OrdinalProbe(torch.nn.Module):
    """A minimal ordinal model built on the real ``OrdinalHead``.

    The head's own target contract, not a restatement of it here, decides whether the synthetic
    ordinal batch is consumable. It also records the batch it was handed.
    """

    def __init__(self, num_ranks: int, in_chans: int = 3) -> None:
        super().__init__()
        self.head = OrdinalHead(in_chans, num_ranks)
        self.seen: list = []

    def forward(self, images, targets=None):
        out = self.head(images.mean(dim=(2, 3)))
        if self.training and targets is not None:
            self.seen.append(targets)
            return {f"head0_{k}": v for k, v in self.head.compute_loss(out, targets).items()}
        return self.head.decode(out)


def _real_detection_item(tmp_path):
    """One ``(image, target)`` from a real ``DetectionDataset``: a non-square frame, a non-square
    box, and a subject that is not the first one in the registry."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    write_registry(tmp_path / "classes.json",
                   ClassRegistry((Subject("bush"), Subject("bud"))))
    Image.new("RGB", (96, 48)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="bud", geometry=BBox(4, 6, 40, 19))], 96, 48)
    return DetectionDataset(str(images_dir), str(labels_dir), subject="bud")[0]


def _real_detection_target(tmp_path):
    return _real_detection_item(tmp_path)[1]


def _real_ordinal_items(tmp_path):
    """Two ``(image, target)`` items from a real ``OrdinalDataset`` over a skewed, sparsely
    populated rank column."""
    from PIL import Image
    from tcip_mcp.pipelines.data.datasets import OrdinalDataset

    images_dir = tmp_path / "ordinal_images"
    images_dir.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (96, 48)).save(images_dir / f"{name}.png")
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("image_stem,rank\na,0\nb,3\n")
    dataset = OrdinalDataset(str(images_dir), str(csv_path))
    return [dataset[0], dataset[1]]


def _real_ordinal_target(tmp_path):
    return _real_ordinal_items(tmp_path)[1][1]


def _synthetic_detection_target():
    recorder = _DetectionTargetRecorder()
    check_model_contract(recorder, "detection", num_classes=1, img_size=64)
    assert len(recorder.seen) == 1, "the contract never ran a train-mode forward"
    targets = recorder.seen[0]
    assert len(targets) == 1
    return targets[0]


def test_synthetic_detection_target_labels_a_foreground_object(tmp_path):
    """Label 0 is background to a torchvision detector. The real loader emits 1-indexed
    foreground labels, and the synthetic stand-in emits foreground labels too, so a green report
    is evidence the model can learn the object rather than learn to predict nothing."""
    synth_labels = _synthetic_detection_target()["labels"]
    real_labels = _real_detection_target(tmp_path)["labels"]

    assert synth_labels.numel() == 1
    assert real_labels.numel() == 1
    assert synth_labels.dtype == real_labels.dtype
    assert int(real_labels.min()) >= 1
    assert int(synth_labels.min()) >= 1


def test_synthetic_detection_box_covers_a_positive_area_inside_the_frame():
    """A degenerate or out-of-frame box is not an object a detector can be asked to learn."""
    img_size = 64
    recorder = _DetectionTargetRecorder()
    check_model_contract(recorder, "detection", num_classes=1, img_size=img_size)
    boxes = recorder.seen[0][0]["boxes"]

    assert boxes.shape == (1, 4)
    x1, y1, x2, y2 = (float(v) for v in boxes[0])
    assert x2 > x1
    assert y2 > y1
    assert 0.0 <= x1 and x2 <= img_size
    assert 0.0 <= y1 and y2 <= img_size


def test_synthetic_detection_target_carries_only_keys_the_real_loader_carries(tmp_path):
    """Every key the synthetic detection target carries is one the real detection loader emits,
    so a model written against the loader's targets can be smoked against this batch."""
    synth = _synthetic_detection_target()
    real = _real_detection_target(tmp_path)

    assert set(synth)
    assert {"boxes", "labels"} <= set(synth)
    assert set(synth) <= set(real), (sorted(synth), sorted(real))


def test_synthetic_ordinal_target_carries_only_keys_the_real_loader_carries(tmp_path):
    """Same agreement for ordinal: the synthetic rank target uses the real loader's key."""
    probe = _OrdinalProbe(num_ranks=4)
    check_model_contract(probe, "ordinal", num_classes=4)
    assert len(probe.seen) == 1, "the contract never ran a train-mode forward"
    synth = probe.seen[0]
    real = _real_ordinal_target(tmp_path)

    assert set(synth)
    assert set(synth) <= set(real), (sorted(synth), sorted(real))


def test_an_ordinal_head_consumes_the_synthetic_ordinal_batch():
    """The rail admits valid work: a model built on the real ordinal head passes the contract on
    the synthetic batch, which is only true while that batch carries what the head reads."""
    report = check_model_contract(_OrdinalProbe(num_ranks=4), "ordinal", num_classes=4)

    assert report["ok"], report["issues"]
    assert report["train_loss"] is not None
    assert report["eval_output_type"] == "dict"


class _BatchRecorder(torch.nn.Module):
    """Records the whole training batch handed to it and answers with a tensor-carrying dict."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.seen: list = []

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            self.seen.append((images, targets))
            return {"loss": self.scale * 1.0}
        return {"values": torch.zeros(1)}


def test_synthetic_detection_batch_is_assembled_by_the_loader_collate(tmp_path):
    """The contract's detection batch is the one the trainer's collate builds over per-sample
    items, so it carries every per-image target key the real loader emits, at the same python
    type, rather than a hand-assembled subset of them that can drift."""
    recorder = _DetectionTargetRecorder()
    check_model_contract(recorder, "detection", num_classes=1, img_size=64)
    assert len(recorder.seen) == 1, "the contract never ran a train-mode forward"
    synth_target = recorder.seen[0][0]

    real_images, real_targets = task_collate("detection")([_real_detection_item(tmp_path)])

    assert isinstance(real_images, list)
    assert isinstance(real_targets, list)
    assert set(synth_target) == set(real_targets[0]), (sorted(synth_target), sorted(real_targets[0]))
    for key, real_value in real_targets[0].items():
        assert type(synth_target[key]) is type(real_value), key


def test_synthetic_ordinal_batch_is_assembled_by_the_loader_collate(tmp_path):
    """Same for a stacking task: the contract's ordinal batch carries the loader's target keys at
    the loader's dtype and rank, batched along the same axis as the images."""
    recorder = _BatchRecorder()
    check_model_contract(recorder, "ordinal", num_classes=4)
    assert len(recorder.seen) == 1, "the contract never ran a train-mode forward"
    synth_images, synth_targets = recorder.seen[0]

    real_images, real_targets = task_collate("ordinal")(_real_ordinal_items(tmp_path))

    assert isinstance(synth_images, torch.Tensor)
    assert synth_images.ndim == real_images.ndim
    assert set(synth_targets) == set(real_targets), (sorted(synth_targets), sorted(real_targets))
    for key, real_value in real_targets.items():
        assert synth_targets[key].dtype == real_value.dtype, key
        assert synth_targets[key].ndim == real_value.ndim, key
        assert synth_targets[key].shape[0] == synth_images.shape[0], key


def test_real_models_still_smoke_green_on_the_collated_batch():
    """The rail admits valid work: a real torchvision detector passes the contract on the collated
    batch, per-image key included, and a model on the real ordinal head still learns on it."""
    pytest.importorskip("torchvision")
    from tests import bespoke_models

    detector = bespoke_models.build_bespoke_detection(num_classes=1, min_size=64, max_size=128)
    report = check_model_contract(detector, "detection", num_classes=1, img_size=64)
    assert report["ok"], report["issues"]
    assert report["eval_output_type"] == "list[dict]"

    learned = overfit_check(_OrdinalProbe(num_ranks=4), "ordinal", steps=20, num_classes=4, seed=0)
    assert learned["passed"], learned["issue"]

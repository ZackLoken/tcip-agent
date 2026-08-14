"""The batch the contract synthesizes stands in for the batch the real loaders build.

``check_model_contract`` and ``overfit_check`` smoke a model against a synthetic batch whenever
the caller supplies none. That batch is what the report is earned against, so its targets carry
the same semantics the run's own loaders emit: 1-indexed foreground detection labels (0 is
background for a torchvision detector), a real positive-area box, and the rank key an ordinal
head consumes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from tcip_mcp.pipelines.components.heads import OrdinalHead  # noqa: E402
from tcip_mcp.pipelines.model_contract import check_model_contract  # noqa: E402


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


def _real_detection_target(tmp_path):
    """One target from a real ``DetectionDataset``: a non-square frame, a non-square box, and a
    subject that is not the first one in the registry."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.class_registry import ClassRegistry, Subject, write_registry
    from tcip_mcp.pipelines.data.datasets import DetectionDataset

    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    write_registry(tmp_path / "classes.json",
                   ClassRegistry((Subject("bush"), Subject("catkin"))))
    Image.new("RGB", (96, 48)).save(images_dir / "a.png")
    json_io.write_annotations(str(labels_dir / "a.json"),
                              [Annotation(subject="catkin", geometry=BBox(4, 6, 40, 19))], 96, 48)
    _img, target = DetectionDataset(str(images_dir), str(labels_dir), subject="catkin")[0]
    return target


def _real_ordinal_target(tmp_path):
    """One target from a real ``OrdinalDataset`` over a skewed, sparsely populated rank column."""
    from PIL import Image
    from tcip_mcp.pipelines.data.datasets import OrdinalDataset

    images_dir = tmp_path / "ordinal_images"
    images_dir.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (96, 48)).save(images_dir / f"{name}.png")
    csv_path = tmp_path / "ranks.csv"
    csv_path.write_text("image_stem,rank\na,0\nb,3\n")
    _img, target = OrdinalDataset(str(images_dir), str(csv_path))[1]
    return target


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

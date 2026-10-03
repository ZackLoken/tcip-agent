"""Building a loader over data on disk the way every door in the platform does.

A loader is built from the samples the producer named, never from a place to look, so a test that
needs one over a directory or a table admits through ``label_queries.admit`` first and builds from
what it answers. These helpers are that one route, so a fixture and the production path cannot
drift into admitting different membership.
"""

from __future__ import annotations

from typing import Any


def seed_leaf_detection_dataset(root) -> tuple:
    """Two training and one validation 128px image under ``root``, each with a label document
    holding one ``leaf`` box; ``(images, labels, val_images, val_labels)``."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir, labels_dir = root / "images", root / "labels"
    val_images, val_labels = root / "val_images", root / "val_labels"
    for d in (images_dir, labels_dir, val_images, val_labels):
        d.mkdir(parents=True)
    leaf = [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30))]
    for images, labels, stem in ((images_dir, labels_dir, "t0"), (images_dir, labels_dir, "t1"),
                                 (val_images, val_labels, "v0")):
        Image.new("RGB", (128, 128)).save(images / f"{stem}.png")
        json_io.write_annotations(str(labels / f"{stem}.json"), leaf, 128, 128)
    return images_dir, labels_dir, val_images, val_labels


def seed_two_bud_images(images_dir, labels_dir) -> None:
    """Make ``images_dir`` and ``labels_dir`` and write two 32px images, each with a label
    document holding one ``bud`` box, so a drawn split holds one out for validation."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (32, 32), color=(10 * i, 0, 0)).save(images_dir / f"img{i}.png")
        json_io.write_annotations(str(labels_dir / f"img{i}.json"),
                                  [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)


def small_detection_config(images_dir, labels_dir, experiment_id: str) -> dict:
    """A one-epoch CPU detection run of the bespoke detector over ``images_dir`` and
    ``labels_dir`` under ``experiment_id``."""
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "labels_dir": str(labels_dir),
                 "scope": {"subject": "bud"}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
        "experiment_id": experiment_id,
    }


def registry_over(dataset_root, registry) -> None:
    """``dataset_root``'s subject registry replaced by ``registry`` through the platform's own save
    (:func:`~tcip_mcp.subject_registry.replace_registry`), whatever it held before."""
    from tcip_mcp.subject_registry import replace_registry

    replace_registry(dataset_root, registry, expect=None, allow_removals=True,
                     allow_type_changes=True, actor=None)


def mark_complete(image_path, label_path, subject: str, *, project, rect=None,
                  proposals_hidden: bool = False, by: str = "user:tester"):
    """``subject`` marked complete over ``rect`` (the whole image when ``None``) on the label
    document at ``label_path`` through the platform's one save door, the document's annotations
    kept as they are; an image with no annotation of ``subject`` reads as a confirmed negative.
    Returns the new version."""
    from tcip_annotation.json_io import client_annotation, read_label_document

    from tcip_mcp.dataset_layout import Gestures, save_label_document
    from tcip_mcp.pipelines.image_utils import image_path_dimensions

    doc = read_label_document(label_path)
    width, height = image_path_dimensions(image_path)
    return save_label_document(
        project, image_path, label_path, [client_annotation(a) for a in doc.annotations],
        width=width, height=height, author=by, actor=by,
        gestures=Gestures(complete={subject: True}, rect=rect, proposals_hidden=proposals_hidden))


def admit_over(
    images_dir, ground_truth, *, subject: str | None = None, members: list[str] | None = None,
):
    """The admission over one place holding ground truth under the class space its registry
    answers for ``subject`` (:func:`~tcip_mcp.pipelines.data.label_queries.registry_scope`),
    refusing an empty one by name."""
    from tcip_mcp.pipelines.data.label_queries import admit, registry_scope, require_admitted

    admitted = admit(images_dir, ground_truth, scope=registry_scope(ground_truth, subject),
                     members=members)
    require_admitted(admitted)
    return admitted


def samples_over(
    images_dir, ground_truth, *, subject: str | None = None, members: list[str] | None = None,
):
    """Every admitted member of one place as samples, each on the training side."""
    return admit_over(images_dir, ground_truth, subject=subject, members=members).every_sample()


def run_over(
    task: str, images_dir, ground_truth, *, subject: str | None = None,
    members: list[str] | None = None, stated: dict[str, Any] | None = None, **kwargs: Any,
):
    """A loader for ``task`` over one place holding ground truth and the data section a run over
    it records (its ``scope`` and sizes), both through the producer.

    ``stated`` is what a config would state about the sizes (a band count, a class count); the
    rest are resolved off the admitted samples the way a run resolves them."""
    from dataclasses import asdict

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.split_construction import run_sizes

    admitted = admit_over(images_dir, ground_truth, subject=subject, members=members)
    samples = admitted.every_sample()
    data = {**(stated or {}), "scope": asdict(admitted.scope)}
    sizes = run_sizes(task, data, samples, kwargs.get("dataset_source"))
    return build_dataset(task, samples=samples, scope=admitted.scope, sizes=sizes, **kwargs), data


def dataset_over(task: str, images_dir, ground_truth, **kwargs: Any):
    """The loader :func:`run_over` builds."""
    return run_over(task, images_dir, ground_truth, **kwargs)[0]

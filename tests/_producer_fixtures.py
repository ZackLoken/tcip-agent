"""Label documents written and loaders built through the platform's own producers
(``label_queries.admit``, ``write_label_document``, the one label save)."""

from __future__ import annotations

from typing import Any

from tcip_mcp.dataset_layout import label_key_of as image_label_key


def label_image(image_path, annotations, width: int, height: int, **kwargs: Any):
    """``annotations`` written as the label document of the image at ``image_path``
    (:func:`~tcip_annotation.json_io.write_label_document` at the image's own key); the new
    version."""
    from tcip_annotation.json_io import write_label_document

    return write_label_document(image_label_key(image_path), annotations, width, height, **kwargs)


def seed_leaf_detection_dataset(root) -> tuple:
    """Two training and one validation 128px image under ``root``'s image tree, captures
    ``train`` and ``val``, each with a label document holding one ``leaf`` box;
    ``(images, val_images)``."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    images_dir, val_images = root / "images" / "train", root / "images" / "val"
    for d in (images_dir, val_images):
        d.mkdir(parents=True)
    leaf = [Annotation(subject="leaf", geometry=BBox(10, 10, 30, 30))]
    for images, stem in ((images_dir, "t0"), (images_dir, "t1"), (val_images, "v0")):
        Image.new("RGB", (128, 128)).save(images / f"{stem}.png")
        label_image(images / f"{stem}.png", leaf, 128, 128)
    return images_dir, val_images


def seed_two_bud_images(images_dir) -> None:
    """Make ``images_dir``, a capture of a dataset image tree, and write two 32px images, each
    with a label document holding one ``bud`` box, so a drawn split holds one out for
    validation."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    images_dir.mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (32, 32), color=(10 * i, 0, 0)).save(images_dir / f"img{i}.png")
        label_image(images_dir / f"img{i}.png",
                    [Annotation(subject="bud", geometry=BBox(1, 1, 9, 9))], 32, 32)


def small_detection_config(images_dir, experiment_id: str) -> dict:
    """A one-epoch CPU detection run of the bespoke detector over ``images_dir`` under
    ``experiment_id``, drawing its split at seed 0."""
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
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


def mark_complete(image_path, subject: str, *, project, rect=None,
                  proposals_hidden: bool = False, by: str = "user:tester"):
    """``subject`` marked complete over ``rect`` (the whole image when ``None``) on the label
    document of the image at ``image_path`` through the platform's one save door, the document's
    annotations kept as they are; an image with no annotation of ``subject`` reads as a confirmed
    negative. Returns the new version."""
    from tcip_annotation.json_io import client_annotation, read_document_versioned

    from tcip_mcp.dataset_layout import Gestures, save_label_document
    from tcip_mcp.pipelines.image_utils import image_path_dimensions

    key = image_label_key(image_path)
    doc, _version = read_document_versioned(key)
    width, height = image_path_dimensions(image_path)
    return save_label_document(
        project, key, [client_annotation(a) for a in doc.annotations],
        width=width, height=height, author=by, actor=by,
        gestures=Gestures(complete={subject: True}, rect=rect, proposals_hidden=proposals_hidden))


def admit_over(
    images_dir, ground_truth=None, *, subject: str | None = None,
    members: list[str] | None = None,
):
    """The admission over the images of ``images_dir`` and their ground truth (their own label
    documents when ``ground_truth`` is ``None``) under the class space their registry answers for
    ``subject`` (:func:`~tcip_mcp.pipelines.data.label_queries.registry_scope`), refusing an empty
    one by name."""
    from tcip_mcp.pipelines.data.label_queries import admit, registry_scope, require_admitted

    admitted = admit(images_dir, ground_truth, scope=registry_scope(images_dir, subject),
                     members=members)
    require_admitted(admitted)
    return admitted


def checkpoint_admission(checkpoint, images_dir, ground_truth=None):
    """The admission ``evaluate_model`` measures ``checkpoint`` over: the images of
    ``images_dir`` and their ground truth under the class space the checkpoint records."""
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.selection import ClassScope

    return admit(images_dir, ground_truth, scope=ClassScope.of(checkpoint.data_config))


def samples_over(
    images_dir, ground_truth=None, *, subject: str | None = None,
    members: list[str] | None = None,
):
    """Every admitted member of one place as samples, each on the training side."""
    return admit_over(images_dir, ground_truth, subject=subject, members=members).every_sample()


def run_over(
    task: str, images_dir, ground_truth=None, *, subject: str | None = None,
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


def dataset_over(task: str, images_dir, ground_truth=None, **kwargs: Any):
    """The loader :func:`run_over` builds."""
    return run_over(task, images_dir, ground_truth, **kwargs)[0]

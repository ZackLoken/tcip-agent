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


def box_annotation(x1: float, y1: float, x2: float, y2: float, *, subject: str = "bud",
                   score: float | None = None, **attributes: str):
    """An annotation of ``subject`` boxed at pixel ``(x1, y1, x2, y2)`` carrying ``attributes``;
    a prediction when ``score`` is given."""
    from tcip_annotation.state import Annotation, BBox

    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2), score=score,
                      attributes=dict(attributes))


def saved_annotations(image) -> list:
    """The annotations of the label document of the image at ``image``."""
    from tcip_annotation.json_io import read_label_document

    return read_label_document(image_label_key(image)).annotations


def write_image(path, size: tuple[int, int] = (100, 80), color: tuple[int, int, int] = (0, 0, 0)):
    """A flat ``color`` three-band ``size`` image written at ``path``, its directory made;
    ``path``."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def gray_frame(where, size: int = 128, name: str = "img.png") -> str:
    """A flat gray ``size``px square ``name`` in ``where`` (:func:`write_image`); its path."""
    return str(write_image(where / name, (size, size), (120, 120, 120)))


def blank_image(root, name: str = "IMG_0001.JPG", size: tuple[int, int] = (100, 80)):
    """:func:`write_image` of ``name`` in ``root``'s undated capture."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    return write_image(root / "images" / UNDATED_BUCKET / name, size)


def one_labeled_capture(root):
    """``root`` holding one image in its ``2026-03-04`` capture with an empty label document over
    it, and the subject registry that decodes it; ``root``."""
    from tcip_mcp.subject_registry import Subject, SubjectRegistry

    image = root / "images" / "2026-03-04" / "a_1.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    label_image(image, [], 8, 8, keep_empty=True)
    registry_over(root, SubjectRegistry(subjects=(Subject(name="bud"),)))
    return root


def seed_labeled_images(images_dir, annotations, *, n: int, width: int, height: int):
    """Make ``images_dir``, a capture of a dataset image tree, and write ``n`` three-band
    ``width`` by ``height`` images ``img<i>.png``, each with a label document holding
    ``annotations``; ``images_dir``."""
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (width, height), color=(10 * i, 0, 0)).save(images_dir / f"img{i}.png")
        label_image(images_dir / f"img{i}.png", annotations, width, height)
    return images_dir


def seed_bud_images(images_dir, *, n: int = 3, size: int = 128,
                    box: tuple[float, float, float, float] = (10, 10, 40, 40)):
    """:func:`seed_labeled_images` of ``n`` square ``size``px images, each holding one ``bud``
    box at ``box``."""
    from tcip_annotation.state import Annotation, BBox

    return seed_labeled_images(images_dir, [Annotation(subject="bud", geometry=BBox(*box))],
                               n=n, width=size, height=size)


def small_detection_config(images_dir) -> dict:
    """A one-epoch CPU detection run of the bespoke detector over ``images_dir``, drawing its
    split at seed 0."""
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"min_size": 64, "max_size": 128},
                         "task": "detection"},
        "data": {"images_dir": str(images_dir), "scope": {"subject": "bud"},
                 "split": {"seed": 0, "val_ratio": 0.15}},
        "batch_size": 1, "stages": [{"freeze_to": -1, "epochs": 1}],
        "mixed_precision": False, "device": "cpu",
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


def fake_popen(monkeypatch, captured: list[list[str]]) -> None:
    """Replace ``subprocess.Popen`` with one recording each argv into ``captured`` and starting
    nothing, and ``launch_tensorboard`` with one starting no TensorBoard, so a launch door is
    driven through its own records without a child process."""
    import subprocess

    import tcip_mcp.tools.training_tools  # noqa: F401  # imported before Popen is replaced

    class _FakeProc:
        pid = 424242

    def _popen(argv, **kwargs):
        captured.append(argv)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})

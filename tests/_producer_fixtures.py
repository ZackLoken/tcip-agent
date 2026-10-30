"""Label documents written and loaders built through the platform's own producers.

The producers are ``label_queries.admit``, ``write_label_document`` and the one label save.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from tcip_mcp.dataset_layout import (
    UNDATED_BUCKET,
    Gestures,
    image_dir,
    image_root,
    label_key_of as image_label_key,
)
from tests._training_values import drawn_split

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import pytest
    from PIL import Image
    from torch.utils.data import Dataset

    from tcip_annotation.json_io import LabelDocument
    from tcip_annotation.state import Annotation, Attribute
    from tcip_mcp.model_registry import VerifiedCheckpoint
    from tcip_mcp.pipelines.data.augmentations import Transforms
    from tcip_mcp.pipelines.data.datasets import BespokeSource
    from tcip_mcp.pipelines.data.label_queries import Admission
    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_mcp.pipelines.data.split_construction import Partition
    from tcip_mcp.pipelines.model_build import SourceLayout
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.subject_registry import SubjectRegistry
    from tcip_mcp.tools.proposal_tools import (
        ProposedCandidates,
        StagedAssignments,
    )
    from tcip_store import Version

    Color = float | tuple[int, ...]
    """A paint color: one value for every band, or one value per band."""


def label_image(
    image_path: str | Path,
    annotations: Iterable[Annotation],
    width: int,
    height: int,
    **kwargs: object,
) -> Version | None:
    """Write ``annotations`` as the label document of the image at ``image_path``.

    The document is written by :func:`~tcip_annotation.json_io.write_label_document` at the
    image's own key. Returns the new version.
    """
    from tcip_annotation.json_io import write_label_document

    return write_label_document(
        image_label_key(image_path), annotations, width, height, **kwargs
    )


def box_annotation(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    subject: str = "bud",
    score: float | None = None,
    **attributes: str,
) -> Annotation:
    """Return an annotation of ``subject`` boxed at pixel ``(x1, y1, x2, y2)``.

    It carries ``attributes``, and is a prediction when ``score`` is given.
    """
    from tcip_annotation.state import Annotation, BBox

    return Annotation(
        subject=subject,
        geometry=BBox(x1, y1, x2, y2),
        score=score,
        attributes=dict(attributes),
    )


def saved_annotations(image: str | Path) -> list:
    """Return the annotations of the label document of the image at ``image``."""
    from tcip_annotation.json_io import read_label_document

    return read_label_document(image_label_key(image)).annotations


def write_image(
    path: Path,
    size: tuple[int, int] = (100, 80),
    color: tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """Save ``Image.new("RGB", size, color)`` at ``path``, its directory made.

    Returns ``path``.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def gray_frame(where: Path, size: int = 128, name: str = "img.png") -> str:
    """Write ``name`` in ``where``, ``size`` square at (120, 120, 120), and return its path.

    The image is written by :func:`write_image`.
    """
    return str(write_image(where / name, (size, size), (120, 120, 120)))


def write_jpegs(
    images_dir: Path, stems: Sequence[str], size: tuple[int, int]
) -> list[Path]:
    """Write a black ``<stem>.jpg`` of ``size`` into ``images_dir`` for each of ``stems``.

    Each is written by :func:`write_image`; returns their paths in ``stems``' order.
    """
    return [write_image(images_dir / f"{stem}.jpg", size) for stem in stems]


class WrittenDataset(NamedTuple):
    """A dataset root, the capture directory its images were written into and those images."""

    root: Path
    images_dir: Path
    images: list[Path]


class GroupedDataset(NamedTuple):
    """A dataset holding one grouped capture and one plain photo, with the paths written.

    ``root`` is the dataset root, ``images_dir`` the capture both sit in, ``manifest`` the
    group's ``.bandgroup`` manifest, ``bands`` its band files by name and ``plain`` the photo.
    """

    root: Path
    images_dir: Path
    manifest: Path
    bands: dict[str, Path]
    plain: Path


def write_band_group(
    images_dir: Path,
    stem: str,
    bands: dict[str, Path],
    *,
    central_wavelength_nm: dict[str, float] | None,
) -> Path:
    """Write the manifest grouping ``bands`` under ``stem`` in ``images_dir``; return its path.

    The record names each band's file, states ``central_wavelength_nm``, the explicit-manifest
    source and no identity tolerance, and is written by ``write_band_group_manifest``.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        BandGroupManifest,
        BandGroupRef,
        band_group_manifest_path,
        write_band_group_manifest,
    )

    return write_band_group_manifest(
        BandGroupRef(
            band_group_manifest_path(images_dir, stem),
            BandGroupManifest(
                bands={name: path.name for name, path in bands.items()},
                source="explicit-manifest",
                identity_tolerance={},
                central_wavelength_nm=central_wavelength_nm,
            ),
        )
    )


def stub_images(images_dir: Path, names: list[str]) -> list[Path]:
    """Write a one-byte file at each of ``names`` in ``images_dir``, its directory made.

    The files carry an image's name and no image; returns their paths in ``names``' order.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = [images_dir / name for name in names]
    for path in paths:
        path.write_bytes(b"x")
    return paths


def one_labeled_capture(root: Path) -> Path:
    """Write a three-byte JPEG stub ``a_1.jpg`` into ``root``'s ``2026-03-04`` capture.

    The stub has a kept empty label document over an 8 by 8 frame, and ``root``'s subject
    registry declares ``bud``. Returns ``root``.
    """
    image = image_dir(root, "2026-03-04") / "a_1.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    label_image(image, [], 8, 8, keep_empty=True)
    registry_over(root, registry_of("bud"))
    return root


BRIGHT = (230, 230, 230)
"""The color a painted object is bright in, 230 of 255 in each band."""


def painted_frame(
    width: int,
    height: int,
    background: Color,
    extents: Iterable[tuple[tuple[float, float, float, float], Color]],
    mode: str = "RGB",
) -> Image.Image:
    """Return a ``mode`` image of ``width`` by ``height`` in ``background`` with painted extents.

    Each of ``extents`` (a half-open pixel box ``(x1, y1, x2, y2)`` beside its color) is filled
    in its color.
    """
    from PIL import Image, ImageDraw

    frame = Image.new(mode, (width, height), background)
    draw = ImageDraw.Draw(frame)
    for (x1, y1, x2, y2), color in extents:
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=color)
    return frame


def painted_array(
    width: int,
    height: int,
    extents: Iterable[tuple[tuple[float, float, float, float], Color]],
    *,
    background: Color = 0,
    mode: str = "L",
) -> np.ndarray:
    """Return ``np.array`` of :func:`painted_frame` over ``extents``, ``background``, ``mode``."""
    import numpy as np

    return np.array(
        painted_frame(width, height, background, extents, mode=mode)
    )


def labeled_frame(
    path: Path,
    annotations: Sequence[Annotation],
    width: int,
    height: int,
    background: tuple[int, int, int],
    **kwargs: object,
) -> Path:
    """Write a labeled :func:`painted_frame` of ``background`` at ``path``.

    The frame is :data:`BRIGHT` over the extent of each of ``annotations`` that has one and is no
    crowd region, and labeled with ``annotations`` (:func:`label_image`, ``kwargs`` its own).
    Returns ``path``.
    """
    from tcip_annotation.state import Point, bbox_of

    path.parent.mkdir(parents=True, exist_ok=True)
    boxes = [
        bbox_of(a.geometry)
        for a in annotations
        if not isinstance(a.geometry, Point) and not a.iscrowd
    ]
    painted_frame(
        width,
        height,
        background,
        [((b.x1, b.y1, b.x2, b.y2), BRIGHT) for b in boxes],
    ).save(path)
    label_image(path, annotations, width, height, **kwargs)
    return path


def seed_labeled_images(
    images_dir: Path,
    annotations: Sequence[Annotation],
    *,
    n: int,
    width: int,
    height: int,
) -> Path:
    """Make ``images_dir``, a capture of a dataset image tree, and write ``n`` labeled frames.

    The frames are :func:`labeled_frame` images ``img<i>.png`` of ``width`` by ``height``, each
    holding ``annotations``. Returns ``images_dir``.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        labeled_frame(
            images_dir / f"img{i}.png",
            annotations,
            width,
            height,
            (10 * i, 0, 0),
        )
    return images_dir


def big_single_source(
    root: Path, width: int, height: int, *, polygons: bool = False
) -> tuple[Path, str]:
    """Write one ``width`` by ``height`` noise mosaic under ``root``'s undated capture.

    The mosaic is unseeded uniform noise below 0.3, labeled at its real size with 20 px
    ``bud`` boxes (``polygons``: squares) every 200 px from (20, 20).
    Returns its image directory and stem.
    """
    import torch
    from torchvision.utils import save_image

    from tcip_annotation.state import Annotation, BBox, Polygon

    images_dir = image_dir(root, UNDATED_BUCKET)
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "mosaic"
    save_image(
        torch.rand(3, height, width) * 0.3, str(images_dir / f"{stem}.png")
    )
    boxes = [
        Annotation(
            subject="bud",
            geometry=Polygon(
                rings=[[(x, y), (x + 20, y), (x + 20, y + 20), (x, y + 20)]]
            )
            if polygons
            else BBox(x, y, x + 20, y + 20),
        )
        for x in range(20, width - 20, 200)
        for y in range(20, height - 20, 200)
    ]
    label_image(images_dir / f"{stem}.png", boxes, width, height)
    return images_dir, stem


def seed_bud_images(
    images_dir: Path,
    *,
    n: int = 3,
    size: int = 128,
    box: tuple[float, float, float, float] = (10, 10, 40, 40),
) -> Path:
    """Write ``n`` square ``size``px images through :func:`seed_labeled_images`.

    Each holds one ``bud`` box at ``box``.
    """
    from tcip_annotation.state import Annotation, BBox

    return seed_labeled_images(
        images_dir,
        [Annotation(subject="bud", geometry=BBox(*box))],
        n=n,
        width=size,
        height=size,
    )


def small_detection_config(
    images_dir: str | Path,
    model_source: dict | None = None,
    *,
    project: Path,
    **data: object,
) -> dict:
    """Return a :func:`~tests._chain_fixtures.training_config` of ``project`` over ``images_dir``.

    It passes ``model_source`` (by default
    :data:`~tests._verified_checkpoint_fixtures.BUILT_DETECTOR`) and a data section of
    ``images_dir``, scope ``bud`` and a drawn split at validation ratio 0.15 grouped by tile
    prefix, with ``data``'s keys over it.
    """
    from tests._chain_fixtures import training_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    return training_config(
        project,
        model_source or BUILT_DETECTOR,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": "bud"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            **data,
        },
    )


def registry_of(subject: str, *attributes: Attribute) -> SubjectRegistry:
    """Return the subject registry declaring ``subject`` alone, with ``attributes`` on it."""
    from tcip_mcp.subject_registry import Subject, SubjectRegistry

    return SubjectRegistry(
        subjects=(
            Subject(name=subject, description="", attributes=attributes),
        )
    )


def registry_over(dataset_root: str | Path, registry: SubjectRegistry) -> None:
    """Replace ``dataset_root``'s subject registry by ``registry``.

    The replacement goes through the platform's own save
    (:func:`~tcip_mcp.subject_registry.replace_registry`).
    """
    from tcip_mcp.subject_registry import replace_registry

    replace_registry(
        dataset_root,
        registry,
        expect=None,
        allow_removals=True,
        allow_type_changes=True,
        actor=None,
    )


def scope_over(
    dataset_root: Path, registry: SubjectRegistry, subject: str
) -> ClassScope:
    """Return ``registry_scope`` of ``subject`` over ``dataset_root``'s image tree.

    ``dataset_root``'s registry is first replaced by ``registry`` (:func:`registry_over`).
    """
    from tcip_mcp.pipelines.data.label_queries import registry_scope

    registry_over(dataset_root, registry)
    return registry_scope(image_root(dataset_root), subject)


def attribute_scoped_dataset(root: Path) -> Path:
    """Write five ``leaf`` frames into ``root``'s capture ``2-11-26`` and return ``root``.

    ``root``'s registry declares ``leaf`` with a categorical ``condition`` over ``healthy`` and
    ``damaged``. Four frames' one instance is assessed for it and ``unassessed``'s is not; each
    is a 64 px :func:`labeled_frame`.
    """
    from tcip_annotation.state import Annotation, Attribute, BBox

    registry_over(
        root,
        registry_of(
            "leaf",
            Attribute("condition", "categorical", ("healthy", "damaged")),
        ),
    )
    images_dir = image_dir(root, "2-11-26")
    for stem, values in (
        ("assessed_a", {"condition": "healthy"}),
        ("assessed_b", {"condition": "damaged"}),
        ("assessed_c", {"condition": "healthy"}),
        ("assessed_d", {"condition": "damaged"}),
        ("unassessed", {}),
    ):
        labeled_frame(
            images_dir / f"{stem}.jpg",
            [
                Annotation(
                    subject="leaf",
                    geometry=BBox(4, 4, 20, 20),
                    attributes=values,
                )
            ],
            64,
            64,
            (100, 120, 90),
        )
    return root


def mark_complete(
    image_path: str | Path,
    subject: str,
    *,
    project: Path,
    rect: tuple[float, float, float, float] | None = None,
    proposals_hidden: bool = False,
    by: str = "user:tester",
) -> tuple[Version, LabelDocument, dict[int, int]]:
    """Mark ``subject`` complete over ``rect`` on the label document of the image at ``image_path``.

    The mark goes through ``dataset_layout.save_label_document`` as ``by``, with gestures of
    ``rect`` and ``proposals_hidden``, passing the document's stored annotations back as client
    annotations. Returns the new version.
    """
    from tcip_annotation.json_io import (
        ClientAnnotation,
        read_document_versioned,
    )

    doc, _version = read_document_versioned(image_label_key(image_path))
    return save_labels(
        project,
        image_path,
        [
            ClientAnnotation.of(a).model_dump(mode="json")
            for a in doc.annotations
        ],
        author=by,
        gestures=Gestures(
            complete={subject: True},
            rect=rect,
            proposals_hidden=proposals_hidden,
        ),
    )


def save_labels(
    project: Path,
    image: str | Path,
    payloads: list[dict],
    *,
    author: str = "user:breeder",
    expect: Version | None = None,
    gestures: Gestures = Gestures(),
) -> tuple[Version, LabelDocument, dict[int, int]]:
    """Save ``payloads`` as the label document of the image at ``image``, as ``author``.

    The save is ``dataset_layout.save_label_document``'s at the image's own dimensions, against
    ``expect`` and with ``gestures``. Returns its answer.
    """
    from tcip_mcp.dataset_layout import save_label_document
    from tcip_mcp.pipelines.image_utils import image_path_dimensions

    width, height = image_path_dimensions(image)
    return save_label_document(
        project,
        image_label_key(image),
        payloads,
        width=width,
        height=height,
        author=author,
        actor=author,
        expect=expect,
        gestures=gestures,
    )


def staged_box(project: Path, image: Path, box: list[float]) -> str:
    """Stage one ``bud`` box over the pixel ``box`` on ``image``; return its bucket's name.

    The box is proposed at 0.9 by the model ``detector`` through ``stage_proposals``, normalized
    by the image's own dimensions.
    """
    from tcip_mcp.pipelines.image_utils import image_path_dimensions
    from tcip_mcp.tools.proposal_tools import StagedShapes, stage_proposals

    width, height = image_path_dimensions(image)
    x1, y1, x2, y2 = box
    staged = stage_proposals(
        project,
        str(image),
        model_name="detector",
        actor=None,
        boxes=[
            {
                "subject": "bud",
                "conf": 0.9,
                "cx": (x1 + x2) / 2 / width,
                "cy": (y1 + y2) / 2 / height,
                "w": (x2 - x1) / width,
                "h": (y2 - y1) / height,
            }
        ],
    )
    assert isinstance(staged, StagedShapes) and staged.bucket, staged
    return staged.bucket


def install_stub_proposer(
    monkeypatch: pytest.MonkeyPatch, candidates: list[dict]
) -> None:
    """Install a proposal engine that hands ``candidates`` back verbatim, whatever the engine.

    The engine is installed by patching ``proposal.resolve_proposer``.
    """
    from tcip_mcp.pipelines import proposal

    class StubProposer:
        def propose(
            self, image_path: str, **params: object
        ) -> proposal.Proposal:
            return proposal.Proposal(candidates, None)

    monkeypatch.setattr(
        proposal, "resolve_proposer", lambda engine: StubProposer()
    )


def admit_over(
    images_dir: str | Path,
    ground_truth: str | Path | None = None,
    *,
    subject: str | None = None,
    members: list[str] | None = None,
) -> Admission:
    """Return the admission of ``images_dir`` under its registry's scope for ``subject``.

    ``label_queries.admit`` runs over ``images_dir``, ``ground_truth`` and ``members`` at
    ``registry_scope(images_dir, subject)``, and ``require_admitted`` refuses an empty
    admission.
    """
    from tcip_mcp.pipelines.data.label_queries import (
        admit,
        registry_scope,
        require_admitted,
    )

    admitted = admit(
        images_dir,
        ground_truth,
        scope=registry_scope(images_dir, subject),
        members=members,
    )
    require_admitted(admitted)
    return admitted


def checkpoint_admission(
    checkpoint: VerifiedCheckpoint,
    images_dir: str | Path,
    ground_truth: str | Path | None = None,
) -> Admission:
    """Return ``admit`` of ``images_dir``, ``ground_truth`` at the checkpoint's recorded scope."""
    from tcip_mcp.pipelines.data.label_queries import admit

    return admit(
        images_dir, ground_truth, scope=checkpoint.spec.data.recorded_scope
    )


def samples_over(
    images_dir: str | Path,
    ground_truth: str | Path | None = None,
    *,
    subject: str | None = None,
    members: list[str] | None = None,
) -> list[Sample]:
    """Return :func:`admit_over`'s ``every_sample()`` over ``images_dir``."""
    return admit_over(
        images_dir, ground_truth, subject=subject, members=members
    ).every_sample()


def run_over(
    task: str,
    images_dir: str | Path,
    ground_truth: str | Path | None = None,
    *,
    subject: str | None = None,
    members: list[str] | None = None,
    stated: dict[str, Any] | None = None,
    tiling: dict | None = None,
    dataset_source: BespokeSource | None = None,
    transforms: Transforms | None = None,
) -> tuple[Dataset, dict]:
    """Return a loader for ``task`` over one place holding ground truth, and its data section.

    The samples are the admission's ``every_sample()``. The data section is ``run_sizes`` over a
    :func:`drawn_split` block with ``stated`` (a band count, a class count), ``tiling`` (its
    ``data.tiling`` block) and ``dataset_source``'s source over it, given its lattice by
    ``split_construction._with_lattice``; ``dataset_source`` pairs a ``datasets.BespokeSource``
    with the layout it imports from. The loader is the training dataset
    ``split_construction.recorded_datasets`` builds at ``transforms``, and the data section is
    returned as its ``record()``.
    """
    from tcip_mcp.pipelines.data.split_construction import (
        _with_lattice,
        recorded_datasets,
        run_sizes,
    )
    from tcip_mcp.pipelines.schemas import DataSpec

    admitted = admit_over(
        images_dir, ground_truth, subject=subject, members=members
    )
    samples = admitted.every_sample()
    block = {
        "split": drawn_split(),
        **(stated or {}),
        **({} if tiling is None else {"tiling": tiling}),
        **(
            {}
            if dataset_source is None
            else {"dataset_source": dataset_source[0]}
        ),
    }
    data = _with_lattice(
        task,
        run_sizes(
            task, DataSpec.model_validate(block), admitted.scope, samples
        ),
        samples,
    )
    dataset, _val = recorded_datasets(
        task,
        data,
        samples,
        None,
        transforms,
        dataset_source and dataset_source[1],
    )
    return dataset, data.record()


def served_regions(dataset: Dataset) -> list:
    """Return the regions a stand-in detection ``dataset`` exposes, one per item it serves.

    Each region is the item's target's objects over its image's pixels
    (``datasets.object_region``).
    """
    from tcip_mcp.pipelines.data.datasets import object_region

    return [
        object_region(target, image.shape[-2] * image.shape[-1])
        for image, target in (dataset[i] for i in range(len(dataset)))
    ]


def staged_layout(project: str | Path, data: dict | DataSpec) -> SourceLayout:
    """Return ``model_build.staged_sources``'s layout for a config of ``project`` over ``data``.

    ``data`` is a data block (a dict, or a ``DataSpec``). The config's model source is
    ``_chain_fixtures.DETECTION_SOURCE`` normalized by its weights' statistics
    (``_chain_fixtures.weights_statistics``).
    """
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import (
        DETECTION_SOURCE,
        training_config,
        weights_statistics,
    )

    block = data if isinstance(data, dict) else data.record()
    return staged_sources(
        train_config(
            training_config(
                Path(project),
                weights_statistics(DETECTION_SOURCE),
                {"split": drawn_split(), **block},
            ),
            Path(project),
        ),
        Path(project),
    ).layout


def train_val(
    project: Path,
    task: str,
    data_cfg: dict,
    transforms: Transforms | None = None,
) -> tuple[Dataset, Dataset | None, Partition, DataSpec]:
    """Run ``split_construction.auto_train_val`` for a ``task`` run of ``project``.

    ``data_cfg`` is validated as a run's data block of ``project``, a dataset builder it declares
    imported from :func:`staged_layout`. Returns ``(train_ds, val_ds, partition, resolved)``.
    """
    from tcip_mcp.pipelines.data.split_construction import auto_train_val
    from tcip_mcp.pipelines.schemas import DataSpec

    data = DataSpec.model_validate(
        data_cfg, context={"project": Path(project)}
    )
    return auto_train_val(
        project, task, data, transforms, staged_layout(project, data_cfg)
    )


def dataset_over(
    task: str,
    images_dir: str | Path,
    ground_truth: str | Path | None = None,
    **kwargs: str | list[str] | dict | BespokeSource | Transforms | None,
) -> Dataset:
    """Return the loader :func:`run_over` builds."""
    return run_over(task, images_dir, ground_truth, **kwargs)[0]


def fake_popen(
    monkeypatch: pytest.MonkeyPatch, captured: list[list[str]]
) -> type:
    """Replace ``subprocess.Popen`` and ``launch_tensorboard`` with stand-ins starting nothing.

    The ``Popen`` stand-in records each argv into ``captured`` and starts no process, and the
    ``launch_tensorboard`` one is :func:`no_tensorboard`'s. Returns the ``Popen`` stand-in class,
    whose instances share one ``pid``.
    """
    import subprocess

    import tcip_mcp.tools.training_tools  # noqa: F401  # imported before Popen is replaced

    class _RecordedChild:
        pid = 424242

        def __init__(self, argv: list[str], **kwargs: object) -> None:
            captured.append(argv)

    monkeypatch.setattr(subprocess, "Popen", _RecordedChild)
    no_tensorboard(monkeypatch)
    return _RecordedChild


def no_tensorboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``launch_tensorboard`` with a stand-in that starts nothing and answers so.

    Its answer is the ``TensorboardFailure`` naming the directory it was handed.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    monkeypatch.setattr(
        tb,
        "launch_tensorboard",
        lambda directory: tb.TensorboardFailure(
            error=f"no TensorBoard starts under this test: {directory}",
            output=None,
        ),
    )


def kept(
    proposed: ProposedCandidates | dict, *assigned: tuple[int, str]
) -> StagedAssignments:
    """Return the assignments keeping each ``(candidate_id, subject)`` of the run ``proposed``.

    ``proposed`` is a ``propose_annotations`` answer, whose ``staging_id`` names the run; a
    refusal fails the test with its text.
    """
    from tcip_mcp.tools.proposal_tools import (
        Assignment,
        ProposedCandidates,
        StagedAssignments,
    )

    assert isinstance(proposed, ProposedCandidates), proposed
    assert proposed.staging_id is not None, proposed
    return StagedAssignments(
        staging_id=proposed.staging_id,
        candidates=[
            Assignment(candidate_id=candidate_id, subject=subject)
            for candidate_id, subject in assigned
        ],
    )


def mapping_of(
    rows: dict[str, list[tuple[str, tuple[str, str] | None, float | None]]],
) -> dict[str, list[plant_mapping.Assignment]]:
    """Return a plant mapping's per-date assignment records.

    ``rows`` maps each date to its ``(image, plant, distance_m)`` rows, ``image`` a file name of
    that date's capture. A ``plant`` ``(plant_id, accession_name)`` is assigned by capture
    sequence at ``distance_m``; ``None`` is an unmapped image, its ``distance_m`` the nearest
    plant's or ``None`` when no distance was measured.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import Assignment

    return {
        date: [
            Assignment(
                image=image,
                date_folder=date,
                plant_id=None if plant is None else plant[0],
                accession_name=None if plant is None else plant[1],
                source="unmapped" if plant is None else "sequence",
                distance_m=distance_m,
            )
            for image, plant, distance_m in assigned
        ]
        for date, assigned in rows.items()
    }

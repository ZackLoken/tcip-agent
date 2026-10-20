"""The correctness loop, ``check_model_contract`` and ``overfit_check``, reached through preflight.

``preflight_config(smoke=True)`` builds and smokes a model at the resolved dims and blocks a
broken builder, and ``TrainContext`` exposes the checks and craft primitives to a hand-rolled
``train(ctx)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET
from tests._chain_fixtures import (
    BESPOKE_CLASSIFIER,
    BESPOKE_MODELS,
    BESPOKE_SEMANTIC_SEG,
    FRAME_STATISTICS,
    training_config,
    weights_normalization,
    weights_statistics,
)
from tests._training_values import (
    drawn_split,
    evaluation_block,
    stage,
    tiling_block,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.data.datasets import BaseDataset  # noqa: E402
from tcip_mcp.pipelines.model_build import resolve_contract_dims  # noqa: E402
from tcip_mcp.pipelines.schemas import TrainGeometry  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PIL import Image
    from torch import Tensor, nn

    from tcip_mcp.pipelines.data.selection import ClassScope, Sample
    from tcip_mcp.pipelines.schemas import TrainConfigSchema


def _broken_builder(**kwargs: object) -> nn.Module:
    """Return an 'agent-written' model whose eval-mode forward returns a bare tensor.

    In training it returns a ``loss`` from one linear layer over random input.
    """
    import torch.nn as nn

    class _Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(
            self, images: list[Tensor], targets: list[dict] | None = None
        ) -> dict[str, Tensor] | Tensor:
            if self.training:
                return {"loss": self.lin(torch.rand(1, 4)).sum()}
            return torch.rand(
                3
            )  # not list[dict], violates the detection boundary

    return _Broken()


class _ValuesDataset(BaseDataset):
    """Four blank 32 px images, each target its index as a scalar value."""

    @property
    def num_samples(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor]]:
        return torch.zeros(3, 32, 32), {"values": torch.tensor(float(idx))}


def _bespoke_task_dataset(**_kwargs: object) -> _ValuesDataset:
    """Return an agent-authored dataset for a task the platform does not enumerate."""
    return _ValuesDataset()


def _strict_bespoke_dataset(
    samples: Sequence[Sample] | None = None,
    scope: ClassScope | None = None,
    transforms: Callable[[Image.Image, dict], tuple[Tensor, dict]]
    | None = None,
    task: str | None = None,
) -> _ValuesDataset:
    """Return a :class:`_ValuesDataset`, accepting the four named builder arguments.

    The helper's signature names ``samples``, ``scope``, ``transforms`` and ``task`` and has no
    ``**kwargs`` catch-all.
    """
    return _ValuesDataset()


def _unbuildable_dataset(**_kwargs: object) -> NoReturn:
    raise RuntimeError("cannot open the source for this task")


def _bespoke_classification_dataset(
    samples: Sequence[Sample] | None = None,
    scope: ClassScope | None = None,
    transforms: Callable[[Image.Image, dict], tuple[Tensor, dict]]
    | None = None,
    task: str | None = None,
) -> BaseDataset:
    """Return a dataset of ``len(samples)`` blank 32 px images, each labeled its index modulo 3."""

    class _DS(BaseDataset):
        @property
        def num_samples(self) -> int:
            return len(samples)

        def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, int]]:
            return torch.zeros(3, 32, 32), {"labels": idx % 3}

    return _DS()


def _admitted_tree(tmp_path: Path) -> Path:
    """Return an images directory whose label documents the platform's producer admits."""
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import labeled_frame, registry_over

    root = tmp_path / "ds"
    imgs = root / "images" / UNDATED_BUCKET
    imgs.mkdir(parents=True)
    registry_over(
        root, SubjectRegistry(subjects=(Subject(name="leaf", description=""),))
    )
    for stem in ("a", "b", "c", "d"):
        labeled_frame(
            imgs / f"{stem}.png",
            [Annotation(subject="leaf", geometry=BBox(2, 2, 10, 10))],
            64,
            64,
            (0, 0, 0),
            keep_empty=True,
        )
    return imgs


TASK_MODEL = f"{Path(__file__).stem}:_bespoke_task_model"
"""The ``model_source`` builder of :func:`_bespoke_task_model`."""


def _bespoke_task_model(**_kwargs: object) -> nn.Module:
    """Return an agent-authored model for the bespoke task, regressing each image's ``values``."""
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(3 * 32 * 32, 1)

        def forward(self, images: Tensor, targets: dict | None = None) -> dict:
            pred = self.head(images.flatten(1)).squeeze(-1)
            if self.training:
                return {"loss": ((pred - targets["values"]) ** 2).mean()}
            return {"values": pred}

    return _Net()


# --------------------------------------------------------------------------
# resolve_contract_dims: the size the run will actually see
# --------------------------------------------------------------------------


def _spec(task: str, data: dict) -> TrainConfigSchema:
    """Return a validated config of a ``task`` model of this repository over ``data``.

    The data section is a default ``drawn_split()`` with ``data`` merged over it, and the model
    source is ``unbuilt_source(task)`` normalized by pretrained weights.
    """
    from tcip_mcp.pipelines.schemas import train_config
    from tests import REPO_ROOT
    from tests._verified_checkpoint_fixtures import unbuilt_source

    return train_config(
        training_config(
            REPO_ROOT,
            weights_statistics(unbuilt_source(task)),
            {"split": drawn_split(), **data},
        )
    )


def test_resolve_contract_dims_reads_the_frame_the_run_resolved() -> None:
    """The contract's smoke frame is the frame the run resolved, and nothing when it resolved none.

    The smoke frame is the square tile a tiled run resolved, else the frame an untiled run's
    sources share, and nothing when the run resolved neither.
    """
    tiled = _spec("detection", {"tiling": tiling_block(tile_size=512)})
    untiled = _spec("detection", {"tiling": None})
    native = TrainGeometry(
        native_size=(96, 64), object_density=None, detector=None
    )
    unshared = TrainGeometry(
        native_size=None, object_density=None, detector=None
    )

    assert resolve_contract_dims(
        tiled, {"in_chans": 4, "num_classes": 5}, unshared
    ) == {
        "in_chans": 4,
        "num_classes": 5,
        "img_size": (512, 512),
    }
    assert resolve_contract_dims(untiled, {"in_chans": 3}, native)[
        "img_size"
    ] == (96, 64)
    assert "img_size" not in resolve_contract_dims(
        _spec("detection", {}), {"in_chans": 3}, unshared
    )


def test_model_dims_states_only_what_the_run_holds() -> None:
    """The model's dimensions state only what the run holds.

    A run that records no width refuses naming ``data.num_channels``; a run that
    holds no count states none; a rank count answers for an ordinal run under its own name, and
    img_size is the only value the contract resolver adds.
    """
    from tcip_mcp.pipelines.data.selection import ClassScope
    from tcip_mcp.pipelines.model_build import model_dims

    scope = ClassScope()
    frame = {"tiling": None}
    shared = TrainGeometry(
        native_size=(32, 32), object_density=None, detector=None
    )

    with pytest.raises(ValueError, match="data.num_channels"):
        model_dims(scope, {})
    detector = model_dims(scope, {"num_channels": 3})
    assert detector == {"in_chans": 3}
    assert resolve_contract_dims(
        _spec("detection", frame), detector, shared
    ) == {
        "in_chans": 3,
        "img_size": (32, 32),
    }  # a detector's synthetic box needs no count
    ordinal = model_dims(scope, {"num_channels": 3, "num_ranks": 4})
    assert ordinal == {"in_chans": 3, "num_ranks": 4}
    assert resolve_contract_dims(_spec("ordinal", frame), ordinal, shared) == {
        "in_chans": 3,
        "num_classes": 4,
        "img_size": (32, 32),
    }


def test_model_dims_hands_the_admitted_subject_and_every_attribute(
    tmp_path: Path,
) -> None:
    """A scoped run's model dimensions carry the admitted subject and every declared attribute."""
    from tcip_mcp.pipelines.data.label_queries import registry_scope
    from tcip_mcp.pipelines.model_build import model_dims
    from tcip_mcp.subject_registry import Attribute, Subject, SubjectRegistry
    from tests._producer_fixtures import registry_over

    attributes = (
        Attribute("color", "categorical", ("red", "blue")),
        Attribute("grade", "ordinal", ("low", "mid", "high")),
    )
    registry_over(
        tmp_path,
        SubjectRegistry(
            subjects=(
                Subject(name="bud", description="", attributes=attributes),
            )
        ),
    )
    scope = registry_scope(tmp_path / "images", "bud")

    assert model_dims(scope, {"num_channels": 3}) == {
        "in_chans": 3,
        "num_classes": 1,
        "attributes": attributes,
    }


# --------------------------------------------------------------------------
# preflight_config(smoke=True): builds + smokes, blocks a broken builder
# --------------------------------------------------------------------------


def test_preflight_smoke_blocks_broken_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = _admitted_tree(tmp_path)
    cfg = training_config(
        tmp_path,
        {
            "builder": f"{Path(__file__).stem}:_broken_builder",
            "source_files": [__file__],
            "task": "detection",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(imgs),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        batch_size=1,
        stages=[stage(0)],
    )
    # Fast path (no smoke) is structurally valid: the builder imports fine.
    assert preflight_config(tmp_path, cfg)["valid"] is True
    # Smoke path builds + runs the contract and catches the measurement-boundary violation.
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert r["valid"] is False
    assert any("model contract" in i for i in r["issues"])
    assert r["smoke"]["dims"]["img_size"] == (
        64,
        64,
    )  # the frame every source shares


def test_preflight_smokes_a_tiled_run_at_the_edge_its_resolution_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiled run stating no tile edge smokes at the edge its own resolution derives.

    The synthetic smoke batch has the run resolution's ``tile_size`` as its edge.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._producer_fixtures import seed_bud_images
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    imgs = seed_bud_images(
        tmp_path / "ds" / "images" / UNDATED_BUCKET,
        n=4,
        size=160,
        box=(10, 10, 26, 26),
    )
    cfg = training_config(
        tmp_path,
        BUILT_DETECTOR,
        {
            "images_dir": str(imgs),
            "scope": {"subject": "bud"},
            "tiling": tiling_block(sliver_frac=0.5),
            "split": drawn_split(val_ratio=0.25, grouping="tile_prefix"),
        },
        batch_size=1,
        stages=[stage(0)],
    )
    spec = train_config(cfg)
    edge = resolve_run(
        spec, staged_sources(spec, tmp_path).layout, project=tmp_path
    ).data.tiling.tile_size

    r = preflight_config(tmp_path, cfg, smoke=True)
    assert r["valid"] is True, r["issues"]
    assert r["smoke"]["batch_source"] == "synthetic"
    assert tuple(r["smoke"]["dims"]["img_size"]) == (edge, edge)


def test_preflight_smokes_an_untiled_run_of_mixed_frames_on_a_real_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An untiled run of mixed frame sizes smokes on one real batch off its own dataset.

    With frames of 64 by 64 and 48 by 40 px, the smoke reports a dataset batch and no
    synthesized dims.
    """
    monkeypatch.chdir(tmp_path)
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._chain_fixtures import stated_detector_geometry
    from tests._producer_fixtures import label_image
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    imgs = _admitted_tree(tmp_path)
    Image.new("RGB", (48, 40)).save(imgs / "e.png")
    label_image(
        imgs / "e.png",
        [Annotation(subject="leaf", geometry=BBox(2, 2, 10, 10))],
        48,
        40,
    )
    cfg = training_config(
        tmp_path,
        {**BUILT_DETECTOR, "detector_geometry": stated_detector_geometry()},
        {
            "images_dir": str(imgs),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        batch_size=1,
        stages=[stage(0)],
    )

    r = preflight_config(tmp_path, cfg, smoke=True)
    assert r["valid"] is True, r["issues"]
    assert (
        r["smoke"]["batch_source"] == "dataset" and r["smoke"]["dims"] is None
    )


def test_preflight_smoke_passes_valid_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR

    imgs = _admitted_tree(tmp_path)
    cfg = training_config(
        tmp_path,
        BUILT_DETECTOR,
        {
            "images_dir": str(imgs),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        batch_size=1,
        stages=[stage(0)],
    )
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
    assert r["valid"] is True, r["issues"]
    assert r["smoke"]["ok"] is True
    assert "overfit_check" in r  # voluntary diagnostic reported, non-gating


# --------------------------------------------------------------------------
# A task the contract has no synthetic schema for is smoked against a real batch:
# no task taxonomy, and no run launching with the contract silently skipped.
# --------------------------------------------------------------------------


def test_preflight_builds_and_smokes_at_the_count_the_run_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preflight builds and smokes the model at the class and band counts the run resolved.

    Over masks of three classes, the smoke dims carry three classes and three bands.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config
    from tests.test_mask_and_table_membership import _three_class_masks

    images_dir, masks_dir = _three_class_masks(tmp_path / "ds")
    cfg = training_config(
        tmp_path,
        {
            "builder": BESPOKE_SEMANTIC_SEG,
            "source_files": [BESPOKE_MODELS],
            "task": "semantic_seg",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        batch_size=1,
        stages=[stage(0)],
    )

    r = preflight_config(tmp_path, cfg, smoke=True)

    assert r["smoke"]["dims"]["num_classes"] == 3, r["smoke"]
    assert r["smoke"]["dims"]["in_chans"] == 3


def test_preflight_smokes_a_single_class_run_within_its_own_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-class run's synthetic smoke batch is shaped within the count the run resolved.

    An all-background mask run smokes a synthetic batch at one class and passes, and the model
    built at that count passes the contract over the run's own batch.
    """
    import numpy as np
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    images_dir, masks_dir = (
        tmp_path / "ds" / "images" / UNDATED_BUCKET,
        tmp_path / "ds" / "masks",
    )
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    for index, stem in enumerate(("a", "b", "c", "d")):
        shade = 80 + 10 * index
        Image.new("RGB", (32, 32), (shade, shade, shade)).save(
            images_dir / f"{stem}.png"
        )
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(
            masks_dir / f"{stem}.png"
        )
    cfg = training_config(
        tmp_path,
        {
            "builder": BESPOKE_SEMANTIC_SEG,
            "source_files": [BESPOKE_MODELS],
            "task": "semantic_seg",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(images_dir),
            "labels_dir": str(masks_dir),
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
        batch_size=1,
        stages=[stage(0)],
    )

    synthetic = preflight_config(tmp_path, cfg, smoke=True)

    assert synthetic["smoke"]["dims"]["num_classes"] == 1
    assert synthetic["smoke"]["batch_source"] == "synthetic"
    assert synthetic["smoke"]["ok"] is True, synthetic["smoke"]["issues"]
    # The same model over the run's own batch reaches the same verdict.
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import (
        build_from_model_source,
        staged_sources,
    )
    from tcip_mcp.pipelines.model_contract import check_model_contract
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import _one_real_batch

    spec = train_config(cfg)
    layout = staged_sources(spec, tmp_path).layout
    resolution = resolve_run(spec, layout, project=tmp_path)
    batch = _one_real_batch("semantic_seg", resolution.train_ds)
    assert not isinstance(batch, str), batch
    smoked = synthetic["smoke"]["dims"]
    model = build_from_model_source(
        spec.model_source,
        layout,
        {"in_chans": smoked["in_chans"], "num_classes": smoked["num_classes"]},
        resolution.geometry,
    )
    assert check_model_contract(
        model, "semantic_seg", device=torch.device("cpu"), sample_batch=batch
    )["ok"]


def test_preflight_smokes_bespoke_task_on_a_real_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bespoke task run smokes and overfit-checks on a real batch from its own dataset.

    The smoke reports a dataset batch and a train loss, and the overfit check passes.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = _admitted_tree(tmp_path)
    cfg = training_config(
        tmp_path,
        {
            "builder": TASK_MODEL,
            "source_files": [__file__],
            "task": "bunch_compactness",
            "normalization": FRAME_STATISTICS,
        },
        {
            "images_dir": str(imgs),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            "dataset_source": {
                "builder": f"{Path(__file__).stem}:_bespoke_task_dataset",
                "source_files": [__file__],
            },
        },
        stages=[stage(0)],
    )
    r = preflight_config(tmp_path, cfg, smoke=True, overfit=True)
    assert r["valid"] is True, r["issues"]
    # The contract actually ran: a real batch stood in for the missing synthetic schema.
    assert r["smoke"]["not_smokeable"] is None
    assert r["smoke"]["ok"] is True
    assert r["smoke"]["train_loss"] is not None
    assert (
        r["smoke"]["batch_source"] == "dataset"
    )  # provenance: which reference proved it
    # The same batch reaches overfit_check; re-synthesizing here would report a false "does not
    # learn" for exactly the bespoke tasks the real-batch path exists to serve.
    assert r["overfit_check"]["issue"] is None, r["overfit_check"]
    assert r["overfit_check"]["passed"] is True, r["overfit_check"]


def test_preflight_smoke_batch_matches_what_the_run_will_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The smoke batch comes from the dataset the run's own resolution builds.

    A bespoke builder whose signature names only the producer's own four keywords yields one
    batch off that dataset with no reason given.
    """
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.pipelines.data.split_construction import resolve_run
    from tcip_mcp.pipelines.model_build import staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tcip_mcp.tools.training_tools import _one_real_batch

    imgs = _admitted_tree(tmp_path)
    data = {
        "images_dir": str(imgs),
        "scope": {"subject": "leaf"},
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        "dataset_source": {
            "builder": f"{Path(__file__).stem}:_strict_bespoke_dataset",
            "source_files": [__file__],
        },
    }
    config = training_config(
        tmp_path,
        {
            "builder": TASK_MODEL,
            "source_files": [__file__],
            "task": "bunch_compactness",
            "normalization": FRAME_STATISTICS,
        },
        data,
    )
    spec = train_config(config)
    resolution = resolve_run(
        spec, staged_sources(spec, tmp_path).layout, project=tmp_path
    )

    batch = _one_real_batch("bunch_compactness", resolution.train_ds)
    assert not isinstance(batch, str), batch


def test_preflight_blocks_when_no_batch_can_be_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset that cannot produce an item blocks preflight, naming the builder's own reason."""
    monkeypatch.chdir(tmp_path)
    from tcip_mcp.tools.training_tools import preflight_config

    imgs = _admitted_tree(tmp_path)
    # Structurally valid, but the dataset cannot produce an item.
    cfg = training_config(
        tmp_path,
        {
            "builder": TASK_MODEL,
            "source_files": [__file__],
            "task": "bunch_compactness",
            "normalization": weights_normalization(),
        },
        {
            "images_dir": str(imgs),
            "scope": {"subject": "leaf"},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            "dataset_source": {
                "builder": f"{Path(__file__).stem}:_unbuildable_dataset",
                "source_files": [__file__],
            },
        },
        stages=[stage(0)],
    )
    r = preflight_config(tmp_path, cfg, smoke=True)
    assert r["valid"] is False
    # Only the builder raises this text, so the refusal is the unbuildable dataset's own.
    assert any("cannot open the source" in i for i in r["issues"]), r["issues"]


# --------------------------------------------------------------------------
# ctx surface: a custom loop self-proves + reuses the craft primitives
# --------------------------------------------------------------------------


def _table_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Write four 32 px frames and a table labeling them 0, 1, 2, 0 under ``tmp_path / "ds"``.

    Returns their images directory and the table.
    """
    from tests.tiny_trainer_fixtures import write_regression_dataset

    return write_regression_dataset(
        tmp_path / "ds",
        [0.1, 0.3, 0.5, 0.7],
        [0, 1, 2, 0],
        height=32,
        width=32,
    )


def _bespoke_classification_data(
    imgs: Path, table: Path, **sizes: int
) -> dict:
    """Return the data block of a run over ``imgs`` labeled by ``table``, drawing its own split.

    Its loaders are built by :func:`_bespoke_classification_dataset`, and the block states
    ``sizes``.
    """
    return {
        "images_dir": str(imgs),
        "labels_dir": str(table),
        **sizes,
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        "dataset_source": {
            "builder": f"{Path(__file__).stem}:_bespoke_classification_dataset",
            "source_files": [__file__],
        },
    }


def test_ctx_check_contract_and_overfit_check(tmp_path: Path) -> None:
    """The run context's contract and overfit checks build the model at what the run recorded.

    The run records three bands, three classes and the 32 px frame its untiled sources share.
    """
    from tests.tiny_trainer_fixtures import run_context

    imgs, table = _table_tree(tmp_path)
    ctx = run_context(
        tmp_path,
        training_config(
            tmp_path,
            {
                "builder": BESPOKE_CLASSIFIER,
                "source_files": [BESPOKE_MODELS],
                "task": "classification",
                "normalization": FRAME_STATISTICS,
            },
            {
                "images_dir": str(imgs),
                "labels_dir": str(table),
                "split": drawn_split(val_ratio=0.0),
            },
            evaluation=evaluation_block(selection_metric="loss"),
        ),
    )
    recorded = ctx.run.spec.data
    assert (
        recorded.num_channels,
        recorded.num_classes,
        ctx.run.geometry.native_size,
    ) == (3, 3, (32, 32))
    report = ctx.check_contract()
    assert report["ok"], report["issues"]
    # overfit is voluntary + non-gating; steps flow through as an override kwarg.
    over = ctx.overfit_check(steps=15)
    assert over["passed"], over["issue"]


def test_ctx_overfit_check_seeds_at_the_runs_own_seed_and_determinism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overfit diagnostic seeds at the seed and determinism its run's config states.

    It seeds once, at those values, and at no value of its own.
    """
    from tcip_mcp.pipelines.training import generic_trainer
    from tests.tiny_trainer_fixtures import run_context

    imgs, table = _table_tree(tmp_path)
    ctx = run_context(
        tmp_path,
        training_config(
            tmp_path,
            {
                "builder": BESPOKE_CLASSIFIER,
                "source_files": [BESPOKE_MODELS],
                "task": "classification",
                "normalization": FRAME_STATISTICS,
            },
            {
                "images_dir": str(imgs),
                "labels_dir": str(table),
                "split": drawn_split(val_ratio=0.0),
            },
            evaluation=evaluation_block(selection_metric="loss"),
            seed=13,
        ),
    )
    seeded: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        generic_trainer,
        "set_seed",
        lambda seed, *, deterministic: seeded.append((seed, deterministic)),
    )

    ctx.overfit_check(steps=2)

    assert seeded == [(13, ctx.run.spec.deterministic)]


def test_ctx_smokes_a_bespoke_dataset_run_at_the_count_its_data_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bespoke dataset run is smoked at the class count its data section states.

    The run records three bands beside the three classes its data section states, and the
    contract and overfit checks run without an issue.
    """
    from tests.tiny_trainer_fixtures import run_context

    monkeypatch.chdir(tmp_path)
    imgs, table = _table_tree(tmp_path)
    ctx = run_context(
        tmp_path,
        training_config(
            tmp_path,
            {
                "builder": BESPOKE_CLASSIFIER,
                "source_files": [BESPOKE_MODELS],
                "task": "classification",
                "normalization": FRAME_STATISTICS,
            },
            _bespoke_classification_data(imgs, table, num_classes=3),
            evaluation=evaluation_block(selection_metric="loss"),
        ),
    )
    resolved = ctx.run.spec.data
    assert (resolved.num_channels, resolved.num_classes) == (3, 3)

    report = ctx.check_contract()
    over = ctx.overfit_check(steps=3)

    assert report["not_smokeable"] is None, report
    assert report["ok"], report["issues"]
    assert report["train_loss"] is not None
    assert over["issue"] is None, over


def test_a_classification_record_stating_no_class_count_builds_no_model() -> (
    None
):
    """A classification record stating no class count is refused by name.

    ``recorded_model_dims`` refuses the record naming ``num_classes``.
    """
    from tcip_mcp.pipelines.model_build import recorded_model_dims

    with pytest.raises(ValueError, match="records no num_classes"):
        recorded_model_dims(
            _spec("classification", {"num_channels": 3, "scope": {}})
        )


def test_ctx_apply_stage_freeze_matches_trainer_guard(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.training.generic_trainer import apply_stage_freeze
    from tests._verified_checkpoint_fixtures import unbuilt_source
    from tests.tiny_trainer_fixtures import frames_data, run_context

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    ctx = run_context(
        tmp_path,
        training_config(
            tmp_path,
            unbuilt_source("regression"),
            frames_data(tmp_path / "ds", [0.1, 0.5], [0.2, 1.0]),
            evaluation=evaluation_block(selection_metric="loss"),
        ),
    )
    full = ctx.apply_stage_freeze(model, 0)
    assert full == sum(p.numel() for p in model.parameters())
    # A shrink relative to the previous stage violates the monotonic guard.
    with pytest.raises(RuntimeError, match="Non-decreasing unfreeze"):
        apply_stage_freeze(model, 0, prev_trainable=full + 1)

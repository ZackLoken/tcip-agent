"""``model_build``'s smoke contract, provenance snapshot, rebuild and handed dimensions.

The smoke contract's class count is the loader's, the provenance snapshot copies the module a
dotted reference names (not its top-level package), a saved checkpoint rebuilds from its own
config's model source, and the dimensions a builder is handed are the run's own.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET

torch = pytest.importorskip("torch")

from tcip_mcp import subject_registry  # noqa: E402
from tcip_mcp.pipelines.data.label_queries import registry_scope  # noqa: E402
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    CONFIG_KEY,
    GEOMETRY_KEY,
    SNAPSHOT_KEY,
    STATE_DICT_KEY,
    recorded_model_dims,
    resolve_contract_dims,
    snapshot_model_source,
)
from tcip_mcp.pipelines.schemas import (  # noqa: E402
    HANDED_TO_BUILDER,
    TrainGeometry,
    checked_train_config,
    train_config,
)
from tcip_mcp.pipelines.training.envelope import TrainContext
from tests import REPO_ROOT
from tests._chain_fixtures import (
    FRAME_STATISTICS,
    built_model,
    training_config,
    weights_statistics,
)
from tests._producer_fixtures import registry_over  # noqa: E402
from tests._training_values import (  # noqa: E402
    drawn_split,
    evaluation_block,
    tiling_block,
)

UNSHARED = TrainGeometry(native_size=None, object_density=None, detector=None)
"""A resolution whose training frames share no one size."""

WEIGHTS_NORMALIZATION = weights_statistics({})["normalization"]
"""The normalization record of the pretrained weights a probe model names: these tests read
the shapes a build takes, never the data its statistics would describe."""

PROBE_NET = f"{Path(__file__).stem}:build_probe_net"
"""The ``model_source`` builder of :func:`build_probe_net`."""
PROBE_FILES = [__file__]
"""The ``source_files`` a model source naming :data:`PROBE_NET` declares."""


def build_probe_net(
    *,
    image_mean: list[float],
    image_std: list[float],
    pretrained_weights=None,
    num_classes: int = 2,
    in_chans: int = 3,
    attributes: tuple = (),
):
    """Build a tiny module whose parameter shapes follow its builder kwargs.

    It has one head per attribute sized by its values, takes the statistics and weights it is
    handed, and loads none.
    """
    import torch.nn as nn

    class ProbeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(in_chans, 6, 3)
            self.head = nn.Conv2d(6, num_classes, 1)
            self.attribute_heads = nn.ModuleList(
                nn.Conv2d(6, len(a.values), 1) for a in attributes
            )

        def forward(self, images, targets=None):
            return self.head(self.stem(images))

    return ProbeNet()


def _param_shapes(model) -> dict:
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


def _write_registry(dataset_root: Path) -> None:
    """Write a registry with three subjects of different shape.

    'leaf' carries a three-value ordinal axis, 'bush' a two-value categorical one, and 'bud'
    none at all.
    """
    registry = subject_registry.SubjectRegistry(
        subjects=(
            subject_registry.Subject(
                name="leaf",
                attributes=(
                    subject_registry.Attribute(
                        name="condition",
                        type="ordinal",
                        values=("healthy", "mild", "severe"),
                    ),
                ),
            ),
            subject_registry.Subject(
                name="bush",
                attributes=(
                    subject_registry.Attribute(
                        name="vigor",
                        type="categorical",
                        values=("low", "high"),
                    ),
                ),
            ),
            subject_registry.Subject(name="bud"),
        )
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    registry_over(dataset_root, registry)


def _agent_package(root: Path, name: str, modules: dict) -> Path:
    """Write an empty ``__init__.py`` and each of ``modules`` under ``root / name``; return it."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for mod_name, body in modules.items():
        (pkg / f"{mod_name}.py").write_text(body, encoding="utf-8")
    importlib.invalidate_caches()
    return pkg


def test_contract_dims_take_the_admitted_attributes_without_the_loader_background_offset(
    tmp_path,
):
    """Contract dims take the admitted attributes without the loader's background offset.

    A scoped detection config smokes at the one subject and the two-value attribute the run was
    admitted under, with no background class added, after its registry has gained a third
    condition value.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import admit_over, label_image

    dataset_root = tmp_path / "currant_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    registry_over(
        dataset_root,
        subject_registry.SubjectRegistry(
            subjects=(
                subject_registry.Subject(
                    name="leaf",
                    attributes=(
                        subject_registry.Attribute(
                            name="condition",
                            type="ordinal",
                            values=("healthy", "mild"),
                        ),
                    ),
                ),
            )
        ),
    )
    for stem, condition in (("leaf_a", "healthy"), ("leaf_b", "mild")):
        Image.new("RGB", (64, 64)).save(images_dir / f"{stem}.png")
        label_image(
            images_dir / f"{stem}.png",
            [
                Annotation(
                    subject="leaf",
                    geometry=BBox(8, 8, 24, 24),
                    attributes={"condition": condition},
                )
            ],
            64,
            64,
        )

    scope = admit_over(images_dir, subject="leaf").scope
    assert [len(a.values) for a in scope.attributes] == [
        2
    ]  # the head this run trains
    spec = train_config(
        training_config(
            tmp_path,
            {
                "builder": PROBE_NET,
                "builder_kwargs": {},
                "task": "detection",
                "normalization": WEIGHTS_NORMALIZATION,
            },
            {
                "scope": asdict(scope),
                "num_channels": 5,
                "images_dir": str(images_dir),
                "tiling": tiling_block(tile_size=640),
                "split": drawn_split(),
            },
        )
    )

    _write_registry(dataset_root)  # a third condition value declared since
    assert len(registry_scope(images_dir, "leaf").attributes[0].values) == 3

    dims = resolve_contract_dims(spec, recorded_model_dims(spec), UNSHARED)

    assert dims == {
        "in_chans": 5,
        "num_classes": 1,
        "attributes": scope.attributes,
        "img_size": (640, 640),
    }


def test_contract_dims_count_only_the_subject_for_a_scope_declaring_no_attributes(
    tmp_path,
):
    """Contract dims count the subject as their one class for a scope declaring no attributes.

    An instance_seg scope whose subject declares no attribute resolves one class and no
    attributes, and a run resolving no frame carries no frame size.
    """
    dataset_root = tmp_path / "subject_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _write_registry(dataset_root)

    spec = train_config(
        training_config(
            tmp_path,
            {
                "builder": PROBE_NET,
                "builder_kwargs": {},
                "task": "instance_seg",
                "normalization": WEIGHTS_NORMALIZATION,
            },
            {
                "scope": asdict(registry_scope(images_dir, "bud")),
                "num_channels": 3,
                "images_dir": str(images_dir),
                "split": drawn_split(),
            },
        )
    )

    dims = resolve_contract_dims(spec, recorded_model_dims(spec), UNSHARED)

    assert dims == {"in_chans": 3, "num_classes": 1}


def test_snapshot_copies_each_dotted_module_at_its_module_path(tmp_path):
    """The snapshot copies each dotted module at its module path.

    Each of the three bespoke seams' modules, declared under its source's ``source_files`` and
    lying outside the project, is copied byte for byte at its path under its import root.
    """
    pkg = _agent_package(
        tmp_path,
        "agent_code_seams",
        {
            "nets": "def build_net(**kwargs):\n    return None\n",
            "loops": "def train(ctx):\n    return {}\n",
            "sources": "def build_ds(**kwargs):\n    return None\n",
        },
    )

    declared = [
        pkg / "nets.py",
        pkg / "loops.py",
        pkg / "sources.py",
        pkg / "__init__.py",
    ]
    spec = train_config(
        training_config(
            tmp_path / "project",
            {
                "builder": "agent_code_seams.nets:build_net",
                "source_files": [
                    str(declared[0]),
                    str(declared[1]),
                    str(declared[3]),
                ],
                "task": "detection",
                "normalization": WEIGHTS_NORMALIZATION,
            },
            {
                "dataset_source": {
                    "builder": "agent_code_seams.sources:build_ds",
                    "source_files": [str(declared[2])],
                }
            },
            training_source="agent_code_seams.loops:train",
        )
    )
    record, copies = snapshot_model_source(spec, tmp_path / "project")

    for src in declared:
        copy = record["files"][str(src.resolve())]["file"]
        assert copy == f"model_src/agent_code_seams/{src.name}"
        assert copies[copy] == src.read_bytes()


def test_snapshot_captures_the_module_of_a_builder_spelled_without_a_colon(
    tmp_path,
):
    """The snapshot captures the module of a builder spelled without a colon.

    For ``agent_code_dotted.detectors.build_net`` the snapshot copies
    ``agent_code_dotted/detectors.py``, the segments ahead of the function name.
    """
    pkg = _agent_package(
        tmp_path,
        "agent_code_dotted",
        {
            "detectors": "def build_net(**kwargs):\n    return None\n",
        },
    )

    record, _copies = snapshot_model_source(
        train_config(
            training_config(
                tmp_path / "project",
                {
                    "builder": "agent_code_dotted.detectors.build_net",
                    "task": "detection",
                    "normalization": WEIGHTS_NORMALIZATION,
                    "source_files": [
                        str(pkg / "detectors.py"),
                        str(pkg / "__init__.py"),
                    ],
                },
                {"split": drawn_split()},
            )
        ),
        tmp_path / "project",
    )

    assert record["files"][str((pkg / "detectors.py").resolve())]["file"] == (
        "model_src/agent_code_dotted/detectors.py"
    )


def test_a_missing_or_empty_builder_refuses_at_the_configs_validation():
    """A missing or empty builder refuses at the config's validation.

    ``builder`` of ``None`` and of ``""`` both refuse where the config is validated, naming
    ``model_source.builder``; a named builder admits.
    """
    for builder in (None, ""):
        _spec, issues = checked_train_config(
            training_config(
                REPO_ROOT,
                {
                    "builder": builder,
                    "task": "detection",
                    "normalization": WEIGHTS_NORMALIZATION,
                },
                {"split": drawn_split()},
            )
        )
        assert any(
            issue.startswith("model_source.builder") for issue in issues
        ), issues

    assert (
        checked_train_config(
            training_config(
                REPO_ROOT,
                {
                    "builder": PROBE_NET,
                    "task": "detection",
                    "normalization": WEIGHTS_NORMALIZATION,
                },
                {"split": drawn_split()},
            )
        )[1]
        == []
    )


def _probe_config() -> dict:
    """Return a classification run's config under the repository root.

    Its data section states five channels, seven classes and the empty scope.
    """
    return training_config(
        REPO_ROOT,
        {
            "builder": PROBE_NET,
            "source_files": PROBE_FILES,
            "task": "classification",
            "normalization": WEIGHTS_NORMALIZATION,
        },
        {"num_channels": 5, "num_classes": 7, "scope": {}},
    )


@pytest.mark.parametrize("restated", HANDED_TO_BUILDER)
def test_builder_kwargs_restating_a_handed_value_refuses_by_name(restated):
    """Builder kwargs restating a handed value refuse by name.

    A builder_kwargs carrying any name the build hands its builder refuses naming it where the
    config is validated, the rank count included though this run resolves none.
    """
    config = _probe_config()
    config["model_source"]["builder_kwargs"] = {restated: 4}

    issues = checked_train_config(config)[1]
    assert any(restated in issue for issue in issues), issues


def handed_kwargs(**kwargs) -> dict:
    """Return the keyword values the builder was handed."""
    return kwargs


def test_the_names_a_build_hands_are_the_names_a_restatement_refuses():
    """The names a build hands its builder are the names a restatement refuses.

    A build stating weights, their statistics and a detector geometry hands its builder a
    subset of :data:`HANDED_TO_BUILDER`, lacking ``num_ranks`` and ``attributes`` alone.
    """
    from tcip_mcp.pipelines.derivations import pretrained_normalization
    from tcip_mcp.pipelines.schemas import PretrainedWeights
    from tests._chain_fixtures import detector_geometry

    weights = {"library": "timm", "name": "resnet18.a1_in1k"}
    config = training_config(
        REPO_ROOT,
        {
            "builder": f"{Path(__file__).stem}:handed_kwargs",
            "source_files": PROBE_FILES,
            "task": "classification",
            "normalization": pretrained_normalization(
                PretrainedWeights(**weights)
            ).model_dump(),
        },
        {"num_channels": 3, "num_classes": 7, "scope": {}},
    )

    handed = built_model(config, detector_geometry())

    assert set(HANDED_TO_BUILDER) - set(handed) == {"num_ranks", "attributes"}
    assert set(handed) <= set(HANDED_TO_BUILDER)


def test_a_model_source_stating_its_own_width_refuses_by_name():
    """A width stated on the model source itself refuses by name at validation."""
    config = _probe_config()
    config["model_source"]["in_chans"] = 4

    with pytest.raises(ValueError, match="in_chans"):
        built_model(config)


def test_an_ordinal_run_recording_no_rank_count_refuses_by_name():
    """An ordinal run recording no rank count refuses at ``recorded_model_dims`` by name."""
    config = training_config(
        REPO_ROOT,
        {
            "builder": PROBE_NET,
            "task": "ordinal",
            "normalization": WEIGHTS_NORMALIZATION,
        },
        {"num_channels": 3, "scope": {}},
    )

    with pytest.raises(ValueError, match="records no num_ranks"):
        recorded_model_dims(train_config(config))


def test_a_run_over_label_documents_recording_a_count_refuses():
    """A run over label documents recording a class count refuses.

    A detection run whose scope names a subject and records ``num_classes`` beside it refuses
    naming ``num_classes``.
    """
    config = training_config(
        REPO_ROOT,
        {
            "builder": PROBE_NET,
            "task": "detection",
            "normalization": WEIGHTS_NORMALIZATION,
        },
        {
            "num_channels": 3,
            "num_classes": 5,
            "scope": {"subject": "bud", "attributes": []},
        },
    )

    with pytest.raises(ValueError, match=r"records \['num_classes'\]"):
        recorded_model_dims(train_config(config))


def test_a_run_builds_at_the_width_and_heads_its_admitted_data_records(
    tmp_path,
):
    """A run builds at the width and heads its admitted data records.

    A run whose data section the platform wrote from its own admission builds a model reading
    one band, one subject class and one three-value attribute head, each differing from
    ``build_probe_net``'s defaults.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import label_image, run_over

    dataset_root = tmp_path / "hazel_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _write_registry(dataset_root)
    for stem, condition in (
        ("leaf_a", "healthy"),
        ("leaf_b", "mild"),
        ("leaf_c", "severe"),
    ):
        Image.new("L", (64, 64)).save(images_dir / f"{stem}.png")
        label_image(
            images_dir / f"{stem}.png",
            [
                Annotation(
                    subject="leaf",
                    geometry=BBox(8, 8, 24, 24),
                    attributes={"condition": condition},
                )
            ],
            64,
            64,
        )
    _dataset, data = run_over("detection", images_dir, subject="leaf")
    config = training_config(
        tmp_path,
        {
            "builder": PROBE_NET,
            "source_files": PROBE_FILES,
            "task": "detection",
            "normalization": WEIGHTS_NORMALIZATION,
        },
        data,
    )

    shapes = _param_shapes(built_model(config))

    assert shapes["stem.weight"][1] == data["num_channels"] == 1
    assert shapes["head.weight"][0] == 1
    assert (
        shapes["attribute_heads.0.weight"][0]
        == len(data["scope"]["attributes"][0]["values"])
        == 3
    )


def test_a_saved_checkpoint_rebuilds_the_architecture_its_config_builds(
    tmp_path,
):
    """A saved checkpoint rebuilds the architecture its config builds.

    The checkpoint the envelope saves carries no model source of its own, and the config it
    records rebuilds, from the run's snapshot, a model whose parameter shapes match the one the
    run's context built and saved.
    """
    from tcip_mcp.experiments import CONFIG_PATHS, observe
    from tcip_mcp.pipelines.model_build import (
        SNAPSHOT_DIR,
        build_from_model_source,
        owning_run_layout,
    )
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.registry_paths import runtime_paths
    from tests._verified_checkpoint_fixtures import (
        fixture_data_dir,
        opened_run,
        table_images,
    )

    config = training_config(
        tmp_path,
        {
            "builder": PROBE_NET,
            "source_files": PROBE_FILES,
            "task": "classification",
            "normalization": FRAME_STATISTICS,
        },
        {
            **table_images(fixture_data_dir(tmp_path, "probe"), n=4),
            "split": drawn_split(val_ratio=0.25, grouping="tile_prefix"),
        },
        evaluation=evaluation_block(selection_metric="loss"),
    )
    ctx = TrainContext(
        run=observed_run(observe(opened_run(tmp_path, config))),
        train_loader=None,
    )
    trained = ctx.build_model()
    assert _param_shapes(trained)["head.weight"] == (
        2,
        6,
        1,
        1,
    )  # the recorded count took effect
    path = ctx.save_checkpoint(
        {STATE_DICT_KEY: trained.state_dict()}, "model_best"
    )

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert (
        "model_source" not in loaded
    )  # the config is the one place the model source lives
    spec = train_config(
        runtime_paths(loaded[CONFIG_KEY], CONFIG_PATHS, tmp_path)
    )
    layout = owning_run_layout(spec, loaded[SNAPSHOT_KEY], tmp_path)
    assert layout.root == ctx.run_dir / SNAPSHOT_DIR
    rebuilt = build_from_model_source(
        spec.model_source,
        layout,
        recorded_model_dims(spec),
        TrainGeometry.model_validate(loaded[GEOMETRY_KEY]),
    )
    assert _param_shapes(rebuilt) == _param_shapes(trained)


def test_child_pythonpath_carries_sys_path_and_the_existing_env_value(
    tmp_path, monkeypatch
):
    """A child's PYTHONPATH carries a prepended ``sys.path`` entry and ends with the old value.

    With a directory prepended to ``sys.path`` and ``PYTHONPATH`` already set, the child's
    path includes that directory and ends with the existing ``PYTHONPATH`` value.
    """
    import os

    from tcip_mcp.pipelines.model_build import child_pythonpath

    extra_dir = str(tmp_path / "bespoke_src")
    monkeypatch.syspath_prepend(extra_dir)
    monkeypatch.setenv("PYTHONPATH", "/already/set/path")

    result = child_pythonpath()
    entries = result.split(os.pathsep)

    assert extra_dir in entries
    assert entries[-1] == "/already/set/path"

"""``model_build``: the smoke contract's class count is the loader's, the provenance snapshot copies
the module a dotted reference names (not its top-level package), a saved checkpoint rebuilds from
its own config's model source, and the dimensions a builder is handed are the run's own.
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import importlib
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp import subject_registry  # noqa: E402
from tcip_mcp.pipelines.data.label_queries import registry_scope  # noqa: E402
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    CONFIG_KEY,
    SNAPSHOT_KEY,
    STATE_DICT_KEY,
    recorded_model_dims,
    resolve_contract_dims,
    snapshot_model_source,
)
from tcip_mcp.pipelines.schemas import checked_train_config, train_config  # noqa: E402
from tcip_mcp.pipelines.training.envelope import TrainContext
from tests._chain_fixtures import built_model, training_config
from tests._producer_fixtures import registry_over  # noqa: E402

PROBE_NET = f"{Path(__file__).stem}:build_probe_net"
"""The ``model_source`` builder of :func:`build_probe_net`."""
PROBE_FILES = [__file__]
"""The ``source_files`` a model source naming :data:`PROBE_NET` declares."""


def build_probe_net(*, num_classes: int = 2, in_chans: int = 3, attributes: tuple = ()):
    """A tiny module whose parameter shapes follow its builder kwargs, one head per attribute
    sized by its values. Its forward is never run here; these tests read parameter shapes
    only."""
    import torch.nn as nn

    class ProbeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(in_chans, 6, 3)
            self.head = nn.Conv2d(6, num_classes, 1)
            self.attribute_heads = nn.ModuleList(
                nn.Conv2d(6, len(a.values), 1) for a in attributes)

        def forward(self, images, targets=None):
            return self.head(self.stem(images))

    return ProbeNet()


def _param_shapes(model) -> dict:
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


def _write_registry(dataset_root: Path) -> None:
    """A registry with three subjects of different shape, so no scope's class count is guessable
    from the file's overall size: 'leaf' carries a three-value ordinal axis, 'bush' a two-value
    categorical one, and 'bud' none at all."""
    registry = subject_registry.SubjectRegistry(subjects=(
        subject_registry.Subject(
            name="leaf",
            attributes=(subject_registry.Attribute(
                name="condition", type="ordinal", values=("healthy", "mild", "severe")),)),
        subject_registry.Subject(
            name="bush",
            attributes=(subject_registry.Attribute(
                name="vigor", type="categorical", values=("low", "high")),)),
        subject_registry.Subject(name="bud"),
    ))
    dataset_root.mkdir(parents=True, exist_ok=True)
    registry_over(dataset_root, registry)


def _agent_package(root: Path, name: str, modules: dict) -> Path:
    """Write an importable package of agent-written modules and return its directory."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for mod_name, body in modules.items():
        (pkg / f"{mod_name}.py").write_text(body, encoding="utf-8")
    importlib.invalidate_caches()
    return pkg


def test_contract_dims_take_the_admitted_attributes_without_the_loader_background_offset(tmp_path):
    """A scoped detection config smokes at the subject count and the attributes the run was
    admitted under, with no background class added: the +1 is the loader's own offset on the
    labels it builds, so applying it here too would prove the model against a head one class
    wider than the one that trains.

    The attributes are the scope the run was admitted under, not a reading of the registry as it
    stands now: this run was admitted when its subject declared two condition values, the registry
    has since gained a third, and a re-resolution would smoke a head one value wider than the head
    that trains."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import admit_over, label_image

    dataset_root = tmp_path / "currant_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    registry_over(dataset_root, subject_registry.SubjectRegistry(
        subjects=(subject_registry.Subject(
            name="leaf", attributes=(subject_registry.Attribute(
                name="condition", type="ordinal", values=("healthy", "mild")),)),)))
    for stem, condition in (("leaf_a", "healthy"), ("leaf_b", "mild")):
        Image.new("RGB", (64, 64)).save(images_dir / f"{stem}.png")
        label_image(images_dir / f"{stem}.png",
                    [Annotation(subject="leaf", geometry=BBox(8, 8, 24, 24),
                                attributes={"condition": condition})], 64, 64)

    scope = admit_over(images_dir, subject="leaf").scope
    assert [len(a.values) for a in scope.attributes] == [2]  # the head this run trains
    spec = train_config(training_config(
        {"builder": PROBE_NET, "builder_kwargs": {}, "task": "detection"},
        {"scope": asdict(scope), "num_channels": 5, "images_dir": str(images_dir),
         "tiling": {"enabled": True, "tile_size": 640}}))

    _write_registry(dataset_root)  # a third condition value declared since
    assert len(registry_scope(images_dir, "leaf").attributes[0].values) == 3

    dims = resolve_contract_dims(spec, recorded_model_dims(spec))

    assert dims == {"in_chans": 5, "num_classes": 1, "attributes": scope.attributes,
                    "img_size": (640, 640)}


def test_contract_dims_count_only_the_subject_for_a_scope_declaring_no_attributes(tmp_path):
    """An instance_seg scope whose subject declares no attribute trains one class, the subject
    itself, and hands no attributes. The resolved count stays at that one class rather than
    gaining a background slot; a run resolving no frame carries no smoke frame."""
    dataset_root = tmp_path / "subject_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _write_registry(dataset_root)

    spec = train_config(training_config(
        {"builder": PROBE_NET, "builder_kwargs": {}, "task": "instance_seg"},
        {"scope": asdict(registry_scope(images_dir, "bud")), "num_channels": 3,
         "images_dir": str(images_dir)}))

    dims = resolve_contract_dims(spec, recorded_model_dims(spec))

    assert dims == {"in_chans": 3, "num_classes": 1}


def test_snapshot_copies_each_dotted_module_at_its_module_path(tmp_path):
    """Each of the three bespoke seams' modules, declared under its source's ``source_files`` and
    lying outside the project, is copied at its path under its import root, the module its own
    reference names, so the copy binds as that module and never as its top-level package."""
    pkg = _agent_package(tmp_path, "agent_code_seams", {
        "nets": "def build_net(**kwargs):\n    return None\n",
        "loops": "def train(ctx):\n    return {}\n",
        "sources": "def build_ds(**kwargs):\n    return None\n",
    })

    declared = [pkg / "nets.py", pkg / "loops.py", pkg / "sources.py", pkg / "__init__.py"]
    spec = train_config(training_config(
        {"builder": "agent_code_seams.nets:build_net",
         "source_files": [str(declared[0]), str(declared[1]), str(declared[3])],
         "task": "detection"},
        {"dataset_source": {"builder": "agent_code_seams.sources:build_ds",
                            "source_files": [str(declared[2])]}},
        training_source="agent_code_seams.loops:train"))
    record, copies = snapshot_model_source(spec, tmp_path / "project")

    for src in declared:
        copy = record["files"][str(src.resolve())]["file"]
        assert copy == f"model_src/agent_code_seams/{src.name}"
        assert copies[copy] == src.read_bytes()


def test_snapshot_captures_the_module_of_a_builder_spelled_without_a_colon(tmp_path):
    """``module.path.function`` is the other accepted builder spelling; the function name is the
    last segment, so the module is everything before it, not the first segment."""
    pkg = _agent_package(tmp_path, "agent_code_dotted", {
        "detectors": "def build_net(**kwargs):\n    return None\n",
    })

    record, _copies = snapshot_model_source(train_config(training_config(
        {"builder": "agent_code_dotted.detectors.build_net", "task": "detection",
         "source_files": [str(pkg / "detectors.py"), str(pkg / "__init__.py")]}, {})),
        tmp_path / "project")

    assert record["files"][str((pkg / "detectors.py").resolve())]["file"] == (
        "model_src/agent_code_dotted/detectors.py")


def test_a_missing_or_empty_builder_refuses_at_the_configs_validation():
    """``builder`` of ``None`` and of ``""`` both refuse where the config is validated, naming
    ``model_source.builder``, before anything reads the model source; a named builder admits."""
    for builder in (None, ""):
        _spec, issues = checked_train_config(
            training_config({"builder": builder, "task": "detection"}, {}))
        assert any(issue.startswith("model_source.builder") for issue in issues), issues

    assert checked_train_config(
        training_config({"builder": PROBE_NET, "task": "detection"}, {}))[1] == []


def _probe_config() -> dict:
    """A classification run's recorded config: table ground truth records the empty scope."""
    return training_config({"builder": PROBE_NET, "source_files": PROBE_FILES,
                            "task": "classification"},
                           {"num_channels": 5, "num_classes": 7, "scope": {}})


@pytest.mark.parametrize("restated", ["in_chans", "num_classes", "num_ranks", "attributes"])
def test_builder_kwargs_restating_a_dimension_refuses_by_name(restated):
    """The band count and the one count reach the builder from the run's data section alone; a
    builder_kwargs carrying any dimension refuses naming it before any build, the rank count
    included though this run resolves none."""
    config = _probe_config()
    config["model_source"]["builder_kwargs"] = {restated: 4}

    with pytest.raises(ValueError, match=restated):
        built_model(config)


def test_a_model_source_stating_its_own_width_refuses_by_name():
    """A width stated on the model source itself refuses by name where the config is
    validated."""
    config = _probe_config()
    config["model_source"]["in_chans"] = 4

    with pytest.raises(ValueError, match="in_chans"):
        built_model(config)


def test_an_ordinal_run_recording_no_rank_count_refuses_by_name():
    """A checkpoint's rank count is read at the recorded-dimension boundary, which names what is
    missing rather than failing on a bare key."""
    config = training_config({"builder": PROBE_NET, "task": "ordinal"},
                             {"num_channels": 3, "scope": {}})

    with pytest.raises(ValueError, match="records no num_ranks"):
        recorded_model_dims(train_config(config))


def test_a_run_over_label_documents_recording_a_count_refuses():
    """A run over label documents is sized by its scope, the subject and its attributes; a count
    recorded beside it would be a second size, so the dimensions refuse rather than choose."""
    config = training_config({"builder": PROBE_NET, "task": "detection"},
                             {"num_channels": 3, "num_classes": 5,
                              "scope": {"subject": "bud", "attributes": []}})

    with pytest.raises(ValueError, match=r"records \['num_classes'\]"):
        recorded_model_dims(train_config(config))


def test_a_run_builds_at_the_width_and_heads_its_admitted_data_records(tmp_path):
    """The admitting case, through the producer: a run whose data section the platform wrote from
    its own admission builds a model reading that run's band count, its one subject and one head
    per attribute its registry declares. The single-band sources, the one subject and the
    three-value attribute all differ from ``build_probe_net``'s defaults, so a build that dropped
    any of them would come out at the default shape."""
    from PIL import Image
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import label_image, run_over

    dataset_root = tmp_path / "hazel_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _write_registry(dataset_root)
    for stem, condition in (("leaf_a", "healthy"), ("leaf_b", "mild"), ("leaf_c", "severe")):
        Image.new("L", (64, 64)).save(images_dir / f"{stem}.png")
        label_image(images_dir / f"{stem}.png",
                    [Annotation(subject="leaf", geometry=BBox(8, 8, 24, 24),
                                attributes={"condition": condition})], 64, 64)
    _dataset, data = run_over("detection", images_dir, subject="leaf")
    config = training_config({"builder": PROBE_NET, "source_files": PROBE_FILES,
                              "task": "detection"}, data)

    shapes = _param_shapes(built_model(config))

    assert shapes["stem.weight"][1] == data["num_channels"] == 1
    assert shapes["head.weight"][0] == 1
    assert shapes["attribute_heads.0.weight"][0] == len(
        data["scope"]["attributes"][0]["values"]) == 3


def test_a_saved_checkpoint_rebuilds_the_architecture_its_config_builds(tmp_path):
    """The checkpoint written by the envelope's save path carries enough of the model source for
    the inference-side rebuild to reconstruct the same architecture the run trained. Both sides
    are produced here by the real build path, so a stamp that records less than the builder was
    called with shows up as a shape difference rather than passing on a restated literal."""
    from tcip_mcp.experiments import CONFIG_PATHS, observe
    from tcip_mcp.pipelines.model_build import (
        SNAPSHOT_DIR, build_from_model_source, owning_run_layout,
    )
    from tcip_mcp.pipelines.training.run_registry import observed_run
    from tcip_mcp.registry_paths import runtime_paths
    from tests._verified_checkpoint_fixtures import fixture_data_dir, opened_run, table_images

    config = training_config({"builder": PROBE_NET, "source_files": PROBE_FILES,
                              "task": "classification"},
                             {**table_images(fixture_data_dir(tmp_path, "probe"), n=4),
                              "split": {"seed": 0, "val_ratio": 0.25}})
    ctx = TrainContext(run=observed_run(observe(opened_run(tmp_path, config))),
                       train_loader=None)
    trained = ctx.build_model()
    assert _param_shapes(trained)["head.weight"] == (2, 6, 1, 1)  # the recorded count took effect
    path = ctx.save_checkpoint({STATE_DICT_KEY: trained.state_dict()}, "model_best")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_source" not in loaded  # the config is the one place the model source lives
    spec = train_config(runtime_paths(loaded[CONFIG_KEY], CONFIG_PATHS, tmp_path))
    layout = owning_run_layout(spec, loaded[SNAPSHOT_KEY], tmp_path)
    assert layout.root == ctx.run_dir / SNAPSHOT_DIR
    rebuilt = build_from_model_source(spec.model_source, layout, recorded_model_dims(spec))
    assert _param_shapes(rebuilt) == _param_shapes(trained)


def test_child_pythonpath_carries_sys_path_and_the_existing_env_value(tmp_path, monkeypatch):
    """The string a spawned process or Ray worker gets must reproduce this interpreter's own
    import search path, with any existing PYTHONPATH the caller already set preserved at the end
    rather than displaced by it."""
    import os

    from tcip_mcp.pipelines.model_build import child_pythonpath

    extra_dir = str(tmp_path / "bespoke_src")
    monkeypatch.syspath_prepend(extra_dir)
    monkeypatch.setenv("PYTHONPATH", "/already/set/path")

    result = child_pythonpath()
    entries = result.split(os.pathsep)

    assert extra_dir in entries
    assert entries[-1] == "/already/set/path"

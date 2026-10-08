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
    STATE_DICT_KEY,
    build_model,
    recorded_model_dims,
    resolve_contract_dims,
    snapshot_model_source,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tests._producer_fixtures import registry_over  # noqa: E402
from tests.tiny_trainer_fixtures import trainer_run  # noqa: E402

PROBE_NET = f"{__name__}:build_probe_net"
"""The ``model_source`` builder of :func:`build_probe_net`."""


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
    cfg = {
        "model_source": {"builder_kwargs": {}, "task": "detection"},
        "data": {"scope": asdict(scope), "num_channels": 5, "images_dir": str(images_dir),
                 "tiling": {"enabled": True, "tile_size": 640}},
    }

    _write_registry(dataset_root)  # a third condition value declared since
    assert len(registry_scope(images_dir, "leaf").attributes[0].values) == 3

    dims = resolve_contract_dims(cfg, "detection", recorded_model_dims(cfg))

    assert dims == {"in_chans": 5, "num_classes": 1, "attributes": scope.attributes,
                    "img_size": 640}


def test_contract_dims_count_only_the_subject_for_a_scope_declaring_no_attributes(tmp_path):
    """An instance_seg scope whose subject declares no attribute trains one class, the subject
    itself, and hands no attributes. The resolved count stays at that one class rather than
    gaining a background slot."""
    dataset_root = tmp_path / "subject_2026"
    images_dir = dataset_root / "images" / UNDATED_BUCKET
    images_dir.mkdir(parents=True)
    _write_registry(dataset_root)

    cfg = {
        "model_source": {"builder_kwargs": {}, "task": "instance_seg"},
        "data": {"scope": asdict(registry_scope(images_dir, "bud")), "num_channels": 3,
                 "images_dir": str(images_dir)},
    }

    dims = resolve_contract_dims(cfg, "instance_seg", recorded_model_dims(cfg))

    assert dims == {"in_chans": 3, "num_classes": 1, "img_size": 224}


def test_snapshot_captures_each_dotted_module_not_its_top_level_package(tmp_path, monkeypatch):
    """Each of the three bespoke seams resolves to the module its own reference names. A snapshot
    that resolved the package instead would record the package's ``__init__`` as the run's code:
    a manifest that looks complete (nothing missing, no errors) while holding none of the agent's
    builder, loop, or dataset source."""
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = _agent_package(tmp_path, "agent_code_seams", {
        "nets": "def build_net(**kwargs):\n    return None\n",
        "loops": "def train(ctx):\n    return {}\n",
        "sources": "def build_ds(**kwargs):\n    return None\n",
    })

    config = {
        "model_source": {"builder": "agent_code_seams.nets:build_net"},
        "training_source": "agent_code_seams.loops:train",
        "data": {"dataset_source": {"builder": "agent_code_seams.sources:build_ds"}},
    }
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    manifest = snapshot_model_source(config, exp_dir)

    captured = {Path(e["src"]).resolve() for e in manifest["files"]}
    assert captured == {(pkg / "nets.py").resolve(), (pkg / "loops.py").resolve(),
                        (pkg / "sources.py").resolve()}
    assert (pkg / "__init__.py").resolve() not in captured
    assert manifest["missing"] == []
    assert manifest["snapshot_errors"] == []
    for entry in manifest["files"]:
        copied = exp_dir / "model_src" / entry["file"]
        assert copied.read_bytes() == Path(entry["src"]).read_bytes()


def test_snapshot_captures_the_module_of_a_builder_spelled_without_a_colon(tmp_path, monkeypatch):
    """``module.path.function`` is the other accepted builder spelling; the function name is the
    last segment, so the module is everything before it, not the first segment."""
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = _agent_package(tmp_path, "agent_code_dotted", {
        "detectors": "def build_net(**kwargs):\n    return None\n",
    })

    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    manifest = snapshot_model_source(
        {"model_source": {"builder": "agent_code_dotted.detectors.build_net"}}, exp_dir)

    captured = {Path(e["src"]).resolve() for e in manifest["files"]}
    assert captured == {(pkg / "detectors.py").resolve()}
    assert manifest["snapshot_errors"] == []


def test_a_missing_or_empty_builder_refuses_through_the_one_callee_message():
    """``builder`` of ``None`` and of ``""`` both reach ``build_model`` -> ``_import_dotted``,
    the one refusal site for a non-string or empty builder, and refuse with its one message
    rather than two different messages from a duplicated caller-side check."""
    for builder in (None, ""):
        with pytest.raises(ValueError, match="non-empty 'module:function' string"):
            build_model({"model_source": {"builder": builder}}, {"in_chans": 3})


def _probe_config() -> dict:
    """A classification run's recorded config: table ground truth records the empty scope."""
    return {
        "model_source": {"builder": PROBE_NET, "task": "classification"},
        "data": {"num_channels": 5, "num_classes": 7, "scope": {}},
        "device": "cpu",
    }


@pytest.mark.parametrize("restated", ["in_chans", "num_classes", "num_ranks", "attributes"])
def test_builder_kwargs_restating_a_dimension_refuses_by_name(restated):
    """The band count and the one count reach the builder from the run's data section alone; a
    builder_kwargs carrying any dimension refuses naming it before any build, the rank count
    included though this run resolves none."""
    config = _probe_config()
    config["model_source"]["builder_kwargs"] = {restated: 4}

    with pytest.raises(ValueError, match=restated):
        build_model(config, recorded_model_dims(config))


def test_a_model_source_stating_its_own_width_refuses_at_the_build():
    """A width stated on the model source itself refuses at the build, by name."""
    config = _probe_config()
    config["model_source"]["in_chans"] = 4

    with pytest.raises(ValueError, match="in_chans"):
        build_model(config, recorded_model_dims(config))


def test_an_ordinal_run_recording_no_rank_count_refuses_by_name():
    """A checkpoint's rank count is read at the recorded-dimension boundary, which names what is
    missing rather than failing on a bare key."""
    config = {"model_source": {"builder": PROBE_NET, "task": "ordinal"},
              "data": {"num_channels": 3, "scope": {}}}

    with pytest.raises(ValueError, match="records no num_ranks"):
        recorded_model_dims(config)


def test_a_run_over_label_documents_recording_a_count_refuses():
    """A run over label documents is sized by its scope, the subject and its attributes; a count
    recorded beside it would be a second size, so the dimensions refuse rather than choose."""
    config = {"model_source": {"builder": PROBE_NET, "task": "detection"},
              "data": {"num_channels": 3, "num_classes": 5,
                       "scope": {"subject": "bud", "attributes": []}}}

    with pytest.raises(ValueError, match=r"records \['num_classes'\]"):
        recorded_model_dims(config)


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
    config = {"model_source": {"builder": PROBE_NET, "task": "detection"},
              "data": data}

    shapes = _param_shapes(build_model(config, recorded_model_dims(config)))

    assert shapes["stem.weight"][1] == data["num_channels"] == 1
    assert shapes["head.weight"][0] == 1
    assert shapes["attribute_heads.0.weight"][0] == len(
        data["scope"]["attributes"][0]["values"]) == 3


def test_a_saved_checkpoint_rebuilds_the_architecture_its_config_builds(tmp_path):
    """The checkpoint written by the envelope's save path carries enough of the model source for
    the inference-side rebuild to reconstruct the same architecture the run trained. Both sides
    are produced here by the real build path, so a stamp that records less than the builder was
    called with shows up as a shape difference rather than passing on a restated literal."""
    config = _probe_config()
    trained = build_model(config, recorded_model_dims(config))
    assert _param_shapes(trained)["head.weight"] == (7, 6, 1, 1)  # the recorded count took effect

    (tmp_path / "out").mkdir()
    from tests._chain_fixtures import training_config

    ctx = TrainContext(run=trainer_run(training_config(config["model_source"], config["data"]),
                                       tmp_path / "out",
                                       project=tmp_path,
                                       has_val_loader=True, id="auto-run-40"),
                       train_loader=None)
    path = ctx.save_checkpoint({STATE_DICT_KEY: trained.state_dict()}, "model_best")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_source" not in loaded  # the config is the one place the model source lives
    rebuilt = build_model(loaded[CONFIG_KEY], recorded_model_dims(loaded[CONFIG_KEY]))
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

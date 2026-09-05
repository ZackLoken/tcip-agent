"""What ``model_build`` must get right for a run to stay reproducible and provable.

Three standing promises of the module, each checked against the real collaborator rather than a
restated literal: the smoke contract resolves the class count the loader will actually train at
(no second background offset), the provenance snapshot copies the module a dotted reference names
(not its top-level package), and the checkpoint stamp records the source that produced the weights
(a value the caller already placed in the payload is never replaced by the config's).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp import class_registry  # noqa: E402
from tcip_mcp.dataset_layout import classes_path  # noqa: E402
from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map  # noqa: E402
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE, detect_kind  # noqa: E402
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    build_model,
    resolve_contract_dims,
    snapshot_model_source,
    stamp_model_ref,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tcip_mcp.pipelines.training.run_registry import create_run  # noqa: E402


def build_probe_net(*, num_classes: int = 2, in_chans: int = 3):
    """A tiny module whose parameter shapes follow its builder kwargs.

    Both kwargs differ from the defaults in every config below, so a rebuild that silently loses
    them produces different shapes rather than the same model twice. Its forward is never run
    here; these tests read parameter shapes only.
    """
    import torch.nn as nn

    class ProbeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(in_chans, 6, 3)
            self.head = nn.Conv2d(6, num_classes, 1)

        def forward(self, images, targets=None):
            return self.head(self.stem(images))

    return ProbeNet()


def _param_shapes(model) -> dict:
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


def _write_registry(dataset_root: Path) -> None:
    """A registry with three subjects of different shape, so no scope's class count is guessable
    from the file's overall size: 'leaf' carries a three-value ordinal axis, 'bush' a two-value
    categorical one, and 'bud' none at all."""
    registry = class_registry.ClassRegistry(subjects=(
        class_registry.Subject(
            name="leaf",
            attributes=(class_registry.Attribute(
                name="condition", type="ordinal", values=("healthy", "mild", "severe")),)),
        class_registry.Subject(
            name="bush",
            attributes=(class_registry.Attribute(
                name="vigor", type="categorical", values=("low", "high")),)),
        class_registry.Subject(name="bud"),
    ))
    dataset_root.mkdir(parents=True, exist_ok=True)
    class_registry.write_registry(classes_path(dataset_root), registry)


def _agent_package(root: Path, name: str, modules: dict) -> Path:
    """Write an importable package of agent-written modules and return its directory."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for mod_name, body in modules.items():
        (pkg / f"{mod_name}.py").write_text(body, encoding="utf-8")
    importlib.invalidate_caches()
    return pkg


def test_contract_dims_take_the_registry_count_without_the_loader_background_offset(tmp_path):
    """A registry-scoped detection config smokes at the class count the loader derives from the
    same map, with no background class added: the +1 is the loader's own offset on the labels it
    builds, so applying it here too would prove the model against a head one class wider than the
    one that trains."""
    dataset_root = tmp_path / "currant_2026"
    labels_dir = dataset_root / "annotations"
    labels_dir.mkdir(parents=True)
    _write_registry(dataset_root)

    cfg = {
        "model_source": {"builder_kwargs": {"num_classes": 9, "in_chans": 5}},
        "data": {"subject": "leaf", "attribute": "condition", "labels_dir": str(labels_dir),
                 "tiling": {"enabled": True, "tile_size": 640}},
    }
    dims = resolve_contract_dims(cfg, "detection")

    _registry, id_map = resolve_registry_id_map(str(labels_dir), "leaf", "condition")
    assert len(id_map) == 3  # the three condition values this registry declares
    assert dims == {"in_chans": 5, "num_classes": len(id_map), "img_size": 640}
    assert dims["num_classes"] != cfg["model_source"]["builder_kwargs"]["num_classes"]


def test_contract_dims_count_only_the_subject_for_a_single_class_scope(tmp_path):
    """An instance_seg scope with no attribute trains one class, the subject itself. The resolved
    count stays at that one class rather than gaining a background slot."""
    dataset_root = tmp_path / "chestnut_2026"
    labels_dir = dataset_root / "annotations"
    labels_dir.mkdir(parents=True)
    _write_registry(dataset_root)

    cfg = {
        "model_source": {"builder_kwargs": {"num_classes": 9}},
        "data": {"subject": "bud", "labels_dir": str(labels_dir)},
    }
    dims = resolve_contract_dims(cfg, "instance_seg")

    _registry, id_map = resolve_registry_id_map(str(labels_dir), "bud", None)
    assert len(id_map) == 1
    assert dims["num_classes"] == len(id_map)


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


def test_a_model_source_the_caller_placed_in_the_payload_is_not_replaced_by_the_config():
    """A hand-rolled loop that built its model from one source and saves through the envelope must
    keep that source on the checkpoint. Overwriting it with the config's would record a builder
    that never produced these weights, so a rebuild at inference would load them into a different
    architecture."""
    caller_source = {"builder": "agent_code.nets:build_wide_detector",
                     "builder_kwargs": {"num_classes": 7, "in_chans": 5}, "task": "detection"}
    config_source = {"builder": "agent_code.nets:build_narrow_detector",
                     "builder_kwargs": {"num_classes": 2, "in_chans": 3}, "task": "instance_seg"}

    payload = stamp_model_ref(
        {"model_state_dict": {}, "model_source": caller_source},
        {"model_source": config_source})

    assert payload["model_source"] == caller_source
    assert payload["model_source"]["builder"] != config_source["builder"]
    assert payload["kind"] == KIND_TCIP_MODULE


def test_a_missing_or_empty_builder_refuses_through_the_one_callee_message():
    """``builder`` of ``None`` and of ``""`` both reach ``build_model`` -> ``_import_dotted``,
    the one refusal site for a non-string or empty builder, and refuse with its one message
    rather than two different messages from a duplicated caller-side check."""
    for builder in (None, ""):
        with pytest.raises(ValueError, match="non-empty 'module:function' string"):
            build_model({"model_source": {"builder": builder}})


def _probe_config() -> dict:
    return {
        "model_source": {"builder": f"{__name__}:build_probe_net",
                         "builder_kwargs": {"num_classes": 7, "in_chans": 5},
                         "task": "detection", "in_chans": 5},
        "device": "cpu",
    }


def test_a_saved_checkpoint_rebuilds_the_architecture_its_config_builds(tmp_path):
    """The checkpoint written by the envelope's save path carries enough of the model source for
    the inference-side rebuild to reconstruct the same architecture the run trained. Both sides
    are produced here by the real build path, so a stamp that records less than the builder was
    called with shows up as a shape difference rather than passing on a restated literal."""
    config = _probe_config()
    trained = build_model(config)
    assert _param_shapes(trained)["head.weight"] == (7, 6, 1, 1)  # the config's kwargs took effect

    ctx = TrainContext(run=create_run(dict(config), str(tmp_path / "out")), train_loader=None)
    path = ctx.save_checkpoint({"model_state_dict": trained.state_dict()}, "model_best")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = build_model(loaded)
    assert _param_shapes(rebuilt) == _param_shapes(trained)


def test_a_saved_checkpoint_is_recognized_after_its_kind_stamp_is_dropped(tmp_path):
    """The structural fallback recognizes a tcip checkpoint from the markers the save path itself
    writes, so it keeps working for a checkpoint whose kind was never stamped. Feeding it the real
    writer's output, rather than a hand-built dict, is what ties the two ends together."""
    config = _probe_config()
    ctx = TrainContext(run=create_run(dict(config), str(tmp_path / "out")), train_loader=None)
    path = ctx.save_checkpoint(
        {"model_state_dict": build_model(config).state_dict()}, "model_best")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload.pop("kind") == KIND_TCIP_MODULE
    unstamped = tmp_path / "unstamped.pt"
    torch.save(payload, unstamped)

    assert detect_kind(str(unstamped)) == KIND_TCIP_MODULE


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

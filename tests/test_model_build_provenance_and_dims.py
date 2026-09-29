"""``model_build``: the smoke contract's class count is the loader's, the provenance snapshot copies
the module a dotted reference names (not its top-level package), a saved checkpoint rebuilds from
its own config's model source, and the dimensions a builder is handed are the run's own.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tcip_mcp import subject_registry  # noqa: E402
from tcip_mcp.dataset_layout import subjects_path  # noqa: E402
from tcip_mcp.pipelines.data.label_queries import resolve_registry_id_map  # noqa: E402
from tcip_mcp.pipelines.data.selection import ClassScope  # noqa: E402
from tcip_mcp.pipelines.inference.predictor import KIND_TCIP_MODULE, detect_kind  # noqa: E402
from tcip_mcp.pipelines.model_build import (  # noqa: E402
    build_model,
    recorded_model_dims,
    resolve_contract_dims,
    snapshot_model_source,
    stamp_model_ref,
)
from tcip_mcp.pipelines.training.envelope import TrainContext  # noqa: E402
from tests.tiny_trainer_fixtures import trainer_run  # noqa: E402


def build_probe_net(*, num_classes: int = 2, in_chans: int = 3):
    """A tiny module whose parameter shapes follow its builder kwargs. Its forward is never run
    here; these tests read parameter shapes only."""
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
    subject_registry.write_registry(subjects_path(dataset_root), registry)


def _agent_package(root: Path, name: str, modules: dict) -> Path:
    """Write an importable package of agent-written modules and return its directory."""
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for mod_name, body in modules.items():
        (pkg / f"{mod_name}.py").write_text(body, encoding="utf-8")
    importlib.invalidate_caches()
    return pkg


def test_contract_dims_take_the_admitted_count_without_the_loader_background_offset(tmp_path):
    """A scoped detection config smokes at the class count the run was admitted under, with no
    background class added: the +1 is the loader's own offset on the labels it builds, so applying
    it here too would prove the model against a head one class wider than the one that trains.

    The count is the scope the run was admitted under, not a reading of the registry as it stands
    now: this run was admitted when its subject declared two condition values, the registry has
    since gained a third, and a re-resolution would smoke it one class wider than the head that
    trains."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import admit_over

    dataset_root = tmp_path / "currant_2026"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    subject_registry.write_registry(subjects_path(dataset_root), subject_registry.SubjectRegistry(
        subjects=(subject_registry.Subject(
            name="leaf", attributes=(subject_registry.Attribute(
                name="condition", type="ordinal", values=("healthy", "mild")),)),)))
    for stem, condition in (("leaf_a", "healthy"), ("leaf_b", "mild")):
        Image.new("RGB", (64, 64)).save(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="leaf", geometry=BBox(8, 8, 24, 24),
                        attributes={"condition": condition})], 64, 64)

    scope = admit_over(images_dir, labels_dir, subject="leaf", attribute="condition").scope
    assert scope.id_map is not None and len(scope.id_map) == 2  # the class space this run trains over
    cfg = {
        "model_source": {"builder_kwargs": {}, "task": "detection"},
        "data": {"scope": asdict(scope), "num_channels": 5, "labels_dir": str(labels_dir),
                 "tiling": {"enabled": True, "tile_size": 640}},
    }

    _write_registry(dataset_root)  # a third condition value declared since
    _registry, id_map = resolve_registry_id_map(
        str(labels_dir), ClassScope(subject="leaf", attribute="condition"))
    assert len(id_map) == 3

    dims = resolve_contract_dims(cfg, "detection", recorded_model_dims(cfg))

    assert dims == {"in_chans": 5, "num_classes": 2, "img_size": 640}
    assert dims["num_classes"] != len(id_map)


def test_contract_dims_count_only_the_subject_for_a_single_class_scope(tmp_path):
    """An instance_seg scope with no attribute trains one class, the subject itself. The resolved
    count stays at that one class rather than gaining a background slot."""
    dataset_root = tmp_path / "chestnut_2026"
    labels_dir = dataset_root / "annotations"
    labels_dir.mkdir(parents=True)
    _write_registry(dataset_root)

    _registry, id_map = resolve_registry_id_map(str(labels_dir), ClassScope(subject="bud"))
    assert len(id_map) == 1
    cfg = {
        "model_source": {"builder_kwargs": {}, "task": "instance_seg"},
        "data": {"scope": {"subject": "bud", "id_map": id_map}, "num_channels": 3,
                 "labels_dir": str(labels_dir)},
    }

    dims = resolve_contract_dims(cfg, "instance_seg", recorded_model_dims(cfg))

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


def test_the_stamp_reads_the_model_source_off_the_payloads_own_config():
    """The checkpoint's config is the one place its model source lives: the stamp names the kind
    from it and writes no second copy beside it."""
    config = {"model_source": {"builder": "agent_code.nets:build_detector", "task": "detection"}}

    payload = stamp_model_ref({"model_state_dict": {}, "config": config})

    assert payload["kind"] == KIND_TCIP_MODULE
    assert "model_source" not in payload


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
        "model_source": {"builder": f"{__name__}:build_probe_net", "task": "classification"},
        "data": {"num_channels": 5, "num_classes": 7, "scope": {}},
        "device": "cpu",
    }


@pytest.mark.parametrize("restated", ["in_chans", "num_classes", "num_ranks"])
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
    config = {"model_source": {"builder": f"{__name__}:build_probe_net", "task": "ordinal"},
              "data": {"num_channels": 3, "scope": {}}}

    with pytest.raises(ValueError, match="records no num_ranks"):
        recorded_model_dims(config)


def test_a_scoped_run_recording_a_second_count_refuses():
    """A scoped run's class count is its map's length; a count recorded beside the map would be a
    second one, so the dimensions refuse rather than choose."""
    config = {"model_source": {"builder": f"{__name__}:build_probe_net", "task": "detection"},
              "data": {"num_channels": 3, "num_classes": 5,
                       "scope": {"subject": "bud", "id_map": {"bud": 0}}}}

    with pytest.raises(ValueError, match="two counts"):
        recorded_model_dims(config)


def test_a_run_builds_at_the_width_and_count_its_admitted_data_records(tmp_path):
    """The admitting case, through the producer: a run whose data section the platform wrote from
    its own admission builds a model reading that run's band count and scoring its class map. The
    single-band sources and the three-value map both differ from ``build_probe_net``'s defaults,
    so a build that dropped either would come out at the default shape."""
    from PIL import Image
    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox

    from tests._producer_fixtures import run_over

    dataset_root = tmp_path / "hazel_2026"
    images_dir, labels_dir = dataset_root / "images", dataset_root / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    _write_registry(dataset_root)
    for stem, condition in (("leaf_a", "healthy"), ("leaf_b", "mild"), ("leaf_c", "severe")):
        Image.new("L", (64, 64)).save(images_dir / f"{stem}.png")
        json_io.write_annotations(
            str(labels_dir / f"{stem}.json"),
            [Annotation(subject="leaf", geometry=BBox(8, 8, 24, 24),
                        attributes={"condition": condition})], 64, 64)
    _dataset, data = run_over("detection", images_dir, labels_dir, subject="leaf",
                              attribute="condition")
    config = {"model_source": {"builder": f"{__name__}:build_probe_net", "task": "detection"},
              "data": data}

    shapes = _param_shapes(build_model(config, recorded_model_dims(config)))

    assert shapes["stem.weight"][1] == data["num_channels"] == 1
    assert shapes["head.weight"][0] == len(data["scope"]["id_map"]) == 3


def test_a_saved_checkpoint_rebuilds_the_architecture_its_config_builds(tmp_path):
    """The checkpoint written by the envelope's save path carries enough of the model source for
    the inference-side rebuild to reconstruct the same architecture the run trained. Both sides
    are produced here by the real build path, so a stamp that records less than the builder was
    called with shows up as a shape difference rather than passing on a restated literal."""
    config = _probe_config()
    trained = build_model(config, recorded_model_dims(config))
    assert _param_shapes(trained)["head.weight"] == (7, 6, 1, 1)  # the recorded count took effect

    (tmp_path / "out").mkdir()
    ctx = TrainContext(run=trainer_run(dict(config), tmp_path / "out", has_val_loader=True,
                                       id="auto-run-40"),
                       train_loader=None)
    path = ctx.save_checkpoint({"model_state_dict": trained.state_dict()}, "model_best")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert detect_kind(path) == KIND_TCIP_MODULE
    rebuilt = build_model(loaded["config"], recorded_model_dims(loaded["config"]))
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

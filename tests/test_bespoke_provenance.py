"""Code provenance for a bespoke (model_source) run.

Locks: snapshot_model_source (copy source files + sha256), a pass rebuilding a bespoke model from
its importable builder (no exec) and predicting, and a completed run's registry entry carrying
its metrics and digest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests._chain_fixtures import BESPOKE_CLASSIFIER, BESPOKE_MODELS, GT_ANCHOR_DETECTOR

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from tcip_mcp.pipelines.execution import Stated, prepare  # noqa: E402
from tcip_mcp.pipelines.model_build import snapshot_model_source  # noqa: E402
from tests import bespoke_models  # noqa: E402


def _model_source() -> dict:
    return {"builder": GT_ANCHOR_DETECTOR,
            "builder_kwargs": {"gt_boxes_wh": [[15, 36], [16, 40], [17, 44]],
                               "min_size": 64, "max_size": 128},
            "task": "detection", "source_files": [__file__, BESPOKE_MODELS]}


def _snapshot(model_source: dict, project: Path) -> tuple[dict, dict]:
    """``snapshot_model_source`` over a run of ``project`` of ``model_source`` over no data
    (:func:`~tests._chain_fixtures.training_config`), validated: its record and its copies."""
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config

    return snapshot_model_source(train_config(training_config(model_source, {})), project)


_DATA = {"num_channels": 3, "scope": {"subject": "bud", "attributes": []}}
"""The data section a one-subject, three-band run records, which its checkpoint carries."""


# snapshot_model_source: one copy per declared file, at its path in its own tree.

def test_snapshot_model_source_copies_each_declared_file_at_its_path_in_its_tree(tmp_path):
    """Each declared file is keyed by its stored path and copied at its path under the import
    root of the declared module holding it, else under the project, so a helper lands beside the
    module that imports it."""
    helper = tmp_path / "agent_code" / "helpers.py"
    helper.parent.mkdir()
    helper.write_text("SCALE = 2\n", encoding="utf-8")
    source = {**_model_source(), "source_files": [str(helper), *_model_source()["source_files"]]}

    record, copies = _snapshot(source, tmp_path)

    expected = {"agent_code/helpers.py": (helper, "model_src/agent_code/helpers.py"),
                str(Path(__file__).resolve()): (__file__,
                                                "model_src/test_bespoke_provenance.py"),
                str(Path(BESPOKE_MODELS).resolve()): (BESPOKE_MODELS,
                                                      "model_src/bespoke_models.py")}
    assert set(record["files"]) == set(expected)
    for key, (original, copy) in expected.items():
        entry = record["files"][key]
        assert entry["file"] == copy
        assert copies[copy] == Path(original).read_bytes()
        assert entry["sha256"] == hashlib.sha256(copies[copy]).hexdigest()
        assert entry["bytes"] == len(copies[copy])


def test_snapshot_model_source_refuses_a_declared_file_it_cannot_read(tmp_path):
    src = _model_source()
    src["source_files"] = [*src["source_files"], str(tmp_path / "does_not_exist.py")]

    with pytest.raises(FileNotFoundError, match="does_not_exist.py"):
        _snapshot(src, tmp_path)


def test_snapshot_model_source_refuses_a_builder_no_declared_file_holds(tmp_path):
    with pytest.raises(ValueError, match="definitely_not_a_real_module_xyz"):
        _snapshot({"builder": "definitely_not_a_real_module_xyz:build", "task": "detection",
                   "source_files": [__file__]}, tmp_path)


def test_a_declared_file_of_the_builders_stem_that_is_not_python_holds_no_module(tmp_path):
    """A declared ``review.txt`` is no module, so the plan of a builder ``review:build`` it is
    declared for refuses, though a ``review`` module of that name is already loaded."""
    import sys
    import types

    from tcip_mcp.pipelines.model_build import module_plan
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config

    (tmp_path / "review.txt").write_text("def build():\n    return 1\n", encoding="utf-8")
    cached = types.ModuleType("review")
    cached.build = lambda: 123  # type: ignore[attr-defined]
    sys.modules["review"] = cached
    try:
        with pytest.raises(ValueError, match="review"):
            module_plan(train_config(training_config(
                {"builder": "review:build", "task": "detection",
                 "source_files": [str(tmp_path / "review.txt")]}, {})), tmp_path)
    finally:
        sys.modules.pop("review", None)


def test_snapshot_model_source_refuses_a_file_outside_every_tree_it_can_place(tmp_path):
    """A declared file outside the project and under no declared module's import root has no
    path in the run's snapshot, so the snapshot refuses it, naming it and the roots it could have
    been placed under."""
    from tests import REPO_ROOT

    stray = tmp_path.parent / "stray" / "helpers.py"
    stray.parent.mkdir()
    stray.write_text("", encoding="utf-8")
    src = {**_model_source(), "source_files": [*_model_source()["source_files"], str(stray)]}

    with pytest.raises(ValueError, match="no place for it") as refused:
        _snapshot(src, tmp_path)
    assert all(str(place) in str(refused.value) for place in (stray, tmp_path, REPO_ROOT))


def test_snapshot_model_source_refuses_two_files_landing_on_one_copy(tmp_path):
    """The project's own ``bespoke_models.py`` and the builder's module of that path under its
    import root would be one copy; the snapshot refuses rather than keep either."""
    shadow = tmp_path / "bespoke_models.py"
    shadow.write_text("", encoding="utf-8")
    src = {**_model_source(), "source_files": [*_model_source()["source_files"], str(shadow)]}

    with pytest.raises(ValueError, match="both lie at"):
        _snapshot(src, tmp_path)


def test_a_project_file_beside_an_external_builder_named_like_the_standard_library_refuses(
        tmp_path):
    """A project-local ``code.py`` declared beside a builder from another tree is placed under
    the one snapshot root as ``code`` once copied, so the plan refuses it by that name."""
    from tcip_mcp.pipelines.model_build import module_plan
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import training_config

    (tmp_path / "code.py").write_text("", encoding="utf-8")
    source = {**_model_source(),
              "source_files": [*_model_source()["source_files"], str(tmp_path / "code.py")]}

    with pytest.raises(ValueError, match="standard-library module code"):
        module_plan(train_config(training_config(source, {})), tmp_path)


def test_a_preflight_over_a_declared_standard_library_name_imports_nothing(tmp_path):
    """The preflight places a config's declared files before it imports any of them, so a
    declared ``code.py`` is refused and the loaded ``code`` module is never replaced."""
    import code
    import sys

    from tcip_mcp.tools.training_tools import preflight_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, detection_config

    (tmp_path / "agent_model.py").write_text(
        "from tests.bespoke_models import build_bespoke_detection as build\n", encoding="utf-8")
    (tmp_path / "code.py").write_text("raise AssertionError('declared code.py ran')\n",
                                      encoding="utf-8")
    config = detection_config(tmp_path / "data", model_source={
        **BUILT_DETECTOR, "builder": "agent_model:build", "source_files": [
            str(tmp_path / "code.py"), str(tmp_path / "agent_model.py")]})

    result = preflight_config(tmp_path, config)

    assert any("standard-library module code" in issue for issue in result["issues"])
    assert sys.modules["code"] is code


def test_a_run_whose_snapshot_refuses_leaves_no_run_directory(tmp_path):
    """A declared file that cannot be read refuses the run before its directory exists, so no
    half-opened run is left behind."""
    from tcip_mcp.experiments import experiment_dir
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, detection_config, opened_run

    config = detection_config(tmp_path / "data", model_source={
        **BUILT_DETECTOR, "source_files": [*BUILT_DETECTOR["source_files"],
                                           str(tmp_path / "gone.py")]})

    with pytest.raises(FileNotFoundError, match="gone.py"):
        opened_run(tmp_path, config, experiment_id="refused")

    assert not experiment_dir("refused", project=tmp_path).exists()


def test_a_declared_file_named_like_the_standard_library_refuses_before_a_run_opens(tmp_path):
    """A declared ``code.py`` beside the builder would import as the standard library's
    ``code`` once the run binds its declared files, so the snapshot refuses it by name and no
    run directory is left."""
    from tcip_mcp.experiments import experiment_dir
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, detection_config, opened_run

    (tmp_path / "agent_model.py").write_text(
        "from tests.bespoke_models import build_bespoke_detection as build\n", encoding="utf-8")
    (tmp_path / "code.py").write_text("", encoding="utf-8")
    config = detection_config(tmp_path / "data", model_source={
        **BUILT_DETECTOR, "builder": "agent_model:build", "source_files": [
            str(tmp_path / "agent_model.py"), str(tmp_path / "code.py")]})

    with pytest.raises(ValueError, match="standard-library"):
        opened_run(tmp_path, config, experiment_id="shadowing")

    assert not experiment_dir("shadowing", project=tmp_path).exists()


def test_snapshot_model_source_keeps_files_sharing_a_name_apart(tmp_path):
    """Two declared files sharing a basename land at their own paths, each with its own
    content."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.py").write_text(f"# builder {name}", encoding="utf-8")
    src = {"builder": GT_ANCHOR_DETECTOR, "task": "detection",
           "source_files": [str(tmp_path / "a" / "model.py"), str(tmp_path / "b" / "model.py"),
                            BESPOKE_MODELS]}

    _record, copies = _snapshot(src, tmp_path)

    assert copies["model_src/a/model.py"] == b"# builder a"
    assert copies["model_src/b/model.py"] == b"# builder b"


# A pass rebuilds the bespoke model from its builder (no exec) and predicts.

def test_a_pass_rebuilds_a_bespoke_detector_and_predicts(tmp_path):
    from PIL import Image

    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    checkpoint = load_registered_checkpoint(registered_checkpoint(
        tmp_path, model_source=_model_source(), data=_DATA,
        metrics={"val_loss": 0.3, "epoch": 1}), project=tmp_path)

    p = prepare(checkpoint, Stated(tile=False, conf=0.0), device="cpu").runnable()
    assert type(p.predictor.model).__qualname__ == bespoke_models.BespokeGNDetector.__qualname__
    assert p.predictor.task == "detection"
    assert p.predictor.in_chans == 3

    img = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(img)
    (out,) = p.predict([str(img)])
    assert {"boxes", "scores", "labels", "count"} <= set(out)  # measurable detection output


def test_predictor_loads_at_the_two_channels_its_run_recorded(tmp_path):
    """The width a run records on its data section is the one its model was built at, and the
    predictor reads that checkpoint's images at it: a two-band run loads at two channels, not a
    silent default of 3."""
    import csv

    import numpy as np

    from tcip_mcp.dataset_layout import UNDATED_BUCKET
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tests._verified_checkpoint_fixtures import fixture_data_dir, registered_checkpoint

    where = fixture_data_dir(tmp_path, "two-band")
    images = where / "images" / UNDATED_BUCKET
    images.mkdir(parents=True)
    with open(where / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "label"))
        for index in range(4):
            np.save(images / f"frame{index}.npy",
                    np.full((16, 16, 2), 40 * index, dtype=np.uint8))
            writer.writerow((f"frame{index}", index % 2))
    checkpoint = load_registered_checkpoint(registered_checkpoint(
        tmp_path, model_source={"builder": BESPOKE_CLASSIFIER, "source_files": [BESPOKE_MODELS],
                                "task": "classification"},
        data={"images_dir": str(images), "labels_dir": str(where / "labels.csv"),
              "num_channels": 2}), project=tmp_path)

    p = prepare(checkpoint, Stated(tile=False), device="cpu").runnable()
    assert p.predictor.in_chans == 2

    arr = (np.random.rand(16, 16, 2) * 255).astype(np.uint8)
    img = tmp_path / "two_band.npy"
    np.save(img, arr)
    (out,) = p.predict([str(img)])
    assert out  # decoded and forwarded at 2 channels with no shape-mismatch error


# A completed run's registry entry.

def test_a_completed_bespoke_runs_entry_carries_its_metrics_and_digest(tmp_path):
    from tcip_mcp.model_registry import ModelRegistry
    from tests._verified_checkpoint_fixtures import finished_run

    finished_run(tmp_path, experiment_id="expB", model_source=_model_source(), data=_DATA,
                 metrics={"val_loss": 0.3, "epoch": 1})

    [entry] = [m for m in ModelRegistry(str(tmp_path)).list_models()
               if m["experiment_id"] == "expB"]
    assert entry["metrics"]["val_loss"] == pytest.approx(0.3)
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert entry["experiment_id"] == "expB"

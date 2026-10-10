"""A relative path a caller hands a project door names a location under the project's root,
never under the process's working directory: each door, called with one relative spelling from
two working directories outside two like projects, succeeds and answers the same result (the
location it recorded, spelled against its project, or what it read there), and an entry point
holding no project refuses a relative path by name.

Each door's inputs come from the platform's own producers (``ingest_images``,
``import_coco``, ``register_model``, a run the launcher opens and completes); only the spelling
of the path handed to the door under test is relative; the per-plant delivery ships under an
acknowledgment the breeder's own recording door records.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest

from tests.test_coco_import import BOX, DATE, STEMS, SUBJECT, _dataset, _document

IMAGE = f"images/{DATE}/{STEMS[0]}.png"
"""The project-relative spelling of one ingested image."""


def _relative(path: str | Path, project: Path) -> str:
    """``path`` spelled relative to ``project``."""
    return os.path.relpath(path, project)


def _register_dataset(project: Path) -> Path:
    from tcip_mcp.tools.project_tools import dataset_entry_path, read_datasets, register_dataset

    result = register_dataset(project, ".", crop="currant")
    assert "error" not in result, result
    (entry,) = read_datasets(project)
    return dataset_entry_path(project, entry)


def _write_subject_registry(project: Path) -> Path:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    result = write_subject_registry(project, ".", subjects={
        SUBJECT: {"description": "one fruit"}, "leaf": {"description": "one leaf"}})
    assert "error" not in result, result
    return Path(result["subjects_path"]).parent


def _register_plant_registry(project: Path) -> Path:
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.tools.phenology_tools import register_plant_registry
    from tests._mapping_fixtures import write_plant_csv
    from tests.test_plant_mapping_binding import PLANTS

    write_plant_csv(project / "plants.csv", PLANTS)
    result = register_plant_registry(project, name="plants", csv_paths=["plants.csv"],
                                     crop="currant", site="north orchard")
    assert "error" not in result, result
    record = plant_mapping.load_registry(project, "plants")
    assert record is not None
    (entry,) = plant_mapping.registry_csv_entries(record, project)
    return Path(entry["path"])


def _labeled(project: Path) -> None:
    """``project``'s images labeled through the COCO door, every path stated absolutely."""
    from tcip_mcp.tools.ingest_tools import import_coco

    annotations = [{"id": i, "image_id": i, "category_id": 7, "bbox": BOX, "iscrowd": 0}
                   for i in range(1, len(STEMS) + 1)]
    document = _document(project / "labels.json", annotations=annotations)
    result = import_coco(project, str(document), str(project), DATE)
    assert "error" not in result, result


def _draw_splits(project: Path) -> Path:
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._audit_fixtures import audit_rows

    _labeled(project)
    result = draw_splits(project, ".", seed=0, val_ratio=0.34, calibration_ratio=0.0,
                         holdout_ratio=0.0, output_path="splits/drawn", subject=SUBJECT)
    assert "error" not in result, result
    (row,) = [r for r in audit_rows(project) if r["tool"] == "draw_splits"]
    assert row["arguments"]["folder_path"] == str(project.resolve())
    return Path(row["arguments"]["output_path"])


def _import_coco(project: Path) -> Path:
    from tcip_mcp.tools.ingest_tools import import_coco

    _document(project / "external.json")
    result = import_coco(project, "external.json", ".", DATE)
    assert "error" not in result, result
    return Path(result["document"])


def _ingest_images(project: Path) -> Path:
    from PIL import Image

    from tcip_mcp.tools.ingest_tools import ingest_images

    raw = project / "more_raw"
    raw.mkdir()
    Image.new("RGB", (16, 16)).save(raw / "extra.png")
    result = ingest_images(project, source="more_raw", date_from=DATE)
    assert "error" not in result and result["found"] == 1, result
    return raw


def _save_annotations(project: Path) -> Path:
    import tcip_store as ts

    from tcip_mcp.dataset_layout import label_key
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.annotation_tools import save_annotations
    from tcip_mcp.workspace import BoundProject, workspace_from_environment

    bound = BoundProject(project, read_record(project)["id"], workspace_from_environment())
    result = save_annotations(bound, IMAGE, [{"subject": SUBJECT, "bbox": BOX}])
    assert "error" not in result, result
    assert ts.exists(label_key(project, DATE, STEMS[0]))
    return result


def _stage_proposals(project: Path) -> Path:
    from tcip_mcp.tools.proposal_tools import stage_proposals

    box = {"subject": SUBJECT, "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}
    result = stage_proposals(project, IMAGE, boxes=[box], model_name="stub")
    assert "error" not in result, result
    return result


def _visualize(project: Path) -> Path:
    from tcip_mcp.tools.vision_tools import visualize

    _labeled(project)
    result = visualize(project, "annotations", IMAGE)
    assert "error" not in result, result
    return {key: value for key, value in result.items() if key != "image_path"}


def _overlay_reference_grid(project: Path) -> Path:
    from tcip_mcp.tools.vision_tools import overlay_reference_grid

    result = overlay_reference_grid(project, IMAGE)
    assert "error" not in result, result
    return {key: value for key, value in result.items() if key != "image_path"}


def _read_audit_log(project: Path) -> Path:
    from tcip_mcp.tools.meta_tools import read_audit_log

    result = read_audit_log(project, scope=".")
    assert "error" not in result and result["entries"], result
    return Path(result["scope_resolved"])


def _focus_human_attention(project: Path) -> Path:
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.gui_tools import focus_human_attention
    from tcip_mcp.workspace import BoundProject, workspace_from_environment

    bound = BoundProject(project, read_record(project)["id"], workspace_from_environment())
    result = focus_human_attention(bound, ".", SUBJECT, DATE, image_index=0, mode="box")
    assert "error" not in result, result
    return result


def _render_failure_cases(project: Path) -> list[str]:
    from tcip_mcp.tools.vision_tools import render_failure_cases

    _published(project)
    result = render_failure_cases(project, ".", "preds")
    assert "error" not in result, result
    return sorted(item["stem"] for item in result.get("worst_images") or [])


def _propose_trait_registry(project: Path) -> list[str]:
    from tcip_mcp.traits import resolve_statement_registry

    return [s.name for s in resolve_statement_registry(project, ".").subjects]


def _register_model(project: Path) -> str:
    from tcip_mcp.model_registry import read_registry_index
    from tcip_mcp.tools.model_tools import register_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    elsewhere = project.parent / f"{project.name}-elsewhere"
    spelling = _relative(registered_checkpoint(elsewhere), project)
    result = register_model(project, name="m", checkpoint_path=spelling)
    assert "error" not in result, result
    (entry,) = read_registry_index(project)
    assert Path(result["checkpoint_path"]) == (project / spelling).resolve()
    return entry["name"]


def _registered(project: Path) -> str:
    """A detector of :data:`SUBJECT` a run of another project trained, registered into
    ``project`` through ``register_model`` and spelled relative to it."""
    from tests._verified_checkpoint_fixtures import project_checkpoint

    data = {"num_channels": 3, "scope": {"subject": SUBJECT},
            "split": {"seed": 0, "val_ratio": 0.15}}
    return _relative(project_checkpoint(project, data=data), project)


def _stated():
    """The execution values every pass here states."""
    from tcip_mcp.pipelines.execution import Stated
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF, SAMPLE_MAX_DETS

    return Stated(tile=False, conf=SAMPLE_CONF, max_dets=SAMPLE_MAX_DETS)


def _published(project: Path) -> None:
    """The bucket ``preds`` the registered detector publishes over the labeled capture, every
    path stated absolutely."""
    from tcip_mcp.tools.inference_tools import run_inference

    _labeled(project)
    result = run_inference(project, str((project / _registered(project)).resolve()),
                           images_dir=str(project / "images" / DATE), bucket="preds",
                           stated=_stated())
    assert "error" not in result, result


def _run_inference(project: Path) -> Path:
    from tcip_mcp.tools.inference_tools import run_inference

    result = run_inference(project, _registered(project), images_dir=f"images/{DATE}",
                           bucket="preds", dry_run=True, stated=_stated())
    assert "error" not in result, result
    return Path(result["dataset_root"])


def _evaluate_model(project: Path) -> dict:
    from tcip_mcp.tools.training_tools import evaluate_model

    _labeled(project)
    result = evaluate_model(project, _registered(project), f"images/{DATE}", stated=_stated())
    assert "error" not in result, result
    return {key: result[key] for key in ("tp", "fp", "fn") if key in result} or sorted(result)


def _prioritize_review_queue(project: Path) -> list[str]:
    from tcip_mcp.tools.feedback_tools import prioritize_review_queue

    result = prioritize_review_queue(project, _registered(project), f"images/{DATE}",
                                     budget=len(STEMS))
    assert "error" not in result, result
    return sorted(entry["image"] for entry in result["queue"])


def _triage_predictions(project: Path) -> int:
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS

    result = triage_predictions(project, _registered(project), f"images/{DATE}",
                                max_dets=SAMPLE_MAX_DETS)
    assert "error" not in result, result
    return result["total_images"]


def _calibrate_physical_scale(project: Path) -> Path:
    from tcip_mcp.assessment import read_assessment
    from tests import _trait_fixtures as fx
    from tests.test_physical_scale_assessment import _author_tolerance, _calibrate, _reference

    fx.seed_delivery_traits(project)
    _author_tolerance(project)
    selection, table = _reference(project)
    scale = _calibrate(project, Path(_relative(selection, project)),
                       Path(_relative(table, project)))
    assert scale["passed"] is True, scale
    recorded = read_assessment(project, scale["assessment_id"])
    (cited,) = [f.ground_truth for f in recorded.reference.ground_truth
                if isinstance(f.ground_truth, str) and f.ground_truth.endswith(".csv")]
    return Path(cited)


DOORS: dict[str, Callable[[Path], object]] = {
    "calibrate_physical_scale": _calibrate_physical_scale,
    "evaluate_model": _evaluate_model,
    "focus_human_attention": _focus_human_attention,
    "prioritize_review_queue": _prioritize_review_queue,
    "propose_trait_registry": _propose_trait_registry,
    "register_model": _register_model,
    "render_failure_cases": _render_failure_cases,
    "run_inference": _run_inference,
    "triage_predictions": _triage_predictions,
    "register_dataset": _register_dataset,
    "write_subject_registry": _write_subject_registry,
    "register_plant_registry": _register_plant_registry,
    "draw_splits": _draw_splits,
    "import_coco": _import_coco,
    "ingest_images": _ingest_images,
    "save_annotations": _save_annotations,
    "stage_proposals": _stage_proposals,
    "visualize": _visualize,
    "overlay_reference_grid": _overlay_reference_grid,
    "read_audit_log": _read_audit_log,
}


def _stub_child(monkeypatch) -> None:
    """Launches spawn no training process and start no TensorBoard."""
    import subprocess

    class _StubChild:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 4242

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(subprocess, "Popen", _StubChild)
    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", lambda *a, **k: {})


def _relative_config(project: Path, where: Path) -> dict:
    """A detection config over a dataset at ``project / where`` whose ``images_dir`` is spelled
    relative to ``project``, and the same dataset laid under the working directory as a decoy."""
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset, _detection_config,
    )

    images_dir = _canonical_dataset(project / where)
    _canonical_dataset(Path.cwd() / where)
    config = _detection_config(images_dir)
    config["data"]["images_dir"] = _relative(images_dir, project)
    return config


@pytest.mark.parametrize("door", ["evaluate_model", "prioritize_review_queue",
                                  "triage_predictions"])
def test_a_checkpoint_door_locates_its_checkpoint_once(tmp_path: Path, monkeypatch, door: str):
    """The checkpoint a door names arrives through its acquisition, which locates it once: of
    every location any step answers, whatever spelling it was handed, one is the checkpoint's."""
    import tcip_mcp.model_registry as model_registry
    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.feedback_tools as feedback_tools
    import tcip_mcp.tools.training_tools as training_tools

    project = _dataset(tmp_path).root
    checkpoint = (project / _registered(project)).resolve()
    real = registry_paths.located
    answered: list[Path] = []

    def counted(path, root):
        answered.append(real(path, root))
        return answered[-1]

    for module in (registry_paths, model_registry, feedback_tools, training_tools):
        monkeypatch.setattr(module, "located", counted)
    DOORS[door](project)

    assert answered.count(checkpoint) == 1, answered


def test_a_scoped_door_locates_its_callers_spelling_once(tmp_path: Path, monkeypatch):
    """The dataset a scoped audited door names is located from the caller's spelling once, and
    that one location reaches the door's act and its audit scope; the audit row, filed in the
    dataset's own log, keeps the caller's spelling."""
    import tcip_mcp.audit as audit
    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.project_tools as project_tools
    import tcip_store as ts
    from tests._web_fixtures import new_project
    from tests.test_check_dataset_identity_script import _real_dataset

    project = new_project(tmp_path / "project").root
    _real_dataset(project / "ds")
    real = registry_paths.located
    spellings: list = []

    def counted(path, root):
        spellings.append(path)
        return real(path, root)

    for module in (registry_paths, audit, project_tools):
        monkeypatch.setattr(module, "located", counted)
    assert "error" not in project_tools.register_dataset(project, "ds", "chestnut")

    assert spellings.count("ds") == 1, spellings
    rows = ts.read_log(audit.audit_log_key((project / "ds").resolve())).records
    assert [r["arguments"]["dataset_root"] for r in rows if r["tool"] == "register_dataset"] == [
        "ds"]


def test_a_sweep_records_and_trains_on_the_location_it_admitted(tmp_path: Path, monkeypatch):
    """A sweep admitted over a relative ``images_dir`` records that project-rooted location in
    its input, and its trial trains on the same one, a same-named directory under the working
    directory never entering either."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    project = tmp_path / "project"
    project.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = _relative_config(project, Path("ds"))
    expected = str((project / config["data"]["images_dir"]).resolve())

    sweep = opened_sweep(project, config)
    trial = open_trial(sweep, "t0", {"optimizer.head_lr": 0.001})

    assert observe(sweep).record["input"]["base_config"]["data"]["images_dir"] == expected
    assert observe(trial).record["config"]["data"]["images_dir"] == expected


def test_a_sweep_records_the_base_the_caller_stated_never_its_first_corner(tmp_path: Path):
    """A base stating ``data.split.val_ratio`` beside an axis over it is recorded as stated, so
    a trial whose point names only another parameter trains on the base's own value."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import open_trial
    from tests._training_values import sweep_space
    from tests._verified_checkpoint_fixtures import opened_sweep
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset, _detection_config,
    )

    project = tmp_path / "project"
    config = _detection_config(_canonical_dataset(project / "ds"))
    config["data"]["split"] = {"seed": 0, "val_ratio": 0.2}
    space = {**sweep_space(), "data.split.val_ratio": {"type": "categorical",
                                                       "choices": [0.35, 0.45]}}

    sweep = opened_sweep(project, config, param_space=space)
    trial = open_trial(sweep, "t0", {"optimizer.head_lr": 0.001})

    assert observe(sweep).record["input"]["base_config"]["data"]["split"]["val_ratio"] == 0.2
    assert observe(trial).record["config"]["data"]["split"]["val_ratio"] == 0.2


def test_a_runs_source_files_are_stored_against_its_project_like_its_data(tmp_path: Path):
    """``model_source.source_files``, the files a run's builder imports from, are stored against
    the project as ``images_dir`` is, so a copied project's run reads its own copy of them."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, observe, read_record
    from tcip_mcp.pipelines.schemas import train_config
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, detection_config, opened_run

    project = tmp_path / "project"
    project.mkdir()
    (project / "agent_code.py").write_text("", encoding="utf-8")
    config = detection_config(project / "ds", model_source={
        **BUILT_DETECTOR, "source_files": ["agent_code.py", *BUILT_DETECTOR["source_files"]]})
    run_dir = opened_run(project, train_config(config, project).record())

    stored = read_record(run_dir / RUN_FILE)["config"]
    assert stored["model_source"]["source_files"][0] == "agent_code.py"
    read = observe(run_dir).record["config"]
    assert read["model_source"]["source_files"][0] == str((project / "agent_code.py").resolve())


def test_a_copied_projects_run_and_sweep_trial_read_the_copys_source_files(tmp_path: Path):
    """Every config path a run or sweep records travels with the project: a copy's run reads its
    resolved dataset builder's source files from the copy, and a copy's sweep trial reads the
    model source files its sampled choice and its baseline name from the copy."""
    pytest.importorskip("torchvision")
    import shutil

    import tcip_store as ts
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import open_trial
    from tests._training_values import sweep_space
    from tests._verified_checkpoint_fixtures import opened_run, opened_sweep
    from tests._chain_fixtures import BESPOKE_MODELS
    from tests.test_dataset_source_seam import BESPOKE_DS, BESPOKE_DS_FILE
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset, _detection_config,
    )

    project = tmp_path / "project"
    config = _detection_config(_canonical_dataset(project / "ds"))
    for name in ("agent_code.py", "other.py"):
        (project / name).write_text("", encoding="utf-8")
    run = opened_run(project, {**config, "data": {
        **config["data"], "dataset_source": {"builder": BESPOKE_DS, "source_files": [
            str(project / "agent_code.py"), BESPOKE_DS_FILE]}}})
    choices = [[str(project / "agent_code.py"), BESPOKE_MODELS],
               [str(project / "other.py"), BESPOKE_MODELS]]
    space = {**sweep_space(), "model_source.source_files": {"type": "categorical",
                                                           "choices": choices}}
    sweep = opened_sweep(project, config, param_space=space,
                         baseline_params={"optimizer.head_lr": 0.001,
                                          "model_source.source_files": choices[1]})

    copy = tmp_path / "copy"
    ts.release_root(project)
    shutil.copytree(project, copy)
    copied_run = observe(copy / run.relative_to(project)).record
    copied_sweep = copy / sweep.relative_to(project)
    point = observe(copied_sweep).record["input"]["baseline_params"]
    trial = open_trial(copied_sweep, "t0", point)

    assert copied_run["resolved"]["data"]["dataset_source"]["source_files"][0] == str(
        (copy / "agent_code.py").resolve())
    assert point["model_source.source_files"][0] == str((copy / "other.py").resolve())
    assert observe(copied_sweep).record["input"]["param_space"][
        "model_source.source_files"]["choices"][0][0] == str((copy / "agent_code.py").resolve())
    assert observe(trial).record["config"]["model_source"]["source_files"][0] == str(
        (copy / "other.py").resolve())


def test_a_copied_sweeps_whole_mapping_choices_open_a_trial_naming_the_copy(tmp_path: Path):
    """A sweep whose choices are the whole ``model_source`` and ``data`` mappings stores the paths
    inside them against its project, so a trial a copied project opens on those choices names the
    copy's files."""
    pytest.importorskip("torchvision")
    import shutil

    import tcip_store as ts
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset, _detection_config,
    )

    project = tmp_path / "project"
    config = _detection_config(_canonical_dataset(project / "ds"))
    (project / "agent_code.py").write_text("", encoding="utf-8")
    model_source = {**config["model_source"],
                    "source_files": [str(project / "agent_code.py"), *config["model_source"][
                        "source_files"]]}
    space = {"model_source": {"type": "categorical", "choices": [model_source]},
             "data": {"type": "categorical", "choices": [config["data"]]}}
    sweep = opened_sweep(project, config, param_space=space)

    copy = tmp_path / "copy"
    ts.release_root(project)
    shutil.copytree(project, copy)
    copied_sweep = copy / sweep.relative_to(project)
    stored = observe(copied_sweep).record["input"]["param_space"]
    point = {name: axis["choices"][0] for name, axis in stored.items()}
    trial = observe(open_trial(copied_sweep, "t0", point)).record["config"]

    assert trial["model_source"]["source_files"][0] == str((copy / "agent_code.py").resolve())
    assert Path(trial["data"]["images_dir"]).is_relative_to(copy.resolve())


def _authored_checkpoint(project: Path, monkeypatch) -> tuple[str, Path]:
    """A run completed in a new project at ``project`` over a builder module of a name of its own
    at ``project/<module>.py``: the module's name and the checkpoint the run registered."""
    import sys
    import uuid

    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._verified_checkpoint_fixtures import BUILT_DETECTOR, registered_checkpoint
    from tests._web_fixtures import new_project

    module = f"builder_{uuid.uuid4().hex[:8]}"
    new_project(project)
    (project / f"{module}_helper.py").write_text("MIN_SIZE = 64\n", encoding="utf-8")
    (project / f"{module}.py").write_text(
        f"import {module}_helper\n\n\n"
        "def build(**kwargs):\n"
        "    from tests.bespoke_models import build_bespoke_detection\n\n"
        f"    kwargs['min_size'] = {module}_helper.MIN_SIZE\n"
        "    return build_bespoke_detection(**kwargs)\n",
        encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    return module, Path(registered_checkpoint(project, model_source={
        **BUILT_DETECTOR, "builder": f"{module}:build",
        "source_files": [str(project / f"{module}.py"), str(project / f"{module}_helper.py"),
                         BESPOKE_MODELS]}))


def _built_from(checkpoint, project: Path) -> Path:
    """The file the builder of ``checkpoint``, loaded for ``project``, was defined in, once the
    model it builds is built (``model_build.build_from_model_source``)."""
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import (
        build_from_model_source, import_source_builder, recorded_model_dims,
    )

    loaded = load_registered_checkpoint(checkpoint, project=project)
    spec = loaded.spec
    build_from_model_source(spec.model_source, loaded.layout, recorded_model_dims(spec))
    build = import_source_builder(spec.model_source.builder, loaded.layout)
    return Path(build.__code__.co_filename).resolve()


def test_a_checkpoint_builds_from_its_runs_snapshot_in_its_own_project(tmp_path: Path, monkeypatch):
    """Admits valid work: a run completed through the platform's own producer, loaded in its own
    project, binds its builder to the copy its run snapshotted and builds."""
    pytest.importorskip("torchvision")
    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR

    project = tmp_path / "project"
    _module, checkpoint = _authored_checkpoint(project, monkeypatch)

    built = _built_from(checkpoint, project)

    assert built.is_relative_to(project.resolve()) and SNAPSHOT_DIR in built.parts


def test_a_copied_project_builds_from_the_copys_snapshot_with_the_original_gone(
        tmp_path: Path, monkeypatch):
    """The whole project copied to a new root and the original deleted: the copy's checkpoint
    builds from the copy's own snapshot, its builder and the helper declared beside it alike,
    with the original's modules of those names still loaded."""
    pytest.importorskip("torchvision")
    import shutil
    import sys

    import tcip_store as ts
    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR

    project, copy = tmp_path / "project", tmp_path / "copy"
    module, checkpoint = _authored_checkpoint(project, monkeypatch)
    assert module in sys.modules
    ts.release_root(project)
    shutil.copytree(project, copy)
    shutil.rmtree(project)

    built = _built_from(copy / checkpoint.relative_to(project), copy)

    assert built.is_relative_to(copy.resolve()) and SNAPSHOT_DIR in built.parts
    helper = Path(sys.modules[f"{module}_helper"].__file__).resolve()
    assert helper.is_relative_to(copy.resolve()) and SNAPSHOT_DIR in helper.parts


def test_a_run_carried_whole_builds_from_its_snapshot_never_a_same_named_file(
        tmp_path: Path, monkeypatch):
    """A run carried whole into another project (``archive_project``, then ``import_project``)
    whose own file at the author's relative location is a different module of the same name
    builds from the run's snapshot; that file never answers."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.project_tools import import_project, write_archive

    project, carried = tmp_path / "project", tmp_path / "carried"
    module, checkpoint = _authored_checkpoint(project, monkeypatch)
    bundle = tmp_path / "bundle.zip"
    assert "error" not in write_archive(project.resolve(), output_path=str(bundle),
                                        include_models=True)
    assert "error" not in import_project(str(bundle), str(carried))
    (carried / f"{module}.py").write_text(
        "def build(**kwargs):\n    raise AssertionError('the same-named file answered')\n",
        encoding="utf-8")

    built = _built_from(carried / checkpoint.relative_to(project), carried)

    assert built.is_relative_to(carried.resolve()) and built != (carried / f"{module}.py").resolve()


def test_a_checkpoint_moved_without_its_run_refuses_before_importing(tmp_path: Path, monkeypatch):
    """A checkpoint file carried alone into a project that holds no snapshot of its run refuses
    at load, naming the run directory its source lies in and the primitive that carries a run
    whole, and imports nothing."""
    pytest.importorskip("torchvision")
    import shutil

    import tcip_mcp.pipelines.model_build as model_build
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import Stated, prepare
    from tests._verified_checkpoint_fixtures import register_checkpoint
    from tests._web_fixtures import new_project

    project, elsewhere = tmp_path / "project", tmp_path / "elsewhere"
    _module, checkpoint = _authored_checkpoint(project, monkeypatch)
    new_project(elsewhere)
    alone = elsewhere / "alone.pt"
    shutil.copyfile(checkpoint, alone)
    register_checkpoint(elsewhere, str(alone), name="alone")
    loaded: list[str] = []
    monkeypatch.setattr(model_build, "import_source_builder",
                        lambda dotted, layout: loaded.append(dotted))

    with pytest.raises(ValueError, match="archive_project") as refused:
        prepare(load_registered_checkpoint(alone, project=elsewhere), Stated(tile=False))

    assert checkpoint.parent.name in str(refused.value)
    assert loaded == []


def test_a_run_missing_its_metrics_log_answers_its_own_error_at_the_checkpoint_doors(
        tmp_path: Path):
    """An existing registered checkpoint whose project holds a run missing its metrics log is
    not an absent checkpoint: evaluation and the review queue answer the broken record's own
    error."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import METRICS_FILE
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import SAMPLE_MAX_DETS, registered_checkpoint

    project = _dataset(tmp_path).root
    checkpoint = Path(registered_checkpoint(project, data={
        "num_channels": 3, "scope": {"subject": SUBJECT},
        "split": {"seed": 0, "val_ratio": 0.15}}))
    (checkpoint.parent / METRICS_FILE).unlink()
    spelling = _relative(checkpoint, project)

    evaluated = evaluate_model(project, spelling, f"images/{DATE}", stated=_stated())
    triaged = triage_predictions(project, spelling, f"images/{DATE}", max_dets=SAMPLE_MAX_DETS)

    for answer in (evaluated, triaged):
        assert f"is missing {checkpoint.parent / METRICS_FILE}" in answer["error"], answer


def test_evaluation_takes_a_run_id_and_a_relative_checkpoint_path(tmp_path: Path):
    """A completed run of the project named by its id and the checkpoint it completed named by a
    relative path both evaluate."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    project = _dataset(tmp_path).root
    _labeled(project)
    checkpoint = Path(registered_checkpoint(project, data={
        "num_channels": 3, "scope": {"subject": SUBJECT},
        "split": {"seed": 0, "val_ratio": 0.15}}))

    by_id = evaluate_model(project, checkpoint.parent.name, f"images/{DATE}", stated=_stated())
    by_path = evaluate_model(project, _relative(checkpoint, project), f"images/{DATE}",
                             stated=_stated())

    assert "error" not in by_id and "error" not in by_path, (by_id, by_path)


def test_a_resumed_run_records_the_project_rooted_checkpoint(tmp_path: Path, monkeypatch):
    """``resume_from`` spelled relative names a checkpoint a run of the project wrote: the run
    record and the launch's audit line both carry that location."""
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import observe
    from tcip_mcp.tools.training_tools import launch_training
    from tests._audit_fixtures import audit_rows
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    project = tmp_path / "project"
    project.mkdir()
    checkpoint = Path(registered_checkpoint(project))
    expected = str(checkpoint.resolve())
    _stub_child(monkeypatch)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = launch_training(project, _relative_config(project, Path("ds")),
                             resume_from=_relative(checkpoint, project), actor=None)

    assert "error" not in result, result
    assert observe(Path(result["output_dir"])).record["resume_from"] == expected
    (row,) = [r for r in audit_rows(project) if r["tool"] == "launch_training"]
    assert row["arguments"]["resume_from"] == expected


def test_a_config_with_a_relative_path_and_no_project_is_refused_by_name():
    """A config validated where no project is held has no root to read a relative path against:
    it refuses naming the path, never reading it under the working directory."""
    from tcip_mcp.pipelines.schemas import checked_train_config

    spec, issues = checked_train_config(
        {"model_source": {"builder": "m:f", "task": "detection"},
         "data": {"images_dir": "images/undated"}})
    assert spec is None
    assert any("data.images_dir" in issue and "absolute path" in issue for issue in issues)


def test_the_preflight_command_reads_a_relative_config_under_its_project(project, tmp_path):
    """``--config`` spelled relative names the config under ``--project``: run from a directory
    holding a decoy of the same name, the project's own config is the one validated."""
    import json

    from tests._cli_fixtures import run_tcip
    from tests.test_preflight_config_script import _fixture_config

    _fixture_config(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()
    (cwd / "config.json").write_text("{}", encoding="utf-8")

    result = run_tcip("preflight-config", ["--config", "config.json", "--project", str(project)],
                      cwd=cwd)

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["valid"] is True


def test_the_identity_command_reads_a_relative_dataset_root_under_its_project(tmp_path: Path):
    """The dataset root spelled relative names the dataset under ``--project``, so the check
    from any working directory reports the registered dataset's own outcome."""
    from tcip_mcp.tools.project_tools import register_dataset
    from tests._cli_fixtures import run_tcip
    from tests._web_fixtures import new_project
    from tests.test_check_dataset_identity_script import _real_dataset

    project = new_project(tmp_path / "project").root
    _real_dataset(project / "ds")
    assert "error" not in register_dataset(project, str(project / "ds"), "chestnut")
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    completed = run_tcip("check-dataset-identity", ["ds", "--project", str(project)], cwd=cwd)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


def test_the_score_command_reads_a_relative_path_under_its_project(project, tmp_path):
    """``--path`` spelled relative names the image under ``--project``, run from a directory
    outside it."""
    import json

    from tests._cli_fixtures import run_tcip
    from tests.test_score_predictions_script import _fixture

    image, bucket = _fixture(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = run_tcip("score-predictions", ["--path", _relative(image, project), "--bucket",
                                            bucket, "--project", str(project)], cwd=cwd)

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["tp"] == 1


def test_the_doctor_reads_the_project_named_by_a_relative_root(tmp_path: Path):
    """``tcip doctor .`` from inside a project checks that project, establishing the root once
    where the command starts."""
    from tests._cli_fixtures import run_tcip

    project = _dataset(tmp_path).root

    result = run_tcip("doctor", ["."], cwd=project)

    assert "doctor:" in result.stdout, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def test_the_shapefile_converter_refuses_relative_paths_by_name(tmp_path: Path, monkeypatch):
    """The converter holds no project, so a relative input or output refuses naming the absolute
    spelling and nothing is written under the working directory."""
    from tcip_mcp.cli.shp_to_plant_csv import convert_shp_to_plant_csv
    from tests.test_shp_to_plant_csv_script import _point_shapefile

    shp = _point_shapefile(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute path"):
        convert_shp_to_plant_csv(shp, "plants.csv")
    with pytest.raises(ValueError, match="absolute path"):
        convert_shp_to_plant_csv(shp.name, tmp_path / "plants.csv")
    assert not (tmp_path / "plants.csv").exists()


def test_the_census_command_refuses_a_relative_dataset_root(tmp_path: Path):
    """``tcip scan-dataset`` holds no project, so a relative dataset root is refused naming the
    absolute spelling rather than read under the operator's working directory, where one is."""
    from tests._cli_fixtures import run_tcip

    _dataset(tmp_path)
    result = run_tcip("scan-dataset", ["fruit_count"], cwd=tmp_path)
    assert result.returncode == 1
    assert "absolute path" in result.stderr


@pytest.mark.parametrize("door", sorted(DOORS))
def test_one_relative_path_names_one_project_location_from_any_working_directory(
        tmp_path: Path, monkeypatch, door: str):
    """The door answers one relative spelling from two working directories outside two like
    projects, each directory holding nothing the spelling names, with a successful result that is
    the same: the location it recorded, spelled against its project, or what it read there."""
    found = []
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        project = _dataset(tmp_path / name).root
        cwd = tmp_path / f"{name}_cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        observed = DOORS[door](project)
        found.append(_relative(observed, project) if isinstance(observed, Path) else observed)
    assert found[0] == found[1]


def test_a_per_plant_delivery_reads_and_writes_under_the_project(tmp_path: Path, monkeypatch):
    """``deliver_per_plant_csv`` with ``dataset_root="ds"`` and ``output_path="out.csv"``, shipped
    under the breeder's recorded acknowledgment, reads the buckets under the project's dataset
    and writes its CSV under the project, from two working directories outside two like projects,
    leaving each working directory empty."""
    pytest.importorskip("torch")
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import DeliveryRefusedError, record_acknowledgment
    from tcip_mcp.pipelines.postprocessing.aggregation import deliver_per_plant_aggregate
    from tcip_mcp.pipelines.postprocessing.phenology import population
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv
    from tests import _trait_fixtures as fx, csv_rows
    from tests.test_deliver_per_plant_csv import KIND, TWO_PLANTS, _bucket, _mapped, _results

    found = []
    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        fx.seed_delivery_traits(project)
        fx.seed_confirmed_aggregate(project, "stem_count", value_keys=["count"])
        dataset_root, date = _mapped(project, TWO_PLANTS)
        bucket = _bucket(project, dataset_root, date, {"P1": 2, "P2": 1}, name="run")
        results = _results(("P1", 2), ("P2", 1))
        with pytest.raises(DeliveryRefusedError) as refused:
            deliver_per_plant_aggregate(
                project, results, str(project / "unacknowledged.csv"),
                delivered_phenotype="stem_count", delivery_kind=KIND,
                buckets=[read_bucket(dataset_root, bucket)], plants=population(["P1", "P2"]),
                crop="currant", door="deliver_per_plant_csv", actor=None)
        act = record_acknowledgment(project, acknowledged_by="user:breeder", reason="a look",
                                    result_sha256=str(refused.value.result_sha256))
        cwd = tmp_path / f"{name}_cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        delivered = deliver_per_plant_csv(project, results, "out.csv", "stem_count", KIND,
                                          ["P1", "P2"], _relative(dataset_root, project), [bucket],
                                          crop="currant", acknowledgment_id=act.acknowledgment_id)

        assert "error" not in delivered, delivered
        assert (project / "out.csv").is_file() and list(cwd.iterdir()) == []
        found.append(sorted(row["plant_id"] for row in csv_rows(project / "out.csv")))
    assert found[0] == found[1]


def test_a_phenology_measurement_reads_the_buckets_under_the_project_rooted_dataset(
        tmp_path: Path, monkeypatch):
    """The phenology delivery door over a series the platform's producers built, its dataset
    root spelled relative to the project, measures the project's own buckets from two working
    directories outside two like projects: each reaches the delivery's own refusal of the
    unassessed buckets it read there, and leaves its working directory empty."""
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import PLANTS, attributed_series

    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        series = attributed_series(project, fractions=(0.0, 1.0), assessed=False)
        cwd = tmp_path / f"{name}_cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        refused = deliver_phenology_milestones(
            project, series.trait, series.mapping_name, _relative(series.root, project),
            list(series.buckets.values()), "out.csv", list(PLANTS),
            require_all_dates_complete=True)
        assert "no assessment answers for bucket" in refused.get("error", ""), refused
        assert list(cwd.iterdir()) == []

"""A relative path handed to a project door names a location under the project's root.

Each door, called with one relative spelling from two working directories outside two like
projects, succeeds and answers the same result (the location it recorded, spelled against its
project, or what it read there), and an entry point holding no project refuses a relative path
by name.

Each door's inputs come from the platform's own producers (``ingest_images``,
``import_coco``, ``register_model``, a run the launcher opens and completes), with the path
handed to the door spelled relative; the per-plant delivery ships under an acknowledgment the
breeder's own recording door records.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir
from tests._training_values import (
    COMBINED_SCORER,
    IOU_THRESHOLD,
    RATIONALE,
    TRIAGE_WEIGHTS,
    VERTEX_SPACING,
    drawn_split,
)
from tests.test_coco_import import (
    BOX,
    DATE,
    STEMS,
    SUBJECT,
    _dataset,
    _document,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tcip_mcp.pipelines.execution import Stated

CAPTURE = image_dir("", DATE).as_posix()
"""The project-relative spelling of the ingested images' capture."""
IMAGE = f"{CAPTURE}/{STEMS[0]}.png"
"""The project-relative spelling of one ingested image."""


def _relative(path: str | Path, project: Path) -> str:
    """Return ``path`` spelled relative to ``project``."""
    return os.path.relpath(path, project)


def _register_dataset(project: Path) -> Path:
    from tcip_mcp.tools.project_tools import (
        dataset_entry_path,
        read_datasets,
        register_dataset,
    )

    result = register_dataset(project, ".", crop="currant", actor=None)
    assert "error" not in result, result
    (entry,) = read_datasets(project)
    return dataset_entry_path(project, entry)


def _write_subject_registry(project: Path) -> Path:
    from tcip_mcp.tools.annotation_tools import write_subject_registry

    result = write_subject_registry(
        project,
        ".",
        subjects={
            SUBJECT: {"description": "one fruit"},
            "leaf": {"description": "one leaf"},
        },
        actor=None,
    )
    assert "error" not in result, result
    return Path(result["subjects_path"]).parent


def _register_plant_registry(project: Path) -> Path:
    from tcip_mcp.pipelines.postprocessing import plant_mapping
    from tcip_mcp.tools.phenology_tools import register_plant_registry
    from tests._mapping_fixtures import write_plots_csv

    write_plots_csv(project / "plants.csv")
    result = register_plant_registry(
        project,
        name="plants",
        csv_paths=["plants.csv"],
        crop="currant",
        site="north orchard",
        actor=None,
    )
    assert "error" not in result, result
    record = plant_mapping.load_registry(project, "plants")
    assert record is not None
    (entry,) = plant_mapping.registry_csv_entries(record, project)
    return Path(entry["path"])


def _labeled(project: Path) -> None:
    """Label ``project``'s images through the COCO door, stating absolute paths."""
    from tcip_mcp.tools.ingest_tools import import_coco

    annotations = [
        {"id": i, "image_id": i, "category_id": 7, "bbox": BOX, "iscrowd": 0}
        for i in range(1, len(STEMS) + 1)
    ]
    document = _document(project / "labels.json", annotations=annotations)
    result = import_coco(
        project,
        str(document),
        str(project),
        DATE,
        VERTEX_SPACING,
        RATIONALE,
        actor=None,
    )
    assert "error" not in result, result


def _draw_splits(project: Path) -> Path:
    from tcip_mcp.tools.data_tools import draw_splits
    from tests._audit_fixtures import audit_rows

    _labeled(project)
    result = draw_splits(
        project,
        ".",
        output_path="splits/drawn",
        subject=SUBJECT,
        **drawn_split(val_ratio=0.34, grouping="tile_prefix"),
        actor=None,
    )
    assert "error" not in result, result
    (row,) = audit_rows(project, "draw_splits")
    assert row["arguments"]["folder_path"] == str(project.resolve())
    return Path(row["arguments"]["output_path"])


def _import_coco(project: Path) -> Path:
    from tcip_mcp.tools.ingest_tools import import_coco

    _document(project / "external.json")
    result = import_coco(
        project,
        "external.json",
        ".",
        DATE,
        VERTEX_SPACING,
        RATIONALE,
        actor=None,
    )
    assert "error" not in result, result
    return Path(result["document"])


def _ingest_images(project: Path) -> Path:
    from datetime import datetime

    from PIL import Image

    from tcip_mcp.tools.ingest_tools import ingest_images
    from tests._image_fixtures import capture_exif

    raw = project / "more_raw"
    raw.mkdir()
    Image.new("RGB", (16, 16)).save(
        raw / "extra.png", exif=capture_exif(datetime.fromisoformat(DATE))
    )
    result = ingest_images(project, source="more_raw", actor=None)
    assert "error" not in result and result["found"] == 1, result
    return raw


def _save_annotations(project: Path) -> Path:
    import tcip_store as ts
    from tcip_mcp.dataset_layout import label_key
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.annotation_tools import save_annotations
    from tcip_mcp.workspace import BoundProject, workspace_from_environment

    bound = BoundProject(
        project, read_record(project).id, workspace_from_environment()
    )
    result = save_annotations(
        bound, IMAGE, [{"subject": SUBJECT, "bbox": BOX}], actor=None
    )
    assert "error" not in result, result
    assert ts.exists(label_key(project, DATE, STEMS[0]))
    return result


def _stage_proposals(project: Path) -> dict:
    """Stage one box on the image and return the answer, its bucket as its record reads back.

    The bucket's name carries the staging's own fresh id, so the answer holds the documents the
    record names in its place.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import parse_image_path
    from tcip_mcp.registry_paths import located
    from tcip_mcp.tools.proposal_tools import stage_proposals

    box = {
        "subject": SUBJECT,
        "conf": 0.9,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.25,
        "h": 0.25,
    }
    result = stage_proposals(
        project, IMAGE, boxes=[box], model_name="stub", actor=None
    )
    assert "error" not in result, result
    root = parse_image_path(located(IMAGE, project))[0]
    published = read_bucket(root, result.bucket)
    return {
        **result.model_dump(exclude={"bucket"}),
        "documents": published.documents,
    }


def _visualize(project: Path) -> Path:
    from tcip_mcp.tools.vision_tools import AnnotationsRender, visualize

    _labeled(project)
    result = visualize(
        project, AnnotationsRender(source="annotations", path=IMAGE)
    )
    assert "error" not in result, result
    return {key: value for key, value in result.items() if key != "image_path"}


def _overlay_reference_grid(project: Path) -> Path:
    from tcip_mcp.tools.vision_tools import overlay_reference_grid

    result = overlay_reference_grid(project, IMAGE, overlap=0.0)
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

    bound = BoundProject(
        project, read_record(project).id, workspace_from_environment()
    )
    result = focus_human_attention(
        bound, ".", SUBJECT, DATE, image_index=0, mode="box"
    )
    assert "error" not in result, result
    return result


def _render_failure_cases(project: Path) -> list[str]:
    from tcip_mcp.tools.vision_tools import render_failure_cases

    _published(project)
    result = render_failure_cases(project, ".", "preds", 10, **TRIAGE_WEIGHTS)
    assert "error" not in result, result
    return sorted(item["stem"] for item in result["worst_images"])


def _propose_trait_registry(project: Path) -> list[str]:
    from tcip_mcp.traits import resolve_statement_registry

    return [s.name for s in resolve_statement_registry(project, ".").subjects]


def _register_model(project: Path) -> str:
    from tcip_mcp.model_registry import read_registry_index
    from tcip_mcp.tools.model_tools import register_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    elsewhere = project.parent / f"{project.name}-elsewhere"
    spelling = _relative(registered_checkpoint(elsewhere), project)
    result = register_model(
        project, name="m", checkpoint_path=spelling, actor=None
    )
    (entry,) = read_registry_index(project)
    assert (
        Path(result.entry.checkpoint_path)
        == (project / spelling).resolve()
    )
    return entry.name


def _registered(project: Path) -> str:
    """Return a detector of :data:`SUBJECT` registered into ``project``, spelled relative to it.

    A finished run under ``project`` registered the detector by completing
    (``_verified_checkpoint_fixtures.project_checkpoint``).
    """
    from tests._verified_checkpoint_fixtures import project_checkpoint

    data = {
        "num_channels": 3,
        "scope": {"subject": SUBJECT},
        "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
    }
    return _relative(project_checkpoint(project, data=data), project)


def _stated() -> Stated:
    """Return the stated execution values of an untiled pass at the sample confidence."""
    from tcip_mcp.pipelines.execution import Stated
    from tests._verified_checkpoint_fixtures import SAMPLE_CONF

    return Stated(tile=False, conf=SAMPLE_CONF, rationale=RATIONALE)


def _published(project: Path) -> None:
    """Publish bucket ``preds`` from the registered detector over the labeled capture.

    The paths it states are absolute.
    """
    from tcip_mcp.tools.inference_tools import run_inference

    _labeled(project)
    result = run_inference(
        project,
        str((project / _registered(project)).resolve()),
        images_dir=str(image_dir(project, DATE)),
        bucket="preds",
        stated=_stated(),
        actor=None,
    )
    assert "error" not in result, result


def _run_inference(project: Path) -> Path:
    from tcip_mcp.tools.inference_tools import run_inference

    result = run_inference(
        project,
        _registered(project),
        images_dir=CAPTURE,
        bucket="preds",
        dry_run=True,
        stated=_stated(),
        actor=None,
    )
    assert "error" not in result, result
    return Path(result["dataset_root"])


def _evaluate_model(project: Path) -> dict:
    from tcip_mcp.tools.training_tools import evaluate_model

    _labeled(project)
    result = evaluate_model(
        project,
        _registered(project),
        CAPTURE,
        stated=_stated(),
        iou_threshold=IOU_THRESHOLD,
    )
    assert "error" not in result, result
    return {
        key: result[key] for key in ("tp", "fp", "fn") if key in result
    } or sorted(result)


def _prioritize_review_queue(project: Path) -> list[str]:
    from tcip_mcp.tools.feedback_tools import (
        ReviewQueue,
        prioritize_review_queue,
    )

    result = prioritize_review_queue(
        project,
        _registered(project),
        CAPTURE,
        scorer=COMBINED_SCORER,
        budget=len(STEMS),
        rationale=RATIONALE,
    )
    assert isinstance(result, ReviewQueue), result
    return sorted(entry.image for entry in result.queue)


def _triage_predictions(project: Path) -> int:
    from tcip_mcp.tools.feedback_tools import triage_predictions

    result = triage_predictions(
        project, _registered(project), CAPTURE, 0.3, 0.8, RATIONALE
    )
    assert "error" not in result, result
    return result["total_images"]


def _calibrate_physical_scale(project: Path) -> Path:
    from tcip_mcp.assessment import read_assessment
    from tests import _trait_fixtures as fx
    from tests.test_physical_scale_assessment import (
        _author_tolerance,
        _calibrate,
        _reference,
    )

    fx.seed_delivery_traits(project)
    _author_tolerance(project)
    selection, table = _reference(project)
    scale = _calibrate(
        project,
        Path(_relative(selection, project)),
        Path(_relative(table, project)),
    )
    recorded = read_assessment(project, scale["assessment_id"])
    assert recorded.passed, scale
    (cited,) = [
        f.ground_truth
        for f in recorded.reference.ground_truth
        if isinstance(f.ground_truth, str) and f.ground_truth.endswith(".csv")
    ]
    return Path(cited)


def _scan_dataset(project: Path) -> dict:
    from tcip_mcp.tools.data_tools import scan_dataset

    result = scan_dataset(project, ".")
    assert "error" not in result and result["image_count"], result
    return {k: v for k, v in result.items() if k != "images_sample"}


def _score_predictions(project: Path) -> dict:
    from tcip_mcp.tools.annotation_tools import score_predictions

    _published(project)
    result = score_predictions(project, CAPTURE, "preds", IOU_THRESHOLD, 0.0)
    assert "error" not in result, result
    return {
        k: v for k, v in result.items() if k not in ("path", "not_predicted")
    }


def _check_dataset_identity(project: Path) -> str:
    from tcip_mcp.tools.project_tools import check_dataset_identity

    _labeled(project)
    _register_dataset(project)
    verdict = check_dataset_identity(project, ".")
    assert verdict.state == "unchanged", verdict
    return verdict.state


def _shp_to_plant_csv(project: Path) -> Path:
    from tcip_mcp.tools.phenology_tools import shp_to_plant_csv
    from tests.test_shp_to_plant_csv_script import _point_shapefile

    _point_shapefile(project)
    result = shp_to_plant_csv(project, "points.shp", "plants.csv", actor=None)
    assert "error" not in result, result
    return Path(result["csv_path"])


DOORS: dict[str, Callable[[Path], object]] = {
    "scan_dataset": _scan_dataset,
    "score_predictions": _score_predictions,
    "check_dataset_identity": _check_dataset_identity,
    "shp_to_plant_csv": _shp_to_plant_csv,
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


def _relative_config(project: Path, where: Path) -> dict:
    """Return a detection config whose ``images_dir`` is spelled relative to ``project``.

    ``_canonical_dataset`` writes a dataset at ``project / where`` and another at
    ``Path.cwd() / where``; the config is over the first.
    """
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset,
        _detection_config,
    )

    images_dir = _canonical_dataset(project / where)
    _canonical_dataset(Path.cwd() / where)
    return _detection_config(project, _relative(images_dir, project))


@pytest.mark.parametrize(
    "door", ["evaluate_model", "prioritize_review_queue", "triage_predictions"]
)
def test_a_checkpoint_door_locates_its_checkpoint_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, door: str
) -> None:
    """A checkpoint door locates the checkpoint it names exactly once.

    Of every location any step answers for the door's relative spelling, one is the
    checkpoint's.
    """
    import tcip_mcp.model_registry as model_registry
    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.feedback_tools as feedback_tools
    import tcip_mcp.tools.training_tools as training_tools

    project = _dataset(tmp_path).root
    checkpoint = (project / _registered(project)).resolve()
    real = registry_paths.located
    answered: list[Path] = []

    def counted(path: str | Path, root: str | Path | None) -> Path:
        answered.append(real(path, root))
        return answered[-1]

    for module in (
        registry_paths,
        model_registry,
        feedback_tools,
        training_tools,
    ):
        monkeypatch.setattr(module, "located", counted)
    DOORS[door](project)

    assert answered.count(checkpoint) == 1, answered


def test_a_scoped_door_locates_its_callers_spelling_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scoped audited door locates the dataset spelling it is handed once.

    The spelling ``ds`` is located once, and the audit row, filed in the dataset's own log,
    keeps ``ds``.
    """
    import tcip_mcp.audit as audit
    import tcip_mcp.registry_paths as registry_paths
    import tcip_mcp.tools.project_tools as project_tools
    from tests._audit_fixtures import audit_rows
    from tests._web_fixtures import new_project
    from tests.test_check_dataset_identity_script import _real_dataset

    project = new_project(tmp_path / "project").root
    _real_dataset(project / "ds")
    real = registry_paths.located
    spellings: list = []

    def counted(path: str | Path, root: str | Path | None) -> Path:
        spellings.append(path)
        return real(path, root)

    for module in (registry_paths, audit, project_tools):
        monkeypatch.setattr(module, "located", counted)
    assert "error" not in project_tools.register_dataset(
        project, "ds", "chestnut", actor=None
    )

    assert spellings.count("ds") == 1, spellings
    (row,) = audit_rows((project / "ds").resolve(), "register_dataset")
    arguments = row["arguments"]
    assert isinstance(arguments, dict) and arguments["dataset_root"] == "ds"


def test_a_sweep_records_and_trains_on_the_location_it_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep over a relative ``images_dir`` and its trial record the project-rooted location.

    The sweep records that location in its input, and an opened trial's config names the same
    one, not the same-named directory under the working directory.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import observe_run, observe_sweep
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

    base_data = observe_sweep(sweep).record.input.base_config["data"]
    assert isinstance(base_data, dict)
    assert base_data["images_dir"] == expected
    assert observe_run(trial).record.spec.data.images_dir == expected


def test_a_sweep_records_the_base_the_caller_stated_never_its_first_corner(
    tmp_path: Path,
) -> None:
    """A sweep records its base's stated validation ratio, not its search space's first value.

    A base stating ``data.split.val_ratio`` 0.2 beside an axis over 0.35 and 0.45 is recorded at
    0.2, and the config of a trial whose point names another parameter records 0.2.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import observe_run, observe_sweep
    from tcip_mcp.pipelines.schemas import DrawnSplit
    from tcip_mcp.tools.training_tools import open_trial
    from tests._training_values import sweep_space
    from tests._verified_checkpoint_fixtures import opened_sweep
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset,
        _detection_config,
    )

    project = tmp_path / "project"
    config = _detection_config(
        project,
        _canonical_dataset(project / "ds"),
        split=drawn_split(val_ratio=0.2, grouping="tile_prefix"),
    )
    space = {
        **sweep_space(),
        "data.split.val_ratio": {
            "type": "categorical",
            "choices": [0.35, 0.45],
        },
    }

    sweep = opened_sweep(project, config, param_space=space)
    trial = open_trial(sweep, "t0", {"optimizer.head_lr": 0.001})

    base_data = observe_sweep(sweep).record.input.base_config["data"]
    assert isinstance(base_data, dict)
    base_split = base_data["split"]
    assert isinstance(base_split, dict) and base_split["val_ratio"] == 0.2
    trial_split = observe_run(trial).record.spec.data.split
    assert isinstance(trial_split, DrawnSplit) and trial_split.val_ratio == 0.2


def test_a_runs_source_files_are_stored_against_its_project_like_its_data(
    tmp_path: Path,
) -> None:
    """A run's ``model_source.source_files`` are stored relative to its project.

    The run's record stores ``agent_code.py`` relative to the project, and reading the record
    back resolves it under the project.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import RUN_FILE, read_record, read_run
    from tcip_mcp.pipelines.schemas import train_config
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        detection_config,
        opened_run,
    )

    project = tmp_path / "project"
    project.mkdir()
    (project / "agent_code.py").write_text("", encoding="utf-8")
    config = detection_config(
        project / "ds",
        project=project,
        model_source={
            **BUILT_DETECTOR,
            "source_files": ["agent_code.py", *BUILT_DETECTOR["source_files"]],
        },
    )
    run_dir = opened_run(project, train_config(config, project).record())

    stored = read_record(run_dir / RUN_FILE)
    assert isinstance(stored, dict) and isinstance(stored["config"], dict)
    stored_source = stored["config"]["model_source"]
    assert isinstance(stored_source, dict)
    assert isinstance(stored_source["source_files"], list)
    assert stored_source["source_files"][0] == "agent_code.py"
    read = read_run(run_dir).spec.model_source.source_files
    assert read is not None
    assert read[0] == str((project / "agent_code.py").resolve())


def test_a_copied_projects_run_and_sweep_trial_read_the_copys_source_files(
    tmp_path: Path,
) -> None:
    """A copied project's run and sweep trial read their source files from the copy.

    The config paths a run or sweep records travel with the project: a copy's run reads its
    resolved dataset builder's source files from the copy, and a copy's sweep trial reads the
    model source files its sampled choice and its baseline name from the copy.
    """
    pytest.importorskip("torchvision")
    import shutil

    import tcip_store as ts
    from tcip_mcp.experiments import observe_run, observe_sweep
    from tcip_mcp.tools.training_tools import open_trial
    from tests._chain_fixtures import BESPOKE_MODELS, stated_detector_geometry
    from tests._training_values import sweep_space
    from tests._verified_checkpoint_fixtures import opened_run, opened_sweep
    from tests.test_dataset_source_seam import BESPOKE_DS, BESPOKE_DS_FILE
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset,
        _detection_config,
    )

    project = tmp_path / "project"
    images_dir = _canonical_dataset(project / "ds")
    config = _detection_config(project, images_dir)
    for name in ("agent_code.py", "other.py"):
        (project / name).write_text("", encoding="utf-8")
    bespoke = _detection_config(
        project,
        images_dir,
        dataset_source={
            "builder": BESPOKE_DS,
            "source_files": [str(project / "agent_code.py"), BESPOKE_DS_FILE],
        },
    )
    run = opened_run(
        project,
        {
            **bespoke,
            "model_source": {
                **bespoke["model_source"],
                "detector_geometry": stated_detector_geometry(),
            },
        },
    )
    choices = [
        [str(project / "agent_code.py"), BESPOKE_MODELS],
        [str(project / "other.py"), BESPOKE_MODELS],
    ]
    space = {
        **sweep_space(),
        "model_source.source_files": {
            "type": "categorical",
            "choices": choices,
        },
    }
    sweep = opened_sweep(
        project,
        config,
        param_space=space,
        baseline_params={
            "optimizer.head_lr": 0.001,
            "model_source.source_files": choices[1],
        },
    )

    copy = tmp_path / "copy"
    ts.release_root(project)
    shutil.copytree(project, copy)
    copied_run = observe_run(copy / run.relative_to(project)).record
    copied_sweep = copy / sweep.relative_to(project)
    sweep_input = observe_sweep(copied_sweep).record.input
    point = sweep_input.baseline_params
    assert point is not None
    trial = open_trial(copied_sweep, "t0", point)

    dataset_source = copied_run.resolution.data.dataset_source
    assert dataset_source is not None and dataset_source.source_files
    assert dataset_source.source_files[0] == str(
        (copy / "agent_code.py").resolve()
    )
    baseline_files = point["model_source.source_files"]
    assert isinstance(baseline_files, list)
    assert baseline_files[0] == str((copy / "other.py").resolve())
    axis = sweep_input.param_space["model_source.source_files"]
    assert isinstance(axis, dict) and isinstance(axis["choices"], list)
    first_choice = axis["choices"][0]
    assert isinstance(first_choice, list)
    assert first_choice[0] == str((copy / "agent_code.py").resolve())
    trial_files = observe_run(trial).record.spec.model_source.source_files
    assert trial_files is not None
    assert trial_files[0] == str((copy / "other.py").resolve())


def test_a_copied_sweeps_whole_mapping_choices_open_a_trial_naming_the_copy(
    tmp_path: Path,
) -> None:
    """A copied sweep's whole-mapping choices open a trial naming the copy's files.

    For a sweep whose choices are the whole ``model_source`` and ``data`` mappings, a trial a
    copied project opens on those choices names the copy's source file and images directory.
    """
    pytest.importorskip("torchvision")
    import shutil

    import tcip_store as ts
    from tcip_mcp.experiments import observe_run, observe_sweep
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep
    from tests.test_launch_artifact_anchoring_and_lineage import (
        _canonical_dataset,
        _detection_config,
    )

    project = tmp_path / "project"
    config = _detection_config(project, _canonical_dataset(project / "ds"))
    (project / "agent_code.py").write_text("", encoding="utf-8")
    model_source = {
        **config["model_source"],
        "source_files": [
            str(project / "agent_code.py"),
            *config["model_source"]["source_files"],
        ],
    }
    space = {
        "model_source": {"type": "categorical", "choices": [model_source]},
        "data": {"type": "categorical", "choices": [config["data"]]},
    }
    sweep = opened_sweep(project, config, param_space=space)

    copy = tmp_path / "copy"
    ts.release_root(project)
    shutil.copytree(project, copy)
    copied_sweep = copy / sweep.relative_to(project)
    stored = observe_sweep(copied_sweep).record.input.param_space
    point: dict[str, JsonValue] = {}
    for name, axis in stored.items():
        assert isinstance(axis, dict) and isinstance(axis["choices"], list)
        point[name] = axis["choices"][0]
    trial = observe_run(open_trial(copied_sweep, "t0", point)).record.spec

    assert trial.model_source.source_files is not None
    assert trial.model_source.source_files[0] == str(
        (copy / "agent_code.py").resolve()
    )
    assert trial.data.images_dir is not None
    assert Path(trial.data.images_dir).is_relative_to(copy.resolve())


def _authored_checkpoint(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, Path]:
    """Return a builder module's name and the checkpoint a run over it registered.

    The run completes in a new project at ``project`` over a uniquely named builder module at
    ``project/<module>.py`` importing ``project/<module>_helper.py``, both listed as its source
    files beside the bespoke models module.
    """
    import sys
    import uuid

    from tests._chain_fixtures import BESPOKE_MODELS
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        registered_checkpoint,
    )
    from tests._web_fixtures import new_project

    module = f"builder_{uuid.uuid4().hex[:8]}"
    new_project(project)
    (project / f"{module}_helper.py").write_text(
        "BOX_NMS_THRESH = 0.4\n", encoding="utf-8"
    )
    (project / f"{module}.py").write_text(
        f"import {module}_helper\n\n\n"
        "def build(**kwargs):\n"
        "    from tests.bespoke_models import build_bespoke_detection\n\n"
        f"    kwargs['box_nms_thresh'] = {module}_helper.BOX_NMS_THRESH\n"
        "    return build_bespoke_detection(**kwargs)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    return module, Path(
        registered_checkpoint(
            project,
            model_source={
                **BUILT_DETECTOR,
                "builder": f"{module}:build",
                "source_files": [
                    str(project / f"{module}.py"),
                    str(project / f"{module}_helper.py"),
                    BESPOKE_MODELS,
                ],
            },
        )
    )


def _built_from(checkpoint: Path, project: Path) -> Path:
    """Return the file the builder of ``checkpoint``, loaded for ``project``, was defined in.

    The model is built from the checkpoint's model source
    (``model_build.build_from_model_source``) before the builder is imported and its file read.
    """
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.model_build import (
        build_from_model_source,
        import_source_builder,
        recorded_model_dims,
    )

    loaded = load_registered_checkpoint(checkpoint, project=project)
    spec = loaded.spec
    build_from_model_source(
        spec.model_source,
        loaded.layout,
        recorded_model_dims(spec),
        loaded.geometry,
    )
    build = import_source_builder(spec.model_source.builder, loaded.layout)
    return Path(build.__code__.co_filename).resolve()


def test_a_checkpoint_builds_from_its_runs_snapshot_in_its_own_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint loaded in its own project builds from its run's snapshot.

    A run completed through the platform's own producer, loaded in its own project, binds its
    builder to the copy its run snapshotted and builds.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR

    project = tmp_path / "project"
    _module, checkpoint = _authored_checkpoint(project, monkeypatch)

    built = _built_from(checkpoint, project)

    assert (
        built.is_relative_to(project.resolve()) and SNAPSHOT_DIR in built.parts
    )


def test_a_copied_project_builds_from_the_copys_snapshot_with_the_original_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied project's checkpoint builds from the copy's snapshot with the original gone.

    The whole project is copied to a new root and the original deleted: the copy's checkpoint
    builds from the copy's own snapshot, its builder and the helper declared beside it alike,
    with the original's modules of those names still loaded.
    """
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
    assert (
        helper.is_relative_to(copy.resolve()) and SNAPSHOT_DIR in helper.parts
    )


def test_a_run_carried_whole_builds_from_its_snapshot_never_a_same_named_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run carried whole builds from its snapshot, not from a same-named file.

    A run carried whole into another project (``archive_project``, then ``import_project``)
    whose own file at the author's relative location is a different module of the same name
    builds from a file under the carried project other than that one.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.project_tools import (
        ZipBundle,
        import_project,
        write_archive,
    )

    project, carried = tmp_path / "project", tmp_path / "carried"
    module, checkpoint = _authored_checkpoint(project, monkeypatch)
    bundle = tmp_path / "bundle.zip"
    assert "error" not in write_archive(
        project.resolve(),
        ZipBundle(output_path=str(bundle)),
        include_models=True,
    )
    assert "error" not in import_project(str(bundle), str(carried))
    (carried / f"{module}.py").write_text(
        "def build(**kwargs):\n    raise AssertionError('the same-named file answered')\n",
        encoding="utf-8",
    )

    built = _built_from(carried / checkpoint.relative_to(project), carried)

    assert (
        built.is_relative_to(carried.resolve())
        and built != (carried / f"{module}.py").resolve()
    )


def test_a_checkpoint_moved_without_its_run_refuses_before_importing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint moved without its run is refused at load, and nothing is imported.

    A checkpoint file carried alone into a project that holds no snapshot of its run refuses
    at load, naming the run directory its source lies in and the primitive that carries a run
    whole, and imports nothing.
    """
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
    monkeypatch.setattr(
        model_build,
        "import_source_builder",
        lambda dotted, layout: loaded.append(dotted),
    )

    with pytest.raises(ValueError, match="archive_project") as refused:
        prepare(
            load_registered_checkpoint(alone, project=elsewhere),
            Stated(tile=False, rationale=RATIONALE),
        )

    assert checkpoint.parent.name in str(refused.value)
    assert loaded == []


def test_a_run_missing_its_metrics_log_answers_its_own_error_at_the_checkpoint_doors(
    tmp_path: Path,
) -> None:
    """A run missing its metrics log answers its own error at the checkpoint doors.

    An existing registered checkpoint whose project holds a run missing its metrics log is
    not an absent checkpoint: evaluation and triage answer the broken record's own error.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import METRICS_FILE
    from tcip_mcp.tools.feedback_tools import triage_predictions
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    project = _dataset(tmp_path).root
    checkpoint = Path(
        registered_checkpoint(
            project,
            data={
                "num_channels": 3,
                "scope": {"subject": SUBJECT},
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        )
    )
    (checkpoint.parent / METRICS_FILE).unlink()
    spelling = _relative(checkpoint, project)

    evaluated = evaluate_model(
        project,
        spelling,
        CAPTURE,
        stated=_stated(),
        iou_threshold=IOU_THRESHOLD,
    )
    triaged = triage_predictions(
        project, spelling, CAPTURE, 0.3, 0.8, RATIONALE
    )

    for answer in (evaluated, triaged):
        assert (
            f"is missing {checkpoint.parent / METRICS_FILE}" in answer["error"]
        ), answer


def test_evaluation_takes_a_run_id_and_a_relative_checkpoint_path(
    tmp_path: Path,
) -> None:
    """Evaluation takes a completed run's id and a relative checkpoint path alike.

    A completed run of the project named by its id and the checkpoint it completed named by a
    relative path both evaluate.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    project = _dataset(tmp_path).root
    _labeled(project)
    checkpoint = Path(
        registered_checkpoint(
            project,
            data={
                "num_channels": 3,
                "scope": {"subject": SUBJECT},
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        )
    )

    by_id = evaluate_model(
        project,
        checkpoint.parent.name,
        CAPTURE,
        stated=_stated(),
        iou_threshold=IOU_THRESHOLD,
    )
    by_path = evaluate_model(
        project,
        _relative(checkpoint, project),
        CAPTURE,
        stated=_stated(),
        iou_threshold=IOU_THRESHOLD,
    )

    assert "error" not in by_id and "error" not in by_path, (by_id, by_path)


def test_a_resumed_run_records_the_project_rooted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative ``resume_from`` is recorded as the project-rooted checkpoint.

    ``resume_from`` spelled relative names a checkpoint a run of the project wrote: the run
    record and the launch's audit line both carry that location.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.experiments import read_run
    from tcip_mcp.tools.training_tools import RunLaunched, launch_training
    from tests._audit_fixtures import audit_rows
    from tests._producer_fixtures import fake_popen
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    project = tmp_path / "project"
    project.mkdir()
    checkpoint = Path(registered_checkpoint(project))
    expected = str(checkpoint.resolve())
    fake_popen(monkeypatch, [])
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = launch_training(
        project,
        _relative_config(project, Path("ds")),
        resume_from=_relative(checkpoint, project),
        actor=None,
    )

    assert isinstance(result, RunLaunched), result
    assert read_run(Path(result.output_dir)).resume_from == expected
    (row,) = audit_rows(project, "launch_training")
    assert row["arguments"]["resume_from"] == expected


def test_a_config_with_a_relative_path_and_no_project_is_refused_by_name() -> (
    None
):
    """A config with a relative path validated where no project is held is refused by name.

    Validation returns no spec and an issue naming ``data.images_dir`` as needing an absolute
    path.
    """
    from tcip_mcp.pipelines.schemas import checked_train_config
    from tests._chain_fixtures import weights_statistics

    spec, issues = checked_train_config(
        {
            "model_source": weights_statistics(
                {"builder": "m:f", "task": "detection"}
            ),
            "data": {"images_dir": image_dir("", UNDATED_BUCKET).as_posix()},
        }
    )
    assert spec is None
    assert any(
        "data.images_dir" in issue and "absolute path" in issue
        for issue in issues
    )


def test_the_preflight_command_reads_a_relative_config_under_its_project(
    project: Path, tmp_path: Path
) -> None:
    """The preflight command reads a relative ``--config @<file>`` under ``--project``.

    Run from a directory holding a decoy of the same name, the project's own config is the one
    validated.
    """
    import json

    from tests._cli_fixtures import run_tcip
    from tests.test_preflight_config_script import _fixture_config

    _fixture_config(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()
    (cwd / "config.json").write_text("{}", encoding="utf-8")

    result = run_tcip(
        "preflight-config",
        [
            *("--config", "@config.json", "--project", str(project)),
            *("--user", "tester"),
        ],
        cwd=cwd,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["valid"] is True


def test_the_identity_command_reads_a_relative_dataset_root_under_its_project(
    tmp_path: Path,
) -> None:
    """The identity command reads a relative dataset root under ``--project``.

    The check run from a working directory outside the project reports the registered
    dataset's own outcome.
    """
    import json

    from tcip_mcp.tools.project_tools import register_dataset
    from tests._cli_fixtures import run_tcip
    from tests._web_fixtures import new_project
    from tests.test_check_dataset_identity_script import _real_dataset

    project = new_project(tmp_path / "project").root
    _real_dataset(project / "ds")
    assert "error" not in register_dataset(
        project, str(project / "ds"), "chestnut", actor=None
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    completed = run_tcip(
        "check-dataset-identity",
        [
            *("--dataset-root", "ds", "--project", str(project)),
            *("--user", "tester"),
        ],
        cwd=cwd,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["state"] == "unchanged"


def test_the_score_command_reads_a_relative_path_under_its_project(
    project: Path, tmp_path: Path
) -> None:
    """The score command reads a relative ``--path`` under ``--project`` from outside it."""
    import json

    from tests._cli_fixtures import run_tcip
    from tests.test_score_predictions_script import SCORED_AT, _fixture

    image, bucket = _fixture(project)
    cwd = tmp_path.parent / "operator_cwd"
    cwd.mkdir()

    result = run_tcip(
        "score-predictions",
        [
            "--path",
            _relative(image, project),
            "--bucket",
            bucket,
            "--project",
            str(project),
            *SCORED_AT,
            *("--user", "tester"),
        ],
        cwd=cwd,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["tp"] == 1


def test_the_doctor_reads_the_project_named_by_a_relative_root(
    tmp_path: Path,
) -> None:
    """``tcip diagnose-project --project .`` run inside a project answers what the tool answers.

    The command's printed findings are those ``diagnose_project`` returns over the project.
    """
    import json

    from tcip_mcp.tools.doctor_tools import diagnose_project
    from tests._cli_fixtures import run_tcip

    project = _dataset(tmp_path).root

    result = run_tcip(
        "diagnose-project",
        ["--project", ".", "--user", "tester"],
        cwd=project,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == json.loads(
        json.dumps(diagnose_project(project))
    )


@pytest.mark.parametrize("door", sorted(DOORS))
def test_one_relative_path_names_one_project_location_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, door: str
) -> None:
    """A door answers one relative spelling alike from two working directories.

    The door answers one relative spelling from two working directories outside two like
    projects, each directory holding nothing the spelling names, with a successful result that is
    the same: the location it recorded, spelled against its project, or what it read there.
    """
    found = []
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        project = _dataset(tmp_path / name).root
        cwd = tmp_path / f"{name}_cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        observed = DOORS[door](project)
        found.append(
            _relative(observed, project)
            if isinstance(observed, Path)
            else observed
        )
    assert found[0] == found[1]


def test_a_per_plant_delivery_reads_and_writes_under_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-plant delivery reads its buckets and writes its CSV under the project.

    ``deliver_per_plant_csv`` with ``dataset_root="ds"`` and ``output_path="out.csv"``, shipped
    under the breeder's recorded acknowledgment, reads the buckets under the project's dataset
    and writes its CSV under the project, from two working directories outside two like projects,
    leaving each working directory empty.
    """
    pytest.importorskip("torch")
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import DeliveryRefusedError, record_acknowledgment
    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )
    from tcip_mcp.pipelines.postprocessing.phenology import population
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv
    from tests import _trait_fixtures as fx, csv_rows
    from tests.test_deliver_per_plant_csv import (
        KIND,
        TWO_PLANTS,
        _bucket,
        _mapped,
        _records,
    )

    found = []
    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        fx.seed_delivery_traits(project)
        fx.seed_confirmed_aggregate(
            project, "stem_count", value_keys=["count"]
        )
        dataset_root, date = _mapped(project, TWO_PLANTS)
        bucket = _bucket(
            project, dataset_root, date, {"P1": 2, "P2": 1}, name="run"
        )
        records = _records(("P1", 2), ("P2", 1))
        with pytest.raises(DeliveryRefusedError) as refused:
            deliver_per_plant_aggregate(
                project,
                records,
                str(project / "unacknowledged.csv"),
                value_key="count",
                delivered_phenotype="stem_count",
                delivery_kind=KIND,
                buckets=[read_bucket(dataset_root, bucket)],
                plants=population(["P1", "P2"]),
                crop="currant",
                door="deliver_per_plant_csv",
                actor=None,
            )
        act = record_acknowledgment(
            project,
            acknowledged_by="user:breeder",
            reason="a look",
            result_sha256=str(refused.value.result_sha256),
        )
        cwd = tmp_path / f"{name}_cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        delivered = deliver_per_plant_csv(
            project,
            records,
            "out.csv",
            "stem_count",
            KIND,
            ["P1", "P2"],
            _relative(dataset_root, project),
            [bucket],
            value_key="count",
            crop="currant",
            acknowledgment_id=act.acknowledgment_id,
            actor=None,
        )

        assert "error" not in delivered, delivered
        assert (project / "out.csv").is_file() and list(cwd.iterdir()) == []
        found.append(
            sorted(row["plant_id"] for row in csv_rows(project / "out.csv"))
        )
    assert found[0] == found[1]


def test_a_phenology_measurement_reads_the_buckets_under_the_project_rooted_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phenology delivery reads the buckets under its project-rooted dataset.

    The phenology delivery door over a series the platform's producers built, its dataset
    root spelled relative to the project, reads the project's own buckets from two working
    directories outside two like projects: each reaches the delivery's own refusal of the
    unassessed buckets it read there, and leaves its working directory empty.
    """
    pytest.importorskip("torchvision")
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones
    from tests._chain_fixtures import PLANTS, attributed_series

    for name in ("first", "second"):
        project = tmp_path / name
        project.mkdir()
        series = attributed_series(
            project, fractions=(0.0, 1.0), assessed=False
        )
        cwd = tmp_path / f"{name}_cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        refused = deliver_phenology_milestones(
            project,
            series.trait,
            series.mapping_name,
            _relative(series.root, project),
            list(series.buckets.values()),
            "out.csv",
            list(PLANTS),
            actor=None,
        )
        assert "no assessment answers for bucket" in refused.get(
            "error", ""
        ), refused
        assert list(cwd.iterdir()) == []

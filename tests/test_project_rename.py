"""Renaming a project changes its display name and nothing else, and a project whose directory
moves still resolves every run, data location and bucket its records name.

``tcip_mcp.project_record.rename_project`` and ``POST /api/projects/rename``; every project is made
through the platform's own creation door under this test's workspace (``tmp_path.parent``).
"""

from __future__ import annotations

from tcip_mcp.dataset_layout import UNDATED_BUCKET

import os
from pathlib import Path

import tcip_store as ts
from tests._audit_fixtures import audit_rows
from tests._web_fixtures import named_project


def _files(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` but the databases, by its path relative to ``root``."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and not p.name.startswith("store.db")}


def _held(root: Path, *, but: set[str]) -> dict[tuple, object]:
    """Every record and log the databases under ``root`` hold, read through the seam, but the
    stores ``but`` names, by root relative to ``root``, store and parts."""
    from tcip_store.file_backend import database_roots

    return {(base.relative_to(root).as_posix(), store, key.parts):
            (ts.read(key, default=None), ts.read_log(key).records)
            for base in database_roots(root) for store in ts.stores(str(base)) if store not in but
            for key in ts.keys(store, str(base))}


def test_a_rename_changes_the_display_name_and_leaves_the_directory_and_records_as_they_were(
    client, tmp_path,
):
    from tcip_mcp.experiments import run_rows
    from tcip_mcp.project_record import read_record
    from tcip_mcp.tools.project_tools import read_datasets, register_dataset
    from tests._verified_checkpoint_fixtures import finished_run

    ws = tmp_path.parent
    project, project_id = named_project(ws / "valley_block", "Valley block")
    finished_run(project, experiment_id="exp-before")
    (project / "ds").mkdir()
    assert "error" not in register_dataset(project, str(project / "ds"), "currant")
    from tcip_mcp.audit import AUDIT_LOG_STORE
    from tcip_mcp.project_record import PROJECT_RECORD_STORE

    renamed = {AUDIT_LOG_STORE, PROJECT_RECORD_STORE}
    files, runs, datasets = _files(project), run_rows(project), read_datasets(project)
    held, audit_lines = _held(project, but=renamed), audit_rows(project)

    resp = client.post("/api/projects/rename", json={
        "id": project_id, "display_name": "Valley block, north half", "user": "tester"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"id": project_id, "display_name": "Valley block, north half",
                           "previous_display_name": "Valley block"}
    assert read_record(project)["display_name"] == "Valley block, north half"
    assert sorted(p.name for p in ws.iterdir() if (p / ".tcip").is_dir()) == ["valley_block"]
    assert _files(project) == files
    assert _held(project, but=renamed) == held
    assert run_rows(project) == runs
    assert read_datasets(project) == datasets
    assert audit_rows(project)[:-1] == audit_lines
    (line,) = audit_rows(project, "project_renamed")
    assert line["arguments"]["previous_display_name"] == "Valley block"
    assert line["actor"] == "user:tester"
    listed = {p["id"]: p for p in client.get("/api/projects").json()["projects"]}
    assert listed[project_id]["display_name"] == "Valley block, north half"


def test_a_display_name_the_record_refuses_answers_400_and_changes_nothing(client, tmp_path):
    from tcip_mcp.project_record import read_record

    project, project_id = named_project(tmp_path.parent / "valley_block", "Valley block")

    resp = client.post("/api/projects/rename", json={
        "id": project_id, "display_name": "   ", "user": "tester"})

    assert resp.status_code == 400
    assert read_record(project)["display_name"] == "Valley block"
    assert audit_rows(project, "project_renamed") == []


def test_a_rename_naming_no_one_answers_400_and_changes_nothing(client, tmp_path, monkeypatch):
    from tcip_mcp.project_record import read_record

    monkeypatch.setenv("TCIP_USER", "osuser")
    project, project_id = named_project(tmp_path.parent / "valley_block", "Valley block")

    resp = client.post("/api/projects/rename", json={
        "id": project_id, "display_name": "Hill block", "user": "  "})

    assert resp.status_code == 400 and "names no one" in resp.text
    assert read_record(project)["display_name"] == "Valley block"
    assert audit_rows(project, "project_renamed") == []


def test_an_id_no_project_holds_answers_404(client, tmp_path):
    named_project(tmp_path.parent / "valley_block", "Valley block")

    resp = client.post("/api/projects/rename", json={
        "id": "0" * 12, "display_name": "Hill block", "user": "tester"})

    assert resp.status_code == 404


def test_a_project_moved_after_training_resolves_every_path_its_records_name(
    tmp_path, monkeypatch,
):
    """Every in-project path a record holds is spelled relative to the project, so a project
    whose directory moves resolves its resumed run's checkpoint, its completed run's checkpoint,
    its registry entry, a run's data locations and partition, and a published bucket's documents
    at their new location, and no run record names the directory it left."""
    import json

    from tcip_mcp.experiments import (
        RUN_FILE, SWEEP_FILE, create_run_directory, experiment_dir, find_run, find_sweep, observe,
        write_record,
    )
    from tcip_mcp.model_registry import ModelRegistry
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.pipelines.data.split_construction import partition_samples
    from tcip_mcp.pipelines.training.subprocess_worker import prepare_run_context
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._verified_checkpoint_fixtures import detection_config, finished_run, opened_run

    ws = tmp_path.parent
    project, _ = named_project(ws / "valley_block", "Valley block")
    first = finished_run(project, experiment_id="exp-first")
    checkpoint = Path(observe(first).checkpoint["path"])
    opened_run(project, detection_config(project / "data"),
               experiment_id="exp-resumed", resume_from=str(checkpoint))
    bucket = "live/2-11-26"
    published = run_inference(project, checkpoint_path=str(checkpoint),
                              images_dir=str(project / "data" / "images" / UNDATED_BUCKET),
                              bucket=bucket,
                              stated=Stated(tile=False))
    assert "error" not in published, published
    other_images = str(project / "data" / "other")
    sweep = create_run_directory(experiment_dir("study", project=project))
    write_record(sweep / SWEEP_FILE, {"input": {
        "base_config": {"data": {"images_dir": str(project / "data" / "images")}},
        "baseline_params": {"data.images_dir": other_images},
        "param_space": {"data.labels_dir": {"type": "categorical", "choices": [other_images]}},
    }})

    moved = ws / "valley_block_moved"
    ts.release_root(project)
    os.rename(project, moved)

    left_behind = json.dumps(str(project.resolve() / "x"))[1:-2]
    for run_file in (moved / ".tcip" / "experiments").glob(f"*/{RUN_FILE}"):
        assert left_behind not in run_file.read_text(encoding="utf-8"), run_file

    observed = observe(find_run("exp-resumed", project=moved))
    resumed = prepare_run_context(observed)
    assert Path(resumed.resume_from) == (moved / checkpoint.relative_to(project)).resolve()
    assert Path(resumed.resume_from).is_file()
    for data in (observed.record["config"]["data"], observed.record["resolved"]["data"]):
        assert Path(data["images_dir"]).is_dir()
        assert Path(data["images_dir"]).is_relative_to(moved.resolve())
    partition = observed.record["resolved"]["partition"]
    for sample in partition_samples(partition):
        assert Path(sample.source).is_file() and ts.exists(sample.ground_truth)
        assert Path(sample.ground_truth.root).is_relative_to(moved.resolve())

    moved_checkpoint = Path(observe(find_run("exp-first", project=moved)).checkpoint["path"])
    assert moved_checkpoint.is_file() and moved_checkpoint.is_relative_to(moved)
    (entry,) = [m for m in ModelRegistry(str(moved)).list_models() if m["name"] == "exp-first"]
    assert Path(entry["checkpoint_path"]).resolve() == moved_checkpoint.resolve()

    record = read_bucket(moved / "data", bucket)
    assert record.document_keys and all(ts.exists(key) for key in record.document_keys)

    moved_other = str(moved.resolve() / "data" / "other")
    swept = observe(find_sweep("study", project=moved)).record["input"]
    assert swept["base_config"]["data"]["images_dir"] == str(moved.resolve() / "data" / "images")
    assert swept["baseline_params"] == {"data.images_dir": moved_other}
    assert swept["param_space"]["data.labels_dir"]["choices"] == [moved_other]

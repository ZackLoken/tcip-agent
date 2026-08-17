"""The whole store catalogue in one import, and where each store's entries live on disk.

A store is registered by importing the module that declares it, so a tool that has to reason
about every store (writing a database back out as files, or reading files into one) needs every
owning module imported first. No single package's own import set covers them all: the web
package owns the learning-capture log the MCP server never imports. This module is that one
import set, and :data:`ADOPTION_SOURCES` beside it states, per record and log store, the kind
of root it hangs off and which of its key parts are constants.

The second half exists because a locator alone cannot say which store owns a file: thirteen
stores place a single json document under ``.tcip/state``, and their locators claim each
other's files exactly. What tells them apart is the constant each store's key constructor
spells, which is what the patterns below carry.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store import registered_stores
from tcip_store.adoption import ANY, StoreSource, literal, starting_with

from tcip_annotation import format_io, json_io, review_engine  # noqa: F401
from tcip_mcp import (  # noqa: F401
    audit,
    dataset_layout,
    experiments,
    model_registry,
    operationalization,
    project_status,
    traits,
    web_client,
    workspace,
)
from tcip_mcp.pipelines import model_build, resolution  # noqa: F401
from tcip_mcp.pipelines.data import band_groups, splits  # noqa: F401
from tcip_mcp.pipelines.feedback import materialize  # noqa: F401
from tcip_mcp.pipelines.postprocessing import plant_mapping  # noqa: F401
from tcip_mcp.pipelines.training import evaluation, generic_trainer, hpo  # noqa: F401
from tcip_mcp.tools import (  # noqa: F401
    data_tools,
    inference_tools,
    meta_tools,
    project_tools,
    training_tools,
    vision_tools,
)
from tcip_web import agent_learning_capture, jobstore  # noqa: F401
from tcip_web import state as web_state  # noqa: F401
from tcip_web.routes import canvas, sessions  # noqa: F401

ROOT = "root"
"""A project or dataset root: the directory holding ``images/``, ``annotations/`` and ``.tcip/``."""

STATE = "state"
"""A root's ``.tcip/state`` directory, the root the review shards and per-trait records hang off."""

EXPERIMENTS = "experiments"
"""A root's ``.tcip/experiments`` directory, the one root every experiment's members share."""

WORKSPACE = "workspace"
"""The workspace directory holding the project folders and the active-project marker."""

HPO_ROOT = "hpo_root"
"""A root's ``.tcip/hpo`` directory, holding one study result and one manifest per sweep."""

SWEEP = "sweep"
"""One sweep's directory under the hpo root, holding a directory per trial."""

SPLITS = "splits"
"""A partition's output directory: one document per split plus the manifest describing them."""

CURATED = "curated"
"""A curated dataset's output directory."""

RUN = "run"
"""A training or evaluation run's output directory."""

PREDICTION_BUCKET = "prediction_bucket"
"""One prediction bucket directory, where a run's operating-point stamps sit beside its output."""

LAYOUTS = (
    ROOT,
    STATE,
    EXPERIMENTS,
    WORKSPACE,
    HPO_ROOT,
    SWEEP,
    SPLITS,
    CURATED,
    RUN,
    PREDICTION_BUCKET,
)
"""Every kind of directory a root can be, for an operator naming one on a command line."""

ADOPTION_SOURCES: dict[str, StoreSource] = {
    # a project or dataset root
    "image_status": StoreSource(ROOT, (literal("image_status"),)),
    "image_status_digest": StoreSource(ROOT, (literal("image_status_digest"),)),
    "view_coverage": StoreSource(ROOT, (literal("view_coverage"),)),
    "region_completeness": StoreSource(ROOT, (literal("region_completeness"),)),
    "region_completeness_digest": StoreSource(ROOT, (literal("region_completeness_digest"),)),
    "gui_snapshot": StoreSource(ROOT, (literal("gui"),)),
    "canvas_meta": StoreSource(ROOT, (literal("canvas_live"),)),
    "canvas_geometry": StoreSource(ROOT, (literal("canvas_shapes"),)),
    "project_status": StoreSource(ROOT, (literal("project_status"),)),
    "annotation_stats": StoreSource(ROOT, (literal("annotation_stats"),)),
    "ray_dashboard": StoreSource(ROOT, (literal("ray_dashboard"),)),
    "backend_port": StoreSource(ROOT, (literal("web_port"),)),
    "model_registry": StoreSource(ROOT, (literal("registry"),)),
    "dataset_registry": StoreSource(ROOT, (literal("datasets"),)),
    "audit_log": StoreSource(ROOT, (literal("audit"),)),
    "learning_capture": StoreSource(ROOT, (literal("learning_capture"),)),
    "proposal_staging": StoreSource(ROOT, (starting_with("proposals_"),)),
    # the one registry the platform persists (tcip_web/routes/inference.py); another name
    # would land in unaccounted_files rather than being taken for a neighbouring document
    "job_registry": StoreSource(ROOT, (literal("inference_jobs"),)),
    "confidence_sweep": StoreSource(ROOT, (ANY,)),
    "cal_holdout_split_lock": StoreSource(ROOT, (ANY,)),
    # <root>/.tcip/state
    "plant_mapping": StoreSource(STATE, (literal("plant_mapping"),)),
    "review_verdicts": StoreSource(STATE, (ANY, ANY)),
    "trait_operationalizations": StoreSource(STATE, (ANY, ANY)),
    # <root>/.tcip/experiments
    "experiment_config": StoreSource(EXPERIMENTS, (ANY, literal("config"))),
    "experiment_status": StoreSource(EXPERIMENTS, (ANY, literal("status"))),
    "experiment_lineage": StoreSource(EXPERIMENTS, (ANY, literal("lineage"))),
    "experiment_artifacts": StoreSource(EXPERIMENTS, (ANY, literal("artifacts"))),
    "experiment_env": StoreSource(EXPERIMENTS, (ANY, literal("env"))),
    "experiment_split": StoreSource(EXPERIMENTS, (ANY, literal("split"))),
    "experiment_metrics": StoreSource(EXPERIMENTS, (ANY, literal("metrics"))),
    "experiment_validations": StoreSource(EXPERIMENTS, (ANY, literal("validations"))),
    "model_snapshot_manifest": StoreSource(EXPERIMENTS, (ANY, literal("manifest"))),
    # the workspace
    "workspace_active_project": StoreSource(WORKSPACE, (literal(".active"),)),
    # <root>/.tcip/hpo and one sweep under it
    "hpo_study_result": StoreSource(HPO_ROOT, (ANY,)),
    "hpo_sweep_manifest": StoreSource(HPO_ROOT, (ANY, literal("manifest"))),
    "hpo_trial_config": StoreSource(SWEEP, (ANY, literal("resolved_config"))),
    "hpo_trial_metrics": StoreSource(SWEEP, (ANY, literal("metrics"))),
    # output directories a caller names
    "split_manifest": StoreSource(SPLITS, (literal("split_manifest"),)),
    "split_stem_list": StoreSource(SPLITS, (ANY,)),
    "curated_manifest": StoreSource(CURATED, (literal("curated_manifest"),)),
    "evaluation_results": StoreSource(RUN, (literal("test_results"),)),
    "run_launch_config": StoreSource(RUN, (literal("launch_config"),)),
    "operating_point_sidecar": StoreSource(PREDICTION_BUCKET, (literal("operating_point"),)),
    "classifier_operating_point_sidecar": StoreSource(
        PREDICTION_BUCKET, (literal("classifier_operating_point"),)
    ),
    "ordinal_operating_point_sidecar": StoreSource(
        PREDICTION_BUCKET, (literal("ordinal_operating_point"),)
    ),
    "regression_operating_point_sidecar": StoreSource(
        PREDICTION_BUCKET, (literal("regression_operating_point"),)
    ),
    "resolve_scale_sidecar": StoreSource(PREDICTION_BUCKET, (literal("resolve_scale"),)),
}


def bootstrapped_stores() -> tuple[str, ...]:
    """Every store this module's imports register, which is every store the platform declares."""
    return registered_stores()


def project_roots(project_root: str | Path) -> tuple[tuple[str, str], ...]:
    """The roots a whole project's records live in, each with the layout it is.

    The registered dataset roots come from the project's own registry rather than from a
    directory guess, so a dataset that lives outside the project tree still travels.
    """
    root = Path(project_root).absolute()
    roots: list[tuple[str, str]] = [
        (str(root), ROOT),
        (str(root / ".tcip" / "state"), STATE),
        (str(root / ".tcip" / "experiments"), EXPERIMENTS),
    ]
    seen = {os.path.normcase(str(root))}
    for entry in project_tools.read_datasets(root):
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            continue
        dataset_root = Path(path).absolute()
        if os.path.normcase(str(dataset_root)) in seen:
            continue
        seen.add(os.path.normcase(str(dataset_root)))
        roots.append((str(dataset_root), ROOT))
        roots.append((str(dataset_root / ".tcip" / "state"), STATE))
    return tuple(roots)

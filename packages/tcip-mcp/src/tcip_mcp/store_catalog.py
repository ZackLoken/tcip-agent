"""The whole store catalog in one import: every module that registers a store.

Importing this module registers every store's descriptor, so a caller invoked on its own sees the
same stores a running MCP server does. Where each store's entries sit under a root is
:mod:`tcip_store.layout_claims`.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store import registered_stores
from tcip_store.layout_claims import (
    CURATED, EXPERIMENTS, HPO_ROOT, PREDICTION_BUCKET, ROOT, RUN, SPLITS, STATE, SWEEP,
)

from tcip_annotation import json_io, review_engine  # noqa: F401
from tcip_mcp import (  # noqa: F401
    audit,
    dataset_layout,
    experiments,
    model_registry,
    operationalization,
    project_record,
    project_status,
    traits,
    web_client,
    workspace,
)
from tcip_mcp.pipelines import model_build, resolution  # noqa: F401
from tcip_mcp.pipelines.data import band_groups, selection, splits  # noqa: F401
from tcip_mcp.pipelines.data.split_construction import bound_selection_dir
from tcip_mcp.pipelines.feedback import materialize  # noqa: F401
from tcip_mcp.pipelines.postprocessing import plant_mapping  # noqa: F401
from tcip_mcp.pipelines.training import eval_runners, generic_trainer, hpo  # noqa: F401
from tcip_mcp.tools import (  # noqa: F401
    data_tools,
    inference_tools,
    meta_tools,
    project_tools,
    proposal_tools,
    training_tools,
)


def bootstrapped_stores() -> tuple[str, ...]:
    """Every store this module's imports register, which is every store the platform declares."""
    return registered_stores()


def _add(roots: list[tuple[str, str]], seen: set[tuple[str, str]], path: Path, layout: str) -> None:
    """Append ``(path, layout)`` unless this exact path/layout pair is already in; a directory can
    be two kinds of root at once.
    """
    key = (os.path.normcase(str(path)), layout)
    if key in seen:
        return
    seen.add(key)
    roots.append((str(path), layout))


def _add_if_dir(roots: list[tuple[str, str]], seen: set[tuple[str, str]], path: Path,
                 layout: str) -> None:
    """Like :func:`_add`, but skips a record-named root whose directory no longer exists."""
    if path.is_dir():
        _add(roots, seen, path, layout)


def project_roots(project_root: str | Path) -> tuple[tuple[str, str], ...]:
    """The roots a whole project's records live in, each with the layout it is.

    Every root here comes from a record the project itself holds, or from walking a directory a
    record already named, never from a directory guessed with no record behind it: the registered
    dataset roots from the project's own registry, an HPO sweep's directory found under the
    project's own recorded HPO root, and a run's output directory, selection binding,
    curated-dataset artifact and prediction-bucket lineage from that run's own
    ``experiments/<id>/`` members. A recorded directory that no longer exists is skipped.

    Per layout:

    - ``ROOT``/``STATE``/``EXPERIMENTS``: the project's own root, plus ``.tcip/state`` and
      ``.tcip/experiments`` under it; ``ROOT`` and ``STATE`` for every registered dataset too.
    - ``HPO_ROOT``: ``.tcip/hpo`` under the project root
      (:func:`tcip_mcp.tools.training_tools.hpo_root`'s default).
    - ``SWEEP``: one root per immediate subdirectory of the project's own HPO root
      (:func:`~tcip_mcp.tools.training_tools.sweep_dir`).
    - ``RUN``: each experiment's own ``status.json["output_dir"]``.
    - ``SPLITS``: each experiment's own ``split.json["selection_binding"]["selection_dir"]``,
      present only for a run bound to an existing selection.
    - ``CURATED``: each experiment's own ``artifacts.json["curated_dataset"]["path"]``.
    - ``PREDICTION_BUCKET``: every model directory and its date subdirectories under each dataset's
      own ``predictions/`` tree, live and cleared alike
      (:func:`tcip_mcp.dataset_layout.prediction_bucket_dirs` with ``include_cleared=True``), and
      each experiment's own ``lineage.json["predictions"]``.
    """
    root = Path(project_root).absolute()
    roots: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    _add(roots, seen, root, ROOT)
    _add(roots, seen, root / ".tcip" / "state", STATE)
    _add(roots, seen, root / ".tcip" / "experiments", EXPERIMENTS)

    hpo_dir = training_tools.hpo_root(root=root)
    _add(roots, seen, hpo_dir, HPO_ROOT)
    if hpo_dir.is_dir():
        for entry in sorted(p for p in hpo_dir.iterdir() if p.is_dir()):
            _add(roots, seen, entry, SWEEP)

    for exp_id in experiments.experiment_ids_with_status(root):
        status = experiments.read_member(experiments.status_key(exp_id, root=root), {})
        output_dir = status.get("output_dir") if isinstance(status, dict) else None
        if output_dir:
            _add_if_dir(roots, seen, Path(output_dir).absolute(), RUN)

        split_doc = experiments.read_member(experiments.split_key(exp_id, root=root), {})
        selection_dir = bound_selection_dir(split_doc)
        if selection_dir:
            _add_if_dir(roots, seen, Path(selection_dir).absolute(), SPLITS)

        artifacts = experiments.read_member(experiments.artifacts_key(exp_id, root=root), {})
        curated = artifacts.get("curated_dataset") if isinstance(artifacts, dict) else None
        curated_path = curated.get("path") if isinstance(curated, dict) else None
        if curated_path:
            _add_if_dir(roots, seen, Path(curated_path).absolute(), CURATED)

        lineage = experiments.read_member(experiments.lineage_key(exp_id, root=root), {})
        predictions = lineage.get("predictions") if isinstance(lineage, dict) else None
        if predictions:
            _add_if_dir(roots, seen, Path(predictions).absolute(), PREDICTION_BUCKET)

    for dataset_entry in project_tools.read_datasets(root):
        dataset_root = project_tools.dataset_entry_path(root, dataset_entry).absolute()
        _add(roots, seen, dataset_root, ROOT)
        _add(roots, seen, dataset_root / ".tcip" / "state", STATE)
        for bucket in dataset_layout.prediction_bucket_dirs(dataset_root, include_cleared=True):
            _add(roots, seen, bucket.absolute(), PREDICTION_BUCKET)

    return tuple(roots)


__all__ = ["bootstrapped_stores", "project_roots"]

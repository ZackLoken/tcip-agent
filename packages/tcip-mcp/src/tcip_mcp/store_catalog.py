"""The whole store catalog in one import: every module that registers a store.

Importing this module registers every store's descriptor, so a caller invoked on its own sees the
same stores a running MCP server does. Where each store's entries sit under a root is
:mod:`tcip_store.layout_claims`.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store import registered_stores
from tcip_store.layout_claims import PREDICTION_BUCKET, ROOT, RUN, SPLITS, STATE

from tcip_annotation import json_io, review_engine  # noqa: F401
from tcip_mcp import experiments
from tcip_mcp import (  # noqa: F401
    audit,
    dataset_layout,
    model_registry,
    operationalization,
    project_record,
    project_status,
    traits,
    web_client,
    workspace,
)
from tcip_mcp.project_paths import project_state_dir
from tcip_mcp.pipelines import resolution  # noqa: F401
from tcip_mcp.pipelines.data import band_groups, selection, splits  # noqa: F401
from tcip_mcp.pipelines.feedback import materialize  # noqa: F401
from tcip_mcp.pipelines.postprocessing import plant_mapping  # noqa: F401
from tcip_mcp.pipelines.training import hpo  # noqa: F401
from tcip_mcp.tools import (  # noqa: F401
    data_tools,
    inference_tools,
    meta_tools,
    project_tools,
    proposal_tools,
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
    dataset roots from the project's own registry, and each run directory with the selection its
    resolved record bound. A recorded directory that no longer exists is skipped.

    Per layout:

    - ``ROOT``/``STATE``: the project's own root and ``.tcip/state`` under it, and every
      registered dataset's.
    - ``RUN``: each run directory (``experiments.run_observations``).
    - ``SPLITS``: the selection directory each bound training run's partition names, off the same
      walk, present only while that directory exists.
    - ``PREDICTION_BUCKET``: every model directory and its date subdirectories under each dataset's
      own ``predictions/`` tree, live and cleared alike
      (:func:`tcip_mcp.dataset_layout.prediction_bucket_dirs` with ``include_cleared=True``).
    """
    root = Path(project_root).absolute()
    roots: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    _add(roots, seen, root, ROOT)
    _add(roots, seen, project_state_dir(root), STATE)

    for run in experiments.run_observations(root):
        _add(roots, seen, run.directory.absolute(), RUN)
        binding = (run.record["resolved"]["partition"]["selection"]
                   if run.record["config"] is not None else None)
        if binding is not None:
            _add_if_dir(roots, seen, Path(binding["selection_dir"]).absolute(), SPLITS)

    for dataset_entry in project_tools.read_datasets(root):
        dataset_root = project_tools.dataset_entry_path(root, dataset_entry).absolute()
        _add(roots, seen, dataset_root, ROOT)
        _add(roots, seen, project_state_dir(dataset_root), STATE)
        for bucket in dataset_layout.prediction_bucket_dirs(dataset_root, include_cleared=True):
            _add(roots, seen, bucket.absolute(), PREDICTION_BUCKET)

    return tuple(roots)


__all__ = ["bootstrapped_stores", "project_roots"]

"""The whole store catalogue in one import: every module that registers a store.

Attributing a file to a store needs that store's descriptor, so a caller that must reason about
every store (the bundle accounting, the ``tcip export-store``/``tcip adopt-store`` commands)
imports this module first: a running MCP server already has every store registered through its
own tool imports, but a caller invoked on its own (a console command, a focused test) must not
silently see fewer stores than the server does. The operator commands that import
:func:`bootstrapped_stores` (``export-store``, ``adopt-store``) and the tests that exercise the
catalogue directly import it from here.

Where each store's entries sit under a root is not here: that is :mod:`tcip_store.layout_claims`,
which the conform rail reads without importing any owning module.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store import registered_stores
from tcip_store.layout_claims import (
    CURATED, EXPERIMENTS, HPO_ROOT, PREDICTION_BUCKET, ROOT, RUN, SPLITS, STATE, SWEEP,
)

from tcip_annotation import format_io, json_io, review_engine  # noqa: F401
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
from tcip_mcp.pipelines.data import band_groups, splits  # noqa: F401
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
from tcip_web import agent_learning_capture, jobstore  # noqa: F401
from tcip_web import state as web_state  # noqa: F401
from tcip_web.routes import canvas, sessions  # noqa: F401


def bootstrapped_stores() -> tuple[str, ...]:
    """Every store this module's imports register, which is every store the platform declares."""
    return registered_stores()


def _add(roots: list[tuple[str, str]], seen: set[tuple[str, str]], path: Path, layout: str) -> None:
    """Append ``(path, layout)`` unless this exact path/layout pair is already in.

    Keyed on the pair rather than the path alone: a directory can be two kinds of root at once
    (a curated-dataset artifact that also happens to be registered as a dataset), and keying on
    the path alone would keep whichever layout was added first and silently drop the other.
    """
    key = (os.path.normcase(str(path)), layout)
    if key in seen:
        return
    seen.add(key)
    roots.append((str(path), layout))


def _add_if_dir(roots: list[tuple[str, str]], seen: set[tuple[str, str]], path: Path,
                 layout: str) -> None:
    """Like :func:`_add`, but skips a record-named root whose directory no longer exists.

    A record can outlive the directory it names (a run's output moved, a curated dataset
    deleted): the record itself is not a fabrication, so it is read without complaint, but
    turning an absent directory into a root would have ``adopt-store`` create it and an empty
    ``store.db`` under a path the operator never asked for.
    """
    if path.is_dir():
        _add(roots, seen, path, layout)


def project_roots(project_root: str | Path) -> tuple[tuple[str, str], ...]:
    """The roots a whole project's records live in, each with the layout it is.

    Every root here comes from a record the project itself holds, or from walking a directory a
    record already named, never from a directory guessed with no record behind it: the
    registered dataset roots from the project's own registry, an HPO sweep's directory found
    under the project's own recorded HPO root, and a run's output directory, split-manifest
    binding, curated-dataset artifact and prediction-bucket lineage from that run's own
    ``experiments/<id>/`` members. Reads through
    :func:`~tcip_mcp.tools.project_tools.read_datasets_raw`, never
    :func:`~tcip_mcp.tools.project_tools.read_datasets`: this only needs a dataset's own
    location to enumerate its roots, not its current fingerprint identity, and a bare
    pre-prefix fingerprint elsewhere in the registry must not stop ``adopt-store``/
    ``export-store`` from reaching the very project a bare fingerprint predates.

    Per layout:

    ``ROOT``/``STATE``/``EXPERIMENTS``: the project's own root, plus ``.tcip/state`` and
    ``.tcip/experiments`` under it, the platform's own fixed convention (:mod:`tcip_mcp.
    project_paths`, :mod:`tcip_mcp.experiments`); the same two, ``ROOT`` and ``STATE``, for every
    registered dataset, since a dataset carries its own ``.tcip/state`` but no experiments of its
    own.

    ``HPO_ROOT``: ``.tcip/hpo`` under the project root, the same fixed convention
    (:func:`tcip_mcp.tools.training_tools.hpo_root`'s own default). A sweep launched with an
    explicit, non-default ``output_dir`` elsewhere is out of reach: nothing in the project's own
    records names that choice apart from the sweep's own manifest, which nothing here has read yet.

    ``SWEEP``: one root per immediate subdirectory of the project's own HPO root, the same walk
    a prediction bucket is found by under a dataset's ``predictions/`` tree: ``hpo_sweep_manifest``
    (:func:`tcip_mcp.tools.training_tools.sweep_manifest_key`) keys every sweep's manifest by
    ``(study_name, "manifest")`` under that root, but its descriptor declares no enumeration, so
    a sweep is found by the directory :func:`~tcip_mcp.tools.training_tools.sweep_dir` names it
    at (``hpo_root/<study_name>``) rather than by listing the store's own keys.

    ``RUN``: each experiment's own ``status.json["output_dir"]``
    (:func:`tcip_mcp.experiments.stamp_run_identity` is the one writer), read for every id
    :func:`tcip_mcp.experiments.experiment_ids_with_status` names. A pre-created experiment that
    was never launched carries no ``output_dir`` and contributes no root; a launched one whose
    output directory has since been moved or deleted is skipped the same way, its record read
    without complaint but not turned into a root ``adopt-store`` would otherwise recreate.

    ``SPLITS``: each experiment's own ``split.json["manifest_binding"]["manifest_dir"]``
    (:mod:`tcip_mcp.pipelines.data.split_construction`'s own field, persisted by
    ``persist_split_manifest``), present only for a run bound to an existing split manifest
    rather than one that drew its own. A manifest ``draw_splits``/``freeze_split_manifest``
    (:mod:`tcip_mcp.tools.data_tools`) wrote but that no run has ever bound to is out of reach:
    nothing in the project's own records names it. A bound manifest directory that no longer
    exists is skipped the same way as a moved run output directory.

    ``CURATED``: each experiment's own ``artifacts.json["curated_dataset"]["path"]``
    (:func:`tcip_mcp.tools.feedback_tools.materialize_review_dataset` records it there only when
    called with an ``experiment_id``). A curated dataset materialized without naming one is out
    of reach the same way; a recorded one since deleted is skipped the same way as a moved run
    output directory.

    ``PREDICTION_BUCKET``: two sources, since a bucket's own directory is never named by a fixed
    convention alone. Every model directory and its date subdirectories under each dataset's own
    ``predictions/`` tree (:func:`tcip_mcp.dataset_layout.prediction_bucket_dirs`, the same walk
    the ``doctor`` command's registry check reads through), for a bucket that lives under a
    registered dataset; and each experiment's own ``lineage.json["predictions"]``
    (:mod:`tcip_mcp.tools.inference_tools`'s own ``update_lineage`` call), for a bucket an
    inference run wrote outside any registered dataset's tree. A recorded lineage bucket since
    deleted is skipped the same way as a moved run output directory; the walk under a registered
    dataset's own tree only ever names directories that exist.
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
        binding = split_doc.get("manifest_binding") if isinstance(split_doc, dict) else None
        manifest_dir = binding.get("manifest_dir") if isinstance(binding, dict) else None
        if manifest_dir:
            _add_if_dir(roots, seen, Path(manifest_dir).absolute(), SPLITS)

        artifacts = experiments.read_member(experiments.artifacts_key(exp_id, root=root), {})
        curated = artifacts.get("curated_dataset") if isinstance(artifacts, dict) else None
        curated_path = curated.get("path") if isinstance(curated, dict) else None
        if curated_path:
            _add_if_dir(roots, seen, Path(curated_path).absolute(), CURATED)

        lineage = experiments.read_member(experiments.lineage_key(exp_id, root=root), {})
        predictions = lineage.get("predictions") if isinstance(lineage, dict) else None
        if predictions:
            _add_if_dir(roots, seen, Path(predictions).absolute(), PREDICTION_BUCKET)

    for dataset_entry in project_tools.read_datasets_raw(root):
        if not dataset_entry.get("path"):
            continue
        dataset_root = project_tools.dataset_entry_path(root, dataset_entry).absolute()
        _add(roots, seen, dataset_root, ROOT)
        _add(roots, seen, dataset_root / ".tcip" / "state", STATE)
        for bucket in dataset_layout.prediction_bucket_dirs(dataset_root):
            _add(roots, seen, bucket.absolute(), PREDICTION_BUCKET)

    return tuple(roots)


__all__ = ["bootstrapped_stores", "project_roots"]

"""Operator-script entry point for the store catalogue and a project's own roots.

The catalogue import itself (every module that registers a store) lives in
``tcip_mcp.store_catalogue``, package-only so :func:`tcip_mcp.tools.bundle.account_for` reaches
it without the repo root on ``sys.path``. This module re-exports :func:`bootstrapped_stores` from
there, so the operator scripts that import it (``export_store.py``, ``adopt_store.py``) keep
working unchanged, and adds :func:`project_roots`, which needs ``project_tools`` and
``tcip_mcp.experiments`` directly rather than the whole catalogue.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store.layout_claims import (
    CURATED, EXPERIMENTS, HPO_ROOT, PREDICTION_BUCKET, ROOT, RUN, SPLITS, STATE, SWEEP,
)

from tcip_mcp import experiments as experiments_module
from tcip_mcp.dataset_layout import prediction_bucket_dirs
from tcip_mcp.store_catalogue import bootstrapped_stores  # noqa: F401
from tcip_mcp.tools import project_tools
from tcip_mcp.tools.training_tools import hpo_root as project_hpo_root


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
    turning an absent directory into a root would have ``adopt_store.py`` create it and an empty
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
    pre-prefix fingerprint elsewhere in the registry must not stop ``adopt_store.py``/
    ``export_store.py`` from reaching the very project a conform script needs to run against.

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
    without complaint but not turned into a root ``adopt_store.py`` would otherwise recreate.

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
    ``scripts/doctor.py``'s registry check reads through), for a bucket that lives under a
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

    hpo_dir = project_hpo_root(root=root)
    _add(roots, seen, hpo_dir, HPO_ROOT)
    if hpo_dir.is_dir():
        for entry in sorted(p for p in hpo_dir.iterdir() if p.is_dir()):
            _add(roots, seen, entry, SWEEP)

    for exp_id in experiments_module.experiment_ids_with_status(root):
        status = experiments_module.read_member(
            experiments_module.status_key(exp_id, root=root), {})
        output_dir = status.get("output_dir") if isinstance(status, dict) else None
        if output_dir:
            _add_if_dir(roots, seen, Path(output_dir).absolute(), RUN)

        split_doc = experiments_module.read_member(
            experiments_module.split_key(exp_id, root=root), {})
        binding = split_doc.get("manifest_binding") if isinstance(split_doc, dict) else None
        manifest_dir = binding.get("manifest_dir") if isinstance(binding, dict) else None
        if manifest_dir:
            _add_if_dir(roots, seen, Path(manifest_dir).absolute(), SPLITS)

        artifacts = experiments_module.read_member(
            experiments_module.artifacts_key(exp_id, root=root), {})
        curated = artifacts.get("curated_dataset") if isinstance(artifacts, dict) else None
        curated_path = curated.get("path") if isinstance(curated, dict) else None
        if curated_path:
            _add_if_dir(roots, seen, Path(curated_path).absolute(), CURATED)

        lineage = experiments_module.read_member(
            experiments_module.lineage_key(exp_id, root=root), {})
        predictions = lineage.get("predictions") if isinstance(lineage, dict) else None
        if predictions:
            _add_if_dir(roots, seen, Path(predictions).absolute(), PREDICTION_BUCKET)

    for dataset_entry in project_tools.read_datasets_raw(root):
        if not dataset_entry.get("path"):
            continue
        dataset_root = project_tools.dataset_entry_path(root, dataset_entry).absolute()
        _add(roots, seen, dataset_root, ROOT)
        _add(roots, seen, dataset_root / ".tcip" / "state", STATE)
        for bucket in prediction_bucket_dirs(dataset_root):
            _add(roots, seen, bucket.absolute(), PREDICTION_BUCKET)

    return tuple(roots)

"""The whole store catalogue in one import, and which roots a project's records live in.

A store is registered by importing the module that declares it, so a tool that has to reason
about every store (writing a database back out as files, or decoding files into one) needs
every owning module imported first. No single package's own import set covers them all: the
web package owns the learning-capture log the MCP server never imports. This module is that
one import set.

Where each store's entries sit under a root is not here: that is :mod:`tcip_store.layout_claims`,
which the conform rail reads without importing any owning module.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store import registered_stores
from tcip_store.layout_claims import EXPERIMENTS, ROOT, STATE

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

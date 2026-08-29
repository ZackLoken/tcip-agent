"""The whole store catalogue in one import: every module that registers a store.

Attributing a file to a store needs that store's descriptor, so a caller that must reason about
every store (the bundle accounting, the export and adopt scripts) imports this module first: a
running MCP server already has every store registered through its own tool imports, but a caller
invoked on its own (a script, a focused test) must not silently see fewer stores than the server
does. Package-only: nothing here needs the repository's own ``scripts`` package, which exists
only with the repo root on ``sys.path``, so :func:`tcip_mcp.tools.bundle.account_for` reaches
this without it. ``scripts/_store_bootstrap.py`` re-exports :func:`bootstrapped_stores` from
here, so the operator scripts that import it (``export_store.py``, ``adopt_store.py``) and the
tests that exercise the catalogue directly keep working unchanged.

Where each store's entries sit under a root is not here: that is :mod:`tcip_store.layout_claims`,
which the conform rail reads without importing any owning module.
"""

from __future__ import annotations

from tcip_store import registered_stores

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


__all__ = ["bootstrapped_stores"]

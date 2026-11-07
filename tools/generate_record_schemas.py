"""Generate the snapshot of the JSON schema of every record the platform stores and reads back.

:data:`STORED_RECORDS` lists, by dotted name, each model (or the validator the reader of a
record holds) that validates a value the platform itself wrote: a ``tcip_store`` key, a
run-directory file, a dataset's own document, a log row, a checkpoint's stamp or an exported
label container. A model that validates a caller's input, a derived value or an engine's output
is no entry, and a model one entry nests renders inside that entry.

Each entry renders as the schema its reader validates against (``mode="validation"``), keyed by
its dotted name, in one document written to :data:`SNAPSHOT_PATH`. A change to a stored record's
schema moves the document, and ``tests/test_stored_record_schemas.py`` fails until it is
regenerated in the same change.

Run it after changing a stored record model; it rewrites the snapshot in place.
"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from tests import REPO_ROOT, write_generated

if TYPE_CHECKING:
    from collections.abc import Mapping

SNAPSHOT_PATH = REPO_ROOT / "tests" / "stored_record_schemas.json"

STORED_RECORDS = (
    "tcip_annotation.flags._FLAGS_RECORD",
    "tcip_annotation.verdicts._VERDICT",
    "tcip_mcp.assessment.ASSESSMENT_RECORD",
    "tcip_mcp.audit.AuditLine",
    "tcip_mcp.buckets._BUCKET",
    "tcip_mcp.dataset_layout.DatasetRecord",
    "tcip_mcp.experiments.CancelRequest",
    "tcip_mcp.experiments.FinalStatus",
    "tcip_mcp.experiments.MetricsRow",
    "tcip_mcp.experiments.RunRecord",
    "tcip_mcp.experiments.SweepRecord",
    "tcip_mcp.model_registry.CheckpointStamp",
    "tcip_mcp.model_registry.RegistryIndex",
    "tcip_mcp.pipelines.data.band_groups.BandGroupManifest",
    "tcip_mcp.pipelines.data.selection._SELECTION",
    "tcip_mcp.pipelines.data.selection._STORED_GROUND_TRUTH",
    "tcip_mcp.pipelines.delivery_events_schema.Acknowledgment",
    "tcip_mcp.pipelines.delivery_events_schema.DeliveryEventRecord",
    "tcip_mcp.pipelines.inference.generic_predictor.SliceBatch",
    "tcip_mcp.pipelines.postprocessing.plant_mapping._MAPPING_RECORD",
    "tcip_mcp.pipelines.postprocessing.plant_mapping._PLANT_REGISTRY",
    "tcip_mcp.pipelines.training.lr_range.LrRangeResult",
    "tcip_mcp.project_record.ProjectRecord",
    "tcip_mcp.subject_registry._REGISTRY",
    "tcip_mcp.tools.inference_tools.RasterPassIdentity",
    "tcip_mcp.tools.ingest_tools.LabelExport",
    "tcip_mcp.tools.meta_tools.FrictionReport",
    "tcip_mcp.tools.meta_tools.Retrospective",
    "tcip_mcp.tools.project_tools._DATASET_REGISTRY",
    "tcip_mcp.tools.proposal_tools.ProposalStaging",
    "tcip_mcp.traits.TraitRecord",
    "tcip_mcp.web_client.CanvasGeometry",
    "tcip_mcp.web_client.CanvasMeta",
    "tcip_mcp.web_client._PersistedGuiState",
    "tcip_web.routes.sessions.AnnotationStats",
)
"""The stored record entries, each the dotted name of a model or of the adapter its reader holds."""


def stored_record_adapters() -> dict[str, TypeAdapter]:
    """Return each :data:`STORED_RECORDS` entry as the adapter validating it, by dotted name.

    An entry naming a ``TypeAdapter`` is that adapter, the one its reader calls; any other is
    wrapped in one.
    """
    adapters: dict[str, TypeAdapter] = {}
    for dotted in STORED_RECORDS:
        module, attribute = dotted.rsplit(".", 1)
        declared = getattr(importlib.import_module(module), attribute)
        adapters[dotted] = (
            declared
            if isinstance(declared, TypeAdapter)
            else TypeAdapter(declared)
        )
    return adapters


def render(adapters: Mapping[str, TypeAdapter]) -> str:
    """Return the snapshot document of ``adapters``, each rendered as its reader validates.

    The document is a JSON object keyed by dotted name, every object's keys sorted, each value
    the adapter's validation-mode JSON schema, indented, ending in a newline.
    """
    schemas = {
        name: adapters[name].json_schema(mode="validation")
        for name in sorted(adapters)
    }
    return json.dumps(schemas, indent=2, sort_keys=True) + "\n"


def main() -> int:
    """Write the snapshot of :data:`STORED_RECORDS` and return 0."""
    return write_generated(SNAPSHOT_PATH, render(stored_record_adapters()))


if __name__ == "__main__":
    raise SystemExit(main())

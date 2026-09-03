"""The ``delivery_events`` record's declared shape, so its writer (``resolution.py``'s
``record_delivery_binding_event``) and its readers (``tcip_web``'s ``list_delivery_events`` route,
``scripts/conform_delivery_events.py``) agree on one shape rather than each independently
tolerating whatever the others happen to have written.

A sibling module to ``resolution.py`` rather than a class inside it, the same split
``pipelines/schemas.py`` already draws for its own pydantic models: ``resolution.py`` states its
own dependency surface as storage-seam-and-stdlib only, and a pydantic import belongs with the
other schema-only module instead of widening that statement.

Both models forbid an undeclared key, so a stored record or disclosure carrying one is refused by
name rather than silently accepted and later misread.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class PlantMappingDisclosure(BaseModel):
    """The ``plant_mapping`` a phenology delivery attributed detections through, exactly as
    :meth:`tcip_mcp.pipelines.postprocessing.plant_mapping.MappingBuild.delivery_disclosure`
    composes it: the mapping's own identity, ``verify_mapping_inputs``'s two unverified
    disclosures, and this delivery's own unattributed-capture count scoped to its delivered dates.

    Every key is required: none of ``dates_delivered``, ``images_unattributed`` or
    ``plant_attribution`` can be reconstructed from a record that lacks it, since none was ever
    computed for that delivery, so a record missing one is refused rather than read as if the gap
    meant something.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    project_root: str
    dataset_id: str
    dataset_root: str
    built_at: str
    record_sha256: str
    nn_tolerance_m: dict
    capture_identity: dict[str, str]
    captures_unverified: list[str]
    plant_csvs_unverified: list[str]
    dates_delivered: list[str]
    images_unattributed: int
    images_unattributed_scope: str
    plant_attribution: str


class DocumentBinding(BaseModel):
    """One bucket's binding evidence, exactly as ``record_delivery_binding_event`` renders a
    :class:`tcip_mcp.pipelines.resolution.StampBinding` into the stored record's ``documents``
    mapping."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    claimed: bool
    experiment_id: Optional[str]
    producing_experiment_id: Optional[str]
    checkpoint_sha256: Optional[str]
    record_digest: Optional[str]
    note: str


class DeliveryEventRecord(BaseModel):
    """The stored per-delivery record, as ``record_delivery_binding_event`` writes it and
    ``list_delivery_events`` serves it back: what shipped, under which trait and kind, and the
    real per-bucket verification evidence the delivering door reconciled at the time."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    trait: Optional[str]
    delivery_kind: Optional[str]
    door: str
    output_path: Optional[str]
    measurement_documents: list[str]
    scale_document: Optional[str]
    plant_mapping: Optional[PlantMappingDisclosure]
    documents: dict[str, DocumentBinding]
    produced_at: str

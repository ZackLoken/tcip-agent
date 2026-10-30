"""Write served delivery events as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), in projects each created through
``tests._web_fixtures.named_project``, deliveries are made through the platform's own producers
and the test suite's own scenes for them: a per-image count delivery
over a bucket its assessment validates, a phenology delivery through a plant mapping shipped
under a breeder's acknowledgment, and the orthomosaic door's nearest-plant and canopy-segment
deliveries. Each project is opened through ``POST /api/projects/open`` and its one record served
by ``GET /api/results/delivery-events``; the phenology project's listing is served again after a
superseding rebuild archives the mapping it cites. The count project's checkpoint is assessed
again at :data:`UNMET_CONF`, which fails, a second checkpoint is registered beside it, and its
bucket is exported through ``POST /api/results/export_count_csv``. The results routes' answers are
recorded beside them, under ``answers``, each refusal with its status: the count project's
mapping names, none, ``no_mapping_names``; the count export repeated while every audit append
fails (``tests._audit_fixtures.refuse_audit_appends``), ``count_export_audit_gap``; its listing
once its first delivery's record is damaged (``tests._record_damage_fixtures.damage_record``),
``unreadable_deliveries``; the phenology project's measurement refused by the delivery gate,
``delivery_refusal``, shown unvalidated, ``phenology``, and naming no stored mapping,
``unknown_mapping``; its curves exported under an acknowledgment whose line lands while
the delivery's fails, ``export_audit_gap``; a second mapping built while every append fails,
``mapping_audit_gap``; a rebuild of the mapping its delivery cites, unsuperseded, whose refusal
names the citing event as ``<acknowledged_mapping event>`` (``fixture_session.normalized``'s
``minted``), ``mapping_cited``; once the count trait is confirmed there, a phenology
measurement of it, ``operationalization_refusal``; its mapping names, ``mapping_names``, and
its first mapping loaded, ``mapping_loaded``; mappings built
over its registry (``mapping_built``), over one placing only its first plant
(``mapping_one_placed``), at a stated radius (``mapping_stated``) and, once a date of
photographs carrying no position is added, over every date (``mapping_unpositioned``); in a
project of its own, beside a plant open on every date, a measurement of each of three plants,
one open but its first capture left without predictions
(``test_plant_mapping_binding.UNREAD_P2``), one undetected and one closed on every date
(``milestones``); in a project of its own, an attributed
series assessed, published and mapped (``tests._chain_fixtures.attributed_series``), its
deliveries, none, ``no_deliveries``, and its first plant measured validated,
``phenology_validated``; and the nearest-plant project's
counts exported unacknowledged, ``count_delivery_refusal``, then exported acknowledged, kept
under ``acknowledged_count_export``. The served listings, keyed by fixture name under
``listings``, the count export (its three delivery headers, its file and the delivery events
once it is written, :func:`exported`), under ``count_export``, the count project's
``GET /api/results/assessments`` listing, under
``assessments``, the listing of a project of its own whose trained detector is assessed once at
:data:`UNMET_CONF` and fails, under ``failed_assessments``, and the count project's registered
models, under ``registered_models``, and the answers are
written, normalized by :data:`NUMBERED` (``fixture_session.normalized``),
each id a scene holds read under the scene's name for it and the archived mapping's key as
``<archived NAME>``, to :data:`GENERATED_PATH`.

    python -m tools.generate_delivery_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    ACTOR,
    answered,
    detail_of,
    normalized,
    opened,
    responded,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, JsonValue

    from tcip_web.state import OpenProject

GENERATED_PATH = FRONTEND_SRC / "test" / "deliveryEvents.json"

NUMBERED = frozenset(
    {
        "event_id",
        "acknowledgment_id",
        "assessment_id",
        "dataset_id",
        "experiment_id",
        "output_sha256",
        "result_sha256",
        "record_sha256",
        "checkpoint_sha256",
        "sha256",
        "digest",
        "curves",
        "milestones",
    }
)
"""The served strings two productions of the same scenes fill differently: ids minted inside a
helper's own steps, and digests over what carries those or a time, or over a file the scene
writes in the platform's own line endings (a run's weights, a written output, a record holding a
time, a plant registry, a measurement's projection keyed by its name under ``digests``).
An id a scene step holds is read under the name the scene gives it instead (:data:`HELD_RUNS`
and the ``minted`` names :func:`deliveries` reads)."""

COUNT_RUN = "exp-fixture"
"""The count chain's experiment id, a sample name."""

FAILED_RUN = "exp-failed"
"""The experiment id of the detector whose one assessment fails, a sample name."""

HELD_RUNS = {
    COUNT_RUN: "<count run>",
    FAILED_RUN: "<failed run>",
    "exp-acknowledged": "<acknowledged run>",
    "exp-attributed": "<attributed run>",
}
"""The experiment ids the scenes state, the count chain's and the chain helpers' own
(``tests._chain_fixtures``), each read under the scene's name for it."""

UNMET_CONF = 0.999
"""A conf above the scores the chain's detector gives, so its assessment there counts nothing
and fails (read from the generator's own run, which asserts the failure)."""

UNPOSITIONED_DATE = "2026-03-11"
"""A capture date whose photographs carry no position, added once the other mappings are built."""

STATED_RADIUS = "the receiver's spec sheet"
"""The reason the stated-radius mapping gives for its radius, a sample wording."""

MILESTONE_PLOTS = {
    "P1": (43.19670, -90.058000),
    "P2": (43.19670, -90.058037),
    "P3": (43.19670, -90.058074),
    "P4": (43.19670, -90.058111),
}
"""The milestone scene's plants, one row apart, each at its ``(lat, lon)``, as
``tests._mapping_fixtures.PLOTS`` spaces them and names the first two; :func:`_milestone_result`
names what each holds."""


def _milestone_result(image: Path) -> dict:
    """Return the published detections of ``image`` by the plant its name begins with.

    ``P1`` and ``P2`` hold one open flower, ``P3`` none and ``P4`` one closed flower.
    """
    from tests.test_second_trait_acceptance import BLOOM_STATE

    states = {
        "P1": [[BLOOM_STATE.values.index("open")]],
        "P2": [[BLOOM_STATE.values.index("open")]],
        "P3": [],
        "P4": [[BLOOM_STATE.values.index("closed")]],
    }[image.stem.split("_")[0]]
    return {
        "image": str(image),
        "width": 8,
        "height": 8,
        "boxes": [[1.0, 1.0, 3.0, 3.0]] * len(states),
        "scores": [0.9] * len(states),
        "labels": [1] * len(states),
        "cap": 2,
        "attributes": states,
    }


def deliveries(workspace: Path, client: TestClient) -> dict[str, JsonValue]:
    """Return the scenes' delivery events, export headers, assessments and route answers.

    The events and the results routes' answers are by fixture name; the assessments and the
    registered models are the count project's listings, the failed assessments the failed
    project's.

    The scenes are made in projects created under ``workspace`` and served through ``client``.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.delivery import DeliveryRefusal, delivery_event_key
    from tcip_mcp.identity import actor
    from tcip_mcp.pipelines.delivery_events_schema import (
        PlantMappingDisclosure,
    )
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        StatedRadius,
        archived_mapping_name,
    )
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tcip_web.routes.results import (
        EXPORT_HEADER,
        AcknowledgmentPayload,
        BuildMappingPayload,
        DeliveryEvents,
        ExportCountCsvPayload,
        ExportCsvPayload,
        LoadMappingPayload,
        OrthomosaicPlantCountsDelivery,
        PerImageCountDelivery,
        PhenologyAnswer,
        PhenologyPayload,
    )
    from tests import (
        _trait_fixtures as fx,
        test_orthomosaic_tools as ortho,
        test_plant_mapping_binding as binding,
    )
    from tests._audit_fixtures import refuse_audit_appends
    from tests._chain_fixtures import (
        assess,
        attributed_series,
        published,
        run_the_chain,
        trained_detector,
    )
    from tests._mapping_fixtures import (
        PLANTS,
        POSITION_ERROR_M,
        geolocated_captures,
        plant_rows,
        register_plant_registry_for,
        write_plant_csv,
    )
    from tests._producer_fixtures import write_image
    from tests._record_damage_fixtures import damage_record
    from tests._verified_checkpoint_fixtures import foreign_checkpoint
    from tests._web_fixtures import BROWSER, named_project
    from tests.test_second_trait_acceptance import (
        FLOWERS,
        _seed_currant_bloom_trait,
    )

    def served(project: OpenProject, name: str) -> JsonValue:
        """Return ``project``'s one delivery event listed.

        Its event id reads as ``<NAME event>``, the name of the first listing that served it.
        """
        opened(client, project)
        listing = answered(client, "list_delivery_events", 200)
        (record,) = DeliveryEvents.model_validate(listing).records
        minted.setdefault(record.event.event_id, f"<{name} event>")
        return listing

    def delivered(result: dict) -> None:
        assert "error" not in result, f"a scene's delivery refused: {result}"

    def browsed(
        route: str, body: BaseModel, status: int, *, landing: int | None = None
    ) -> JsonValue:
        """Post ``body`` to ``route`` as a browser (:func:`~tools.fixture_session.answered`).

        Returns its answer, a refusal with its status. With ``landing`` the first ``landing``
        audit appends land and every later one fails
        (``tests._audit_fixtures.refuse_audit_appends``).
        """
        with pytest.MonkeyPatch.context() as patch:
            if landing is not None:
                refuse_audit_appends(patch, landing=landing)
            return answered(
                client,
                route,
                status,
                body,
                headers=BROWSER,
            )

    def exported(
        body: BaseModel, route: str = "export_count_csv"
    ) -> dict[str, JsonValue]:
        """Export ``body`` through the export door ``route`` names as a browser; return its answer.

        That is the export header the door sent, the file it sent and the open project's delivery
        events once it has, the export's own the latest; any answer but 200 raises. The export's
        event id is read as ``<export N>`` in every place it is named, and the acknowledgment it
        shipped under, if any, as ``<export N acknowledgment>``.
        """
        response = responded(client, route, 200, body, headers=BROWSER)
        events = answered(client, "list_delivery_events", 200)
        latest = DeliveryEvents.model_validate(events).records[-1].event
        name = f"<export {len(exports)}"
        exports.append(latest.event_id)
        minted[latest.event_id] = f"{name}>"
        if latest.acknowledgment is not None:
            minted[latest.acknowledgment.acknowledgment_id] = (
                f"{name} acknowledgment>"
            )
        return {
            "headers": {EXPORT_HEADER: response.headers[EXPORT_HEADER]},
            "file": response.text,
            "events": events,
        }

    minted: dict[str, str] = dict(HELD_RUNS)
    exports: list[str] = []
    crop = sorted(registered_crops())[0]

    out: dict[str, JsonValue] = {}
    answers: dict[str, JsonValue] = {}

    root = workspace / "counts"
    project = named_project(root, "Counts")
    chain = run_the_chain(root, experiment_id=COUNT_RUN)
    minted[chain.assessment["assessment_id"]] = "<chain assessment>"
    delivered(
        deliver_per_image_counts(
            root,
            str(chain.root),
            chain.bucket,
            output_path=str(root / "counts.csv"),
            trait=fx.COUNT_TRAIT,
            actor=None,
        )
    )
    out["validated"] = served(project, "validated")
    answers["no_mapping_names"] = answered(client, "list_plant_mappings", 200)
    unmet = assess(
        root, chain.checkpoint_path, chain.selection_dir, conf=UNMET_CONF
    )
    assert unmet["failures"], unmet
    minted[unmet["assessment_id"]] = "<unmet assessment>"
    second_run = Path(foreign_checkpoint(root, name="other")).parent.name
    models = answered(client, "registered_models", 200)
    assessments = answered(client, "assessments", 200)
    registered = register_dataset(root, str(chain.root), crop=crop, actor=None)
    delivered(registered)
    minted[registered["id"]] = "<count dataset>"
    count_export = ExportCountCsvPayload(
        delivery=PerImageCountDelivery(
            kind="per_image_count",
            dataset_root=str(chain.root),
            bucket=chain.bucket,
            trait=fx.COUNT_TRAIT,
        ),
        filename="counts.csv",
        user=ACTOR,
    )
    export = exported(count_export)
    answers["count_export_audit_gap"] = browsed(
        "export_count_csv", count_export, 409, landing=0
    )
    (record,) = DeliveryEvents.model_validate(out["validated"]).records
    damage_record(delivery_event_key(root, record.event.event_id), b"[]")
    answers["unreadable_deliveries"] = answered(
        client, "list_delivery_events", 400
    )

    root = workspace / "phenology"
    project = binding._init(root)
    dataset_root = binding._dataset(root)
    plant_csv, preds_by_date = binding._write_scene(dataset_root)
    registry = register_plant_registry_for(root, [plant_csv])
    delivered(
        build_plant_mapping(
            root,
            name="valley",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            actor=None,
        )
    )
    _seed_currant_bloom_trait(root)
    delivered(
        binding._deliver(
            root,
            trait="currant_bloom",
            mapping_name="valley",
            plants=[str(plot) for plot in binding.POPULATION],
            dataset_root=dataset_root,
            buckets=preds_by_date.values(),
            output_csv_path=str(root / "phenology.csv"),
            by=actor(ACTOR),
        )
    )
    out["acknowledged_mapping"] = served(project, "acknowledged_mapping")
    answers["mapping_cited"] = browsed(
        "build_plant_mapping",
        BuildMappingPayload(
            name="valley",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            user=ACTOR,
        ),
        409,
    )
    delivered(
        build_plant_mapping(
            root,
            name="valley",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            supersede=True,
            actor=None,
        )
    )
    out["archived_mapping"] = served(project, "archived_mapping")
    (archived,) = DeliveryEvents.model_validate(
        out["archived_mapping"]
    ).records
    disclosure = archived.event.plant_mapping
    assert isinstance(disclosure, PlantMappingDisclosure), disclosure
    measured = PhenologyPayload(
        mapping_name="valley",
        dataset_root=str(dataset_root),
        buckets=list(preds_by_date.values()),
        trait="currant_bloom",
        plants=[str(plot) for plot in binding.POPULATION],
    )
    answers["delivery_refusal"] = browsed(
        "phenology_measurement", measured, 400
    )
    phenology = browsed(
        "phenology_measurement",
        measured.model_copy(update={"show_unvalidated": True}),
        200,
    )
    answers["phenology"] = phenology
    answers["unknown_mapping"] = browsed(
        "phenology_measurement",
        measured.model_copy(update={"mapping_name": "absent"}),
        404,
    )
    unvalidated = PhenologyAnswer.model_validate(phenology).unvalidated
    assert unvalidated is not None, phenology
    curves_export = ExportCsvPayload(
        **measured.model_dump(exclude={"show_unvalidated"}),
        payload="curves",
        user=ACTOR,
        acknowledgment=AcknowledgmentPayload(
            reason="the mapping's rails, not the assessment",
            result_sha256=unvalidated.digests.curves,
        ),
    )
    answers["export_audit_gap"] = browsed(
        "export_csv", curves_export, 409, landing=1
    )
    curves = exported(curves_export, "export_csv")
    answers["mapping_audit_gap"] = browsed(
        "build_plant_mapping",
        BuildMappingPayload(
            name="orchard",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            user=ACTOR,
        ),
        409,
        landing=0,
    )
    fx.propose_and_confirm(root, fx.COUNT_SPEC)
    answers["operationalization_refusal"] = browsed(
        "phenology_measurement",
        measured.model_copy(update={"trait": fx.COUNT_TRAIT}),
        400,
    )
    answers["mapping_names"] = answered(client, "list_plant_mappings", 200)
    answers["mapping_loaded"] = browsed(
        "load_plant_mapping", LoadMappingPayload(name="valley"), 200
    )
    registries = {
        "mapping_built": registry,
        "mapping_one_placed": register_plant_registry_for(
            root,
            [write_plant_csv(root / "one_plant.csv", PLANTS[:1])],
            name="one-plant",
        ),
    }
    for fixture, plant_registry in registries.items():
        answers[fixture] = browsed(
            "build_plant_mapping",
            BuildMappingPayload(
                name=fixture.replace("_", "-"),
                dataset_root=str(dataset_root),
                plant_registry=plant_registry,
                user=ACTOR,
            ),
            200,
        )
    answers["mapping_stated"] = browsed(
        "build_plant_mapping",
        BuildMappingPayload(
            name="mapping-stated",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            match_radius=StatedRadius(value=4.0, rationale=STATED_RADIUS),
            user=ACTOR,
        ),
        200,
    )
    for plant in PLANTS:
        write_image(
            image_dir(dataset_root, UNPOSITIONED_DATE)
            / f"{plant['plot']}_{UNPOSITIONED_DATE.replace('-', '')}.jpg"
        )
    answers["mapping_unpositioned"] = browsed(
        "build_plant_mapping",
        BuildMappingPayload(
            name="mapping-unpositioned",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            user=ACTOR,
        ),
        200,
    )

    root = workspace / "milestones"
    project = binding._init(root)
    dataset_root = binding._dataset(root)
    plants = plant_rows(MILESTONE_PLOTS)
    live = [
        published(
            root,
            f"live/{date}",
            [
                _milestone_result(image)
                for image in images
                if image.stem not in binding.UNREAD_P2
            ],
            scope={"subject": "flower"},
            registry=FLOWERS,
        ).name
        for date, images in geolocated_captures(
            dataset_root, binding.DATES, plants
        ).items()
    ]
    delivered(
        build_plant_mapping(
            root,
            name="valley",
            dataset_root=str(dataset_root),
            plant_registry=register_plant_registry_for(
                root, [write_plant_csv(root / "plants.csv", plants)]
            ),
            actor=None,
        )
    )
    _seed_currant_bloom_trait(root)
    opened(client, project)
    answers["milestones"] = [
        browsed(
            "phenology_measurement",
            PhenologyPayload(
                mapping_name="valley",
                dataset_root=str(dataset_root),
                buckets=live,
                trait="currant_bloom",
                plants=[plot],
                show_unvalidated=True,
            ),
            200,
        )
        for plot in list(MILESTONE_PLOTS)[1:]
    ]

    series = attributed_series(workspace / "series")
    opened(client, series.project)
    answers["no_deliveries"] = answered(client, "list_delivery_events", 200)
    validated = PhenologyPayload(**series.body())
    answers["phenology_validated"] = browsed(
        "phenology_measurement",
        validated.model_copy(update={"plants": validated.plants[:1]}),
        200,
    )

    root = workspace / "nearest"
    project = named_project(root, "Nearest")
    fx.seed_delivery_traits(root)
    fx.seed_confirmed_aggregate(root, "stem_count", value_keys=["count"])
    raster_path = ortho._raster(root)
    ortho._write_geo_raster(raster_path)
    bucket = ortho._raster_bucket(
        root, raster_path, [(8.0, 8.0, 12.0, 12.0), (48.0, 8.0, 52.0, 12.0)]
    )
    registry = ortho._plant_registry(
        root,
        ortho._plants_csv_at(
            root,
            raster_path,
            [
                ("plot0", 10.0, 10.0),
                ("plot2", 50.0, 10.0),
                ("plot9", 200.0, 200.0),
            ],
        ),
    )
    delivered(
        ortho._deliver(
            root, bucket, registry, ["plot0", "plot2"], by=actor(ACTOR)
        )
    )
    out["registry"] = served(project, "registry")
    nearest = register_dataset(root, str(bucket.root), crop=crop, actor=None)
    delivered(nearest)
    minted[nearest["id"]] = "<nearest dataset>"
    plant_counts = ExportCountCsvPayload(
        delivery=OrthomosaicPlantCountsDelivery(
            kind="orthomosaic_plant_counts",
            dataset_root=str(bucket.root),
            bucket=bucket.name,
            plant_registry=registry,
            delivered_phenotype="stem_count",
            plants=["plot0", "plot2"],
        ),
        filename="plant_counts.csv",
        user=ACTOR,
    )
    answers["count_delivery_refusal"] = browsed(
        "export_count_csv", plant_counts, 400
    )
    refusal = DeliveryRefusal.model_validate(
        detail_of(answers["count_delivery_refusal"])
    )
    assert refusal.result_sha256 is not None, refusal
    acknowledged_export = exported(
        plant_counts.model_copy(
            update={
                "acknowledgment": AcknowledgmentPayload(
                    reason="the assessment is not run yet",
                    result_sha256=refusal.result_sha256,
                )
            }
        )
    )

    root = workspace / "segmented"
    project = named_project(root, "Segmented")
    fx.seed_delivery_traits(root)
    fx.seed_confirmed_aggregate(root, "stem_count", value_keys=["count"])
    _dataset_root, raster_path, bucket = ortho._canopy_setup(
        root,
        [
            (8.0, 8.0, 12.0, 12.0),
            (48.0, 8.0, 52.0, 12.0),
            (49.0, 49.0, 51.0, 51.0),
            (30.0, 30.0, 32.0, 32.0),
        ],
    )
    registry = ortho._plant_registry(
        root,
        ortho._plants_csv_at(
            root,
            raster_path,
            [
                ("plot0", 10.0, 10.0),
                ("plot1", 10.0, 50.0),
                ("plot2", 50.0, 10.0),
                ("plot3", 20.5, 45.0),
                ("plot4", 44.0, 44.0),
                ("plot9", 200.0, 200.0),
            ],
        ),
    )
    ortho._write_canopy_document(
        raster_path,
        [
            (5.0, 5.0, 15.0, 15.0),
            (45.0, 5.0, 55.0, 15.0),
            (20.0, 40.0, 30.0, 50.0),
            (40.0, 40.0, 52.0, 52.0),
            (48.0, 48.0, 60.0, 60.0),
        ],
    )
    delivered(
        ortho._deliver(
            root,
            bucket,
            registry,
            ["plot0", "plot2"],
            canopy_subject="canopy",
            position_error_m=POSITION_ERROR_M,
            by=actor(ACTOR),
        )
    )
    out["canopy"] = served(project, "canopy")

    root = workspace / "failed"
    project = named_project(root, "Failed")
    _, _, selection_dir, checkpoint_path = trained_detector(
        root, experiment_id=FAILED_RUN
    )
    failed = assess(root, checkpoint_path, selection_dir, conf=UNMET_CONF)
    assert failed["failures"], failed
    minted[failed["assessment_id"]] = "<failed assessment>"
    opened(client, project)
    failed_assessments = answered(client, "assessments", 200)
    return normalized(
        {
            "listings": out,
            "count_export": export,
            "acknowledged_count_export": acknowledged_export,
            "curves_export": curves,
            "assessments": assessments,
            "failed_assessments": failed_assessments,
            "registered_models": models,
            "answers": answers,
        },
        workspace,
        numbered=NUMBERED,
        rounded=True,
        minted={
            **minted,
            second_run: "<second run>",
            archived_mapping_name(
                disclosure.name, disclosure.record_sha256
            ): f"<archived {disclosure.name}>",
        },
    )


def produce() -> dict[str, JsonValue]:
    """Return the fixture's content, made in a scratch session."""
    with scratch_session() as (workspace, client):
        return deliveries(workspace, client)


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))

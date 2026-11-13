"""Tests for the phenology MCP tools, ``build_plant_mapping`` and ``deliver_phenology_milestones``.

``build_plant_mapping`` builds and persists a mapping and reports a compact summary or an error.
``deliver_phenology_milestones`` writes the column schema over a persisted plant mapping from
buckets whose scope declares the positive state's attribute. It delivers when each bucket was
published under an assessment that answers for the delivery, and refuses when one was not or when
no bucket's scope declares the attribute.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import get_args

import pytest

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.pipelines.postprocessing.plant_mapping import (
    GpsSource,
    StatedRadius,
    plant_mapping_key,
)
from tcip_mcp.tools.phenology_tools import build_plant_mapping
from tests import csv_rows
from tests._audit_fixtures import audit_rows
from tests._chain_fixtures import deliver_milestones


def _served_tool_description() -> str:
    """Return the description the tool listing serves for ``build_plant_mapping``."""
    from tests._server_calls import listed_tools

    served = listed_tools()["build_plant_mapping"].description
    assert served is not None
    return served


def _served_plant_mapping_section() -> str:
    """Return the "Plant mapping" section of the ``phenology`` content as served."""
    from tcip_mcp.tools.knowledge_tools import serve_domain_knowledge

    content = serve_domain_knowledge(name="phenology")["content"]
    _, _, tail = content.partition("\n## Plant mapping")
    assert tail, "the served phenology document has no 'Plant mapping' section"
    section, _, _ = tail.partition("\n## ")
    return section


@pytest.mark.parametrize(
    "served_text",
    [_served_tool_description, _served_plant_mapping_section],
    ids=["tool-description", "knowledge-section"],
)
def test_the_served_text_states_the_claiming_method_and_names_no_row_run(
    served_text: Callable[[], str],
) -> None:
    """Each text the agent is served about the mapping states what the mapping does.

    Each image on a date takes the nearest unclaimed plant, in capture-time order; neither the
    tool description nor the knowledge document names a row run.
    """
    text = " ".join(served_text().split())
    assert "row run" not in text
    assert "nearest unclaimed plant" in text
    assert "capture-time order" in text


def test_the_served_plant_mapping_section_names_every_source_an_assignment_takes() -> (
    None
):
    """The served "Plant mapping" section names each ``GpsSource`` value in backticks."""
    section = _served_plant_mapping_section()
    for source in get_args(GpsSource):
        assert f"`{source}`" in section


def test_build_plant_mapping_wraps_build_and_persists(tmp_path: Path) -> None:
    from datetime import datetime

    from tests._image_fixtures import write_geo_image
    from tests._mapping_fixtures import (
        CROP,
        register_plant_registry_for,
        write_plant_csv,
    )
    from tests._producer_fixtures import register_dataset_at
    from tests._web_fixtures import named_project

    named_project(tmp_path, "Orchard")
    write_geo_image(
        image_dir(tmp_path, "2026-02-11") / "img1.jpg",
        43.19670,
        -90.058000,
        datetime(2026, 2, 11, 9, 30),
    )
    register_dataset_at(tmp_path, tmp_path, crop=CROP)
    csv_path = write_plant_csv(
        tmp_path / "plants.csv",
        [{"plot": "P1", "accession": "acc-9", "lat": 43.197, "lon": -90.058}],
    )
    name = "valley"

    registry = register_plant_registry_for(tmp_path, [csv_path])
    res = build_plant_mapping(
        tmp_path,
        name=name,
        dataset_root=str(tmp_path),
        plant_registry=registry,
        match_radius=StatedRadius(value=10.0, rationale="the receiver's spec"),
        actor=None,
    )

    assert "error" not in res, res
    assert res["name"] == name
    assert list(res["per_date"]) == ["2026-02-11"]
    assert res["per_date"]["2026-02-11"]["n_images"] == 1
    assert res["match_radius"] == {
        "2026-02-11": {
            "value": 10.0,
            "source": "stated",
            "rationale": "the receiver's spec",
        }
    }
    persisted = ts.read(plant_mapping_key(tmp_path, name))
    assert isinstance(persisted, dict)
    assignments = persisted["assignments"]
    assert isinstance(assignments, dict)
    assert list(assignments.keys()) == ["2026-02-11"]
    dated = assignments["2026-02-11"]
    assert isinstance(dated, list) and isinstance(dated[0], dict)
    assert Path(str(dated[0]["image"])).stem == "img1"
    assert "confidence" not in dated[0]


def test_build_plant_mapping_missing_dataset_root(tmp_path: Path) -> None:
    from tests._web_fixtures import named_project

    named_project(tmp_path, "Orchard")
    res = build_plant_mapping(
        tmp_path,
        dataset_root=str(tmp_path / "nope"),
        plant_registry="unregistered",
        name="m",
        actor=None,
    )
    assert "error" in res
    assert "register_dataset" in res["error"]


def test_build_plant_mapping_missing_registry(tmp_path: Path) -> None:
    from tcip_mcp.traits import registered_crops
    from tests._producer_fixtures import register_dataset_at
    from tests._web_fixtures import named_project

    named_project(tmp_path, "Orchard")
    image_dir(tmp_path, "2026-02-11").mkdir(parents=True)
    register_dataset_at(tmp_path, tmp_path, crop=sorted(registered_crops())[0])
    res = build_plant_mapping(
        tmp_path,
        dataset_root=str(tmp_path),
        plant_registry="does-not-exist",
        name="m",
        actor=None,
    )
    assert "error" in res
    assert "plant registry not found" in res["error"]
    assert "register_plant_registry" in res["error"]


# ── deliver_phenology_milestones over a published, assessed series ─────────────────────────


def test_an_assessed_series_delivers_validated_milestones_under_its_revision(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.pipelines.postprocessing.phenology import (
        phenology_csv_columns,
    )
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path)
    out_csv = tmp_path / "out" / "bud_phenology.csv"

    res = deliver_milestones(tmp_path, series.body(), out_csv)

    assert "error" not in res, res
    revision = latest_confirmed("bud_opening", tmp_path)
    assert res["shipped"] == {"kind": "validated"}
    assert res["trait_revision"] == revision.number
    assert res["columns"] == phenology_csv_columns(revision.entry)
    assert (
        res["plant_mapping"]["images_unattributed"] > 0
    )  # the reference frames stand at no plant
    rows = csv_rows(out_csv)
    assert [r["plant_id"] for r in rows] == ["PLANT_A", "PLANT_B"]
    assert {r["validated"] for r in rows} == {"True"}
    assert {r["delivery_event_id"] for r in rows} == {res["event_id"]}
    (event,) = read_delivery_events(tmp_path)
    assert event.door == "deliver_phenology_milestones"
    assert event.producer.model_dump() == series.assessment["producer"]
    assert {b.assessment_id for b in event.buckets} == {
        series.assessment["assessment_id"]
    }


def test_the_delivery_line_lands_in_the_log_of_the_dataset_its_buckets_sit_in(
    tmp_path: Path,
) -> None:
    """The delivery's one audit line, in the log that travels with the data, names the event."""
    pytest.importorskip("torch")
    from tcip_mcp.delivery import read_delivery_events
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    res = deliver_milestones(
        tmp_path, series.body(), tmp_path / "out" / "bud.csv"
    )
    assert "error" not in res, res

    (event,) = read_delivery_events(tmp_path)
    lines = audit_rows(series.root, "delivery_event")
    assert [e["arguments"] for e in lines] == [{"event_id": event.event_id}]
    assert audit_rows(tmp_path, "delivery_event") == []


def test_an_unassessed_series_refuses_at_the_door_that_takes_no_acknowledgment(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(
        tmp_path, fractions=(0.0, 1.0), assessed=False
    ).body()
    out_csv = tmp_path / "out" / "bud.csv"

    res = deliver_milestones(tmp_path, body, out_csv)

    assert "no assessment answers" in res["error"], res
    assert not out_csv.exists()


def test_an_unreadable_prediction_document_is_reported_by_name(
    tmp_path: Path,
) -> None:
    """A present, unreadable prediction document is returned as an error naming its stem."""
    pytest.importorskip("torch")
    from tcip_mcp.dataset_layout import prediction_key
    from tests._chain_fixtures import attributed_series
    from tests._record_damage_fixtures import UNDECODABLE, damage_record

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    date = sorted(series.buckets)[1]
    stem = f"PLANT_A_{date}_0"
    damage_record(
        prediction_key(series.root, series.buckets[date], stem), UNDECODABLE
    )

    res = deliver_milestones(
        tmp_path, series.body(), tmp_path / "out" / "bud.csv"
    )

    assert stem in res["error"], res


def test_predictions_that_never_classified_the_positive_state_refuse(
    tmp_path: Path,
) -> None:
    """Predictions that classify no positive state refuse.

    Detector buckets over the same captures carry no opening axis at all, so the delivery refuses
    saying they classify no ``opening='open'``, and writes no CSV.
    """
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series, predicted, published

    series = attributed_series(tmp_path, fractions=(0.0, 1.0), assessed=False)
    bare = []
    for date in series.buckets:
        images = sorted(image_dir(series.root, date).iterdir())
        results = [predicted(p, ["bud", "bud"]) for p in images]
        bare.append(
            published(
                tmp_path, f"bare/{date}", results, scope={"subject": "bud"}
            ).name
        )
    out_csv = tmp_path / "out" / "bud.csv"

    res = deliver_milestones(tmp_path, series.body(buckets=bare), out_csv)

    assert "classify no opening='open'" in res["error"], res
    assert not out_csv.exists()


def test_each_bucket_stands_for_the_date_its_record_states_and_two_on_one_date_refuse(
    tmp_path: Path,
) -> None:
    """Each bucket stands for the date its record states, and two on one date refuse.

    A bucket published under a name naming another date delivers as the date its record states,
    and two buckets recording one date refuse naming that shared date, writing no CSV.
    """
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.inference_tools import run_inference
    from tests._chain_fixtures import attributed_series

    series = attributed_series(tmp_path, fractions=(0.0, 1.0))
    first, second = sorted(series.buckets)
    misnamed = "misnamed/2099-12-31"
    published = run_inference(
        tmp_path,
        checkpoint_path=series.checkpoint_path,
        images_dir=str(image_dir(series.root, second)),
        bucket=misnamed,
        assessment_id=series.assessment["assessment_id"],
        stated=Stated(),
        actor=None,
    )
    assert "error" not in published, published
    assert published["date"] == second

    shipped = deliver_milestones(
        tmp_path,
        series.body(buckets=[series.buckets[first], misnamed]),
        tmp_path / "out" / "a.csv",
    )
    both = deliver_milestones(
        tmp_path,
        series.body(buckets=[*series.buckets.values(), misnamed]),
        tmp_path / "out" / "b.csv",
    )

    assert "error" not in shipped, shipped
    assert csv_rows(tmp_path / "out" / "a.csv")[0]["n_dates"] == "2"
    assert f"both record capture date {second}" in both["error"]
    assert not (tmp_path / "out" / "b.csv").exists()


def test_a_missing_mapping_and_an_unknown_trait_each_refuse(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from tests._chain_fixtures import attributed_series

    body = attributed_series(tmp_path, fractions=(0.0, 1.0)).body()

    missing = deliver_milestones(
        tmp_path, {**body, "mapping_name": "nope"}, tmp_path / "a.csv"
    )
    unknown = deliver_milestones(
        tmp_path, {**body, "trait": "not-a-real-trait"}, tmp_path / "b.csv"
    )

    assert "not found" in missing["error"]
    assert "error" in unknown
    assert (
        not (tmp_path / "a.csv").exists() and not (tmp_path / "b.csv").exists()
    )

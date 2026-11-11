"""Tests for the general per-plant CSV door.

Per-image results, each naming its image by capture date and stem, are attributed to plants by a
mapping built by ``build_plant_mapping``, aggregated, and delivered over published buckets.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tcip_mcp.dataset_layout import image_dir
from tests import _trait_fixtures as fx, csv_rows
from tests._mapping_fixtures import COUNT_CROP, PLOTS, write_plots_csv

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import JsonValue

torch = pytest.importorskip("torch")

KIND = "per_plant_count_aggregate"
SCOPE = {"subject": fx.COUNT_SUBJECT}
pytestmark = pytest.mark.usefixtures("confirmed_count_aggregate")

PLANT_CSV = "plants.csv"
"""The file name, under the project, of the plant CSV :func:`_scene` writes and registers."""

ONE_PLANT = {"P1": (43.19670, -90.058000)}
TWO_PLANTS = {**ONE_PLANT, "P2": (43.19680, -90.057000)}


def _scene(
    tmp_path: Path,
    plants: Mapping[str, tuple[float, float]],
    captures: Mapping[str, Mapping[str, tuple[float, float]]],
) -> tuple[Path, Path]:
    """Write a registered dataset of geolocated images and a plant CSV naming ``plants``.

    ``captures`` holds each date's images by stem, each taken at its ``(lat, lon)``, five minutes
    apart in the order given. Returns ``(dataset root, plant csv)``.
    """
    from tcip_mcp.tools.project_tools import initialize_project
    from tests._mapping_fixtures import PHOTO_POSITION_ERROR_M, write_photos
    from tests._producer_fixtures import register_dataset_at
    from tests._web_fixtures import INVARIANCES

    assert "error" not in initialize_project(
        str(tmp_path),
        "Orchard",
        site="orchard block",
        invariances=INVARIANCES,
        actor=None,
    )
    dataset_root = tmp_path / "ds"
    for date, taken in captures.items():
        write_photos(
            image_dir(dataset_root, date),
            {
                f"{stem}.jpg": (lat, lon, PHOTO_POSITION_ERROR_M)
                for stem, (lat, lon) in taken.items()
            },
            datetime.fromisoformat(date).replace(hour=9, minute=30),
            timedelta(minutes=5),
        )
    register_dataset_at(tmp_path, dataset_root, crop=COUNT_CROP)
    plant_csv = write_plots_csv(tmp_path / PLANT_CSV, plants)
    return dataset_root, plant_csv


DATE = "2026-02-11"


def _registered(
    tmp_path: Path,
    plants: Mapping[str, tuple[float, float]],
    captures: Mapping[str, Mapping[str, tuple[float, float]]],
) -> tuple[Path, str]:
    """Write :func:`_scene` over ``captures``, register its plants; return the root and registry."""
    from tests._mapping_fixtures import register_plant_registry_for

    dataset_root, plant_csv = _scene(tmp_path, plants, captures)
    return dataset_root, register_plant_registry_for(
        tmp_path, [plant_csv], crop=COUNT_CROP
    )


def _build(
    tmp_path: Path,
    dataset_root: Path,
    registry: str,
    dates: list[str] | None,
) -> dict:
    """Call the tool ``build_plant_mapping`` for the mapping ``valley`` over ``dates``."""
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    return build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=registry,
        dates=dates,
        actor=None,
    )


def _mapped(
    tmp_path: Path,
    plants: Mapping[str, tuple[float, float]] = ONE_PLANT,
    captures: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
    dates: list[str] | None = None,
) -> tuple[Path, str]:
    """Build :func:`_scene` with the mapping ``valley`` over it; return ``(dataset root, date)``.

    The captures are ``captures``, by default one image per plant on :data:`DATE`, each named by
    its plant and taken at its place; the date returned is the first. The mapping covers
    ``dates`` of them, every date when ``None``.
    """
    taken = captures or {DATE: dict(plants)}
    dataset_root, registry = _registered(tmp_path, plants, taken)
    mapped = _build(tmp_path, dataset_root, registry, dates)
    assert "error" not in mapped, mapped
    return dataset_root, next(iter(taken))


def _bucket(
    project: Path,
    dataset_root: Path,
    date: str,
    counts: dict[str, int],
    name: str = "manual",
) -> str:
    """Publish a bucket named ``<name>/<date>`` under ``project`` and return its name.

    The bucket lies under ``dataset_root``, registered at :data:`COUNT_CROP` when it carries no
    identity (``_chain_fixtures.published``), and holds ``counts[stem]`` detections on each
    image ``images/<date>/<stem>.jpg``.
    """
    from tests._chain_fixtures import detected, detection_pass, published

    results = [
        detection_pass(
            image_dir(dataset_root, date) / f"{stem}.jpg",
            (8, 8),
            [detected((1.0, 1.0, 3.0, 3.0), 0.9)] * n,
            cap=n + 1,
        )
        for stem, n in counts.items()
    ]
    return published(
        project, f"{name}/{date}", results, scope=SCOPE, crop=COUNT_CROP
    ).name


def door_results(rows: list[dict]) -> list:
    """Return ``rows`` as ``deliver_per_plant_csv`` takes its ``results``, through its schema."""
    from typing import get_type_hints

    from pydantic import TypeAdapter

    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    return TypeAdapter(
        get_type_hints(deliver_per_plant_csv)["results"]
    ).validate_python(rows)


def _results(date: str, *stem_values: tuple[str, int | None]) -> list:
    """Return one result per ``(stem, value)`` on ``date``, ``None`` an unobserved value."""
    return door_results(
        [
            {"date": date, "stem": stem, "value": value}
            for stem, value in stem_values
        ]
    )


def _deliver(
    tmp_path: Path,
    results: list,
    plants: list[str],
    dataset_root: Path,
    buckets: list[str],
    **kwargs: str | None,
) -> dict:
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    return deliver_per_plant_csv(
        tmp_path,
        results,
        str(tmp_path / "o.csv"),
        "stem_count",
        KIND,
        plants,
        str(dataset_root),
        buckets,
        value_key="count",
        **kwargs,
        actor=None,
    )


def test_a_verified_mapping_over_an_unassessed_bucket_reaches_the_gate_and_refuses_there(
    tmp_path: Path,
) -> None:
    """A verified mapping over an unassessed bucket reaches the gate and refuses there.

    Over a mapping built on a registered registry, a published bucket and per-image results, the
    delivery reaches the one gate; the bucket was published under no assessment, so it refuses
    there, writing no output CSV.
    """
    dataset_root, date = _mapped(tmp_path, TWO_PLANTS)
    bucket = _bucket(
        tmp_path, dataset_root, date, {"P1": 2, "P2": 1}, name="run"
    )

    refused = _deliver(
        tmp_path,
        _results(date, ("P1", 2), ("P2", 1)),
        ["P1", "P2"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "no assessment answers" in refused["error"]
    assert not (tmp_path / "o.csv").exists()


def test_an_unknown_plant_mapping_refuses_by_name(tmp_path: Path) -> None:
    """A ``plant_mapping`` with no stored record is refused by name, and no CSV is written.

    The refusal names the mapping and ``build_plant_mapping``.
    """
    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})

    res = _deliver(
        tmp_path,
        _results(date, ("P1", 3)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="no-such-mapping",
    )

    assert "no-such-mapping" in res["error"]
    assert "build_plant_mapping" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_delivered_date_the_mapping_does_not_cover_refuses(
    tmp_path: Path,
) -> None:
    """A bucket on a date the mapping does not cover refuses, its step naming what a rebuild needs.

    The step puts ingesting the date's captures before the rebuild.
    """
    from tests._image_fixtures import write_geo_image

    dataset_root, date = _mapped(tmp_path)
    uncovered = "2026-03-01"
    write_geo_image(
        image_dir(dataset_root, uncovered) / "P1.jpg",
        *ONE_PLANT["P1"],
        datetime(2026, 3, 1, 9, 30),
    )

    res = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        dataset_root,
        [
            _bucket(tmp_path, dataset_root, date, {"P1": 1}),
            _bucket(tmp_path, dataset_root, uncovered, {"P1": 1}),
        ],
        plant_mapping="valley",
    )

    assert "does not cover" in res["error"] and uncovered in res["error"]
    assert (
        "ingest any of those dates' captures the dataset lacks, rebuild the mapping to cover "
        "them, or drop those buckets"
    ) in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_predictions_under_another_dataset_than_the_mappings_refuse(
    tmp_path: Path,
) -> None:
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image
    from tests._producer_fixtures import register_dataset_at

    _dataset_root, date = _mapped(tmp_path)
    other = tmp_path / "ds2"
    write_geo_image(
        image_dir(other, date) / "P1.jpg",
        *ONE_PLANT["P1"],
        datetime(2026, 2, 11, 9, 30),
    )
    register_dataset_at(tmp_path, other, crop=sorted(registered_crops())[0])

    res = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        other,
        [_bucket(tmp_path, other, date, {"P1": 1})],
        plant_mapping="valley",
    )

    assert "different dataset" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_mapping_with_no_capture_at_all_for_a_delivered_date_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mapping with no capture for a delivered date refuses on the record's own evidence.

    The mapping is recorded with no capture at all for that date.
    """
    from tests.test_ungeoreferenced_capture_refusal import (
        DATE,
        _delivery_scene,
        _persist_synthetic_mapping,
    )

    _, dataset_root, preds_by_date = _delivery_scene(tmp_path, monkeypatch)
    plant_csv = write_plots_csv(tmp_path / "plants.csv")
    _persist_synthetic_mapping(
        tmp_path,
        dataset_root,
        "valley",
        plant_csvs=[
            {
                "path": str(plant_csv),
                "sha256": "0" * 64,
                "n_plants": len(PLOTS),
            }
        ],
        assignments={DATE: []},
    )

    res = _deliver(
        tmp_path,
        _results(DATE, ("P1", 3)),
        ["P1"],
        dataset_root,
        list(preds_by_date.values()),
        plant_mapping="valley",
    )

    assert "recorded no capture at all" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_delivered_plant_the_mapping_never_assigned_refuses_by_name(
    tmp_path: Path,
) -> None:
    """A delivered plant the mapping never assigned on a delivered date is refused by name.

    The refusal names the unassigned plant and not the assigned one, and no output CSV is
    written.
    """
    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})

    res = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1", "GHOST"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "GHOST" in res["error"] and "'P1'" not in res["error"]
    assert not (tmp_path / "o.csv").exists()


def test_a_capture_added_since_the_mapping_was_built_refuses(
    tmp_path: Path,
) -> None:
    """A capture added since the mapping was built refuses the delivery.

    The refusal says the mapping does not cover what is on disk now, and no CSV is written.
    """
    from tests._image_fixtures import write_geo_image

    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    write_geo_image(
        image_dir(dataset_root, date) / "P2.jpg",
        43.19680,
        -90.057000,
        datetime(2026, 2, 11, 9, 35),
    )

    res = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert "does not cover what is on disk now" in res["error"]
    assert not (tmp_path / "o.csv").exists()


def _removed(plant_csv: Path) -> str:
    """Remove ``plant_csv``; return the phrase a refusal of a missing CSV states."""
    plant_csv.unlink()
    return "not found"


def _rewritten(plant_csv: Path) -> str:
    """Rewrite ``plant_csv`` with its plant moved; return the phrase a refusal of it states."""
    write_plots_csv(plant_csv, {"P1": (43.19675, -90.058000)})
    return "rewritten"


@pytest.mark.parametrize(
    "change", [_removed, _rewritten], ids=lambda f: f.__name__.strip("_")
)
def test_a_registered_plant_csv_gone_or_rewritten_refuses_the_delivery_naming_it(
    tmp_path: Path, change: Callable[[Path], str]
) -> None:
    """A plant CSV removed or rewritten since its registration refuses the door, naming the file.

    The refusal comes before the gate, and no output is written.
    """
    dataset_root, date = _mapped(tmp_path)
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    plant_csv = tmp_path / PLANT_CSV
    phrase = change(plant_csv)

    res = _deliver(
        tmp_path,
        _results(date, ("P1", 1)),
        ["P1"],
        dataset_root,
        [bucket],
        plant_mapping="valley",
    )

    assert phrase in res["error"] and str(plant_csv) in res["error"], res
    assert not (tmp_path / "o.csv").exists()


# ── each result counts toward the plant the mapping assigned its image ──

FIRST, SECOND = "2026-02-11", "2026-02-25"
FAR = (43.19700, -90.058500)
"""A place about 33 m from either of :data:`TWO_PLANTS`, beyond the photographs' reported error."""
CAPTURES = {
    FIRST: {
        "a": TWO_PLANTS["P1"],
        "b": TWO_PLANTS["P1"],
        "c": TWO_PLANTS["P2"],
        "far": FAR,
    },
    SECOND: {"a": TWO_PLANTS["P2"], "c": TWO_PLANTS["P1"]},
}
"""Each date's photographs by stem and place, in capture order.

Stem ``a`` stands at ``P1`` on the first date and at ``P2`` on the second; ``b`` finds ``P1``
already claimed by ``a``, so its assignment's source is not the others'; ``far`` stands at
:data:`FAR`, so the mapping attributes it to no plant."""
READ = (
    (FIRST, "a", 1),
    (FIRST, "b", 2),
    (FIRST, "c", 4),
    (SECOND, "a", 8),
    (SECOND, "c", 16),
)
"""A value for each image of :data:`CAPTURES` the mapping attributes, each a distinct power of
two."""
NO_READ_IMAGE = "the delivery read none of the images of plant"
"""The phrase of the refusal of a delivered plant the delivery read no image of."""
UNCOVERED = "no result names them"
"""The phrase of the refusal of read images no result names."""
UNREAD_CAUSE = (
    "no prediction document in the delivered bucket, or no longer on disk"
)
"""Why an image was not read, as both refusals of one state it."""
NOT_COVERED = "on a date the mapping does not cover"
NOT_COVERED_STEP = (
    "ingest any of those dates' captures the dataset lacks, rebuild the mapping to cover those "
    "dates and deliver those dates' buckets, or drop those results"
)
UNDELIVERED = "on a date that is not a delivered date"
UNDELIVERED_STEP = "deliver those dates' buckets or drop those results"
NO_IMAGE = "the mapping records no image of that stem on that date"
NO_IMAGE_STEP = "correct the date and stem, or drop those results"
NO_PLANT = "the mapping assigned to no plant"
NO_PLANT_STEP = "drop those results, which belong to no plant"
OUTSIDE = "of a plant the population does not name"
OUTSIDE_STEP = "deliver those plants or drop those results"
UNREAD = f"not read by the delivery ({UNREAD_CAUSE})"
UNREAD_STEP = "drop those results, or deliver a bucket documenting them with the images on disk"
REPEATED = "named by more than one result"
REPEATED_STEP = "name each image in one result only"
REASONS = (
    (NOT_COVERED, NOT_COVERED_STEP),
    (UNDELIVERED, UNDELIVERED_STEP),
    (NO_IMAGE, NO_IMAGE_STEP),
    (NO_PLANT, NO_PLANT_STEP),
    (OUTSIDE, OUTSIDE_STEP),
    (UNREAD, UNREAD_STEP),
    (REPEATED, REPEATED_STEP),
)
"""Each reason a result is refused for, as its phrase and its next step, in the order the
refusal gives them."""


def _rows(*images: tuple[str, str, int | None]) -> list[dict]:
    """Return one result row per ``(date, stem, value)``."""
    return [
        {"date": date, "stem": stem, "value": value}
        for date, stem, value in images
    ]


def _but(*left_out: tuple[str, str]) -> list[dict]:
    """Return the rows of :data:`READ` but those of the ``(date, stem)`` images ``left_out``."""
    return _rows(*(r for r in READ if r[:2] not in left_out))


def _published(
    project: Path,
    dataset_root: Path,
    without: frozenset = frozenset(),
    name: str = "manual",
    captures: Mapping[str, Mapping[str, tuple[float, float]]] = CAPTURES,
) -> list[str]:
    """Publish one bucket per date of ``captures``, named ``<name>/<date>``; return their names.

    Each holds a document for every image of its date but the ``(date, stem)`` pairs ``without``
    names.
    """
    return [
        _bucket(
            project,
            dataset_root,
            date,
            {stem: 1 for stem in taken if (date, stem) not in without},
            name,
        )
        for date, taken in captures.items()
    ]


def _both(project: Path, dataset_root: Path) -> list[str]:
    """Publish both dates (:func:`_published`)."""
    return _published(project, dataset_root)


def _first_only(project: Path, dataset_root: Path) -> list[str]:
    """Publish both dates and deliver only the first's bucket."""
    return _published(project, dataset_root)[:1]


def _b_undocumented(project: Path, dataset_root: Path) -> list[str]:
    """Publish both dates, the first's bucket holding no document for ``b``."""
    return _published(project, dataset_root, frozenset({(FIRST, "b")}))


def _b_off_disk(project: Path, dataset_root: Path) -> list[str]:
    """Publish both dates, then remove the first date's ``b`` from disk."""
    buckets = _published(project, dataset_root)
    (image_dir(dataset_root, FIRST) / "b.jpg").unlink()
    return buckets


def _second_folder_gone(project: Path, dataset_root: Path) -> list[str]:
    """Publish both dates, then remove the second date's image folder."""
    buckets = _published(project, dataset_root)
    shutil.rmtree(image_dir(dataset_root, SECOND))
    return buckets


def _p2_unread(
    project: Path, dataset_root: Path, name: str = "manual"
) -> list[str]:
    """Publish both dates, the first's ``c`` undocumented, then remove the second's folder.

    The delivery reads no image of ``P2``: its first-date image has no document and its
    second-date image is gone with its folder.
    """
    buckets = _published(
        project, dataset_root, frozenset({(FIRST, "c")}), name
    )
    shutil.rmtree(image_dir(dataset_root, SECOND))
    return buckets


def _arguments_over(
    project: Path,
    dataset_root: Path,
    buckets: list[str],
    rows: list[dict],
    plants: tuple[str, ...] = ("P1", "P2"),
) -> dict:
    """Return the door's arguments delivering ``rows`` for ``plants`` over ``buckets``."""
    return {
        "results": rows,
        "output_path": str(project / "o.csv"),
        "delivered_phenotype": "stem_count",
        "delivery_kind": KIND,
        "plants": list(plants),
        "dataset_root": str(dataset_root),
        "buckets": buckets,
        "value_key": "count",
        "plant_mapping": "valley",
    }


def _called(project: Path, arguments: dict) -> JsonValue:
    """Call the door through the server bound to ``project``; return its answer."""
    from tcip_mcp.server import binding_for
    from tests._server_calls import called_through_the_server

    return called_through_the_server(
        "deliver_per_plant_csv", arguments, bound=binding_for(project)
    )


def _delivered(project: Path, arguments: dict) -> JsonValue:
    """Call the door through the server under an acknowledgment of exactly ``arguments``."""
    from tests.test_delivery_events import _acknowledged_per_plant_csv

    _, acknowledged = _acknowledged_per_plant_csv(project, arguments)
    return _called(project, acknowledged)


def _by_plant(project: Path) -> dict[str, dict]:
    """Return the rows of the CSV the door wrote under ``project``, by plant."""
    return {row["plant_id"]: row for row in csv_rows(project / "o.csv")}


def test_each_row_is_the_aggregate_of_the_images_the_mapping_assigned_its_plant(
    tmp_path: Path,
) -> None:
    """Each row's plant, value and per-image cells come from the mapping's own assignments.

    Stem ``a`` stands at a different plant on each date and ``P1``'s images are assigned by
    different sources; each plant's value is the median of its own images' values, and its
    ``n_images``, ``plant_id_source`` and ``plant_id_distance_m_max`` are those of the
    assignments the mapping persisted for them, read back from the written CSV.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    arguments = _arguments_over(
        tmp_path, dataset_root, _both(tmp_path, dataset_root), _rows(*READ)
    )

    answer = _delivered(tmp_path, arguments)

    assert isinstance(answer, dict) and "error" not in answer, answer
    build = load_mapping(tmp_path, "valley")
    assert build is not None
    of_plant = {
        plant: [
            build.by_stem[date][stem]
            for date, stem, _ in READ
            if build.by_stem[date][stem].plant_id == plant
        ]
        for plant in ("P1", "P2")
    }
    assert {a.source for a in of_plant["P1"]} == {
        "sequence",
        "nearest_neighbor",
    }
    rows = _by_plant(tmp_path)
    assert {p: float(r["value"]) for p, r in rows.items()} == {
        "P1": 2.0,
        "P2": 6.0,
    }
    for plant, assigned in of_plant.items():
        sources = {a.source for a in assigned}
        assert rows[plant]["n_images"] == str(len(assigned))
        assert rows[plant]["plant_id_source"] == (
            sources.pop() if len(sources) == 1 else "mixed"
        )
        assert float(rows[plant]["plant_id_distance_m_max"]) == pytest.approx(
            max(a.distance_m for a in assigned if a.distance_m is not None)
        )


@pytest.mark.parametrize(
    ("publish", "rows", "plants", "phrase", "named"),
    [
        (
            _first_only,
            _rows(*READ[:3], (SECOND, "a", 8)),
            ("P1", "P2"),
            UNDELIVERED,
            f"{SECOND}/a",
        ),
        (
            _both,
            _rows(*READ, ("2026-03-01", "x", 32)),
            ("P1", "P2"),
            NOT_COVERED,
            "2026-03-01/x",
        ),
        (
            _both,
            _rows(*READ, (SECOND, "b", 32)),
            ("P1", "P2"),
            NO_IMAGE,
            f"{SECOND}/b",
        ),
        (
            _both,
            _rows(*READ, (FIRST, "far", 32)),
            ("P1", "P2"),
            NO_PLANT,
            f"{FIRST}/far",
        ),
        (
            _both,
            _but((SECOND, "a")),
            ("P1",),
            OUTSIDE,
            f"{FIRST}/c (plant 'P2')",
        ),
        (
            _b_undocumented,
            _rows(*READ),
            ("P1", "P2"),
            UNREAD,
            f"{FIRST}/b",
        ),
        (
            _b_off_disk,
            _rows(*READ),
            ("P1", "P2"),
            UNREAD,
            f"{FIRST}/b",
        ),
        (
            _second_folder_gone,
            _rows(*READ),
            ("P1", "P2"),
            UNREAD,
            f"{SECOND}/a",
        ),
        (
            _both,
            _rows(*READ, READ[0]),
            ("P1", "P2"),
            REPEATED,
            f"{FIRST}/a",
        ),
        (
            _both,
            _but((FIRST, "b")),
            ("P1", "P2"),
            UNCOVERED,
            f"{FIRST}/b",
        ),
        (
            _p2_unread,
            _rows(*READ[:2]),
            ("P1", "P2"),
            NO_READ_IMAGE,
            f"'P2' ({FIRST}/c, {SECOND}/a)",
        ),
        (
            _p2_unread,
            _rows(READ[0]),
            ("P1", "P2"),
            NO_READ_IMAGE,
            f"'P2' ({FIRST}/c, {SECOND}/a)",
        ),
    ],
    ids=[
        "undelivered_date",
        "date_the_mapping_does_not_cover",
        "stem_of_another_date",
        "unattributed",
        "plant_outside_the_population",
        "no_document",
        "not_on_disk",
        "absent_folder",
        "repeated_image",
        "uncovered_image",
        "plant_with_no_read_image",
        "plant_with_no_read_image_and_an_uncovered_image",
    ],
)
def test_a_result_the_mapping_does_not_attribute_to_a_read_image_of_a_delivered_plant_refuses(
    tmp_path: Path,
    publish: Callable[[Path, Path], list[str]],
    rows: list[dict],
    plants: tuple[str, ...],
    phrase: str,
    named: str,
) -> None:
    """A result the mapping cannot place, or a read image no result covers, refuses by image.

    The refusal names the image and its reason with that reason's next step alone, through the
    server's door, and no delivery event or CSV is written. A delivered plant the delivery read
    no image of refuses naming it and its images, ahead of the read images no result names.
    """
    from tcip_mcp.delivery import read_delivery_events

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    arguments = _arguments_over(
        tmp_path,
        dataset_root,
        publish(tmp_path, dataset_root),
        rows,
        plants,
    )

    refused = _called(tmp_path, arguments)

    assert isinstance(refused, dict), refused
    assert phrase in refused.get("error", ""), refused
    assert named in refused["error"], refused
    assert (UNCOVERED in refused["error"]) == (phrase == UNCOVERED), refused
    own = [(p, s) for p, s in REASONS if p == phrase]
    assert [
        (p, s)
        for p, s in REASONS
        if p in refused["error"] or s in refused["error"]
    ] == own, refused
    if own:
        _named_under(refused["error"], tuple(own))
    assert "image '" not in refused["error"], refused
    assert "['" not in refused["error"], refused
    assert read_delivery_events(tmp_path) == []
    assert not (tmp_path / "o.csv").exists()


@pytest.mark.parametrize(
    ("covered", "phrase", "step"),
    [
        ([FIRST], NOT_COVERED, NOT_COVERED_STEP),
        (None, UNDELIVERED, UNDELIVERED_STEP),
    ],
    ids=["date_the_mapping_does_not_cover", "covered_date_not_delivered"],
)
def test_the_next_step_of_a_result_on_a_date_not_delivered_leads_to_a_delivery(
    tmp_path: Path,
    covered: list[str] | None,
    phrase: str,
    step: str,
) -> None:
    """Following the next step of a refused date's result reaches a delivery that succeeds.

    The first call delivers the first date's bucket alone, with a result on the second date: the
    mapping covers only the first date in one case and both in the other, so the refusal gives
    that case's own step. In the first case the step rebuilds the mapping to cover the second
    date; in both it delivers both dates' buckets, with every read image's result.
    """
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping
    from tcip_mcp.tools.phenology_tools import build_plant_mapping

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES, covered)
    buckets = _both(tmp_path, dataset_root)
    first = _arguments_over(
        tmp_path,
        dataset_root,
        buckets[:1],
        _rows(*READ[:3], (SECOND, "a", 8)),
    )

    refused = _called(tmp_path, first)

    assert isinstance(refused, dict), refused
    assert _named_under(refused["error"], ((phrase, step),)) == {
        phrase: [f"{SECOND}/a"]
    }
    assert not [s for p, s in REASONS if p != phrase and s in refused["error"]]
    if covered is not None:
        built = load_mapping(tmp_path, first["plant_mapping"])
        assert built is not None
        rebuilt = build_plant_mapping(
            tmp_path,
            name=built.name,
            dataset_root=str(dataset_root),
            plant_registry=built.plant_registry.name,
            actor=None,
        )
        assert "error" not in rebuilt, rebuilt
    answer = _delivered(
        tmp_path,
        _arguments_over(tmp_path, dataset_root, buckets, _rows(*READ)),
    )

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert len(read_delivery_events(tmp_path)) == 1
    assert set(_by_plant(tmp_path)) == {"P1", "P2"}


IMAGE_NAME = r"\d{4}-\d{2}-\d{2}/\w+"
"""The pattern of an image named as ``<date>/<stem>``."""
MIXED_PLANTS = {**TWO_PLANTS, "P3": (43.19640, -90.059000)}
MIXED_CAPTURES = {
    FIRST: {
        "a": MIXED_PLANTS["P1"],
        "b": MIXED_PLANTS["P1"],
        "c": MIXED_PLANTS["P2"],
        "d": MIXED_PLANTS["P3"],
        "far": FAR,
    },
    SECOND: {"a": MIXED_PLANTS["P2"], "c": MIXED_PLANTS["P1"]},
}
"""A scene with a third plant, so two images can be of two different plants outside a
population."""


def _named_under(
    error: str, reasons: tuple[tuple[str, str], ...] = REASONS
) -> dict[str, list[str]]:
    """Return, for each of ``reasons`` the ``error`` states, the images it names under it.

    Each reason's phrase and next step must appear once, the phrases in the order given and the
    step after its phrase and before the next one; an image is named between its phrase and its
    step, as ``<date>/<stem>`` with the plant of one outside the population in parentheses.
    """
    assert all(
        error.count(p) == 1 and error.count(s) == 1 for p, s in reasons
    ), error
    phrase_at = [error.index(phrase) for phrase, _ in reasons]
    step_at = [error.index(step) for _, step in reasons]
    after = [*phrase_at[1:], len(error)]
    assert all(
        start < step < end
        for start, step, end in zip(phrase_at, step_at, after, strict=True)
    ), error
    return {
        phrase: re.findall(
            rf"{IMAGE_NAME}(?: \(plant '\w+'\))?",
            error[start + len(phrase) : step],
        )
        for (phrase, _), start, step in zip(
            reasons, phrase_at, step_at, strict=True
        )
    }


def test_one_refusal_names_every_inadmissible_image_under_its_reason(
    tmp_path: Path,
) -> None:
    """Every reason a result can be refused for is stated once, with all its images and its step.

    One call carries every reason beside an admissible image, submitted out of the order the
    refusal gives them: an image on a date the mapping does not cover, an image on a date the
    mapping covers and the delivery does not deliver, a stem the mapping records no image
    of, an unmapped capture submitted twice, two images of two different plants outside the
    population, an image the delivery did not read, and the admissible image submitted twice.
    The refusal names each image once as ``<date>/<stem>`` in the order submitted, the plant of
    each image outside the population after it, and leaves no delivery event or CSV.
    """
    from tcip_mcp.delivery import read_delivery_events

    dataset_root, _ = _mapped(tmp_path, MIXED_PLANTS, MIXED_CAPTURES)
    buckets = _published(
        tmp_path,
        dataset_root,
        frozenset({(FIRST, "b")}),
        captures={FIRST: MIXED_CAPTURES[FIRST]},
    )
    rows = _rows(
        (FIRST, "d", 1),
        (FIRST, "far", 1),
        (FIRST, "a", 1),
        (FIRST, "b", 1),
        (SECOND, "a", 1),
        (FIRST, "zz", 1),
        (FIRST, "c", 1),
        (FIRST, "far", 1),
        (FIRST, "a", 1),
        ("2026-03-02", "y", 1),
    )

    refused = _called(
        tmp_path,
        _arguments_over(tmp_path, dataset_root, buckets, rows, ("P1",)),
    )

    assert isinstance(refused, dict), refused
    for stem in ("zz", "far", "d", "c", "b"):
        assert f"{FIRST}/{stem}" in refused["error"], refused
    assert _named_under(refused["error"]) == {
        NOT_COVERED: ["2026-03-02/y"],
        UNDELIVERED: [f"{SECOND}/a"],
        NO_IMAGE: [f"{FIRST}/zz"],
        NO_PLANT: [f"{FIRST}/far"],
        OUTSIDE: [f"{FIRST}/d (plant 'P3')", f"{FIRST}/c (plant 'P2')"],
        UNREAD: [f"{FIRST}/b"],
        REPEATED: [f"{FIRST}/a"],
    }
    assert "image '" not in refused["error"], refused
    assert read_delivery_events(tmp_path) == []
    assert not (tmp_path / "o.csv").exists()


def test_a_refusal_names_every_unmapped_capture_without_a_cap(
    tmp_path: Path,
) -> None:
    """More than twenty captures the mapping attributes to no plant are all named.

    Each is submitted, in reverse of the order its photograph was taken, beside the plant's own
    image; the one refusal names every one in the order submitted, and writes no event or CSV.
    """
    from tcip_mcp.delivery import read_delivery_events

    unmapped = [f"u{i:02d}" for i in range(25)]
    dataset_root, date = _mapped(
        tmp_path,
        ONE_PLANT,
        {DATE: {"P1": ONE_PLANT["P1"], **dict.fromkeys(unmapped, FAR)}},
    )
    bucket = _bucket(tmp_path, dataset_root, date, {"P1": 1})
    submitted = [*reversed(unmapped)]
    arguments = _arguments_over(
        tmp_path,
        dataset_root,
        [bucket],
        _rows((date, "P1", 1), *((date, stem, 1) for stem in submitted)),
        ("P1",),
    )

    refused = _called(tmp_path, arguments)

    assert isinstance(refused, dict), refused
    for stem in unmapped:
        assert f"{date}/{stem}" in refused["error"], refused
    assert _named_under(refused["error"], ((NO_PLANT, NO_PLANT_STEP),)) == {
        NO_PLANT: [f"{date}/{stem}" for stem in submitted]
    }
    assert read_delivery_events(tmp_path) == []
    assert not (tmp_path / "o.csv").exists()


@pytest.mark.parametrize(
    ("publish", "rows", "plants", "named", "reason", "absent"),
    [
        (
            _b_undocumented,
            _rows(*READ),
            ("P2",),
            [
                f"{FIRST}/a (plant 'P1')",
                f"{FIRST}/b (plant 'P1')",
                f"{SECOND}/c (plant 'P1')",
            ],
            (OUTSIDE, OUTSIDE_STEP),
            UNREAD,
        ),
        (
            _b_undocumented,
            _rows(*READ, (FIRST, "b", 2)),
            ("P1", "P2"),
            [f"{FIRST}/b"],
            (UNREAD, UNREAD_STEP),
            REPEATED,
        ),
    ],
    ids=["outside_the_population_and_unread", "unread_and_repeated"],
)
def test_an_image_failing_two_conditions_is_named_under_the_first(
    tmp_path: Path,
    publish: Callable[[Path, Path], list[str]],
    rows: list[dict],
    plants: tuple[str, ...],
    named: list[str],
    reason: tuple[str, str],
    absent: str,
) -> None:
    """The checks run in the order the refusal gives the reasons, and an image takes one.

    An image of a plant outside the population the delivery also did not read is named under
    the population reason; an unread image submitted twice is named once under the unread one.
    """
    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)

    refused = _called(
        tmp_path,
        _arguments_over(
            tmp_path,
            dataset_root,
            publish(tmp_path, dataset_root),
            rows,
            plants,
        ),
    )

    assert isinstance(refused, dict), refused
    assert _named_under(refused["error"], (reason,)) == {reason[0]: named}
    assert absent not in refused["error"], refused


def test_an_inadmissible_result_refuses_before_a_plant_nothing_was_read_of(
    tmp_path: Path,
) -> None:
    """A result the door does not admit is refused alone, ahead of the delivery-wide refusals.

    The delivery reads no image of ``P2`` and a read image of ``P1`` is left unnamed, yet the
    refusal states neither: it names the one inadmissible result.
    """
    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    arguments = _arguments_over(
        tmp_path,
        dataset_root,
        _p2_unread(tmp_path, dataset_root),
        _rows(READ[0], ("2026-03-01", "x", 1)),
    )

    refused = _called(tmp_path, arguments)

    assert isinstance(refused, dict), refused
    assert _named_under(
        refused["error"], ((NOT_COVERED, NOT_COVERED_STEP),)
    ) == {NOT_COVERED: ["2026-03-01/x"]}
    assert UNDELIVERED not in refused["error"]
    assert NO_READ_IMAGE not in refused["error"]
    assert UNCOVERED not in refused["error"]


@pytest.mark.parametrize(
    ("rows", "unnamed"),
    [([], READ), (_rows(*READ[:3]), READ[3:])],
    ids=["no_results", "one_of_two_dates"],
)
def test_results_leaving_read_images_unnamed_refuse_naming_every_one(
    tmp_path: Path, rows: list[dict], unnamed: tuple[tuple[str, str, int], ...]
) -> None:
    """Read images of the delivered plants no result names are all named, as dates and stems.

    With no result at all, and with the results of the first date alone, each plant has read
    images on both dates; the refusal names every read image no result names, one spelling.
    """
    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)

    refused = _called(
        tmp_path,
        _arguments_over(
            tmp_path, dataset_root, _both(tmp_path, dataset_root), rows
        ),
    )

    assert isinstance(refused, dict), refused
    assert UNCOVERED in refused["error"], refused
    assert re.findall(IMAGE_NAME, refused["error"]) == [
        f"{date}/{stem}" for date, stem, _ in unnamed
    ]
    assert "['" not in refused["error"], refused


@pytest.mark.parametrize(
    ("publish", "plants"),
    [
        (_both, ("P1", "P2")),
        (_both, ("P1",)),
        (_first_only, ("P1", "P2")),
        (_b_undocumented, ("P1", "P2")),
        (_second_folder_gone, ("P1", "P2")),
    ],
    ids=["all", "one_plant", "first_date", "no_document", "absent_folder"],
)
def test_the_door_admits_exactly_the_images_read_images_returns(
    tmp_path: Path,
    publish: Callable[[Path, Path], list[str]],
    plants: tuple[str, ...],
) -> None:
    """The images ``read_images`` returns deliver, and any other image the build records refuses.

    Both are produced from the one resolved mapping: the images read, each submitted once, are
    delivered, and the same images with every other image the build records added are refused,
    naming exactly those others.
    """
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        plant_mapping_disclosure,
        read_images,
    )

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    names = publish(tmp_path, dataset_root)
    resolved = plant_mapping_disclosure(
        tmp_path,
        "valley",
        by_recorded_date(read_bucket(dataset_root, n) for n in names),
        list(plants),
    )
    read = [
        i for images in read_images(resolved, plants).values() for i in images
    ]
    outside = [
        (date, a.stem)
        for date, rows in resolved.build.assignments.items()
        for a in rows
        if (date, a.stem) not in read
    ]
    assert outside

    refused = _called(
        tmp_path,
        _arguments_over(
            tmp_path,
            dataset_root,
            names,
            _rows(*((d, s, 1) for d, s in [*read, *outside])),
            plants,
        ),
    )
    answer = _delivered(
        tmp_path,
        _arguments_over(
            tmp_path,
            dataset_root,
            names,
            _rows(*((d, s, 1) for d, s in read)),
            plants,
        ),
    )

    assert isinstance(refused, dict), refused
    assert sorted(re.findall(IMAGE_NAME, refused["error"])) == sorted(
        f"{d}/{s}" for d, s in outside
    )
    assert isinstance(answer, dict) and "error" not in answer, answer
    assert set(_by_plant(tmp_path)) == set(plants)
    assert len(read_delivery_events(tmp_path)) == 1


def test_the_two_refusals_of_an_unread_image_state_one_cause(
    tmp_path: Path,
) -> None:
    """An unread image's refusal and an unread plant's refusal give the same cause.

    The first is the refusal of a result naming an image the delivered bucket holds no document
    for; the second, of a plant the delivery read no image of. Both are produced over the one
    scene, and each states the cause.
    """
    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    by_image = _called(
        tmp_path,
        _arguments_over(
            tmp_path,
            dataset_root,
            _b_undocumented(tmp_path, dataset_root),
            _rows(*READ),
        ),
    )
    by_plant = _called(
        tmp_path,
        _arguments_over(
            tmp_path,
            dataset_root,
            _p2_unread(tmp_path, dataset_root, "other"),
            _rows(*READ[:2]),
        ),
    )

    assert isinstance(by_image, dict) and isinstance(by_plant, dict)
    assert UNREAD in by_image["error"], by_image
    assert NO_READ_IMAGE in by_plant["error"], by_plant
    assert UNREAD_CAUSE in by_image["error"], by_image
    assert UNREAD_CAUSE in by_plant["error"], by_plant


@pytest.mark.parametrize(
    ("publish", "left_out", "disclosed"),
    [
        (_b_off_disk, ((FIRST, "b"),), f"{FIRST}/b.jpg"),
        (_second_folder_gone, ((SECOND, "a"), (SECOND, "c")), SECOND),
        (_b_undocumented, ((FIRST, "b"),), f"{FIRST}/b.jpg"),
    ],
    ids=["not_on_disk", "absent_folder", "no_document"],
)
def test_an_image_the_delivery_did_not_read_is_disclosed_rather_than_demanded(
    tmp_path: Path,
    publish: Callable[[Path, Path], list[str]],
    left_out: tuple[tuple[str, str], ...],
    disclosed: str,
) -> None:
    """A delivery leaving out each image it could not read succeeds and discloses the absence.

    The image is off disk, its date's folder is gone, or its bucket holds no document for it;
    the remaining results cover every read image of the delivered plants, and the event's
    disclosure names what was not read.
    """
    from tcip_mcp.delivery import read_delivery_events

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    arguments = _arguments_over(
        tmp_path,
        dataset_root,
        publish(tmp_path, dataset_root),
        _but(*left_out),
    )

    answer = _delivered(tmp_path, arguments)

    assert isinstance(answer, dict) and "error" not in answer, answer
    (event,) = read_delivery_events(tmp_path)
    assert event.plant_mapping is not None
    assert disclosed in event.plant_mapping.captures_unverified
    assert set(_by_plant(tmp_path)) == {"P1", "P2"}


def test_the_verifications_read_images_and_its_disclosed_ones_partition_the_documented_ones(
    tmp_path: Path,
) -> None:
    """Each date's read images and its disclosed ones split its attributed, documented images.

    On the first date one documented image is off disk and another has no document; the second
    date's folder is gone. Per date, the attributed images its bucket documents are exactly the
    verification's read stems and the disclosed ones among them, never both; a bare date
    disclosed stands for each of its images.
    """
    from tcip_mcp.buckets import by_recorded_date, read_bucket
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        assignment_is_attributed,
        plant_mapping_disclosure,
    )

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    names = _published(tmp_path, dataset_root, frozenset({(FIRST, "c")}))
    buckets = by_recorded_date(read_bucket(dataset_root, n) for n in names)
    (image_dir(dataset_root, FIRST) / "b.jpg").unlink()
    shutil.rmtree(image_dir(dataset_root, SECOND))

    resolved = plant_mapping_disclosure(
        tmp_path, "valley", buckets, ["P1", "P2"]
    )

    disclosed = resolved.disclosure.captures_unverified
    for date, rows in resolved.build.assignments.items():
        documented = {
            a.stem
            for a in rows
            if assignment_is_attributed(a)
            and a.stem in buckets[date].documents
        }
        unverified = (
            documented
            if date in disclosed
            else {
                Path(entry).stem
                for entry in disclosed
                if entry.startswith(f"{date}/")
            }
            & documented
        )
        read = resolved.verified.read_stems[date]
        assert read | unverified == documented, date
        assert not read & unverified, date
    assert resolved.verified.read_stems == {
        FIRST: frozenset({"a"}),
        SECOND: frozenset(),
    }


def test_an_image_yielding_no_value_counts_as_an_image_and_not_toward_the_statistic(
    tmp_path: Path,
) -> None:
    """A ``None`` value counts toward its plant's ``n_images`` and never enters its median."""
    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    rows = _rows(*READ)
    rows[1]["value"] = None

    answer = _delivered(
        tmp_path,
        _arguments_over(
            tmp_path, dataset_root, _both(tmp_path, dataset_root), rows
        ),
    )

    assert isinstance(answer, dict) and "error" not in answer, answer
    p1 = _by_plant(tmp_path)["P1"]
    assert (p1["n_images"], float(p1["value"])) == ("3", 8.5)


def test_a_sub_population_leaving_out_a_plant_with_no_value_delivers_without_its_images(
    tmp_path: Path,
) -> None:
    """A plant whose every image yields no value refuses; delivering without it succeeds.

    The sub-population's delivery neither needs nor takes the left-out plant's images.
    """
    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    buckets = _both(tmp_path, dataset_root)
    p2_images = {(FIRST, "c"), (SECOND, "a")}
    unobserved = _rows(
        *((d, s, None if (d, s) in p2_images else v) for d, s, v in READ)
    )

    refused = _called(
        tmp_path, _arguments_over(tmp_path, dataset_root, buckets, unobserved)
    )
    answer = _delivered(
        tmp_path,
        _arguments_over(
            tmp_path, dataset_root, buckets, _but(*p2_images), ("P1",)
        ),
    )

    assert isinstance(refused, dict), refused
    assert "carry no 'stem_count' observation" in refused["error"], refused
    assert isinstance(answer, dict) and "error" not in answer, answer
    assert list(_by_plant(tmp_path)) == ["P1"]


def test_no_parameter_of_the_aggregate_or_the_door_names_a_results_plant() -> (
    None
):
    """A result names its image and value, never a plant, source or distance.

    The door's result model holds exactly ``date``, ``stem`` and ``value``, and the per-image
    aggregate takes the resolved mapping in place of any caller-stated plant or disclosure.
    """
    import inspect
    from typing import get_args, get_type_hints

    from tcip_mcp.pipelines.postprocessing.aggregation import (
        deliver_per_plant_aggregate,
    )
    from tcip_mcp.tools.delivery_tools import deliver_per_plant_csv

    (result,) = get_args(get_type_hints(deliver_per_plant_csv)["results"])
    parameters = inspect.signature(deliver_per_plant_aggregate).parameters

    assert set(result.model_fields) == {"date", "stem", "value"}
    assert {"results", "mapping"} <= set(parameters)


def _arguments(rows: list[dict]) -> dict:
    """Return the door's arguments over ``rows``, naming no ``plant_mapping``."""
    return {
        "results": rows,
        "output_path": "o.csv",
        "delivered_phenotype": "stem_count",
        "delivery_kind": KIND,
        "plants": ["P1"],
        "dataset_root": "ds",
        "buckets": ["manual/2026-02-11"],
        "value_key": "count",
    }


def _called_with(rows: list[dict]) -> str:
    """Call ``deliver_per_plant_csv`` through the MCP server over ``rows``; return its answer.

    The server is started for no project, so a call its argument schema admits is refused for
    that instead.
    """
    from tests._server_calls import called_through_the_server

    return str(
        called_through_the_server(
            "deliver_per_plant_csv",
            {**_arguments(rows), "plant_mapping": "valley"},
        )
    )


ROW = {"date": DATE, "stem": "P1", "value": 2}
"""A row naming one image and its value."""


def test_the_door_refuses_a_call_naming_no_plant_mapping_at_its_schema() -> (
    None
):
    """A call naming no ``plant_mapping`` is refused by the door's own schema, by the name.

    The refusal is the argument schema's, before any delivery runs.
    """
    from tests._server_calls import called_through_the_server

    refused = called_through_the_server(
        "deliver_per_plant_csv", _arguments([ROW])
    )

    assert "validation error" in str(refused)
    assert "plant_mapping" in str(refused)


def test_the_door_offers_the_plant_aggregate_kinds_and_refuses_the_mosaic_count() -> (
    None
):
    """The door's schema offers the kinds a per-plant statistic reduces, and no other.

    The schema the server lists names the three aggregate kinds; a call naming the orthomosaic
    count's kind is refused by that schema, by the parameter's name, before any delivery runs.
    """
    from tests._server_calls import called_through_the_server, client_server

    tools = {t.name: t for t in client_server(None)._tool_manager.list_tools()}
    refused = called_through_the_server(
        "deliver_per_plant_csv",
        {
            **_arguments([ROW]),
            "delivery_kind": "per_plant_orthomosaic_count",
            "plant_mapping": "valley",
        },
    )

    assert "validation error" in str(refused), refused
    assert "delivery_kind" in str(refused), refused
    offered = tools["deliver_per_plant_csv"].parameters["properties"]
    assert offered["delivery_kind"].get("enum") == [
        "per_plant_count_aggregate",
        "per_plant_ordinal_aggregate",
        "per_plant_regression_aggregate",
    ]


EXTRA = "Extra inputs are not permitted"
"""The schema's sentence for a key its record does not hold."""


@pytest.mark.parametrize(
    ("row", "named", "reason"),
    [
        ({"stem": "P1", "value": 2}, "date", "Field required"),
        ({"date": DATE, "value": 2}, "stem", "Field required"),
        ({"date": DATE, "stem": "P1"}, "value", "Field required"),
        ({**ROW, "stem": ""}, "stem", "at least 1 character"),
        ({**ROW, "count": 2}, "count", EXTRA),
    ],
)
def test_the_door_refuses_a_row_its_record_does_not_admit_by_the_key(
    row: dict, named: str, reason: str
) -> None:
    """A row missing its date, stem or value, or carrying an unknown key, is refused by the key.

    The refusal is the door's own schema's, naming the row's place, the key and why, before any
    delivery runs.
    """
    refused = _called_with([row])

    assert "validation error" in refused
    assert f"results.0.{named}\n  " in refused, refused
    assert reason in refused.split(f"results.0.{named}\n  ")[1].splitlines()[0]


@pytest.mark.parametrize("key", ["plant_id", "source", "distance_m"])
def test_the_door_refuses_a_row_stating_its_plant_or_its_assignment(
    key: str,
) -> None:
    """A row carrying a plant, a source or a distance is refused as an extra key.

    The mapping states each image's plant and assignment; the door's schema admits no row
    stating its own.
    """
    refused = _called_with([{**ROW, key: "P1" if key == "plant_id" else 1.0}])

    assert "validation error" in refused
    assert f"results.0.{key}\n  {EXTRA}" in refused, refused


def test_the_door_admits_a_row_its_record_states() -> None:
    """A row naming its image and value, or its image yielding no value, passes the schema."""
    answered = _called_with([ROW, {**ROW, "stem": "P2", "value": None}])

    assert "validation error" not in answered


# ── a build asked for a date the dataset lists no capture of ──

ABSENT = "2026-03-01"
"""A date with no capture folder in :data:`CAPTURES`' scene."""
ALSO_ABSENT = "2026-05-01"
"""A second date with no capture folder in :data:`CAPTURES`' scene."""
EMPTY = "2026-04-01"
"""A date whose capture folder the tests make and leave holding no image."""
INGEST_FIRST = "ingest the missing dates' images first"
"""The step of the build's refusal of a date the dataset lists no capture of."""


@pytest.mark.parametrize(
    "dates",
    [[FIRST, ABSENT], [ABSENT, FIRST, ALSO_ABSENT], [ABSENT]],
    ids=["one_absent_beside_a_listed", "two_absent", "absent_alone"],
)
def test_a_build_asked_for_a_date_the_dataset_lists_no_capture_of_refuses_naming_it(
    tmp_path: Path, dates: list[str]
) -> None:
    """A build asked for a date with no capture folder refuses, storing and recording nothing.

    The sentence names each such date, the captures the dataset lists and the step; no mapping is
    stored under the name and the project's audit log gains no line.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping
    from tests._audit_fixtures import audit_rows

    dataset_root, registry = _registered(tmp_path, TWO_PLANTS, CAPTURES)
    logged = audit_rows(tmp_path)

    refused = _build(tmp_path, dataset_root, registry, dates)

    absent = [d for d in dates if d not in (FIRST, SECOND)]
    assert list(refused) == ["error"], refused
    assert str(absent) in refused["error"], refused
    assert str([FIRST, SECOND]) in refused["error"], refused
    assert INGEST_FIRST in refused["error"], refused
    assert load_mapping(tmp_path, "valley") is None
    assert audit_rows(tmp_path) == logged


def test_a_build_asked_for_a_name_the_lister_does_not_list_refuses_naming_it(
    tmp_path: Path,
) -> None:
    """A dot-prefixed folder is no capture the dataset lists, so a build asked for it refuses.

    The folder exists under the image tree; the refusal names it and offers the listed captures.
    """
    dataset_root, registry = _registered(tmp_path, TWO_PLANTS, CAPTURES)
    image_dir(dataset_root, ".2026-03-01").mkdir(parents=True)

    refused = _build(tmp_path, dataset_root, registry, [FIRST, ".2026-03-01"])

    assert str([".2026-03-01"]) in refused["error"], refused
    assert str([FIRST, SECOND]) in refused["error"], refused


def test_a_build_asked_for_dates_holding_no_image_names_the_captures_it_scanned(
    tmp_path: Path,
) -> None:
    """A build over only dates whose folders hold no image refuses naming those captures.

    The refusal no longer says "the requested dates".
    """
    dataset_root, registry = _registered(tmp_path, TWO_PLANTS, CAPTURES)
    image_dir(dataset_root, EMPTY).mkdir(parents=True)

    refused = _build(tmp_path, dataset_root, registry, [EMPTY])

    assert "no capture under" in refused["error"], refused
    assert "the requested dates" not in refused["error"], refused
    assert str([EMPTY]) in refused["error"], refused


@pytest.mark.parametrize(
    ("dates", "covered"),
    [
        ([FIRST], [FIRST]),
        ([FIRST, SECOND], [FIRST, SECOND]),
        (None, [FIRST, SECOND, EMPTY]),
        ([FIRST, EMPTY], [FIRST, EMPTY]),
    ],
    ids=[
        "a_listed_date",
        "listed_dates",
        "no_dates_stated",
        "beside_an_empty_folder",
    ],
)
def test_a_build_over_dates_the_dataset_lists_covers_them(
    tmp_path: Path, dates: list[str] | None, covered: list[str]
) -> None:
    """A build over listed dates, or over none stated, succeeds covering exactly those dates.

    A listed folder holding no image is covered, served with no images.
    """
    dataset_root, registry = _registered(tmp_path, TWO_PLANTS, CAPTURES)
    image_dir(dataset_root, EMPTY).mkdir(parents=True)

    built = _build(tmp_path, dataset_root, registry, dates)

    assert "error" not in built, built
    assert list(built["per_date"]) == covered
    assert built["per_date"].get(EMPTY, {"n_images": 0})["n_images"] == 0


def test_the_next_step_of_a_result_on_a_date_with_no_capture_leads_to_a_delivery(
    tmp_path: Path,
) -> None:
    """Following the door's step for a date with no capture folder reaches a delivery.

    The result on the date is refused with the step; a rebuild over the date is refused naming it;
    once the date's capture is written, the mapping rebuilt and the date's bucket published, the
    delivery with every read image's result succeeds.
    """
    from tcip_mcp.delivery import read_delivery_events
    from tcip_mcp.pipelines.postprocessing.plant_mapping import load_mapping
    from tests._mapping_fixtures import PHOTO_POSITION_ERROR_M, write_photos

    dataset_root, _ = _mapped(tmp_path, TWO_PLANTS, CAPTURES)
    buckets = _both(tmp_path, dataset_root)
    rows = _rows(*READ, (ABSENT, "x", 32))
    refused = _called(
        tmp_path, _arguments_over(tmp_path, dataset_root, buckets, rows)
    )

    assert isinstance(refused, dict), refused
    assert _named_under(
        refused["error"], ((NOT_COVERED, NOT_COVERED_STEP),)
    ) == {NOT_COVERED: [f"{ABSENT}/x"]}
    built = load_mapping(tmp_path, "valley")
    assert built is not None
    rebuilt = _build(
        tmp_path,
        dataset_root,
        built.plant_registry.name,
        [FIRST, SECOND, ABSENT],
    )
    assert ABSENT in rebuilt["error"], rebuilt
    write_photos(
        image_dir(dataset_root, ABSENT),
        {"x.jpg": (*TWO_PLANTS["P1"], PHOTO_POSITION_ERROR_M)},
        datetime(2026, 3, 1, 9, 30),
        timedelta(minutes=5),
    )
    rebuilt = _build(tmp_path, dataset_root, built.plant_registry.name, None)
    assert "error" not in rebuilt, rebuilt
    buckets.append(_bucket(tmp_path, dataset_root, ABSENT, {"x": 1}))
    answer = _delivered(
        tmp_path, _arguments_over(tmp_path, dataset_root, buckets, rows)
    )

    assert isinstance(answer, dict) and "error" not in answer, answer
    assert len(read_delivery_events(tmp_path)) == 1

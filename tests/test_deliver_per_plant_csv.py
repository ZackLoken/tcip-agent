"""Tests for the general per-plant CSV door.

Per-image results, each naming its image by capture date and stem, are attributed to plants by a
mapping built by ``build_plant_mapping``, aggregated, and delivered over published buckets.
"""

from __future__ import annotations

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
    from tcip_mcp.tools.project_tools import (
        initialize_project,
        register_dataset,
    )
    from tests._mapping_fixtures import PHOTO_POSITION_ERROR_M, write_photos
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
    result = register_dataset(
        tmp_path, str(dataset_root), crop=COUNT_CROP, actor=None
    )
    assert "error" not in result, result
    plant_csv = write_plots_csv(tmp_path / PLANT_CSV, plants)
    return dataset_root, plant_csv


DATE = "2026-02-11"


def _mapped(
    tmp_path: Path,
    plants: Mapping[str, tuple[float, float]] = ONE_PLANT,
    captures: Mapping[str, Mapping[str, tuple[float, float]]] | None = None,
) -> tuple[Path, str]:
    """Build :func:`_scene` with the mapping ``valley`` over it; return ``(dataset root, date)``.

    The captures are ``captures``, by default one image per plant on :data:`DATE`, each named by
    its plant and taken at its place; the date returned is the first.
    """
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._mapping_fixtures import register_plant_registry_for

    taken = captures or {DATE: dict(plants)}
    dataset_root, plant_csv = _scene(tmp_path, plants, taken)
    mapped = build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry=register_plant_registry_for(
            tmp_path, [plant_csv], crop=COUNT_CROP
        ),
        actor=None,
    )
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

    The bucket lies under ``dataset_root`` and holds ``counts[stem]`` detections on each image
    ``images/<date>/<stem>.jpg``.
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
    return published(project, f"{name}/{date}", results, scope=SCOPE).name


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
    assert not (tmp_path / "o.csv").exists()


def test_predictions_under_another_dataset_than_the_mappings_refuse(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.traits import registered_crops
    from tests._image_fixtures import write_geo_image

    _dataset_root, date = _mapped(tmp_path)
    other = tmp_path / "ds2"
    write_geo_image(
        image_dir(other, date) / "P1.jpg",
        *ONE_PLANT["P1"],
        datetime(2026, 2, 11, 9, 30),
    )
    assert "error" not in register_dataset(
        tmp_path, str(other), crop=sorted(registered_crops())[0], actor=None
    )

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
    project: Path, dataset_root: Path, without: frozenset = frozenset()
) -> list[str]:
    """Publish one bucket per date of :data:`CAPTURES`; return their names.

    Each holds a document for every image of its date but the ``(date, stem)`` pairs ``without``
    names.
    """
    return [
        _bucket(
            project,
            dataset_root,
            date,
            {stem: 1 for stem in taken if (date, stem) not in without},
        )
        for date, taken in CAPTURES.items()
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


def _p2_unread(project: Path, dataset_root: Path) -> list[str]:
    """Publish both dates, the first's ``c`` undocumented, then remove the second's folder.

    The delivery reads no image of ``P2``: its first-date image has no document and its
    second-date image is gone with its folder.
    """
    buckets = _published(project, dataset_root, frozenset({(FIRST, "c")}))
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
            "is not a delivered date",
            f"image 'a' of {SECOND}",
        ),
        (
            _both,
            _rows(*READ, (SECOND, "b", 32)),
            ("P1", "P2"),
            "records no image of that stem on that date",
            f"image 'b' of {SECOND}",
        ),
        (
            _both,
            _rows(*READ, (FIRST, "far", 32)),
            ("P1", "P2"),
            "assigned it to no plant",
            f"image 'far' of {FIRST}",
        ),
        (
            _both,
            _but((SECOND, "a")),
            ("P1",),
            "which the population ['P1'] does not name",
            f"image 'c' of {FIRST}",
        ),
        (
            _b_undocumented,
            _rows(*READ),
            ("P1", "P2"),
            "was not read by the delivery",
            f"image 'b' of {FIRST}",
        ),
        (
            _b_off_disk,
            _rows(*READ),
            ("P1", "P2"),
            "was not read by the delivery",
            f"image 'b' of {FIRST}",
        ),
        (
            _second_folder_gone,
            _rows(*READ),
            ("P1", "P2"),
            "was not read by the delivery",
            f"image 'a' of {SECOND}",
        ),
        (
            _both,
            _rows(*READ, READ[0]),
            ("P1", "P2"),
            "is named by two results",
            f"image 'a' of {FIRST}",
        ),
        (
            _both,
            _but((FIRST, "b")),
            ("P1", "P2"),
            UNCOVERED,
            f"'{FIRST}/b'",
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

    The refusal names the image and its reason, through the server's door, and no delivery
    event or CSV is written. A delivered plant the delivery read no image of refuses naming it
    and its images, ahead of the read images no result names.
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
    assert read_delivery_events(tmp_path) == []
    assert not (tmp_path / "o.csv").exists()


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

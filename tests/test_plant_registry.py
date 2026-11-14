"""Tests of ``register_plant_registry``.

The registry is a named, project-scoped record of a plant-locations CSV set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tcip_mcp.tools.phenology_tools import (
    build_plant_mapping,
    register_plant_registry,
)
from tests._mapping_fixtures import PLANTS, write_plant_csv
from tests.test_plant_mapping_binding import (
    POPULATION,
    _dataset,
    _init,
    _write_scene,
)


def test_registers_a_csv_and_persists_the_frozen_record(
    tmp_path: Path,
) -> None:
    """Registering a CSV persists the frozen registry record.

    The recorded shape: per-file {path, sha256, n_plants}, crop, site, registered_at, and a digest
    over the parsed rows; who registered it is the audit line's.
    """
    csv_path = write_plant_csv(tmp_path / "plants.csv", PLANTS)

    res = register_plant_registry(
        tmp_path,
        name="valley-plants",
        csv_paths=[str(csv_path)],
        crop="currant",
        site="north orchard",
        actor=None,
    )

    assert "error" not in res, res
    assert res["name"] == "valley-plants"
    assert res["n_plants"] == len(PLANTS)
    assert "registered_by" not in res
    assert len(res["digest"]) == 64
    from tcip_mcp.registry_paths import stored_path

    assert res["csvs"] == [
        {
            "path": stored_path(csv_path, tmp_path),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "n_plants": len(PLANTS),
        }
    ]

    stored = plant_mapping.load_registry(tmp_path, "valley-plants")
    assert stored is not None
    assert stored["digest"] == res["digest"]
    assert stored["csvs"] == res["csvs"]


def test_a_registered_csvs_hash_is_of_the_bytes_its_rows_were_parsed_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered CSV's hash and count are of the bytes its rows were parsed from.

    The file is rewritten with more plants between the parse and anything after it.
    """
    csv_path = write_plant_csv(tmp_path / "plants.csv", PLANTS[:1])
    parsed_bytes = csv_path.read_bytes()
    real = plant_mapping.read_plant_csv_bytes

    def parse_then_rewrite(data: bytes) -> list[plant_mapping.PlantRecord]:
        rows = real(data)
        write_plant_csv(csv_path, PLANTS)
        return rows

    monkeypatch.setattr(
        plant_mapping, "read_plant_csv_bytes", parse_then_rewrite
    )
    csvs, _digest, n_plants = plant_mapping.parse_plant_registry_csvs(
        [csv_path], tmp_path
    )

    assert n_plants == csvs[0]["n_plants"] == 1
    assert csvs[0]["sha256"] == hashlib.sha256(parsed_bytes).hexdigest()


def test_refuses_a_name_outside_name_segment(tmp_path: Path) -> None:
    csv_path = write_plant_csv(tmp_path / "plants.csv", PLANTS)

    res = register_plant_registry(
        tmp_path,
        name="Not Legal!",
        csv_paths=[str(csv_path)],
        crop="currant",
        site="orchard",
        actor=None,
    )

    match res:
        case {"error": str() as error}:
            assert "lowercase" in error
        case _:
            raise AssertionError(res)


def test_refuses_a_missing_file(tmp_path: Path) -> None:
    res = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(tmp_path / "missing.csv")],
        crop="currant",
        site="orchard",
        actor=None,
    )

    match res:
        case {"error": str() as error}:
            assert (
                error == f"plant CSV(s) not found: {tmp_path / 'missing.csv'}"
            )
        case _:
            raise AssertionError(res)


def test_a_missing_file_is_named_by_its_path_with_each_backslash_once(
    tmp_path: Path,
) -> None:
    """A missing file whose name holds a backslash is named by the path as resolved.

    The name carries the backslash itself, so the path holds one on every host.
    """
    from tcip_mcp.registry_paths import located

    name = "plants\\missing.csv"

    res = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[name],
        crop="currant",
        site="orchard",
        actor=None,
    )

    assert res == {
        "error": f"plant CSV(s) not found: {located(name, tmp_path)}"
    }


def test_refuses_naming_a_file_that_parses_no_georeferenced_plant(
    tmp_path: Path,
) -> None:
    """A file parsing to no plant refuses beside a good one, naming only the failing file."""
    good = write_plant_csv(tmp_path / "good.csv", PLANTS)
    bad = write_plant_csv(tmp_path / "bad.csv", [])

    res = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(good), str(bad)],
        crop="currant",
        site="orchard",
        actor=None,
    )

    assert res == {"error": _NO_PLANT_SENTENCE.format(file=_located(bad))}
    assert not ts.exists(plant_mapping.plant_registry_key(tmp_path, "valley"))


_NO_PLANT_SENTENCE = (
    "{file} parsed no georeferenced plant, or is not UTF-8 text (need columns plant_id, "
    "WGS84_centroid_x, WGS84_centroid_y with usable values); register only files that carry "
    "at least one."
)

_NO_ID_SENTENCE = (
    "{file}: {unnamed} of {parsed} parsed plants have no id; an id is read from the column "
    "'plant_id', and every plant needs one."
)


def _located(path: Path) -> Path:
    from tcip_mcp.registry_paths import located

    return located(path, path.parent)


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _register(tmp_path: Path, *files: Path) -> dict:
    return register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(f) for f in files],
        crop="currant",
        site="orchard",
        actor=None,
    )


def _no_id_column_file(path: Path) -> Path:
    return _write_text(
        path,
        "WGS84_centroid_x,WGS84_centroid_y,accession_name,plot_name,row_number,col_number\n"
        "-90.0580,43.1967,acc-1,P1,1,1\n"
        "-90.0581,43.1968,acc-2,P1,1,2\n"
        "-90.0582,43.1969,acc-3,P2,2,1\n",
    )


def test_refuses_a_file_with_no_id_column_naming_the_file_the_counts_and_the_column(
    tmp_path: Path,
) -> None:
    """A file with no ``plant_id`` column is refused whole at registration; nothing registers."""
    plants = _no_id_column_file(tmp_path / "plants.csv")

    res = _register(tmp_path, plants)

    assert res == {
        "error": _NO_ID_SENTENCE.format(
            file=_located(plants), unnamed=3, parsed=3
        )
    }
    assert not ts.exists(plant_mapping.plant_registry_key(tmp_path, "valley"))


def test_the_count_of_plants_without_an_id_is_of_parsed_plants(
    tmp_path: Path,
) -> None:
    """A blank cell and a short row both count as no id; a row with no coordinates is not a plant.

    The id column is last, so a short row keeps its coordinates and loses only its id.
    """
    plants = _write_text(
        tmp_path / "plants.csv",
        "WGS84_centroid_x,WGS84_centroid_y,accession_name,plant_id\n"
        "-90.0580,43.1967,acc-1,A1\n"
        "-90.0581,43.1968,acc-2,\n"
        "-90.0582,43.1969,acc-3\n"
        ",,acc-4,\n",
    )

    res = _register(tmp_path, plants)

    assert res == {
        "error": _NO_ID_SENTENCE.format(
            file=_located(plants), unnamed=2, parsed=3
        )
    }


def test_each_file_without_an_id_is_refused_in_its_own_sentence_and_nothing_registers(
    tmp_path: Path,
) -> None:
    """Two failing files beside a whole one: a sentence each, with its own counts."""
    good = write_plant_csv(tmp_path / "good.csv", PLANTS)
    no_column = _no_id_column_file(tmp_path / "no_column.csv")
    one_blank = _write_text(
        tmp_path / "one_blank.csv",
        "plant_id,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "A1,acc-1,-90.0580,43.1967\n"
        ",acc-2,-90.0581,43.1968\n",
    )

    res = _register(tmp_path, good, no_column, one_blank)

    assert res == {
        "error": " ".join(
            [
                _NO_ID_SENTENCE.format(
                    file=_located(no_column), unnamed=3, parsed=3
                ),
                _NO_ID_SENTENCE.format(
                    file=_located(one_blank), unnamed=1, parsed=2
                ),
            ]
        )
    }
    assert not ts.exists(plant_mapping.plant_registry_key(tmp_path, "valley"))


def test_a_file_parsing_no_plant_beside_a_file_without_ids_is_refused_in_a_sentence_each(
    tmp_path: Path,
) -> None:
    """A call with both kinds of failing file names every one, in file order."""
    empty = write_plant_csv(tmp_path / "empty.csv", [])
    no_column = _no_id_column_file(tmp_path / "no_column.csv")

    res = _register(tmp_path, no_column, empty)

    assert res == {
        "error": " ".join(
            [
                _NO_ID_SENTENCE.format(
                    file=_located(no_column), unnamed=3, parsed=3
                ),
                _NO_PLANT_SENTENCE.format(file=_located(empty)),
            ]
        )
    }
    assert not ts.exists(plant_mapping.plant_registry_key(tmp_path, "valley"))


def test_registers_several_files_whose_plants_all_have_an_id(
    tmp_path: Path,
) -> None:
    """Several whole files register together as one registry (a legitimate call)."""
    first = write_plant_csv(tmp_path / "first.csv", PLANTS[:1])
    second = write_plant_csv(tmp_path / "second.csv", PLANTS[1:])

    res = _register(tmp_path, first, second)

    assert "error" not in res, res
    assert res["n_plants"] == len(PLANTS)
    assert [c["n_plants"] for c in res["csvs"]] == [1, 1]


def test_a_second_registration_under_the_same_name_and_content_is_a_no_op(
    tmp_path: Path,
) -> None:
    csv_path = write_plant_csv(tmp_path / "plants.csv", PLANTS)

    first = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(csv_path)],
        crop="currant",
        site="orchard",
        actor=None,
    )
    assert "error" not in first, first

    second = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(csv_path)],
        crop="currant",
        site="orchard",
        actor=None,
    )

    assert "error" not in second, second
    assert second["digest"] == first["digest"]


def test_a_second_registration_under_the_same_name_and_different_plants_refuses(
    tmp_path: Path,
) -> None:
    csv_path = write_plant_csv(tmp_path / "plants.csv", PLANTS)
    other_csv = write_plant_csv(
        tmp_path / "other.csv",
        [{"plot": "P9", "accession": "acc-Z", "lat": 1.0, "lon": 1.0}],
    )

    first = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(csv_path)],
        crop="currant",
        site="orchard",
        actor=None,
    )
    assert "error" not in first, first

    conflict = register_plant_registry(
        tmp_path,
        name="valley",
        csv_paths=[str(other_csv)],
        crop="currant",
        site="orchard",
        actor=None,
    )

    match conflict:
        case {"error": str() as error}:
            assert first["digest"] in error
        case _:
            raise AssertionError(conflict)
    stored = plant_mapping.load_registry(tmp_path, "valley")
    assert stored["digest"] == first["digest"], (
        "the taken name's record must not have moved"
    )


def test_build_plant_mapping_reads_a_registered_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_plant_mapping builds a mapping over a name register_plant_registry minted."""
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    plant_csv, _ = _write_scene(dataset_root, dates=["2026-02-11"])

    reg = register_plant_registry(
        tmp_path,
        name="valley-plants",
        csv_paths=[str(plant_csv)],
        crop="currant",
        site="orchard",
        actor=None,
    )
    assert "error" not in reg, reg

    res = build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry="valley-plants",
        actor=None,
    )

    assert "error" not in res, res
    build = plant_mapping.load_mapping(tmp_path, "valley")
    assert build is not None
    assert build.plant_registry == plant_mapping.PlantRegistryReference(
        name="valley-plants", digest=reg["digest"]
    )


def test_build_plant_mapping_refuses_naming_register_plant_registry_when_registry_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    _write_scene(dataset_root, dates=["2026-02-11"])

    res = build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry="nope",
        actor=None,
    )

    assert "error" in res
    assert "register_plant_registry" in res["error"]


def _deliver_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, dict[str, str]]:
    """Build a one-date phenology scene; return its plant registry's name and its buckets by date.

    The registry ``valley-plants`` and the mapping ``valley`` are made through the platform's own
    producers, and the second trait is seeded.
    """
    from tests.test_second_trait_acceptance import _seed_currant_bloom_trait

    _init(tmp_path)
    dataset_root = _dataset(tmp_path)
    plant_csv, preds_by_date = _write_scene(dataset_root, dates=["2026-02-11"])
    reg = register_plant_registry(
        tmp_path,
        name="valley-plants",
        csv_paths=[str(plant_csv)],
        crop="currant",
        site="orchard",
        actor=None,
    )
    assert "error" not in reg, reg
    build_res = build_plant_mapping(
        tmp_path,
        name="valley",
        dataset_root=str(dataset_root),
        plant_registry="valley-plants",
        actor=None,
    )
    assert "error" not in build_res, build_res
    _seed_currant_bloom_trait(tmp_path)
    return "valley-plants", preds_by_date


def test_the_happy_path_through_the_platforms_own_producers_refuses_at_the_classifier_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivery over an intact registry refuses at the assessment gate, not at the registry.

    With the scene's buckets unassessed, the delivery's error says no assessment answers and does
    not name the registry.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    registry_name, preds_by_date = _deliver_scene(tmp_path, monkeypatch)
    out_csv = tmp_path / "out.csv"

    res = deliver_phenology_milestones(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=str(tmp_path / "ds"),
        buckets=list(preds_by_date.values()),
        output_csv_path=str(out_csv),
        actor=None,
    )

    assert "error" in res, res
    assert "no assessment answers" in res["error"]
    assert registry_name not in res["error"]


def test_a_deleted_registry_refuses_at_delivery_naming_the_registry_and_the_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry deleted after the mapping was built refuses the delivery by name.

    The refusal names the registry and the mapping, and no CSV is written.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    registry_name, preds_by_date = _deliver_scene(tmp_path, monkeypatch)
    key = plant_mapping.plant_registry_key(tmp_path, registry_name)
    ts.delete(key, expect=ts.read_versioned(key).version)
    out_csv = tmp_path / "out.csv"

    res = deliver_phenology_milestones(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=str(tmp_path / "ds"),
        buckets=list(preds_by_date.values()),
        output_csv_path=str(out_csv),
        actor=None,
    )

    assert "error" in res
    assert registry_name in res["error"]
    assert "valley" in res["error"]
    assert not out_csv.exists()


def test_a_registry_digest_mismatch_refuses_at_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry whose stored digest does not match the mapping's refuses at delivery.

    Such a registry (a hand-edited or store-corrupted record) refuses naming itself, and no CSV is
    written.
    """
    from tcip_mcp.tools.phenology_tools import deliver_phenology_milestones

    registry_name, preds_by_date = _deliver_scene(tmp_path, monkeypatch)
    key = plant_mapping.plant_registry_key(tmp_path, registry_name)
    versioned = ts.read_versioned(key)
    ts.replace(
        key, {**versioned.value, "digest": "0" * 64}, expect=versioned.version
    )
    out_csv = tmp_path / "out.csv"

    res = deliver_phenology_milestones(
        tmp_path,
        trait="currant_bloom",
        mapping_name="valley",
        plants=POPULATION,
        dataset_root=str(tmp_path / "ds"),
        buckets=list(preds_by_date.values()),
        output_csv_path=str(out_csv),
        actor=None,
    )

    assert "error" in res
    assert registry_name in res["error"]
    assert not out_csv.exists()

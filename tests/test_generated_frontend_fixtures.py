"""The browser's delivery-event fixtures are what the platform's own deliveries record and serve.

``tools/generate_delivery_fixture.py`` makes real deliveries in scratch projects and writes the
records ``GET /api/results/delivery-events`` serves for them to
``frontend/src/test/deliveryEvents.json``; this test regenerates them and holds the checked-in
file to what a regeneration serves, each value compared but the ones a run mints. The trait
listings ``tools/generate_trait_fixture.py`` writes, the project fixtures
``tools/generate_project_fixture.py`` writes, the TensorBoard serving answers
``tools/generate_tensorboard_fixture.py`` writes, the open answer and state envelope
``tools/generate_open_fixture.py`` writes and the training routes' answers
``tools/generate_training_fixture.py`` writes are held the same way, and the scratch
session the generators make their scenes in (``tools/fixture_session.py``) is held to leaving
nothing behind.
"""

from __future__ import annotations

import copy
import importlib
import json
from typing import TYPE_CHECKING, cast

import pytest

from tests._web_fixtures import named_project
from tools.fixture_session import normalized, opened, scratch_session

if TYPE_CHECKING:
    from types import ModuleType

    from pydantic import JsonValue


def _generator(name: str) -> ModuleType:
    """Import and return the fixture generator module ``tools.<name>``."""
    return importlib.import_module(f"tools.{name}")


GENERATED = _generator("generate_delivery_fixture").GENERATED_PATH


PER_RUN = frozenset(
    {
        "event_id",
        "acknowledgment_id",
        "assessment_id",
        "dataset_id",
        "experiment_id",
        "produced_at",
        "recorded_at",
        "built_at",
        "output_sha256",
        "result_sha256",
        "record_sha256",
        "checkpoint_sha256",
        "digest",
    }
)
"""The record fields two productions of the same scenes fill differently: minted ids, the
times things happened, and digests over what carries those or over a file the scene writes in
the platform's own line endings (a run's weights, a written output, a record holding a time, a
plant registry)."""


def _normalized(fixture: dict[str, dict]) -> dict[str, JsonValue]:
    """Return ``fixture`` with each value a run mints made what it stands for.

    A walked mapping's served disclosure (``ServedPlantMappingDisclosure``, each ``records``
    entry's ``plant_mapping`` read through the route's ``ServedDisclosure``) whose resolved key is
    the name :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.archived_mapping_name` gives
    the record it cites has that key read as ``<archived NAME>``; then each :data:`PER_RUN` string
    reads as a placeholder numbered by its first appearance across the fixture and each float is
    rounded (``fixture_session.normalized``). Other values are kept as served.
    """
    from pydantic import TypeAdapter

    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        archived_mapping_name,
    )
    from tcip_web.routes.results import (
        ServedDisclosure,
        ServedPlantMappingDisclosure,
    )

    disclosure = TypeAdapter[ServedDisclosure | None](ServedDisclosure | None)
    archived = copy.deepcopy(fixture)
    for record in archived["records"].values():
        pm = disclosure.validate_python(record["plant_mapping"])
        if isinstance(
            pm, ServedPlantMappingDisclosure
        ) and pm.resolved_key == archived_mapping_name(
            pm.name, pm.record_sha256
        ):
            record["plant_mapping"]["resolved_key"] = f"<archived {pm.name}>"
    return normalized(archived, numbered=PER_RUN, rounded=True)


def _changed(fixture: dict, name: str, *path_and_value: JsonValue) -> dict:
    """Return a copy of ``fixture`` with one value under record ``name`` replaced.

    The value at ``path`` under record ``name`` is set to the last argument.
    """
    *path, value = path_and_value
    out = copy.deepcopy(fixture)
    target = out["records"][name]
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return out


def test_the_comparison_keeps_every_value_a_run_does_not_mint() -> None:
    """The fixture comparison breaks on a changed served value and holds across a reminted id.

    A served value (a disclosure count, an output path) changing, a
    minted value changing its type or going absent, two records sharing a minted value no longer
    sharing it, and an archived mapping key that is not the archive's own name each break the
    comparison; a minted id changing its value does not.
    """
    checked = json.loads(GENERATED.read_text(encoding="utf-8"))
    records = checked["records"]
    count = records["canopy"]["plant_mapping"]["segments_without_plant"]
    digest = records["archived_mapping"]["plant_mapping"]["record_sha256"]
    broken = [
        _changed(
            checked,
            "canopy",
            "plant_mapping",
            "segments_without_plant",
            count + 99,
        ),
        _changed(
            checked, "validated", "output_path", "/scratch/elsewhere.csv"
        ),
        _changed(checked, "validated", "event_id", 7),
        _changed(checked, "validated", "output_sha256", 7),
        _changed(checked, "registry", "producer", "experiment_id", None),
        _changed(
            checked,
            "archived_mapping",
            "producer",
            "experiment_id",
            "another-run",
        ),
        *(
            _changed(
                checked,
                "archived_mapping",
                "plant_mapping",
                "resolved_key",
                key,
            )
            for key in ("valley@", "valley@3", f"valley@{digest}")
        ),
    ]

    for fixture in broken:
        assert _normalized(fixture) != _normalized(checked)
    reminted = _changed(checked, "validated", "event_id", "another-id")
    assert _normalized(reminted) == _normalized(checked)


def test_each_disclosure_normalizes_through_its_own_type() -> None:
    """Each delivery record's disclosure normalizes as its schema type states.

    A walked mapping's disclosure whose resolved key is its cited record's archive name reads as
    archived, one still resolving to its own name keeps it, and a registry or canopy disclosure,
    which carries no resolved key, gains none.
    """
    checked = json.loads(GENERATED.read_text(encoding="utf-8"))
    normal = cast("dict[str, dict]", _normalized(checked)["records"])

    assert normal["archived_mapping"]["plant_mapping"]["resolved_key"] == (
        "<archived valley>"
    )
    assert normal["acknowledged_mapping"]["plant_mapping"]["resolved_key"] == (
        "valley"
    )
    for name in ("registry", "canopy"):
        assert "resolved_key" not in normal[name]["plant_mapping"]


@pytest.mark.parametrize(
    "name",
    [
        "generate_trait_fixture",
        "generate_project_fixture",
        "generate_tensorboard_fixture",
        "generate_open_fixture",
        "generate_training_fixture",
    ],
)
def test_the_fixture_is_what_a_regeneration_serves_value_for_value(
    name: str,
) -> None:
    """A trait, project, TensorBoard, open or training fixture is what a regeneration serves.

    Each generator's checked-in file equals what its ``produce`` serves, value for value.
    """
    module = _generator(name)

    served = module.produce()

    assert (
        json.loads(module.GENERATED_PATH.read_text(encoding="utf-8")) == served
    ), f"{module.GENERATED_PATH} is out of date; run python -m tools.{name}"


def test_a_tensorboard_whose_stop_is_unconfirmed_fails_the_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TensorBoard fixture's production raises when a stop answers its kill unconfirmed.

    Each stop is the manager's own, its answer then reported unconfirmed.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    stop = tb.stop_tensorboard
    monkeypatch.setattr(
        tb,
        "stop_tensorboard",
        lambda logdir: {**stop(logdir), "status": "kill_unconfirmed"},
    )

    with pytest.raises(SystemExit, match="kills unconfirmed"):
        _generator("generate_tensorboard_fixture").produce()


def test_an_unconfirmed_stop_after_a_failed_production_is_refused_over_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production that fails after a launch still refuses its unconfirmed stop.

    The failure is the first answer's normalization raising; the stop is the manager's own, its
    answer reported unconfirmed; the refusal carries the failure as its context.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb

    class NormalizationFailedError(Exception):
        pass

    def failing(*_served: object, **_policy: object) -> dict:
        raise NormalizationFailedError

    module = _generator("generate_tensorboard_fixture")
    stop = tb.stop_tensorboard
    monkeypatch.setattr(
        tb,
        "stop_tensorboard",
        lambda logdir: {**stop(logdir), "status": "kill_unconfirmed"},
    )
    monkeypatch.setattr(module, "normalized", failing)

    with pytest.raises(SystemExit, match="kills unconfirmed") as refused:
        module.produce()
    assert isinstance(refused.value.__context__, NormalizationFailedError)


def test_a_wrong_typed_stamped_value_is_kept_for_the_staleness_comparison() -> (
    None
):
    """A value of another type than the producer writes under a stamped key is kept as served.

    Under the training fixture's own stamps, a ``None`` where a timestamp belongs and a string
    or a boolean where a pid belongs survive normalization, so a regeneration carrying them
    differs from the checked-in fixture, while the producer's own types read as their
    placeholders.
    """
    stamped = _generator("generate_training_fixture").STAMPED

    assert normalized({"created": None, "pid": "1"}, stamped=stamped) == {
        "created": None,
        "pid": "1",
    }
    assert normalized({"pid": True}, stamped=stamped) == {"pid": True}
    assert normalized(
        {"created": "2026-01-01T00:00:00+00:00", "pid": 7}, stamped=stamped
    ) == {"created": "<created>", "pid": 0}


def test_a_request_validation_refusal_is_never_recorded_as_the_routes_own() -> (
    None
):
    """The training fixture's refusal refuses a refusal the request model gave, not the route.

    A body the ranking route's request model refuses answers 422 listing its issues; recorded as
    the route's own refusal at that status, it raises instead.
    """
    from tcip_web.routes._body_common import EmptyBodyPayload

    refusal = _generator("generate_training_fixture").refusal
    with scratch_session() as (_workspace, client):
        with pytest.raises(AssertionError, match="Field required"):
            refusal(
                client,
                "/api/training/compare/best",
                EmptyBodyPayload(),
                422,
                lambda _detail: True,
            )


def test_the_delivery_fixture_is_what_a_regeneration_serves() -> None:
    """The delivery fixture is what a regeneration of its scenes serves.

    With both passed through :func:`_normalized`, the checked-in fixture equals what a
    regeneration of the scenes produces.
    """
    pytest.importorskip("torch")
    produced = _generator("generate_delivery_fixture").produce()
    checked = json.loads(GENERATED.read_text(encoding="utf-8"))
    assert _normalized(checked) == _normalized(produced), (
        "packages/tcip-web/frontend/src/test/deliveryEvents.json is out of date; "
        "run python -m tools.generate_delivery_fixture"
    )


def test_a_scratch_session_whose_body_raises_releases_store_and_directory() -> (
    None
):
    """A scratch session whose body raises releases its store connections and directory.

    The bound store is left holding no connection, the session's directory is removed, and the
    body's own exception propagates out of the session.
    """
    from tcip_store import store as seam

    class BodyFailedError(Exception):
        pass

    with pytest.raises(BodyFailedError):
        with scratch_session() as (workspace, _client):
            named_project(workspace / "valley", "valley")
            backend, scratch = seam._bound, workspace.parent
            assert backend is not None and backend._connections
            raise BodyFailedError

    assert backend._connections == {}
    assert not scratch.exists()


def test_a_scratch_session_holds_no_project_an_earlier_session_opened() -> (
    None
):
    """A scratch session holds no project an earlier session opened.

    A second scratch session in the same interpreter lists no open project, though the first
    left the project it opened open when its body ended.
    """
    with scratch_session() as (workspace, client):
        opened(client, named_project(workspace / "valley", "valley"))
    with scratch_session() as (_workspace, client):
        listing = client.get("/api/projects")

    assert listing.json()["open_id"] is None

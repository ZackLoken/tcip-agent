"""The browser's delivery-event fixtures are what the platform's own deliveries record and serve.

``tools/generate_delivery_fixture.py`` makes real deliveries in scratch projects and writes the
records ``GET /api/results/delivery-events`` serves for them to
``frontend/src/test/deliveryEvents.json``; this test regenerates them and holds the checked-in
file to what a regeneration serves, each value compared but the ones a run mints. The trait
listings ``tools/generate_trait_fixture.py`` writes, the project fixtures
``tools/generate_project_fixture.py`` writes and the TensorBoard serving answers
``tools/generate_tensorboard_fixture.py`` writes are held the same way, and the scratch session the
generators make their scenes in (``tools/fixture_session.py``) is held to leaving nothing
behind.
"""

from __future__ import annotations

import copy
import importlib
import json
from typing import TYPE_CHECKING

import pytest

from tests._web_fixtures import named_project
from tools.fixture_session import opened, replaced, scratch_session

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

FLOAT_PLACES = 6
"""The decimal places a served float is compared to: a geometric value's last bits differ
between the platforms the scenes run on, and a delivered meter value carries no more."""


def _normalized(fixture: dict[str, dict]) -> dict[str, dict]:
    """Return ``fixture`` with each minted value replaced by a numbered placeholder.

    Each :data:`PER_RUN` string is replaced by a placeholder numbered by its first appearance
    across the fixture; each float is rounded to
    :data:`FLOAT_PLACES`; and an archived mapping key equal to the name
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.archived_mapping_name` gives the
    cited record is replaced by that record's placeholder. Other values are kept as served.
    """
    from tcip_mcp.pipelines.postprocessing.plant_mapping import (
        archived_mapping_name,
    )

    seen: dict[str, str] = {}

    def leaf(key: str | None, value: JsonValue) -> JsonValue:
        if key in PER_RUN and isinstance(value, str):
            return seen.setdefault(value, f"<{key} {len(seen)}>")
        if isinstance(value, float):
            return round(value, FLOAT_PLACES)
        return value

    out: dict[str, dict] = {}
    for name, record in fixture.items():
        normalized = replaced(record, leaf)
        pm, resolved = (
            record["plant_mapping"],
            record.get("plant_mapping_resolved_key"),
        )
        if (
            pm is not None
            and resolved is not None
            and "record_sha256" in pm
            and resolved
            == archived_mapping_name(pm["name"], pm["record_sha256"])
        ):
            normalized["plant_mapping_resolved_key"] = (
                f"<archived {pm['name']} at {seen[pm['record_sha256']]}>"
            )
        out[name] = normalized
    return out


def _changed(fixture: dict, name: str, *path_and_value: JsonValue) -> dict:
    """Return a copy of ``fixture`` with one value under record ``name`` replaced.

    The value at ``path`` under record ``name`` is set to the last argument.
    """
    *path, value = path_and_value
    out = copy.deepcopy(fixture)
    target = out[name]
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
    count = checked["canopy"]["plant_mapping"]["segments_without_plant"]
    digest = checked["archived_mapping"]["plant_mapping"]["record_sha256"]
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
                checked, "archived_mapping", "plant_mapping_resolved_key", key
            )
            for key in ("valley@", "valley@3", f"valley@{digest}")
        ),
    ]

    for fixture in broken:
        assert _normalized(fixture) != _normalized(checked)
    reminted = _changed(checked, "validated", "event_id", "another-id")
    assert _normalized(reminted) == _normalized(checked)


@pytest.mark.parametrize(
    "name",
    [
        "generate_trait_fixture",
        "generate_project_fixture",
        "generate_tensorboard_fixture",
    ],
)
def test_the_fixture_is_what_a_regeneration_serves_value_for_value(
    name: str,
) -> None:
    """A trait, project or TensorBoard fixture is what a regeneration serves, value for value.

    Each generator's checked-in file equals what its ``produce`` serves.
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

    def failing(_served: dict) -> dict:
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

"""The browser's fixtures are what the platform's own producers record and serve.

The file of each generator :data:`SERVED_FIXTURES` names is held to what a regeneration serves,
value for value, each value a run fills read as its stand-in by the generator itself
(``fixture_session.normalized``). The scratch session the generators make their scenes in
(``tools/fixture_session.py``) is held to leaving nothing behind.
"""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._web_fixtures import named_project
from tools.fixture_session import (
    INSTANT_BASE,
    normalized,
    opened,
    scratch_session,
    served_instants,
)

if TYPE_CHECKING:
    from types import ModuleType


def _generator(name: str) -> ModuleType:
    """Import and return the fixture generator module ``tools.<name>``."""
    return importlib.import_module(f"tools.{name}")


WORKSPACE = Path("scratch") / "workspace"
"""A workspace standing where a scratch session's would, for a normalization alone."""


def test_a_numbered_value_keeps_its_sharing_and_no_more() -> None:
    """Strings a run fills read by their first appearance, so sharing survives and only sharing.

    Two keys holding one value read as one placeholder, a reminted value reads as the one it
    replaced, a value that stopped sharing reads as a new one, and a value of another type than a
    string is kept as served.
    """
    numbered = frozenset({"event_id", "cited"})
    served = {
        "a": {"event_id": "x1"},
        "b": {"cited": "x1"},
        "c": {"event_id": "y"},
    }

    assert normalized(served, WORKSPACE, numbered=numbered) == {
        "a": {"event_id": "<event_id n0>"},
        "b": {"cited": "<event_id n0>"},
        "c": {"event_id": "<event_id n1>"},
    }
    reminted = {
        "a": {"event_id": "z"},
        "b": {"cited": "z"},
        "c": {"event_id": "w"},
    }
    assert normalized(reminted, WORKSPACE, numbered=numbered) == normalized(
        served, WORKSPACE, numbered=numbered
    )
    unshared = {
        "a": {"event_id": "x1"},
        "b": {"cited": "x2"},
        "c": {"event_id": "y"},
    }
    assert normalized(unshared, WORKSPACE, numbered=numbered) != normalized(
        served, WORKSPACE, numbered=numbered
    )
    assert normalized({"event_id": 7}, WORKSPACE, numbered=numbered) == {
        "event_id": 7
    }


def test_an_instant_reads_as_an_instant_keeping_its_order_and_sharing() -> (
    None
):
    """Served instants read as instants a second apart in time order, spelled as served.

    The later instant served first still reads as the later one, two spellings of one instant
    read as one, and epoch seconds stay epoch seconds of their own type.
    """
    instants = frozenset({"created", "modified"})
    served = {
        "later": {"created": "2026-03-04T12:00:05+00:00"},
        "earlier": {"created": "2026-03-04T12:00:00.250000+00:00"},
        "same": {"created": "2026-03-04T14:00:05+02:00"},
        "stamp": {"modified": 1772625600.25},
        "whole": {"modified": 1772625605},
    }

    def second(n: int) -> datetime:
        return INSTANT_BASE + timedelta(seconds=n)

    assert normalized(served, WORKSPACE, instants=instants) == {
        "later": {"created": second(1).isoformat()},
        "earlier": {"created": second(0).isoformat()},
        "same": {"created": second(1).isoformat()},
        "stamp": {"modified": second(0).timestamp()},
        "whole": {"modified": int(second(1).timestamp())},
    }


def test_the_served_instants_are_the_fields_a_served_model_types_as_an_instant() -> (
    None
):
    """A model's ``AwareDatetime`` field is a served instant; a string or a number one is not.

    ``created`` is a run record's instant, ``started`` a session's; ``modified`` is a project
    summary's epoch seconds and ``created_at`` a label document's string stamp.
    """
    found = served_instants()

    assert {"created", "started"} <= found
    assert not {"modified", "created_at"} & found


SERVED_FIXTURES = (
    "generate_delivery_fixture",
    "generate_trait_fixture",
    "generate_project_fixture",
    "generate_tensorboard_fixture",
    "generate_open_fixture",
    "generate_training_fixture",
    "generate_labels_fixture",
    "generate_bands_fixture",
    "generate_sessions_fixture",
    "generate_inference_fixture",
    "generate_panel_events_fixture",
    "generate_terminal_fixture",
    "generate_images_fixture",
    "generate_meta_fixture",
    "generate_annotate_fixture",
)
"""The fixture generators whose files hold no value a run mints, compared value for value."""


@pytest.mark.parametrize("name", SERVED_FIXTURES)
def test_the_fixture_is_what_a_regeneration_serves_value_for_value(
    name: str,
) -> None:
    """The generator's checked-in file equals what its ``produce`` serves, value for value."""
    module = _generator(name)

    served = module.produce()

    assert (
        json.loads(module.GENERATED_PATH.read_text(encoding="utf-8")) == served
    ), (
        f"{module.GENERATED_PATH} is out of date; run python -m {module.__name__}"
    )


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

    with pytest.raises(AssertionError, match="kills unconfirmed"):
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

    with pytest.raises(AssertionError, match="kills unconfirmed") as refused:
        module.produce()
    assert isinstance(refused.value.__context__, NormalizationFailedError)


def test_a_wrong_typed_stamped_value_is_kept_for_the_staleness_comparison() -> (
    None
):
    """A value of another type than the producer writes under its key is kept as served.

    Under the training fixture's own rules, a ``None`` where a timestamp belongs and a string
    or a boolean where a pid belongs survive normalization, so a regeneration carrying them
    differs from the checked-in fixture, while the producer's own types read as their
    placeholders.
    """
    module = _generator("generate_training_fixture")

    def under_its_rules(served: dict) -> dict:
        return normalized(
            served,
            WORKSPACE,
            stamped=module.STAMPED,
            numbered=module.NUMBERED,
        )

    assert under_its_rules({"created": None, "pid": "1"}) == {
        "created": None,
        "pid": "1",
    }
    assert under_its_rules({"pid": True}) == {"pid": True}
    assert under_its_rules(
        {"created": "2026-01-01T00:00:00+00:00", "pid": 7}
    ) == {"created": INSTANT_BASE.isoformat(), "pid": 0}


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

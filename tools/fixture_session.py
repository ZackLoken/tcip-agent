"""Scratch-session utilities for producing served JSON as fixture files.

They are a scratch workspace with a web-backend client serving it (:func:`scratch_session`), a
request to a named route asserting its answer's status (:func:`answered`), a named socket's URL
(:func:`socket_url`), the opening of a project through the picker's route (:func:`opened`), a walk
that replaces each leaf of a served value (:func:`replaced`), the normalization of what a run or a
platform fills differently in a served value (:func:`normalized`), the points a route's background
worker holds at until the scene steps it on (:class:`Gate`), and the write of a fixture file
(:func:`write_fixture`). Every scene's requests name :data:`ACTOR`.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from tests import formatted, write_generated
from tests._web_fixtures import backend_session

if TYPE_CHECKING:
    from fastapi.testclient import TestClient
    from pydantic import JsonValue
    from starlette.testclient import httpx

    from tcip_web.jobstore import Job, JobRegistry
    from tcip_web.state import OpenProject


@contextmanager
def scratch_session() -> Iterator[tuple[Path, TestClient]]:
    """Yield a fresh scratch workspace and a client of the web backend serving it.

    The client is ``tests._web_fixtures.backend_session``'s; the directory is removed on exit,
    after the backend's session has closed.
    """
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory).resolve() / "workspace"
        workspace.mkdir()
        with backend_session(workspace) as client:
            yield workspace, client


ACTOR = "grower"
"""The person every scene's requests name."""


def responded(
    client: TestClient,
    route: str,
    status: int,
    body: BaseModel | Mapping[str, JsonValue] | None = None,
    *,
    path: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """Send a request to the route named ``route`` through ``client``; return its response.

    The route's path is the app's ``url_path_for(route, **path)``. With ``body`` the request
    POSTs it, a model as it dumps to JSON and a mapping as it stands, else it is a GET; ``params``
    are its query. A response of any other status than ``status`` raises (``AssertionError``)
    carrying its text.
    """
    from tcip_web.app import app

    url = app.url_path_for(route, **(path or {}))
    if body is None:
        response = client.get(url, params=params, headers=headers)
    else:
        response = client.post(
            url,
            json=body.model_dump(mode="json")
            if isinstance(body, BaseModel)
            else dict(body),
            params=params,
            headers=headers,
        )
    assert response.status_code == status, response.text
    return response


def answered(
    client: TestClient,
    route: str,
    status: int,
    body: BaseModel | Mapping[str, JsonValue] | None = None,
    **request: Mapping[str, str] | None,
) -> JsonValue:
    """Send a request as :func:`responded` does; return what the route answered.

    A success answers its JSON body, a refusal ``{"status": status, "body": body}`` so the status
    travels with the body.
    """
    response = responded(client, route, status, body, **request)
    answer = response.json()
    return (
        answer if response.is_success else {"status": status, "body": answer}
    )


def detail_of(refusal: JsonValue) -> JsonValue:
    """Return the ``detail`` of a refusal :func:`answered` returned."""
    assert isinstance(refusal, dict), refusal
    body = refusal["body"]
    assert isinstance(body, dict), refusal
    return body["detail"]


def socket_url(client: TestClient, route: str, **path: str) -> str:
    """Return the websocket URL of the route named ``route`` on ``client``'s backend."""
    from tcip_web.app import app

    joined = client.base_url.join(app.url_path_for(route, **path))
    return str(joined.copy_with(scheme="ws"))


def opened(client: TestClient, project: OpenProject) -> JsonValue:
    """Open ``project`` through ``POST /api/projects/open`` as :data:`ACTOR`; return its answer.

    Raises on a refusal.
    """
    return answered(
        client, "open_project", 200, {"id": project.id, "user": ACTOR}
    )


type Leaf = Callable[[str | None, JsonValue], JsonValue]
"""A replacement of one leaf of a served value, given the dict key it sits under."""


def replaced(
    served: Mapping[str, JsonValue], leaf: Leaf
) -> dict[str, JsonValue]:
    """Return the served mapping with each leaf (anything not a dict or a list) through ``leaf``.

    A leaf becomes ``leaf(key, leaf value)``, ``key`` being the dict key the leaf sits under,
    ``None`` for a list item.
    """
    return {k: _replaced(v, leaf, k) for k, v in served.items()}


def _replaced(value: JsonValue, leaf: Leaf, key: str | None) -> JsonValue:
    """Return ``value``, under ``key``, with each of its leaves replaced through ``leaf``."""
    if isinstance(value, dict):
        return replaced(value, leaf)
    if isinstance(value, list):
        return [_replaced(v, leaf, None) for v in value]
    return leaf(key, value)


SCRATCH_ROOT = "/scratch"
"""What a scratch session's directory reads as in a served string."""

FLOAT_PLACES = 6
"""The decimal places a served float is compared to: a computed value's last bits differ between
the platforms a scene runs on, and a delivered meter value carries no more."""

type Stamp = tuple[type, JsonValue]
"""The exact type of the value a producer writes under a key, and what that value reads as."""

_NOTHING: Mapping = MappingProxyType({})


INSTANT_BASE = datetime(2000, 1, 1, tzinfo=UTC)
"""The instant a fixture's earliest served instant reads as, a sample value: the N-th distinct
served instant in time order reads as N seconds after it."""


def _states_an_instant(schema: JsonValue) -> bool:
    """Return whether a property's JSON schema ``schema`` types its value as an instant."""
    return isinstance(schema, dict) and (
        schema.get("format") == "date-time"
        or any(_states_an_instant(m) for m in cast("list", schema.get("anyOf", [])))
    )


@cache
def served_instants() -> frozenset[str]:
    """Return every field name a served model types as an instant (``AwareDatetime``).

    The models are the generated frontend types' (``tools.generate_frontend_types``), each
    nested model among them.
    """
    from tools.generate_frontend_types import _schemas

    _names, defs = _schemas()
    return frozenset(
        prop
        for schema in defs.values()
        for prop, prop_schema in schema.get("properties", {}).items()
        if _states_an_instant(prop_schema)
    )


LABEL_DOCUMENT_STAMPS = frozenset({"created_at", "accepted_at", "at"})
"""The instants an annotation and a completion mark stamp, strings no served model types as an
instant because ``tcip_annotation.json_io`` reads the label document by hand."""


def _instant(value: JsonValue) -> datetime | None:
    """Return the instant ``value`` states as pydantic reads a ``datetime``, else ``None``."""
    try:
        return TypeAdapter(datetime).validate_python(value)
    except ValidationError:
        return None


def normalized(
    served: Mapping[str, JsonValue],
    workspace: Path,
    *,
    instants: frozenset[str] = frozenset(),
    stamped: Mapping[str, Stamp] = _NOTHING,
    numbered: frozenset[str] = frozenset(),
    minted: Mapping[str, str] = _NOTHING,
    places: Mapping[Path, str] = _NOTHING,
    rounded: bool = False,
) -> dict[str, JsonValue]:
    """Return ``served`` with each value a run or a platform fills differently read as its stand-in.

    An instant a run fills sits under a key a served model types as an instant
    (:func:`served_instants`) or under an ``instants`` key, one no served model types so, and
    becomes the instant of its rank
    among the distinct served instants in time order, counted from :data:`INSTANT_BASE` in
    seconds and spelled as it was served (an ISO-8601 string or epoch seconds), so it still parses
    and two places sharing one still share it, a string spelling of one becoming the same wherever
    another string quotes it; a string a run fills sits under a ``numbered`` key and becomes
    ``<key nN>``, ``N`` counting the distinct such strings in order of first appearance, unless
    it is an id ``minted`` names; a number a run fills sits under a ``stamped`` key and, when its
    type is exactly the one its :data:`Stamp` names, becomes the placeholder beside it; with
    ``rounded``, each
    float is rounded to :data:`FLOAT_PLACES`; and in each other string each id ``minted`` names
    becomes its name and each spelling of the scratch session's directory (``workspace``'s
    parent, read as :data:`SCRATCH_ROOT`) and of each directory ``places`` names, its own, its
    forward-slash one and the one a ``repr`` quotes, becomes its placeholder, a string naming one
    then reading with forward slashes alone. Every other value, a wrong-typed one under a
    ``stamped``, ``numbered`` or ``instants`` key among them, is kept as served.
    """
    places = {workspace.parent: SCRATCH_ROOT, **places}
    instants = served_instants() | instants
    seen: dict[str, str] = {}
    found: dict[str | float, datetime] = {}

    def noted(key: str | None, value: JsonValue) -> JsonValue:
        if (
            key in instants
            and isinstance(value, str | int | float)
            and (at := _instant(value)) is not None
        ):
            found[value] = at
        return value

    replaced(served, noted)
    rank = {at: n for n, at in enumerate(sorted(set(found.values())))}

    def read_as(value: str | float) -> str | float:
        at = INSTANT_BASE + timedelta(seconds=rank[found[value]])
        if isinstance(value, str):
            return at.isoformat()
        return (
            int(at.timestamp()) if isinstance(value, int) else at.timestamp()
        )

    spelled_instants = {
        value: str(read_as(value)) for value in found if isinstance(value, str)
    }

    def leaf(key: str | None, value: JsonValue) -> JsonValue:
        if (
            key in instants
            and isinstance(value, str | int | float)
            and value in found
        ):
            return read_as(value)
        if key is not None and key in stamped:
            kind, placeholder = stamped[key]
            return placeholder if type(value) is kind else value
        if key in numbered and isinstance(value, str) and value not in minted:
            return seen.setdefault(value, f"<{key} n{len(seen)}>")
        if rounded and isinstance(value, float):
            return round(value, FLOAT_PLACES)
        if not isinstance(value, str):
            return value
        for minted_id, name in (*minted.items(), *spelled_instants.items()):
            value = value.replace(minted_id, name)
        spelled = value
        for directory, placeholder in places.items():
            for spelling in (
                str(directory),
                directory.as_posix(),
                repr(str(directory))[1:-1],
            ):
                spelled = spelled.replace(spelling, placeholder)
        return (
            value
            if spelled == value
            else spelled.replace("\\\\", "/").replace("\\", "/")
        )

    return replaced(served, leaf)


WAIT_S = 60.0
"""How long a scene waits on a route's background worker: for it to end, to pause at a
:class:`Gate`, or to be stepped on. A sample bound, far above a scratch session's worker time."""


def ended[J: Job](registry: JobRegistry[J], job_id: str) -> J:
    """Wait for the job ``job_id`` names in ``registry`` to end and return its snapshot then.

    The job's own thread is joined for at most :data:`WAIT_S`; one still running raises.
    """
    job = registry.get(job_id)
    assert job is not None and job.thread is not None, job_id
    job.thread.join(timeout=WAIT_S)
    assert not job.thread.is_alive(), job_id
    done = registry.get(job_id)
    assert done is not None, job_id
    return done


class Gate:
    """The points a route's background worker holds at, each until the scene steps it on."""

    def __init__(self) -> None:
        """Start with no worker paused and none stepped on."""
        self._reached = threading.Semaphore(0)
        self._stepped = threading.Semaphore(0)

    def pause(self) -> None:
        """Hold the worker calling it until the scene steps it on."""
        self._reached.release()
        assert self._stepped.acquire(timeout=WAIT_S), "never stepped on"

    def reached(self) -> None:
        """Wait for a worker to pause."""
        assert self._reached.acquire(timeout=WAIT_S), "no worker paused"

    def step(self) -> None:
        """Step the paused worker on."""
        self._stepped.release()


def write_fixture(path: Path, content: JsonValue) -> int:
    """Write ``content`` to ``path`` as JSON the frontend's Prettier formats; return exit code 0."""
    return write_generated(path, formatted(json.dumps(content), path))

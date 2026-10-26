"""The browser and the backend meet on one declaration of the API's paths.

The FastAPI app registers the paths; ``tools/generate_frontend_routes.py`` projects the ones the
frontend's sources read into the module at its ``GENERATED_PATH``. These tests hold that
projection to the app's registered routes and the sources' reads, check that the frontend writes
no backend path of its own beside it, and check that each path the browser asks for is one the dev
server forwards, as Vite resolves its proxy from ``vite.config.ts``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI

from tcip_mcp.web_client import LOOPBACK_HOST, Panel
from tcip_web.__main__ import DEFAULT_PORT
from tcip_web.app import app
from tests import FRONTEND, VITE_CONFIG, printed_json
from tests._route_walk import route_generator as _generator

GENERATED = _generator().GENERATED_PATH
PROXY_GENERATED = _generator().PROXY_GENERATED_PATH


@pytest.fixture(scope="module")
def frontend() -> dict:
    """Return the generator's own read of the frontend's sources, run once for the module."""
    return _generator().frontend_reads()


@pytest.fixture(scope="module")
def dev_server_proxy() -> dict:
    """Return the dev server's proxy rules, as Vite resolves them from the real ``vite.config.ts``.

    They are the ``proxy`` :data:`~tests.VITE_CONFIG` prints over the frontend directory, resolved
    with no ``TCIP_WEB_BACKEND`` override in its environment, so they forward to the default
    backend.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("TCIP_WEB_BACKEND", raising=False)
        return printed_json(VITE_CONFIG, FRONTEND)["proxy"]


def test_the_generated_path_module_is_what_the_registered_routes_produce(
    frontend: dict,
) -> None:
    """The checked-in path module equals what the generator renders over the sources' reads."""
    generated = _generator().render(app, frontend["reads"])
    assert GENERATED.read_text(encoding="utf-8") == generated, (
        f"{GENERATED} is out of date; run python -m {_generator().__name__}"
    )


def test_no_frontend_source_writes_a_backend_path_of_its_own(
    frontend: dict,
) -> None:
    """Outside the generated module, no non-test frontend source holds a backend path literal."""
    offenders = [
        literal
        for literal in frontend["literals"]
        if Path(literal["file"]) != GENERATED
    ]
    assert not offenders, (
        f"these write a backend path instead of using ROUTES: {offenders}"
    )


def test_the_generated_proxy_module_is_what_the_registered_routes_produce() -> (
    None
):
    """The checked-in dev-proxy module equals what the generator renders from the app's routes."""
    generated = _generator().render_proxy(app)
    assert PROXY_GENERATED.read_text(encoding="utf-8") == generated, (
        f"{PROXY_GENERATED} is out of date; run python -m {_generator().__name__}"
    )


def test_vite_config_proxies_each_generated_entry_to_the_backend(
    dev_server_proxy: dict,
) -> None:
    """The real dev server proxies exactly the generated entries, each to the backend.

    Each entry's prefix is a proxy rule carrying its ``ws``, the origin changed, and the backend's
    loopback host at the port it asks for by default as the target.
    """
    target = f"http://{LOOPBACK_HOST}:{DEFAULT_PORT}"
    assert dev_server_proxy == {
        path: {"target": target, "changeOrigin": True, "ws": ws}
        for path, ws in _generator().collect_proxy_entries(app)
    }


def test_the_real_apis_websocket_routes_make_the_api_prefix_proxy_websockets() -> (
    None
):
    """The ``/api`` proxy entry generated from the app's routes carries ``ws: true``."""
    entries = dict(_generator().collect_proxy_entries(app))
    assert entries.get("/api") is True


def test_the_frontend_serving_paths_are_never_proxy_entries() -> None:
    """The app's frontend-serving paths do not become proxy entries.

    Over an app serving ``/``, ``/health`` and one ``/api`` route, the proxy entries are
    exactly ``{"/api"}``, and none is a frontend-serving path.
    """
    from tcip_web.app import FRONTEND_SERVING_PATHS

    fresh = FastAPI()

    @fresh.get("/")
    def _root() -> dict:
        return {}

    @fresh.get("/health")
    def _health() -> dict:
        return {}

    @fresh.get("/api/orchard/blocks")
    def _list_blocks() -> dict:
        return {}

    prefixes = {
        path for path, _ws in _generator().collect_proxy_entries(fresh)
    }
    assert prefixes == {"/api"}
    assert not (prefixes & FRONTEND_SERVING_PATHS)


def test_every_path_the_browser_asks_for_is_forwarded_by_the_dev_server(
    frontend: dict, dev_server_proxy: dict
) -> None:
    """The paths the frontend reads all fall under a prefix the dev server forwards.

    The prefixes are the dev server's proxy rules as Vite resolves them, and the routes read are
    the generator's own read of the frontend's sources.
    """
    prefixes = tuple(dev_server_proxy)
    assert prefixes, "Vite resolves no proxy rule from vite.config.ts"

    read = _generator().read_routes(app, frontend["reads"])
    assert read, (
        "no ROUTES reference found in the frontend, so nothing was checked"
    )
    unproxied = sorted(
        f"{name} ({path})"
        for name, path, _, _ in read
        if not path.startswith(prefixes)
    )
    assert not unproxied, f"the dev server forwards none of these: {unproxied}"


def _orchard_app() -> FastAPI:
    """Return a fresh app registering two HTTP routes and two sockets."""
    fresh = FastAPI()

    @fresh.get("/api/orchard/blocks")
    def _list_blocks() -> dict:
        return {}

    @fresh.post("/api/orchard/blocks/{block_id}/close")
    def _close_block(block_id: str) -> dict:
        return {}

    @fresh.websocket("/ws/orchard/{block_id}")
    async def _block_socket(block_id: str) -> None:
        return None

    @fresh.websocket("/ws/panel/{panel}")
    async def _panel_socket(panel: Panel) -> None:
        return None

    return fresh


def test_a_route_the_generator_has_not_seen_before_still_reaches_the_browser() -> (
    None
):
    """A newly registered route is projected without anything being taught about it.

    The names carry the method and the path parameters, the rendered module carries exactly
    the routes named to it, formatted, and it passes a parameter through
    ``encodeURIComponent`` where it is substituted. A parameter declared as a generated
    vocabulary is typed by its generated alias, imported from the generated types module.
    """
    generator = _generator()
    fresh = _orchard_app()

    names = {
        name: path for name, path, _, _ in generator.collect_routes(fresh)
    }
    assert names["getOrchardBlocks"] == "/api/orchard/blocks"
    assert (
        names["postOrchardBlocksByBlockIdClose"]
        == "/api/orchard/blocks/{block_id}/close"
    )
    assert names["socketWsOrchardByBlockId"] == "/ws/orchard/{block_id}"

    read = {
        "getOrchardBlocks",
        "postOrchardBlocksByBlockIdClose",
        "socketWsPanelByPanel",
    }
    assert generator.render(fresh, read) == (
        generator.HEADER
        + 'import type { Panel } from "./types.generated";\n\n'
        "export const ROUTES = {\n"
        '  getOrchardBlocks: "/api/orchard/blocks",\n'
        "  postOrchardBlocksByBlockIdClose: (blockId: string) =>\n"
        "    `/api/orchard/blocks/${encodeURIComponent(blockId)}/close`,\n"
        "  socketWsPanelByPanel: (panel: Panel) =>"
        " `/ws/panel/${encodeURIComponent(panel)}`,\n" + generator.FOOTER
    )


def test_a_route_under_an_included_routers_prefix_is_named_at_its_path() -> (
    None
):
    """A route and a socket a router includes under its own prefix keep that prefix.

    Each is named and projected at the path the app serves it on, the include prefix with the
    route's own path.
    """
    router = APIRouter()

    @router.get("/items")
    def _list_items() -> dict:
        return {}

    @router.websocket("/ws/{item_id}")
    async def _item_socket(item_id: str) -> None:
        return None

    fresh = FastAPI()
    fresh.include_router(router, prefix="/api/farm")

    routes = {
        name: (path, method)
        for name, path, method, _ in _generator().collect_routes(fresh)
    }
    assert routes == {
        "getFarmItems": ("/api/farm/items", "GET"),
        "socketFarmWsByItemId": ("/api/farm/ws/{item_id}", "WEBSOCKET"),
    }


def test_a_read_naming_no_registered_route_is_refused_naming_it() -> None:
    """A read of a name no registered route generates is refused, naming the read."""
    with pytest.raises(SystemExit, match="getOrchardTrees"):
        _generator().render(
            _orchard_app(), {"getOrchardBlocks", "getOrchardTrees"}
        )


def test_two_routes_that_would_share_a_name_are_refused_rather_than_collapsed() -> (
    None
):
    """Two routes generating one name are refused, naming it."""
    generator = _generator()
    clashing = FastAPI()

    @clashing.get("/api/orchard/blocks")
    def _by_path() -> dict:
        return {}

    @clashing.get("/api/orchard-blocks")
    def _by_dash() -> dict:
        return {}

    with pytest.raises(SystemExit, match="getOrchardBlocks"):
        generator.collect_routes(clashing)

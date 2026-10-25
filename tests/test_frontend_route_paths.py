"""The browser and the backend meet on one declaration of the API's paths.

The FastAPI app registers the paths; ``tools/generate_frontend_routes.py`` projects the ones the
frontend's sources read into ``frontend/src/api/routes.ts``. These tests hold that projection to
the app's registered routes and the sources' reads, check that the frontend writes no backend
path of its own beside it, and check that each path the browser asks for is one the dev server
forwards.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI

from tcip_web.app import app
from tests import FRONTEND_SRC
from tests._route_walk import route_generator as _generator

GENERATED = _generator().GENERATED_PATH
PROXY_GENERATED = _generator().PROXY_GENERATED_PATH
VITE_CONFIG = FRONTEND_SRC.parent / "vite.config.ts"

_PROXY_ENTRY_RE = re.compile(
    r'\{\s*path:\s*"([^"]+)",\s*ws:\s*(true|false)\s*\}'
)
_PROXY_BUILD_RE = re.compile(
    r"Object\.fromEntries\(\s*DEV_PROXY\.map\(\(\s*\{\s*path,\s*ws\s*\}\s*\)\s*=>\s*"
    r"\[\s*path\s*,\s*\{\s*target:\s*BACKEND\s*,\s*changeOrigin:\s*true\s*,\s*ws\s*\}\s*\]\s*\)\s*,?\s*\)",
    re.S,
)


@pytest.fixture(scope="module")
def frontend() -> dict:
    """Return the generator's own read of the frontend's sources, run once for the module."""
    return _generator().frontend_reads()


def test_the_generated_path_module_is_what_the_registered_routes_produce(
    frontend: dict,
) -> None:
    """The checked-in path module equals what the generator renders over the sources' reads."""
    generated = _generator().render(app, frontend["reads"])
    assert GENERATED.read_text(encoding="utf-8") == generated, (
        "packages/tcip-web/frontend/src/api/routes.ts is out of date; "
        "run python -m tools.generate_frontend_routes"
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
        "packages/tcip-web/frontend/src/api/devProxy.generated.ts is out of date; "
        "run python -m tools.generate_frontend_routes"
    )


def _proxy_entries_from_generated_module() -> tuple[tuple[str, bool], ...]:
    """Return the ``DEV_PROXY`` entries as the checked-in generated module declares them."""
    text = PROXY_GENERATED.read_text(encoding="utf-8")
    return tuple(
        (path, ws == "true") for path, ws in _PROXY_ENTRY_RE.findall(text)
    )


def test_vite_config_builds_its_real_proxy_from_the_generated_module() -> None:
    """The real ``vite.config.ts`` builds its proxy from each generated ``DEV_PROXY`` entry.

    The config imports the generated module, and its ``server.proxy`` maps each ``DEV_PROXY``
    entry's ``path`` and ``ws`` fields into a proxy rule.
    """
    vite_text = VITE_CONFIG.read_text(encoding="utf-8")
    assert "./src/api/devProxy.generated" in vite_text, (
        "vite.config.ts no longer imports the generated proxy module"
    )
    proxy_block = re.search(r"proxy:\s*(.*?),\n\s*\},", vite_text, re.S)
    assert proxy_block is not None, (
        "the Vite server.proxy assignment is no longer where this test reads it"
    )
    assert _PROXY_BUILD_RE.search(proxy_block.group(1)), (
        "vite.config.ts's real server.proxy is not built from each DEV_PROXY entry's own "
        "path/ws fields: " + proxy_block.group(1)
    )


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
    frontend: dict,
) -> None:
    """The paths the frontend reads all fall under a prefix the generated dev proxy forwards.

    The prefixes are parsed from the checked-in proxy module, and the routes read are the
    generator's own read of the frontend's sources.
    """
    prefixes = tuple(
        path for path, _ws in _proxy_entries_from_generated_module()
    )
    assert prefixes, (
        "no proxied prefixes parsed out of the generated proxy module"
    )

    read = _generator().read_routes(app, frontend["reads"])
    assert read, (
        "no ROUTES reference found in the frontend, so nothing was checked"
    )
    unproxied = sorted(
        f"{name} ({path})"
        for name, path, _ in read
        if not path.startswith(prefixes)
    )
    assert not unproxied, f"the dev server forwards none of these: {unproxied}"


def _orchard_app() -> FastAPI:
    """Return a fresh app registering two HTTP routes and one socket."""
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

    return fresh


def test_a_route_the_generator_has_not_seen_before_still_reaches_the_browser() -> (
    None
):
    """A newly registered route is projected without anything being taught about it.

    The names carry the method and the path parameters, the rendered module carries exactly
    the routes named to it, and it passes a parameter through ``encodeURIComponent`` where it
    is substituted.
    """
    generator = _generator()
    fresh = _orchard_app()

    names = {name: path for name, path, _ in generator.collect_routes(fresh)}
    assert names["getOrchardBlocks"] == "/api/orchard/blocks"
    assert (
        names["postOrchardBlocksByBlockIdClose"]
        == "/api/orchard/blocks/{block_id}/close"
    )
    assert names["socketWsOrchardByBlockId"] == "/ws/orchard/{block_id}"

    read = {"getOrchardBlocks", "postOrchardBlocksByBlockIdClose"}
    assert generator.render(fresh, read) == (
        generator.HEADER + '  getOrchardBlocks: "/api/orchard/blocks",\n'
        "  postOrchardBlocksByBlockIdClose: (blockId: string) =>\n"
        "    `/api/orchard/blocks/${encodeURIComponent(blockId)}/close`,\n"
        + generator.FOOTER
    )


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

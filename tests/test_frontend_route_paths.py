"""The browser and the backend meet on one declaration of the API's paths.

The FastAPI app registers the paths; ``tools/generate_frontend_routes.py`` projects them into
``frontend/src/api/routes.ts``. These tests hold that projection to the routes the app really has,
keep the frontend from writing a path of its own beside it, and check that every path the browser
asks for is one the dev server forwards.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from fastapi import FastAPI

from tcip_web.app import app

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"
GENERATED = FRONTEND_SRC / "api" / "routes.ts"
PROXY_GENERATED = FRONTEND_SRC / "api" / "devProxy.generated.ts"
VITE_CONFIG = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "vite.config.ts"
GENERATOR = REPO_ROOT / "tools" / "generate_frontend_routes.py"

_LITERAL_RE = re.compile(r"""["'`](/(?:api|ws)/[^"'`]*)["'`]""")
_ROUTES_USE_RE = re.compile(r"\bROUTES\.([A-Za-z0-9_]+)")
_PROXY_ENTRY_RE = re.compile(r'\{\s*path:\s*"([^"]+)",\s*ws:\s*(true|false)\s*\}')
_PROXY_BUILD_RE = re.compile(
    r"Object\.fromEntries\(\s*DEV_PROXY\.map\(\(\s*\{\s*path,\s*ws\s*\}\s*\)\s*=>\s*"
    r"\[\s*path\s*,\s*\{\s*target:\s*BACKEND\s*,\s*changeOrigin:\s*true\s*,\s*ws\s*\}\s*\]\s*\)\s*,?\s*\)",
    re.S,
)


def _generator():
    """The path-module generator loaded as a module, so the test regenerates the same way CI does."""
    spec = importlib.util.spec_from_file_location("tcip_frontend_route_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frontend_sources(include_tests: bool) -> list[Path]:
    files = [p for p in FRONTEND_SRC.rglob("*.ts")] + [p for p in FRONTEND_SRC.rglob("*.tsx")]
    if include_tests:
        return sorted(files)
    return sorted(p for p in files if ".test." not in p.name and "test" not in p.parent.name)


def test_the_generated_path_module_is_what_the_registered_routes_produce() -> None:
    """The checked-in module is a projection of the app's routes, not a hand-edited copy.

    Regenerating and comparing is the whole guarantee: an added, renamed or re-methoded route that
    never reached the module would leave the browser asking for a path the backend no longer serves.
    """
    generated = _generator().render(app)
    assert GENERATED.read_text(encoding="utf-8") == generated, (
        "packages/tcip-web/frontend/src/api/routes.ts is out of date; "
        "run python tools/generate_frontend_routes.py"
    )


def test_no_frontend_module_writes_a_backend_path_of_its_own() -> None:
    """Outside the generated module, the frontend holds no path string of its own.

    A call site that spells a path itself is a second declaration of the contract, which is what
    drifts: the backend renames the route and the browser keeps asking for the old one. Frontend
    test files are left out: a URL written there is the expectation being asserted, not a caller.
    """
    registered_shapes = {
        re.sub(r"\{[^}]*\}", "{}", path) for _, path, _ in _generator().collect_routes(app)
    }

    offenders: list[str] = []
    for source in _frontend_sources(include_tests=False):
        if source == GENERATED:
            continue
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            for literal in _LITERAL_RE.findall(line):
                shape = re.sub(r"\$\{[^}]*\}", "{}", literal.split("?")[0])
                if literal.startswith("/api/") or shape in registered_shapes:
                    offenders.append(
                        f"{source.relative_to(REPO_ROOT).as_posix()}:{line_no} {literal}"
                    )
    assert not offenders, "these write a backend path instead of using ROUTES:\n" + "\n".join(offenders)


def test_the_generated_proxy_module_is_what_the_registered_routes_produce() -> None:
    """The checked-in dev-proxy module is a projection of the app's routes, not a hand-edited
    copy: a socket route added under a new prefix that never reached the module would leave the
    dev server unable to proxy it."""
    generated = _generator().render_proxy(app)
    assert PROXY_GENERATED.read_text(encoding="utf-8") == generated, (
        "packages/tcip-web/frontend/src/api/devProxy.generated.ts is out of date; "
        "run python tools/generate_frontend_routes.py"
    )


def _proxy_entries_from_generated_module() -> tuple[tuple[str, bool], ...]:
    """The ``DEV_PROXY`` entries as the checked-in generated module actually declares them."""
    text = PROXY_GENERATED.read_text(encoding="utf-8")
    return tuple((path, ws == "true") for path, ws in _PROXY_ENTRY_RE.findall(text))


def test_vite_config_builds_its_real_proxy_from_the_generated_module() -> None:
    """The one assertion that reads the real ``vite.config.ts``: its ``server.proxy`` is built by
    mapping each ``DEV_PROXY`` entry's own ``path``/``ws`` fields into a proxy rule, structurally,
    never by a literal of its own a substring match could mistake for the real thing."""
    vite_text = VITE_CONFIG.read_text(encoding="utf-8")
    assert "./src/api/devProxy.generated" in vite_text, (
        "vite.config.ts no longer imports the generated proxy module"
    )
    proxy_block = re.search(r"proxy:\s*(.*?),\n\s*\},", vite_text, re.S)
    assert proxy_block is not None, "the Vite server.proxy assignment is no longer where this test reads it"
    assert _PROXY_BUILD_RE.search(proxy_block.group(1)), (
        "vite.config.ts's real server.proxy is not built from each DEV_PROXY entry's own "
        "path/ws fields: " + proxy_block.group(1)
    )


def test_the_real_apis_websocket_routes_make_the_api_prefix_proxy_websockets() -> None:
    """Three sockets are mounted under ``/api`` rather than under ``/ws``, so the generated
    ``/api`` entry must itself carry ``ws: true`` or the dev server never proxies them."""
    entries = dict(_generator().collect_proxy_entries(app))
    assert entries.get("/api") is True


def test_the_frontend_serving_paths_are_never_proxy_entries() -> None:
    """A route the app itself serves the frontend or its health probe through never becomes a
    proxy entry, so the dev server keeps serving its own root, modules and HMR there."""
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

    prefixes = {path for path, _ws in _generator().collect_proxy_entries(fresh)}
    assert prefixes == {"/api"}
    assert not (prefixes & FRONTEND_SERVING_PATHS)


def test_every_path_the_browser_asks_for_is_forwarded_by_the_dev_server() -> None:
    """The paths the frontend references all fall under a prefix the generated dev proxy forwards.

    Sockets are the case worth stating: two of them are mounted under the API prefix rather than
    under /ws, so a proxy rule for /ws alone would leave them unreachable in development. The
    prefixes come from the checked-in proxy module, what the config's own ``server.proxy`` is
    held to build from, rather than only from the generator function that produced it.
    """
    prefixes = tuple(path for path, _ws in _proxy_entries_from_generated_module())
    assert prefixes, "no proxied prefixes parsed out of the generated proxy module"

    by_name = {name: path for name, path, _ in _generator().collect_routes(app)}
    referenced = {
        name
        for source in _frontend_sources(include_tests=False)
        if source != GENERATED
        for name in _ROUTES_USE_RE.findall(source.read_text(encoding="utf-8"))
    }
    assert referenced, "no ROUTES reference found in the frontend, so nothing was checked"
    unknown = sorted(name for name in referenced if name not in by_name)
    assert not unknown, f"the frontend asks for routes the backend does not register: {unknown}"
    unproxied = sorted(
        f"{name} ({by_name[name]})" for name in referenced if not by_name[name].startswith(prefixes)
    )
    assert not unproxied, f"the dev server forwards none of these: {unproxied}"


def test_a_route_the_generator_has_not_seen_before_still_reaches_the_browser() -> None:
    """A newly registered route is projected without anything being taught about it.

    The names carry the method and the path parameters, and a parameter is encoded where it is
    substituted, so a value with a slash in it cannot widen the path it was meant to fill.
    """
    generator = _generator()
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

    names = {name: path for name, path, _ in generator.collect_routes(fresh)}
    assert names["getOrchardBlocks"] == "/api/orchard/blocks"
    assert names["postOrchardBlocksByBlockIdClose"] == "/api/orchard/blocks/{block_id}/close"
    assert names["socketWsOrchardByBlockId"] == "/ws/orchard/{block_id}"

    module = generator.render(fresh)
    assert '  getOrchardBlocks: "/api/orchard/blocks",\n' in module
    assert (
        "  postOrchardBlocksByBlockIdClose: (blockId: string) =>\n"
        "    `/api/orchard/blocks/${encodeURIComponent(blockId)}/close`,\n"
    ) in module


def test_two_routes_that_would_share_a_name_are_refused_rather_than_collapsed() -> None:
    """One name cannot stand for two routes: the browser would reach the wrong one silently."""
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

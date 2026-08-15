"""The browser and the backend meet on one declaration of the API's paths.

The FastAPI app registers the paths; ``scripts/generate_frontend_routes.py`` projects them into
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
VITE_CONFIG = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "vite.config.ts"
GENERATOR = REPO_ROOT / "scripts" / "generate_frontend_routes.py"

_LITERAL_RE = re.compile(r"""["'`](/(?:api|ws)/[^"'`]*)["'`]""")
_ROUTES_USE_RE = re.compile(r"\bROUTES\.([A-Za-z0-9_]+)")


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
        "run python scripts/generate_frontend_routes.py"
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


def test_every_path_the_browser_asks_for_is_forwarded_by_the_dev_server() -> None:
    """The paths the frontend references all fall under a prefix the Vite proxy forwards.

    Sockets are the case worth stating: two of them are mounted under the API prefix rather than
    under /ws, so a proxy rule for /ws alone would leave them unreachable in development.
    """
    proxy_block = re.search(r"proxy:\s*\{(.*?)\n\s*\},", VITE_CONFIG.read_text(encoding="utf-8"), re.S)
    assert proxy_block is not None, "the Vite proxy block is no longer where this test reads it"
    prefixes = tuple(re.findall(r'"([^"]+)":', proxy_block.group(1)))
    assert prefixes, "no proxied prefixes parsed out of vite.config.ts"

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

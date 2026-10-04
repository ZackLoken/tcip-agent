"""MCP server entry point: every domain tool, served on stdio for the project named at start
(``--project <path>``)."""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tcip_mcp import agent_identity

logger = logging.getLogger(__name__)

_Tool = TypeVar("_Tool", bound=Callable[..., Any])

_TOOLS: list[tuple[Callable[..., Any], dict[str, Any]]] = []
"""Every function a tool module declared with :func:`tool`, with its registration options."""


class NoProject(ToolError):
    """A tool that acts on a project was called on a server started for none; the server hands
    the agent its message."""


def tool(**options: Any) -> Callable[[_Tool], _Tool]:
    """Declare the decorated function an MCP tool, with ``options`` passed to the server's own
    registration when :func:`build_server` builds a server, and return it unchanged."""
    def decorate(fn: _Tool) -> _Tool:
        _TOOLS.append((fn, options))
        return fn

    return decorate


# Import tool modules to declare their handlers. A tool module that needs torch imports it inside
# its own functions, so every tool registers whether or not torch is installed.
import tcip_mcp.tools.data_tools  # noqa: F401, E402
import tcip_mcp.tools.project_tools  # noqa: F401, E402
import tcip_mcp.tools.ingest_tools  # noqa: F401, E402
import tcip_mcp.tools.experiment_tools  # noqa: F401, E402
import tcip_mcp.tools.meta_tools  # noqa: F401, E402
import tcip_mcp.tools.knowledge_tools  # noqa: F401, E402
import tcip_mcp.tools.phenology_tools  # noqa: F401, E402
import tcip_mcp.tools.trait_tools  # noqa: F401, E402
import tcip_mcp.tools.annotation_tools  # noqa: F401, E402
import tcip_mcp.tools.vision_tools  # noqa: F401, E402
import tcip_mcp.tools.proposal_tools  # noqa: F401, E402
import tcip_mcp.tools.gui_tools  # noqa: F401, E402
import tcip_mcp.tools.feedback_tools  # noqa: F401, E402
import tcip_mcp.tools.training_tools  # noqa: F401, E402
import tcip_mcp.tools.inference_tools  # noqa: F401, E402
import tcip_mcp.tools.calibration_tools  # noqa: F401, E402
import tcip_mcp.tools.model_tools  # noqa: F401, E402
import tcip_mcp.tools.orthomosaic_tools  # noqa: F401, E402
import tcip_mcp.tools.delivery_tools  # noqa: F401, E402


def _bound(fn: Callable[..., Any], binding: tuple[Path, Path] | None) -> Callable[..., Any]:
    """``fn`` as the server registers it: a ``project`` or ``workspace`` parameter is registered
    without it, every call passing the server's own ``binding``, and an ``actor`` parameter is
    registered without it, every call passing ``None``, since no person makes an act an MCP client
    calls. With no binding, a call to a function taking a project or workspace raises
    :class:`NoProject` naming ``--project``."""
    sig = inspect.signature(fn, eval_str=True)
    bound = {"project", "workspace", "actor"} & sig.parameters.keys()
    if not bound:
        return fn

    @functools.wraps(fn)
    def entry(*args: Any, **kwargs: Any) -> Any:
        if binding is None:
            raise NoProject("this MCP server was started for no project, so no tool that acts on "
                            "one can run; restart it with --project <path> naming the project")
        values = {**dict(zip(("project", "workspace"), binding)), "actor": None}
        return fn(*args, **{name: values[name] for name in bound}, **kwargs)

    entry.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[p for name, p in sig.parameters.items() if name not in bound])
    return entry


def build_server(binding: tuple[Path, Path] | None) -> MCPServer:
    """A server registering every declared tool, each acting on the project of ``binding``
    (``(project, workspace)``, the workspace naming the backend that serves it) or, with
    ``None``, a server started for no project, which serves only the tools needing neither."""
    server = MCPServer("tcip-pipeline", lifespan=agent_identity.session_lifespan)
    # Records which harness connected, from the handshake, before any tool call is served.
    server.middleware.append(agent_identity.record_connecting_client)
    for fn, options in _TOOLS:
        server.add_tool(_bound(fn, binding), **options)
    return server


def list_registered_tools() -> list[str]:
    """Return the sorted names of all tools a server registers."""
    return sorted(fn.__name__ if "name" not in options else options["name"]
                  for fn, options in _TOOLS)


def main(argv: list[str] | None = None) -> None:
    """Start the MCP server on stdio for ``--project`` and the workspace ``TCIP_WORKSPACE`` names,
    or for no project and no workspace when ``--project`` is omitted. Exits naming the path when
    ``--project`` names a directory holding no project record; refuses what
    :func:`~tcip_mcp.workspace.workspace_from_environment` refuses when a project is named."""
    parser = argparse.ArgumentParser(prog="python -m tcip_mcp")
    parser.add_argument("--project", default=None,
                        help="the project every project tool acts on; omitted, only the tools "
                             "that need no project run")
    args = parser.parse_args(argv)

    from tcip_store import bind

    from tcip_mcp.workspace import workspace_from_environment

    bind()
    binding = None
    if args.project is not None:
        from tcip_mcp.project_record import existing_project

        try:
            binding = (existing_project(args.project), workspace_from_environment())
        except ValueError as exc:
            raise SystemExit(f"--project: {exc}") from exc
    # Size GDAL's block cache once per process, at the entry point, never at source construction.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    build_server(binding).run(transport="stdio")

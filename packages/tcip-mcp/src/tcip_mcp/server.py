"""MCP server entry point: every domain tool, served on stdio.

The server serves the project named at start (``--project <path>``), or, started for none, the
first project its creation door creates.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tcip_mcp import agent_identity
from tcip_mcp.workspace import BoundProject, workspace_from_environment

logger = logging.getLogger(__name__)

_Tool = TypeVar("_Tool", bound=Callable[..., Any])

_TOOLS: list[tuple[Callable[..., Any], str | None, bool]] = []
"""Every function a tool module declared with :func:`tool`, with its stated description and
whether it creates a project."""


class NoProjectError(ToolError):
    """A tool that acts on a project was called on a server started for none."""


def tool(
    *, description: str | None = None, creates_project: bool = False
) -> Callable[[_Tool], _Tool]:
    """Declare the decorated function an MCP tool and return it unchanged.

    ``description`` is passed to the server's own registration when :func:`build_server` builds
    a server; omitted, the registration reads the function's docstring. ``creates_project``
    marks the door whose answer, a project's ``project_path``, a server started for no project
    acts on from then on (:func:`_creating`).
    """

    def decorate(fn: _Tool) -> _Tool:
        _TOOLS.append((fn, description, creates_project))
        return fn

    return decorate


class Session:
    """The project a server's calls act on and the person they are made by, until bound none."""

    binding: BoundProject | None = None
    actor: str | None = None

    def bind(self, binding: BoundProject | None, actor: str | None) -> None:
        """Make ``binding`` the project and ``actor`` the person of every call from now on.

        ``actor`` is :func:`~tcip_mcp.identity.actor`'s spelling, ``None`` for calls no person
        makes, an MCP client's.
        """
        self.binding, self.actor = binding, actor


def binding_for(project: str | Path) -> BoundProject:
    """Return the binding of the project at ``project``.

    That is its directory, the id its record holds and the workspace ``TCIP_WORKSPACE`` names.
    Raises (``ValueError``) what :func:`~tcip_mcp.project_record.existing_project` and
    :func:`~tcip_mcp.workspace.workspace_from_environment` raise.
    """
    from tcip_mcp.project_record import existing_project

    root, record = existing_project(project)
    return BoundProject(root, record["id"], workspace_from_environment())


# Import tool modules to declare their handlers. A tool module that needs torch imports it inside
# its own functions, so every tool registers whether or not torch is installed.
import tcip_mcp.tools.annotation_tools  # noqa: F401, E402
import tcip_mcp.tools.calibration_tools  # noqa: F401, E402
import tcip_mcp.tools.data_tools  # noqa: F401, E402
import tcip_mcp.tools.delivery_tools  # noqa: F401, E402
import tcip_mcp.tools.doctor_tools  # noqa: F401, E402
import tcip_mcp.tools.experiment_tools  # noqa: F401, E402
import tcip_mcp.tools.feedback_tools  # noqa: F401, E402
import tcip_mcp.tools.gui_tools  # noqa: F401, E402
import tcip_mcp.tools.inference_tools  # noqa: F401, E402
import tcip_mcp.tools.ingest_tools  # noqa: F401, E402
import tcip_mcp.tools.knowledge_tools  # noqa: F401, E402
import tcip_mcp.tools.meta_tools  # noqa: F401, E402
import tcip_mcp.tools.model_tools  # noqa: F401, E402
import tcip_mcp.tools.orthomosaic_tools  # noqa: F401, E402
import tcip_mcp.tools.phenology_tools  # noqa: F401, E402
import tcip_mcp.tools.project_tools  # noqa: F401, E402
import tcip_mcp.tools.proposal_tools  # noqa: F401, E402
import tcip_mcp.tools.training_tools  # noqa: F401, E402
import tcip_mcp.tools.trait_tools  # noqa: F401, E402
import tcip_mcp.tools.vision_tools  # noqa: F401, E402

_CONTEXT = ("project", "bound", "actor")
"""The parameters the server supplies and no client states: ``project``, the bound project's
directory; ``bound``, the :class:`~tcip_mcp.workspace.BoundProject` whole; ``actor``, the
person the server's calls are made by, ``None`` for an MCP client's."""


def _bound(fn: Callable[..., Any], session: Session) -> Callable[..., Any]:
    """Return ``fn`` as the server registers it.

    Each :data:`_CONTEXT` parameter it takes is registered without it, every call passing the
    server's own value as ``session`` holds it at the call. While the session has no project, a
    call to a function taking ``project`` or ``bound`` raises :class:`NoProjectError` naming
    ``initialize_project`` and ``--project``.
    """
    sig = inspect.signature(fn, eval_str=True)
    bound = set(_CONTEXT) & sig.parameters.keys()
    if not bound:
        return fn

    @functools.wraps(fn)
    def entry(*args: object, **kwargs: object) -> object:
        binding = session.binding
        values: dict[str, object] = {"actor": session.actor}
        if binding is not None:
            values |= {"project": binding.root, "bound": binding}
        elif bound - values.keys():
            raise NoProjectError(
                "this MCP server has no project, so no tool that acts on one can run; "
                "create one with initialize_project, which this server then acts on, or "
                "restart it with --project <path> naming an existing one"
            )
        return fn(*args, **{name: values[name] for name in bound}, **kwargs)

    entry.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[
            p for name, p in sig.parameters.items() if name not in bound
        ]
    )
    return entry


def _creating(
    fn: Callable[..., dict], session: Session
) -> Callable[..., dict]:
    """Return the project-creation door ``fn`` as the server registers it.

    A creation answered without an ``error`` binds a session that has no project to the created
    one (:func:`binding_for`), and every answer that creates carries ``session_project``, the
    directory the server's project tools act on after the call; a session that has a project
    keeps it.
    """

    @functools.wraps(fn)
    def entry(*args: object, **kwargs: object) -> dict:
        result = fn(*args, **kwargs)
        if "error" in result:
            return result
        binding = session.binding
        if binding is None:
            binding = binding_for(result["project_path"])
            session.bind(binding, session.actor)
        return {**result, "session_project": str(binding.root)}

    return entry


def build_server(session: Session) -> MCPServer:
    """Return a server registering every declared tool, each call acting in ``session``.

    Each call acts on the project ``session`` is bound to at that call and is made by its person
    (:class:`Session`). With no project bound, only the tools taking neither ``project`` nor
    ``bound`` can run, and from the first project the creation door creates, every tool acts on
    that one.
    """
    server = MCPServer(
        "tcip-pipeline", lifespan=agent_identity.session_lifespan
    )
    # Records which harness connected, from the handshake, before any tool call is served.
    server.middleware.append(agent_identity.record_connecting_client)
    for fn, description, creates_project in _TOOLS:
        registered = _bound(fn, session)
        if creates_project:
            registered = _creating(registered, session)
        server.add_tool(registered, description=description)
    return server


def server_for(session: Session) -> MCPServer:
    """Bind this process's store and return the server acting in ``session``, not yet running.

    The process's GDAL cache is sized here, once, at the entry point.
    """
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache
    from tcip_store import bind

    bind()
    configure_gdal_cache()
    return build_server(session)


def project_binding(project: str | None) -> BoundProject | None:
    """Return the binding of the project an entry point's ``--project`` names, ``None`` for none.

    The binding is :func:`binding_for`'s, with the workspace ``TCIP_WORKSPACE`` names. Exits
    naming the path when ``project`` names a directory holding no project record; refuses what
    :func:`~tcip_mcp.workspace.workspace_from_environment` refuses.
    """
    if project is None:
        return None
    try:
        return binding_for(project)
    except ValueError as exc:
        raise SystemExit(f"--project: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """Start the server :func:`server_for` returns on stdio, bound to ``--project``, no person."""
    parser = argparse.ArgumentParser(prog="python -m tcip_mcp")
    parser.add_argument(
        "--project",
        default=None,
        help="the project every project tool acts on; omitted, only the tools "
        "that need no project run",
    )
    project = parser.parse_args(argv).project
    session = Session()
    server = server_for(session)
    session.bind(project_binding(project), None)
    server.run(transport="stdio")

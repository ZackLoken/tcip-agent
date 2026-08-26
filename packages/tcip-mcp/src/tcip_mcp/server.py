"""MCP server entry point: register all domain tools and run on stdio."""

from __future__ import annotations

import logging

from mcp.server import MCPServer

from tcip_mcp import agent_identity

mcp = MCPServer("tcip-pipeline", lifespan=agent_identity.session_lifespan)
# Records which harness connected, from the handshake, before any tool call is served.
mcp.middleware.append(agent_identity.record_connecting_client)
logger = logging.getLogger(__name__)

# Import tool modules to register their handlers with the server. A tool module that needs
# torch imports it inside its own functions, so every tool registers whether or not torch is installed.
import tcip_mcp.tools.data_tools  # noqa: F401, E402
import tcip_mcp.tools.project_tools  # noqa: F401, E402
import tcip_mcp.tools.ingest_tools  # noqa: F401, E402
import tcip_mcp.tools.experiment_tools  # noqa: F401, E402
import tcip_mcp.tools.meta_tools  # noqa: F401, E402
import tcip_mcp.tools.phenology_tools  # noqa: F401, E402
import tcip_mcp.tools.operationalization_tools  # noqa: F401, E402
import tcip_mcp.tools.trait_spec_authoring_tools  # noqa: F401, E402
import tcip_mcp.tools.scale_tools  # noqa: F401, E402
import tcip_mcp.tools.annotation_tools  # noqa: F401, E402
import tcip_mcp.tools.vision_tools  # noqa: F401, E402
import tcip_mcp.tools.feedback_tools  # noqa: F401, E402
import tcip_mcp.tools.training_tools  # noqa: F401, E402
import tcip_mcp.tools.inference_tools  # noqa: F401, E402
import tcip_mcp.tools.model_tools  # noqa: F401, E402
import tcip_mcp.tools.orthomosaic_tools  # noqa: F401, E402


def list_registered_tools() -> list[str]:
    """Return the sorted names of all tools currently registered on the server.

    This is the single source of truth for "how many tools are there": docs and
    tests read it instead of hard-coding a number.
    """
    manager = getattr(mcp, "_tool_manager", None)
    if manager is not None and hasattr(manager, "list_tools"):
        return sorted(t.name for t in manager.list_tools())
    # Fallback for SDK versions without the sync tool-manager accessor.
    import asyncio

    return sorted(t.name for t in asyncio.run(mcp.list_tools()))


def main() -> None:
    """Start the MCP server on stdio transport."""
    from tcip_store.binding import bind_default

    bind_default()
    # Pin the platform-state root before any tool resolves a ``.tcip`` path, so a server
    # launched from a subdir still agrees with the web backend on one root. ``set_active_project``
    # later repins this to the adopted project.
    from tcip_mcp.project_paths import pin_project_root

    pin_project_root()
    # Size GDAL's block cache once per process, at the entry point, never at source construction.
    from tcip_mcp.pipelines.raster_source import configure_gdal_cache

    configure_gdal_cache()
    mcp.run(transport="stdio")

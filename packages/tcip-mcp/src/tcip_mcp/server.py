"""MCP server entry point: register all domain tools and run on stdio."""

from __future__ import annotations

import logging

from mcp.server import MCPServer

mcp = MCPServer("tcip-pipeline")
logger = logging.getLogger(__name__)

# Import tool modules to register their handlers with the server.
# Non-torch tools always load; torch-dependent tools load when torch is available.
import tcip_mcp.tools.data_tools  # noqa: F401, E402
import tcip_mcp.tools.project_tools  # noqa: F401, E402
import tcip_mcp.tools.ingest_tools  # noqa: F401, E402
import tcip_mcp.tools.experiment_tools  # noqa: F401, E402
import tcip_mcp.tools.meta_tools  # noqa: F401, E402
import tcip_mcp.tools.phenology_tools  # noqa: F401, E402

try:
    import tcip_mcp.tools.annotation_tools  # noqa: F401, E402
except ImportError as e:
    logger.warning("Annotation tools unavailable: %s", e)

try:
    import tcip_mcp.tools.vision_tools  # noqa: F401, E402
except ImportError as e:
    logger.warning("Vision tools unavailable: %s", e)

try:
    import tcip_mcp.tools.feedback_tools  # noqa: F401, E402
except ImportError as e:
    logger.warning("Feedback tools unavailable: %s", e)

try:
    import tcip_mcp.tools.training_tools  # noqa: F401, E402
    import tcip_mcp.tools.inference_tools  # noqa: F401, E402
    import tcip_mcp.tools.model_tools  # noqa: F401, E402
    import tcip_mcp.tools.orthomosaic_tools  # noqa: F401, E402
except (ImportError, OSError) as e:
    logger.warning("Torch-dependent tools unavailable: %s", e)


def list_registered_tools() -> list[str]:
    """Return the sorted names of all tools currently registered on the server.

    This is the single source of truth for "how many tools are there": docs and
    tests read it instead of hard-coding a number. Note the count reflects what
    actually imported in this environment: torch-dependent tool modules only
    register when their dependencies are present (see the guarded imports above).
    """
    manager = getattr(mcp, "_tool_manager", None)
    if manager is not None and hasattr(manager, "list_tools"):
        return sorted(t.name for t in manager.list_tools())
    # Fallback for SDK versions without the sync tool-manager accessor.
    import asyncio

    return sorted(t.name for t in asyncio.run(mcp.list_tools()))


def main() -> None:
    """Start the MCP server on stdio transport."""
    # Pin the platform-state root before any tool resolves a ``.tcip`` path, so a server
    # launched from a subdir still agrees with the web backend on one root. ``set_active_project``
    # later repins this to the adopted project.
    from tcip_mcp.project_paths import pin_project_root

    pin_project_root()
    mcp.run(transport="stdio")

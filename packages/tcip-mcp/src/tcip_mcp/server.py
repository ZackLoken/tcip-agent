"""MCP server entry point — register all domain tools and run on stdio."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tcip-pipeline")
logger = logging.getLogger(__name__)

# Import tool modules to register their handlers with the server.
# Non-torch tools always load; torch-dependent tools load when torch is available.
import tcip_mcp.tools.data_tools  # noqa: F401, E402
import tcip_mcp.tools.project_tools  # noqa: F401, E402
import tcip_mcp.tools.experiment_tools  # noqa: F401, E402
import tcip_mcp.tools.meta_tools  # noqa: F401, E402

try:
    import tcip_mcp.tools.annotation_tools  # noqa: F401, E402
except ImportError as e:
    logger.warning("Annotation tools unavailable: %s", e)

try:
    import tcip_mcp.tools.vision_tools  # noqa: F401, E402
except ImportError as e:
    logger.warning("Vision tools unavailable: %s", e)

try:
    import tcip_mcp.tools.training_tools  # noqa: F401, E402
    import tcip_mcp.tools.inference_tools  # noqa: F401, E402
    import tcip_mcp.tools.model_tools  # noqa: F401, E402
    import tcip_mcp.tools.pipeline_tools  # noqa: F401, E402
    import tcip_mcp.tools.active_learning_tools  # noqa: F401, E402
except (ImportError, OSError) as e:
    logger.warning("Torch-dependent tools unavailable: %s", e)


def main() -> None:
    """Start the MCP server on stdio transport."""
    mcp.run(transport="stdio")

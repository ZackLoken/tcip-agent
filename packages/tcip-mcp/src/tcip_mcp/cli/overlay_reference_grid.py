r"""Render an image with a labeled reference-grid overlay for spatial referencing, from the command
line.

Wraps ``vision_tools.overlay_reference_grid``: square cells of ``--tile-size`` native pixels named
spreadsheet-style ('A1' top-left), rendered in yellow on the cells' true boundaries. Every response
echoes the full grid geometry (tile_size, overlap, cols, rows, width, height): pass the echoed
tile_size/overlap to ``propose_annotations(grid_cells=...)`` so a cell name resolves against the
grid that was actually rendered. --project is required: the artifact lands under it and a
relative --image lies under it.

Usage:
    tcip overlay-reference-grid --image <path> --project <project> \
        [--tile-size <native_pixels>] [--overlap 0.0]
"""

from __future__ import annotations

import argparse
import json

from tcip_mcp.cli import bound_project


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("--image", required=True,
                        help="The image file; a relative one is under --project.")
    parser.add_argument("--project", required=True, help="The project the artifact lands under.")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Cell edge in native pixels; omitted derives a legible default.")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Cell overlap as a fraction of tile_size, training tiling's "
                             "semantics.")
    args = parser.parse_args(argv)

    project = bound_project(args.project)

    from tcip_mcp.tools.vision_tools import overlay_reference_grid

    result = overlay_reference_grid(
        project, args.image, tile_size=args.tile_size, overlap=args.overlap
    )
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())

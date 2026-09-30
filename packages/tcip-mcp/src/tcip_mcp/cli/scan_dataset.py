"""Scan a folder for images, labels, and predictions.

The read-only census of what a dataset folder holds: image/label/prediction counts, and which files
are excluded from every bucket walk because their own stem or filename collides with a prediction
bucket's provenance stamp. Wraps ``tcip_mcp.tools.data_tools.scan_dataset``.

    tcip scan-dataset <folder_path>
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("folder_path", help="Dataset root directory to scan.")
    args = parser.parse_args(argv)

    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    from tcip_mcp.tools.data_tools import scan_dataset

    bind_default()

    result = scan_dataset(args.folder_path)
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

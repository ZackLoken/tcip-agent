"""Restamp a bare legacy dataset fingerprint onto the formula-version-prefixed form.

``dataset_fingerprint`` now returns ``v<n>:<hex>`` rather than a bare hex, so a fingerprint a
dataset registered before this family recorded still carries the old bare form in
``dataset.json`` and the project's ``.tcip/datasets.json``. This script compares the bare hex
this code recomputes right now under the current formula to the bare value already recorded:
equal, it re-registers through ``register_dataset``'s own path, which stamps the prefixed form
the writer spells today; unequal, it refuses and reports the mismatch rather than papering over
it, since a formula match cannot be assumed and papering over a real content change would carry
a stale identity forward as if nothing had changed. A dataset already carrying a prefixed
fingerprint is left alone: nothing here to restamp.

    python scripts/restamp_dataset_fingerprint.py <dataset_root> [--project <project_root>]

Never a runtime migration: the writer already produces the prefixed form, so this exists only
to carry an already-registered dataset onto it. Run ``export_store.py`` on the affected project
afterwards, and ``doctor.py`` to confirm.

Exit codes: 0 restamped or nothing to do, 1 the dataset is unregistered, 2 the recorded bare
fingerprint disagrees with the current recompute (refused, not restamped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tcip_store.binding import bind_default  # noqa: E402

from tcip_mcp.dataset_layout import require_dataset_identity  # noqa: E402
from tcip_mcp.pipelines.resolution import (  # noqa: E402
    dataset_fingerprint,
    fingerprint_formula_version,
)
from tcip_mcp.tools.project_tools import register_dataset  # noqa: E402


def restamp(dataset_root: str, project_root: str | None = None) -> int:
    """Restamp one dataset's fingerprint if it is a bare legacy value that still matches the
    current recompute; refuse and report if it does not. Returns the exit code to use."""
    root = Path(dataset_root).absolute()
    try:
        identity = require_dataset_identity(root)
    except ValueError as exc:
        print(f"UNREGISTERED: {exc}")
        return 1

    recorded = identity.get("fingerprint")
    if fingerprint_formula_version(recorded) is not None:
        print(f"OK: {root} already carries a formula-versioned fingerprint ({recorded!r}); "
              "nothing to restamp")
        return 0

    current = dataset_fingerprint(root)
    if current is None:
        print(f"UNCOMPUTABLE: {root} has no images or labels to fingerprint right now, so this "
              "run cannot verify the recorded value before restamping it")
        return 2
    current_bare = current.split(":", 1)[1]
    if recorded != current_bare:
        print(f"MISMATCH: {root} recorded bare fingerprint {recorded!r} does not match the "
              f"current recompute {current_bare!r}; refusing to restamp over a real content "
              "change. Nothing was written.")
        return 2

    result = register_dataset(str(root), identity.get("crop", ""), project_root or str(root))
    if "error" in result:
        print(f"REFUSED: {root} could not be re-registered: {result['error']}")
        return 2
    print(f"RESTAMPED: {root} fingerprint {recorded!r} -> {current!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_root")
    ap.add_argument("--project", default=None, help="project root to register under "
                     "(default: dataset_root)")
    args = ap.parse_args()

    bind_default()

    return restamp(args.dataset_root, args.project)


if __name__ == "__main__":
    sys.exit(main())

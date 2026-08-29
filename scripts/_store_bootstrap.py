"""Operator-script entry point for the store catalogue and a project's own roots.

The catalogue import itself (every module that registers a store) lives in
``tcip_mcp.store_catalogue``, package-only so :func:`tcip_mcp.tools.bundle.account_for` reaches
it without the repo root on ``sys.path``. This module re-exports :func:`bootstrapped_stores` from
there, so the operator scripts that import it (``export_store.py``, ``adopt_store.py``) keep
working unchanged, and adds :func:`project_roots`, which needs ``project_tools`` directly rather
than the whole catalogue.
"""

from __future__ import annotations

import os
from pathlib import Path

from tcip_store.layout_claims import EXPERIMENTS, ROOT, STATE

from tcip_mcp.store_catalogue import bootstrapped_stores  # noqa: F401
from tcip_mcp.tools import project_tools


def project_roots(project_root: str | Path) -> tuple[tuple[str, str], ...]:
    """The roots a whole project's records live in, each with the layout it is.

    The registered dataset roots come from the project's own registry rather than from a
    directory guess, so a dataset that lives outside the project tree still travels.
    """
    root = Path(project_root).absolute()
    roots: list[tuple[str, str]] = [
        (str(root), ROOT),
        (str(root / ".tcip" / "state"), STATE),
        (str(root / ".tcip" / "experiments"), EXPERIMENTS),
    ]
    seen = {os.path.normcase(str(root))}
    for entry in project_tools.read_datasets(root):
        if not entry.get("path"):
            continue
        dataset_root = project_tools.dataset_entry_path(root, entry).absolute()
        if os.path.normcase(str(dataset_root)) in seen:
            continue
        seen.add(os.path.normcase(str(dataset_root)))
        roots.append((str(dataset_root), ROOT))
        roots.append((str(dataset_root / ".tcip" / "state"), STATE))
    return tuple(roots)

"""Shared platform-state-root pinning for the operator scripts that call an ``@audited`` tool
function directly, outside the MCP server or web backend.

Neither entry point pins ``$TCIP_PROJECT_ROOT`` for a bare script, so left alone
``tcip_mcp.project_paths.resolve_state`` and the audit log both fall back to the process cwd: a
script run from any directory other than the project would read state, and write its audit line
and a fresh ``.tcip/store.db``, wherever the operator happened to be standing rather than under
the project. :func:`pin_project_root` resolves the root from the script's own explicit argument
or ``$TCIP_PROJECT_ROOT``, refuses naming both when neither is set, and pins the environment
variable before the caller imports or calls its tool function, so that resolution and every
later one in the process, including a store bound after this call, land under the project.
"""

from __future__ import annotations

import os
from pathlib import Path


def pin_project_root(explicit: str | None) -> Path:
    """Resolve and pin ``$TCIP_PROJECT_ROOT`` to an absolute path, or refuse.

    ``explicit`` is the script's own project-root argument, when the operator passed one;
    an already-set ``$TCIP_PROJECT_ROOT`` is the fallback. Refuses, naming both, when neither
    names a root, rather than silently defaulting to the current directory.
    """
    from tcip_mcp.project_paths import ENV_VAR

    root = explicit or os.environ.get(ENV_VAR)
    if not root:
        raise SystemExit(
            f"no project root: pass --project <path> or set ${ENV_VAR} before running this "
            "script, so its audit line and any state it reads or writes land under the "
            "project rather than the current directory."
        )
    resolved = Path(root).resolve()
    os.environ[ENV_VAR] = str(resolved)
    return resolved

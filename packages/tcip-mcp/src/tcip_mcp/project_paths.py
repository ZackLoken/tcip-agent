"""Stable resolution of the platform state root, independent of a process's cwd.

Durable *platform* state, the platform's own audit log, the experiment store, and the model
registry, anchors here so a whole project is self-contained under one ``<root>/.tcip/``.

Resolution order (``platform_state_root`` / ``resolve_state``, evaluated at use time):
  1. ``$TCIP_STATE_ROOT`` if set.
  2. otherwise the current working directory, the historical default, so nothing changes
     for tests or an un-pinned run.

A long-lived process binds the variable once at startup, through :func:`pin_platform_root`.
A process that opts in (``from_marker=True``: the web backend always, the MCP server inside
the platform's own agent terminal) binds from the workspace's active-project marker when one
names an adoptable project, else keeps whatever it inherited, else the repo root. A process
that does not opt in keeps the historical ``setdefault``: the inherited variable, else the repo
root. Either way the decision is recorded in a :class:`RootBinding`, kept module-level and
returned by :func:`root_binding`, since no process in this repo configures logging and an info
line would otherwise reach nothing; ``inspect_project`` and the workspace projects list route
report it.

Adopting a project (``workspace.activate_project``) *repins* the adopting process's own
variable to ``<workspace>/<project>`` through :func:`repin_platform_root`, so the platform's own
audit log (now this project's, one file at one key), experiments, and registry all land under
that project from then on; a training run in flight
keeps writing to the root it started under (the launch snapshots it once) until it is
deliberately adopted. The repin is explicit, never a passive marker read: no operation other
than an adopt itself changes a running process's root, so the window between one process
adopting and another converging is stated rather than closed here.

Data-side project state (images, ``gui.json``, reports, retrospectives) is addressed by an
explicit ``project_path`` (the workspace project); after adoption the platform root equals
that project, so the two coincide.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ENV_VAR = "TCIP_STATE_ROOT"


def platform_state_root() -> Path:
    """The platform state root: ``$TCIP_STATE_ROOT`` if set, else the current directory."""
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else Path.cwd()


def require_and_pin_platform_root(explicit: str | None) -> Path:
    """Resolve and pin ``$TCIP_STATE_ROOT`` to an absolute path, or refuse.

    For an operator command that calls an ``@audited`` tool function directly, outside the MCP
    server or web backend, so its resolution and its audit line land under the intended project
    root rather than the process cwd. Neither entry point pins ``$TCIP_STATE_ROOT`` for a bare
    command, so left alone :func:`resolve_state` and the audit log both fall back to the process
    cwd: a command run from any directory other than the target project would read state, and
    write its audit line and a fresh ``.tcip/store.db``, wherever the operator happened to be
    standing rather than under the platform state root.

    ``explicit`` is the command's own project-root argument, when the operator passed one; an
    already-set ``$TCIP_STATE_ROOT`` is the fallback. Refuses, naming both, when neither names a
    root, rather than silently defaulting to the current directory. Pins the environment variable
    before the caller imports or calls its tool function, so that resolution and every later one
    in the process, including a store bound after this call, land under the platform state root.
    """
    root = explicit or os.environ.get(ENV_VAR)
    if not root:
        raise SystemExit(
            f"no platform state root: pass --project <path> or set ${ENV_VAR} before running "
            "this command, so its audit line and any state it reads or writes land under the "
            "platform state root rather than the current directory."
        )
    resolved = Path(root).resolve()
    os.environ[ENV_VAR] = str(resolved)
    return resolved


def resolve_output_path(path: "str | Path") -> Path:
    """An output-artifact path (weights, prediction buckets, delivery CSVs, curated datasets)
    anchored to the platform state root.

    An absolute path is the caller's own explicit choice and is returned unchanged. A relative
    path resolves against :func:`platform_state_root`, never the process cwd: the MCP server and
    the web backend both run with a cwd unrelated to any project (the repo root), so a relative
    output path means inside the project, not wherever the server process happens to have been
    launched.
    """
    p = Path(path)
    return p if p.is_absolute() else platform_state_root() / p


def resolve_state(path: Path) -> Path:
    """Resolve a platform-state path against the pinned root, at use time.

    - An already-absolute ``path`` is returned unchanged (e.g. a test that rebinds
      ``AUDIT_PATH``/``EXPERIMENTS_DIR`` to a tmp dir).
    - A relative ``path`` is prefixed with ``$TCIP_STATE_ROOT`` when pinned, so a process
      launched from a subdir still writes to the one platform ``.tcip/``.
    - When unpinned, a relative ``path`` is returned as-is → resolved against the current
      directory at use, preserving the historical default (and per-test cwd isolation).
    """
    if path.is_absolute():
        return path
    override = os.environ.get(ENV_VAR)
    return Path(override) / path if override else path


def resolve_state_or(path: Path, fallback: Path) -> Path:
    """The pinned resolution of a relative platform-state path, or the caller's own fallback.

    ``$TCIP_STATE_ROOT`` set: the same resolution :func:`resolve_state` gives. Unset: returns
    ``fallback`` rather than ``path`` unchanged, for a caller (e.g. an unpinned render cache) that
    needs somewhere real to write when there is no pinned project, not a path resolved against
    whatever the process cwd happens to be.
    """
    if os.environ.get(ENV_VAR):
        return resolve_state(path)
    return fallback


def repo_root_from_here() -> Path:
    """The repo root inferred from this file's location: the nearest ancestor holding
    ``.mcp.json``, checked across every ancestor before falling back to the nearest one holding
    ``CLAUDE.md``. Stable regardless of cwd; used to *pin* the env var.

    ``.mcp.json`` lives only at the true repo root, so it is checked in a full pass first; each
    package under ``packages/`` carries its own ``CLAUDE.md`` too, so checking both markers in a
    single climb would stop at a package's own file instead of continuing to the repo root.
    """
    here = Path(__file__).resolve()
    parents = list(here.parents)
    for parent in parents:
        if (parent / ".mcp.json").is_file():
            return parent
    for parent in parents:
        if (parent / "CLAUDE.md").is_file():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class RootBinding:
    """How a process's platform-state root was decided by :func:`pin_platform_root`, or set
    since by :func:`repin_platform_root`.

    ``source`` is ``"marker"`` (the workspace's active-project marker supplied the root),
    ``"inherited"`` (the process kept a root it was launched with), ``"repo_root"`` (neither
    was available), or ``"adopted"`` (a later :func:`repin_platform_root` call, an explicit
    adopt rather than the startup bind). ``inherited_root`` is the variable's value before this
    bind, ``None`` when nothing was inherited, kept so an overridden value stays visible; for
    an ``"adopted"`` binding it is the root this process was pinned to just before the repin.
    ``marker_problem`` is ``None`` when the process either did not consult the marker or the
    marker named no project; otherwise the text of why the marker could not be used: a store
    refusal, a lock timeout, or a name :func:`tcip_mcp.workspace.adoptable_project_root`
    refuses to open. Always ``None`` on an ``"adopted"`` binding, since a repin only ever
    follows an adopt that already resolved a real root.
    """

    root: Path
    source: str
    inherited_root: Optional[str]
    marker_problem: Optional[str]


_binding: Optional[RootBinding] = None


def root_binding() -> Optional[RootBinding]:
    """This process's :class:`RootBinding`, or ``None`` before either :func:`pin_platform_root`
    or :func:`repin_platform_root` has run: every test (until it calls ``activate_project``,
    which repins) and any other standalone use, since none of those call either."""
    return _binding


def restore_binding(binding: Optional[RootBinding]) -> None:
    """Restore this process's :class:`RootBinding` to a snapshot a caller took earlier.

    For a caller that must undo its own :func:`pin_platform_root`/:func:`repin_platform_root`
    calls once it is done, the way a test's fixture restores the ``$TCIP_STATE_ROOT``
    environment variable around a test that adopts a project.
    """
    global _binding
    _binding = binding


def repin_platform_root(root: Path) -> None:
    """Set ``$TCIP_STATE_ROOT`` to ``root`` and record the change as an ``"adopted"``
    :class:`RootBinding`.

    The one writer of the variable: :func:`pin_platform_root`'s startup bind,
    ``workspace.activate_project``'s adopt, and the web backend's repin on the agent's
    adopt signal all go through here, so there is one place that changes it, and
    :func:`root_binding` reports the current root immediately after any of them.
    """
    global _binding
    inherited_root = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = str(root)
    _binding = RootBinding(
        root=Path(root), source="adopted", inherited_root=inherited_root, marker_problem=None
    )


def pin_platform_root(*, from_marker: bool) -> RootBinding:
    """Bind this process's platform-state root at startup and record how it was decided.

    ``from_marker=True`` (the web backend always; the MCP server inside the platform's own
    agent terminal, see ``server.binds_from_marker``): the workspace's active-project
    marker's project, when :func:`tcip_mcp.workspace.active_project_if_present` (with
    ``create=False``) finds one adoptable; otherwise the inherited variable, when set;
    otherwise :func:`repo_root_from_here`. A store refusal, a lock timeout, or a marker
    naming a project that is not adoptable is caught into the returned binding's
    ``marker_problem`` rather than raised: the process must start regardless.

    ``from_marker=False`` (every other process): the inherited variable, else the repo root,
    the historical ``setdefault`` behaviour, now recorded rather than only applied.

    Call once, after the process has bound a storage backend (a marker read needs one) and
    before anything resolves a ``.tcip`` path. Returns the :class:`RootBinding`, which
    :func:`root_binding` also keeps for later reporting.
    """
    global _binding
    inherited = os.environ.get(ENV_VAR)
    marker_problem: Optional[str] = None
    root: Optional[Path] = None
    source = ""

    if from_marker:
        from tcip_mcp import workspace

        try:
            found = workspace.active_project_if_present(create=False)
        except Exception as exc:  # noqa: BLE001 - a store refusal or lock timeout; the process starts anyway
            found = None
            marker_problem = str(exc)
        if found is not None:
            _, root = found
            source = "marker"
        elif marker_problem is None:
            marker_problem = workspace.marker_problem(create=False)

    if root is None:
        if inherited:
            root, source = Path(inherited), "inherited"
        else:
            root, source = repo_root_from_here(), "repo_root"

    binding = RootBinding(
        root=root, source=source, inherited_root=inherited, marker_problem=marker_problem
    )
    repin_platform_root(root)
    _binding = binding
    return binding

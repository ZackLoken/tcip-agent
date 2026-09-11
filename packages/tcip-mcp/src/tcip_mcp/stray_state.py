"""What a stray file under a project's ``.tcip/state`` root is, and whether one path may be
deleted: one predicate, read by the doctor's own listing and by :func:`delete_stray_state_file`
(``tools/project_tools.py``), so the two cannot disagree (CLAUDE.md: when two code paths must
agree, call one from the other).

A stray is a file under ``<project>/.tcip/state`` that
:func:`~tcip_mcp.tools.bundle.account_for`'s layout-aware accounting calls unaccounted: no
store's own template claims it (``tcip_store.layout_claims.claimants_of`` is not used here, since
it filters by no layout and would call the commonest stray, a loose ``.json`` directly under
``.tcip/state``, claimed by a one-segment wildcard template meant for a different root entirely),
it is not the storage backend's own bookkeeping (a lock file, a temp file, the database and its
WAL sidecars), and it is not a recognized blob. The state root's own database
(``<project>/.tcip/state/.tcip/store.db``) is excluded a second way, by location, since nothing
under it is ever a stray whatever the accounting says.

``project_root`` is resolved (:meth:`Path.resolve`) before the accounting runs, and every
membership test below compares :func:`os.path.normcase` of the target against the same
normalization of the accounting's own paths, exactly the rule ``account_for`` applies internally:
a case-different or junction-aliased ``project_root`` spelling cannot make a real stray misclassify
as claimed, or a claimed file misclassify as a stray. The target itself is joined onto that
resolved root and normalized lexically rather than resolved, so the path classified is the path
the caller named; a link on any segment below the state root is refused, never followed.

The accounting itself can refuse: :class:`~tcip_mcp.tools.bundle.AnchorMisplaced` (a split or
curated manifest sitting somewhere the derivation excludes) and ``tcip_store.StoreError`` (one
root's own plan in which two stores claim the same file, from
:func:`~tcip_store.adoption.plan_root`). Both are caught here and
answered as a refusal naming the exception, never left to escape as a traceback: a project whose
tree the accounting cannot classify at all should not crash a tool call or a doctor run over it,
and the caller already has a real door (fixing the misplaced anchor, or the layout claim) to clear
before either of those two would recur.

``accounting.collisions`` is a different fact and nothing here reads it: it holds a path that two
derived roots' separate plans each claim, each unambiguously, rather than the two-stores-in-one-plan
case the paragraph above names. Such a path needs no branch of its own, since it is present in
more than one plan's entries and the same "which plan's entries name this path" walk that answers
a plainly claimed file finds it too, refusing under whichever store the iteration reaches first.
Stated here rather than left to look arbitrary: a path two roots claim is exactly as unsafe to
delete under either name, so naming only one of them is not a defect in the refusal, only in the
claim tables that let two roots agree on one file.

No staleness gate is taken over this predicate: see :func:`~tcip_mcp.cli.doctor.check_stray_state_files`'s
own docstring for why.
"""

from __future__ import annotations

import os
from pathlib import Path


def _under_normcased(path: Path, root: Path) -> bool:
    """Whether ``path`` is ``root`` itself or sits under it, compared by
    :func:`os.path.normcase` rather than :meth:`Path.relative_to`, the same comparison
    ``bundle.account_for`` applies to its own classification, so a case-different root spelling
    cannot make this disagree with the accounting."""
    p, r = os.path.normcase(str(path)), os.path.normcase(str(root))
    return p == r or p.startswith(r + os.sep)


def stray_state_files(project_root: "Path | str") -> tuple[Path, ...]:
    """Every stray file under ``project_root``'s ``.tcip/state``: a path
    :func:`~tcip_mcp.tools.bundle.account_for` calls unaccounted, that lies under the state root
    and not under the state root's own ``.tcip`` (its database home).

    Raises :class:`~tcip_mcp.tools.bundle.AnchorMisplaced` or ``tcip_store.StoreError`` when the
    accounting itself refuses; callers that must never raise (the doctor) catch these themselves,
    the same way :func:`stray_state_file_refusal` does for a single path.
    """
    from tcip_mcp.tools import bundle

    resolved_root = Path(project_root).resolve()
    accounting = bundle.account_for(resolved_root)
    state_root = resolved_root / ".tcip" / "state"
    database_home = state_root / ".tcip"
    return tuple(
        path for path in accounting.unaccounted
        if _under_normcased(path, state_root) and not _under_normcased(path, database_home)
    )


def _is_link_or_junction(path: Path) -> bool:
    """A symbolic link (POSIX) or a Windows junction/reparse point: never deleted by name here,
    since deleting the link entry while the accounting classified its target is not the single,
    unambiguous act this door promises."""
    import stat

    try:
        st = os.lstat(path)
    except OSError:
        return False
    if getattr(st, "st_file_attributes", 0) & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
        return True
    return stat.S_ISLNK(st.st_mode)


def _first_link_on(target: Path, state_root: Path) -> "Path | None":
    """The first link or junction on ``target``'s own path below ``state_root``, target included,
    or ``None`` when every segment is a real directory or file.

    Walked segment by segment rather than compared against :meth:`Path.resolve`'s answer, because
    resolving is what must not happen: a resolved path names the file a link points at, and this
    door deletes only the path its caller named and its audit line records.
    """
    walked = state_root
    for segment in target.relative_to(state_root).parts:
        walked = walked / segment
        if _is_link_or_junction(walked):
            return walked
    return None


def stray_state_file_refusal(
    project_root: "Path | str", relative_path: str,
) -> "tuple[Path, str | None]":
    """The reason ``<project_root>/.tcip/state/<relative_path>`` may not be deleted, or ``None``
    when it is a stray and the deletion may proceed; always paired with the resolved target path.

    The target path is joined onto the state root and normalized lexically
    (:func:`os.path.normpath`), never resolved: :meth:`Path.resolve` answers the file a link
    points at, and this door acts on the path its caller named and its audit line records, so a
    link anywhere on that path is refused instead of followed.

    Checked in this order: an empty, absolute (either grammar,
    :func:`~tcip_mcp.registry_paths.is_external_form`) or ``..``-carrying ``relative_path``, or one
    that normalizes outside the state root (traversal, so a ``../hpo/<study>/result.json`` never
    reaches the accounting); a path under the state root's own ``.tcip`` (the database home,
    named as such); a link or junction on any segment below the state root, the target itself
    included (:func:`_first_link_on`); a path that does not exist; a directory; then the
    accounting itself (see the module docstring for its two caught raises): bookkeeping refuses as
    the backend's own artifact, a path in a plan's entries refuses naming that entry's store, a
    recognized blob refuses as a blob home's, and a path in ``accounting.unaccounted`` is a stray
    and answers ``None``.
    """
    from tcip_mcp.registry_paths import is_external_form
    from tcip_mcp.tools import bundle

    resolved_root = Path(project_root).resolve()
    state_root = resolved_root / ".tcip" / "state"
    database_home = state_root / ".tcip"

    if not relative_path or is_external_form(relative_path) or ".." in Path(relative_path).parts:
        return state_root, (
            f"{relative_path!r} is not a plain path relative to the state root (empty, absolute, "
            "or carrying a .. segment); name a path under .tcip/state with no traversal")

    target = Path(os.path.normpath(str(state_root / relative_path)))
    if not _under_normcased(target, state_root):
        return target, (
            f"{relative_path!r} normalizes to {target}, outside the state root {state_root}; "
            "refusing to act outside .tcip/state")

    if _under_normcased(target, database_home):
        return target, f"{target} is under the state root's own database home, never a stray file"

    link = _first_link_on(target, state_root)
    if link is not None:
        return target, (
            f"{link} is a link or junction, refused rather than followed: the file it points at "
            f"is not the path this call names, so deleting through it would delete something the "
            f"response and the audit line do not name. Name a path whose every segment under "
            f".tcip/state is a real directory or file")

    if not os.path.lexists(target):
        return target, f"{target} does not exist"

    if target.is_dir():
        return target, f"{target} is a directory; this door deletes one file at a time"

    from tcip_store.errors import StoreError

    try:
        accounting = bundle.account_for(resolved_root)
    except (bundle.AnchorMisplaced, StoreError) as exc:
        return target, f"the state root's accounting refused: {exc}"

    marker = os.path.normcase(str(target))
    if any(os.path.normcase(str(path)) == marker for path in accounting.bookkeeping):
        return target, f"{target} is the storage backend's own artifact, never a stray"

    for plan in accounting.plans:
        for entry in plan.entries:
            if os.path.normcase(str(entry.path)) == marker:
                return target, (
                    f"{target} belongs to the {entry.store!r} store; use that store's own doors "
                    "(tcip export-store / tcip adopt-store), never a direct delete")

    if any(os.path.normcase(str(path)) == marker for path in accounting.blobs):
        return target, f"{target} is a recognized blob (imagery, labels, a checkpoint, ...), never a stray"

    return target, None

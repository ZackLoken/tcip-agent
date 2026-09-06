"""The browser and the backend meet on one declaration of the dataset's own vocabulary.

Two facts the backend owns reach the browser as literals: the statuses the image-status store
holds, and the extension a per-image label or prediction record is written under. Both are stated
once on each side, and these tests hold the two sides equal, so a token added or renamed in
``dataset_layout`` cannot leave the browser filtering on a value the store never records.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO_ROOT / "packages" / "tcip-web" / "frontend" / "src"
STATUS_DECLARATION = FRONTEND_SRC / "api" / "classes.ts"
PATH_DECLARATION = FRONTEND_SRC / "lib" / "paths.ts"

_STATUS_UNION_RE = re.compile(r"export type ImageStatus = ([^;]+);")
_RECORD_EXT_RE = re.compile(r'export const RECORD_EXT = "([^"]+)";')
_FINISHED_STATUSES_RE = re.compile(
    r"export const FINISHED_STATUSES: readonly ImageStatus\[\] = \[([^\]]+)\];"
)


def _statuses() -> tuple[str, ...]:
    from tcip_mcp.dataset_layout import IMAGE_STATUSES

    return IMAGE_STATUSES


def test_the_browsers_finished_statuses_are_the_ones_dataset_layout_declares() -> None:
    """The frontend's own FINISHED_STATUSES declaration, parsed independently of the wider
    ImageStatus union check above, holds equal to dataset_layout.FINISHED_STATUSES (coverage)."""
    from tcip_mcp.dataset_layout import FINISHED_STATUSES

    match = _FINISHED_STATUSES_RE.search(STATUS_DECLARATION.read_text(encoding="utf-8"))
    assert match is not None, "FINISHED_STATUSES is no longer where this test reads it"
    declared = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    assert declared == FINISHED_STATUSES


def _frontend_sources(include_tests: bool) -> list[Path]:
    files = list(FRONTEND_SRC.rglob("*.ts")) + list(FRONTEND_SRC.rglob("*.tsx"))
    if include_tests:
        return sorted(files)
    return sorted(p for p in files if ".test." not in p.name and "test" not in p.parent.name)


def test_the_browsers_image_statuses_are_the_ones_the_store_records() -> None:
    """The union the browser types against is the store's vocabulary, in the same order."""
    match = _STATUS_UNION_RE.search(STATUS_DECLARATION.read_text(encoding="utf-8"))
    assert match is not None, "the ImageStatus union is no longer where this test reads it"
    declared = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    assert declared == _statuses()


def test_the_browser_names_a_record_the_way_the_resolver_does() -> None:
    """The extension the browser appends to a stem is the one the label resolver would have used."""
    from tcip_mcp.dataset_layout import label_ext

    match = _RECORD_EXT_RE.search(PATH_DECLARATION.read_text(encoding="utf-8"))
    assert match is not None, "RECORD_EXT is no longer where this test reads it"
    assert match.group(1) == label_ext("json")


def test_no_other_frontend_module_restates_the_status_vocabulary() -> None:
    """One declaration in the browser, so the backend has one counterpart to stay equal to.

    A second copy of the union is what drifts: a status added to the store reaches the module that
    was updated and silently narrows every module that was not. Frontend test files are left out,
    since a union written there is the expectation being asserted, not a declaration.
    """
    statuses = _statuses()
    offenders: list[str] = []
    for source in _frontend_sources(include_tests=False):
        if source == STATUS_DECLARATION:
            continue
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            quoted = re.findall(r'"([^"]+)"', line)
            if len(set(quoted) & set(statuses)) >= 3 and "|" in line:
                offenders.append(f"{source.relative_to(REPO_ROOT).as_posix()}:{line_no}")
    assert not offenders, (
        "these restate the image-status union instead of importing ImageStatus:\n"
        + "\n".join(offenders)
    )


def test_no_frontend_module_composes_a_dataset_path_of_its_own() -> None:
    """Outside the one path module, the browser builds no dataset path from the tree's layout.

    The backend resolves every directory the browser is handed; a module that spells the images
    tree itself is a second copy of the layout, which is what drifts when the resolver changes.
    """
    layout_re = re.compile(r"\$\{[^}]*\}/(?:images|annotations|predictions)/")
    offenders: list[str] = []
    for source in _frontend_sources(include_tests=False):
        if source == PATH_DECLARATION:
            continue
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if layout_re.search(line):
                offenders.append(f"{source.relative_to(REPO_ROOT).as_posix()}:{line_no}")
    assert not offenders, (
        "these compose a dataset path instead of using the dirs the backend resolved:\n"
        + "\n".join(offenders)
    )

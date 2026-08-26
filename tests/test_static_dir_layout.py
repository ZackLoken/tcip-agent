"""Where Vite writes the built frontend and where the backend looks for it must agree.

``_find_static_dir`` in ``app.py`` prefers a packaged copy (``tcip_web/static/``) over the
src-layout checkout (``packages/tcip-web/static/``) when both exist, so a stale packaged copy
left over from an earlier wheel build would shadow the directory ``npm run build`` just wrote. A
source checkout never carries the packaged candidate, so the two cannot collide there.
"""

from __future__ import annotations

import re
from pathlib import Path

from tcip_web.app import _static_dir_candidates

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "packages" / "tcip-web" / "frontend"
VITE_CONFIG = FRONTEND_DIR / "vite.config.ts"
GITIGNORE = REPO_ROOT / ".gitignore"
PRETTIERIGNORE = FRONTEND_DIR / ".prettierignore"

_OUT_DIR_RE = re.compile(r'outDir:\s*["\']([^"\']+)["\']')


def _vite_out_dir() -> Path:
    """The directory Vite's ``outDir`` literal names, resolved against the frontend directory."""
    text = VITE_CONFIG.read_text(encoding="utf-8")
    match = _OUT_DIR_RE.search(text)
    assert match is not None, f"no outDir literal found in {VITE_CONFIG}"
    return (FRONTEND_DIR / match.group(1)).resolve()


def _packaged_static_dir() -> Path:
    return _static_dir_candidates()[0].resolve()


def _src_layout_static_dir() -> Path:
    return _static_dir_candidates()[-1].resolve()


def test_vite_outdir_is_the_src_layout_static_candidate():
    assert _vite_out_dir() == _src_layout_static_dir()


def test_gitignores_frontend_build_entries_resolve_to_the_same_static_dir():
    static_dir = _src_layout_static_dir()
    text = GITIGNORE.read_text(encoding="utf-8")
    candidates = [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    entries = [
        entry for entry in candidates
        if (resolved := (REPO_ROOT / entry.rstrip("/")).resolve()) == static_dir
        or static_dir in resolved.parents
    ]
    assert len(entries) == 4, entries


def test_prettierignore_static_entry_resolves_to_the_same_static_dir():
    text = PRETTIERIGNORE.read_text(encoding="utf-8")
    entries = [line.strip() for line in text.splitlines() if line.strip() == "../static"]
    assert len(entries) == 1, entries
    resolved = (FRONTEND_DIR / entries[0]).resolve()
    assert resolved == _src_layout_static_dir()


def test_packaged_static_candidate_does_not_exist_in_a_source_checkout():
    # A stale packaged copy would be served ahead of the directory Vite writes, so a source
    # checkout must never carry one; only a wheel build places files there.
    assert not (_packaged_static_dir() / "index.html").exists()

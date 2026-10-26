"""The frontend's source directory is the one its ``tsconfig.json`` includes.

:data:`tests.FRONTEND_SRC` reads it from the frontend's ``tsconfig.json``, as TypeScript, Vite and
the frontend's Node-side configuration do. Two further statements of it are fixed by their
formats: the ``@/*`` alias in the same file's ``paths``, and the entry module ``index.html``
loads. These tests hold both to the directory.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import PurePosixPath

from tests import FRONTEND, FRONTEND_SRC


class _ModuleScripts(HTMLParser):
    """The ``src`` of every ``<script type="module">`` in the page fed to it."""

    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "script" and ("type", "module") in attrs:
            self.sources.extend(
                value
                for name, value in attrs
                if name == "src" and value is not None
            )


def test_the_alias_names_the_source_directory() -> None:
    """``tsconfig.json``'s ``@/*`` alias maps onto every module under the directory."""
    tsconfig = json.loads(
        (FRONTEND / "tsconfig.json").read_text(encoding="utf-8")
    )
    assert tsconfig["compilerOptions"]["paths"]["@/*"] == [
        str(PurePosixPath(FRONTEND_SRC.relative_to(FRONTEND).as_posix(), "*"))
    ]


def test_the_page_entry_module_lies_in_the_source_directory() -> None:
    """Each module ``index.html`` loads, resolved against the page's root, lies in it."""
    page = _ModuleScripts()
    page.feed((FRONTEND / "index.html").read_text(encoding="utf-8"))
    assert page.sources, "index.html loads no module script"
    for source in page.sources:
        entry = FRONTEND / PurePosixPath("/", source).relative_to("/")
        assert entry.resolve().is_relative_to(FRONTEND_SRC.resolve()), source

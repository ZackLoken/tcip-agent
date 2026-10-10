"""Write every record and log a project's store databases hold as files a person can read.

Read-only: each database under the project is read through the store's own readers. A key's
record lands at ``<out_dir>/<root>/<store>/<part>/.../<last part>.json`` and its log entries at
the same path ending ``.jsonl``, a key with no parts at ``<out_dir>/<root>/<store>.json``;
``<root>`` is the database's root relative to the project (``.`` for the project itself) as one
name. Every name is spelled by :func:`spelled`, which percent-encodes every character outside
lower-case letters, digits, ``_`` and ``-`` as its UTF-8 bytes, a lone surrogate included
(``urllib.parse.unquote(name, errors="surrogatepass")`` reads it back).

    tcip dump-store <project_path> <out_dir>
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import tcip_store
from tcip_store import encode_log_line, encode_record

from tcip_mcp.cli import bound_project

_PLAIN = frozenset(string.ascii_lowercase + string.digits + "_-")
_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul",
                           *(f"{d}{i}" for d in ("com", "lpt") for i in range(1, 10))})


def _encoded(text: str) -> str:
    return "".join(f"%{b:02X}" for b in text.encode("utf-8", "surrogatepass"))


def spelled(name: str) -> str:
    """``name`` as one path component: every character outside lower-case letters, digits, ``_``
    and ``-`` percent-encoded with upper-case hex, and the first letter of a Windows device name
    too. The spelling is reversible, never ``.`` or ``..``, and no two names' spellings differ by
    case alone."""
    text = "".join(c if c in _PLAIN else _encoded(c) for c in name)
    return _encoded(text[0]) + text[1:] if text in _DEVICE_NAMES else text


def dump_store(project: Path, out_dir: Path) -> list[Path]:
    """Write every record and log of every store database under ``project``, an established
    project root (:func:`~tcip_mcp.project_record.existing_project`), into ``out_dir``
    (:func:`~tcip_mcp.registry_paths.located` against ``project``), a record through
    :func:`tcip_store.encode_record` and a log one :func:`tcip_store.encode_log_line` per line, and
    return the files written. Refuses (``ValueError``) an ``out_dir`` inside the project and a log
    holding entries that will not decode."""
    from tcip_store.file_backend import database_roots

    from tcip_mcp.registry_paths import located

    out_dir = located(out_dir, project)
    if out_dir.is_relative_to(project):
        raise ValueError(f"{out_dir} is inside the project {project}; dump it somewhere outside")
    written: list[Path] = []
    for root in database_roots(project):
        base = out_dir / spelled(root.relative_to(project).as_posix())
        for store in tcip_store.stores(str(root)):
            for key in tcip_store.keys(store, str(root)):
                *dirs, last = (spelled(name) for name in (key.store, *key.parts))
                stem = base.joinpath(*dirs, last)
                files = []
                if tcip_store.exists(key):
                    files.append((stem.with_name(f"{last}.json"),
                                  encode_record(tcip_store.read(key))))
                page = tcip_store.read_log(key)
                if page.corrupt:
                    raise ValueError(f"the log {key} holds entries at {page.corrupt} that will "
                                     "not decode")
                if page.records:
                    files.append((stem.with_name(f"{last}.jsonl"),
                                  b"".join(encode_log_line(e) + b"\n" for e in page.records)))
                for path, data in files:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    written.append(path)
    return written


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog=prog)
    parser.add_argument("project_path", help="Root directory of the project.")
    parser.add_argument("out_dir", help="Directory outside the project to write the files into.")
    args = parser.parse_args(argv)
    try:
        written = dump_store(bound_project(args.project_path), Path(args.out_dir))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(written)} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

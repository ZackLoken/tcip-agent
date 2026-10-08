"""``tcip dump-store``: every record and log of a project's store databases written out as files
a person reads, each decoding to the value the database holds."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pytest

import tcip_store as ts
from tcip_mcp.cli.dump_store import dump_store
from tcip_mcp.tools.meta_tools import report_friction, write_retrospective
from tests._cli_fixtures import run_tcip
from tests._web_fixtures import new_project


def _project_with_records(tmp_path: Path) -> Path:
    """A project holding records and log entries in two databases, written by their producers."""
    from tests import _trait_fixtures as fx

    project = new_project(tmp_path).root
    report_friction(project, category="missing_tool", detail="a separator / in the detail")
    write_retrospective(project, project_id="p", task="t", worked="w", did_not_work="d")
    fx.seed_confirmed_count(project)
    return project


def _held(project: Path) -> dict[tuple, object]:
    """Every record and every log each database under ``project`` holds, read through the seam,
    by ``(root relative to the project, store, parts, suffix)``."""
    held: dict[tuple, object] = {}
    for root in {database.parent.parent for database in project.rglob("store.db")}:
        relative = root.relative_to(project).as_posix()
        for store in ts.stores(str(root)):
            for key in ts.keys(store, str(root)):
                if ts.exists(key):
                    held[(relative, store, key.parts, ".json")] = ts.read(key)
                if entries := ts.read_log(key).records:
                    held[(relative, store, key.parts, ".jsonl")] = entries
    return held


def _dumped(out: Path) -> dict[tuple, object]:
    """Every file under ``out``, its names read back through ``unquote``, by the same tuple."""
    dumped: dict[tuple, object] = {}
    for path in (p for p in out.rglob("*") if p.is_file()):
        relative, store, *parts = (unquote(name, errors="surrogatepass") for name in
                                   path.relative_to(out).with_suffix("").parts)
        data = path.read_bytes()
        dumped[(relative, store, tuple(parts), path.suffix)] = (
            [ts.decode_value(line) for line in data.splitlines()] if path.suffix == ".jsonl"
            else ts.decode_value(data))
    return dumped


def test_the_records_a_database_holds_and_the_records_its_dump_writes_agree(tmp_path: Path):
    """Store by store, the dump's files read back to exactly the keys and values each database
    answers."""
    project = _project_with_records(tmp_path)
    out = tmp_path.parent / "dump"

    written = dump_store(project, out)

    held = _held(project)
    assert len({relative for relative, *_ in held}) == 2
    assert len(written) == len(held)
    assert _dumped(out) == held


def test_every_key_the_store_admits_dumps_to_its_own_file_inside_the_destination(
    tmp_path: Path,
):
    """A key with no parts, parts that would climb out of the destination, two parts that differ
    only by case and a part holding a lone surrogate each land in a file of their own under
    ``out_dir``, and a key holding both a record and log entries dumps both."""
    project = new_project(tmp_path).root
    root = str(project)
    for parts, value in (((), {"n": 0}), (("..", "..", "escaped"), {"n": 1}),
                         (("A",), {"n": 2}), (("a",), {"n": 3}), (("both",), {"n": 4}),
                         ((chr(0xD800),), {"n": 5})):
        ts.replace(ts.Key("probe", root, parts), value)
    ts.append(ts.Key("probe", root, ("both",)), {"i": 1})
    out = tmp_path.parent / "dump"

    written = dump_store(project, out)

    assert all(path.resolve().is_relative_to(out.resolve()) for path in written)
    assert len(set(path.as_posix().lower() for path in written)) == len(written)
    dumped = _dumped(out)
    assert {parts: value for (_, store, parts, suffix), value in dumped.items()
            if store == "probe" and suffix == ".json"} == {
        (): {"n": 0}, ("..", "..", "escaped"): {"n": 1}, ("A",): {"n": 2}, ("a",): {"n": 3},
        ("both",): {"n": 4}, (chr(0xD800),): {"n": 5}}
    assert dumped[(".", "probe", ("both",), ".jsonl")] == [{"i": 1}]


def test_the_dump_refuses_a_directory_inside_the_project_and_admits_one_outside(tmp_path: Path):
    project = _project_with_records(tmp_path)

    with pytest.raises(ValueError, match="inside the project"):
        dump_store(project, project / "dump")
    assert not (project / "dump").exists()

    assert dump_store(project, tmp_path.parent / "dump")


def test_the_command_writes_the_dump_and_answers_how_many_files(tmp_path: Path):
    project = _project_with_records(tmp_path)
    out = tmp_path.parent / "dump"
    ts.release_root(project)

    done = run_tcip("dump-store", [str(project), str(out)])

    assert done.returncode == 0, done.stderr
    assert "file(s) written" in done.stdout
    assert (out / "%2E" / "friction_reports").is_dir()

"""The test tree's shared ways of spoiling a store are used, not written out by hand.

Each helper in ``tests._record_damage_fixtures`` is exercised here, and a scan over the test tree
finds no test spelling the same steps out again.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import tcip_store as ts
from tests._record_damage_fixtures import (
    NON_DATABASE,
    damage_last_log_entry,
    damage_log_entry,
    unopenable_database,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

TEST_TREE = Path(__file__).resolve().parent
DAMAGE_FIXTURES = "_record_damage_fixtures.py"
DAMAGING_SQL = re.compile(r"\bupdate\s+(records|log_entries)\b", re.IGNORECASE)


def _test_sources() -> Iterator[tuple[str, str]]:
    """Yield each Python file under the test tree as its posix path from the tree and its text."""
    for path in sorted(TEST_TREE.rglob("*.py")):
        yield (
            path.relative_to(TEST_TREE).as_posix(),
            path.read_text(encoding="utf-8"),
        )


def _record_key(root: Path) -> ts.Key:
    return ts.Key("fixture_record", str(root), ("a",))


def _log_key(root: Path) -> ts.Key:
    return ts.Key("fixture_log", str(root), ("entries",))


def _names_a_database(expr: ast.AST) -> bool:
    """Return whether ``expr`` calls or names ``database_file`` or spells ``"store.db"``."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id == "database_file":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "database_file":
            return True
        if isinstance(node, ast.Constant) and node.value == "store.db":
            return True
    return False


def _hand_spoils(source: str) -> list[int]:
    """Return the lines of ``source`` writing bytes to a path built as a database's.

    The receiver is the path expression itself, or a name assigned from one in the same function.
    """
    module = ast.parse(source)
    scopes = [
        module,
        *(
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
    ]
    lines: set[int] = set()
    for scope in scopes:
        assigned: dict[str, list[ast.expr]] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned.setdefault(target.id, []).append(node.value)
        for node in ast.walk(scope):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_bytes"
            ):
                continue
            receiver = node.func.value
            sources = (
                assigned.get(receiver.id, [])
                if isinstance(receiver, ast.Name)
                else [receiver]
            )
            if any(_names_a_database(source) for source in sources):
                lines.add(node.lineno)
    return sorted(lines)


def test_a_spoiled_database_is_refused_by_the_identity_check_and_gone_after_the_block(
    tmp_path: Path,
) -> None:
    """A root whose store was open reads as refused inside the block; after it, no file is left.

    The refusal is the identity check's sentence naming the path, and the next write builds a
    fresh database that reads back.
    """
    key = _record_key(tmp_path)
    ts.replace(key, {"n": 1})

    with unopenable_database(tmp_path) as path:
        assert path.read_bytes() == NON_DATABASE
        with pytest.raises(ts.DatabaseRefusedError) as refused:
            ts.read(key, default=None)

    assert "could not complete this store's identity check" in str(
        refused.value
    )
    assert str(path) in str(refused.value)
    assert not path.exists()
    ts.replace(key, {"n": 2})
    assert ts.read(key) == {"n": 2}
    assert path.is_file()


def test_a_root_with_no_database_yet_is_spoiled_and_left_without_one(
    tmp_path: Path,
) -> None:
    """A root holding no ``.tcip`` directory is refused inside the block and has no file after."""
    root = tmp_path / "fresh"
    root.mkdir()
    key = _record_key(root)

    with unopenable_database(root) as path:
        assert path.parent.is_dir()
        with pytest.raises(ts.DatabaseRefusedError):
            ts.read(key, default=None)

    assert not path.exists()
    assert ts.read(key, default=None) is None


def test_no_test_writes_bytes_over_a_database_by_hand() -> None:
    """Only the damage fixtures write bytes over ``database_file(...)`` or a ``store.db`` path."""
    found = [
        f"{name}:{line}"
        for name, source in _test_sources()
        if name != DAMAGE_FIXTURES
        for line in _hand_spoils(source)
    ]

    assert not found, (
        "these write bytes over a database; use "
        f"tests._record_damage_fixtures.unopenable_database: {found}"
    )


def test_a_log_entry_damaged_at_a_position_is_reported_at_that_position(
    tmp_path: Path,
) -> None:
    """The second of three entries, damaged, reads as ``corrupt`` at position 1; the rest decode.

    Read from a cursor, the positions count from the first entry past it.
    """
    key = _log_key(tmp_path)
    ts.append(key, {"i": 0})
    first = ts.read_log(key).cursor
    ts.append(key, {"i": 1})
    ts.append(key, {"i": 2})

    damage_log_entry(key, 1, b'{"i": bro')

    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 2]
    assert page.corrupt == (1,)
    past_first = ts.read_log(key, after=first)
    assert [r["i"] for r in past_first.records] == [2]
    assert past_first.corrupt == (0,)


def test_the_last_log_entry_is_damaged_and_the_cursor_returned_is_past_it(
    tmp_path: Path,
) -> None:
    """The last of three entries reads as ``corrupt``; the cursor returned is the log's own."""
    key = _log_key(tmp_path)
    for i in range(3):
        ts.append(key, {"i": i})

    past = damage_last_log_entry(key)

    page = ts.read_log(key)
    assert [r["i"] for r in page.records] == [0, 1]
    assert page.corrupt == (2,)
    assert page.cursor == past


def test_no_test_writes_the_sql_that_damages_a_record_or_a_log_entry() -> None:
    """Only the damage fixtures hold the statements that rewrite a stored record or log entry."""
    found = [
        f"{name}:{number}"
        for name, source in _test_sources()
        if name != DAMAGE_FIXTURES
        for number, line in enumerate(source.splitlines(), start=1)
        if DAMAGING_SQL.search(line)
    ]

    assert not found, (
        "these rewrite a stored entry by hand; use damage_record, damage_log_entry or "
        f"damage_last_log_entry from tests._record_damage_fixtures: {found}"
    )

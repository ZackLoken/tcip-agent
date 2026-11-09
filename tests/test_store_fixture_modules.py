"""The test tree's shared ways of spoiling a store are used, not written out by hand.

Each helper in ``tests._record_damage_fixtures`` is exercised here, and a scan over the test tree
finds no test spelling the same steps out again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tcip_store as ts
from tests._record_damage_fixtures import NON_DATABASE, unopenable_database

TEST_TREE = Path(__file__).resolve().parent
DAMAGE_FIXTURES = "_record_damage_fixtures.py"


def _record_key(root: Path) -> ts.Key:
    return ts.Key("fixture_record", str(root), ("a",))


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
    found = {
        f"{path.relative_to(TEST_TREE)}:{line}"
        for path in sorted(TEST_TREE.rglob("*.py"))
        if path.name != DAMAGE_FIXTURES
        for line in _hand_spoils(path.read_text(encoding="utf-8"))
    }

    assert not found, (
        "these write bytes over a database; use "
        f"tests._record_damage_fixtures.unopenable_database: {sorted(found)}"
    )

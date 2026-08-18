"""``scripts/conform_trait_specs.py --migrate-reroot``: the one-off step for a project whose
trait-spec record still sits in the old, self-rooted ``trait_specs`` database (the shape the
store used before it was re-rooted onto the shared ``.tcip/state`` database every sibling
project-state store already shares).

The migration copies exactly one thing, the spec record, and must neither touch the
trait-spec-statement record already on file at the shared state root nor delete the old
database once it has copied out of it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend, database_path

from tcip_mcp import traits

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_trait_specs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_trait_specs_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    """This migration is a SQLite-only artifact; bind it explicitly rather than trust whichever
    backend the ambient environment happens to default to."""
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def _seed_old_database(root: Path, trait: str, spec: dict) -> None:
    """Write a spec straight into the old, self-rooted database, the shape a pre-re-root process
    wrote: the key's own root is the specs directory itself, not the shared state root."""
    old_specs_root = root / ".tcip" / "state" / "trait_specs"
    old_key = ts.Key(traits.TRAIT_SPECS_STORE, str(old_specs_root), (trait,))
    ts.replace(old_key, spec, expect=ts.Version.ABSENT)


def _seed_statement(root: Path, trait: str) -> dict:
    statement = {
        "trait": trait,
        "statement_fields": {"delivers": ["leaf_length"]},
        "rationale": "pre-existing statement, written before this migration ran",
        "stated_by": "author_trait_spec",
        "stated_at": "2026-03-04T00:00:00+00:00",
        "relayed_note": "",
        "confirmed_by": None, "confirmed_at": None,
        "identity_from_request": None, "record_seen": None,
    }
    key = traits.trait_spec_statement_key(traits.trait_spec_statements_scope(root), trait)
    ts.replace(key, statement, expect=ts.Version.ABSENT)
    return statement


def test_migrates_the_spec_and_leaves_the_statement_and_old_database_untouched(tmp_path: Path):
    module = _load_script()
    backend = _bind_sqlite()
    spec = {"name": "leaf", "delivers": ["leaf_length"]}
    _seed_old_database(tmp_path, "leaf", spec)
    statement_before = _seed_statement(tmp_path, "leaf")
    old_db = database_path(str(tmp_path / ".tcip" / "state" / "trait_specs"))
    assert old_db.is_file()

    outcome = module.migrate_reroot_one(tmp_path, "leaf", backend)

    assert outcome.startswith("migrated:")
    new_key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert ts.read(new_key) == spec
    assert old_db.is_file()

    statement_key = traits.trait_spec_statement_key(traits.trait_spec_statements_scope(tmp_path), "leaf")
    assert ts.read(statement_key) == statement_before


def test_a_second_run_is_an_idempotent_no_op(tmp_path: Path):
    module = _load_script()
    backend = _bind_sqlite()
    spec = {"name": "leaf", "delivers": ["leaf_length"]}
    _seed_old_database(tmp_path, "leaf", spec)

    first = module.migrate_reroot_one(tmp_path, "leaf", backend)
    assert first.startswith("migrated:")

    second = module.migrate_reroot_one(tmp_path, "leaf", backend)
    assert second.startswith("no-op:")


def test_a_disagreeing_new_value_is_refused_rather_than_overwritten(tmp_path: Path):
    module = _load_script()
    backend = _bind_sqlite()
    _seed_old_database(tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]})
    new_key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    ts.replace(new_key, {"name": "leaf", "delivers": ["leaf_width"]}, expect=ts.Version.ABSENT)

    with pytest.raises(module.RerootConflict, match="disagrees"):
        module.migrate_reroot_one(tmp_path, "leaf", backend)

    assert ts.read(new_key) == {"name": "leaf", "delivers": ["leaf_width"]}


def test_a_missing_old_database_is_refused_rather_than_silently_created(tmp_path: Path):
    module = _load_script()
    backend = _bind_sqlite()

    with pytest.raises(RuntimeError, match="does not exist"):
        module.migrate_reroot_one(tmp_path, "leaf", backend)


def test_the_file_backend_is_refused_rather_than_silently_no_op(tmp_path: Path):
    module = _load_script()
    ts.bind(FileBackend())

    with pytest.raises(RuntimeError, match="SQLite backend"):
        module.migrate_reroot_one(tmp_path, "leaf", FileBackend())

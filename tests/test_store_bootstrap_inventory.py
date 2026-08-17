"""The bootstrap import set and the contract suite's inventory are one list or they are wrong.

Two maintained lists of every store exist for two different reasons: the contract suite names
each store to pin its bytes and its placement, and the bootstrap imports each store's owning
module so a tool that has to write every store back out can see it. A store added to one and
not the other is a store that either has no byte identity or silently exports as nothing, so
the lists are compared rather than each trusted on its own.
"""

from __future__ import annotations

import tcip_store as ts
from scripts._store_bootstrap import ADOPTION_SOURCES, LAYOUTS, bootstrapped_stores
from tests.test_store_contract import REGISTERED


def _platform_stores() -> set[str]:
    """Every registered store the platform owns, leaving out a test's own scaffolding."""
    return {
        name
        for name in bootstrapped_stores()
        if not ts.get_descriptor(name).declared_in.startswith(("tests", "test_"))
    }


def test_the_bootstrap_imports_exactly_the_stores_the_contract_suite_names():
    """A store the bootstrap does not import has no locator to export through, and a store the
    contract suite does not name has no bytes anyone checked."""
    assert _platform_stores() == set(REGISTERED)


def test_every_record_and_log_store_states_where_its_entries_live():
    """Adoption reads files rather than rows, so a store with no source is a file layout no
    adoption would take in, left behind to read as absent under a database."""
    owed = {
        name for name in _platform_stores() if ts.get_descriptor(name).kind in ("record", "log")
    }

    assert set(ADOPTION_SOURCES) == owed


def test_no_adoption_source_names_a_blob_or_an_unregistered_store():
    """A blob's file stays a file under every backend, so adopting one would move bytes the
    database is not supposed to hold."""
    blobs = {name for name in ADOPTION_SOURCES if ts.get_descriptor(name).kind == "blob"}

    assert blobs == set()


def test_each_adoption_source_matches_its_store_key_shape_and_a_known_layout():
    """A pattern per key part, in a layout an operator can name: a source of the wrong arity
    silently matches nothing, which reads as "this store has no files"."""
    mismatched = {
        name: (len(source.parts), len(ts.get_descriptor(name).key_fields))
        for name, source in ADOPTION_SOURCES.items()
        if len(source.parts) != len(ts.get_descriptor(name).key_fields)
    }
    unknown = {name: source.layout for name, source in ADOPTION_SOURCES.items()
               if source.layout not in LAYOUTS}

    assert mismatched == {}
    assert unknown == {}

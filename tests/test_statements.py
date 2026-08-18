"""``statements.py``'s shared primitives, proven identical to what they replaced.

``canonical`` and the content-hash-over-declared-fields pattern used to live inside
``operationalization.py``, hand-written for its own eight statement fields. They now live in
``tcip_mcp.statements`` as generic functions any statement kind can call, and
``operationalization.record_seen_hash`` becomes a one-line wrapper over them. This module checks
that the move changed nothing observable: the wrapper's hash still equals a direct call through the
real store's own round-tripped record, and ``canonical`` still normalizes the same way a JSON
round trip does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp import operationalization as op
from tcip_mcp.statements import canonical, content_hash, now_iso
from tests import _operationalization_fixtures as fx


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return fx.seed_project(tmp_path / "project")


def test_the_wrapper_matches_a_direct_call_over_a_record_round_tripped_through_the_real_store(
    project: Path,
) -> None:
    """``record_seen_hash`` is now ``content_hash(record, STATEMENT_FIELDS)``; prove they agree."""
    written = fx.state_crossing(project)
    _spec, stored, _specs_dir = fx.resolve(project, fx.CROSSING_TRAIT, op.STATE_CROSSING_DATES)

    assert op.record_seen_hash(stored.value) == content_hash(stored.value, op.STATEMENT_FIELDS)
    assert op.record_seen_hash(written) == content_hash(written, op.STATEMENT_FIELDS)


def test_content_hash_does_not_vary_with_sequence_type(project: Path) -> None:
    record = fx.state_crossing(project)
    as_tuples = {**record, "delivered_phenotypes": tuple(record["delivered_phenotypes"])}

    assert content_hash(as_tuples, op.STATEMENT_FIELDS) == content_hash(record, op.STATEMENT_FIELDS)


def test_content_hash_takes_the_field_tuple_as_a_parameter_rather_than_a_fixed_set() -> None:
    """The generalization S1 asked for: a caller names its own statement kind's field set."""
    narrow = content_hash({"a": 1, "b": 2}, ("a",))
    wide = content_hash({"a": 1, "b": 2}, ("a", "b"))

    assert narrow != wide
    assert narrow == content_hash({"a": 1, "b": "irrelevant"}, ("a",))


def test_canonical_sorts_mapping_keys_and_lists_sequences_recursively() -> None:
    assert canonical({"b": 2, "a": (1, 2)}) == {"a": [1, 2], "b": 2}
    assert canonical("a string is a sequence but stays a scalar") == (
        "a string is a sequence but stays a scalar"
    )
    assert canonical(3) == 3


def test_now_iso_returns_a_parseable_utc_timestamp() -> None:
    from datetime import datetime

    stamp = now_iso()

    assert datetime.fromisoformat(stamp).utcoffset() is not None

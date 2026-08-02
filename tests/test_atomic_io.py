"""atomic_io: atomicity + concurrency (no corruption, no lost updates)."""

import json
from concurrent.futures import ThreadPoolExecutor

from tcip_mcp.utils.atomic_io import (
    append_jsonl,
    atomic_write_json,
    file_transaction,
    read_json,
)


def test_atomic_write_json_roundtrip_and_no_temp_leftovers(tmp_path):
    p = tmp_path / "sub" / "x.json"  # parent dir is created automatically
    atomic_write_json(p, {"a": 1, "b": [1, 2]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2]}
    # The temp file must have been renamed away; only the target remains.
    assert [q.name for q in p.parent.iterdir()] == ["x.json"]


def test_read_json_returns_default_for_missing_and_corrupt(tmp_path):
    assert read_json(tmp_path / "missing.json", default={"d": 1}) == {"d": 1}
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert read_json(bad, default=[]) == []


def test_append_jsonl_concurrent_preserves_every_line(tmp_path):
    p = tmp_path / "log.jsonl"
    n = 200
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda i: append_jsonl(p, {"i": i}), range(n)))
    lines = p.read_text().strip().splitlines()
    assert len(lines) == n  # no interleaved/torn/lost appends
    assert sorted(json.loads(ln)["i"] for ln in lines) == list(range(n))


def test_file_transaction_serializes_rmw_no_lost_updates(tmp_path):
    p = tmp_path / "counter.json"
    atomic_write_json(p, {"n": 0})
    n = 200

    def increment(_):
        with file_transaction(p):
            data = read_json(p, default={"n": 0})
            data["n"] += 1
            atomic_write_json(p, data)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(increment, range(n)))
    assert read_json(p)["n"] == n  # without the lock, concurrent RMW would lose updates

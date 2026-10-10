"""The browser's delivery-event fixtures are what the platform's own deliveries record and serve.

``tools/generate_delivery_fixture.py`` makes real deliveries in scratch projects and writes the
records ``GET /api/results/delivery-events`` serves for them to
``frontend/src/test/deliveryEvents.json``; this test regenerates them and holds the checked-in
file to what a regeneration serves, every value compared but those a run mints.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from typing import Any

import pytest

from tests import FRONTEND_SRC, REPO_ROOT

GENERATED = FRONTEND_SRC / "test" / "deliveryEvents.json"
GENERATOR = REPO_ROOT / "tools" / "generate_delivery_fixture.py"


def _generator():
    """The fixture generator loaded as a module, so the test regenerates the way the tool
    does."""
    spec = importlib.util.spec_from_file_location("tcip_delivery_fixture_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PER_RUN = frozenset({
    "event_id", "acknowledgment_id", "assessment_id", "dataset_id", "experiment_id",
    "produced_at", "recorded_at", "built_at", "output_sha256", "result_sha256", "record_sha256",
    "checkpoint_sha256"})
"""The record fields two productions of the same scenes fill differently: minted ids, the
times things happened, and digests over what carries those (a run's weights, a written output, a
record holding a time)."""


def _normalized(fixture: dict[str, dict]) -> dict[str, dict]:
    """``fixture`` with each :data:`PER_RUN` string replaced by a string placeholder numbered by
    first appearance, so equal values stay equal and distinct ones distinct and a value of any
    other type is kept as served, and an archived mapping key that is exactly the name
    :func:`~tcip_mcp.pipelines.postprocessing.plant_mapping.archived_mapping_name` gives the
    cited record replaced by that record's placeholder. Every other value is kept as served."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import archived_mapping_name

    seen: dict[str, str] = {}

    def walk(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if key in PER_RUN and isinstance(value, str):
            return seen.setdefault(value, f"<{key} {len(seen)}>")
        return value

    out: dict[str, dict] = {}
    for name, record in fixture.items():
        normalized = walk(record)
        pm, resolved = record["plant_mapping"], record.get("plant_mapping_resolved_key")
        if (pm is not None and resolved is not None and "record_sha256" in pm
                and resolved == archived_mapping_name(pm["name"], pm["record_sha256"])):
            normalized["plant_mapping_resolved_key"] = (
                f"<archived {pm['name']} at {seen[pm['record_sha256']]}>")
        out[name] = normalized
    return out


def _changed(fixture: dict, name: str, *path_and_value: Any) -> dict:
    """A copy of ``fixture`` with the value at ``path`` under record ``name`` set to the last
    argument."""
    *path, value = path_and_value
    out = copy.deepcopy(fixture)
    target = out[name]
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return out


def test_the_comparison_keeps_every_value_a_run_does_not_mint() -> None:
    """A served fact the frontend asserts (a disclosure count, an output path) changing, a
    minted value changing its type or going absent, two records sharing a minted value no longer
    sharing it, and an archived mapping key that is not the archive's own name each break the
    comparison; a minted id changing its value does not."""
    checked = json.loads(GENERATED.read_text(encoding="utf-8"))
    count = checked["canopy"]["plant_mapping"]["segments_without_plant"]
    digest = checked["archived_mapping"]["plant_mapping"]["record_sha256"]
    broken = [
        _changed(checked, "canopy", "plant_mapping", "segments_without_plant", count + 99),
        _changed(checked, "validated", "output_path", "/scratch/elsewhere.csv"),
        _changed(checked, "validated", "event_id", 7),
        _changed(checked, "validated", "output_sha256", 7),
        _changed(checked, "registry", "producer", "experiment_id", None),
        _changed(checked, "archived_mapping", "producer", "experiment_id", "another-run"),
        *(_changed(checked, "archived_mapping", "plant_mapping_resolved_key", key)
          for key in ("valley@", "valley@3", f"valley@{digest}")),
    ]

    for fixture in broken:
        assert _normalized(fixture) != _normalized(checked)
    reminted = _changed(checked, "validated", "event_id", "another-id")
    assert _normalized(reminted) == _normalized(checked)


def test_the_delivery_fixture_is_what_a_regeneration_serves() -> None:
    """The checked-in fixture is each record the scenes' deliveries serve, value for value but
    for the minted strings :func:`_normalized` replaces: a record field added, renamed or given
    another type, or a scene that no longer delivers what it did, fails this."""
    pytest.importorskip("torch")
    produced = _generator().produce()
    checked = json.loads(GENERATED.read_text(encoding="utf-8"))
    assert _normalized(checked) == _normalized(produced), (
        "packages/tcip-web/frontend/src/test/deliveryEvents.json is out of date; "
        "run python tools/generate_delivery_fixture.py")

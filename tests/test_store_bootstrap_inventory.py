"""The bootstrap import set, the contract suite's inventory, and the claim table are one list.

Three maintained statements about every store exist for three different reasons: the contract
suite names each store to pin its bytes and its placement, the bootstrap imports each store's
owning module so a tool that has to write every store back out can see it, and the claim table
says where each store's files sit so the conform rail can answer without importing anything. A
store in one and not the others either has no byte identity, silently exports as nothing, or is
invisible to the rail that decides whether its files were left behind, so the three are compared
rather than each trusted on its own.

The claim table restates each locator's path shape as data, so the two are held together here
by round-tripping a golden key through the locator onto a path the row must match, and by a set
of near misses the row must reject.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_store.layout_claims import (
    ANY,
    LAYOUTS,
    PLATFORM_CLAIMS,
    Claim,
    Constant,
    Patterned,
    Template,
    literal,
    matches_template,
)

from tcip_mcp.store_catalog import bootstrapped_stores

from tests.test_store_contract import REGISTERED

_PACKAGES = Path(__file__).resolve().parent.parent / "packages"


def _platform_stores() -> set[str]:
    """Every registered store the platform owns, leaving out a test's own scaffolding."""
    return {
        name
        for name in bootstrapped_stores()
        if not ts.get_descriptor(name).declared_in.startswith(("tests", "test_"))
    }


def _stores_owed_a_claim() -> set[str]:
    """Every platform store whose files the rail has to be able to recognize."""
    return {
        name for name in _platform_stores() if ts.get_descriptor(name).kind in ("record", "log")
    }


def _matching_templates(claim: Claim, relative: str) -> list[Template]:
    """Every template of this claim the path satisfies."""
    segments = tuple(relative.split("/"))
    return [template for template in claim.templates if matches_template(template, segments)]


def test_the_bootstrap_imports_exactly_the_stores_the_contract_suite_names():
    """A store the bootstrap does not import has no locator to export through, and a store the
    contract suite does not name has no bytes anyone checked."""
    assert _platform_stores() == set(REGISTERED)


def test_every_record_and_log_store_has_exactly_one_claim_row_and_no_row_is_orphaned():
    """A store with no row is invisible to the rail, so its files read as absent under a
    database; a row for a store nothing declares claims files no store would ever adopt.

    ``bootstrapped_stores`` answers the process-global registry, and a suite that imports an
    owning module anywhere registers its store for every test after it, so this states the
    property of a session rather than of the catalog's own imports.
    :func:`test_the_catalog_registers_every_store_the_claim_table_speaks_for` is the one that
    answers for a fresh process, which is the state ``export-store`` and the freeze manifest run in.
    """
    assert set(PLATFORM_CLAIMS) == _stores_owed_a_claim()


def test_no_claim_row_names_a_blob_or_an_unknown_layout():
    """A blob's file stays a file under every backend, so claiming one would make every dataset
    root read as unconformed; a layout no operator can name is a row nothing ever matches."""
    blobs = {name for name in PLATFORM_CLAIMS if ts.get_descriptor(name).kind == "blob"}
    unknown = {
        name: claim.layout
        for name, claim in PLATFORM_CLAIMS.items()
        if claim.layout not in LAYOUTS
    }

    assert blobs == set()
    assert unknown == {}


def _golden_path(name: str, root: Path) -> str:
    """Where this store's own locator places a real key of it, relative to its root."""
    descriptor = ts.get_descriptor(name)
    assert descriptor.locator is not None
    key = REGISTERED[name].key_of(root)
    return descriptor.locator.relative_path(key.root, key.parts).as_posix()


@pytest.mark.parametrize("name", sorted(PLATFORM_CLAIMS))
def test_a_golden_key_lands_where_its_claim_row_says_it_would(name: str, tmp_path: Path):
    """The row and the locator state one path shape twice, so the locator's own placement of a
    real key is what the row is checked against rather than a path this test spells."""
    placed = _golden_path(name, tmp_path)

    assert _matching_templates(PLATFORM_CLAIMS[name], placed), (
        f"{name} places {placed}, which its claim row does not match"
    )


@pytest.mark.parametrize("name", sorted(PLATFORM_CLAIMS))
def test_a_claim_row_rejects_the_near_misses_of_its_own_golden_path(name: str, tmp_path: Path):
    """Recall is half the property. A row that also matched a wrong constant, a wrong extension
    or a wrong depth would claim files of stores that never wrote them, which under the rail is
    a root permanently refused and under the planner a document attributed to the wrong store."""
    claim = PLATFORM_CLAIMS[name]
    placed = _golden_path(name, tmp_path)
    segments = placed.split("/")
    template = _matching_templates(claim, placed)[0]

    misses = []
    for index, matcher in enumerate(template):
        spelled = list(segments)
        if isinstance(matcher, Constant) or matcher.pattern.literal is not None or matcher.lead:
            spelled[index] = f"not_{segments[index]}"
            misses.append("/".join(spelled))
        if isinstance(matcher, Patterned) and matcher.tail:
            spelled = list(segments)
            spelled[index] = segments[index][: -len(matcher.tail)] + ".wrong"
            misses.append("/".join(spelled))
    arities = {len(one) for one in claim.templates}
    if len(segments) + 1 not in arities:
        misses.append(f"{placed}/deeper.json")
    if len(segments) - 1 not in arities and len(segments) > 1:
        misses.append("/".join(segments[1:]))

    assert misses, f"{name} spells no constant text at all, so nothing tells its files apart"
    assert [miss for miss in misses if _matching_templates(claim, miss)] == []


def test_the_row_checks_notice_a_row_that_is_missing_or_misshapen(tmp_path: Path):
    """The checks above are only worth their runtime if a wrong table fails them, so each is fed
    a row deliberately made wrong: one dropped from the table, one whose constant directory is
    misspelled so the store's own placement no longer matches it, and one widened until it
    swallows a document belonging to another store."""
    dropped = {name for name in PLATFORM_CLAIMS if name != "image_status"}
    placed = _golden_path("image_status", tmp_path)
    misspelled = Claim(
        PLATFORM_CLAIMS["image_status"].layout,
        ((Constant(".tcip"), Constant("stat"), Patterned(literal("image_status"), tail=".json")),),
    )
    widened = Claim(
        PLATFORM_CLAIMS["image_status"].layout,
        ((Constant(".tcip"), Constant("state"), Patterned(ANY, tail=".json")),),
    )

    assert dropped != _stores_owed_a_claim()
    assert _matching_templates(misspelled, placed) == []
    assert _matching_templates(widened, ".tcip/state/not_image_status.json")
    assert _matching_templates(PLATFORM_CLAIMS["image_status"], ".tcip/state/not_image_status.json") == []


def test_both_review_verdict_depths_match_the_one_row_that_speaks_for_them():
    """A verdict names a prediction bucket or names none, and its shard sits one directory
    deeper in the first case. One row covers both or half the shards read as unclaimed."""
    claim = PLATFORM_CLAIMS["review_verdicts"]

    assert _matching_templates(claim, "review/predictions/live/a_1.jpg.json") == []
    assert _matching_templates(claim, "review/predictions_live/a_1.jpg.json")
    assert _matching_templates(claim, "review/a_1.jpg.json")
    assert _matching_templates(claim, "a_1.jpg.json") == []


def test_every_platform_registration_is_an_import_side_effect():
    """The rail's declared-claim window is confined to stores registered at runtime, and that
    confinement is only real if no shipped store is one. A call nested in a function body could
    run at any time, so the source itself is what says every shipped store registers on import.
    """
    nested: list[str] = []
    for path in sorted(_PACKAGES.glob("*/src/**/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and _names_register_store(inner.func):
                    nested.append(f"{path.name}:{inner.lineno}")

    assert nested == []


def _names_register_store(func: ast.expr) -> bool:
    """Whether a called expression is ``register_store`` however it was imported."""
    if isinstance(func, ast.Name):
        return func.id == "register_store"
    return isinstance(func, ast.Attribute) and func.attr == "register_store"


_CATALOG_IMPORT_ONLY = """\
import sys

import tcip_mcp
from tcip_mcp.store_catalog import bootstrapped_stores

stores = bootstrapped_stores()
assert "tcip_web" not in sys.modules, sorted(sys.modules)
assert "learning_capture" in stores, stores
assert "job_registry" in stores, stores
assert "annotation_stats" in stores, stores
print(tcip_mcp.__file__)
"""


_CATALOG_AGAINST_CLAIMS = """\
import json

import tcip_mcp
from tcip_store import get_descriptor
from tcip_store.layout_claims import platform_claim_stores
from tcip_mcp.store_catalog import bootstrapped_stores

registered = bootstrapped_stores()
record_and_log = {n for n in registered if get_descriptor(n).kind in ("record", "log")}
claimed = platform_claim_stores()
print(json.dumps({
    "tcip_mcp": tcip_mcp.__file__,
    "claimed_but_unregistered": sorted(claimed - record_and_log),
    "registered_but_unclaimed": sorted(record_and_log - claimed),
}))
"""


def test_the_catalog_registers_every_store_the_claim_table_speaks_for():
    """In a process that imports nothing but the catalog, the claim table and the catalog
    name one set of stores.

    The direction this exists for is claim row to catalog. The claim table is what the conform
    rail answers from without importing any owning module, so a row whose store the catalog
    never registers means the rail speaks for files that ``export-store``, ``adopt-store`` and
    ``generate_frozen_manifest`` cannot see: a record kind declared frozen, sitting outside the
    shipped freeze commitment, with the manifest check agreeing with the gap. The other direction
    already refuses at the point of use, where :func:`~tcip_store.layout_claims.claim_of` raises
    naming the declaration the store owes, so it is asserted here only because it is free.

    A fresh child, not this process: ``bootstrapped_stores`` reads the process-global registry,
    and any import of an owning module anywhere in the suite hides a missing catalog import
    from every in-session check. The child prints the ``tcip_mcp`` it imported so a run whose
    environment resolves an installed copy elsewhere fails here rather than proving the fact
    about another tree.
    """
    result = subprocess.run(
        [sys.executable, "-c", _CATALOG_AGAINST_CLAIMS],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    answered = json.loads(result.stdout)
    imported = Path(answered["tcip_mcp"]).resolve()
    assert Path(__file__).resolve().parents[1] / "packages" in imported.parents, imported
    assert answered["claimed_but_unregistered"] == []
    assert answered["registered_but_unclaimed"] == []


def test_the_catalog_reaches_every_web_owned_store_without_importing_tcip_web():
    """A caller that only needs the catalog (``export-store``, ``adopt-store``, this test
    suite) must not pull the web package in as a side effect: the three stores the web package
    owns register through :mod:`tcip_mcp.web_client`, which the catalog already imports.

    The child prints the ``tcip_mcp`` it imported, and the assertion holds it to this
    repository's own package, so a run whose environment resolves an installed copy elsewhere
    fails here rather than proving the fact about another tree. The catalog's import chain
    reaches torch, so a cold child takes longer than a warm one.
    """
    result = subprocess.run(
        [sys.executable, "-c", _CATALOG_IMPORT_ONLY],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    imported = Path(result.stdout.strip()).resolve()
    repository_packages = Path(__file__).resolve().parents[1] / "packages"
    assert repository_packages in imported.parents, imported

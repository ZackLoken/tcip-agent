"""``scripts/drop_trait_spec_provenance.py``: the one-off conform step for a project whose
trait-spec records still carry the retired free-text ``provenance`` field, and whose
``trait_specs`` directory still holds the stray database and stale YAML file an earlier
YAML-to-record conform step left behind.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import tcip_store as ts
from tcip_store.binding import bind_default
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend, database_path

from tcip_mcp import traits

SCRIPT = Path(__file__).parent.parent / "scripts" / "drop_trait_spec_provenance.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("drop_trait_spec_provenance_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_sqlite() -> SqliteBackend:
    """Bind the SQLite backend explicitly rather than trust the ambient default.

    For a test that only calls the script's functions directly: none of them ever rebind, so
    pinning the backend here and reading it back through the same pin proves the function's own
    logic once, without also re-proving the storage seam's own read/write agreement, which
    ``test_store_contract.py`` already holds both backends to.
    """
    backend = SqliteBackend()
    ts.bind(backend)
    return backend


def _bind_ambient_backend() -> FileBackend | SqliteBackend:
    """Bind whichever backend the suite's own environment names.

    ``main()`` calls ``bind_default()`` itself, so a test that drives ``main()`` has to seed
    its fixture through that same call, not through a backend it picked itself: otherwise the
    fixture and the script under test can disagree about which store holds the records.
    """
    return bind_default()


def _seed_live_record(root: Path, trait: str, spec: dict) -> None:
    key = traits.trait_spec_key(traits.trait_specs_dir(root), trait)
    ts.replace(key, spec, expect=ts.Version.ABSENT)


def _seed_stray_database(root: Path, trait: str, spec: dict, *, restore: FileBackend | SqliteBackend) -> Path:
    """Write a record straight into the old, self-rooted database, the shape a pre-re-root
    conform step left behind, and return the database file it creates.

    The stray database is a SQLite artifact by construction: ``remove_stray_database`` finds it
    by its literal file path, never through the store API, so it has no equivalent under the
    file backend. Seeding it always goes through a ``SqliteBackend`` of its own regardless of
    which backend the test itself is exercising, then restores ``restore`` (the backend the
    caller had bound) so whatever runs next keeps reading and writing through that one.
    """
    old_specs_root = root / ".tcip" / "state" / "trait_specs"
    old_key = ts.Key(traits.TRAIT_SPECS_STORE, str(old_specs_root), (trait,))
    sqlite_backend = SqliteBackend()
    ts.bind(sqlite_backend)
    try:
        ts.replace(old_key, spec, expect=ts.Version.ABSENT)
    finally:
        sqlite_backend.close()
        ts.bind(restore)
    return database_path(str(old_specs_root))


def _seed_stale_yaml(root: Path, trait: str) -> Path:
    specs_dir = root / ".tcip" / "state" / "trait_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = specs_dir / f"{trait}.yml"
    yaml_path.write_text("name: leaf\ndelivers: [leaf_length]\n", encoding="utf-8")
    return yaml_path


def test_drops_provenance_from_the_live_record_and_leaves_other_fields_alone(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    spec = {
        "name": "leaf", "delivers": ["leaf_length"], "notes": "kept as-is",
        "provenance": ["notes: domain_expert_confirmed"],
    }
    _seed_live_record(tmp_path, "leaf", spec)

    outcomes = module.drop_provenance_from_records(tmp_path, plan=False)

    assert outcomes == ["leaf: dropped provenance"]
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    stored = ts.read(key)
    assert "provenance" not in stored
    assert stored["notes"] == "kept as-is"
    # the loader, which refuses an unknown field, now reads the trait again
    assert [s.name for s in traits.load_trait_specs(project_root=tmp_path)] == ["leaf"]


def test_plan_mode_writes_nothing(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    spec = {"name": "leaf", "delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(tmp_path, "leaf", spec)

    outcomes = module.drop_provenance_from_records(tmp_path, plan=True)

    assert outcomes == ["leaf: would drop provenance"]
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert "provenance" in ts.read(key)


def test_a_record_with_no_provenance_field_is_reported_unchanged(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed_live_record(tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]})

    outcomes = module.drop_provenance_from_records(tmp_path, plan=False)

    assert outcomes == ["leaf: no provenance field, unchanged"]


def test_a_record_with_no_provenance_field_that_is_invalid_anyway_is_refused(tmp_path: Path):
    """No ``provenance`` to strip does not exempt a record from validation: a record broken for
    an unrelated reason must still be refused, not waved through as unchanged."""
    _bind_sqlite()
    module = _load_script()
    _seed_live_record(tmp_path, "leaf", {"name": "leaf", "delivers": ["not_a_real_phenotype"]})

    outcomes = module.drop_provenance_from_records(tmp_path, plan=False)

    assert len(outcomes) == 1 and outcomes[0].startswith("leaf: refused")


def test_a_record_invalid_once_provenance_is_dropped_is_refused_and_left_on_file(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    # off-vocab delivers: still invalid once provenance is stripped, so nothing should be written.
    spec = {"name": "leaf", "delivers": ["not_a_real_phenotype"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(tmp_path, "leaf", spec)

    outcomes = module.drop_provenance_from_records(tmp_path, plan=False)

    assert len(outcomes) == 1 and outcomes[0].startswith("leaf: refused")
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert ts.read(key) == spec


def test_removes_the_stray_self_rooted_database(tmp_path: Path):
    """The database, its journal sidecars and the lock file beside it all go, and so does the
    directory that held them. The lock file is seeded explicitly because ``filelock`` keeps it
    after release under Unix and deletes it under Windows: without seeding, a Windows run never
    sees the file a Unix run leaves, and the directory it keeps alive. The file is named here
    the way ``filelock`` names it beside a data file, rather than through the backend's helper,
    so the test collects on a tree that has no helper yet."""
    backend = _bind_sqlite()
    module = _load_script()
    db = _seed_stray_database(
        tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]}, restore=backend
    )
    assert db.is_file()
    backend.close()  # release the file handle before the script deletes the file, Windows-safe
    lock = db.with_name(db.name + ".lock")
    lock.touch()

    outcome = module.remove_stray_database(tmp_path, plan=False)

    assert outcome is not None and outcome.startswith("removed stray database")
    assert str(lock) in outcome
    assert not db.is_file()
    assert not lock.is_file()
    assert not db.parent.is_dir()


def test_remove_stray_database_plan_mode_leaves_it_in_place(tmp_path: Path):
    backend = _bind_sqlite()
    module = _load_script()
    db = _seed_stray_database(
        tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]}, restore=backend
    )
    backend.close()

    outcome = module.remove_stray_database(tmp_path, plan=True)

    assert outcome is not None and outcome.startswith("would remove")
    assert db.is_file()


def test_no_stray_database_reports_none(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    assert module.remove_stray_database(tmp_path, plan=False) is None


def test_removes_a_stale_yaml_file_beside_the_record(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed_live_record(tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]})
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")
    assert yaml_path.is_file()

    outcomes = module.remove_stale_yaml(tmp_path, plan=False)

    assert outcomes == [f"removed stale spec file {yaml_path}"]
    assert not yaml_path.is_file()


def test_remove_stale_yaml_plan_mode_leaves_it_in_place(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    _seed_live_record(tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]})
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")

    outcomes = module.remove_stale_yaml(tmp_path, plan=True)

    assert outcomes == [f"would remove stale spec file {yaml_path}"]
    assert yaml_path.is_file()


def test_a_stale_yaml_with_no_record_in_the_store_is_kept_and_reported(tmp_path: Path):
    _bind_sqlite()
    module = _load_script()
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")

    outcomes = module.remove_stale_yaml(tmp_path, plan=False)

    assert len(outcomes) == 1
    assert outcomes[0].startswith(f"kept {yaml_path}")
    assert "no 'leaf' record loads through the platform's trait-spec loader" in outcomes[0]
    assert yaml_path.is_file()


def test_a_stale_yaml_beside_a_record_that_fails_to_load_is_kept_and_reported(tmp_path: Path):
    """A store key with the trait's name is not enough: the record itself must load. A key
    present but refused (here, off-vocab ``delivers``, unrelated to ``provenance``) leaves the
    YAML the only recoverable copy, so it must be kept, not removed."""
    _bind_sqlite()
    module = _load_script()
    _seed_live_record(tmp_path, "leaf", {"name": "leaf", "delivers": ["not_a_real_phenotype"]})
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")

    outcomes = module.remove_stale_yaml(tmp_path, plan=False)

    assert len(outcomes) == 1
    assert outcomes[0].startswith(f"kept {yaml_path}")
    assert yaml_path.is_file()
    assert yaml_path.is_file()


def test_a_recorded_and_an_unrecorded_stale_yaml_coexist_and_only_the_recorded_one_is_removed(
    tmp_path: Path,
):
    _bind_sqlite()
    module = _load_script()
    _seed_live_record(tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]})
    recorded_yaml = _seed_stale_yaml(tmp_path, "leaf")
    unrecorded_yaml = _seed_stale_yaml(tmp_path, "twig")

    outcomes = module.remove_stale_yaml(tmp_path, plan=False)

    assert f"removed stale spec file {recorded_yaml}" in outcomes
    assert any(o.startswith(f"kept {unrecorded_yaml}") for o in outcomes)
    assert not recorded_yaml.is_file()
    assert unrecorded_yaml.is_file()


def test_end_to_end_against_a_fixture_shaped_like_the_real_on_disk_state(tmp_path: Path):
    """One project fixture carrying all three artifacts the real on-disk state holds (a live
    record with provenance, a stray old-rooted database, a stale YAML file), conformed by the
    same three calls ``main()`` makes, in the same order."""
    backend = _bind_sqlite()
    live_spec = {
        "name": "bud_opening", "delivers": ["leaf_out_05per_date"], "positive_class_name": "open",
        "provenance": ["positive_class_name: domain_expert_confirmed"],
    }
    _seed_live_record(tmp_path, "bud_opening", live_spec)
    stray_db = _seed_stray_database(
        tmp_path, "bud_opening", {"name": "bud_opening", "delivers": ["leaf_out_05per_date"]},
        restore=backend
    )
    yaml_path = _seed_stale_yaml(tmp_path, "bud_opening")

    module = _load_script()
    record_outcomes = module.drop_provenance_from_records(tmp_path, plan=False)
    backend.close()  # release the stray database's file handle before removing it
    db_outcome = module.remove_stray_database(tmp_path, plan=False)
    yaml_outcomes = module.remove_stale_yaml(tmp_path, plan=False)

    assert record_outcomes == ["bud_opening: dropped provenance"]
    assert db_outcome is not None and db_outcome.startswith("removed stray database")
    assert yaml_outcomes == [f"removed stale spec file {yaml_path}"]

    _bind_sqlite()  # the previous connection was closed above; re-bind to read the record back
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "bud_opening")
    stored = ts.read(key)
    assert "provenance" not in stored
    assert stored["positive_class_name"] == "open"
    assert not stray_db.is_file()
    assert not yaml_path.is_file()
    assert [s.name for s in traits.load_trait_specs(project_root=tmp_path)] == ["bud_opening"]


def test_a_root_whose_write_raises_is_refused_and_a_second_root_still_conforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    """A ``StoreError`` from one root's write must not abort the whole invocation: the exit-code
    contract (0 conformed, 2 refused) only holds if a refusal is scoped to the root that raised,
    leaving every other root named on the same command line to conform normally."""
    _bind_ambient_backend()
    module = _load_script()
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    spec = {"name": "leaf", "delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(root_a, "leaf", spec)
    _seed_live_record(root_b, "leaf", spec)

    failing_state_root = str(traits._trait_specs_state_root(traits.trait_specs_dir(root_a)))
    real_replace = module.ts.replace

    def flaky_replace(key, value, *, expect):
        if key.root == failing_state_root:
            raise module.ts.StoreError("simulated write refusal")
        return real_replace(key, value, expect=expect)

    monkeypatch.setattr(module.ts, "replace", flaky_replace)
    monkeypatch.setattr(sys, "argv", ["drop_trait_spec_provenance.py", str(root_a), str(root_b)])

    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{root_a.resolve()}: refused, simulated write refusal" in output
    assert f"{root_b.resolve()}: leaf: dropped provenance" in output
    key_b = traits.trait_spec_key(traits.trait_specs_dir(root_b), "leaf")
    assert "provenance" not in ts.read(key_b)


def test_main_leaves_the_stray_database_and_stale_yaml_in_place_when_a_record_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    """The destructive-ordering fix: a root whose live record is refused (here, off-vocab
    ``delivers``, unrelated to ``provenance``) must not have its legacy copies removed. Those
    copies may be the only recoverable source of the trait's measurement definition."""
    backend = _bind_ambient_backend()
    module = _load_script()
    spec = {"name": "leaf", "delivers": ["not_a_real_phenotype"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(tmp_path, "leaf", spec)
    stray_db = _seed_stray_database(
        tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]}, restore=backend
    )
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")
    backend.close()  # release the stray database's file handle before main() opens it again

    monkeypatch.setattr(sys, "argv", ["drop_trait_spec_provenance.py", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{tmp_path.resolve()}: leaf: refused" in output
    assert "nothing is written or removed for this root" in output
    assert stray_db.is_file()
    assert yaml_path.is_file()
    _bind_ambient_backend()  # the previous connection was closed above; re-bind to read the record back
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert ts.read(key) == spec


def test_main_does_not_write_a_conforming_sibling_record_when_another_record_in_the_root_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Validation runs for the whole root before anything is written: a good record shares its
    root's refusal, rather than being conformed while its sibling is left broken."""
    _bind_ambient_backend()
    module = _load_script()
    good_spec = {"name": "leaf", "delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"]}
    bad_spec = {"name": "twig", "delivers": ["not_a_real_phenotype"]}
    _seed_live_record(tmp_path, "leaf", good_spec)
    _seed_live_record(tmp_path, "twig", bad_spec)

    monkeypatch.setattr(sys, "argv", ["drop_trait_spec_provenance.py", str(tmp_path)])
    exit_code = module.main()

    assert exit_code == 2
    leaf_key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert ts.read(leaf_key) == good_spec


def test_main_still_conforms_and_removes_legacy_copies_for_a_root_that_validates_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The rail admits valid work: a root whose record conforms cleanly still gets its stray
    database and stale YAML removed, the legitimate case the destructive-ordering fix must not
    break."""
    backend = _bind_ambient_backend()
    module = _load_script()
    spec = {"name": "leaf", "delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(tmp_path, "leaf", spec)
    stray_db = _seed_stray_database(
        tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]}, restore=backend
    )
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")
    backend.close()

    monkeypatch.setattr(sys, "argv", ["drop_trait_spec_provenance.py", str(tmp_path)])
    exit_code = module.main()

    assert exit_code == 0
    assert not stray_db.is_file()
    assert not yaml_path.is_file()
    _bind_ambient_backend()
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert "provenance" not in ts.read(key)


def test_main_plan_mode_reports_the_refusal_and_writes_or_removes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    """``--plan`` must surface a would-be refusal (so an operator sees it before running for
    real) while still writing and removing nothing."""
    backend = _bind_ambient_backend()
    module = _load_script()
    spec = {"name": "leaf", "delivers": ["not_a_real_phenotype"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(tmp_path, "leaf", spec)
    stray_db = _seed_stray_database(
        tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]}, restore=backend
    )
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")
    backend.close()

    monkeypatch.setattr(sys, "argv", ["drop_trait_spec_provenance.py", "--plan", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 2
    assert f"{tmp_path.resolve()}: leaf: refused" in output
    assert stray_db.is_file()
    assert yaml_path.is_file()
    _bind_ambient_backend()
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert ts.read(key) == spec


def test_main_plan_mode_previews_removal_for_a_root_that_validates_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    backend = _bind_ambient_backend()
    module = _load_script()
    spec = {"name": "leaf", "delivers": ["leaf_length"], "provenance": ["name: vocabulary_derived"]}
    _seed_live_record(tmp_path, "leaf", spec)
    stray_db = _seed_stray_database(
        tmp_path, "leaf", {"name": "leaf", "delivers": ["leaf_length"]}, restore=backend
    )
    yaml_path = _seed_stale_yaml(tmp_path, "leaf")
    backend.close()

    monkeypatch.setattr(sys, "argv", ["drop_trait_spec_provenance.py", "--plan", str(tmp_path)])
    exit_code = module.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert f"{tmp_path.resolve()}: leaf: would drop provenance" in output
    assert "would remove stray database" in output
    assert f"would remove stale spec file {yaml_path}" in output
    assert stray_db.is_file()
    assert yaml_path.is_file()
    _bind_ambient_backend()
    key = traits.trait_spec_key(traits.trait_specs_dir(tmp_path), "leaf")
    assert ts.read(key) == spec

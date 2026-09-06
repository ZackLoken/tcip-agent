"""The conform rail against the layouts the platform's own stores actually have.

The rail's mechanics are exercised with synthetic stores in ``test_store_conform_rail.py``.
What is asked here is whether the shipped claim table is right about real directories: a
project root holding a spreadsheet and a nested dataset, a workspace holding project folders,
an output directory that is a curated root and a dataset root at once. Every case addresses
shipped stores through their own key constructors, because the table's rows are what is under
test and a harness store would only restate the table's assumptions back at it.

Every store-owning module is imported, which is the condition the MCP server and the web
backend run in and the condition under which locator shapes collide worst: the workspace
marker's bare one-field key shape reads any single-segment file as its own, and the experiment
members' two-field json shape reads any json two directories down as theirs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

import tcip_store as ts
from tcip_store.file_backend import FileBackend
from tcip_store.sqlite_backend import SqliteBackend, database_path

import tcip_mcp.store_catalogue  # noqa: F401


@contextmanager
def bound(backend):
    """Bind one backend for a block, since these cases hand a root between two."""
    ts.bind(backend)
    try:
        yield backend
    finally:
        ts.unbind()
        backend.close()


def _image_status(root: Path) -> ts.Key:
    from tcip_mcp import dataset_layout

    return dataset_layout.image_status_key(root)


def _plant_mapping(root: Path) -> ts.Key:
    """A store rooted on ``.tcip/state``, addressed here at whatever root a case gives it."""
    from tcip_mcp.pipelines.postprocessing import plant_mapping

    return ts.Key(plant_mapping.PLANT_MAPPING_STORE, str(root), ("plant_mapping",))


def _coco_target(directory: Path, stem: str) -> ts.Key:
    """The caller-named blob target a coco export writes, which can be aimed anywhere."""
    from tcip_annotation.format_io import coco_document_key

    return coco_document_key(directory, stem)


# ── what the rail must admit ─────────────────────────────────────────────────


def test_a_project_root_holding_a_stray_csv_and_a_nested_dataset_is_admitted(tmp_path):
    """A project root holds files nothing in the seam owns and datasets whose own documents
    travel with them, and none of that is record state left behind. Asking every registered
    locator whether it could parse a path claimed the spreadsheet and the dataset's own
    documents, and refused this root permanently."""
    (tmp_path / "plants.csv").write_text("plot,accession\n1,ü\n", encoding="utf-8")
    dataset = tmp_path / "datasets" / "currant"
    (dataset / "images" / "2026-03-04").mkdir(parents=True)
    (dataset / "annotations" / "2026-03-04").mkdir(parents=True)
    (dataset / "images" / "2026-03-04" / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
    (dataset / "annotations" / "2026-03-04" / "a_1.json").write_text("[]", encoding="utf-8")
    (dataset / "classes.json").write_text('{"subjects": []}', encoding="utf-8")
    (dataset / "dataset.json").write_text('{"identity": "ü"}', encoding="utf-8")

    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)

        assert ts.read(_image_status(tmp_path)) == {"bud": {}}
    assert database_path(str(tmp_path)).is_file()


def test_a_workspace_root_holding_only_foreign_files_is_admitted(tmp_path, monkeypatch):
    """A workspace is a directory of project folders and a marker. Nothing else in it is the
    seam's, and the marker's one-part key shape claimed every single-segment file there, which
    made a spreadsheet beside the projects read as record state."""
    from tcip_mcp import workspace

    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path))
    (tmp_path / "notes.txt").write_text("ü", encoding="utf-8")
    (tmp_path / "measurements.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "currant_2026").mkdir()

    with bound(SqliteBackend()):
        ts.replace(workspace.active_project_key(), "currant_2026")

        assert ts.read(workspace.active_project_key()).strip() == "currant_2026"


# ── one directory, more than one kind of root ────────────────────────────────


def test_a_directory_that_is_two_kinds_of_root_is_served_end_to_end(tmp_path):
    """A curated materialization writes its manifest and the dataset documents it carries into
    one output directory, so that directory is a curated root and a dataset root at once. Every
    door has to serve it: the writes, the reads, and one transaction spanning both kinds."""
    from tcip_mcp import dataset_layout
    from tcip_mcp.pipelines.feedback import materialize

    out = tmp_path / "curated"
    out.mkdir()
    manifest = materialize.curated_manifest_key(out)
    digest = dataset_layout.image_status_digest_key(out)

    with bound(SqliteBackend()):
        ts.replace(manifest, {"images": ["a_1.jpg"]})
        ts.replace(digest, {"bud/2026-03-04": {"a_1.jpg": "9f2c"}}, expect=ts.Version.ABSENT)
        with ts.transaction(manifest, digest) as txn:
            txn.write(manifest, {"images": ["a_1.jpg", "b_2.jpg"]})
            txn.write(digest, {"bud/2026-03-04": {"a_1.jpg": "9f2c", "b_2.jpg": "1ab3"}})

        assert ts.read(manifest) == {"images": ["a_1.jpg", "b_2.jpg"]}
        assert ts.read(digest)["bud/2026-03-04"]["b_2.jpg"] == "1ab3"

    with bound(SqliteBackend()):
        assert ts.read(manifest) == {"images": ["a_1.jpg", "b_2.jpg"]}


def test_a_second_kinds_files_refuse_on_the_connection_that_already_served_the_first(tmp_path):
    """A connection opened for one kind of root must still answer for the second before it
    serves it. Checking once per connection would let a directory's other half stay invisible
    for that connection's whole life, which is the absence a database beside files produces."""
    from tcip_mcp.pipelines.feedback import materialize

    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)
        assert ts.read(_image_status(tmp_path)) == {"bud": {}}
        (tmp_path / "curated_manifest.json").write_text('{"images": []}', encoding="utf-8")

        with pytest.raises(ts.StoreError) as raised:
            ts.read(materialize.curated_manifest_key(tmp_path), default=None)

    message = str(raised.value)
    assert "curated_manifest" in message
    assert "tcip adopt-store" in message


def test_a_file_two_kinds_of_root_claim_equally_refuses_naming_every_claimant(tmp_path):
    """Two layouts' templates can describe one path: a free directory's ``metrics.jsonl`` is an
    experiment's metrics log under one kind of root and a trial's under another. At a directory
    serving both, no marker says whose the file is, so both claimants are named and neither is
    picked, the way the planner refuses a tie rather than attributing one store's log to
    another."""
    experiment = ts.Key("experiment_metrics", str(tmp_path), ("exp_1", "metrics"))
    trial = ts.Key("hpo_trial_metrics", str(tmp_path), ("trial_1", "metrics"))

    with bound(SqliteBackend()):
        ts.append(experiment, {"epoch": 1})
        (tmp_path / "trial_1").mkdir()
        (tmp_path / "trial_1" / "metrics.jsonl").write_text('{"epoch": 2}\n', encoding="utf-8")

        with pytest.raises(ts.StoreError) as raised:
            ts.read_log(trial)

    message = str(raised.value)
    assert "experiment_metrics" in message and "hpo_trial_metrics" in message
    assert "metrics.jsonl" in message


def test_a_file_of_a_store_the_database_never_held_refuses_beside_it(tmp_path):
    """With a database present a claimed file is ordinarily that store's own export, so the
    accounting is per store: a store the database has never held is the one whose file no
    export can explain, and reading past it answers every one of its entries with absence."""
    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)
    state = tmp_path / ".tcip" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "view_coverage.json").write_text('{"bud/2026-03-04": {}}', encoding="utf-8")

    with bound(SqliteBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.read(_image_status(tmp_path), default=None)

    assert "view_coverage" in str(raised.value)
    assert "tcip adopt-store" in str(raised.value)


def test_a_process_that_imported_one_owning_module_still_sees_another_stores_files(tmp_path):
    """Which store owns which path is data in the seam rather than the live registry, so which
    modules a process happened to import cannot decide whether a root's state was left behind.
    Asked of the registry, a process that never imported the owning module read this root as
    fresh, created a database over its confirmed negatives, and answered them as absent."""
    with bound(FileBackend()):
        ts.replace(_image_status(tmp_path), {"bud/2026-03-04": {}}, expect=ts.Version.ABSENT)
    program = (
        "import json, sys\n"
        "import tcip_annotation.review_engine\n"
        "from tcip_store.layout_claims import ROOT, unconformed_files\n"
        "print(json.dumps(sorted(p.name for p in unconformed_files(sys.argv[1], ROOT))))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        capture_output=True, text=True, check=True,
    )

    assert json.loads(result.stdout) == ["image_status.json"]


def test_the_priority_queue_and_hpo_registries_are_claimed_beside_inference_jobs(tmp_path):
    """``job_registry`` places three documents under ``.tcip/state``: ``inference_jobs.json``,
    ``review_priority_jobs.json`` and ``hpo_sweeps.json``. A claim naming only the first left
    the other two invisible to the conform rail and the adoption planner even though the same
    store and locator write them; each has to be claimed once written through its own route."""
    from tcip_store.layout_claims import ROOT, unconformed_files
    from tcip_web.routes import review, tuning

    with bound(FileBackend()):
        review._pq_register(
            review.PriorityQueueJob(job_id="pq1", checkpoint_path="c", images_dir="i",
                                     dataset_root="d")
        )
        with tuning._lock:
            tuning._registry.jobs["hpo1"] = tuning.HPOJob(sweep_id="hpo1")
        tuning._persist()

        names = {p.name for p in unconformed_files(str(tmp_path), ROOT)}
    review._pq_registry.jobs.clear()
    tuning._registry.jobs.clear()

    assert "review_priority_jobs.json" in names
    assert "hpo_sweeps.json" in names


# ── the blob write that would land on a record's own path ────────────────────


def test_a_blob_written_onto_a_records_claimed_path_beside_a_database_is_refused(tmp_path):
    """A blob write bypasses the record rail by design, and a caller-named target can be aimed
    at a record's own file. Landing it beside a database is a write that database never sees,
    whichever door it came through, and the refusal does not wait to see whether the claiming
    store is one this database happens to hold: a test against markers is defeatable in both
    directions, since a store's first write can mint markers on a cached connection and a read
    can serve honest absence while the file idles, neither of which reaches another process."""
    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)

    with bound(FileBackend()):
        with pytest.raises(ts.StoreError) as held_store:
            ts.put_blob(_coco_target(tmp_path / ".tcip" / "state", "image_status"), b"{}")
        with pytest.raises(ts.StoreError) as never_held_store:
            ts.put_blob(_coco_target(tmp_path / ".tcip" / "state", "view_coverage"), b"{}")

    assert "image_status" in str(held_store.value)
    assert "view_coverage" in str(never_held_store.value)
    assert not (tmp_path / ".tcip" / "state" / "image_status.json").exists()
    assert not (tmp_path / ".tcip" / "state" / "view_coverage.json").exists()


def test_a_caller_named_output_that_collides_with_a_claim_is_refused_by_name(tmp_path):
    """The accepted cost of refusing unconditionally, pinned as behavior rather than left for
    someone to discover: an export whose filename happens to be a record's own is refused beside
    a database even where the author meant no harm. The message has to carry the store whose
    path it is and what to do, because renaming the output is the whole remedy."""
    from tcip_mcp.pipelines.feedback import materialize

    with bound(SqliteBackend()):
        ts.replace(materialize.curated_manifest_key(tmp_path), {"images": []})

    with bound(FileBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.put_blob(_coco_target(tmp_path, "curated_manifest"), b"{}")

    message = str(raised.value)
    assert "curated_manifest" in message
    assert "Rename the output" in message


def test_a_stores_first_write_is_refused_while_a_file_it_claims_predates_it(tmp_path):
    """Once a store holds rows here, a file claiming it reads as its own export, so the moment
    that matters is the write that gives it its first row: after it, a file that predated the
    store stops being visible to the accounting. The connection here has already served this
    kind of root, so nothing would walk it again; the guard is what refuses."""
    from tcip_mcp import dataset_layout

    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)
        state = tmp_path / ".tcip" / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "view_coverage.json").write_text('{"bud": {}}', encoding="utf-8")

        with pytest.raises(ts.StoreError) as raised:
            ts.replace(
                dataset_layout.view_coverage_key(tmp_path), {"bud": {}},
                expect=ts.Version.ABSENT,
            )

    message = str(raised.value)
    assert "view_coverage.json" in message
    assert "tcip adopt-store" in message


def test_a_stores_first_write_lands_when_no_file_of_its_own_predates_it(tmp_path):
    """The partner of the guard above: a store arriving at a root that holds a database is
    ordinary work, and the guard may only fire on a file that would become invisible."""
    from tcip_mcp import dataset_layout

    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)

        ts.replace(
            dataset_layout.view_coverage_key(tmp_path),
            {"bud/2026-03-04": {"a_1.jpg": {"grid": {"rows": 1, "cols": 1}}}},
            expect=ts.Version.ABSENT,
        )

        assert ts.read(dataset_layout.view_coverage_key(tmp_path))["bud/2026-03-04"]


def test_an_ordinary_blob_write_beside_a_database_takes_no_lock_and_creates_no_state_dir(
    tmp_path, monkeypatch
):
    """The admit half, asserted rather than assumed: imagery, labels and an export to a plain
    ``<name>.json`` must not pay for the collision above, and a lock taken on their behalf
    would put a ``.tcip`` directory inside every output directory the platform writes into."""
    from tcip_annotation import json_io
    from tcip_mcp import dataset_layout
    from tcip_store import file_backend as file_backend_module

    with bound(SqliteBackend()):
        ts.replace(_image_status(tmp_path), {"bud": {}}, expect=ts.Version.ABSENT)
    output = tmp_path / "exports" / "2026-03-04"
    output.mkdir(parents=True)
    locked: list[str] = []
    real_lock = file_backend_module.transition_lock

    def counted(root, **kwargs):
        locked.append(str(root))
        return real_lock(root, **kwargs)

    monkeypatch.setattr(file_backend_module, "transition_lock", counted)

    with bound(FileBackend()):
        ts.put_blob(_coco_target(output, "annotations"), b"{}")
        ts.put_blob(
            json_io.annotation_record_key(
                dataset_layout.annotation_dir(tmp_path, "2026-03-04"), "a_1"
            ),
            b"[]",
        )
        ts.put_blob(dataset_layout.image_key(tmp_path, "2026-03-04", "a_1", ".jpg"), b"\xff\xd8")

    assert locked == []
    assert not (output / ".tcip").exists()
    assert (output / "annotations.json").read_bytes() == b"{}"


def test_a_blob_target_matching_two_claims_locks_both_roots_and_refuses_at_the_database(
    tmp_path,
):
    """One path can be a legal entry of one store under two different roots, so the writer
    cannot pick one and hope: it holds every candidate before it decides, in a fixed order, so
    a concurrent creator cannot deadlock against it, and it refuses at whichever candidate
    holds a database."""
    inner = tmp_path / "review"
    inner.mkdir()
    with bound(SqliteBackend()):
        ts.replace(_plant_mapping(inner), {"2026-03-04": []})

    with bound(FileBackend()):
        with pytest.raises(ts.StoreError) as raised:
            ts.put_blob(_coco_target(inner / "review", "a_1.jpg"), b"{}")

    assert "review_verdicts" in str(raised.value)
    assert (tmp_path / ".tcip").is_dir()
    assert not (inner / "review" / "a_1.jpg.json").exists()


def test_a_colliding_blob_write_and_a_database_creation_never_both_land(tmp_path):
    """The two publishers race on one lock, so whichever runs second sees the first: either the
    database exists and the blob was refused, or the blob's file is there and the creation
    refused the root it had made unconformed. Both landing is the interleaving that loses a
    write, and it must be out of reach from either starting order."""
    for order in (0, 1):
        root = tmp_path / f"order_{order}"
        root.mkdir()
        outcomes: dict[str, str] = {}
        ready = threading.Barrier(2)

        def create(root=root, outcomes=outcomes, ready=ready) -> None:
            backend = SqliteBackend()
            try:
                ready.wait(timeout=30)
                backend.replace(_image_status(root), {"bud": {}}, expect=ts.Version.ABSENT)
                outcomes["create"] = "created"
            except ts.StoreError as exc:
                outcomes["create"] = f"refused: {exc}"
            finally:
                backend.close()

        def collide(root=root, outcomes=outcomes, ready=ready) -> None:
            backend = FileBackend()
            try:
                ready.wait(timeout=30)
                backend.put_blob(_coco_target(root / ".tcip" / "state", "image_status"), b"{}")
                outcomes["blob"] = "written"
            except ts.StoreError as exc:
                outcomes["blob"] = f"refused: {exc}"
            finally:
                backend.close()

        threads = [threading.Thread(target=create), threading.Thread(target=collide)]
        for thread in threads if order == 0 else list(reversed(threads)):
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        made_a_database = database_path(str(root)).is_file()
        wrote_the_file = (root / ".tcip" / "state" / "image_status.json").is_file()
        assert not (made_a_database and wrote_the_file), outcomes
        assert made_a_database or wrote_the_file, outcomes
        if made_a_database:
            assert outcomes["blob"].startswith("refused"), outcomes
        else:
            assert outcomes["create"].startswith("refused"), outcomes

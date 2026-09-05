"""Shared test fixtures."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _pin_torch_single_thread():
    """Pin torch to one intra-op thread for the test session.

    The fixtures are tiny (1-2 epochs on ≤4 images), so a multi-thread pool is pure overhead;
    single-thread is also a prerequisite for pytest-xdist (else N workers oversubscribe the
    cores). Lazy-import so a torch-less collection still works.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


def _drain_background_store_writers() -> None:
    """Join any background thread still writing through the bound backend.

    A database backend closes its connections, so closing one under a thread mid-statement
    frees the connection that statement is running on, which native SQLite answers by taking
    the whole worker process down rather than by raising. The modules that spawn such threads
    are looked up in ``sys.modules`` rather than imported, so a test that never touched the web
    package pays nothing for this. A worker that outlasts the wait is raised rather than closed
    under, so the failure is legible instead of a crash.
    """
    tuning = sys.modules.get("tcip_web.routes.tuning")
    if tuning is None:
        return
    from tcip_store.file_backend import DEFAULT_LOCK_TIMEOUT_S

    # A worker's slowest single act is one store write, bounded by the seam's own lock wait.
    bound = 2 * DEFAULT_LOCK_TIMEOUT_S
    still_running = tuning.wait_for_workers(timeout_s=bound)
    if still_running:
        raise RuntimeError(
            f"sweep workers {', '.join(still_running)} were still writing after {bound}s, so "
            "this test's storage backend was not closed under them. The test that launched "
            "them has to wait on tcip_web.routes.tuning.wait_for_workers before returning."
        )


@pytest.fixture(autouse=True)
def _bind_storage_backend():
    """Bind the storage backend before every test, the way a process entry point does.

    Every code path that reaches a store needs one bound; a suite that left it unbound would
    report the absence of a backend where the behavior under test is what the store does. Per
    test rather than per session, so a test that binds its own backend and drops it on the way
    out leaves the next one a bound process rather than an unbound one. Closed on the way out
    so a database backend leaves no open handle on the test's tmp_path, which Windows would
    then refuse to remove, and drained first so no background writer is still holding it.
    """
    from tcip_store.binding import bind_default

    backend = bind_default()
    yield
    _drain_background_store_writers()
    backend.close()


@pytest.fixture(scope="session", autouse=True)
def _stop_leaked_tensorboards():
    """Leave no TensorBoard process behind at the end of the test session.

    A test that drives the real ``launch_training`` gets TensorBoard as a best-effort side
    effect; a test that forgets to stub it out leaks the child process, its cwd pinned to the
    test's own (soon-deleted) directory. The module is looked up in ``sys.modules`` rather than
    imported, so a session that never touched it pays nothing; a session with nothing tracked
    stops nothing.
    """
    yield
    tb = sys.modules.get("tcip_mcp.pipelines.training.tensorboard_manager")
    if tb is None:
        return
    for entry in tb.list_tensorboard():
        tb.stop_tensorboard(run_id=entry["key"])


@pytest.fixture(autouse=True)
def _discard_leftover_launch_marks():
    """Leave no pre-manifest sweep-launch mark behind for the next test.

    A test that calls ``mark_sweep_launching`` directly, or drives a relaunch whose worker
    never reaches ``run_hyperparameter_search``, can leave a study name in the process-global mark set; the next
    test's ``cancel_hyperparameter_search`` call must not find it. Discarded through the same call ``run_hyperparameter_search``
    and the relaunch worker use, not by reaching into the dict directly. The module is looked
    up in ``sys.modules`` rather than imported, so a session that never touched it pays nothing.
    """
    yield
    tt = sys.modules.get("tcip_mcp.tools.training_tools")
    if tt is None:
        return
    for study_name in list(tt._LAUNCHING_SWEEPS):
        tt.discard_sweep_launching(study_name)


def pytest_collection_modifyitems(config, items):
    """Guardrail: fail loudly when far fewer tests collect than expected.

    Catches the failure mode where a missing dependency makes ~15 files module-level
    ``importorskip`` at collection time, shrinking the suite, while CI still reports
    green. CI sets ``TCIP_MIN_TESTS`` to a floor safely below the real count; a large
    shortfall means a core dep (torch/torchvision/pycocotools/...) is absent.
    """
    floor = os.environ.get("TCIP_MIN_TESTS")
    if floor and len(items) < int(floor):
        raise pytest.UsageError(
            f"Collected only {len(items)} tests (< TCIP_MIN_TESTS={floor}). A core "
            "dependency is likely missing: module-level importorskip silently skipped files."
        )


@pytest.fixture(autouse=True)
def _restore_platform_root_env():
    """Keep the process-global platform-state root hermetic across tests.

    ``activate_project`` repins ``TCIP_STATE_ROOT`` in-process (so a project's audit /
    experiments / registry co-locate under it). Since pytest runs in one process, a test
    that adopts a tmp project would otherwise leak that now-deleted root, and the
    :class:`~tcip_mcp.project_paths.RootBinding` naming it, into later tests. Snapshot and
    restore both the var and the binding around every test.
    """
    from tcip_mcp import project_paths

    saved = os.environ.get("TCIP_STATE_ROOT")
    saved_binding = project_paths.root_binding()
    yield
    if saved is None:
        os.environ.pop("TCIP_STATE_ROOT", None)
    else:
        os.environ["TCIP_STATE_ROOT"] = saved
    # getattr, not a plain import: a fail-before run against a tree that predates
    # restore_binding must still collect every other test in this file.
    restore = getattr(project_paths, "restore_binding", None)
    if restore is not None:
        restore(saved_binding)


@pytest.fixture
def tmp_path(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest,
             monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test's ``tmp_path`` is a project directory inside its own fresh workspace.

    The web layer's path guard admits only the workspace, the roots registered to its projects,
    and ``TCIP_IMAGE_ROOTS``. Laying ``tmp_path`` out as ``<workspace>/project`` gives every test
    the production topology (a workspace root distinct from the project under it, and the
    platform-state pin below on the project, as ``workspace.activate_project`` leaves it) with
    no fixture naming the layout. A refusal case uses a directory beside the workspace
    (``tmp_path_factory.mktemp``); a test of the additive roots sets ``TCIP_IMAGE_ROOTS`` itself.
    """
    name = re.sub(r"[\W]", "_", request.node.name)[:30]
    workspace = tmp_path_factory.mktemp(name, numbered=True) / "workspace"
    project = workspace / "project"
    project.mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(workspace))
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    return project


@pytest.fixture(autouse=True)
def _close_open_project():
    """Leave the web layer's open project closed after each test.

    The GUI state store is a process-wide singleton; a Results door serves only the project it
    has open, so a test that opened one must not hand it to the next test.
    """
    yield
    state = sys.modules.get("tcip_web.state")
    if state is not None:
        state.store.close_project()


@pytest.fixture(autouse=True)
def _pin_platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pin every test's platform-state root to its own unique ``tmp_path``.

    Without this, any unpinned write (audit, experiments, jobstore, the vision candidates
    cache) resolves relative to the process CWD and lands in the repo's real ``.tcip/``,
    shared across tests and, under xdist, across worker processes. Uses monkeypatch so it
    auto-restores; a test that manages the var itself (setenv/delenv in its body) overrides
    this and is unaffected.
    """
    monkeypatch.setenv("TCIP_STATE_ROOT", str(tmp_path))


@pytest.fixture
def seed_bud_trait_spec(tmp_path: Path, _pin_platform_root):
    """Seed a real bud_opening trait-spec record into this test's pinned platform state root.

    There are no built-in traits anymore: ``get_trait("bud_opening")`` only resolves where a spec
    record actually exists (``traits.py``). Writing the same values
    ``tests/_trait_fixtures.BUD_OPENING`` holds keeps a test that calls
    ``get_trait("bud_opening")``/``registered_traits()`` without authoring its own spec working,
    the same as when a builtin was unconditionally present. Not autouse: an unrelated test's
    platform state root should stay empty by default; request this explicitly in a test that
    actually needs bud_opening registered.
    """
    import dataclasses

    import tcip_store as ts

    from tcip_mcp import traits
    from tests._trait_fixtures import BUD_OPENING

    specs_dir = tmp_path / ".tcip" / "state" / "trait_specs"
    data = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(BUD_OPENING).items()}
    ts.replace(traits.trait_spec_key(specs_dir, "bud_opening"), data, expect=ts.Version.ABSENT)


@pytest.fixture
def seed_bud_operationalization(tmp_path: Path, seed_bud_trait_spec):
    """Give the seeded bud_opening spec a confirmed ``state_crossing_dates`` record in the same root.

    The crossing delivery doors refuse a trait whose delivered number has no breeder-confirmed
    meaning, so a test whose subject is the delivery itself seeds one here and goes on testing what
    it was written to test. Requested alongside ``seed_bud_trait_spec`` rather than autouse, for
    the same reason that one is not: a test of the refusal needs the root without a record in it.
    The trait's own name is not the subject it measures, so the measured subject is named
    explicitly rather than left to default to the trait name.
    """
    from tests._operationalization_fixtures import seed_confirmed_crossing

    seed_confirmed_crossing(tmp_path, "bud_opening", measured_subject="bud")


@pytest.fixture
def real_hpo_base_config(tmp_path: Path) -> dict:
    """A base config the sweep door's own structural preflight admits: an importable builder and
    a data section whose directories exist, so a sweep test exercises the search itself rather
    than the door's refusal. Held admitted by
    test_split_draws.test_real_hpo_base_config_is_admitted_by_preflight, which runs it through
    preflight_config directly."""
    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir(exist_ok=True)
    lbls.mkdir(exist_ok=True)
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {"num_classes": 1}, "task": "detection"},
        "data": {"images_dir": str(imgs), "labels_dir": str(lbls)},
    }


#: The single detection subject the canonical test dataset declares (``data_dir``).
DATA_DIR_SUBJECT = "bud"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A minimal dataset in the canonical layout: name-based per-image labels + a registry.

    One file per image under ``annotations/<date>/`` holding every subject (here one detection
    subject, ``bud``), predictions under ``predictions/<model>/<date>/``, and one nested
    ``classes.json``. Geometry is two boxes per image on a 640x480 frame, matching the
    count/geometry expectations downstream.
    """
    from PIL import Image

    from tcip_annotation import json_io
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp import class_registry
    from tcip_mcp.class_registry import ClassRegistry, Subject

    date = "2-11-26"
    subject = DATA_DIR_SUBJECT
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)
    labels_dir = tmp_path / "annotations" / date
    labels_dir.mkdir(parents=True)
    preds_dir = tmp_path / "predictions" / "live" / date
    preds_dir.mkdir(parents=True)

    # One nested registry travelling with the labels: a single detection subject, no attributes.
    class_registry.write_registry(
        tmp_path / "classes.json",
        ClassRegistry(subjects=(Subject(name=subject, description="a bud"),)),
    )

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{name}.jpg")
        # GT: 2 boxes per image (pixel xyxy), by subject name.
        json_io.write_annotations(
            labels_dir / f"{name}.json",
            [Annotation(subject=subject, geometry=BBox(288, 216, 352, 264)),
             Annotation(subject=subject, geometry=BBox(176, 132, 208, 156))],
            640, 480,
        )
        # Predictions: 1 matching (TP) + 1 elsewhere (FP), the confidence in each annotation's score.
        json_io.write_annotations(
            preds_dir / f"{name}.json",
            [Annotation(subject=subject, geometry=BBox(288, 216, 352, 264), score=0.9),
             Annotation(subject=subject, geometry=BBox(496, 372, 528, 396), score=0.7)],
            640, 480,
        )

    return tmp_path

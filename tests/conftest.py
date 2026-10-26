"""Shared test fixtures."""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient

    from tcip_mcp.workspace import BoundProject
    from tcip_web.state import OpenProject

from tests.hung_test_watchdog import pytest_timeout_set_timer
from tests.ray_cluster_lock import pytest_configure, pytest_runtest_protocol

__all__ = [
    "pytest_configure",
    "pytest_runtest_protocol",
    "pytest_timeout_set_timer",
]


@pytest.fixture(scope="session", autouse=True)
def _pin_torch_single_thread() -> None:
    """Default ``OMP_NUM_THREADS`` to 1 and pin torch, when installed, to one intra-op thread."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


@pytest.fixture(scope="session", autouse=True)
def _import_ray_before_any_popen_substitute() -> None:
    """Import ``ray._private.services`` at the session's start, when Ray is installed."""
    try:
        import ray._private.services  # noqa: F401
    except ImportError:
        return


@pytest.fixture(scope="session", autouse=True)
def _stop_leaked_tensorboards() -> Iterator[None]:
    """Stop every TensorBoard process the session started, when the manager module is loaded."""
    yield
    tb = sys.modules.get("tcip_mcp.pipelines.training.tensorboard_manager")
    if tb is None:
        return
    tb._stop_all_tracked()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Refuse a collection of fewer items than ``TCIP_MIN_TESTS`` names, when it is set.

    The refusal is a ``pytest.UsageError`` naming the count collected and the floor.
    """
    floor = os.environ.get("TCIP_MIN_TESTS")
    if floor and len(items) < int(floor):
        raise pytest.UsageError(
            f"Collected only {len(items)} tests (< TCIP_MIN_TESTS={floor}). A core "
            "dependency is likely missing: module-level importorskip silently skipped files."
        )


@pytest.fixture
def _backend(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Iterator[tuple[Path, TestClient]]:
    """Yield each test's own fresh workspace and a client of the web backend serving it.

    The backend is ``tests._web_fixtures.backend_session``'s. The workspace's root is emptied
    after a test that passed under the ``failed`` retention policy and kept whole after one that
    failed; the emptied root itself stays.
    """
    from _pytest.tmpdir import tmppath_result_key

    from tests._web_fixtures import backend_session

    name = re.sub(r"[\W]", "_", request.node.name)[:30]
    root = tmp_path_factory.mktemp(name, numbered=True).resolve()
    workspace = root / "workspace"
    workspace.mkdir()
    with backend_session(workspace) as client:
        yield workspace, client
    policy = request.config.getini("tmp_path_retention_policy")
    outcome = request.node.stash.get(tmppath_result_key, {})
    if policy == "failed" and outcome.get("call", True):
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


@pytest.fixture
def tmp_path(_backend: tuple[Path, TestClient]) -> Path:
    """Make and return ``<workspace>/project`` in :func:`_backend`'s workspace as ``tmp_path``."""
    project = _backend[0] / "project"
    project.mkdir()
    return project


@pytest.fixture(autouse=True)
def _own_workspace(tmp_path: Path) -> None:
    """Give every test its own workspace, whether or not it names ``tmp_path``."""


@pytest.fixture
def made(tmp_path: Path) -> OpenProject:
    """Make ``tmp_path`` a project through the platform's own creation door.

    The door is reached through ``tests._web_fixtures.new_project``.
    """
    from tests._web_fixtures import new_project

    return new_project(tmp_path)


@pytest.fixture
def project(made: OpenProject) -> Path:
    """Return :func:`made`'s directory."""
    return made.root


@pytest.fixture
def bound(made: OpenProject) -> BoundProject:
    """Return :func:`made` as a ``BoundProject`` (``tests._web_fixtures.bound_to``)."""
    from tests._web_fixtures import bound_to

    return bound_to(made)


@pytest.fixture
def opened(tmp_path: Path) -> OpenProject:
    """Make ``tmp_path`` a project and open it in the web backend.

    It is opened by ``tests._web_fixtures.open_new_project``.
    """
    from tests._web_fixtures import open_new_project

    return open_new_project(tmp_path)


@pytest.fixture
def opened_project(opened: OpenProject) -> Path:
    """Return :func:`opened`'s directory."""
    return opened.root


@pytest.fixture
def client(_backend: tuple[Path, TestClient]) -> TestClient:
    """Return a client of the web backend serving this test's workspace (:func:`_backend`).

    The client's base URL is ``http://127.0.0.1``.
    """
    return _backend[1]


@pytest.fixture
def opened_client(opened_project: Path, client: TestClient) -> TestClient:
    """Return :func:`client` with :func:`opened_project` open in the backend."""
    return client


@pytest.fixture
def fake_terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Launch the fake agent; afterwards end every session and forget the survivors of it.

    The fake is ``tests/fake_terminal_app.py`` through ``TCIP_TERMINAL_CMD``.
    """
    from tcip_web.routes import terminal as terminal_routes
    from tcip_web.terminal import TERMINAL_CMD_ENV
    from tests._terminal_fixtures import FAKE

    monkeypatch.setenv(TERMINAL_CMD_ENV, f"{sys.executable} -u {FAKE}")
    yield
    terminal_routes.shutdown_all()
    terminal_routes._SESSIONS.clear()


@pytest.fixture
def tb_launches(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Path]]:
    """Record every directory handed to ``launch_tensorboard``, its child a listening stand-in.

    The launch itself is the manager's own; every child it tracks is stopped at teardown.
    """
    from tcip_mcp.pipelines.training import tensorboard_manager as tb
    from tests._tensorboard_parent import listening_argv

    calls: list[Path] = []
    launch = tb.launch_tensorboard

    def recording_launch(directory: Path) -> tb.TensorboardLaunch:
        calls.append(directory)
        return launch(directory)

    monkeypatch.setattr(
        tb,
        "_tensorboard_argv",
        lambda logdir, port, prefix: listening_argv(port, seconds=30),
    )
    monkeypatch.setattr(tb, "launch_tensorboard", recording_launch)
    yield calls
    tb._stop_all_tracked()


@pytest.fixture
def seed_bud_trait_spec(tmp_path: Path) -> None:
    """Propose and confirm ``tests/_trait_fixtures.BUD_OPENING`` in ``tmp_path``'s project."""
    from tests._trait_fixtures import BUD_OPENING, propose_and_confirm

    propose_and_confirm(tmp_path, BUD_OPENING)


@pytest.fixture
def seed_bud_operationalization(
    tmp_path: Path, seed_bud_trait_spec: None
) -> None:
    """Confirm a revision of bud_opening stating a ``state_crossing_dates`` operationalization.

    The operationalization measures the subject ``bud``, in ``tmp_path``'s project.
    """
    from tests._trait_fixtures import seed_confirmed_crossing

    seed_confirmed_crossing(tmp_path, "bud_opening", measured_subject="bud")


@pytest.fixture
def confirmed_count_aggregate(tmp_path: Path) -> None:
    """Confirm every delivery trait of ``tests/_trait_fixtures`` and one aggregate meaning.

    The project is ``tmp_path``; the aggregate meaning confirmed is the per-plant one of the
    trait delivering ``stem_count``.
    """
    from tests import _trait_fixtures as fx

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])


@pytest.fixture
def hpo_base_config_at(tmp_path: Path) -> Callable[[int], dict]:
    """Return a producer of ``tmp_path``'s base configs, one per split seed.

    The producer takes the seed its split draws at; each config holds an importable builder and
    a data section over two labeled frames of its subject
    (``_verified_checkpoint_fixtures.detection_images``), at the sample regime and the statistics
    of the draw's train side (``_chain_fixtures.training_config``).
    """
    from tests._chain_fixtures import training_config
    from tests._training_values import drawn_split
    from tests._verified_checkpoint_fixtures import (
        BUILT_DETECTOR,
        detection_images,
    )

    scope = {"subject": DATA_DIR_SUBJECT}

    def at(seed: int) -> dict:
        return training_config(
            tmp_path,
            BUILT_DETECTOR,
            {
                **detection_images(tmp_path / "hpo-data", scope),
                "scope": scope,
                "split": drawn_split(
                    seed=seed, val_ratio=0.15, grouping="tile_prefix"
                ),
            },
        )

    return at


@pytest.fixture
def real_hpo_base_config(
    hpo_base_config_at: Callable[[int], dict],
) -> dict:
    """Return the base config :func:`hpo_base_config_at` produces at seed 0."""
    return hpo_base_config_at(0)


#: The single detection subject the canonical test dataset declares (``data_dir``).
DATA_DIR_SUBJECT = "bud"
#: The bucket the canonical test dataset publishes (``data_dir``).
DATA_DIR_BUCKET = "live/2-11-26"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Write three labeled images of capture ``2-11-26``, a registry and a published bucket.

    The registry declares the one subject ``bud``. Each image is 640x480 gray, its label
    document holding boxes (288, 216, 352, 264) and (176, 132, 208, 156); the bucket
    ``live/2-11-26`` publishes boxes (288, 216, 352, 264) at 0.9 and (496, 372, 528, 396) at 0.7
    on each.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import Subject, SubjectRegistry
    from tests._producer_fixtures import label_image, registry_over

    date = "2-11-26"
    subject = DATA_DIR_SUBJECT
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)

    # One nested registry traveling with the labels: a single detection subject, no attributes.
    registry_over(
        tmp_path,
        SubjectRegistry(
            subjects=(Subject(name=subject, description="a bud"),)
        ),
    )

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(
            images_dir / f"{name}.jpg"
        )
        # GT: 2 boxes per image (pixel xyxy), by subject name.
        label_image(
            images_dir / f"{name}.jpg",
            [
                Annotation(subject=subject, geometry=BBox(288, 216, 352, 264)),
                Annotation(subject=subject, geometry=BBox(176, 132, 208, 156)),
            ],
            640,
            480,
        )
    # Predictions, published as one bucket: 1 matching (TP) + 1 elsewhere (FP) per image.
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(
        tmp_path,
        DATA_DIR_BUCKET,
        [
            {
                "image": str(images_dir / f"{name}.jpg"),
                "width": 640,
                "height": 480,
                "boxes": [[288, 216, 352, 264], [496, 372, 528, 396]],
                "scores": [0.9, 0.7],
                "labels": [1, 1],
                "cap": 3,
            }
            for name in ("img_001", "img_002", "img_003")
        ],
        scope={"subject": subject},
    )
    return tmp_path

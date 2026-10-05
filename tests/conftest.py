"""Shared test fixtures."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from tests.hung_test_watchdog import pytest_timeout_set_timer
from tests.ray_cluster_lock import pytest_configure, pytest_runtest_protocol

__all__ = ["pytest_configure", "pytest_runtest_protocol", "pytest_timeout_set_timer"]


@pytest.fixture(scope="session", autouse=True)
def _pin_torch_single_thread():
    """Pin torch to one intra-op thread for the test session, when torch is installed."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _bind_storage_backend():
    """Bind the storage backend before every test, the way a process entry point does, and close
    it after the test."""
    import tcip_store

    backend = tcip_store.bind()
    yield
    backend.close()


@pytest.fixture(scope="session", autouse=True)
def _stop_leaked_tensorboards():
    """Stop every TensorBoard process the session started, when the manager module is loaded."""
    yield
    tb = sys.modules.get("tcip_mcp.pipelines.training.tensorboard_manager")
    if tb is None:
        return
    tb._stop_all_tracked()


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


@pytest.fixture
def tmp_path(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest,
             monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each test's ``tmp_path`` is ``<workspace>/project`` inside its own fresh workspace, which
    ``TCIP_WORKSPACE`` names and the web backend is started with, no additive image roots.

    A refusal case uses a directory beside the workspace (``tmp_path_factory.mktemp``); a test of
    the additive roots starts the backend with them itself.
    """
    from tcip_web.state import store

    name = re.sub(r"[\W]", "_", request.node.name)[:30]
    workspace = (tmp_path_factory.mktemp(name, numbered=True) / "workspace").resolve()
    project = workspace / "project"
    project.mkdir(parents=True)
    monkeypatch.setenv("TCIP_WORKSPACE", str(workspace))
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    store.configure(workspace, ())
    return project


@pytest.fixture(autouse=True)
def _own_workspace(tmp_path: Path) -> None:
    """Give every test its own workspace, whether or not it names ``tmp_path``, so a record kept
    under the workspace root (the backend's port, the last-opened pointer) never reaches the next
    test."""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """``tmp_path`` made a project through the platform's own creation door."""
    from tests._web_fixtures import new_project

    return new_project(tmp_path)


@pytest.fixture
def opened_project(tmp_path: Path) -> Path:
    """``tmp_path`` made a project and open in the web backend, the way the picker's open
    leaves it."""
    from tests._web_fixtures import open_new_project

    return open_new_project(tmp_path)


@pytest.fixture
def client():
    """A client of the web backend at the loopback address a browser on this machine uses."""
    from fastapi.testclient import TestClient

    from tcip_web.app import app

    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def opened_client(opened_project: Path, client):
    """:func:`client` with :func:`opened_project` open in the backend."""
    return client


@pytest.fixture
def tb_launches(monkeypatch) -> list[str]:
    """Record every logdir handed to ``launch_tensorboard`` instead of starting a real one."""
    calls: list[str] = []

    def fake_launch(logdir: str) -> dict:
        calls.append(logdir)
        from tcip_mcp.web_client import LOOPBACK_HOST

        return {"url": f"http://{LOOPBACK_HOST}:6006", "port": 6006, "pid": 1, "logdir": logdir}

    monkeypatch.setattr(
        "tcip_mcp.pipelines.training.tensorboard_manager.launch_tensorboard", fake_launch
    )
    return calls


@pytest.fixture(autouse=True)
def _close_open_project():
    """Leave the web layer's process-wide state store with no project open after each test."""
    yield
    state = sys.modules.get("tcip_web.state")
    if state is not None and state.store.project_root is not None:
        import asyncio

        asyncio.run(state.store.close_project())


@pytest.fixture
def seed_bud_trait_spec(tmp_path: Path):
    """Propose and confirm ``tests/_trait_fixtures.BUD_OPENING`` in this test's ``tmp_path``
    project, so measurement readers of ``bud_opening`` resolve. Not autouse: an unrelated
    test's project stays empty.
    """
    from tests._trait_fixtures import BUD_OPENING, propose_and_confirm

    propose_and_confirm(tmp_path, BUD_OPENING)


@pytest.fixture
def seed_bud_operationalization(tmp_path: Path, seed_bud_trait_spec):
    """Confirm a revision of bud_opening stating a ``state_crossing_dates`` operationalization of
    the subject ``bud``, in the same root. Never autouse, so a test of the refusal gets a root
    without one.
    """
    from tests._trait_fixtures import seed_confirmed_crossing

    seed_confirmed_crossing(tmp_path, "bud_opening", measured_subject="bud")


@pytest.fixture
def confirmed_count_aggregate(tmp_path: Path) -> None:
    """Confirm, in this test's ``tmp_path`` project, every delivery trait of
    ``tests/_trait_fixtures`` and the per-plant aggregate meaning of the one delivering
    ``stem_count``."""
    from tests import _trait_fixtures as fx

    fx.seed_delivery_traits(tmp_path)
    fx.seed_confirmed_aggregate(tmp_path, "stem_count", value_keys=["count"])


@pytest.fixture
def real_hpo_base_config(tmp_path: Path) -> dict:
    """A base config the sweep door's own preflight admits: an importable builder and a data
    section over two labeled frames of its subject (``_verified_checkpoint_fixtures.
    detection_images``), drawn at seed 0."""
    from tests._verified_checkpoint_fixtures import detection_images

    scope = {"subject": DATA_DIR_SUBJECT}
    return {
        "model_source": {"builder": "tests.bespoke_models:build_bespoke_detection",
                         "builder_kwargs": {}, "task": "detection"},
        "data": {**detection_images(tmp_path / "hpo-data", scope), "scope": scope,
                 "split": {"seed": 0, "val_ratio": 0.15}},
    }


#: The single detection subject the canonical test dataset declares (``data_dir``).
DATA_DIR_SUBJECT = "bud"
#: The bucket the canonical test dataset publishes (``data_dir``).
DATA_DIR_BUCKET = "live/2-11-26"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A minimal dataset in the canonical layout: name-based per-image labels + a registry.

    One label document per image of capture ``2-11-26`` holding every subject (here one detection
    subject, ``bud``), a published bucket named ``live/2-11-26``, and one nested
    ``subjects.json``. Geometry is two boxes per image on a 640x480 frame.
    """
    from PIL import Image

    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tests._producer_fixtures import label_image, registry_over

    date = "2-11-26"
    subject = DATA_DIR_SUBJECT
    images_dir = tmp_path / "images" / date
    images_dir.mkdir(parents=True)

    # One nested registry traveling with the labels: a single detection subject, no attributes.
    registry_over(tmp_path, SubjectRegistry(subjects=(Subject(name=subject, description="a bud"),)))

    for name in ("img_001", "img_002", "img_003"):
        Image.new("RGB", (640, 480), color=(128, 128, 128)).save(images_dir / f"{name}.jpg")
        # GT: 2 boxes per image (pixel xyxy), by subject name.
        label_image(images_dir / f"{name}.jpg",
                    [Annotation(subject=subject, geometry=BBox(288, 216, 352, 264)),
                     Annotation(subject=subject, geometry=BBox(176, 132, 208, 156))], 640, 480)
    # Predictions, published as one bucket: 1 matching (TP) + 1 elsewhere (FP) per image.
    pytest.importorskip("torch")
    from tests._chain_fixtures import published

    published(tmp_path, DATA_DIR_BUCKET, [
        {"image": str(images_dir / f"{name}.jpg"), "width": 640, "height": 480,
         "boxes": [[288, 216, 352, 264], [496, 372, 528, 396]], "scores": [0.9, 0.7],
         "labels": [1, 1]} for name in ("img_001", "img_002", "img_003")],
        scope={"subject": subject})
    return tmp_path

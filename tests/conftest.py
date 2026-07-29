"""Shared test fixtures."""

from __future__ import annotations

import os
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
            "dependency is likely missing — module-level importorskip silently skipped files."
        )


@pytest.fixture(autouse=True)
def _restore_platform_root_env():
    """Keep the process-global platform-state root hermetic across tests.

    ``set_active_project`` repins ``TCIP_PROJECT_ROOT`` in-process (so a project's audit /
    experiments / registry co-locate under it). Since pytest runs in one process, a test
    that adopts a tmp project would otherwise leak that now-deleted root into later tests.
    Snapshot and restore the var around every test.
    """
    saved = os.environ.get("TCIP_PROJECT_ROOT")
    yield
    if saved is None:
        os.environ.pop("TCIP_PROJECT_ROOT", None)
    else:
        os.environ["TCIP_PROJECT_ROOT"] = saved


@pytest.fixture(autouse=True)
def _pin_platform_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Pin every test's platform-state root to its own unique ``tmp_path``.

    Without this, any unpinned write (audit, experiments, jobstore, the vision candidates
    cache) resolves relative to the process CWD and lands in the repo's real ``.tcip/`` —
    shared across tests and, under xdist, across worker processes. Uses monkeypatch so it
    auto-restores; a test that manages the var itself (setenv/delenv in its body) overrides
    this and is unaffected.
    """
    monkeypatch.setenv("TCIP_PROJECT_ROOT", str(tmp_path))


@pytest.fixture
def seed_catkin_trait_spec(tmp_path: Path, _pin_platform_root):
    """Seed a real catkin.yml into this test's pinned project root (round 10, 2026-07-29).

    There are no built-in traits anymore — ``get_trait("catkin")`` only resolves where a config
    file actually exists (``traits.py``). Writing the same values ``tests/_trait_fixtures.CATKIN``
    holds keeps a test that calls ``get_trait("catkin")``/``registered_traits()`` without authoring
    its own config working, the same as when a builtin was unconditionally present. NOT autouse: an
    unrelated test's project root should stay exactly as empty as it would without this cluster's
    change — request this explicitly in a test that actually needs catkin registered.
    """
    import dataclasses

    import yaml

    from tests._trait_fixtures import CATKIN

    specs_dir = tmp_path / ".tcip" / "state" / "trait_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    data = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(CATKIN).items()}
    (specs_dir / "catkin.yml").write_text(yaml.safe_dump(data), encoding="utf-8")


#: The single detection subject the canonical test dataset declares (``data_dir``).
DATA_DIR_SUBJECT = "catkin"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A minimal dataset in the canonical (K13.5) layout: name-based per-image labels + a registry.

    One file per image under ``annotations/<date>/`` holding every subject (here one detection
    subject, ``catkin``), predictions under ``predictions/<model>/<date>/``, and one nested
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
        ClassRegistry(subjects=(Subject(name=subject, description="a hazelnut catkin"),)),
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

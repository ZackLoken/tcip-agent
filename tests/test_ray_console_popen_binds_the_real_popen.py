"""Ray's ``ConsolePopen`` keeps the real ``Popen`` as its base through a patched window.

``ConsolePopen`` is the ``subprocess.Popen`` subclass Ray's process launcher defines when first
imported. It keeps the real ``Popen`` as its base through the patched window of each of four
test substitutes (``fake_popen``'s class, ``NoExecutableChild``, ``DyingChild`` and
``LivingChild``), and one built after the window closes starts a real child.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from tests._producer_fixtures import fake_popen
from tests.test_run_launcher_identity import NoExecutableChild
from tests.test_tensorboard_launch import DyingChild, LivingChild


@pytest.mark.skipif(
    importlib.util.find_spec("ray") is None, reason="Ray is not installed"
)
@pytest.mark.parametrize(
    "apply",
    [
        lambda monkeypatch: fake_popen(monkeypatch, []),
        lambda monkeypatch: monkeypatch.setattr(
            subprocess, "Popen", NoExecutableChild
        ),
        lambda monkeypatch: monkeypatch.setattr(
            subprocess, "Popen", DyingChild
        ),
        lambda monkeypatch: monkeypatch.setattr(
            subprocess, "Popen", LivingChild
        ),
    ],
    ids=["fake_popen", "no_executable", "dying", "living"],
)
def test_ray_console_popen_keeps_the_real_popen_through_a_patched_window(
    monkeypatch: pytest.MonkeyPatch,
    apply: Callable[[pytest.MonkeyPatch], type | None],
) -> None:
    """A ``ConsolePopen`` imported inside a patched window keeps the real ``Popen`` base.

    Importing Ray's process launcher inside the window leaves ``ConsolePopen`` based on the
    real ``Popen``, and a ``ConsolePopen`` built after the window runs a real child to exit.
    """
    real_popen = subprocess.Popen
    with monkeypatch.context() as window:
        apply(window)
        services = importlib.import_module("ray._private.services")

    assert services.ConsolePopen.__bases__[0] is real_popen
    child = services.ConsolePopen(
        [sys.executable, "-c", "pass"], creationflags=0
    )
    assert child.wait(timeout=30) == 0

"""The fixture a test requests to read the frontend's install fails in its setup without one."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_the_frontend_install_fixture_fails_in_its_setup_naming_the_folder_and_the_command(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suite install that is no folder fails the request, naming that folder and ``npm ci``."""
    missing = tmp_path / "no_install"
    monkeypatch.setattr("tests.FRONTEND_INSTALL", missing)

    with pytest.raises(pytest.fail.Exception) as refused:
        request.getfixturevalue("frontend_install")

    assert str(missing) in str(refused.value)
    assert "npm ci" in str(refused.value)

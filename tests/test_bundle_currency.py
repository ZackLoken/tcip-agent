"""The backend says when the bundle it serves is not the one its source builds.

At start, a checkout's backend compares the build record in ``STATIC_DIR`` with the identity the
frontend's own script gives its source now; a bundle with no record or another identity is logged
and named on every state snapshot's ``bundle``. A bundle that matches, a package with no
``frontend/`` beside it and an output with no bundle in it say nothing.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

import tcip_web.app as web_app

if TYPE_CHECKING:
    from pathlib import Path

REBUILD = "run build"
"""A fragment of the sentence naming the command that rebuilds the bundle."""


@pytest.fixture
def static(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a scratch output directory holding a page, served as the backend's ``STATIC_DIR``."""
    served = tmp_path / "static"
    served.mkdir()
    (served / "index.html").write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setattr(web_app, "STATIC_DIR", served)
    return served


def _started_frame(caplog: pytest.LogCaptureFixture) -> tuple[dict, list[str]]:
    """Start the backend and return its first state frame and the warnings it logged."""
    with caplog.at_level(logging.WARNING, logger=web_app.logger.name):
        with TestClient(web_app.app, base_url="http://127.0.0.1") as started:
            with started.websocket_connect("ws://127.0.0.1/ws/state") as ws:
                frame = ws.receive_json()
    return frame, [r.getMessage() for r in caplog.records]


def test_a_bundle_built_from_other_source_is_logged_and_named_on_the_first_frame(
    static: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = tmp_path / "other-frontend"
    (other / "src").mkdir(parents=True)
    (other / "tsconfig.json").write_text(
        json.dumps({"include": ["src"]}), encoding="utf-8"
    )
    (other / "src" / "main.ts").write_text("export {};\n", encoding="utf-8")
    record = web_app.bundle_record(other)
    (static / record["record"]).write_text(
        json.dumps(record), encoding="utf-8"
    )

    frame, warnings = _started_frame(caplog)

    assert record != web_app.bundle_record(web_app.FRONTEND_DIR)
    assert frame.get("bundle") is not None
    assert [w for w in warnings if REBUILD in w] == [frame["bundle"]]


def test_a_bundle_with_no_build_record_is_logged_and_named_on_the_first_frame(
    static: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frame, warnings = _started_frame(caplog)

    assert frame.get("bundle") is not None
    assert [w for w in warnings if REBUILD in w] == [frame["bundle"]]


def test_a_bundle_built_from_the_source_says_nothing(
    static: Path, caplog: pytest.LogCaptureFixture
) -> None:
    record = web_app.bundle_record(web_app.FRONTEND_DIR)
    (static / record["record"]).write_text(
        json.dumps(record), encoding="utf-8"
    )

    frame, warnings = _started_frame(caplog)

    assert frame["bundle"] is None
    assert [w for w in warnings if REBUILD in w] == []


def test_a_package_with_no_frontend_beside_it_says_nothing(
    static: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        web_app, "FRONTEND_DIR", tmp_path / "no-frontend", raising=False
    )

    frame, warnings = _started_frame(caplog)

    assert frame["bundle"] is None
    assert [w for w in warnings if REBUILD in w] == []


def test_an_output_with_no_bundle_says_nothing(
    static: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (static / "index.html").unlink()

    frame, warnings = _started_frame(caplog)

    assert frame["bundle"] is None
    assert [w for w in warnings if REBUILD in w] == []

"""The frontend build writes its bundle where the backend serves it from.

Vite's build output and its assets directory, as Vite's builder resolves them, are
the backend's ``tcip_web.app.STATIC_DIR`` and ``ASSETS_DIR``; the repository's ignore file and the
image build's ignore file each name that output, and the package's wheel carries it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from tcip_web.app import ASSETS_DIR, STATIC_DIR
from tests import FRONTEND, REPO_ROOT, VITE_CONFIG, printed_json


@pytest.fixture(scope="module")
def vite_config() -> dict:
    """Return what :data:`~tests.VITE_CONFIG` prints over the frontend, run once for the module."""
    return printed_json(VITE_CONFIG, FRONTEND)


def test_the_build_writes_its_bundle_where_the_backend_serves_it(
    vite_config: dict,
) -> None:
    """Vite's ``outDir`` is ``STATIC_DIR`` and its ``assetsDir`` under it is ``ASSETS_DIR``."""
    out_dir = Path(vite_config["outDir"])

    assert out_dir.resolve() == STATIC_DIR.resolve()
    assert (
        out_dir / vite_config["assetsDir"]
    ).resolve() == ASSETS_DIR.resolve()


@pytest.mark.parametrize(
    "ignore_file", [".gitignore", "packages/tcip-web/Dockerfile.dockerignore"]
)
def test_the_ignore_file_names_the_build_output(ignore_file: str) -> None:
    """The ignore file holds ``STATIC_DIR``, relative to the repository, as one directory entry."""
    entry = f"{STATIC_DIR.relative_to(REPO_ROOT).as_posix()}/"
    lines = (REPO_ROOT / ignore_file).read_text(encoding="utf-8").splitlines()

    assert entry in lines


def test_the_wheel_carries_the_build_output() -> None:
    """The package's wheel target lists ``STATIC_DIR``, relative to the package, as an artifact."""
    package = FRONTEND.parent
    wheel = tomllib.loads(
        (package / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert (
        f"{STATIC_DIR.relative_to(package).as_posix()}/" in wheel["artifacts"]
    )

"""The Node versions conda and the Docker image install lie within the frontend's ``engines.node``.

The frontend's ``package.json`` ``engines.node`` states which Node versions the platform runs on.
``environment.yml`` pins one Node release, ``nodejs==<version>``, and the Dockerfile's ``node``
base tag floats over a range of releases; npm's own ``semver`` compares each with it.
"""

from __future__ import annotations

import json
import re
import subprocess

import yaml

from tests import FRONTEND_SRC, REPO_ROOT

FRONTEND = FRONTEND_SRC.parent


def _admitted(version: str, tag: str) -> list[bool]:
    """Return whether the frontend's ``engines.node`` admits ``version`` and all of ``tag``.

    ``version`` is one release, read by ``semver.satisfies``, which answers false for anything that
    is not one; ``tag`` is the range of releases an image tag floats over, read by
    ``semver.subset``. Both run under ``node`` from the frontend's directory.
    """
    manifest = json.loads(
        (FRONTEND / "package.json").read_text(encoding="utf-8")
    )
    script = (
        "const semver = require('semver');"
        "const {engines, version, tag} = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        "process.stdout.write(JSON.stringify("
        "[semver.satisfies(version, engines), semver.subset(tag, engines)]));"
    )
    ran = subprocess.run(
        ["node", "-e", script],
        cwd=FRONTEND,
        capture_output=True,
        text=True,
        check=True,
        input=json.dumps(
            {
                "engines": manifest["engines"]["node"],
                "version": version,
                "tag": tag,
            }
        ),
    )
    return json.loads(ran.stdout)


def test_conda_and_the_docker_image_install_only_node_versions_the_frontend_admits() -> (
    None
):
    dependencies = yaml.safe_load(
        (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    )["dependencies"]
    (conda,) = [
        spec
        for spec in dependencies
        if isinstance(spec, str) and spec.startswith("nodejs")
    ]
    pinned = re.fullmatch(r"nodejs==(.+)", conda)
    assert pinned, f"{conda!r} is not one pinned release, nodejs==<version>"
    (tag,) = re.findall(
        r"^FROM node:(\S+)",
        (REPO_ROOT / "packages" / "tcip-web" / "Dockerfile").read_text(
            encoding="utf-8"
        ),
        re.MULTILINE,
    )

    assert _admitted(pinned[1], tag) == [True, True], (conda, tag)

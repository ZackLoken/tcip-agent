"""Write the image routes' answers as a fixture file, produced by the platform itself.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` and opened through the picker's route holds the rasters
:data:`RASTERS` names (``tests.test_tcip_web_image_serving._wide_raster``) and a file that is no
raster. ``GET /api/images/view`` answers, under :data:`DISPLAY_PIXELS`, the native cells a view
within that cap intersects (``native``), the overview level's cells a whole view of a raster
whose pyramid is built serves (``level``), and the cells of a view down a one-pixel-wide raster
(``skinny``). ``GET /api/images`` serves the native raster whole (``image``) and the
one-pixel-wide raster whole (``image_skinny``), refuses a raster past the cap whose pyramid is not
built (``image_overviews_required``) and a file that does not exist (``image_missing``), each
answer's status and image headers recorded. An overview build is started through
``POST /api/images/overviews`` and runs on the route's own worker, whose builder
(``overviews.build_overviews``) holds at its first report of bytes written until the scene steps
it on (``fixture_session.Gate``): a second request then joins it (``build``), the status route
answers it running (``running``) and, once it ends, completed (``completed``); a build over the
file that is no raster, whose builder raises :data:`BUILD_FAILURE`, fails (``failed``), and a job
no one started answers 404 (``unknown``). The answers, each refusal with its status, are written,
each minted job id read as what it names (``fixture_session.normalized``), to
:data:`GENERATED_PATH`.

    python -m tools.generate_images_fixture
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from tests import FRONTEND_SRC
from tools.fixture_session import (
    Gate,
    answered,
    ended,
    normalized,
    opened,
    responded,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic import JsonValue

GENERATED_PATH = FRONTEND_SRC / "test" / "imageAnswers.json"

DISPLAY_PIXELS = 360_000
"""The device-pixel count of the display each view is asked for, a sample small enough that a
view of the native raster crosses several cells."""

RASTERS = {
    "native": (1000, 600),
    "level": (4000, 2400),
    "skinny": (1, 100_000),
    "unbuilt": (4000, 2400),
    "pyramidless": (4000, 2400),
}
"""Each raster's width and height in pixels, by the scene it serves: one whose native cells a view
within the cap crosses, one whose pyramid is built, one a pixel wide, one whose pyramid the build
scene makes, and one past the cap whose pyramid is never built."""

VIEWS = {
    "native": (250, 0, 850, 600),
    "level": (0, 0, *RASTERS["level"]),
    "skinny": (0, 0, 1, 800),
}
"""The view each scene asks for, ``(x0, y0, x1, y1)`` in the raster's full-resolution pixels: a
window of the native raster within the cap, the whole of the built raster, and the top of the
one-pixel-wide raster."""

BUILD_FAILURE = "a sample failure the scene's builder raises for this file"
"""What the build over the file that is no raster stops on."""


def produce() -> dict[str, JsonValue]:
    """Serve views and build a pyramid in a scratch session; return the routes' answers.

    A route refusing what it should admit, or admitting what it should refuse, raises.
    """
    from tcip_mcp.pipelines import overviews
    from tcip_web.routes import images
    from tcip_web.routes.images import OverviewBuildPayload, OverviewJob
    from tests._web_fixtures import named_project
    from tests.test_tcip_web_image_serving import _wide_raster

    builder = overviews.build_overviews
    gate = Gate()
    held = threading.Event()

    def staged(path: str, progress_cb: Callable[[int], None]) -> object:
        if path != str(rasters["unbuilt"]):
            raise OSError(BUILD_FAILURE)

        def reported(written: int) -> None:
            progress_cb(written)
            if written and not held.is_set():
                held.set()
                gate.pause()

        return builder(path, progress_cb=reported)

    served: dict[str, JsonValue] = {}
    with (
        scratch_session() as (workspace, client),
        pytest.MonkeyPatch.context() as patch,
    ):
        root = workspace / "images"
        opened(client, named_project(root, "Images"))
        rasters: dict[str, Path] = {}
        for name, (width, height) in RASTERS.items():
            rasters[name] = root / f"{name}.tif"
            _wide_raster(rasters[name], width=width, height=height)
        builder(str(rasters["level"]))
        for name, (x0, y0, x1, y1) in VIEWS.items():
            served[name] = answered(
                client,
                "get_view_reads",
                200,
                params={
                    "path": str(rasters[name]),
                    "x0": str(x0),
                    "y0": str(y0),
                    "x1": str(x1),
                    "y1": str(y1),
                    "display_pixels": str(DISPLAY_PIXELS),
                },
            )

        def image_served(path: Path, status: int) -> JsonValue:
            response = responded(
                client,
                "serve_image",
                status,
                params={
                    "path": str(path),
                    "display_pixels": str(DISPLAY_PIXELS),
                },
            )
            return {
                "status": status,
                "headers": {
                    name: response.headers[name]
                    for name in (
                        images.SERVED_SIZE_HEADER,
                        images.IMAGE_ERROR_HEADER,
                    )
                    if name in response.headers
                },
            }

        served["image"] = image_served(rasters["native"], 200)
        served["image_skinny"] = image_served(rasters["skinny"], 200)
        served["image_overviews_required"] = image_served(
            rasters["pyramidless"], 400
        )
        served["image_missing"] = image_served(root / "absent.tif", 404)
        patch.setattr(overviews, "build_overviews", staged)

        def build(path: Path) -> JsonValue:
            return answered(
                client,
                "build_image_overviews",
                200,
                OverviewBuildPayload(path=str(path)),
            )

        def status_once_ended(job_id: str) -> JsonValue:
            ended(images._overview_registry, job_id)
            return answered(
                client, "get_overview_job", 200, params={"job_id": job_id}
            )

        started = OverviewJob.model_validate(build(rasters["unbuilt"])).job_id
        gate.reached()
        served["build"] = build(rasters["unbuilt"])
        served["running"] = answered(
            client, "get_overview_job", 200, params={"job_id": started}
        )
        gate.step()
        served["completed"] = status_once_ended(started)
        not_a_raster = root / "broken.tif"
        not_a_raster.write_bytes(b"this is not a raster")
        broken = OverviewJob.model_validate(build(not_a_raster)).job_id
        served["failed"] = status_once_ended(broken)
        served["unknown"] = answered(
            client, "get_overview_job", 404, params={"job_id": "ovr-unknown"}
        )
        return normalized(
            served,
            workspace,
            minted={started: "<built job>", broken: "<failed job>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))

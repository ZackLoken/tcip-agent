"""Write a project's open answer, the state envelope after it and its selections as a fixture file.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` is opened through the picker's route; the route's answer
(``opened``) and the envelope ``/ws/state`` then sends on connect
(``tcip_web.app.state_snapshot_message``, ``envelope``) are written, with the select route's
answers for a capture of photographs (``raw``) and a capture of one band group (``grouped``)
of datasets inside the project and a selection of the photographs under a separator-ended root
(``separated``), normalized (``fixture_session.normalized``), to :data:`GENERATED_PATH`.

    python -m tools.generate_open_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
    normalized,
    opened,
    scratch_session,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tools.fixture_session import Stamp

GENERATED_PATH = FRONTEND_SRC / "test" / "openFixtures.json"

STAMPED: dict[str, Stamp] = {"epoch": (str, "<epoch>"), "version": (int, 0)}
"""The process's epoch, read as ``<epoch>``, and the version, which counts every state the
process has held, read as ``0``."""

RAW_IMAGES = [f"img{i}.jpg" for i in range(1, 6)]
"""The photographs of the raw capture, five so a test can step through several."""


def _captures(raw_root: Path, grouped_root: Path) -> list[tuple[Path, str]]:
    """Write a capture of :data:`RAW_IMAGES` and a capture of one two-band group.

    Returns each capture's dataset root and date, the photographs' first.
    """
    import numpy as np
    import tifffile

    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.pipelines.data.band_groups import write_band_group_manifest
    from tests._producer_fixtures import write_image

    raw, grouped = "2026-01-01", "2026-01-02"
    for name in RAW_IMAGES:
        write_image(image_dir(raw_root, raw) / name, (8, 8))
    group_dir = image_dir(grouped_root, grouped)
    group_dir.mkdir(parents=True)
    bands = {band: group_dir / f"cap_{band}.tif" for band in ("Green", "Red")}
    for band_path in bands.values():
        tifffile.imwrite(band_path, np.zeros((8, 8), np.uint16))
    write_band_group_manifest(
        group_dir,
        "cap",
        bands,
        central_wavelength_nm=None,
        identity_tolerance={},
    )
    return [(raw_root, raw), (grouped_root, grouped)]


def produce() -> dict[str, JsonValue]:
    """Open the fixture project; return its open answer, the envelope after it and two selections.

    The selections are the select route's answers for the two captures :func:`_captures` writes,
    in the project's datasets ``ds`` and ``grouped``, made after the envelope, and
    :func:`~tcip_mcp.web_client.selection_for`'s selection of the photographs' capture under its
    root spelled with a trailing separator (``separated``), which keeps that spelling while each
    image's place is the one its scan found, so no join of the root, the capture and an image's
    name rebuilds it. All are normalized by :data:`STAMPED`, the project's minted id read as
    ``<valley id>`` and the workspace as ``<workspace>``.
    """
    import os

    from tcip_mcp.web_client import DatasetChoice, selection_for
    from tcip_web.app import app
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        valley = named_project(workspace / "valley", "valley")
        answer = opened(client, valley)
        socket = client.base_url.join(app.url_path_for("state_ws"))
        with client.websocket_connect(
            str(socket.copy_with(scheme="ws"))
        ) as ws:
            envelope = ws.receive_json()
        captures = _captures(valley.root / "ds", valley.root / "grouped")
        selected = {
            kind: client.post(
                "/api/dataset/select",
                json={"dataset_root": str(root), "date": date},
            )
            for kind, (root, date) in zip(
                ("raw", "grouped"), captures, strict=True
            )
        }
        for response in selected.values():
            response.raise_for_status()
        raw_root, raw_date = captures[0]
        separated = selection_for(
            DatasetChoice(
                dataset_root=f"{raw_root}{os.sep}",
                date=raw_date,
                current_image_index=0,
            )
        )
        return normalized(
            {
                "opened": answer,
                "envelope": envelope,
                **{kind: r.json() for kind, r in selected.items()},
                "separated": separated.model_dump(mode="json"),
            },
            stamped=STAMPED,
            minted={valley.id: "<valley id>"},
            places={workspace: "<workspace>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))

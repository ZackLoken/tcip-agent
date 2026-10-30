"""Write a project's open answer, the state envelope after it and its selections as a fixture file.

In a scratch session (``tools/fixture_session.py``), a project created through
``tests._web_fixtures.named_project`` is opened through the picker's route; the route's answer
(``opened``) and the envelope ``/ws/state`` then sends on connect
(``tcip_web.app.state_snapshot_message``, ``envelope``) are written, with the select route's
answers for a capture of photographs (``raw``), a capture of one band group (``grouped``) and a
capture whose one label document will not read (``troubled``) of datasets inside the project,
the nav route's answer to a step to the first image of the last (``navigated``), the dataset
trees of the photographs' dataset, the troubled one and one holding no capture, the select
route's answers for the photographs made for :data:`SUBJECT` (``raw_subject``), for
:data:`FOCUSED_SUBJECT` (``focused``) and for it with the first bucket the photographs' tree
names over that capture (``focused_bucket``), and a selection of the photographs under a
separator-ended root (``separated``), normalized (``fixture_session.normalized``), to
:data:`GENERATED_PATH`.

    python -m tools.generate_open_fixture
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tests import FRONTEND_SRC
from tools.fixture_session import (
    answered,
    normalized,
    opened,
    scratch_session,
    socket_url,
    write_fixture,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from tools.fixture_session import Stamp

GENERATED_PATH = FRONTEND_SRC / "test" / "openFixtures.json"

STAMPED: dict[str, Stamp] = {"version": (int, 0)}
"""The version, which counts every state the process has held, read as ``0``; the process's
``epoch`` is numbered."""

RAW_IMAGES = [f"img{i}.jpg" for i in range(1, 6)]
"""The photographs of the raw capture, five so a test can step through several."""

FRAMES = [(1000, 800), (1010, 800), (1020, 800), (1030, 800), (600, 900)]
"""Each raw photograph's width and height, sample frames the annotate scenes draw within: each
its own, so two photographs' documents holding nothing are two documents, and the last taller
than wide."""


BUCKETS = ("baseline", "v2")
"""The buckets published over the photographs' capture, two so a date offers a choice."""

SUBJECT = "subject_a"
"""The subject a selection of the photographs is made for, a sample name."""

FOCUSED_SUBJECT = "bud"
"""The subject an agent's focus names and the label scenes annotate, not :data:`SUBJECT`."""


def captures(project: Path) -> dict[str, tuple[Path, str]]:
    """Write the project's datasets; return each named capture's dataset root and date.

    ``ds`` holds a capture of :data:`RAW_IMAGES` (``raw``), each of its own of :data:`FRAMES`, with
    each of :data:`BUCKETS` published over it
    (``tests._chain_fixtures.published``), and a later capture of one photograph over which
    nothing is published; ``grouped`` a capture of one two-band group (``grouped``);
    ``troubled`` a capture of one image whose label document is stored as a list, which no label
    document reads as (``tests._record_damage_fixtures.damage_record``, ``troubled``); and
    ``empty`` is a dataset root holding no capture.
    """
    import numpy as np
    import tifffile

    from tcip_mcp.dataset_layout import image_dir, label_key_of
    from tcip_mcp.pipelines.postprocessing.export import DetectionResult
    from tests._chain_fixtures import published
    from tests._producer_fixtures import (
        box_annotation,
        label_image,
        write_band_group,
        write_image,
    )
    from tests._record_damage_fixtures import damage_record

    raw_root, grouped_root, troubled_root = (
        project / name for name in ("ds", "grouped", "troubled")
    )
    raw, later, grouped, troubled = (
        "2026-01-01",
        "2026-01-08",
        "2026-01-02",
        "2026-01-03",
    )
    photographs = [
        (write_image(image_dir(raw_root, raw) / name, frame), frame)
        for name, frame in zip(RAW_IMAGES, FRAMES, strict=True)
    ]
    for bucket in BUCKETS:
        published(
            project,
            f"{bucket}/{raw}",
            [
                DetectionResult(
                    image=str(image),
                    width=width,
                    height=height,
                    boxes=[(1.0, 1.0, 4.0, 4.0)],
                    scores=[0.9],
                    cap=2,
                ).model_dump()
                for image, (width, height) in photographs
            ],
            scope={"subject": SUBJECT},
        )
    write_image(image_dir(raw_root, later) / "img1.jpg", (8, 8))
    group_dir = image_dir(grouped_root, grouped)
    group_dir.mkdir(parents=True)
    bands = {band: group_dir / f"cap_{band}.tif" for band in ("Green", "Red")}
    for band_path in bands.values():
        tifffile.imwrite(band_path, np.zeros((8, 8), np.uint16))
    write_band_group(group_dir, "cap", bands, central_wavelength_nm=None)
    damaged = write_image(image_dir(troubled_root, troubled) / "a.jpg", (8, 8))
    label_image(damaged, [box_annotation(1, 1, 4, 4)], 8, 8)
    damage_record(label_key_of(damaged), b"[]")
    (project / "empty").mkdir()
    return {
        "raw": (raw_root, raw),
        "grouped": (grouped_root, grouped),
        "troubled": (troubled_root, troubled),
    }


def produce() -> dict[str, JsonValue]:
    """Open the fixture project; return its open answer, the envelope, selections and trees.

    The selections are the select route's answers for the three captures :func:`captures`
    names, made after the envelope; the trees ``GET /api/dataset/tree`` answers for the datasets
    ``ds``, ``troubled`` and ``empty`` (``ds_tree``, ``troubled_tree``, ``empty_tree``); and
    :func:`~tcip_mcp.web_client.selection_for`'s selection of the photographs' capture under its
    root spelled with a trailing separator (``separated``), which keeps that spelling while each
    image's place is the one its scan found. All are normalized by :data:`STAMPED`, the epoch
    numbered and the project's minted id read as ``<valley id>``.
    """
    import os

    from tcip_mcp.web_client import CaptureChoice, DatasetChoice, selection_for
    from tcip_web.routes.dataset import DatasetTree, NavRequest
    from tests._web_fixtures import named_project

    with scratch_session() as (workspace, client):
        valley = named_project(workspace / "valley", "valley")
        answer = opened(client, valley)
        with client.websocket_connect(socket_url(client, "state_ws")) as ws:
            envelope = ws.receive_json()
        written = captures(valley.root)
        selected = {
            kind: answered(
                client,
                "select_dataset",
                200,
                CaptureChoice(dataset_root=str(root), date=date),
            )
            for kind, (root, date) in written.items()
        }
        navigated = answered(
            client, "set_current_image", 200, NavRequest(current_image_index=0)
        )
        trees = {
            f"{name}_tree": answered(
                client,
                "get_dataset_tree",
                200,
                params={"dataset_root": str(valley.root / name)},
            )
            for name in ("ds", "troubled", "empty")
        }
        raw_root, raw_date = written["raw"]
        first_bucket, *_ = DatasetTree.model_validate(
            trees["ds_tree"]
        ).buckets_by_date[raw_date]
        made_for = {
            name: answered(
                client,
                "select_dataset",
                200,
                CaptureChoice(
                    dataset_root=str(raw_root),
                    subject=subject,
                    date=raw_date,
                    bucket=bucket,
                ),
            )
            for name, subject, bucket in (
                ("raw_subject", SUBJECT, None),
                ("focused", FOCUSED_SUBJECT, None),
                ("focused_bucket", FOCUSED_SUBJECT, first_bucket),
            )
        }
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
                **selected,
                "navigated": navigated,
                **trees,
                **made_for,
                "separated": separated.model_dump(mode="json"),
            },
            workspace,
            stamped=STAMPED,
            numbered=frozenset({"epoch"}),
            minted={valley.id: "<valley id>"},
        )


if __name__ == "__main__":
    raise SystemExit(write_fixture(GENERATED_PATH, produce()))

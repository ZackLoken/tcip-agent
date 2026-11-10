"""run_inference publishes a bucket and refuses one that exists.

It publishes one prediction document per image and the bucket's record under the images' dataset
root.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_mcp.workspace import BoundProject

torch = pytest.importorskip("torch")
import tcip_store  # noqa: E402
from tcip_mcp.dataset_layout import (  # noqa: E402
    bucket_key,
    image_dir,
    image_root,
)
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests._audit_fixtures import audit_rows  # noqa: E402
from tests._mapping_fixtures import CROP  # noqa: E402
from tests._predictor_fixtures import StubPredictor, install  # noqa: E402
from tests._producer_fixtures import (  # noqa: E402
    dataset_registered,
    gray_frame,
    ingested_unregistered,
    left_written,
    register_dataset_at,
    unregistered_refusal,
)
from tests._server_calls import called_through_the_server  # noqa: E402
from tests._verified_checkpoint_fixtures import (  # noqa: E402
    SAMPLE_CONF,
    SAMPLE_DETECTOR_PASS,
    project_checkpoint,
)


@pytest.fixture(autouse=True)
def _registered(tmp_path: Path) -> None:
    """Register ``tmp_path``, the dataset root most images here are written under, in itself."""
    dataset_registered(tmp_path, tmp_path, crop=CROP)


UNTILED = Stated(
    tile=False, conf=SAMPLE_CONF, rationale="a sample reason the test states"
)
"""The stated execution values of an untiled run at the sample confidence."""

BUCKET = "out/2026-01-01"


def _stubbed(monkeypatch: pytest.MonkeyPatch) -> StubPredictor:
    return install(monkeypatch, StubPredictor(task="detection", in_chans=3))


def _images(directory: Path) -> Path:
    gray_frame(directory, 100)
    return directory


def test_run_inference_publishes_each_document_the_record_and_one_audit_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    from tcip_annotation.json_io import read_predictions
    from tcip_annotation.state import BBox
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.tools.inference_tools import run_inference

    ckpt = Path(project_checkpoint(tmp_path))
    images_dir = _images(image_dir(tmp_path, "2026-01-01"))
    _stubbed(monkeypatch)

    ran = run_inference(
        tmp_path,
        str(ckpt),
        str(images_dir),
        bucket=BUCKET,
        stated=UNTILED,
        actor=None,
    )

    assert "error" not in ran, ran
    bucket = read_bucket(tmp_path, BUCKET)
    key = bucket.document_key("img")
    assert key is not None
    data = tcip_store.read(key)
    assert isinstance(data, dict)
    assert key.parts[-1] == "img" and "image" not in data
    document = read_predictions(key)
    assert (document.width, document.height) == (100, 100)
    (ann,) = document.annotations
    assert ann.subject == "bud"  # label 1 -> id 0 -> the checkpoint's own map
    assert ann.score == pytest.approx(0.9)
    assert ann.geometry == BBox(10.0, 10.0, 30.0, 30.0)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert ann.created_by == f"model:{ckpt.stem}@{digest[:12]}"
    assert bucket.documents == {"img": "img.png"}
    assert bucket.producer.kind == "checkpoint"
    assert bucket.producer.checkpoint_sha256 == digest
    assert bucket.assessment_id is None
    rows = audit_rows(tmp_path, "prediction_bucket_published")
    assert [r["arguments"] for r in rows] == [
        {"dataset_root": str(tmp_path), "bucket": BUCKET, "dropped_boxes": 0}
    ]


def test_a_second_run_into_a_published_bucket_refuses_and_leaves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run into a published bucket refuses and leaves the bucket as it was.

    A second run naming the same bucket refuses, leaving the first bucket's documents and record
    as they were, and a new bucket admits the same run.
    """
    from tcip_mcp.dataset_layout import PREDICTION_DOCUMENTS
    from tcip_mcp.tools.inference_tools import run_inference

    root = dataset_registered(tmp_path, tmp_path / "dataset", crop=CROP)
    images_dir = _images(image_dir(root, "2026-01-01"))
    _stubbed(monkeypatch)
    ckpt = project_checkpoint(tmp_path)
    first = run_inference(
        tmp_path,
        ckpt,
        str(images_dir),
        bucket=BUCKET,
        stated=UNTILED,
        actor=None,
    )
    assert "error" not in first, first

    def published() -> dict:
        keys = [
            bucket_key(root, BUCKET),
            *tcip_store.keys(PREDICTION_DOCUMENTS, str(root), (BUCKET,)),
        ]
        return {key: tcip_store.read_versioned(key).version for key in keys}

    before = published()

    again = run_inference(
        tmp_path,
        ckpt,
        str(images_dir),
        bucket=BUCKET,
        stated=UNTILED,
        actor=None,
    )
    preview = run_inference(
        tmp_path,
        ckpt,
        str(images_dir),
        bucket=BUCKET,
        dry_run=True,
        stated=UNTILED,
        actor=None,
    )
    fresh = run_inference(
        tmp_path,
        ckpt,
        str(images_dir),
        bucket="baseline-2/2026-01-01",
        stated=UNTILED,
        actor=None,
    )

    assert "already" in again["error"]
    assert published() == before
    assert preview["bucket_exists"] is True
    assert "error" not in fresh, fresh


def test_a_run_over_an_empty_images_directory_refuses_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tcip_mcp.tools.inference_tools import run_inference

    empty = image_dir(tmp_path, "2026-01-01")
    empty.mkdir(parents=True)
    _stubbed(monkeypatch)

    ran = run_inference(
        tmp_path,
        project_checkpoint(tmp_path),
        str(empty),
        bucket=BUCKET,
        stated=UNTILED,
        actor=None,
    )

    assert "no document" in ran["error"]
    assert not tcip_store.exists(bucket_key(tmp_path, BUCKET))


@pytest.mark.parametrize(
    "where",
    [
        lambda root: root / "captures",
        image_root,
        lambda root: image_dir(root, "2026-01-01") / "sub",
    ],
    ids=["no_image_tree", "the_image_tree", "inside_a_capture"],
)
def test_an_images_dir_that_is_no_capture_refuses_naming_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: Callable[[Path], Path],
) -> None:
    """An images directory that is no capture is refused at the door naming it.

    A directory that is no capture (``<root>/images/<capture>``) refuses at the door naming it,
    and no bucket is published.
    """
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(where(tmp_path))
    _stubbed(monkeypatch)

    ran = run_inference(
        tmp_path,
        project_checkpoint(tmp_path),
        str(images_dir),
        bucket=BUCKET,
        stated=UNTILED,
        actor=None,
    )

    assert f"{str(images_dir)!r} is not a capture" in ran["error"]
    assert not tcip_store.exists(bucket_key(tmp_path, BUCKET))


TILED = Stated(tile_size=32, overlap=0.0, **SAMPLE_DETECTOR_PASS)
"""The stated execution values of a tiled pass over 32 px tiles, as a raster pass runs."""


def _mosaic(path: Path) -> Path:
    """Write a black 64 px raster at ``path``, its directory made, and return it.

    No platform producer places a raster outside a capture, so a test placing one writes it.
    """
    import numpy as np
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))
    return path


def test_a_raster_outside_a_capture_publishes_a_bucket_of_its_roots_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raster in the image tree and in no capture publishes a bucket recording its root's dataset.

    The bucket records the id ``tmp_path`` was registered with and no capture date.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import require_dataset_identity
    from tcip_mcp.tools.inference_tools import run_inference

    raster = _mosaic(image_root(tmp_path) / "mosaic.tif")
    _stubbed(monkeypatch)

    result = run_inference(
        tmp_path,
        project_checkpoint(tmp_path),
        raster_path=str(raster),
        bucket=BUCKET,
        stated=TILED,
        actor=None,
    )

    assert "error" not in result, result
    bucket = read_bucket(tmp_path, BUCKET)
    assert (result["dataset_root"], bucket.dataset_id, bucket.date) == (
        str(tmp_path),
        require_dataset_identity(tmp_path).id,
        None,
    )


@pytest.mark.parametrize(
    "raster", [False, True], ids=["image_pass", "raster_pass"]
)
def test_a_pass_publishes_under_the_root_its_dry_run_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raster: bool
) -> None:
    """A pass publishes its bucket under the root its dry run states for the same sources.

    The images are a capture named ``images`` of the root ``data``, and the raster lies in the
    image tree of a root itself named ``images``, each position read two ways and settled by the
    root carrying its own state. The dry run's root is the door's reading at entry; the answer's
    and the publication's audit line's are the publication's own.
    """
    from tcip_mcp.tools.inference_tools import run_inference

    if raster:
        root = dataset_registered(
            tmp_path, tmp_path / "scratch" / "images", crop=CROP
        )
        sources = {
            "raster_path": str(_mosaic(image_root(root) / "mosaic.tif"))
        }
    else:
        root = dataset_registered(tmp_path, tmp_path / "data", crop=CROP)
        sources = {"images_dir": str(_images(image_dir(root, "images")))}
    _stubbed(monkeypatch)

    def run(**options: bool) -> dict:
        return run_inference(
            tmp_path,
            project_checkpoint(tmp_path),
            bucket=BUCKET,
            stated=TILED if raster else UNTILED,
            actor=None,
            **sources,
            **options,
        )

    stated = run(dry_run=True)
    ran = run()

    assert "error" not in ran, ran
    rows = audit_rows(
        Path(stated["dataset_root"]), "prediction_bucket_published"
    )
    assert [r["arguments"]["dataset_root"] for r in rows] == [
        stated["dataset_root"]
    ]
    assert ran["dataset_root"] == stated["dataset_root"] == str(root)


def _unregistered_source(tmp_path: Path, *, raster: bool) -> Path:
    """Ingest one source into the dataset root ``tmp_path / "field"``, never registering it.

    The source is a black 64 px raster ``mosaic.tif`` when ``raster``, else a 100 px gray
    ``img.png`` (:func:`~tests._producer_fixtures.gray_frame`); returns its ingested path.
    """
    import numpy as np
    import tifffile

    raw = tmp_path / "raw"
    if raster:
        raw.mkdir()
        tifffile.imwrite(
            str(raw / "mosaic.tif"), np.zeros((64, 64, 3), dtype=np.uint8)
        )
    else:
        gray_frame(raw, 100)
    (name,) = (p.name for p in raw.iterdir())
    return ingested_unregistered(tmp_path / "field", raw) / name


@pytest.mark.parametrize(
    ("raster", "options", "once_registered", "writes"),
    [
        (False, {}, lambda answer: "error" not in answer, True),
        (True, {}, lambda answer: "error" not in answer, True),
        (
            False,
            {"dry_run": True},
            lambda answer: answer.get("dry_run"),
            False,
        ),
        (
            True,
            {"resume": True},
            lambda answer: "no raster-pass progress" in answer["error"],
            False,
        ),
    ],
    ids=["image_pass", "raster_pass", "dry_run", "resume"],
)
def test_a_pass_under_a_root_never_registered_refuses_naming_it_and_writes_nothing(
    tmp_path: Path,
    bound: BoundProject,
    monkeypatch: pytest.MonkeyPatch,
    raster: bool,
    options: dict[str, bool],
    once_registered: Callable[[dict], bool],
    writes: bool,
) -> None:
    """A pass under a root ingested and never registered refuses before any model runs.

    Called through the server, the answer is the one sentence ``require_dataset_identity``
    refuses the root with, naming it and ``register_dataset``; the predictor is never called and
    nothing is written. Registered, the root answers the same call as any registered root does,
    a pass writing what a dry run and a refused resume do not.
    """
    source = _unregistered_source(tmp_path, raster=raster)
    root = tmp_path / "field"
    stub = _stubbed(monkeypatch)
    arguments = {
        "checkpoint_path": project_checkpoint(tmp_path),
        **(
            {"raster_path": str(source)}
            if raster
            else {"images_dir": str(source.parent)}
        ),
        "bucket": BUCKET,
        "stated": (TILED if raster else UNTILED).model_dump(
            mode="json", exclude_unset=True
        ),
        **options,
    }
    before = left_written(tmp_path, root)

    refused = called_through_the_server(
        "run_inference", arguments, bound=bound
    )

    assert refused == {"error": unregistered_refusal(root)}, refused
    assert stub.calls == 0
    assert left_written(tmp_path, root) == before
    register_dataset_at(tmp_path, root, crop=CROP)
    registered = left_written(tmp_path, root)
    answer = called_through_the_server("run_inference", arguments, bound=bound)
    assert isinstance(answer, dict) and once_registered(answer), answer
    assert (left_written(tmp_path, root) != registered) is writes


def test_a_pass_under_a_root_never_registered_refuses_before_its_checkpoint_is_read(
    tmp_path: Path, bound: BoundProject
) -> None:
    """A pass naming no checkpoint file under a root never registered answers the root's refusal.

    The checkpoint is never read, so no model is built and no pass runs.
    """
    source = _unregistered_source(tmp_path, raster=False)

    refused = called_through_the_server(
        "run_inference",
        {
            "checkpoint_path": str(tmp_path / "absent.pt"),
            "images_dir": str(source.parent),
            "bucket": BUCKET,
            "stated": UNTILED.model_dump(mode="json", exclude_unset=True),
        },
        bound=bound,
    )

    assert refused == {"error": unregistered_refusal(tmp_path / "field")}


def test_a_dry_run_previews_the_both_sources_refusal(tmp_path: Path) -> None:
    """A dry run naming both an images directory and a raster is refused, naming the rule."""
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = image_root(tmp_path)
    images_dir.mkdir()
    raster_path = tmp_path / "mosaic.tif"
    raster_path.write_bytes(b"stub")

    result = run_inference(
        tmp_path,
        project_checkpoint(tmp_path),
        stated=Stated(),
        images_dir=str(images_dir),
        raster_path=str(raster_path),
        bucket=BUCKET,
        dry_run=True,
        actor=None,
    )

    assert "exactly one of images_dir or raster_path" in result["error"]


def test_a_dry_run_names_the_bucket_and_the_execution_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tcip_mcp.tools.inference_tools import run_inference

    images_dir = _images(image_dir(tmp_path, "2026-01-01"))
    _stubbed(monkeypatch)

    result = run_inference(
        tmp_path,
        project_checkpoint(tmp_path),
        str(images_dir),
        bucket=BUCKET,
        stated=UNTILED,
        dry_run=True,
        actor=None,
    )

    assert "error" not in result, result
    assert (result["dataset_root"], result["bucket"]) == (
        str(tmp_path),
        BUCKET,
    )
    assert result["bucket_exists"] is False
    assert result["execution"]["tile_size"] is None
    assert not tcip_store.exists(bucket_key(tmp_path, BUCKET))

"""The logical-image refusals and the doors that read them.

The resolver (``image_utils.resolve_image_paths``) and the enumerator
(``image_utils.list_logical_images``) refuse through ``LogicalImageError`` alone, none of its
subclasses a builtin error, so a door answers one only by naming the class. Each door is called
over a capture written through the platform's producers: a two-band grouped capture whose
manifest ``write_band_group_manifest`` writes, a plain photograph and a file no image suffix
names, the images labeled through the label save. A band removed after its manifest is written
makes an incomplete group, and a second photograph of one case-folded stem written into the
capture makes an ambiguous stem, as ``ingest_images`` refuses to bring one in.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from tests._mapping_fixtures import CROP
from tests._producer_fixtures import (
    GroupedDataset,
    label_image,
    register_dataset_at,
    registry_of,
    registry_over,
    write_band_group,
    write_image,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

CAPTURE = "2026-04-01"
SIZE = 16
STRAY = "notes.txt"
"""The capture's file whose suffix names no image."""


def _written(root: Path, project: Path) -> GroupedDataset:
    """Write a grouped capture, a photograph and :data:`STRAY` into ``root``'s :data:`CAPTURE`.

    The group ``capture_001`` holds two single-band TIFFs, the photograph is ``plain_002.jpg``,
    each image holds one ``bud`` box in its label document, the dataset's registry declares
    ``bud`` and ``root`` is registered in ``project`` at :data:`~tests._mapping_fixtures.CROP`.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.dataset_layout import image_dir

    images_dir = image_dir(root, CAPTURE)
    images_dir.mkdir(parents=True)
    registry_over(root, registry_of("bud"))
    bands = {
        "Green": images_dir / "capture_001_G.tif",
        "Red": images_dir / "capture_001_R.tif",
    }
    for band, fill in zip(bands.values(), (111, 222), strict=True):
        tifffile.imwrite(str(band), np.full((SIZE, SIZE), fill, np.uint16))
    manifest = write_band_group(
        images_dir, "capture_001", bands, central_wavelength_nm=None
    )
    plain = write_image(images_dir / "plain_002.jpg", (SIZE, SIZE), (5, 5, 5))
    for source in (manifest, plain):
        label_image(
            source,
            [Annotation(subject="bud", geometry=BBox(2, 2, 6, 6))],
            SIZE,
            SIZE,
        )
    written = GroupedDataset(root, images_dir, manifest, bands, plain)
    (written.images_dir / STRAY).write_bytes(b"field notes")
    register_dataset_at(project, root, crop=CROP)
    return written


@pytest.fixture
def written(tmp_path: Path) -> GroupedDataset:
    """Return :func:`_written` of ``tmp_path / "ds"``, registered in ``tmp_path``."""
    return _written(tmp_path / "ds", tmp_path)


def _sentence(read: Callable[[Path], object], path: Path) -> str:
    """Return the sentence ``read`` refuses ``path`` with."""
    with pytest.raises(Exception) as refused:
        read(path)
    return str(refused.value)


def _resolved(path: Path) -> str:
    """Return the sentence the resolver refuses ``path`` with."""
    from tcip_mcp.pipelines.image_utils import resolve_image_path

    return _sentence(resolve_image_path, path)


def _listed(images_dir: Path) -> str:
    """Return the sentence the enumerator refuses ``images_dir`` with."""
    from tcip_mcp.pipelines.image_utils import list_logical_images

    return _sentence(list_logical_images, images_dir)


def _ambiguous(written: GroupedDataset) -> Path:
    """Write a second photograph of ``plain_002``'s case-folded stem beside it; return it."""
    return write_image(
        written.images_dir / "Plain_002.png", (SIZE, SIZE), (9, 9, 9)
    )


def test_every_refusal_of_the_resolver_and_the_enumerator_is_a_logical_image_error(
    written: GroupedDataset,
) -> None:
    """Each refusal is its own subclass of ``LogicalImageError``, and none is a builtin error.

    A band member and a file no image suffix names name no logical image, an ambiguous stem is
    refused by the resolver and the enumerator alike, and a group missing a band is incomplete.
    """
    from tcip_mcp.pipelines.image_utils import (
        AmbiguousImageStemError,
        BandGroupIncompleteError,
        ImageNotNamedError,
        LogicalImageError,
        list_logical_images,
        resolve_image_path,
    )

    def refusal(read: Callable[[Path], object], path: Path) -> Exception:
        with pytest.raises(LogicalImageError) as refused:
            read(path)
        return refused.value

    refusals: list[tuple[Exception, type[LogicalImageError]]] = [
        (
            refusal(resolve_image_path, written.bands["Red"]),
            ImageNotNamedError,
        ),
        (
            refusal(resolve_image_path, written.images_dir / STRAY),
            ImageNotNamedError,
        ),
    ]
    _ambiguous(written)
    refusals += [
        (refusal(resolve_image_path, written.plain), AmbiguousImageStemError),
        (
            refusal(list_logical_images, written.images_dir),
            AmbiguousImageStemError,
        ),
    ]
    written.bands["Red"].unlink()
    refusals.append(
        (
            refusal(resolve_image_path, written.manifest),
            BandGroupIncompleteError,
        )
    )

    assert [type(refused) for refused, _ in refusals] == [
        kind for _, kind in refusals
    ]
    assert not any(
        isinstance(refused, (FileNotFoundError, ValueError))
        for refused, _ in refusals
    ), refusals


def test_stating_a_capture_date_answers_a_path_naming_no_logical_image(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``state_capture_date`` answers each refused path with the resolver's sentence.

    A band member, a file no image suffix names and a path naming no file each answer the
    sentence, nothing moved; the photograph itself moves.
    """
    from tcip_mcp.tools.ingest_tools import state_capture_date

    def stated(image: Path) -> dict:
        return state_capture_date(
            tmp_path,
            images=[str(image)],
            date=datetime.date(2026, 4, 2),
            rationale="the field book dates this walk",
            actor=None,
        )

    for refused in (
        written.bands["Green"],
        written.images_dir / STRAY,
        written.images_dir / "absent.jpg",
    ):
        assert stated(refused) == {"error": _resolved(refused)}
    assert written.plain.is_file()
    assert [move["source"] for move in stated(written.plain)["moved"]] == [
        str(written.plain)
    ]


def test_rendering_failure_cases_answers_an_image_that_no_longer_resolves(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``render_failure_cases`` answers a bucket image the resolver refuses with its sentence.

    A bucket published for the photograph renders; once a second photograph of its stem sits
    beside it, the render answers the resolver's sentence for it.
    """
    from tcip_annotation.state import Annotation, BBox
    from tcip_mcp.tools.vision_tools import render_failure_cases
    from tests._producer_fixtures import published_document

    published_document(
        tmp_path,
        "preds",
        written.plain,
        [Annotation(subject="bud", geometry=BBox(2, 2, 6, 6), score=0.9)],
        SIZE,
        SIZE,
    )

    def rendered() -> dict:
        return render_failure_cases(
            tmp_path, str(written.root), "preds", 1, 1.0, 1.0, 1.0
        )

    assert "error" not in rendered(), rendered()
    _ambiguous(written)
    assert rendered() == {"error": _resolved(written.plain)}


def test_evaluating_over_an_incomplete_band_group_answers_its_refusal(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``evaluate_model`` answers a labeled group missing a band as its error, naming it."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._training_values import IOU_THRESHOLD
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    checkpoint = registered_checkpoint(tmp_path)
    written.bands["Red"].unlink()

    evaluated = evaluate_model(
        tmp_path,
        checkpoint,
        str(written.images_dir),
        stated=Stated(),
        iou_threshold=IOU_THRESHOLD,
    )

    assert _resolved(written.manifest) in evaluated["error"], evaluated


def test_evaluating_over_an_ambiguous_stem_answers_the_enumerators_refusal(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``evaluate_model`` answers a capture holding two images of one stem as its error."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.execution import Stated
    from tcip_mcp.tools.training_tools import evaluate_model
    from tests._training_values import IOU_THRESHOLD
    from tests._verified_checkpoint_fixtures import registered_checkpoint

    checkpoint = registered_checkpoint(tmp_path)
    _ambiguous(written)

    evaluated = evaluate_model(
        tmp_path,
        checkpoint,
        str(written.images_dir),
        stated=Stated(),
        iou_threshold=IOU_THRESHOLD,
    )

    assert _listed(written.images_dir) in evaluated["error"], evaluated


def test_a_plant_aware_draw_answers_an_ambiguous_stem(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``plant_aware_group_splits`` answers a capture holding two images of one stem."""
    from tcip_mcp.tools.data_tools import plant_aware_group_splits
    from tests._training_values import RATIONALE

    _ambiguous(written)

    drawn = plant_aware_group_splits(
        tmp_path,
        str(written.root),
        [str(tmp_path / "plants.csv")],
        "bud",
        0,
        0.25,
        0.0,
        0.0,
        False,
        RATIONALE,
        actor=None,
    )

    assert drawn == {"error": _listed(written.images_dir)}


def test_a_dataset_sample_render_skips_an_incomplete_band_group(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """A dataset sample render leaves out a group missing a band and renders the photograph."""
    from tcip_mcp.tools.vision_tools import DatasetRender, visualize

    written.bands["Red"].unlink()

    rendered = visualize(
        tmp_path,
        DatasetRender(source="dataset", path=str(written.root), n=SIZE),
    )

    assert "error" not in rendered, rendered
    assert rendered["total_images"] == 2


def test_building_a_plant_mapping_answers_an_ambiguous_capture(
    tmp_path: Path,
) -> None:
    """Both mapping doors answer a mapped capture holding two images of one stem.

    The tool answers the enumerator's sentence as its error and the web route as a 400; with
    the second image gone the tool builds the mapping.
    """
    from tcip_mcp.dataset_layout import image_dir
    from tcip_mcp.tools.phenology_tools import build_plant_mapping
    from tests._mapping_fixtures import (
        geolocated_captures,
        register_plant_registry_for,
        write_plots_csv,
    )

    dataset_root = tmp_path / "ds"
    dataset_root.mkdir()
    register_dataset_at(tmp_path, dataset_root, crop=CROP)
    (first, *_others) = geolocated_captures(dataset_root, [CAPTURE])[CAPTURE]
    second = write_image(first.with_suffix(".png"), (8, 8))
    registry = register_plant_registry_for(
        tmp_path, [write_plots_csv(tmp_path / "plants.csv")], crop=CROP
    )

    def built() -> dict:
        return build_plant_mapping(
            tmp_path,
            name="valley",
            dataset_root=str(dataset_root),
            plant_registry=registry,
            actor=None,
        )

    assert built() == {"error": _listed(image_dir(dataset_root, CAPTURE))}
    second.unlink()
    assert "error" not in built(), built()


def test_the_web_mapping_build_answers_an_ambiguous_capture_400(
    client: TestClient, opened: OpenProject
) -> None:
    """The web mapping build answers a mapped capture holding two images of one stem 400."""
    from tcip_mcp.dataset_layout import image_dir
    from tests._mapping_fixtures import (
        geolocated_captures,
        register_plant_registry_for,
        write_plots_csv,
    )

    dataset_root = opened.root / "ds"
    dataset_root.mkdir()
    register_dataset_at(opened.root, dataset_root, crop=CROP)
    (first, *_others) = geolocated_captures(dataset_root, [CAPTURE])[CAPTURE]
    write_image(first.with_suffix(".png"), (8, 8))
    registry = register_plant_registry_for(
        opened.root, [write_plots_csv(opened.root / "plants.csv")], crop=CROP
    )

    answered = client.post(
        "/api/results/plant_mapping/build",
        json={
            "name": "valley",
            "dataset_root": str(dataset_root),
            "plant_registry": registry,
            "user": "tester",
        },
    )

    assert answered.status_code == 400, answered.text
    assert answered.json()["detail"] == _listed(
        image_dir(dataset_root, CAPTURE)
    )


def test_selecting_an_ambiguous_capture_answers_400_and_a_snapshot_over_it_refuses(
    client: TestClient, opened: OpenProject
) -> None:
    """A capture holding two images of one stem refuses its selection and the snapshot naming it.

    The capture selects while it holds one image per stem. With a second image of one stem the
    select route answers the enumerator's sentence 400, and the persisted GUI snapshot that
    selected it refuses its read naming the project, with that sentence.
    """
    import tcip_store as ts
    from tcip_mcp.web_client import read_gui_snapshot

    written = _written(opened.root / "ds", opened.root)

    def selected() -> object:
        return client.post(
            "/api/dataset/select",
            json={"dataset_root": str(written.root), "date": CAPTURE},
        )

    assert selected().status_code == 200
    _ambiguous(written)
    sentence = _listed(written.images_dir)

    refused = selected()
    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == sentence
    with pytest.raises(ts.ReadRefusedError) as unread:
        read_gui_snapshot(opened.root)
    assert str(opened.root) in str(unread.value)
    assert sentence in str(unread.value)


def test_the_image_door_answers_an_ambiguous_stem_400(
    client: TestClient, opened: OpenProject
) -> None:
    """The image door answers a path whose stem names two images 400 with the resolver's sentence.

    The photograph serves while it is the one image of its stem.
    """
    written = _written(opened.root / "ds", opened.root)

    def served() -> object:
        return client.get(
            "/api/images",
            params={"path": str(written.plain), "display_pixels": SIZE * SIZE},
        )

    assert served().status_code == 200
    _ambiguous(written)

    refused = served()
    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == _resolved(written.plain)

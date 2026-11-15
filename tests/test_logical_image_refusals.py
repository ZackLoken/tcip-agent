"""The logical-image refusals and the doors that read them.

The resolver (``image_utils.resolve_image_paths``) and the enumerator
(``image_utils.list_logical_images``) refuse through ``LogicalImageError`` alone, none of its
subclasses a builtin error, so a door answers one only by naming the class. Each door is called
over a capture written through the platform's producers: a two-band grouped capture whose
manifest ``write_band_group_manifest`` writes, a plain photograph and a file no image suffix
names, the images labeled through the label save. A band removed after its manifest is written
makes an incomplete group, a second photograph of one case-folded stem written into the capture
makes an ambiguous stem, as ``ingest_images`` refuses to bring one in, and the manifest's record
written back without its bands, or a directory in its place, makes a manifest that will not read.
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

from tcip_mcp.pipelines.image_utils import (
    list_logical_images,
    resolve_image_path,
)
from tests._mapping_fixtures import CROP
from tests._producer_fixtures import (
    GroupedDataset,
    labeled_grouped_capture,
    refusal_sentence,
    register_dataset_at,
    unreadable_manifest,
    without_bands,
    write_image,
    write_stray,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx
    from fastapi.testclient import TestClient

    from tcip_web.state import OpenProject

CAPTURE = "2026-04-01"
SIZE = 16
"""The edge, in pixels, of every image :func:`labeled_grouped_capture` writes."""


def _written(root: Path, project: Path) -> GroupedDataset:
    """Return :func:`labeled_grouped_capture` of ``root``'s :data:`CAPTURE`, ``root`` registered.

    ``root`` is registered in ``project`` at :data:`~tests._mapping_fixtures.CROP`.
    """
    written = labeled_grouped_capture(root, CAPTURE)
    register_dataset_at(project, root, crop=CROP)
    return written


@pytest.fixture
def written(tmp_path: Path) -> GroupedDataset:
    """Return :func:`_written` of ``tmp_path / "ds"``, registered in ``tmp_path``."""
    return _written(tmp_path / "ds", tmp_path)


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
            refusal(resolve_image_path, write_stray(written.images_dir)),
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
        write_stray(written.images_dir),
        written.images_dir / "absent.jpg",
    ):
        assert stated(refused) == {
            "error": refusal_sentence(resolve_image_path, refused)
        }
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
    assert rendered() == {
        "error": refusal_sentence(resolve_image_path, written.plain)
    }


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

    assert (
        refusal_sentence(resolve_image_path, written.manifest)
        in evaluated["error"]
    ), evaluated


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

    assert (
        refusal_sentence(list_logical_images, written.images_dir)
        in evaluated["error"]
    ), evaluated


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
    """The mapping tool answers a mapped capture holding two images of one stem.

    It answers the enumerator's sentence as its error; with the second image gone it builds the
    mapping.
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

    assert built() == {
        "error": refusal_sentence(
            list_logical_images, image_dir(dataset_root, CAPTURE)
        )
    }
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
    assert answered.json()["detail"] == refusal_sentence(
        list_logical_images, image_dir(dataset_root, CAPTURE)
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
    sentence = refusal_sentence(list_logical_images, written.images_dir)

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
    assert refused.json()["detail"] == refusal_sentence(
        resolve_image_path, written.plain
    )


LATER = "2026-04-02"
"""A second capture, beside :data:`CAPTURE`, holding one photograph."""


def _assert_names_the_unread_manifest(answer: str, manifest: Path) -> None:
    """Assert ``answer`` names ``manifest``, why its reader refuses it and the next step.

    The why is the sentence ``read_band_group_manifest`` raises over ``manifest``.
    """
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest

    with pytest.raises((OSError, ValueError)) as cause:
        read_band_group_manifest(manifest)
    assert str(manifest) in answer, answer
    assert str(cause.value) in answer, answer
    assert "delete the manifest" in answer, answer


def _scanned(project: Path, written: GroupedDataset) -> dict:
    """Return ``scan_dataset``'s census of ``written``'s dataset root."""
    from tcip_mcp.tools.data_tools import scan_dataset

    return scan_dataset(project, str(written.root))


def test_scanning_a_dataset_refuses_a_capture_whose_band_group_manifest_will_not_read(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``scan_dataset`` refuses a capture whose manifest its model refuses, listing none of it.

    Before the damage the group counts as one image beside the photograph.
    """
    assert _scanned(tmp_path, written)["image_count"] == 2
    unreadable_manifest(written.manifest, without_bands)

    scanned = _scanned(tmp_path, written)

    assert set(scanned) == {"error"}, scanned
    _assert_names_the_unread_manifest(scanned["error"], written.manifest)


def test_scanning_a_dataset_refuses_a_capture_whose_band_group_manifest_cannot_be_read(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``scan_dataset`` refuses a capture whose manifest's bytes the operating system refuses.

    A directory stands where the manifest file was.
    """
    written.manifest.unlink()
    written.manifest.mkdir()

    scanned = _scanned(tmp_path, written)

    assert set(scanned) == {"error"}, scanned
    _assert_names_the_unread_manifest(scanned["error"], written.manifest)


def test_stating_a_capture_date_refuses_every_image_of_a_capture_whose_manifest_will_not_read(
    written: GroupedDataset, tmp_path: Path
) -> None:
    """``state_capture_date`` refuses a former band and the photograph beside it, moving neither."""
    from tcip_mcp.tools.ingest_tools import state_capture_date

    unreadable_manifest(written.manifest, without_bands)

    for image in (written.bands["Green"], written.plain):
        stated = state_capture_date(
            tmp_path,
            images=[str(image)],
            date=datetime.date(2026, 4, 2),
            rationale="the field book dates this walk",
            actor=None,
        )
        assert set(stated) == {"error"}, stated
        _assert_names_the_unread_manifest(stated["error"], written.manifest)
        assert image.is_file()


def test_the_image_door_answers_a_band_of_an_unread_manifest_409(
    client: TestClient, opened: OpenProject
) -> None:
    """The image door answers a former band 409 once its manifest will not read.

    While the manifest reads, the band names no logical image and answers 404.
    """
    written = _written(opened.root / "ds", opened.root)

    def served() -> httpx.Response:
        return client.get(
            "/api/images",
            params={
                "path": str(written.bands["Green"]),
                "display_pixels": SIZE * SIZE,
            },
        )

    assert served().status_code == 404
    unreadable_manifest(written.manifest, without_bands)

    refused = served()
    assert refused.status_code == 409, refused.text
    _assert_names_the_unread_manifest(
        refused.json()["detail"], written.manifest
    )


class _Census(NamedTuple):
    """A project whose :data:`CAPTURE` manifest will not read, beside a sound project.

    ``damaged`` holds :data:`CAPTURE`'s grouped capture and a photograph of :data:`LATER`;
    ``sound`` one photograph of :data:`CAPTURE`.
    """

    damaged: OpenProject
    written: GroupedDataset
    sound: OpenProject


def _census(made: OpenProject) -> _Census:
    """Write :class:`_Census` with ``made`` the damaged project and its manifest made unreadable."""
    from tcip_mcp.dataset_layout import image_dir
    from tests._web_fixtures import named_project

    written = labeled_grouped_capture(made.root, CAPTURE)
    write_image(image_dir(made.root, LATER) / "later.jpg", (SIZE, SIZE))
    sound = named_project(made.root.parent / "sound", "sound")
    write_image(image_dir(sound.root, CAPTURE) / "a.jpg", (SIZE, SIZE))
    unreadable_manifest(written.manifest, without_bands)
    return _Census(made, written, sound)


def _listed_image_counts(client: TestClient) -> dict[str, object]:
    """Return the ``image_count`` of each readable project the listing serves, by its root."""
    listed = client.get("/api/projects")
    assert listed.status_code == 200, listed.text
    return {
        p["root"]: p["image_count"]
        for p in listed.json()["projects"]
        if p["kind"] == "read"
    }


def test_the_project_listing_states_an_uncounted_capture_and_counts_a_sound_project(
    client: TestClient, made: OpenProject
) -> None:
    """The listing serves the damaged project's refusal as its image count, the sound's total."""
    census = _census(made)

    counts = _listed_image_counts(client)

    assert counts[str(census.sound.root)] == 1
    uncounted = counts[str(census.damaged.root)]
    assert isinstance(uncounted, dict) and set(uncounted) == {"problem"}, (
        uncounted
    )
    _assert_names_the_unread_manifest(
        uncounted["problem"], census.written.manifest
    )


def test_the_overview_states_an_uncounted_capture_and_keeps_its_date(
    made: OpenProject,
) -> None:
    """``inspect_project`` serves the refusal as its image count and lists every capture's date."""
    from tcip_mcp.tools.project_tools import inspect_project

    census = _census(made)

    status = inspect_project(census.damaged.root)

    assert status["dates"] == [CAPTURE, LATER]
    uncounted = status["image_count"]
    assert isinstance(uncounted, dict) and set(uncounted) == {"problem"}, (
        uncounted
    )
    _assert_names_the_unread_manifest(
        uncounted["problem"], census.written.manifest
    )


def test_the_project_listing_and_the_overview_serve_one_image_count(
    client: TestClient, made: OpenProject
) -> None:
    """The listing's ``image_count`` of each project is the one ``inspect_project`` serves."""
    from tcip_mcp.tools.project_tools import inspect_project

    census = _census(made)

    counts = _listed_image_counts(client)

    for project in (census.damaged, census.sound):
        assert (
            counts[str(project.root)]
            == inspect_project(project.root)["image_count"]
        )


def test_reopening_a_project_whose_saved_selection_lies_in_an_unread_capture_answers_409(
    client: TestClient, opened: OpenProject
) -> None:
    """A selection saved in a capture whose manifest then will not read refuses the reopen.

    The capture is selected through the select route, the manifest made unreadable, and the
    project closed; its open answers 409 carrying the refusal, and nothing is open.
    """
    from tcip_web.state import store

    written = _written(opened.root / "ds", opened.root)
    selected = client.post(
        "/api/dataset/select",
        json={"dataset_root": str(written.root), "date": CAPTURE},
    )
    assert selected.status_code == 200, selected.text
    unreadable_manifest(written.manifest, without_bands)
    asyncio.run(store.close_project(opened.id))

    reopened = client.post(
        "/api/projects/open", json={"id": opened.id, "user": "grower"}
    )

    assert reopened.status_code == 409, reopened.text
    _assert_names_the_unread_manifest(
        reopened.json()["detail"]["refusal"]["message"], written.manifest
    )
    assert client.get("/api/projects").json()["open_id"] is None

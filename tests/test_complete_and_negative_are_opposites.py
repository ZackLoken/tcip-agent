"""``complete`` and ``negative`` are opposite outcomes of one human action, not degrees of one.

Marking an image done when it holds content for the subject records ``complete``; marking it done
when it holds none records ``negative``, the confirmed negative that trains as an empty image. The
derivation, the store the GUI writes through, and the reader training uses are all driven here, so
a flipped token or a dropped subject scope shows up as an image changing sides.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_annotation.json_io import write_annotations
from tcip_annotation.state import Annotation, BBox
from tcip_mcp.pipelines.data.label_queries import confirmed_negative_names
from tcip_web.app import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1")


def _box(subject: str, x1: float, y1: float, x2: float, y2: float) -> Annotation:
    return Annotation(subject=subject, geometry=BBox(x1, y1, x2, y2))


@pytest.fixture
def labelled_dataset(tmp_path: Path) -> Path:
    """Three labelled images carrying different subjects, and one image with no label file."""
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(
        str(labels / "img_buds.json"),
        [_box("bud", 12, 30, 48, 140), _box("bud", 300, 44, 372, 70)],
        900, 500,
    )
    write_annotations(str(labels / "img_bush.json"), [_box("bush", 5, 9, 640, 480)], 900, 500)
    write_annotations(str(labels / "img_blank.json"), [], 900, 500, keep_empty=True)
    return tmp_path


def _derive(client: TestClient, root: Path, subject: str, images: list[str],
            done: list[str]) -> dict[str, str]:
    resp = client.post(
        "/api/classes/image_status/derive",
        json={"project_root": str(root), "annotations_dir": str(root / "annotations"),
              "subject": subject, "image_list": images, "complete_override": done},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["statuses"]


def _store(client: TestClient, root: Path, subject: str, statuses: dict[str, str]) -> None:
    resp = client.post(
        "/api/classes/image_status/bulk",
        json={"project_root": str(root), "dataset_root": str(root), "subject": subject,
              "statuses": statuses},
    )
    assert resp.status_code == 200, resp.text


def _read_back(client: TestClient, root: Path, subject: str) -> dict[str, str]:
    resp = client.get(
        "/api/classes/image_status",
        params={"project_root": str(root), "dataset_root": str(root), "subject": subject},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["statuses"]


def test_an_image_finished_with_content_never_reads_back_as_a_confirmed_negative(
    client: TestClient, labelled_dataset: Path
) -> None:
    """Content for the subject makes a finished image ``complete``; content belonging only to some
    other subject leaves it empty for this one, which is what a confirmed negative means."""
    root = labelled_dataset
    images = ["img_buds.jpg", "img_bush.jpg", "img_blank.jpg", "img_missing.jpg"]
    derived = _derive(client, root, "bud", images,
                      ["img_buds.jpg", "img_bush.jpg", "img_blank.jpg"])
    assert derived == {
        "img_buds.jpg": "complete",
        "img_bush.jpg": "negative",
        "img_blank.jpg": "negative",
        "img_missing.jpg": "unannotated",
    }

    _store(client, root, "bud", derived)
    negatives = confirmed_negative_names(root / "annotations", subject="bud", date=None)
    assert negatives == {"img_bush.jpg", "img_blank.jpg"}
    assert "img_buds.jpg" not in negatives
    assert _read_back(client, root, "bud")["img_buds.jpg"] == "complete"


def test_one_image_is_a_negative_for_one_subject_and_finished_for_another(
    client: TestClient, labelled_dataset: Path
) -> None:
    """A confirmation is a statement about one subject on one image: the bush image is a bud
    negative while it is a finished bush image, and the bud image is the mirror of that."""
    root = labelled_dataset
    images = ["img_buds.jpg", "img_bush.jpg"]
    for subject in ("bud", "bush"):
        _store(client, root, subject, _derive(client, root, subject, images, images))

    assert confirmed_negative_names(root / "annotations", subject="bud", date=None) == {"img_bush.jpg"}
    assert confirmed_negative_names(root / "annotations", subject="bush", date=None) == {"img_buds.jpg"}
    assert _read_back(client, root, "bud")["img_buds.jpg"] == "complete"
    assert _read_back(client, root, "bush")["img_bush.jpg"] == "complete"


def test_confirmations_recorded_under_different_dates_stay_separate(
    client: TestClient, tmp_path: Path
) -> None:
    """The same image name reviewed under two capture dates carries two independent statuses, so a
    negative recorded on one date can never overwrite a finished image on the other."""
    for date, status in (("2026-03-02", "negative"), ("2026-03-09", "complete")):
        resp = client.post(
            "/api/classes/image_status",
            json={"project_root": str(tmp_path), "dataset_root": str(tmp_path),
                  "image_name": "IMG_0007.JPG", "status": status, "subject": "bud",
                  "date": date},
        )
        assert resp.status_code == 200, resp.text

    def read(date: str | None) -> dict[str, str]:
        params = {"project_root": str(tmp_path), "dataset_root": str(tmp_path),
                  "subject": "bud"}
        if date:
            params["date"] = date
        return client.get("/api/classes/image_status", params=params).json()["statuses"]

    assert read("2026-03-02") == {"IMG_0007.JPG": "negative"}
    assert read("2026-03-09") == {"IMG_0007.JPG": "complete"}
    assert read(None) == {}


def test_confirmed_negative_with_an_empty_label_file_still_carries(
    client: TestClient, tmp_path: Path
) -> None:
    """A confirmed negative whose label file records nothing for the subject is admitted: the
    rail must not refuse the case it exists to let through."""
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(str(labels / "img_blank.json"), [], 900, 500, keep_empty=True)
    _store(client, tmp_path, "subject_a", {"img_blank.jpg": "negative"})

    assert confirmed_negative_names(labels, subject="subject_a", date=None) == {"img_blank.jpg"}


def test_confirmed_negative_excludes_a_name_the_label_file_now_contradicts(
    client: TestClient, tmp_path: Path
) -> None:
    """A stored negative and a label file that now carries the subject disagree about the same
    image; the name is excluded from the negative set (never trained as an empty image on the
    store's say-so) and reported through ``contradicted_out`` rather than silently dropped."""
    labels = tmp_path / "annotations"
    labels.mkdir()
    write_annotations(str(labels / "img_bush.json"), [_box("bush", 5, 9, 640, 480)], 900, 500)
    _store(client, tmp_path, "subject_a", {"img_bush.jpg": "negative"})
    write_annotations(
        str(labels / "img_bush.json"),
        [_box("bush", 5, 9, 640, 480), _box("subject_a", 12, 30, 48, 140)],
        900, 500,
    )

    contradicted: set[str] = set()
    negatives = confirmed_negative_names(
        labels, subject="subject_a", date=None, contradicted_out=contradicted)
    assert negatives == set()
    assert contradicted == {"img_bush.jpg"}


def test_a_contradicted_negative_still_trains_on_its_actual_content(
    client: TestClient, tmp_path: Path
) -> None:
    """The exclusion never drops the image: its label file holds real content, so a trainable-stems
    enumeration over the same directory admits it by that content, the rail admitting valid work
    rather than silently shrinking the run's negative count."""
    from tcip_mcp.pipelines.data.label_queries import trainable_stems

    labels = tmp_path / "annotations"
    labels.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    (images / "img_bush.jpg").write_bytes(b"")
    write_annotations(str(labels / "img_bush.json"), [_box("bush", 5, 9, 640, 480)], 900, 500)
    _store(client, tmp_path, "subject_a", {"img_bush.jpg": "negative"})
    write_annotations(
        str(labels / "img_bush.json"),
        [_box("bush", 5, 9, 640, 480), _box("subject_a", 12, 30, 48, 140)],
        900, 500,
    )

    contradicted: set[str] = set()
    stems, counts = trainable_stems(
        labels, images, subject="subject_a", date=None, contradicted_out=contradicted)
    assert stems == ["img_bush"]
    assert counts["annotated"] == 1
    assert counts["confirmed_negative"] == 0
    assert contradicted == {"img_bush.jpg"}

"""Review flags through the one label save.

Covers a comment raised on a place, a proposal or the image, the reply that resolves it, the
removal that resolves it, and what the editor reads back.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from tcip_annotation.flags import (
    FlagRequest,
    PointTarget,
    ProposalTarget,
    flag_key,
    flags_of,
    flags_record,
    read_flags,
)
from tcip_annotation.json_io import read_label_document
from tcip_mcp.dataset_layout import Gestures, image_dir
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import (
    image_label_key,
    save_labels,
    staged_box,
    write_image,
)

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from tcip_annotation.flags import Flag

DATE = "2026-02-11"
WIDTH, HEIGHT = 100, 80
BOX = [40.0, 30.0, 60.0, 50.0]
INSIDE = (50.0, 40.0)
ON_BOX = PointTarget(point=INSIDE, subject="bud")


def _flags(image: Path) -> list[Flag]:
    return read_flags(flag_key(image_label_key(image)))


def test_a_flag_on_an_annotation_is_recorded_with_who_and_when(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path,
        image,
        [{"subject": "bud", "bbox": BOX}],
        gestures=Gestures(
            flag=(FlagRequest(text="  is this one open?  ", target=ON_BOX),)
        ),
    )

    (flag,) = _flags(image)
    assert (flag.request.text, flag.by, flag.request.target) == (
        "is this one open?",
        "user:breeder",
        ON_BOX,
    )
    assert flag.open and flag.at
    assert flags_of(flags_record([flag])) == [flag]


def test_a_flag_on_the_image_and_one_on_a_proposal_are_each_their_own_target(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    save_labels(
        tmp_path,
        image,
        [],
        gestures=Gestures(
            flag=(
                FlagRequest(text="glare across the frame", target=None),
                FlagRequest(
                    text="bud or gall?",
                    target=ProposalTarget(bucket=bucket, index=0),
                ),
            )
        ),
    )

    whole, on_proposal = _flags(image)
    assert whole.request.target is None
    assert on_proposal.request.target == ProposalTarget(bucket=bucket, index=0)


@pytest.mark.parametrize(
    "fields",
    [
        {"text": "   ", "target": None},
        {"text": "where?", "target": {"point": INSIDE}},
        {"text": "which?", "target": {"subject": "bud"}},
        {
            "text": "both?",
            "target": {"point": INSIDE, "subject": "bud", "index": 0},
        },
    ],
    ids=[
        "blank_comment",
        "point_without_subject",
        "subject_without_point",
        "point_beside_a_proposal",
    ],
)
def test_a_malformed_flag_request_refuses_at_its_construction(
    fields: dict,
) -> None:
    with pytest.raises(ValidationError, match="for FlagRequest"):
        FlagRequest(**fields)


def test_a_flag_or_a_resolution_by_no_person_refuses_before_any_write(
    tmp_path: Path,
) -> None:
    """A flag raised or resolved by no person is refused, and nothing is written.

    A save whose producer names no person, as an agent's save names itself, neither raises a
    flag nor resolves one by removing its annotation; the document and the audit log are
    unchanged.
    """
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    box = [{"subject": "bud", "bbox": BOX}]
    with pytest.raises(
        ValueError, match="is not a person's recorded identity"
    ):
        save_labels(
            tmp_path,
            image,
            box,
            author="save_annotations",
            gestures=Gestures(flag=(FlagRequest(text="open?", target=None),)),
        )
    assert _flags(image) == []

    save_labels(
        tmp_path,
        image,
        box,
        gestures=Gestures(flag=(FlagRequest(text="open?", target=ON_BOX),)),
    )
    held, logged = (
        read_label_document(image_label_key(image)),
        audit_rows(tmp_path),
    )
    with pytest.raises(
        ValueError, match="is not a person's recorded identity"
    ):
        save_labels(tmp_path, image, [], author="save_annotations")
    assert _flags(image)[0].open
    assert read_label_document(image_label_key(image)) == held
    assert audit_rows(tmp_path) == logged


def test_a_flag_on_a_proposal_the_bucket_does_not_hold_refuses(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    bucket = staged_box(tmp_path, image, BOX)
    with pytest.raises(ValueError, match="not one at index 3 to flag"):
        save_labels(
            tmp_path,
            image,
            [],
            gestures=Gestures(
                flag=(
                    FlagRequest(
                        text="?", target=ProposalTarget(bucket=bucket, index=3)
                    ),
                )
            ),
        )
    assert _flags(image) == []


def test_resolving_keeps_the_flag_with_the_reply_and_refuses_a_second_resolution(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    box = [{"subject": "bud", "bbox": BOX}]
    save_labels(
        tmp_path,
        image,
        box,
        gestures=Gestures(flag=(FlagRequest(text="open?", target=ON_BOX),)),
    )
    (raised,) = _flags(image)

    save_labels(
        tmp_path,
        image,
        box,
        author="user:second",
        gestures=Gestures(resolve={raised.id: "yes, open"}),
    )

    (flag,) = _flags(image)
    assert flag.resolution is not None
    assert (
        flag.resolution.by,
        flag.resolution.reply,
        flag.resolution.removed,
    ) == ("user:second", "yes, open", False)
    assert (flag.request, flag.by) == (raised.request, raised.by)

    with pytest.raises(ValueError, match="are not open"):
        save_labels(
            tmp_path,
            image,
            box,
            gestures=Gestures(resolve={raised.id: "again"}),
        )
    with pytest.raises(ValueError, match="are not open"):
        save_labels(
            tmp_path,
            image,
            box,
            gestures=Gestures(resolve={"no-such-flag": ""}),
        )


def test_removing_the_flagged_annotation_resolves_its_flag_and_no_other(
    tmp_path: Path,
) -> None:
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    other = [5.0, 5.0, 20.0, 20.0]
    both = [{"subject": "bud", "bbox": BOX}, {"subject": "bud", "bbox": other}]
    save_labels(
        tmp_path,
        image,
        both,
        gestures=Gestures(
            flag=(
                FlagRequest(text="on the first", target=ON_BOX),
                FlagRequest(
                    text="on the second",
                    target=PointTarget(point=(10.0, 10.0), subject="bud"),
                ),
                FlagRequest(text="on the image", target=None),
            )
        ),
    )

    save_labels(tmp_path, image, both[1:], author="user:second")

    first, second, whole = _flags(image)
    assert first.resolution is not None
    assert (first.resolution.removed, first.resolution.by) == (
        True,
        "user:second",
    )
    assert second.open and whole.open


def test_moving_a_flagged_annotation_off_its_flag_resolves_it_as_removed(
    tmp_path: Path,
) -> None:
    """Moving a flagged annotation off its flag's place resolves the flag as removed.

    A small correction leaves the flag open, and a correction that no longer covers the flag's
    place resolves it as removed.
    """
    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path,
        image,
        [{"subject": "bud", "bbox": BOX}],
        gestures=Gestures(flag=(FlagRequest(text="open?", target=ON_BOX),)),
    )

    save_labels(
        tmp_path, image, [{"subject": "bud", "bbox": [41.0, 31.0, 61.0, 51.0]}]
    )
    assert _flags(image)[0].open

    save_labels(
        tmp_path, image, [{"subject": "bud", "bbox": [0.0, 0.0, 10.0, 10.0]}]
    )
    resolution = _flags(image)[0].resolution
    assert resolution is not None and resolution.removed


def test_the_editor_reads_the_flags_at_load_and_after_a_save(
    tmp_path: Path, client: TestClient
) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    body = {
        "image_path": str(image),
        "user": "breeder",
        "annotations": [{"subject": "bud", "bbox": BOX}],
    }

    saved = client.post(
        "/api/annotate/labels",
        json={
            **body,
            "gestures": {
                "flag": [
                    {
                        "text": "open?",
                        "target": {"point": list(INSIDE), "subject": "bud"},
                    }
                ]
            },
        },
    )
    assert saved.status_code == 200, saved.text
    (flag,) = saved.json()["document"]["flags"]
    assert flag["request"]["text"] == "open?" and flag["resolution"] is None

    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()
    assert loaded["flags"] == [flag]

    resolved = client.post(
        "/api/annotate/labels",
        json={**body, "gestures": {"resolve": {flag["id"]: "confirmed open"}}},
    )
    assert resolved.status_code == 200, resolved.text
    (resolved_flag,) = resolved.json()["document"]["flags"]
    assert resolved_flag["resolution"]["reply"] == "confirmed open"

    again = client.post(
        "/api/annotate/labels",
        json={**body, "gestures": {"resolve": {flag["id"]: "x"}}},
    )
    assert again.status_code == 400, again.text


def test_each_served_flag_names_the_annotation_whose_rings_hold_its_point(
    tmp_path: Path, client: TestClient
) -> None:
    """A served flag names every annotation of its subject whose geometry holds its point.

    Beside a box, a flag inside an L-shaped polygon's rings and a box overlapping it is served
    on both, by their indices in the document, and one inside the polygon's bounding box but
    outside its rings on no annotation, as is a flag on the image as a whole and one inside the
    first box naming another subject; the save and the load answer the same flags.
    """
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT))
    ring = [[10, 10], [70, 10], [70, 20], [20, 20], [20, 70], [10, 70]]
    saved = client.post(
        "/api/annotate/labels",
        json={
            "image_path": str(image),
            "user": "breeder",
            "annotations": [
                {"subject": "bud", "bbox": [80.0, 5.0, 95.0, 15.0]},
                {"subject": "bud", "rings": [ring]},
                {"subject": "bud", "bbox": [5.0, 30.0, 25.0, 50.0]},
            ],
            "gestures": {
                "flag": [
                    {
                        "text": "in the notch",
                        "target": {"point": [50.0, 50.0], "subject": "bud"},
                    },
                    {
                        "text": "on the arm",
                        "target": {"point": [15.0, 40.0], "subject": "bud"},
                    },
                    {"text": "glare", "target": None},
                    {
                        "text": "a leaf here?",
                        "target": {"point": [85.0, 10.0], "subject": "leaf"},
                    },
                ]
            },
        },
    )
    assert saved.status_code == 200, saved.text

    loaded = client.get(
        "/api/annotate/labels", params={"image_path": str(image)}
    ).json()
    assert [f["covered_by"] for f in loaded["flags"]] == [[], [1, 2], [], []]
    assert saved.json()["document"]["flags"] == loaded["flags"]


def test_doctor_names_a_flags_record_that_will_not_read_and_passes_one_that_does(
    tmp_path: Path,
) -> None:
    import tcip_store as ts
    from tcip_mcp.tools.doctor_tools import _buckets, check_state

    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(
        tmp_path,
        image,
        [{"subject": "bud", "bbox": BOX}],
        gestures=Gestures(flag=(FlagRequest(text="open?", target=ON_BOX),)),
    )
    findings: list = []
    check_state(tmp_path, findings, buckets=_buckets(tmp_path, findings))
    assert findings == []

    ts.replace(flag_key(image_label_key(image)), {"flags": [{"id": "x"}]})
    check_state(tmp_path, findings, buckets=_buckets(tmp_path, findings))
    assert [level for level, _ in findings] == ["warn"]
    assert (
        "flags record" in findings[0][1] and "will not read" in findings[0][1]
    )


def test_a_save_that_touches_no_flag_writes_no_flags_record(
    tmp_path: Path,
) -> None:
    import tcip_store as ts

    image = write_image(image_dir(tmp_path, DATE) / "a.jpg", (WIDTH, HEIGHT))
    save_labels(tmp_path, image, [{"subject": "bud", "bbox": BOX}])
    assert not ts.exists(flag_key(image_label_key(image)))

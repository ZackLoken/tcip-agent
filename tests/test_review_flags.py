"""Review flags through the one label save: a comment raised on a place, a proposal or the
image, the reply that resolves it, the removal that resolves it, and what the editor reads
back."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_annotation.flags import FlagRequest, decode_flag, encode_flag, flag_key, read_flags
from tcip_annotation.json_io import read_label_document
from tcip_mcp.dataset_layout import Gestures, image_dir, save_label_document
from tests._audit_fixtures import audit_rows
from tests._producer_fixtures import image_label_key, write_image

DATE = "2026-02-11"
WIDTH, HEIGHT = 100, 80
BOX = [40.0, 30.0, 60.0, 50.0]
INSIDE = (50.0, 40.0)


def _image(root: Path) -> Path:
    return write_image(image_dir(root, DATE) / "a.jpg", (WIDTH, HEIGHT), (90, 110, 70))


def _save(root: Path, image: Path, payloads: list[dict], *, author: str = "user:breeder",
          gestures: Gestures = Gestures()):
    return save_label_document(root, image_label_key(image), payloads, width=WIDTH,
                               height=HEIGHT, author=author, actor=author, gestures=gestures)


def _flags(image: Path):
    return read_flags(flag_key(image_label_key(image)))


def _bucket(root: Path, image: Path) -> str:
    from tcip_mcp.tools.proposal_tools import stage_proposals

    x1, y1, x2, y2 = BOX
    staged = stage_proposals(root, str(image), model_name="detector", boxes=[{
        "subject": "bud", "conf": 0.9, "cx": (x1 + x2) / 2 / WIDTH, "cy": (y1 + y2) / 2 / HEIGHT,
        "w": (x2 - x1) / WIDTH, "h": (y2 - y1) / HEIGHT}])
    assert "error" not in staged, staged
    return staged["bucket"]


def test_a_flag_on_an_annotation_is_recorded_with_who_and_when(tmp_path: Path) -> None:
    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}], gestures=Gestures(
        flag=(FlagRequest(text="  is this one open?  ", point=INSIDE, subject="bud"),)))

    (flag,) = _flags(image)
    assert (flag.text, flag.by, flag.point, flag.subject) == (
        "is this one open?", "user:breeder", INSIDE, "bud")
    assert flag.open and flag.at
    assert decode_flag(encode_flag(flag)) == flag


def test_a_flag_on_the_image_and_one_on_a_proposal_are_each_their_own_target(
        tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    _save(tmp_path, image, [], gestures=Gestures(flag=(
        FlagRequest(text="glare across the frame"),
        FlagRequest(text="bud or gall?", proposal=(bucket, 0)))))

    whole, on_proposal = _flags(image)
    assert (whole.point, whole.proposal) == (None, None)
    assert on_proposal.proposal == (bucket, 0)


@pytest.mark.parametrize("request_, message", [
    (FlagRequest(text="   "), "does not read|is not a comment"),
    (FlagRequest(text="where?", point=INSIDE), "is not a comment"),
    (FlagRequest(text="which?", subject="bud"), "is not a comment"),
], ids=["blank_comment", "point_without_subject", "subject_without_point"])
def test_a_malformed_flag_refuses_before_any_write(
        tmp_path: Path, request_: FlagRequest, message: str) -> None:
    import tcip_store as ts

    image = _image(tmp_path)
    with pytest.raises(ValueError, match=message):
        _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}],
              gestures=Gestures(flag=(request_,)))

    assert ts.read(image_label_key(image), default=None) is None
    assert _flags(image) == []


def test_a_flag_or_a_resolution_by_no_person_refuses_before_any_write(tmp_path: Path) -> None:
    """A flag's ``by`` and ``resolved_by`` name a person; a producer that names none, as an
    agent's save names itself, neither raises a flag nor resolves one by removing its annotation."""
    image = _image(tmp_path)
    box = [{"subject": "bud", "bbox": BOX}]
    with pytest.raises(ValueError, match="by a person"):
        _save(tmp_path, image, box, author="save_annotations",
              gestures=Gestures(flag=(FlagRequest(text="open?"),)))
    assert _flags(image) == []

    _save(tmp_path, image, box, gestures=Gestures(
        flag=(FlagRequest(text="open?", point=INSIDE, subject="bud"),)))
    held, logged = read_label_document(image_label_key(image)), audit_rows(tmp_path)
    with pytest.raises(ValueError, match="by a person"):
        _save(tmp_path, image, [], author="save_annotations")
    assert _flags(image)[0].open
    assert read_label_document(image_label_key(image)) == held
    assert audit_rows(tmp_path) == logged


def test_a_flag_on_a_proposal_the_bucket_does_not_hold_refuses(tmp_path: Path) -> None:
    image = _image(tmp_path)
    bucket = _bucket(tmp_path, image)
    with pytest.raises(ValueError, match="not one at index 3 to flag"):
        _save(tmp_path, image, [], gestures=Gestures(
            flag=(FlagRequest(text="?", proposal=(bucket, 3)),)))
    assert _flags(image) == []


def test_resolving_keeps_the_flag_with_the_reply_and_refuses_a_second_resolution(
        tmp_path: Path) -> None:
    image = _image(tmp_path)
    box = [{"subject": "bud", "bbox": BOX}]
    _save(tmp_path, image, box, gestures=Gestures(
        flag=(FlagRequest(text="open?", point=INSIDE, subject="bud"),)))
    (raised,) = _flags(image)

    _save(tmp_path, image, box, author="user:second",
          gestures=Gestures(resolve={raised.id: "yes, open"}))

    (flag,) = _flags(image)
    assert not flag.open
    assert (flag.resolved_by, flag.reply, flag.removed) == ("user:second", "yes, open", False)
    assert (flag.text, flag.by) == (raised.text, raised.by)

    with pytest.raises(ValueError, match="are not open"):
        _save(tmp_path, image, box, gestures=Gestures(resolve={raised.id: "again"}))
    with pytest.raises(ValueError, match="are not open"):
        _save(tmp_path, image, box, gestures=Gestures(resolve={"no-such-flag": ""}))


def test_removing_the_flagged_annotation_resolves_its_flag_and_no_other(tmp_path: Path) -> None:
    image = _image(tmp_path)
    other = [5.0, 5.0, 20.0, 20.0]
    both = [{"subject": "bud", "bbox": BOX}, {"subject": "bud", "bbox": other}]
    _save(tmp_path, image, both, gestures=Gestures(flag=(
        FlagRequest(text="on the first", point=INSIDE, subject="bud"),
        FlagRequest(text="on the second", point=(10.0, 10.0), subject="bud"),
        FlagRequest(text="on the image"))))

    _save(tmp_path, image, both[1:], author="user:second")

    first, second, whole = _flags(image)
    assert (first.open, first.removed, first.resolved_by) == (False, True, "user:second")
    assert second.open and whole.open


def test_moving_a_flagged_annotation_off_its_flag_resolves_it_as_removed(tmp_path: Path) -> None:
    """The flag marks a place: an annotation corrected so far that it no longer covers the place
    leaves nothing there to look at twice."""
    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}], gestures=Gestures(
        flag=(FlagRequest(text="open?", point=INSIDE, subject="bud"),)))

    _save(tmp_path, image, [{"subject": "bud", "bbox": [41.0, 31.0, 61.0, 51.0]}])
    assert _flags(image)[0].open

    _save(tmp_path, image, [{"subject": "bud", "bbox": [0.0, 0.0, 10.0, 10.0]}])
    assert _flags(image)[0].removed


def test_the_editor_reads_the_flags_at_load_and_after_a_save(tmp_path: Path, client) -> None:
    from tests._web_fixtures import open_new_project

    root = open_new_project(tmp_path / "proj").root
    image = _image(root)
    body = {"image_path": str(image), "user": "breeder",
            "annotations": [{"subject": "bud", "bbox": BOX}]}

    saved = client.post("/api/annotate/labels", json={**body, "flag": [
        {"text": "open?", "point": list(INSIDE), "subject": "bud"}]})
    assert saved.status_code == 200, saved.text
    (flag,) = saved.json()["flags"]
    assert flag["text"] == "open?" and flag["resolved_by"] is None

    loaded = client.get("/api/annotate/labels", params={"image_path": str(image)}).json()
    assert loaded["flags"] == [flag]

    resolved = client.post("/api/annotate/labels", json={
        **body, "resolve": {flag["id"]: "confirmed open"}})
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["flags"][0]["reply"] == "confirmed open"

    again = client.post("/api/annotate/labels", json={**body, "resolve": {flag["id"]: "x"}})
    assert again.status_code == 400, again.text


def test_doctor_names_a_flags_record_that_will_not_read_and_passes_one_that_does(
        tmp_path: Path) -> None:
    import tcip_store as ts

    from tcip_mcp.cli.doctor import check_state

    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}], gestures=Gestures(
        flag=(FlagRequest(text="open?", point=INSIDE, subject="bud"),)))
    findings: list = []
    check_state(tmp_path, findings)
    assert findings == []

    ts.replace(flag_key(image_label_key(image)), {"flags": [{"id": "x"}]})
    check_state(tmp_path, findings)
    assert [level for level, _ in findings] == ["warn"]
    assert "flags record" in findings[0][1] and "will not read" in findings[0][1]


def test_a_save_that_touches_no_flag_writes_no_flags_record(tmp_path: Path) -> None:
    import tcip_store as ts

    image = _image(tmp_path)
    _save(tmp_path, image, [{"subject": "bud", "bbox": BOX}])
    assert not ts.exists(flag_key(image_label_key(image)))

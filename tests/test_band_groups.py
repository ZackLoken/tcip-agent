"""Sensor-agnostic band-group correlation (``pipelines.data.band_groups``).

Real DJI sample data (16 captures x G/NIR/R/RE single-band GeoTIFFs, embedded XMP), copied into
``tmp_path``, backs the detection tests where it is present. Synthetic fixtures build minimal
TIFFs with the DJI XMP tag shape (an attribute form and a nested-element form), among them a
duplicate band identity and a sensor with no embedded metadata.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir, image_root
from tcip_mcp.pipelines.data.band_groups import _EMBEDDED_METADATA_STRATEGIES

if TYPE_CHECKING:
    from collections.abc import Callable

    from tcip_store import Version

REAL_DJI_DIR = image_dir(
    r"C:\Users\breeder\tcip-projects\currant_multispectral_valley-farm-2023",
    "2023-05-23",
)
requires_real_dji_data = pytest.mark.skipif(
    not REAL_DJI_DIR.is_dir(),
    reason="real DJI multispectral sample project not present on this machine",
)


STRATEGY = _EMBEDDED_METADATA_STRATEGIES[0]
"""The DJI-shaped strategy row whose tags the synthetic packets carry."""

((TIME_TAG,), POSITION_TAGS) = (tags for _, tags in STRATEGY.identity_signals)


def _xmp(
    capture_uuid: str,
    band_name: str,
    wavelength: float,
    *,
    utc: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> bytes:
    """Return a minimal XMP description in the DJI shape, its tags those :data:`STRATEGY` names.

    The group id and each of the timestamp ``utc`` and the fix ``lat``, ``lon`` given are
    attributes; the band name and the wavelength are child elements. ElementTree serializes it.
    """
    from xml.etree.ElementTree import Element, SubElement, tostring

    namespaces = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        STRATEGY.group_id_tag.partition(":")[
            0
        ]: "http://www.dji.com/drone-dji/1.0/",
        STRATEGY.band_id_tag.partition(":")[0]: "http://pix4d.com/camera/1.0",
    }
    tags = (STRATEGY.group_id_tag, TIME_TAG, *POSITION_TAGS)
    description = Element(
        "rdf:Description",
        {
            **{f"xmlns:{prefix}": uri for prefix, uri in namespaces.items()},
            **{
                tag: str(value)
                for tag, value in zip(tags, (capture_uuid, utc, lat, lon))
                if value is not None
            },
        },
    )
    for tag, text in (
        (STRATEGY.band_id_tag, band_name),
        (STRATEGY.wavelength_tag, wavelength),
    ):
        SubElement(description, tag).text = str(text)
    return tostring(description)


def _write_packet(path: Path, packet: bytes, fill: int = 100) -> Path:
    """Write a 4 by 4 TIFF of ``fill`` carrying ``packet`` as its XMP tag; return ``path``."""
    tifffile.imwrite(
        str(path),
        np.full((4, 4), fill, dtype=np.uint16),
        extratags=[(700, "B", 0, packet, True)],
    )
    return path


@pytest.fixture
def dji_copy(tmp_path: Path) -> Path:
    """Return the real 64-file sample, copied into an isolated tmp dir (never mutated in place)."""
    dest = image_root(tmp_path)
    dest.mkdir()
    for p in REAL_DJI_DIR.iterdir():
        shutil.copy2(p, dest / p.name)
    return dest


# ── embedded-metadata detection, against the real sample ──────────────────────────────────


@requires_real_dji_data
def test_detects_all_16_captures_from_real_dji_data(dji_copy: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    result = detect_and_write_band_groups(dji_copy)
    assert len(result["formed"]) == 16
    assert result["refused"] == []
    for g in result["formed"]:
        assert sorted(g["bands"]) == ["Green", "NIR", "Red", "RedEdge"]
        assert g["source"] == "embedded-metadata"
    # No physical rename/move: every original file is still exactly where it was.
    assert len(list(dji_copy.glob("*.TIF"))) == 64


@requires_real_dji_data
def test_real_data_manifest_records_true_wavelengths(dji_copy: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
        read_band_group_manifest,
    )

    result = detect_and_write_band_groups(dji_copy)
    ref = read_band_group_manifest(Path(result["manifests"][0]))
    assert ref.manifest.central_wavelength_nm == {
        "Green": 560.0,
        "NIR": 860.0,
        "Red": 650.0,
        "RedEdge": 730.0,
    }
    for p in ref.bands.values():
        assert p.is_file()


@requires_real_dji_data
def test_real_data_detection_is_idempotent(dji_copy: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    first = detect_and_write_band_groups(dji_copy)
    assert len(first["manifests"]) == 16
    manifest_mtimes = {
        p: Path(p).stat().st_mtime_ns for p in first["manifests"]
    }

    second = detect_and_write_band_groups(dji_copy)
    assert second["formed"] == []  # a recorded fact is not re-inferred
    for p, mtime in manifest_mtimes.items():
        assert Path(p).stat().st_mtime_ns == mtime  # never rewritten


@requires_real_dji_data
def test_real_data_stale_manifest_recovers_by_deleting_it(
    dji_copy: Path,
) -> None:
    """A stale manifest refuses to resolve, and deleting it lets the survivors re-form.

    With one sibling's file deleted, deleting the stem's .bandgroup lets the next detection pass
    re-form the surviving bands under that stem.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        BandGroupIncompleteError,
        detect_and_write_band_groups,
    )
    from tcip_mcp.pipelines.image_utils import resolve_image_path

    result = detect_and_write_band_groups(dji_copy)
    stem = result["formed"][0]["stem"]
    manifest_path = Path(result["manifests"][0])
    resolve_image_path(manifest_path).bands["Green"].unlink()

    with pytest.raises(BandGroupIncompleteError):
        resolve_image_path(manifest_path)

    manifest_path.unlink()
    redo = detect_and_write_band_groups(dji_copy)
    # The 3 surviving siblings still share a CaptureUUID, so they re-form under the same
    # canonical stem, now a 3-band group, since the deleted G file is gone for good.
    reformed = {g["stem"]: g for g in redo["formed"]}
    assert stem in reformed
    assert sorted(reformed[stem]["bands"]) == ["NIR", "Red", "RedEdge"]
    ref = resolve_image_path(manifest_path)
    assert sorted(ref.manifest.bands) == ["NIR", "Red", "RedEdge"]


def _expected_tolerances(images_dir: Path) -> dict[str, tuple[float, float]]:
    """Return each identity signal's gap between captures and spread within one capture.

    Per signal, the gap is the smallest separation between two distinct captures' files and the
    spread the largest within one capture, recomputed from each file's own tags: seconds by numpy
    over the sorted capture times, meters by pyproj's geodesic over every pair of files.
    """
    from datetime import datetime
    from itertools import combinations

    from pyproj import Geod

    from tcip_mcp.pipelines.data.band_groups import _read_xmp_tags

    tags = [
        _read_xmp_tags(p) for p in sorted(images_dir.glob("*.[tT][iI][fF]"))
    ]
    gids = np.array([t[STRATEGY.group_id_tag] for t in tags])
    times = np.array(
        [datetime.fromisoformat(t[TIME_TAG]).timestamp() for t in tags]
    )
    order = np.argsort(times)
    steps = np.diff(times[order])
    crossing = gids[order][1:] != gids[order][:-1]
    time_within = max(np.ptp(times[gids == g]) for g in set(gids))

    lat, lon = (
        np.array([float(t[tag]) for t in tags]) for tag in POSITION_TAGS
    )
    pairs = np.array(list(combinations(range(len(tags)), 2)))
    _, _, meters = Geod(ellps="WGS84").inv(
        lon[pairs[:, 0]], lat[pairs[:, 0]], lon[pairs[:, 1]], lat[pairs[:, 1]]
    )
    same = gids[pairs[:, 0]] == gids[pairs[:, 1]]
    return {
        TIME_TAG: (float(steps[crossing].min()), float(time_within)),
        ",".join(POSITION_TAGS): (
            float(meters[~same].min()),
            float(meters[same].max(initial=0.0)),
        ),
    }


def _recorded_tolerances(detected: dict) -> list[dict]:
    """Return the recorded identity tolerance of each manifest a detection pass answered."""
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest

    return [
        read_band_group_manifest(Path(p)).manifest.identity_tolerance
        for p in detected["manifests"]
    ]


def _assert_tolerances_agree(images_dir: Path, detected: dict) -> None:
    """Assert every capture's spread sits below its signal's gap, which every manifest records.

    The gap recorded in each manifest of the detection answer ``detected`` is the one
    :func:`_expected_tolerances` recomputes over ``images_dir``.
    """
    expected = _expected_tolerances(images_dir)
    for gap, within in expected.values():
        assert within < gap
    for recorded in _recorded_tolerances(detected):
        assert recorded.keys() == expected.keys()
        for key, (gap, _within) in expected.items():
            assert recorded[key]["value"] == pytest.approx(gap, rel=1e-4)


@requires_real_dji_data
def test_real_data_tolerance_agrees_with_the_samples_own_gap(
    dji_copy: Path,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    result = detect_and_write_band_groups(dji_copy)

    assert len(result["formed"]) == 16
    _assert_tolerances_agree(dji_copy, result)


def test_tolerance_agrees_with_a_synthetic_buckets_own_gap(
    tmp_path: Path,
) -> None:
    """Each synthetic capture forms, recording the identity gap the test recomputes.

    Three two-band captures are taken seconds apart along a row, each band's time and fix offset
    slightly from its capture's other band.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    captures = [
        ("17:06:28.931", 43.1969463, -90.0580036),
        ("17:06:31.402", 43.1969471, -90.0579662),
        ("17:06:33.887", 43.1969458, -90.0579285),
    ]
    for c, (clock, lat, lon) in enumerate(captures):
        for b, (band, wavelength) in enumerate((("Green", 560), ("NIR", 860))):
            _write_packet(
                d / f"cap{c}_{band}.tif",
                _xmp(
                    f"uuid-{c}",
                    band,
                    wavelength,
                    utc=f"2023-05-23T{clock}{b}4",
                    lat=lat + b * 3e-7,
                    lon=lon - b * 2e-7,
                ),
            )

    result = detect_and_write_band_groups(d)

    assert len(result["formed"]) == 3
    _assert_tolerances_agree(d, result)


def test_a_lone_capture_forms_recording_that_its_identity_had_no_reference(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    for band, wavelength in (("Green", 560), ("NIR", 860)):
        _write_packet(
            d / f"solo_{band}.tif",
            _xmp(
                "only-capture",
                band,
                wavelength,
                utc="2023-05-23T17:06:28.931664",
                lat=43.196946355,
                lon=-90.058003633,
            ),
        )

    result = detect_and_write_band_groups(d)

    assert [g["stem"] for g in result["formed"]] == ["solo"]
    (recorded,) = _recorded_tolerances(result)
    assert set(recorded) == {TIME_TAG, ",".join(POSITION_TAGS)}
    for entry in recorded.values():
        assert entry["value"] is None
        assert "no other capture group" in entry["source"]


# ── XMP packet reading ──────────────────────────────────────────────────────────────────

XMP_PACKET = (
    '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
    ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
    '  <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"\n'
    "   xmlns:Camera='http://pix4d.com/camera/1.0'\n"
    '   drone-dji:CaptureUUID="cap &amp; one"\n'
    "   drone-dji:UTCAtExposure='2023-05-23T17:06:28.931664'>\n"
    "   <Camera:BandName>Red &amp; Green</Camera:BandName>\n"
    "   <Camera:CentralWavelength>650</Camera:CentralWavelength>\n"
    "   <Camera:CentralWavelength>560</Camera:CentralWavelength>\n"
    "  </rdf:Description>\n"
    " </rdf:RDF>\n"
    "</x:xmpmeta>\n"
    "                                \n"
    '<?xpacket end="w"?>'
)
"""An XMP packet holding an entity in an attribute and in an element, a single-quoted
attribute, an element-form and an attribute-form tag, and a repeated element."""

XMP_TAGS = {
    "rdf:about": "",
    "drone-dji:CaptureUUID": "cap & one",
    "drone-dji:UTCAtExposure": "2023-05-23T17:06:28.931664",
    "Camera:BandName": "Red & Green",
    "Camera:CentralWavelength": "650",
}
"""The tags :data:`XMP_PACKET` decodes to, known in advance: entities decoded, the repeated
element's first value kept."""


def _regex_reference(text: str) -> dict[str, str]:
    """Return the tags two regular expressions find in ``text``, the first value of each kept.

    They are an attribute pattern over double-quoted values and an element pattern over
    childless elements.
    """
    tags: dict[str, str] = {}
    for pattern in (
        r'([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)="([^"]*)"',
        r"<([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)>([^<]*)</\1>",
    ):
        for name, value in re.findall(pattern, text):
            tags.setdefault(name, value)
    return tags


@pytest.mark.parametrize(
    "terminator",
    [b"", b"\0", b"\0\0\0"],
    ids=["bare", "one NUL", "three NULs"],
)
def test_an_xmp_packet_reads_as_the_xml_it_is(
    tmp_path: Path, terminator: bytes
) -> None:
    """An XMP packet's tags are the XML's own decoded attributes and element texts.

    The packet reads the same followed by the NUL bytes that terminate a TIFF byte tag.
    """
    from tcip_mcp.pipelines.data.band_groups import _read_xmp_tags

    path = _write_packet(
        tmp_path / "packet.tif", XMP_PACKET.encode("utf-8") + terminator
    )

    assert _read_xmp_tags(path) == XMP_TAGS
    reference = _regex_reference(XMP_PACKET)
    assert reference["drone-dji:CaptureUUID"] == "cap &amp; one"
    assert reference["Camera:BandName"] == "Red &amp; Green"
    assert "drone-dji:UTCAtExposure" not in reference


def test_an_xmp_packet_that_is_not_xml_offers_no_metadata(
    tmp_path: Path,
) -> None:
    """A packet naming a prefix it never declares is not XML, and offers no tags."""
    from tcip_mcp.pipelines.data.band_groups import _read_xmp_tags

    packet = XMP_PACKET.replace(" xmlns:drone-dji=", " xmlns:undeclared=")
    path = _write_packet(tmp_path / "packet.tif", packet.encode("utf-8"))

    assert _read_xmp_tags(path) is None


def test_a_prefix_declared_again_names_only_the_tags_in_its_scope(
    tmp_path: Path,
) -> None:
    """A namespace declared under a second prefix inside an element names that element's tags.

    A tag outside the inner declaration keeps the prefix in scope where it sits.
    """
    from tcip_mcp.pipelines.data.band_groups import _read_xmp_tags

    packet = (
        '<root xmlns:c="urn:test">\n'
        "  <c:Band>first</c:Band>\n"
        '  <child xmlns:d="urn:test"><d:Band>second</d:Band></child>\n'
        "</root>"
    )
    path = _write_packet(tmp_path / "packet.tif", packet.encode("utf-8"))

    assert _read_xmp_tags(path) == {"c:Band": "first", "d:Band": "second"}


def test_a_prefix_bound_again_names_only_its_namespace_where_it_is_bound(
    tmp_path: Path,
) -> None:
    """A prefix bound to another namespace on an element no longer names its former one there."""
    from tcip_mcp.pipelines.data.band_groups import _read_xmp_tags

    packet = (
        '<root xmlns:d="urn:a" xmlns:c="urn:a">\n'
        '  <child xmlns:c="urn:b">\n'
        "    <d:Band>old-uri</d:Band>\n"
        "    <c:Other>new-uri</c:Other>\n"
        "  </child>\n"
        "</root>"
    )
    path = _write_packet(tmp_path / "packet.tif", packet.encode())

    assert _read_xmp_tags(path) == {"d:Band": "old-uri", "c:Other": "new-uri"}


# ── duplicate-band-identity guard (synthetic, the real sample never hits this case) ──────


def test_duplicate_band_identity_is_refused_not_silently_overwritten(
    tmp_path: Path,
) -> None:
    """A capture holding one band twice is refused, never one file overwriting the other.

    The three files carry agreeing identity tags, and the refusal names the duplicated band and
    the group, writing no manifest.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    utc, lat, lon = "2023-05-23T17:06:28.931664", 43.196946355, -90.058003633
    for name, band, wavelength, fill in (
        ("cap_G1.tif", "Green", 560, 10),
        ("cap_G2.tif", "Green", 560, 20),  # same band, same capture
        ("cap_R.tif", "Red", 650, 30),
    ):
        _write_packet(
            d / name,
            _xmp("capA", band, wavelength, utc=utc, lat=lat, lon=lon),
            fill,
        )

    result = detect_and_write_band_groups(d)
    assert (
        result["formed"] == []
    )  # refused, never one file silently overwriting the other
    assert len(result["refused"]) == 1
    refusal = result["refused"][0]
    assert refusal["band"] == "Green"
    assert refusal["group_id"] == "capA"
    assert not list(d.glob("*.bandgroup"))


# ── identity-disagreement guard ─────────────────────────────


def test_two_unrelated_captures_sharing_a_group_id_are_refused_not_spliced(
    tmp_path: Path,
) -> None:
    """Two unrelated captures sharing a group id are refused, not spliced; a genuine one forms.

    Two disjoint-band-name captures (A_G/A_NIR, B_R/B_RE) sharing one group id, taken about two
    minutes and several hundred meters apart, draw one refusal naming all four files and no band;
    capture C, taken between them under its own id, forms the only group and manifest.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    same_gid = "collided-group-id"
    for name, gid, band, wavelength, utc, lat, lon in (
        # Capture A: one place/time.
        (
            "A_G",
            same_gid,
            "Green",
            560,
            "17:06:28.931664",
            43.196946355,
            -90.058003633,
        ),
        (
            "A_NIR",
            same_gid,
            "NIR",
            860,
            "17:06:28.931804",
            43.196946275,
            -90.058003629,
        ),
        # Capture B: a different place and time entirely, but the same (reused) group id.
        (
            "B_R",
            same_gid,
            "Red",
            650,
            "17:08:17.691590",
            43.197026713,
            -90.051619588,
        ),
        (
            "B_RE",
            same_gid,
            "RedEdge",
            730,
            "17:08:17.691682",
            43.197026632,
            -90.051619582,
        ),
        # Capture C: a genuine capture of its own id taken between the two.
        (
            "C_Green",
            "genuine-capture",
            "Green",
            560,
            "17:07:20.000000",
            43.19699,
            -90.0548,
        ),
        (
            "C_NIR",
            "genuine-capture",
            "NIR",
            860,
            "17:07:20.000000",
            43.19699,
            -90.0548,
        ),
    ):
        _write_packet(
            d / f"{name}.tif",
            _xmp(
                gid,
                band,
                wavelength,
                utc=f"2023-05-23T{utc}",
                lat=lat,
                lon=lon,
            ),
        )

    result = detect_and_write_band_groups(d)
    # refused, never spliced into one 4-band group
    assert [g["stem"] for g in result["formed"]] == ["C"]
    assert [p.name for p in d.glob("*.bandgroup")] == ["C.bandgroup"]
    assert len(result["refused"]) == 1
    refusal = result["refused"][0]
    assert refusal["group_id"] == same_gid
    # this is the identity-disagreement refusal, not the duplicate-band one
    assert refusal["band"] is None
    assert set(refusal["files"]) == {
        str(d / "A_G.tif"),
        str(d / "A_NIR.tif"),
        str(d / "B_R.tif"),
        str(d / "B_RE.tif"),
    }


def test_a_colliding_group_id_with_no_identity_signal_at_all_is_refused(
    tmp_path: Path,
) -> None:
    """A colliding group id with no identity signal at all is refused.

    Four files sharing a group id with every identity-check tag absent form no group and write no
    manifest; one refusal names the group id and no band.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    same_gid = "no-identity-tags-at-all"
    _write_packet(d / "A_G.tif", _xmp(same_gid, "Green", 560))
    _write_packet(d / "A_NIR.tif", _xmp(same_gid, "NIR", 860))
    _write_packet(d / "B_R.tif", _xmp(same_gid, "Red", 650))
    _write_packet(d / "B_RE.tif", _xmp(same_gid, "RedEdge", 730))

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []
    assert not list(d.glob("*.bandgroup"))
    assert len(result["refused"]) == 1
    refusal = result["refused"][0]
    assert refusal["group_id"] == same_gid
    assert refusal["band"] is None


def test_a_genuine_capture_with_identity_tags_still_forms_normally(
    tmp_path: Path,
) -> None:
    """4 files sharing a group id and agreeing timestamp/GPS (one capture) form a group."""
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    gid = "real-capture"
    bands = [
        ("Green", 560, "cap_G.tif"),
        ("NIR", 860, "cap_NIR.tif"),
        ("Red", 650, "cap_R.tif"),
        ("RedEdge", 730, "cap_RE.tif"),
    ]
    for band_name, wl, filename in bands:
        _write_packet(
            d / filename,
            _xmp(
                gid,
                band_name,
                wl,
                utc="2023-05-23T17:06:28.931664",
                lat=43.196946355,
                lon=-90.058003633,
            ),
        )

    result = detect_and_write_band_groups(d)
    assert result["refused"] == []
    assert len(result["formed"]) == 1
    assert sorted(result["formed"][0]["bands"]) == [
        "Green",
        "NIR",
        "Red",
        "RedEdge",
    ]


def test_no_metadata_and_no_manifest_leaves_files_independent(
    tmp_path: Path,
) -> None:
    """Files with no correlation metadata and no manifest stay independent, no group guessed.

    A file with no embedded correlation metadata and no explicit manifest forms no group; there
    is no filename-pattern fallback.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    arr = np.zeros((4, 4), dtype=np.uint16)
    tifffile.imwrite(str(d / "plain_a.tif"), arr)
    tifffile.imwrite(str(d / "plain_b.tif"), arr)

    result = detect_and_write_band_groups(d)
    assert result == {"formed": [], "refused": [], "manifests": []}


def test_a_lone_file_with_a_group_id_forms_no_group(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    _write_packet(d / "solo_G.tif", _xmp("capSolo", "Green", 560))

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []


# ── stem-collision guard (the ownership-aware inventory) ─────────────────────────────────


def test_a_group_whose_stem_a_standalone_file_holds_is_refused_through_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingest refuses a group whose stem a standalone file holds, naming the file.

    Through ingest_images: a group whose own canonical stem a raw file in the ingest holds is
    refused, naming that file; an unrelated group in that pass, with no such collision, forms.
    """
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "workspace"))
    from tcip_mcp.tools.ingest_tools import ingest_images

    src = tmp_path / "raw"
    src.mkdir()
    utc, lat, lon = "2023-05-23T17:06:28.931664", 43.196946355, -90.058003633
    later_utc, later_lat = "2023-05-23T17:06:30.931664", 43.196956355
    for capture, at, north in (("A", utc, lat), ("B", later_utc, later_lat)):
        for suffix, band, wavelength in (
            ("G", "Green", 560),
            ("R", "Red", 650),
        ):
            _write_packet(
                src / f"cap{capture}_{suffix}.tif",
                _xmp(
                    f"gid{capture}",
                    band,
                    wavelength,
                    utc=at,
                    lat=north,
                    lon=lon,
                ),
            )
    # A standalone raw file whose stem is exactly capA's own canonical stem.
    tifffile.imwrite(str(src / "capA.tif"), np.zeros((4, 4), dtype=np.uint16))

    project = tmp_path / "proj_bandcollide_case"
    manifest = ingest_images(project, source=str(src), actor=None)
    assert "error" not in manifest
    bg = manifest["band_groups"]
    assert len(bg["refused"]) == 1
    refusal = bg["refused"][0]
    assert refusal["group_id"] == "capA"
    assert "capA.tif" in refusal["reason"]
    assert refusal["bucket"] == UNDATED_BUCKET
    assert {g["stem"] for g in bg["formed"]} == {"capB"}
    bucket_dir = image_dir(project, UNDATED_BUCKET)
    assert not (bucket_dir / "capA.bandgroup").exists()
    assert (bucket_dir / "capB.bandgroup").exists()


def test_explicit_group_id_equal_to_one_of_its_own_members_stem_forms(
    tmp_path: Path,
) -> None:
    """An explicit group whose id equals one of its own members' stems forms.

    The collision check excludes the members the group claims.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )
    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = image_root(tmp_path)
    d.mkdir()
    for name in ("cap.tif", "cap_other.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    result = detect_and_write_band_groups(
        d,
        explicit_groups={"cap": {"Blue": "cap.tif", "Green": "cap_other.tif"}},
    )

    assert result["refused"] == []
    assert len(result["formed"]) == 1
    assert result["formed"][0]["stem"] == "cap"
    assert set(list_logical_images(d)) == {"cap"}


def test_two_explicit_group_ids_folding_to_one_key_form_one_and_refuse_the_other(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    d.mkdir()
    for name in ("a1.tif", "a2.tif", "b1.tif", "b2.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    result = detect_and_write_band_groups(
        d,
        explicit_groups={
            "Cap": {"Blue": "a1.tif", "Green": "a2.tif"},
            "cap": {"Blue": "b1.tif", "Green": "b2.tif"},
        },
    )

    assert len(result["formed"]) == 1
    assert len(result["refused"]) == 1
    formed_stem = result["formed"][0]["stem"]
    refused_group_id = result["refused"][0]["group_id"]
    assert {formed_stem, refused_group_id} == {"Cap", "cap"}


def test_a_group_that_claims_a_members_stem_lets_a_later_group_form_under_it(
    tmp_path: Path,
) -> None:
    """A group claiming a member's stem lets a later group in the pass form under that stem.

    A second group in one pass whose stem equals a member the first group claims (``y.tif``)
    forms: that member's raw identity is gone when the first manifest is written.
    """
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )
    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = image_root(tmp_path)
    d.mkdir()
    for name in ("y.tif", "other.tif", "y_band1.tif", "y_band2.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    result = detect_and_write_band_groups(
        d,
        explicit_groups={
            "capA": {"Blue": "y.tif", "Green": "other.tif"},
            "y": {"Blue": "y_band1.tif", "Green": "y_band2.tif"},
        },
    )

    assert result["refused"] == []
    assert {g["stem"] for g in result["formed"]} == {"capA", "y"}
    assert set(list_logical_images(d)) == {"capA", "y"}


# ── explicit-manifest strategy ──────────────────────────────────────────────────────────


def test_explicit_manifest_groups_a_sensor_with_no_embedded_metadata(
    tmp_path: Path,
) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
        read_band_group_manifest,
    )

    d = image_root(tmp_path)
    d.mkdir()
    for name in ("cap1_band1.tif", "cap1_band2.tif", "cap1_band3.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    explicit = {
        "cap1": {
            "Blue": "cap1_band1.tif",
            "Green": "cap1_band2.tif",
            "Red": "cap1_band3.tif",
        },
    }
    result = detect_and_write_band_groups(d, explicit_groups=explicit)
    assert len(result["formed"]) == 1
    g = result["formed"][0]
    assert g["stem"] == "cap1"
    assert sorted(g["bands"]) == ["Blue", "Green", "Red"]
    # No identity signal was checked and no wavelength stated on an explicit group, facts its
    # record states.
    (manifest,) = result["manifests"]
    ref = read_band_group_manifest(Path(manifest))
    assert ref.manifest.identity_tolerance == {}
    assert ref.manifest.central_wavelength_nm is None


@pytest.mark.parametrize(
    "named", ["./cap.tif", "sub/cap_G.tif", "sub/../cap.tif"]
)
def test_an_explicit_group_naming_a_file_through_a_directory_refuses(
    tmp_path: Path, named: str
) -> None:
    """An explicit group naming a file through a directory part refuses, recording nothing."""
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups,
    )

    d = image_root(tmp_path)
    (d / "sub").mkdir(parents=True)
    for name in ("cap.tif", "other.tif", "sub/cap_G.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    with pytest.raises(ValueError, match="pattern"):
        detect_and_write_band_groups(
            d, explicit_groups={"cap": {"G": named, "R": "other.tif"}}
        )
    assert not list(d.glob("*.bandgroup"))


def test_explicit_manifest_drops_a_group_missing_files(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import (
        groups_from_explicit_mapping,
    )

    d = image_root(tmp_path)
    d.mkdir()
    tifffile.imwrite(
        str(d / "only_one.tif"), np.zeros((4, 4), dtype=np.uint16)
    )

    groups = groups_from_explicit_mapping(
        d, {"capX": {"Blue": "only_one.tif", "Green": "missing.tif"}}
    )
    assert groups == []  # fewer than 2 resolvable bands


# ── manifest read/write round trip ─────────────────────────────────────────────────────


def test_manifest_round_trip(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest
    from tests._producer_fixtures import write_band_group

    d = image_root(tmp_path)
    d.mkdir()
    band_a = d / "a.tif"
    band_b = d / "b.tif"
    band_a.write_bytes(b"x")
    band_b.write_bytes(b"y")

    manifest_path = write_band_group(
        d,
        "cap42",
        {"Red": band_a, "Blue": band_b},
        central_wavelength_nm={"Red": 650.0, "Blue": 470.0},
    )
    assert manifest_path == d / "cap42.bandgroup"
    # Path.stem recovers the canonical stem directly: the whole reason .bandgroup was chosen
    # over a compound ".bandset.json"-style suffix.
    assert manifest_path.stem == "cap42"

    ref = read_band_group_manifest(manifest_path)
    assert ref.stem == "cap42"
    assert ref.manifest_path == manifest_path
    assert ref.bands == {"Red": band_a, "Blue": band_b}
    assert ref.manifest.central_wavelength_nm == {"Red": 650.0, "Blue": 470.0}


@pytest.mark.parametrize(
    "malform",
    [
        pytest.param(lambda doc: doc.pop("bands"), id="no bands"),
        pytest.param(lambda doc: doc.update(bands={}), id="empty bands"),
        pytest.param(
            lambda doc: doc.update(bands=list(doc["bands"].values())),
            id="bands not a mapping",
        ),
        pytest.param(
            lambda doc: doc.pop("central_wavelength_nm"),
            id="no central wavelength",
        ),
        pytest.param(
            lambda doc: doc["bands"].update(Green="sub/cap_G.tif"),
            id="a band in a subdirectory",
        ),
    ],
)
def test_a_malformed_manifest_refuses_and_is_no_identity(
    tmp_path: Path, malform: Callable[[dict], object]
) -> None:
    """A manifest its record's fields refuse raises ``ValueError`` and claims nothing.

    The manifest is the writer's, one field then removed or replaced; the scan lists no identity
    under its stem and each band file as its own.
    """
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest
    from tcip_mcp.pipelines.image_utils import (
        scan_identities,
        stem_collision_key,
    )
    from tcip_store import decode_value, encode_record
    from tests._producer_fixtures import write_band_group

    d = image_root(tmp_path)
    d.mkdir()
    bands = {"Green": d / "cap_G.tif", "Red": d / "cap_R.tif"}
    for band in bands.values():
        tifffile.imwrite(str(band), np.zeros((4, 4), dtype=np.uint16))
    manifest = write_band_group(
        d,
        "cap",
        bands,
        central_wavelength_nm={"Green": 560.0, "Red": 650.0},
    )
    document = decode_value(manifest.read_bytes())
    malform(document)
    manifest.write_bytes(encode_record(document))

    with pytest.raises(ValueError):
        read_band_group_manifest(manifest)
    identities = scan_identities(d)
    assert stem_collision_key("cap") not in identities
    for band in bands.values():
        assert identities[stem_collision_key(band.stem)] == [band]


# ── ingest_images ──────────────────────────────────────────────────────────────────────


@requires_real_dji_data
def test_ingest_images_groups_real_dji_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "workspace"))
    from tcip_mcp.tools.ingest_tools import ingest_images

    proj = tmp_path / "ms_valley_farm"
    manifest = ingest_images(proj, source=str(REAL_DJI_DIR), actor=None)
    assert "error" not in manifest
    assert manifest["copied"] == 64
    assert len(manifest["band_groups"]["formed"]) == 16
    assert manifest["band_groups"]["refused"] == []

    (bucket,) = {f["bucket"] for f in manifest["files"]}
    images_dir = image_dir(proj, bucket)
    assert len(list(images_dir.glob("*.bandgroup"))) == 16
    assert (
        len(list(images_dir.glob("*.TIF"))) == 64
    )  # originals untouched, still all present


def test_an_ingest_of_per_band_captures_forms_their_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingest of per-band captures forms their groups with nothing stated about them."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "workspace"))
    from tcip_mcp.tools.ingest_tools import ingest_images

    src = tmp_path / "raw"
    src.mkdir()
    for capture, utc in (("capA", "17:06:28.931664"), ("capB", "17:06:30.1")):
        for band, wavelength in (("Green", 560), ("Red", 650)):
            _write_packet(
                src / f"{capture}_{band}.tif",
                _xmp(
                    f"gid-{capture}",
                    band,
                    wavelength,
                    utc=f"2023-05-23T{utc}",
                    lat=43.196946355,
                    lon=-90.058003633 if capture == "capA" else -90.057990000,
                ),
            )

    manifest = ingest_images(tmp_path / "proj", source=str(src), actor=None)

    assert sorted(g["stem"] for g in manifest["band_groups"]["formed"]) == [
        "capA",
        "capB",
    ]
    assert len(manifest["band_groups"]["manifests"]) == 2


def test_a_manifest_recorded_mid_detection_is_kept_rather_than_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest recorded mid-detection is kept, not overwritten.

    A competing manifest is written as the pass's own manifest write begins: the recorded bytes
    stay as written, and the result lists no formed group and no manifest.
    """
    from tcip_mcp.pipelines.data import band_groups

    images = image_root(tmp_path)
    images.mkdir()
    (images / "a.tif").write_bytes(b"a")
    (images / "b.tif").write_bytes(b"b")
    (images / "c.tif").write_bytes(b"c")

    original = band_groups.write_band_group_manifest
    raced: list[tuple[Path, bytes]] = []

    def racing_writer(
        group: band_groups.BandGroupRef, *, expect: Version | None = None
    ) -> Path:
        if not raced:
            competing = band_groups.BandGroupRef(
                group.manifest_path,
                band_groups.BandGroupManifest(
                    bands={"Red": "b.tif", "Blue": "c.tif"},
                    source="explicit-manifest",
                    identity_tolerance={},
                    central_wavelength_nm=None,
                ),
            )
            raced.append(
                (original(competing), competing.manifest_path.read_bytes())
            )
        return original(group, expect=expect)

    monkeypatch.setattr(
        band_groups, "write_band_group_manifest", racing_writer
    )

    result = band_groups.detect_and_write_band_groups(
        images,
        explicit_groups={"cap": {"Green": "a.tif", "Red": "b.tif"}},
    )

    ((manifest_path, recorded),) = raced
    assert manifest_path.stem == "cap"
    assert manifest_path.read_bytes() == recorded
    assert result["formed"] == []
    assert result["manifests"] == []

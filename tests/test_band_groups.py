"""Sensor-agnostic band-group correlation (``pipelines.data.band_groups``).

Real DJI sample data (16 captures x G/NIR/R/RE single-band GeoTIFFs, embedded XMP) is used
directly for the detection tests, grounded against real data, never mutated in place, always
copied into ``tmp_path`` first. The synthetic fixtures below build minimal TIFFs with the same
real XMP tag shape (attribute and nested-element forms, both present in the real files) to
exercise the strategies' edge cases the 16-capture sample doesn't happen to hit (a duplicate band
identity, a sensor with no embedded metadata at all).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

REAL_DJI_DIR = Path(
    r"C:\Users\breeder\tcip-projects\currant_multispectral_valley-farm-2023\images\2023-05-23"
)
requires_real_dji_data = pytest.mark.skipif(
    not REAL_DJI_DIR.is_dir(),
    reason="real DJI multispectral sample project not present on this machine",
)


def _xmp(capture_uuid: str, band_name: str, wavelength: float) -> bytes:
    """A minimal XMP packet in the real DJI shape: ``drone-dji:CaptureUUID`` as an XML
    attribute, ``Camera:BandName``/``Camera:CentralWavelength`` as nested elements: both shapes
    are present in the real files, so the reader has to understand both."""
    xmp = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
        'xmlns:Camera="http://pix4d.com/camera/1.0"\n'
        f'   drone-dji:CaptureUUID="{capture_uuid}">\n'
        f"   <Camera:BandName>{band_name}</Camera:BandName>\n"
        f"   <Camera:CentralWavelength>{wavelength}</Camera:CentralWavelength>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return xmp.encode("utf-8")


def _write_band_file(path: Path, capture_uuid: str, band_name: str, wavelength: float,
                     fill: int = 100, shape=(4, 4)) -> None:
    arr = np.full(shape, fill, dtype=np.uint16)
    tifffile.imwrite(
        str(path), arr,
        extratags=[(700, "B", 0, _xmp(capture_uuid, band_name, wavelength), True)],
    )


def _xmp_with_identity(capture_uuid: str, band_name: str, wavelength: float, *,
                      utc: str, lat: float, lon: float) -> bytes:
    """As :func:`_xmp`, plus the secondary identity tags (timestamp + GPS fix) real DJI files
    carry, needed to exercise the identity-disagreement guard, which the bare :func:`_xmp` shape
    (no identity tags at all) can never trigger (every check is skipped when its tag is absent)."""
    xmp = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about="" xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
        'xmlns:Camera="http://pix4d.com/camera/1.0"\n'
        f'   drone-dji:CaptureUUID="{capture_uuid}"\n'
        f'   drone-dji:UTCAtExposure="{utc}"\n'
        f'   drone-dji:GpsLatitude="{lat}"\n'
        f'   drone-dji:GpsLongitude="{lon}">\n'
        f"   <Camera:BandName>{band_name}</Camera:BandName>\n"
        f"   <Camera:CentralWavelength>{wavelength}</Camera:CentralWavelength>\n"
        "  </rdf:Description>\n"
        " </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return xmp.encode("utf-8")


def _write_band_file_with_identity(path: Path, capture_uuid: str, band_name: str, wavelength: float,
                                   *, utc: str, lat: float, lon: float,
                                   fill: int = 100, shape=(4, 4)) -> None:
    arr = np.full(shape, fill, dtype=np.uint16)
    tifffile.imwrite(
        str(path), arr,
        extratags=[(700, "B", 0,
                   _xmp_with_identity(capture_uuid, band_name, wavelength, utc=utc, lat=lat, lon=lon),
                   True)],
    )


@pytest.fixture
def dji_copy(tmp_path: Path) -> Path:
    """The real 64-file sample, copied into an isolated tmp dir (never mutated in place)."""
    dest = tmp_path / "images"
    dest.mkdir()
    for p in REAL_DJI_DIR.iterdir():
        shutil.copy2(p, dest / p.name)
    return dest


# ── embedded-metadata detection, against the real sample ──────────────────────────────────


@requires_real_dji_data
def test_detects_all_16_captures_from_real_dji_data(dji_copy):
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    result = detect_and_write_band_groups(dji_copy)
    assert len(result["formed"]) == 16
    assert result["refused"] == []
    for g in result["formed"]:
        assert sorted(g["bands"]) == ["Green", "NIR", "Red", "RedEdge"]
        assert g["source"] == "embedded-metadata"
    # No physical rename/move: every original file is still exactly where it was.
    assert len(list(dji_copy.glob("*.TIF"))) == 64


@requires_real_dji_data
def test_real_data_manifest_records_true_wavelengths(dji_copy):
    from tcip_mcp.pipelines.data.band_groups import (
        detect_and_write_band_groups, read_band_group_manifest,
    )

    result = detect_and_write_band_groups(dji_copy)
    ref = read_band_group_manifest(Path(result["manifests"][0]))
    assert ref.central_wavelength_nm == {
        "Green": 560.0, "NIR": 860.0, "Red": 650.0, "RedEdge": 730.0,
    }
    for p in ref.bands.values():
        assert p.is_file()


@requires_real_dji_data
def test_real_data_detection_is_idempotent(dji_copy):
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    first = detect_and_write_band_groups(dji_copy)
    assert len(first["manifests"]) == 16
    manifest_mtimes = {p: Path(p).stat().st_mtime_ns for p in first["manifests"]}

    second = detect_and_write_band_groups(dji_copy)
    assert second["formed"] == []  # a recorded fact is not re-inferred
    for p, mtime in manifest_mtimes.items():
        assert Path(p).stat().st_mtime_ns == mtime  # never rewritten


@requires_real_dji_data
def test_real_data_stale_manifest_recovers_by_deleting_it(dji_copy):
    """Deleting a stem's .bandgroup is the one recovery path for a manifest whose sibling was
    since deleted; the next detection pass sees the survivors as ungrouped again."""
    from tcip_mcp.pipelines.data.band_groups import BandGroupIncomplete, detect_and_write_band_groups
    from tcip_mcp.pipelines.image_utils import resolve_image_source

    result = detect_and_write_band_groups(dji_copy)
    stem = result["formed"][0]["stem"]
    manifest_path = dji_copy / f"{stem}.bandgroup"
    (dji_copy / f"{stem}_G.TIF").unlink()

    with pytest.raises(BandGroupIncomplete):
        resolve_image_source(dji_copy, stem)

    manifest_path.unlink()
    redo = detect_and_write_band_groups(dji_copy)
    # The 3 surviving siblings still share a CaptureUUID, so they re-form under the same
    # canonical stem, now a 3-band group, since the deleted G file is gone for good.
    reformed = {g["stem"]: g for g in redo["formed"]}
    assert stem in reformed
    assert sorted(reformed[stem]["bands"]) == ["NIR", "Red", "RedEdge"]
    ref = resolve_image_source(dji_copy, stem)
    assert sorted(ref.bands) == ["NIR", "Red", "RedEdge"]


# ── duplicate-band-identity guard (synthetic, the real sample never hits this case) ──────


def test_duplicate_band_identity_is_refused_not_silently_overwritten(tmp_path):
    """Uses agreeing identity tags on all three files so this isolates the duplicate-band-name
    guard specifically; without them, the newer "no secondary signal at all" guard would refuse
    the group first, for a different reason than the one this test pins."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    utc, lat, lon = "2023-05-23T17:06:28.931664", 43.196946355, -90.058003633
    _write_band_file_with_identity(
        d / "cap_G1.tif", "capA", "Green", 560, utc=utc, lat=lat, lon=lon, fill=10,
    )
    _write_band_file_with_identity(
        d / "cap_G2.tif", "capA", "Green", 560, utc=utc, lat=lat, lon=lon, fill=20,
    )  # same band, same capture
    _write_band_file_with_identity(
        d / "cap_R.tif", "capA", "Red", 650, utc=utc, lat=lat, lon=lon, fill=30,
    )

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []  # refused, never one file silently overwriting the other
    assert len(result["refused"]) == 1
    refusal = result["refused"][0]
    assert refusal["band"] == "Green"
    assert refusal["group_id"] == "capA"
    assert not list(d.glob("*.bandgroup"))


# ── identity-disagreement guard ─────────────────────────────


def test_two_unrelated_captures_sharing_a_group_id_are_refused_not_spliced(tmp_path):
    """Two disjoint-band-name captures (A_G/A_NIR, B_R/B_RE) sharing one fabricated group id,
    with disagreeing timestamp+GPS, must be refused, not silently merged into one confident,
    wrong 4-band composite."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    same_gid = "collided-group-id"
    # Capture A: one place/time.
    _write_band_file_with_identity(
        d / "A_G.tif", same_gid, "Green", 560,
        utc="2023-05-23T17:06:28.931664", lat=43.196946355, lon=-90.058003633,
    )
    _write_band_file_with_identity(
        d / "A_NIR.tif", same_gid, "NIR", 860,
        utc="2023-05-23T17:06:28.931804", lat=43.196946275, lon=-90.058003629,
    )
    # Capture B: a different place and time entirely, but the same (colliding/reused) group id.
    _write_band_file_with_identity(
        d / "B_R.tif", same_gid, "Red", 650,
        utc="2023-05-23T17:08:17.691590", lat=43.197026713, lon=-90.051619588,
    )
    _write_band_file_with_identity(
        d / "B_RE.tif", same_gid, "RedEdge", 730,
        utc="2023-05-23T17:08:17.691682", lat=43.197026632, lon=-90.051619582,
    )

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []  # refused, never spliced into one 4-band group
    assert not list(d.glob("*.bandgroup"))
    assert len(result["refused"]) == 1
    refusal = result["refused"][0]
    assert refusal["group_id"] == same_gid
    assert refusal["band"] is None  # this is the identity-disagreement refusal, not the duplicate-band one
    assert set(refusal["files"]) == {
        str(d / "A_G.tif"), str(d / "A_NIR.tif"), str(d / "B_R.tif"), str(d / "B_RE.tif"),
    }


def test_a_colliding_group_id_with_no_identity_signal_at_all_is_refused(tmp_path):
    """A group id match with every identity-check tag absent (not disagreeing, never recorded)
    must refuse too, not silently accept: an unconfirmed match is exactly as unproven as a
    disagreeing one."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    same_gid = "no-identity-tags-at-all"
    _write_band_file(d / "A_G.tif", same_gid, "Green", 560)
    _write_band_file(d / "A_NIR.tif", same_gid, "NIR", 860)
    _write_band_file(d / "B_R.tif", same_gid, "Red", 650)
    _write_band_file(d / "B_RE.tif", same_gid, "RedEdge", 730)

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []
    assert not list(d.glob("*.bandgroup"))
    assert len(result["refused"]) == 1
    refusal = result["refused"][0]
    assert refusal["group_id"] == same_gid
    assert refusal["band"] is None


def test_a_genuine_capture_with_identity_tags_still_forms_normally(tmp_path):
    """A rail must admit valid work: 4 files sharing a group id and agreeing timestamp/GPS (a
    real single capture) must still form, not be caught by the new guard."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    gid = "real-capture"
    bands = [("Green", 560, "cap_G.tif"), ("NIR", 860, "cap_NIR.tif"),
             ("Red", 650, "cap_R.tif"), ("RedEdge", 730, "cap_RE.tif")]
    for band_name, wl, filename in bands:
        _write_band_file_with_identity(
            d / filename, gid, band_name, wl,
            # Sub-millisecond/sub-meter jitter across the 4 siblings, matching the real sample.
            utc="2023-05-23T17:06:28.931664", lat=43.196946355, lon=-90.058003633,
        )

    result = detect_and_write_band_groups(d)
    assert result["refused"] == []
    assert len(result["formed"]) == 1
    assert sorted(result["formed"][0]["bands"]) == ["Green", "NIR", "Red", "RedEdge"]


def test_a_group_whose_canonical_stem_is_reserved_is_not_written(tmp_path):
    """A group whose siblings' common prefix names a bucket's own provenance stamp is not written
    as a manifest: minting a logical image under that stem would make its label indistinguishable
    from the stamp everywhere a prediction bucket is walked. The members stay standalone files."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    gid = "reserved-stem-capture"
    for band_name, wl, filename in (
        ("Green", 560, "operating_point_G.tif"), ("NIR", 860, "operating_point_NIR.tif"),
    ):
        _write_band_file_with_identity(
            d / filename, gid, band_name, wl,
            utc="2023-05-23T17:06:28.931664", lat=43.196946355, lon=-90.058003633,
        )

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []
    assert not list(d.glob("*.bandgroup"))
    assert len(result["reserved_name_skips"]) == 1
    assert result["reserved_name_skips"][0]["stem"] == "operating_point"
    assert sorted(d.iterdir()) == sorted(
        [d / "operating_point_G.tif", d / "operating_point_NIR.tif"]
    )


def test_no_metadata_and_no_manifest_leaves_files_independent(tmp_path):
    """Refuse, don't guess: a file with no embedded correlation metadata and no explicit
    manifest stays exactly as independent as it is today; no filename-pattern fallback."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    arr = np.zeros((4, 4), dtype=np.uint16)
    tifffile.imwrite(str(d / "plain_a.tif"), arr)
    tifffile.imwrite(str(d / "plain_b.tif"), arr)

    result = detect_and_write_band_groups(d)
    assert result == {"formed": [], "refused": [], "manifests": [], "reserved_name_skips": []}


def test_a_lone_file_with_a_group_id_forms_no_group(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    _write_band_file(d / "solo_G.tif", "capSolo", "Green", 560)

    result = detect_and_write_band_groups(d)
    assert result["formed"] == []


# ── stem-collision guard (the ownership-aware inventory) ─────────────────────────────────


def test_a_group_whose_stem_a_standalone_file_holds_is_refused_through_ingest(tmp_path, monkeypatch):
    """Through ingest_images(detect_band_groups=True): a group whose own canonical stem a raw
    file already placed in the bucket holds is refused, naming that file; an unrelated group in
    the same pass, with no such collision, still forms."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "workspace"))
    from tcip_mcp.tools.ingest_tools import ingest_images

    src = tmp_path / "raw"
    src.mkdir()
    utc, lat, lon = "2023-05-23T17:06:28.931664", 43.196946355, -90.058003633
    _write_band_file_with_identity(
        src / "capA_G.tif", "gidA", "Green", 560, utc=utc, lat=lat, lon=lon,
    )
    _write_band_file_with_identity(
        src / "capA_R.tif", "gidA", "Red", 650, utc=utc, lat=lat, lon=lon,
    )
    # A standalone raw file whose stem is exactly capA's own canonical stem.
    tifffile.imwrite(str(src / "capA.tif"), np.zeros((4, 4), dtype=np.uint16))
    _write_band_file_with_identity(
        src / "capB_G.tif", "gidB", "Green", 560, utc=utc, lat=lat, lon=lon,
    )
    _write_band_file_with_identity(
        src / "capB_R.tif", "gidB", "Red", 650, utc=utc, lat=lat, lon=lon,
    )

    manifest = ingest_images(
        source=str(src), name="proj_bandcollide_case", site="north orchard",
        date_from="none", detect_band_groups=True,
    )
    assert "error" not in manifest
    bg = manifest["band_groups"]
    assert len(bg["refused"]) == 1
    refusal = bg["refused"][0]
    assert refusal["group_id"] == "capA"
    assert "capA.tif" in refusal["reason"]
    assert refusal["bucket"] == "undated"
    assert {g["stem"] for g in bg["formed"]} == {"capB"}
    bucket_dir = Path(manifest["project_path"]) / "images" / "undated"
    assert not (bucket_dir / "capA.bandgroup").exists()
    assert (bucket_dir / "capB.bandgroup").exists()


def test_explicit_group_id_equal_to_one_of_its_own_members_stem_forms(tmp_path):
    """A rail must admit valid work: the ownership-aware inventory excludes a group's own
    about-to-be-claimed members from the collision check, so a group_id matching one of them
    forms cleanly rather than refusing against its own file."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups
    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    for name in ("cap.tif", "cap_other.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    result = detect_and_write_band_groups(
        d, explicit_groups={"cap": {"Blue": "cap.tif", "Green": "cap_other.tif"}},
    )

    assert result["refused"] == []
    assert len(result["formed"]) == 1
    assert result["formed"][0]["stem"] == "cap"
    assert set(list_logical_images(d)) == {"cap"}


def test_two_explicit_group_ids_folding_to_one_key_form_one_and_refuse_the_other(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    for name in ("a1.tif", "a2.tif", "b1.tif", "b2.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    result = detect_and_write_band_groups(
        d, explicit_groups={
            "Cap": {"Blue": "a1.tif", "Green": "a2.tif"},
            "cap": {"Blue": "b1.tif", "Green": "b2.tif"},
        },
    )

    assert len(result["formed"]) == 1
    assert len(result["refused"]) == 1
    formed_stem = result["formed"][0]["stem"]
    refused_group_id = result["refused"][0]["group_id"]
    assert {formed_stem, refused_group_id} == {"Cap", "cap"}


def test_a_group_that_claims_a_members_stem_lets_a_later_group_form_under_it(tmp_path):
    """A rail must admit valid work: the first group's own claimed member (``y.tif``) leaves its
    raw identity behind once the manifest is written, so a second group in the same pass whose
    own stem equals that member's stem forms cleanly rather than refusing against a member the
    first group already claims."""
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups
    from tcip_mcp.pipelines.image_utils import list_logical_images

    d = tmp_path / "images"
    d.mkdir()
    for name in ("y.tif", "other.tif", "y_band1.tif", "y_band2.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    result = detect_and_write_band_groups(
        d, explicit_groups={
            "capA": {"Blue": "y.tif", "Green": "other.tif"},
            "y": {"Blue": "y_band1.tif", "Green": "y_band2.tif"},
        },
    )

    assert result["refused"] == []
    assert {g["stem"] for g in result["formed"]} == {"capA", "y"}
    assert set(list_logical_images(d)) == {"capA", "y"}


# ── explicit-manifest strategy ──────────────────────────────────────────────────────────


def test_explicit_manifest_groups_a_sensor_with_no_embedded_metadata(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import detect_and_write_band_groups

    d = tmp_path / "images"
    d.mkdir()
    for name in ("cap1_band1.tif", "cap1_band2.tif", "cap1_band3.tif"):
        tifffile.imwrite(str(d / name), np.zeros((4, 4), dtype=np.uint16))

    explicit = {
        "cap1": {"Blue": "cap1_band1.tif", "Green": "cap1_band2.tif", "Red": "cap1_band3.tif"},
    }
    result = detect_and_write_band_groups(d, explicit_groups=explicit)
    assert len(result["formed"]) == 1
    g = result["formed"][0]
    assert g["stem"] == "cap1"
    assert sorted(g["bands"]) == ["Blue", "Green", "Red"]
    assert (d / "cap1.bandgroup").is_file()


def test_explicit_manifest_drops_a_group_missing_files(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import groups_from_explicit_mapping

    d = tmp_path / "images"
    d.mkdir()
    tifffile.imwrite(str(d / "only_one.tif"), np.zeros((4, 4), dtype=np.uint16))

    groups, used = groups_from_explicit_mapping(
        d, {"capX": {"Blue": "only_one.tif", "Green": "missing.tif"}}
    )
    assert groups == []  # fewer than 2 resolvable bands
    assert used == set()


# ── manifest read/write round trip ─────────────────────────────────────────────────────


def test_manifest_round_trip(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest, write_band_group_manifest

    d = tmp_path / "images"
    d.mkdir()
    band_a = d / "a.tif"
    band_b = d / "b.tif"
    band_a.write_bytes(b"x")
    band_b.write_bytes(b"y")

    manifest_path = write_band_group_manifest(
        d, "cap42", {"Red": band_a, "Blue": band_b},
        central_wavelength_nm={"Red": 650.0, "Blue": 470.0}, source="explicit-manifest",
    )
    assert manifest_path == d / "cap42.bandgroup"
    # Path.stem recovers the canonical stem directly: the whole reason .bandgroup was chosen
    # over a compound ".bandset.json"-style suffix.
    assert manifest_path.stem == "cap42"

    ref = read_band_group_manifest(manifest_path)
    assert ref.stem == "cap42"
    assert ref.manifest_path == manifest_path
    assert ref.bands == {"Red": band_a, "Blue": band_b}
    assert ref.central_wavelength_nm == {"Red": 650.0, "Blue": 470.0}


def test_manifest_with_missing_bands_key_is_rejected(tmp_path):
    from tcip_mcp.pipelines.data.band_groups import read_band_group_manifest

    bad = tmp_path / "bad.bandgroup"
    bad.write_text('{"source": "embedded-metadata"}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_band_group_manifest(bad)


# ── ingest_images(detect_band_groups=True) ─────────────────────────────────────────────


@requires_real_dji_data
def test_ingest_images_groups_real_dji_bucket(tmp_path, monkeypatch):
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "workspace"))
    from tcip_mcp.tools.ingest_tools import ingest_images

    manifest = ingest_images(
        source=str(REAL_DJI_DIR), name="ms_valley_farm", site="north orchard",
        date_from="2023-05-23", detect_band_groups=True,
    )
    assert "error" not in manifest
    assert manifest["copied"] == 64
    assert len(manifest["band_groups"]["formed"]) == 16
    assert manifest["band_groups"]["refused"] == []

    proj = Path(manifest["project_path"])
    images_dir = proj / "images" / "2023-05-23"
    assert len(list(images_dir.glob("*.bandgroup"))) == 16
    assert len(list(images_dir.glob("*.TIF"))) == 64  # originals untouched, still all present


def test_ingest_images_default_skips_band_group_detection(tmp_path, monkeypatch):
    """A rail must admit valid work: a project with no multi-band capture pays nothing, and
    detect_band_groups defaults False so an ordinary ingest is entirely unaffected."""
    monkeypatch.setenv("TCIP_WORKSPACE", str(tmp_path / "workspace"))
    from PIL import Image

    from tcip_mcp.tools.ingest_tools import ingest_images

    src = tmp_path / "raw"
    src.mkdir()
    Image.new("RGB", (16, 16)).save(src / "a.jpg")

    manifest = ingest_images(source=str(src), name="plain_proj_default", site="north orchard")
    assert manifest["band_groups"] == {
        "formed": [], "refused": [], "manifests": [], "reserved_name_skips": [],
    }


def test_a_manifest_recorded_mid_detection_is_kept_rather_than_overwritten(tmp_path, monkeypatch):
    """A recorded band group is a fact, and a detection pass that reached the same stem a moment
    later must not replace it with its own inference.

    The window is between deciding a stem has no manifest and writing one. Here a competing
    manifest lands inside that window, so the write has to be conditional on the stem still being
    unrecorded: it conflicts, the pass leaves the recorded manifest exactly as it found it, and
    reports the stem as neither formed nor written.
    """
    from tcip_mcp.pipelines.data import band_groups

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.tif").write_bytes(b"a")
    (images / "b.tif").write_bytes(b"b")
    (images / "c.tif").write_bytes(b"c")
    recorded = b'{"bands": {"Red": "b.tif", "Blue": "c.tif"}, "source": "explicit-manifest"}\n'

    manifest_path = band_groups.band_group_manifest_path(images, "cap")
    original = band_groups.write_band_group_manifest
    raced: list[str] = []

    def racing_writer(images_dir, stem, bands, **kwargs):
        if not raced:
            raced.append(stem)
            manifest_path.write_bytes(recorded)
        return original(images_dir, stem, bands, **kwargs)

    monkeypatch.setattr(band_groups, "write_band_group_manifest", racing_writer)

    result = band_groups.detect_and_write_band_groups(
        images, explicit_groups={"cap": {"Green": "a.tif", "Red": "b.tif"}},
    )

    assert raced == ["cap"]
    assert manifest_path.read_bytes() == recorded
    assert result["formed"] == []
    assert result["manifests"] == []

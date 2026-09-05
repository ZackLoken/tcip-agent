"""Sensor-agnostic band-group correlation: sibling single-band raster files that are really one
logical multi-band capture (some multispectral drone sensors write one file per band instead of
one multi-band file per image), and the ``.bandgroup`` manifest that records a found group.

A group is recorded, never physically stacked to disk, the originals keep their real names and
locations; the manifest just names which sibling files belong together and in what band order.

Detection is a capability with pluggable strategies, tried in order, never a closed switch on
sensor name:

1. Embedded-metadata grouping, a small declarative table of (group-id tag, band-id tag)
   pairs to look for in each file's XMP. A new sensor that exposes an exact-match, shared
   group-id-per-capture tag (most multi-file multispectral rigs) is a new table row, not new code.
   This does not generalize to a sensor needing tolerance/clustering correlation as its primary
   grouping key (no exact-match group id at all, only GPS proximity or skewed per-file timestamps
   to cluster on), that would need new matching code, not a table row. A shared group-id value is
   still only one signal, though: each strategy also names independently-recorded secondary
   identity tags (a timestamp, a GPS fix) that must be present and agree within a tight tolerance
   across every file claiming that group id, so a colliding/reused group-id string can never splice
   two unrelated captures into one, disagreement refuses the group, and so does a candidate where
   every declared secondary signal is simply absent (a shared group id with zero independent
   confirmation is exactly as unproven as one with a disagreeing confirmation).
2. Explicit manifest, a caller-supplied ``{group_id: {band_name: filename}}`` mapping, for a
   sensor with no embedded correlation metadata at all.
3. Refuse, don't guess: no embedded match and no explicit mapping leaves every file exactly
   as independent as it is today. No filename-pattern fallback: a guessed grouping that happens to
   be wrong would silently corrupt every downstream annotation/measurement.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tcip_store
from tcip_store import (
    RECORD_JSON,
    Key,
    StoreDescriptor,
    Version,
    VersionConflict,
    check_schema_version,
    get_descriptor,
    register_store,
)
from tcip_store.file_backend import RootedFileLocator

MANIFEST_EXT = ".bandgroup"

BAND_GROUP_MANIFEST_STORE = "band_group_manifest"
_MANIFEST_FILE = RootedFileLocator(suffix=MANIFEST_EXT)
"""A manifest sits beside the sibling files it names, so its scope is that image directory."""

register_store(
    StoreDescriptor(
        name=BAND_GROUP_MANIFEST_STORE,
        kind="blob",
        key_fields=("stem",),
        frozen=True,
        locator=_MANIFEST_FILE,
    )
)


def band_group_manifest_key(images_dir: str | Path, stem: str) -> Key:
    """The manifest recording which sibling files form the capture ``stem``.

    Scoped to the image directory rather than a dataset root: a group is a fact about files
    that sit beside each other, and detection runs against a directory that need not be
    inside a dataset at all.

    A blob because a ``.bandgroup`` file is itself an enumerated logical image
    (``image_utils``), so it has to sit in the image directory the enumerators walk. The
    payload is encoded through the canonical ``RECORD_JSON`` codec.
    """
    return Key(BAND_GROUP_MANIFEST_STORE, str(images_dir), (stem,))


def band_group_manifest_path(images_dir: str | Path, stem: str) -> Path:
    """Where ``stem``'s manifest lives, for a reader holding the directory and the stem."""
    relative = _MANIFEST_FILE.relative_path(str(images_dir), (stem,))
    return Path(images_dir, *relative.parts)


# Extensions an embedded-metadata scan bothers reading, the DJI-shaped rigs this generalizes to
# write one XMP-bearing TIFF per band. A container format with no embedded correlation metadata
# (NPY/NPZ) is only ever grouped via the explicit-manifest strategy.
_METADATA_BEARING_EXTS = (".tif", ".tiff")

_ATTR_RE = re.compile(r'([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)="([^"]*)"')
_ELEM_RE = re.compile(r"<([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)>([^<]*)</\1>")


@dataclass(frozen=True)
class _MetadataStrategy:
    """One (group-id tag, band-id tag) declarative row, plus the secondary identity signal(s) that
    must independently agree (within a tight tolerance) before files sharing that group-id value are
    trusted to be the same physical capture.

    A shared group-id string is necessary but not sufficient: it is one value an upstream tool wrote
    (or a corrupt/reused one could collide), never itself proof two files came off the same shutter
    event. ``identity_checks`` names an ``(tag, kind, tolerance)`` triple per signal, ``kind`` is
    ``"timestamp"`` (ISO-8601, compared in seconds) or ``"degrees"`` (a plain float, e.g. GPS
    lat/lon), read from each candidate file's own XMP independently of the group-id tag itself. A
    tag absent from any one file in the candidate group is skipped for that check (unusable, not a
    disagreement); a tag present on every file but disagreeing beyond its tolerance refuses the whole
    group, and so does a candidate group where every declared check ends up skipped this way (no
    secondary signal was ever actually cross-checked), since an unconfirmed group-id match is no
    safer than a disagreeing one.
    """

    group_id_tag: str
    band_id_tag: str
    wavelength_tag: str | None = None
    identity_checks: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)


# DJI's multispectral schema is the first instance of the exact-match-shared-group-id class this
# generalizes to, not the mechanism itself, a new sensor of the same class is a new row here.
_EMBEDDED_METADATA_STRATEGIES: list[_MetadataStrategy] = [
    _MetadataStrategy(
        "drone-dji:CaptureUUID", "Camera:BandName", "Camera:CentralWavelength",
        identity_checks=(
            # Verified against the real 16-capture DJI sample: same-capture siblings are exposed
            # within ~1ms of each other and agree on GPS to sub-meter precision (max observed
            # intra-capture spread ~0.0007s / ~3e-7 degrees), while the closest two different
            # captures in that flight are ~109s apart. These tolerances are plain, documented
            # platform defaults (same shape as derivations.py's jitter_px), generous relative to the real
            # intra-capture jitter measured, tight relative to any realistic inter-capture gap, not
            # validated against every DJI multispectral rig or flight speed.
            ("drone-dji:UTCAtExposure", "timestamp", 1.0),
            ("drone-dji:GpsLatitude", "degrees", 0.0001),
            ("drone-dji:GpsLongitude", "degrees", 0.0001),
        ),
    ),
]


def _identity_disagreement(
    paths: list[Path], tags_by_path: dict[Path, dict[str, str]],
    checks: tuple[tuple[str, str, float], ...],
) -> tuple[str | None, bool]:
    """The first identity-check tag that disagrees beyond its tolerance across ``paths`` (or
    ``None``), paired with whether any signal was actually checkable at all. A shared group id with
    zero independently-verifiable secondary signal is exactly as unproven as one with a disagreeing
    signal, a caller must refuse both, not silently accept a candidate no signal ever confirmed."""
    any_checked = False
    for tag, kind, tolerance in checks:
        raw_values = [tags_by_path[p].get(tag) for p in paths]
        if any(v is None for v in raw_values):
            continue  # this signal isn't recorded for this candidate group; nothing to cross-check
        values = [v for v in raw_values if v is not None]  # none are, by the check just above
        try:
            if kind == "timestamp":
                parsed = [datetime.fromisoformat(v).timestamp() for v in values]
            else:
                parsed = [float(v) for v in values]
        except ValueError:
            continue  # an unparsable value can't be used as a signal; don't refuse on a parse issue
        any_checked = True
        if max(parsed) - min(parsed) > tolerance:
            return tag, any_checked
    return None, any_checked


@dataclass(frozen=True)
class BandGroupRef:
    """One logical multi-band image, virtually assembled from sibling single-band files.

    ``manifest_path`` is the ``.bandgroup`` file's own path, a first-class field (not re-derived
    by every consumer) since ``serve_image`` and the dataset gallery route both need it directly.
    ``bands`` is ``{band_name: file_path}`` in the manifest's declared order (the order pixels are
    stacked into ``[H, W, C]``).
    """

    stem: str
    manifest_path: Path
    bands: dict[str, Path]
    central_wavelength_nm: dict[str, float] | None = None


class BandGroupIncomplete(FileNotFoundError):
    """A ``.bandgroup`` manifest references a sibling file that no longer exists on disk.

    Raised at the resolver (``resolve_image_source``), never surfaced as a bare decode error deep
    inside a stacking loop, a stale manifest is a named, actionable refusal.
    """


def _read_xmp_tags(path: Path) -> dict[str, str] | None:
    """Flat ``{qualified_tag: value}`` from a raster's embedded XMP packet, or ``None``.

    Reads both the XMP attribute shape (``drone-dji:CaptureUUID="..."``) and the nested-element
    shape (``<Camera:BandName>Green</Camera:BandName>``), a real sensor's own XMP packet (DJI's)
    uses both shapes in the same file, so a reader that only understood one would silently miss
    tags the declarative table names.
    """
    if path.suffix.lower() not in _METADATA_BEARING_EXTS:
        return None
    try:
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(str(path)) as tif:
            # tifffile types page 0 as TiffPage | TiffFrame; only TiffPage carries parsed tags,
            # and a page with none reads exactly as a page with no XMP tag.
            page_tags = getattr(tif.pages[0], "tags", None)
            tag = page_tags.get(700) if page_tags is not None else None  # the TIFF XMP packet tag id
            if tag is None:
                return None
            raw = tag.value
    except Exception:  # noqa: BLE001, an unreadable/corrupt file offers no metadata, not a crash
        return None
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    tags: dict[str, str] = {}
    for m in _ATTR_RE.finditer(text):
        tags.setdefault(m.group(1), m.group(2))
    for m in _ELEM_RE.finditer(text):
        tags.setdefault(m.group(1), m.group(2))
    return tags or None


def _canonical_stem(paths: list[Path]) -> str:
    """The group's own stem, the sibling filenames' common prefix, trimmed of a trailing
    separator (``DJI_..._0001_MS_G`` / ``..._NIR`` / ``..._R`` / ``..._RE`` -> ``DJI_..._0001_MS``).
    No physical rename happens; this is only the manifest's own filename stem."""
    stems = [p.stem for p in paths]
    prefix = os.path.commonprefix(stems)
    return prefix.rstrip("_-") or stems[0]


def _candidate_single_band_files(images_dir: Path) -> list[Path]:
    """Raster files in ``images_dir`` a group could plausibly be built from."""
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _METADATA_BEARING_EXTS
    )


def detect_embedded_metadata_groups(
    candidates: list[Path],
) -> tuple[list[dict], list[dict]]:
    """Group ``candidates`` by each declarative (group_id_tag, band_id_tag) strategy in turn.

    Returns ``(groups, refused)``. ``groups`` is a list of ``{"stem", "bands",
    "central_wavelength_nm", "source"}`` dicts, not yet written to disk. ``refused`` is a list of
    ``{"group_id", "band", "files", "reason"}`` for a candidate group where two files claimed the
    same band identity, OR where files sharing a group id disagree on (or never had) a strategy's
    own secondary identity signal (:func:`_identity_disagreement`), every case logged, never
    silently formed (one file overwriting another, or two unrelated captures spliced into one).
    """
    tags_by_path: dict[Path, dict[str, str]] = {}
    for p in candidates:
        tags = _read_xmp_tags(p)
        if tags:
            tags_by_path[p] = tags

    groups: list[dict] = []
    refused: list[dict] = []
    grouped_paths: set[Path] = set()

    for strategy in _EMBEDDED_METADATA_STRATEGIES:
        group_id_tag, band_id_tag = strategy.group_id_tag, strategy.band_id_tag
        wavelength_tag = strategy.wavelength_tag
        by_group: dict[str, list[Path]] = defaultdict(list)
        for p, tags in tags_by_path.items():
            if p in grouped_paths:
                continue
            gid = tags.get(group_id_tag)
            if gid:
                by_group[gid].append(p)

        for gid, paths in by_group.items():
            if len(paths) < 2:
                continue  # a lone file with a group id is not a sibling group yet
            if strategy.identity_checks:
                disagreeing_tag, any_checked = _identity_disagreement(
                    paths, tags_by_path, strategy.identity_checks,
                )
                if disagreeing_tag is not None:
                    refused.append({
                        "group_id": gid, "band": None,
                        "files": [str(p) for p in sorted(paths)],
                        "reason": f"files sharing group id {gid!r} disagree on {disagreeing_tag!r} "
                                  "beyond its tolerance, a shared group id alone is not proof of "
                                  "one physical capture, refusing rather than splicing unrelated "
                                  "captures together",
                    })
                    continue
                if not any_checked:
                    refused.append({
                        "group_id": gid, "band": None,
                        "files": [str(p) for p in sorted(paths)],
                        "reason": f"files sharing group id {gid!r} carry none of this strategy's "
                                  "independent identity signals (every declared check's tag is "
                                  "missing from at least one file), a shared group id alone is not "
                                  "proof of one physical capture, and there is no secondary signal "
                                  "left to confirm it, so refusing rather than guessing",
                    })
                    continue
            bands: dict[str, Path] = {}
            wavelengths: dict[str, float] = {}
            duplicate = False
            for p in sorted(paths):
                band_name = tags_by_path[p].get(band_id_tag)
                if not band_name:
                    continue
                if band_name in bands:
                    refused.append({
                        "group_id": gid, "band": band_name,
                        "files": [str(bands[band_name]), str(p)],
                        "reason": f"two files both claim band {band_name!r} for capture {gid!r}",
                    })
                    duplicate = True
                    break
                bands[band_name] = p
                if wavelength_tag:
                    wl = tags_by_path[p].get(wavelength_tag)
                    if wl is not None:
                        try:
                            wavelengths[band_name] = float(wl)
                        except ValueError:
                            pass
            if duplicate or len(bands) < 2:
                continue
            group_paths = list(bands.values())
            groups.append({
                "stem": _canonical_stem(group_paths), "bands": bands,
                "central_wavelength_nm": wavelengths or None, "source": "embedded-metadata",
            })
            grouped_paths.update(group_paths)

    return groups, refused


def groups_from_explicit_mapping(
    images_dir: Path, mapping: dict[str, dict[str, str]],
) -> tuple[list[dict], set[str]]:
    """Groups from a caller-supplied ``{group_id: {band_name: filename}}`` sidecar mapping, for a
    sensor with no embedded correlation metadata at all.

    A named file missing on disk is dropped from its group (not refused, nothing was claimed to
    overwrite); a group left with fewer than 2 resolvable bands is skipped entirely. Returns
    ``(groups, used_filenames)``.
    """
    groups: list[dict] = []
    used: set[str] = set()
    for group_id, band_map in mapping.items():
        bands: dict[str, Path] = {}
        for band_name, filename in band_map.items():
            p = images_dir / filename
            if p.is_file():
                bands[band_name] = p
        if len(bands) < 2:
            continue
        groups.append({
            "stem": str(group_id), "bands": bands,
            "central_wavelength_nm": None, "source": "explicit-manifest",
        })
        used.update(p.name for p in bands.values())
    return groups, used


def read_band_group_manifest(manifest_path: Path) -> BandGroupRef:
    """Parse a ``.bandgroup`` file into a :class:`BandGroupRef`. Raises ``ValueError`` on a
    malformed manifest (missing/empty ``bands``).

    A ``schema_version`` this reader does not accept propagates as
    :class:`tcip_store.SchemaVersionRefused`, uncaught, rather than wrapped as ``ValueError``: a
    version refusal is a policy fact about a newer writer, never the same fact as a malformed
    manifest, and a caller's ``except (OSError, ValueError)`` softener (both enumerators of a
    directory's manifests) must not absorb it and silently dissolve the group into its
    individual band files.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_schema_version(get_descriptor(BAND_GROUP_MANIFEST_STORE), data)
    bands_field = data.get("bands")
    if not isinstance(bands_field, dict) or not bands_field:
        raise ValueError(f"{manifest_path}: 'bands' must be a non-empty {{name: filename}} mapping")
    directory = manifest_path.parent
    bands = {str(name): directory / str(filename) for name, filename in bands_field.items()}
    wl_field = data.get("central_wavelength_nm")
    wavelengths = (
        {str(k): float(v) for k, v in wl_field.items()} if isinstance(wl_field, dict) else None
    )
    return BandGroupRef(
        stem=manifest_path.stem, manifest_path=manifest_path, bands=bands,
        central_wavelength_nm=wavelengths,
    )


def write_band_group_manifest(
    images_dir: Path, stem: str, bands: dict[str, Path], *,
    central_wavelength_nm: dict[str, float] | None = None, source: str = "embedded-metadata",
    expect: Version | None = None,
) -> Path:
    """Write ``<images_dir>/<stem>.bandgroup`` recording ``bands`` (by filename, not full path,
    the originals never move) and return its path.

    ``expect`` carries the seam's own meaning: ``Version.ABSENT`` records a newly detected group
    only while none is recorded, and raises ``VersionConflict`` rather than overwriting one a
    concurrent detection pass just wrote.
    """
    payload: dict = {"bands": {name: p.name for name, p in bands.items()}, "source": source}
    if central_wavelength_nm:
        payload["central_wavelength_nm"] = central_wavelength_nm
    tcip_store.put_blob(
        band_group_manifest_key(images_dir, stem), RECORD_JSON.encode(payload), expect=expect,
    )
    return band_group_manifest_path(images_dir, stem)


def detect_and_write_band_groups(
    images_dir: str | Path, *, explicit_groups: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Run the band-group detection strategies over ``images_dir`` and write a ``.bandgroup``
    manifest for each newly found group.

    Idempotent: a stem with an existing manifest is never regenerated (a recorded fact is not
    re-inferred), only files not already claimed by some manifest are offered as candidates.
    ``explicit_groups`` (strategy 2) is tried first when given; the embedded-metadata table
    (strategy 1) runs over whatever candidates it leaves unclaimed. No match under either -> those
    files are left exactly as independent as they are today (strategy 3, refuse-don't-guess).

    A group whose own stem (the siblings' common prefix, not any one source file's stem) is
    reserved for a prediction bucket's provenance stamp (``tcip_annotation.json_io.
    is_sidecar_name``) is not written as a manifest: minting a logical image under that stem would
    make its label indistinguishable from the stamp everywhere a bucket is walked. Its members are
    left exactly as the standalone files they were, and the group is reported in
    ``"reserved_name_skips"`` instead of ``"formed"``.

    Checked next, against the bucket's own identities (``image_utils.bucket_logical_identities``,
    read once before this loop and updated in place after each manifest this call writes): a key
    (``image_utils.stem_collision_key`` of ``group["stem"]``, the canonical common-prefix stem for
    an embedded-metadata group, the caller's own ``group_id`` for an explicit one) held by any
    identity other than a manifest already recorded under this group's own exact stem, or one of
    this group's own about-to-be-claimed members, puts the group under ``"refused"`` naming the
    colliding file, and writes no manifest -- the same fold the ingest door's pre-scan refuses by,
    so this pass never mints the ambiguity the door exists to keep out. A manifest already
    recorded under the exact stem keeps the ``Version.ABSENT`` idempotence below rather than
    counting as a collision. Writing a manifest also removes its own claimed members from the
    keys they sat under as raw identities, so a later group in this same pass whose stem equals a
    now-claimed member's stem is checked against the bucket as it now stands, not as it stood
    before this group formed.

    Returns ``{"formed": [...], "refused": [...], "manifests": [...], "reserved_name_skips": [...]}``.
    """
    from tcip_annotation.json_io import is_sidecar_name

    from tcip_mcp.pipelines.image_utils import bucket_logical_identities, stem_collision_key

    d = Path(images_dir)
    if not d.is_dir():
        return {"formed": [], "refused": [], "manifests": [], "reserved_name_skips": []}

    already_claimed: set[str] = set()
    for mp in sorted(d.glob(f"*{MANIFEST_EXT}")):
        try:
            ref = read_band_group_manifest(mp)
        except (OSError, ValueError):
            continue  # SchemaVersionRefused is neither: it propagates rather than dissolving the group
        already_claimed.update(p.name for p in ref.bands.values())

    candidates = [p for p in _candidate_single_band_files(d) if p.name not in already_claimed]

    explicit_found: list[dict] = []
    if explicit_groups:
        explicit_found, explicit_used = groups_from_explicit_mapping(d, explicit_groups)
        candidates = [p for p in candidates if p.name not in explicit_used]

    embedded_found, refused = detect_embedded_metadata_groups(candidates)

    identities = bucket_logical_identities(d)

    formed: list[dict] = []
    manifests: list[str] = []
    reserved_name_skips: list[dict] = []
    for group in (*explicit_found, *embedded_found):
        stem = group["stem"]
        if is_sidecar_name(f"{stem}.json"):
            reserved_name_skips.append(
                {"stem": stem, "bands": sorted(group["bands"]), "source": group["source"]}
            )
            continue
        key = stem_collision_key(stem)
        own_members = {p.name for p in group["bands"].values()}
        collision = next(
            (p for p in identities.get(key, [])
             if not (p.suffix.lower() == MANIFEST_EXT and p.stem == stem)
             and p.name not in own_members),
            None,
        )
        if collision is not None:
            refused.append({
                "group_id": stem, "band": None,
                "files": sorted(str(p) for p in group["bands"].values()),
                "reason": f"stem {stem!r} collides with {collision} already in this bucket; "
                          "writing a manifest here would mint the same ambiguity the ingest door "
                          "refuses, so this group is refused instead",
            })
            continue
        try:
            mp = write_band_group_manifest(
                d, stem, group["bands"],
                central_wavelength_nm=group.get("central_wavelength_nm"), source=group["source"],
                expect=Version.ABSENT,
            )
        except VersionConflict:
            continue  # idempotent: a recorded fact is not re-inferred, whoever recorded it
        formed.append({"stem": stem, "bands": sorted(group["bands"]), "source": group["source"]})
        manifests.append(str(mp))
        identities[key] = [mp]
        for member_path in group["bands"].values():
            member_key = stem_collision_key(member_path.stem)
            if member_key == key:
                continue
            remaining = [p for p in identities.get(member_key, []) if p.name != member_path.name]
            if remaining:
                identities[member_key] = remaining
            else:
                identities.pop(member_key, None)

    return {
        "formed": formed, "refused": refused, "manifests": manifests,
        "reserved_name_skips": reserved_name_skips,
    }

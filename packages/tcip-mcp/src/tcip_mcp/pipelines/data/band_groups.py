"""Sensor-agnostic band-group correlation and the ``.bandgroup`` manifest recording a group.

A band group is sibling single-band raster files that are really one logical multi-band capture
(some multispectral drone sensors write one file per band instead of one multi-band file per
image).

A group is recorded, never physically stacked to disk: the manifest names which sibling files
belong together and in what band order. Detection strategies, tried in this order:

1. Explicit manifest, a caller-supplied ``{group_id: {band_name: filename}}`` mapping, for a sensor
  with no embedded correlation metadata at all, when one is given.
2. Embedded-metadata grouping over the files the mapping leaves unclaimed, a declarative table of
  (group-id tag, band-id tag) pairs to look for in each file's XMP. Each row also names secondary
  identity signals (a timestamp, a GPS fix) whose spread across the files claiming one group id
  must stay below the bucket's own separation between distinct captures; a spread reaching it
  refuses the group, and so does a candidate where every declared secondary signal is absent.
3. Otherwise every file stays independent; there is no filename-pattern fallback.
"""

from __future__ import annotations

import io
import os
from collections import ChainMap, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated
from xml.etree.ElementTree import ParseError, iterparse

import numpy as np
from pydantic import BaseModel, Field, JsonValue

import tcip_store
from tcip_store import Version, VersionConflictError, encode_record

MANIFEST_EXT = ".bandgroup"


def band_group_manifest_path(images_dir: str | Path, stem: str) -> Path:
    """Return where ``stem``'s manifest lives, beside the sibling files it names."""
    return Path(images_dir, f"{stem}{MANIFEST_EXT}")


# Extensions an embedded-metadata scan bothers reading, the DJI-shaped rigs this generalizes to
# write one XMP-bearing TIFF per band. A container format with no embedded correlation metadata
# (NPY/NPZ) is only ever grouped via the explicit-manifest strategy.
_METADATA_BEARING_EXTS = (".tif", ".tiff")


@dataclass(frozen=True)
class _MetadataStrategy:
    """One (group-id tag, band-id tag) declarative row, plus its secondary identity signals.

    The signals' values must agree across the files sharing one group-id value before they are
    trusted to be one physical capture.

    ``identity_signals`` names each signal's kind and its tags, read from each candidate file's
    own XMP independently of the group-id tag: ``"timestamp"``, one ISO-8601 tag compared in
    seconds, or ``"position"``, a latitude and a longitude tag in decimal degrees compared in
    meters (:data:`_SIGNAL_KINDS`).
    """

    group_id_tag: str
    band_id_tag: str
    wavelength_tag: str | None = None
    identity_signals: tuple[tuple[str, tuple[str, ...]], ...] = ()


# DJI's multispectral schema is the first instance of the exact-match-shared-group-id class this
# generalizes to, not the mechanism itself, a new sensor of the same class is a new row here.
_EMBEDDED_METADATA_STRATEGIES: list[_MetadataStrategy] = [
    _MetadataStrategy(
        "drone-dji:CaptureUUID",
        "Camera:BandName",
        "Camera:CentralWavelength",
        identity_signals=(
            ("timestamp", ("drone-dji:UTCAtExposure",)),
            ("position", ("drone-dji:GpsLatitude", "drone-dji:GpsLongitude")),
        ),
    ),
]


def _seconds_apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return absolute seconds between paired rows of epoch times."""
    return np.abs(a[:, 0] - b[:, 0])


def _meters_apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the WGS84 geodesic, in meters, between paired rows of ``(lat, lon)`` degrees."""
    from tcip_mcp.pipelines.postprocessing.plant_mapping import WGS84

    return np.asarray(WGS84.inv(a[:, 1], a[:, 0], b[:, 1], b[:, 0])[2])


_SIGNAL_KINDS = {
    "timestamp": (
        lambda raw: (datetime.fromisoformat(raw[0]).timestamp(),),
        _seconds_apart,
        "seconds",
    ),
    "position": (
        lambda raw: tuple(float(v) for v in raw),
        _meters_apart,
        "meters",
    ),
}
"""Each identity-signal kind's reading of a file's tag values, the distance between two
readings, and that distance's unit."""


def _signal_values(
    kind: str, tags: tuple[str, ...], files: list[dict[str, str]]
) -> list[tuple[float, ...] | None]:
    """Return each file's reading of one identity signal (:data:`_SIGNAL_KINDS`).

    A reading is seconds since the epoch for a timestamp, ``(lat, lon)`` degrees for a position;
    ``None`` for a file whose tags are absent or do not parse.
    """
    read = _SIGNAL_KINDS[kind][0]
    values: list[tuple[float, ...] | None] = []
    for file_tags in files:
        raw = [file_tags.get(tag) for tag in tags]
        try:
            values.append(None if None in raw else read(raw))
        except ValueError:
            values.append(None)
    return values


def _gap_and_spreads(
    values: np.ndarray, group_ids: np.ndarray, kind: str
) -> tuple[float | None, dict[str, float]]:
    """Return one identity signal's between-capture gap and each capture group's own spread.

    Both come off one pairwise distance matrix over a bucket's files in the signal's own
    distance (:data:`_SIGNAL_KINDS`), the condensed pairs in
    ``scipy.spatial.distance.pdist``'s order.

    ``values`` holds one reading per file and ``group_ids`` its group id. The gap is the smallest
    distance between two files of distinct groups (``None`` when one group holds every file); a
    group's spread is the largest distance between two of its own files (``0.0`` for one file).
    """
    from scipy.spatial.distance import squareform

    i, j = np.triu_indices(len(values), 1)
    distance = squareform(_SIGNAL_KINDS[kind][1](values[i], values[j]))
    same = group_ids[:, None] == group_ids[None, :]
    spread = {
        str(g): float(distance[np.ix_(group_ids == g, group_ids == g)].max())
        for g in np.unique(group_ids)
    }
    crossing = distance[~same]
    return (float(crossing.min()) if crossing.size else None), spread


class BandGroupManifest(BaseModel):
    """The record a ``.bandgroup`` manifest holds.

    ``bands`` maps each band's name to its file's name beside the manifest, at least one, in the
    order pixels are stacked into ``[H, W, C]``; a name holding a directory part (either slash)
    is refused. ``source`` names the strategy that formed the group. ``identity_tolerance``
    holds, per identity signal the group was checked on, the gap it was checked against, empty
    when none was. ``central_wavelength_nm`` is each band's stated wavelength, ``None`` when
    nothing stated one.
    """

    bands: Annotated[
        dict[str, Annotated[str, Field(pattern=r"^[^/\\]+$")]],
        Field(min_length=1),
    ]
    source: str
    identity_tolerance: dict[str, dict[str, JsonValue]]
    central_wavelength_nm: dict[str, float] | None


@dataclass(frozen=True)
class BandGroupRef:
    """One logical multi-band image, virtually assembled from sibling single-band files.

    ``manifest_path`` is the ``.bandgroup`` file's own path and ``manifest`` the record it holds.
    """

    manifest_path: Path
    manifest: BandGroupManifest

    @property
    def stem(self) -> str:
        """The logical image's stem, its manifest's."""
        return self.manifest_path.stem

    @property
    def bands(self) -> dict[str, Path]:
        """Each band's file, its recorded name resolved beside the manifest, in recorded order."""
        return {
            name: self.manifest_path.parent / filename
            for name, filename in self.manifest.bands.items()
        }


class BandGroupIncompleteError(FileNotFoundError):
    """A ``.bandgroup`` manifest references a sibling file that no longer exists on disk."""


def _read_xmp_tags(path: Path) -> dict[str, str] | None:
    """Return flat qualified tags from an embedded XMP packet, or ``None``.

    The packet, its trailing NUL bytes stripped, is parsed as XML. Each attribute
    (``drone-dji:CaptureUUID="..."``) and each childless element's text
    (``<Camera:BandName>Green</Camera:BandName>``) is keyed ``prefix:Name`` by a prefix bound to
    its namespace where the element sits, attributes before element texts and the first value of
    a repeated tag kept; a name in no namespace or the default one is not offered. A packet that
    is not well-formed XML offers no tags.
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
            # the TIFF XMP packet tag id
            tag = page_tags.get(700) if page_tags is not None else None
            if tag is None:
                return None
            raw = tag.value
    except Exception:  # noqa: BLE001, an unreadable/corrupt file offers no metadata, not a crash
        return None
    packet = raw if isinstance(raw, bytes) else str(raw).encode()
    source = io.BytesIO(packet.rstrip(b"\0"))
    try:
        events = list(iterparse(source, ("start-ns", "end-ns", "start")))
    except ParseError:
        return None
    bindings: ChainMap[str, str] = ChainMap()
    elements = []
    for event, item in events:
        if event == "start-ns":
            bindings = bindings.new_child(dict([item]))
        elif event == "end-ns":
            bindings = bindings.parents
        else:
            elements.append((item, {uri: p for p, uri in bindings.items()}))
    tags: dict[str, str] = {}
    for prefixes, name, value in [
        *((ns, *item) for e, ns in elements for item in e.attrib.items()),
        *((ns, e.tag, e.text or "") for e, ns in elements if len(e) == 0),
    ]:
        uri, brace, local = name[1:].partition("}")
        if brace and prefixes.get(uri):
            tags.setdefault(f"{prefixes[uri]}:{local}", value)
    return tags or None


def _canonical_stem(paths: list[Path]) -> str:
    """Return the group's own stem, the sibling filenames' common prefix.

    The prefix is trimmed of a trailing separator (``DJI_..._0001_MS_G`` / ``..._NIR`` /
    ``..._R`` / ``..._RE`` -> ``DJI_..._0001_MS``), the first sibling's stem when they share
    none.
    """
    stems = [p.stem for p in paths]
    prefix = os.path.commonprefix(stems)
    return prefix.rstrip("_-") or stems[0]


def _candidate_single_band_files(images_dir: Path) -> list[Path]:
    """Return candidate raster files in ``images_dir``."""
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _METADATA_BEARING_EXTS
    )


def _signal_checks(
    strategy: _MetadataStrategy,
    group_of: dict[Path, str],
    tags_by_path: dict[Path, dict[str, str]],
) -> list[tuple[str, str, float | None, dict[str, float]]]:
    """Return ``(tags key, kind, gap, spreads)`` of each of ``strategy``'s identity signals.

    Each is taken over the files ``group_of`` assigns a group id (:func:`_gap_and_spreads`); a
    group some file of which lacks the signal takes no part in it and has no spread.
    """
    paths = list(group_of)
    checks = []
    for kind, tags in strategy.identity_signals:
        values = _signal_values(kind, tags, [tags_by_path[p] for p in paths])
        lacking = {group_of[p] for p, v in zip(paths, values) if v is None}
        kept = [
            (group_of[p], v)
            for p, v in zip(paths, values)
            if group_of[p] not in lacking
        ]
        gap, spreads = (
            _gap_and_spreads(
                np.array([v for _, v in kept]),
                np.array([g for g, _ in kept]),
                kind,
            )
            if kept
            else (None, {})
        )
        checks.append((",".join(tags), kind, gap, spreads))
    return checks


def detect_embedded_metadata_groups(
    candidates: list[Path], reference: list[Path]
) -> tuple[list[BandGroupRef], list[dict]]:
    """Group ``candidates`` by each declarative (group_id_tag, band_id_tag) strategy in turn.

    Returns ``(groups, refused)``. ``groups`` holds each formed group's record at its manifest
    path beside its files, not yet written to disk. ``refused`` is a list of ``{"group_id",
    "band", "files", "reason"}`` for a candidate group where two files claimed the same band
    identity, where its files carry none of the strategy's identity signals, or where its own
    spread on a signal reaches that signal's between-capture gap, every case reported in
    ``refused``, never silently formed (one file overwriting another, or two unrelated captures
    spliced into one).

    The gap and the spreads are derived over ``reference``, every metadata-bearing file of the
    bucket ``candidates`` sit in, grouped ones included (:func:`_signal_checks`). A formed group's
    ``identity_tolerance`` holds, per signal it was checked on and keyed by the signal's tags,
    ``{"value", "source"}``: the gap and how it was derived, or ``None`` with the fact that no
    other capture group of the bucket carries the signal, so the group formed on its id alone.
    """
    tags_by_path: dict[Path, dict[str, str]] = {}
    for p in sorted({*candidates, *reference}):
        tags = _read_xmp_tags(p)
        if tags:
            tags_by_path[p] = tags

    groups: list[BandGroupRef] = []
    refused: list[dict] = []
    grouped_paths: set[Path] = set()
    offered = set(candidates)

    for strategy in _EMBEDDED_METADATA_STRATEGIES:
        group_id_tag, band_id_tag = strategy.group_id_tag, strategy.band_id_tag
        wavelength_tag = strategy.wavelength_tag
        group_of = {
            p: tags[group_id_tag]
            for p, tags in tags_by_path.items()
            if tags.get(group_id_tag)
        }
        checks = _signal_checks(strategy, group_of, tags_by_path)
        by_group: dict[str, list[Path]] = defaultdict(list)
        for p, gid in group_of.items():
            if p in offered and p not in grouped_paths:
                by_group[gid].append(p)

        for gid, paths in by_group.items():
            if len(paths) < 2:
                continue  # a lone file with a group id is not a sibling group yet
            checked = [
                (key, kind, gap, spreads[gid])
                for key, kind, gap, spreads in checks
                if gid in spreads
            ]
            reached = next(
                (c for c in checked if c[2] is not None and c[3] >= c[2]), None
            )
            if reached is not None:
                key, kind, gap, spread = reached
                refused.append(
                    {
                        "group_id": gid,
                        "band": None,
                        "files": [str(p) for p in sorted(paths)],
                        "reason": f"files sharing group id {gid!r} spread {spread:.6g} "
                        f"{_SIGNAL_KINDS[kind][2]} on {key!r}, reaching the {gap:.6g} that "
                        "separates the nearest two capture groups of this bucket: a shared "
                        "group id alone is not proof of one physical capture, refusing rather "
                        "than splicing unrelated captures together",
                    }
                )
                continue
            if strategy.identity_signals and not checked:
                refused.append(
                    {
                        "group_id": gid,
                        "band": None,
                        "files": [str(p) for p in sorted(paths)],
                        "reason": f"files sharing group id {gid!r} carry none of this "
                        "strategy's independent identity signals (every declared signal's "
                        "tag is missing from at least one file), a shared group id alone is "
                        "not proof of one physical capture, and there is no secondary signal "
                        "left to confirm it, so refusing rather than guessing",
                    }
                )
                continue
            bands: dict[str, Path] = {}
            wavelengths: dict[str, float] = {}
            duplicate = False
            for p in sorted(paths):
                band_name = tags_by_path[p].get(band_id_tag)
                if not band_name:
                    continue
                if band_name in bands:
                    refused.append(
                        {
                            "group_id": gid,
                            "band": band_name,
                            "files": [str(bands[band_name]), str(p)],
                            "reason": f"two files both claim band {band_name!r} for capture "
                            f"{gid!r}",
                        }
                    )
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
            bucket = group_paths[0].parent
            groups.append(
                BandGroupRef(
                    band_group_manifest_path(
                        bucket, _canonical_stem(group_paths)
                    ),
                    BandGroupManifest(
                        bands={name: p.name for name, p in bands.items()},
                        source="embedded-metadata",
                        identity_tolerance={
                            key: {
                                "value": gap,
                                "source": "the smallest separation in "
                                f"{_SIGNAL_KINDS[kind][2]} between two capture groups' "
                                f"values in bucket {bucket.name!r}, where the group was "
                                "detected"
                                if gap is not None
                                else "no other capture group in bucket "
                                f"{bucket.name!r}, where the group was detected, carries "
                                "this signal, so the group formed on its group id alone",
                            }
                            for key, kind, gap, _spread in checked
                        },
                        central_wavelength_nm=wavelengths or None,
                    ),
                )
            )
            grouped_paths.update(group_paths)

    return groups, refused


def groups_from_explicit_mapping(
    images_dir: Path,
    mapping: dict[str, dict[str, str]],
) -> list[BandGroupRef]:
    """Return groups from a caller-supplied ``{group_id: {band_name: filename}}`` mapping.

    The sidecar mapping serves a sensor with no embedded correlation metadata at all. Each group
    is its record at ``images_dir``'s manifest path for its ``group_id``, not yet written.

    A named file missing on disk is dropped from its group; a group left with fewer than 2
    resolvable bands is skipped entirely. A group naming a file it resolves through a directory
    part raises ``ValueError`` (:class:`BandGroupManifest`'s ``bands``).
    """
    groups = []
    for group_id, band_map in mapping.items():
        bands = {
            band_name: filename
            for band_name, filename in band_map.items()
            if (images_dir / filename).is_file()
        }
        if len(bands) >= 2:
            groups.append(
                BandGroupRef(
                    band_group_manifest_path(images_dir, group_id),
                    BandGroupManifest(
                        bands=bands,
                        source="explicit-manifest",
                        identity_tolerance={},
                        central_wavelength_nm=None,
                    ),
                )
            )
    return groups


def read_band_group_manifest(manifest_path: Path) -> BandGroupRef:
    """Return the :class:`BandGroupRef` of the ``.bandgroup`` file at ``manifest_path``.

    Raises ``ValueError`` (``pydantic.ValidationError``) when the file's bytes are not a
    :class:`BandGroupManifest`.
    """
    return BandGroupRef(
        manifest_path,
        BandGroupManifest.model_validate_json(manifest_path.read_bytes()),
    )


def write_band_group_manifest(
    group: BandGroupRef, *, expect: Version | None = None
) -> Path:
    """Write ``group``'s record to its manifest path and return that path.

    ``expect`` carries the seam's own meaning: ``Version.ABSENT`` records a newly detected group
    only while none is recorded, and raises ``VersionConflictError`` rather than overwriting one a
    concurrent detection pass just wrote.
    """
    tcip_store.put_blob(
        group.manifest_path,
        encode_record(group.manifest.model_dump(mode="json")),
        expect=expect,
    )
    return group.manifest_path


def detect_and_write_band_groups(
    images_dir: str | Path,
    *,
    explicit_groups: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Run the band-group detection strategies over ``images_dir`` and record what they find.

    A ``.bandgroup`` manifest is written for each newly found group.

    Idempotent: a stem with an existing manifest is never regenerated, and only files not
    already claimed by some manifest are offered as candidates. ``explicit_groups`` (strategy 1)
    is tried first when given; the embedded-metadata table (strategy 2) runs over whatever
    candidates it leaves unclaimed. Files no strategy matches are left independent.

    A group whose ``image_utils.stem_collision_key`` (of the common-prefix stem, or the caller's
    ``group_id`` for an explicit group) is held by any identity in the bucket
    (``image_utils.scan_identities``, the one read of its manifests) other than a manifest
    already recorded under this group's exact stem or one of its own members goes under
    ``"refused"`` naming the colliding file, and writes no manifest. A member is the file its
    recorded name resolves to beside the manifest. The identities are updated after each manifest
    this call writes, so a later group is checked against the bucket as it now stands.

    Returns ``{"formed": [...], "refused": [...], "manifests": [...]}``.
    """
    from tcip_mcp.pipelines.image_utils import (
        scan_identities,
        source_path_of,
        stem_collision_key,
    )

    d = Path(images_dir)
    if not d.is_dir():
        return {"formed": [], "refused": [], "manifests": []}

    identities = scan_identities(d)
    explicit_found = groups_from_explicit_mapping(d, explicit_groups or {})
    recorded = [
        entry
        for entries in identities.values()
        for entry in entries
        if isinstance(entry, BandGroupRef)
    ]
    claimed = {
        p for g in (*recorded, *explicit_found) for p in g.bands.values()
    }
    bucket_files = _candidate_single_band_files(d)
    embedded_found, refused = detect_embedded_metadata_groups(
        [p for p in bucket_files if p not in claimed], bucket_files
    )

    formed: list[dict] = []
    manifests: list[str] = []
    for group in (*explicit_found, *embedded_found):
        key = stem_collision_key(group.stem)
        members = group.bands
        collision = next(
            (
                source_path_of(entry)
                for entry in identities.get(key, [])
                if (
                    entry.stem != group.stem
                    if isinstance(entry, BandGroupRef)
                    else entry not in members.values()
                )
            ),
            None,
        )
        if collision is not None:
            refused.append(
                {
                    "group_id": group.stem,
                    "band": None,
                    "files": sorted(str(p) for p in members.values()),
                    "reason": f"stem {group.stem!r} collides with {collision} already in this "
                    "bucket; writing a manifest here would mint the same ambiguity the ingest "
                    "door refuses, so this group is refused instead",
                }
            )
            continue
        try:
            write_band_group_manifest(group, expect=Version.ABSENT)
        except VersionConflictError:
            continue  # idempotent: a recorded fact is not re-inferred, whoever recorded it
        formed.append(
            {
                "stem": group.stem,
                "bands": sorted(group.manifest.bands),
                "source": group.manifest.source,
            }
        )
        manifests.append(str(group.manifest_path))
        identities[key] = [group]
        for member in members.values():
            member_key = stem_collision_key(member.stem)
            if member_key == key:
                continue
            remaining = [
                entry
                for entry in identities.get(member_key, [])
                if entry != member
            ]
            if remaining:
                identities[member_key] = remaining
            else:
                identities.pop(member_key, None)

    return {"formed": formed, "refused": refused, "manifests": manifests}

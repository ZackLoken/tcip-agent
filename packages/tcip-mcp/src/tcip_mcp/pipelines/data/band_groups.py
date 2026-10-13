"""Sensor-agnostic band-group correlation: sibling single-band raster files that are really one
logical multi-band capture (some multispectral drone sensors write one file per band instead of one
multi-band file per image), and the ``.bandgroup`` manifest that records a found group.

A group is recorded, never physically stacked to disk: the manifest names which sibling files
belong together and in what band order. Detection strategies, tried in order:

1. Embedded-metadata grouping, a declarative table of (group-id tag, band-id tag) pairs to look for
  in each file's XMP; a sensor with an exact-match, shared group-id-per-capture tag is a new table
  row. Each row also names secondary identity signals (a timestamp, a GPS fix) whose spread across
  the files claiming one group id must stay below the bucket's own separation between distinct
  captures; a spread reaching it refuses the group, and so does a candidate where every declared
  secondary signal is absent.
2. Explicit manifest, a caller-supplied ``{group_id: {band_name: filename}}`` mapping, for a sensor
  with no embedded correlation metadata at all.
3. Otherwise every file stays independent; there is no filename-pattern fallback.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import tcip_store
from tcip_store import Version, VersionConflictError, encode_record

MANIFEST_EXT = ".bandgroup"


def band_group_manifest_path(images_dir: str | Path, stem: str) -> Path:
    """Where ``stem``'s manifest lives: beside the sibling files it names, itself an enumerated
    logical image of that directory.
    """
    return Path(images_dir, f"{stem}{MANIFEST_EXT}")


# Extensions an embedded-metadata scan bothers reading, the DJI-shaped rigs this generalizes to
# write one XMP-bearing TIFF per band. A container format with no embedded correlation metadata
# (NPY/NPZ) is only ever grouped via the explicit-manifest strategy.
_METADATA_BEARING_EXTS = (".tif", ".tiff")

_ATTR_RE = re.compile(r'([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)="([^"]*)"')
_ELEM_RE = re.compile(r"<([A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*)>([^<]*)</\1>")


@dataclass(frozen=True)
class _MetadataStrategy:
    """One (group-id tag, band-id tag) declarative row, plus the secondary identity signals whose
    values must agree across the files sharing one group-id value before they are trusted to be
    one physical capture.

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
    """Seconds between paired rows of epoch times."""
    return np.abs(a[:, 0] - b[:, 0])


def _meters_apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The WGS84 geodesic, in meters, between paired rows of ``(lat, lon)`` degrees."""
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
    """Each file's reading of one identity signal (:data:`_SIGNAL_KINDS`): seconds since the epoch
    for a timestamp, ``(lat, lon)`` degrees for a position; ``None`` for a file whose tags are
    absent or do not parse.
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
    """The between-capture gap of one identity signal over a bucket's files, and each capture
    group's own spread, both off one pairwise distance matrix in the signal's own distance
    (:data:`_SIGNAL_KINDS`), the condensed pairs in ``scipy.spatial.distance.pdist``'s order.

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


@dataclass(frozen=True)
class BandGroupRef:
    """One logical multi-band image, virtually assembled from sibling single-band files.

    ``manifest_path`` is the ``.bandgroup`` file's own path. ``bands`` is ``{band_name:
    file_path}`` in the manifest's declared order (the order pixels are stacked into ``[H, W,
    C]``). ``central_wavelength_nm`` is each band's stated wavelength, ``None`` when nothing stated
    one.
    """

    stem: str
    manifest_path: Path
    bands: dict[str, Path]
    central_wavelength_nm: dict[str, float] | None


class BandGroupIncompleteError(FileNotFoundError):
    """A ``.bandgroup`` manifest references a sibling file that no longer exists on disk."""


def _read_xmp_tags(path: Path) -> dict[str, str] | None:
    """Flat ``{qualified_tag: value}`` from a raster's embedded XMP packet, or ``None``.

    Reads both the XMP attribute shape (``drone-dji:CaptureUUID="..."``) and the nested-element
    shape (``<Camera:BandName>Green</Camera:BandName>``).
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
    text = (
        raw.decode("utf-8", errors="replace")
        if isinstance(raw, bytes)
        else str(raw)
    )
    tags: dict[str, str] = {}
    for m in _ATTR_RE.finditer(text):
        tags.setdefault(m.group(1), m.group(2))
    for m in _ELEM_RE.finditer(text):
        tags.setdefault(m.group(1), m.group(2))
    return tags or None


def _canonical_stem(paths: list[Path]) -> str:
    """The group's own stem, the sibling filenames' common prefix, trimmed of a trailing
    separator (``DJI_..._0001_MS_G`` / ``..._NIR`` / ``..._R`` / ``..._RE`` -> ``DJI_..._0001_MS``).
    No physical rename happens; this is only the manifest's own filename stem.
    """
    stems = [p.stem for p in paths]
    prefix = os.path.commonprefix(stems)
    return prefix.rstrip("_-") or stems[0]


def _candidate_single_band_files(images_dir: Path) -> list[Path]:
    """Raster files in ``images_dir`` a group could plausibly be built from."""
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
    """``(tags key, kind, gap, spreads)`` of each of ``strategy``'s identity signals over the
    files ``group_of`` assigns a group id (:func:`_gap_and_spreads`); a group some file of which
    lacks the signal takes no part in it and has no spread.
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
) -> tuple[list[dict], list[dict]]:
    """Group ``candidates`` by each declarative (group_id_tag, band_id_tag) strategy in turn.

    Returns ``(groups, refused)``. ``groups`` is a list of ``{"stem", "bands",
    "central_wavelength_nm", "source", "identity_tolerance"}`` dicts, not yet written to disk.
    ``refused`` is a list of ``{"group_id", "band", "files", "reason"}`` for a candidate group
    where two files claimed the same band identity, where its files carry none of the strategy's
    identity signals, or where its own spread on a signal reaches that signal's between-capture
    gap, every case logged, never silently formed (one file overwriting another, or two unrelated
    captures spliced into one).

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

    groups: list[dict] = []
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
            bucket = group_paths[0].parent.name
            groups.append(
                {
                    "stem": _canonical_stem(group_paths),
                    "bands": bands,
                    "central_wavelength_nm": wavelengths or None,
                    "source": "embedded-metadata",
                    "identity_tolerance": {
                        key: {
                            "value": gap,
                            "source": f"the smallest separation in {_SIGNAL_KINDS[kind][2]} "
                            "between two capture groups' values in bucket "
                            f"{bucket!r}, where the group was detected"
                            if gap is not None
                            else f"no other capture group in bucket {bucket!r}, where the "
                            "group was detected, carries this signal, so the group formed "
                            "on its group id alone",
                        }
                        for key, kind, gap, _spread in checked
                    },
                }
            )
            grouped_paths.update(group_paths)

    return groups, refused


def groups_from_explicit_mapping(
    images_dir: Path,
    mapping: dict[str, dict[str, str]],
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
        groups.append(
            {
                "stem": str(group_id),
                "bands": bands,
                "central_wavelength_nm": None,
                "source": "explicit-manifest",
                "identity_tolerance": {},
            }
        )
        used.update(p.name for p in bands.values())
    return groups, used


def read_band_group_manifest(manifest_path: Path) -> BandGroupRef:
    """Parse a ``.bandgroup`` file into a :class:`BandGroupRef`. Raises ``ValueError`` on a
    malformed manifest (missing/empty ``bands``) and ``KeyError`` on one carrying no
    ``central_wavelength_nm``.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    bands_field = data.get("bands")
    if not isinstance(bands_field, dict) or not bands_field:
        raise ValueError(
            f"{manifest_path}: 'bands' must be a non-empty {{name: filename}} mapping"
        )
    directory = manifest_path.parent
    bands = {
        str(name): directory / str(filename)
        for name, filename in bands_field.items()
    }
    wl_field = data["central_wavelength_nm"]
    wavelengths = (
        None
        if wl_field is None
        else {str(k): float(v) for k, v in wl_field.items()}
    )
    return BandGroupRef(
        stem=manifest_path.stem,
        manifest_path=manifest_path,
        bands=bands,
        central_wavelength_nm=wavelengths,
    )


def write_band_group_manifest(
    images_dir: Path,
    stem: str,
    bands: dict[str, Path],
    *,
    central_wavelength_nm: dict[str, float] | None,
    source: str = "embedded-metadata",
    identity_tolerance: dict[str, dict],
    expect: Version | None = None,
) -> Path:
    """Write ``<images_dir>/<stem>.bandgroup`` recording ``bands`` by filename (so a group moved
    whole with its band files still resolves), each band's ``central_wavelength_nm`` (``None``
    when nothing stated one) and the ``identity_tolerance`` each identity signal was checked
    against, empty for a group no signal was checked on; return its path.

    ``expect`` carries the seam's own meaning: ``Version.ABSENT`` records a newly detected group
    only while none is recorded, and raises ``VersionConflictError`` rather than overwriting one a
    concurrent detection pass just wrote.
    """
    payload: dict = {
        "bands": {name: p.name for name, p in bands.items()},
        "source": source,
        "identity_tolerance": identity_tolerance,
        "central_wavelength_nm": central_wavelength_nm,
    }
    path = band_group_manifest_path(images_dir, stem)
    tcip_store.put_blob(path, encode_record(payload), expect=expect)
    return path


def detect_and_write_band_groups(
    images_dir: str | Path,
    *,
    explicit_groups: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Run the band-group detection strategies over ``images_dir`` and write a ``.bandgroup``
    manifest for each newly found group.

    Idempotent: a stem with an existing manifest is never regenerated, and only files not already
        claimed by some manifest are offered as candidates. ``explicit_groups`` (strategy 2) is
        tried first when given; the embedded-metadata table (strategy 1) runs over whatever
        candidates it leaves unclaimed. Files no strategy matches are left independent.

    A group whose ``image_utils.stem_collision_key`` (of the common-prefix stem, or the caller's
    ``group_id`` for an explicit group) is held by any identity in the bucket
    (``image_utils.bucket_logical_identities``) other than a manifest already recorded under this
    group's exact stem or one of its own members goes under ``"refused"`` naming the colliding
    file, and writes no manifest. The identities are updated after each manifest this call writes,
    so a later group is checked against the bucket as it now stands.

    Returns ``{"formed": [...], "refused": [...], "manifests": [...]}``.
    """
    from tcip_mcp.pipelines.image_utils import (
        bucket_logical_identities,
        stem_collision_key,
    )

    d = Path(images_dir)
    if not d.is_dir():
        return {"formed": [], "refused": [], "manifests": []}

    already_claimed: set[str] = set()
    for mp in sorted(d.glob(f"*{MANIFEST_EXT}")):
        try:
            ref = read_band_group_manifest(mp)
        except (OSError, ValueError):
            continue
        already_claimed.update(p.name for p in ref.bands.values())

    bucket_files = _candidate_single_band_files(d)
    candidates = [p for p in bucket_files if p.name not in already_claimed]

    explicit_found: list[dict] = []
    if explicit_groups:
        explicit_found, explicit_used = groups_from_explicit_mapping(
            d, explicit_groups
        )
        candidates = [p for p in candidates if p.name not in explicit_used]

    embedded_found, refused = detect_embedded_metadata_groups(
        candidates, bucket_files
    )

    identities = bucket_logical_identities(d)

    formed: list[dict] = []
    manifests: list[str] = []
    for group in (*explicit_found, *embedded_found):
        stem = group["stem"]
        key = stem_collision_key(stem)
        own_members = {p.name for p in group["bands"].values()}
        collision = next(
            (
                p
                for p in identities.get(key, [])
                if not (p.suffix.lower() == MANIFEST_EXT and p.stem == stem)
                and p.name not in own_members
            ),
            None,
        )
        if collision is not None:
            refused.append(
                {
                    "group_id": stem,
                    "band": None,
                    "files": sorted(str(p) for p in group["bands"].values()),
                    "reason": f"stem {stem!r} collides with {collision} already in this bucket; "
                    "writing a manifest here would mint the same ambiguity the ingest door "
                    "refuses, so this group is refused instead",
                }
            )
            continue
        try:
            mp = write_band_group_manifest(
                d,
                stem,
                group["bands"],
                central_wavelength_nm=group["central_wavelength_nm"],
                source=group["source"],
                identity_tolerance=group["identity_tolerance"],
                expect=Version.ABSENT,
            )
        except VersionConflictError:
            continue  # idempotent: a recorded fact is not re-inferred, whoever recorded it
        formed.append(
            {
                "stem": stem,
                "bands": sorted(group["bands"]),
                "source": group["source"],
            }
        )
        manifests.append(str(mp))
        identities[key] = [mp]
        for member_path in group["bands"].values():
            member_key = stem_collision_key(member_path.stem)
            if member_key == key:
                continue
            remaining = [
                p
                for p in identities.get(member_key, [])
                if p.name != member_path.name
            ]
            if remaining:
                identities[member_key] = remaining
            else:
                identities.pop(member_key, None)

    return {"formed": formed, "refused": refused, "manifests": manifests}

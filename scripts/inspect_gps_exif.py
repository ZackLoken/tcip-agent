"""Print GPS EXIF for a sample of images per acquisition date.

Tells us whether RTK was actually used (centimeter precision, HPositioningError
~0.01-0.10) or whether it's iPhone internal GPS (~3-10 m precision).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

from _paths import vf_root

ROOT = vf_root() / "images"


def read_gps(path: Path) -> dict:
    im = Image.open(path)
    # _getexif is a JpegImageFile/MpoImageFile accessor, absent from the ImageFile stub.
    raw = cast(Any, im)._getexif() or {}
    for tag_id, value in raw.items():
        if TAGS.get(tag_id) == "GPSInfo":
            return {GPSTAGS.get(k, k): v for k, v in value.items()}
    return {}


def to_decimal(dms, ref) -> float | None:
    if not dms:
        return None
    d, m, s = (float(x) for x in dms)
    val = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        val = -val
    return val


def summarize(path: Path) -> None:
    gps = read_gps(path)
    lat = to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    print(f"\n{path.name}")
    print(f"  lat/lon:   {lat!r}, {lon!r}")
    print(f"  altitude:  {gps.get('GPSAltitude')!r}")
    print(f"  DOP:       {gps.get('GPSDOP')!r}")
    print(f"  HPosErr:   {gps.get('GPSHPositioningError')!r}")
    print(f"  ProcMeth:  {gps.get('GPSProcessingMethod')!r}")
    print(f"  all keys:  {sorted(gps.keys())}")


def main() -> None:
    for date_dir in sorted(ROOT.iterdir()):
        if not date_dir.is_dir():
            continue
        imgs = sorted(p for p in date_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".heic"})
        if not imgs:
            continue
        print(f"\n=== {date_dir.name} ({len(imgs)} images) ===")
        sample = [imgs[0], imgs[len(imgs) // 2], imgs[-1]]
        for p in sample:
            try:
                summarize(p)
            except Exception as e:
                print(f"  [error on {p.name}: {e}]")


if __name__ == "__main__":
    main()

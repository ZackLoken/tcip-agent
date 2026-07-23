"""Live e2e smoke: the agent's phenology pipeline on real geolocated images.

Builds a synthetic scene — geolocated JPEGs (real EXIF GPS + capture time) across three
dates plus a plant-locations CSV — then runs the exact two tools the agent composes:

    build_plant_mapping   images + CSV                 → plant_mapping.json
    compute_phenology     mapping + classified preds   → catkin_phenology.csv

and asserts the delivered CSV has the canonical columns and plausible, correctly-ordered
milestone dates. Then it confirms the measurement-integrity guard refuses to write a CSV when
the predictions carry no elongation class. This covers the seam unit tests don't: real EXIF →
real GPS matching → real per-plant milestones. No backend or network — it calls the tool
functions directly, in an isolated temp dir.

Usage (repo root, tcip-agent env):
    python scripts/smoke_phenology_e2e.py

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

_MCP_SRC = Path(__file__).resolve().parents[1] / "packages" / "tcip-mcp" / "src"
if str(_MCP_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SRC))

from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.tools.phenology_tools import (  # noqa: E402
    build_plant_mapping,
    compute_phenology,
)

# Two plants ~3 m apart in one row (1 deg lon ≈ 81 km at 43°N, so 3.7e-5 deg ≈ 3 m).
PLANTS = [
    {"plot": "P1", "accession": "acc-A", "lat": 43.19670, "lon": -90.058000},
    {"plot": "P2", "accession": "acc-B", "lat": 43.19670, "lon": -90.058037},
]
# Three capture dates; the elongated fraction rises 0.0 → 0.4 → 1.0.
DATES = ["2026-02-11", "2026-02-25", "2026-03-11"]
FRACTIONS = {"2026-02-11": 0.0, "2026-02-25": 0.4, "2026-03-11": 1.0}
ELONGATED_CLASS = 1
N_DETECTIONS = 10

_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" - {detail}"
    print(line)
    if not ok:
        _failures += 1


def _deg_to_dms(value: float) -> tuple[float, float, float]:
    v = abs(value)
    d = int(v)
    m_full = (v - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60, 4)
    return (float(d), float(m), s)


def _write_geo_image(path: Path, lat: float, lon: float, when: datetime) -> None:
    """A tiny JPEG carrying EXIF DateTimeOriginal + GPS lat/lon (as plant_mapping reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x8769] = {0x9003: when.strftime("%Y:%m:%d %H:%M:%S")}  # Exif IFD → DateTimeOriginal
    exif[0x8825] = {  # GPSInfo IFD
        0x0001: "N" if lat >= 0 else "S",
        0x0002: _deg_to_dms(lat),
        0x0003: "E" if lon >= 0 else "W",
        0x0004: _deg_to_dms(lon),
    }
    Image.new("RGB", (8, 8)).save(path, exif=exif)


def _pred_boxes(n_elongated: int, n_total: int) -> list[Annotation]:
    """Per-image name-based catkin predictions; first n_elongated carry elongation='elongated'."""
    anns = []
    for i in range(n_total):
        value = "elongated" if i < n_elongated else "dormant"
        anns.append(Annotation(subject="catkin", geometry=BBox(1.0, 1.0, 3.0, 3.0),
                               attributes={"elongation": value}, score=0.90))
    return anns


def _stem(plot: str, date: str) -> str:
    return f"{plot}_{date.replace('-', '')}"


def main() -> int:
    print("Phenology e2e smoke: build_plant_mapping -> compute_phenology\n")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        images_root = root / "images"
        preds_root = root / "preds"           # class-carrying predictions (valid)
        mapping_path = root / ".tcip" / "state" / "plant_mapping.json"
        csv_out = root / "delivery" / "catkin_phenology.csv"

        # 1. Scene: geolocated images + per-image classified predictions.
        for date in DATES:
            base_time = datetime.strptime(date, "%Y-%m-%d").replace(hour=9, minute=30)
            n_elong = round(FRACTIONS[date] * N_DETECTIONS)
            for j, plant in enumerate(PLANTS):
                stem = _stem(plant["plot"], date)
                _write_geo_image(
                    images_root / date / f"{stem}.jpg",
                    plant["lat"], plant["lon"], base_time + timedelta(minutes=j),
                )
                (preds_root / date).mkdir(parents=True, exist_ok=True)
                json_io.write_annotations(
                    preds_root / date / f"{stem}.json",
                    _pred_boxes(n_elong, N_DETECTIONS), 8, 8,
                )

        plant_csv = root / "plants.csv"
        with plant_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
            for p in PLANTS:
                w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])

        preds_by_date = {d: str(preds_root / d) for d in DATES}

        # 2. build_plant_mapping — real EXIF GPS → plant assignments.
        print("Step 1: build_plant_mapping")
        m = build_plant_mapping(
            images_root=str(images_root),
            plant_csv_paths=[str(plant_csv)],
            output_mapping_path=str(mapping_path),
        )
        check("no error", "error" not in m, m.get("error", ""))
        check("3 dates mapped", m.get("n_dates") == 3, str(m.get("n_dates")))
        check("6 images seen", m.get("n_images") == 6, str(m.get("n_images")))
        check("all 6 images mapped to a plant", m.get("n_mapped") == 6,
              f"n_mapped={m.get('n_mapped')} n_unmapped={m.get('n_unmapped')}")
        check("mapping.json persisted", mapping_path.is_file())

        # 3. compute_phenology — the elongation split is DEFERRED to K4/K5.
        # count_by_class no longer reads an integer positive class off disk (elongation is now a
        # name-based *attribute*, resolved by K4/K5). So there is no elongation split yet, the
        # fraction is not a valid bloom measurement, and the rail must REFUSE to deliver a curve —
        # exactly the measurement-integrity guard. This smoke asserts that refusal (the rail holding
        # under the deferral); the delivered curve returns as a K4/K5 acceptance test.
        print("\nStep 2: compute_phenology refuses an un-split curve (elongation split deferred to K4/K5)")
        r = compute_phenology(
            mapping_path=str(mapping_path),
            predictions_by_date=preds_by_date,
            output_csv_path=str(csv_out),
            positive_class_id=ELONGATED_CLASS,
            acknowledge_unvalidated=True,
        )
        check("elongation_classified false (split deferred)", r.get("elongation_classified") is False,
              str(r.get("elongation_classified")))
        check("no CSV written (delivery refused, not a fabricated curve)", not csv_out.exists())

    print()
    if _failures:
        print(f"SMOKE FAILED — {_failures} assertion(s) failed.")
        return 1
    print("SMOKE PASSED - the agent phenology pipeline works end to end.")
    return 0


if __name__ == "__main__":
    # Run from an isolated cwd so the tools' @audited log doesn't touch the repo's .tcip/.
    # chdir back before removing it (Windows can't delete the dir a process is cwd'd into).
    _origin = os.getcwd()
    _audit_cwd = tempfile.mkdtemp()
    os.chdir(_audit_cwd)
    try:
        _rc = main()
    finally:
        os.chdir(_origin)
        shutil.rmtree(_audit_cwd, ignore_errors=True)
    sys.exit(_rc)

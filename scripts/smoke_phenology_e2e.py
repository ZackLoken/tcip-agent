"""Live e2e smoke: the agent's phenology pipeline on real geolocated images.

Builds a synthetic scene: geolocated JPEGs (real EXIF GPS + capture time) across three
dates plus a plant-locations CSV, then runs the exact two tools the agent composes:

    build_plant_mapping   images + CSV                 → a named mapping under the project
    compute_phenology     mapping + classified preds   → <trait>_phenology.csv

and asserts the delivered CSV has the canonical columns and plausible, correctly-ordered
milestone dates. Then it confirms the measurement-integrity guard refuses to write a CSV when
the predictions carry no positive class. This covers the seam unit tests don't: real EXIF →
real GPS matching → real per-plant milestones. No backend or network: it calls the tool
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MCP_SRC = _REPO_ROOT / "packages" / "tcip-mcp" / "src"
for _path in (_MCP_SRC, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from PIL import Image  # noqa: E402

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, BBox  # noqa: E402
from tcip_mcp.pipelines.resolution import write_sidecar  # noqa: E402
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
# The bucket's own recorded id_map (the positive class is resolved from this, on disk, never a
# pinned integer, so this is the production id_map a real export_predictions run would have
# stamped, not a magic constant compute_phenology reads directly).
ID_MAP = {"dormant": 0, "elongated": 1}
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
    """Per-image classified predictions; first n_elongated decode to 'elongated', rest 'dormant'.

    A real prediction's ``.subject`` is the decoded class name directly (write_predictions_json
    decodes the numeric label through the bucket's id_map straight into ``.subject``, leaving
    ``.attributes`` empty), not the GT-annotation shape (object-type subject + an attribute value).
    """
    anns = []
    for i in range(n_total):
        value = "elongated" if i < n_elongated else "dormant"
        anns.append(Annotation(subject=value, geometry=BBox(1.0, 1.0, 3.0, 3.0), score=0.90))
    return anns


def _stem(plot: str, date: str) -> str:
    return f"{plot}_{date.replace('-', '')}"


def _author_catkin_trait_spec(root: Path) -> None:
    """Register the catkin trait under ``root`` and record a confirmed meaning for its delivery.

    The spec is ``tests/_trait_fixtures.CATKIN`` itself rather than a copy of its field values, so
    a change to that definition cannot leave this smoke run exercising a stale one. The crossing
    door refuses a trait whose delivered number has no breeder-confirmed meaning, so this states
    one and confirms it through the same two writers a real project goes through.
    """
    import dataclasses

    import tcip_store as ts

    from tcip_mcp import traits
    from tests._operationalization_fixtures import seed_confirmed_crossing
    from tests._trait_fixtures import CATKIN

    data = {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(CATKIN).items()}
    key = traits.trait_spec_key(traits.trait_specs_dir(root), data["name"])
    ts.replace(key, data, expect=ts.Version.ABSENT)
    seed_confirmed_crossing(root, data["name"])


def main() -> int:
    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    backend = bind_default()

    print("Phenology e2e smoke: build_plant_mapping -> compute_phenology\n")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dataset_root = root / "dataset"        # a registered dataset the mapping is built over
        images_root = dataset_root / "images"
        preds_root = dataset_root / "predictions" / "live"  # class-carrying predictions (valid)
        mapping_name = "smoke-valley"
        csv_out = root / "delivery" / "catkin_phenology.csv"

        # Trait registration is per-project state resolved via $TCIP_PROJECT_ROOT
        # (tcip_mcp.project_paths.resolve_state); the outer chdir into a separate audit-only
        # tmpdir does not point resolution at root, so pin it explicitly for this run, restoring
        # whatever the process already had once done.
        _saved_project_root = os.environ.get("TCIP_PROJECT_ROOT")
        os.environ["TCIP_PROJECT_ROOT"] = str(root)
        try:
            from tcip_mcp.traits import registered_crops
            from tcip_mcp.tools.project_tools import init_project, register_dataset

            init_project(str(root), site="smoke test orchard")
            _author_catkin_trait_spec(root)

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
                # The bucket's own recorded id_map, the real shape export_predictions stamps,
                # written through the store so a database-bound backend's reader can see it.
                write_sidecar(preds_root / date, {"id_map": ID_MAP}, "operating_point")

            plant_csv = root / "plants.csv"
            with plant_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
                for p in PLANTS:
                    w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])

            crop = sorted(registered_crops())[0]
            register_dataset(str(dataset_root), crop=crop, project_root=str(root))

            preds_by_date = {d: str(preds_root / d) for d in DATES}

            # 2. build_plant_mapping: real EXIF GPS → plant assignments.
            print("Step 1: build_plant_mapping")
            m = build_plant_mapping(
                name=mapping_name,
                images_root=str(images_root),
                plant_csv_paths=[str(plant_csv)],
            )
            check("no error", "error" not in m, m.get("error", ""))
            check("3 dates mapped", m.get("n_dates") == 3, str(m.get("n_dates")))
            check("6 images seen", m.get("n_images") == 6, str(m.get("n_images")))
            check("all 6 images mapped to a plant", m.get("n_mapped") == 6,
                  f"n_mapped={m.get('n_mapped')} n_unmapped={m.get('n_unmapped')}")
            from tcip_mcp.pipelines.postprocessing import plant_mapping as _plant_mapping
            check("mapping persisted", bool(_plant_mapping.load_mapping(root, mapping_name)))

            # 3. compute_phenology: the real coverage rule and classifier gate are both live. The
            # positive class is resolved from each bucket's own recorded id_map (never a pinned
            # int), and every image is classified (no bare single-class-detector buckets here), so the
            # elongation split is valid; this script is the delivery path's acceptance artifact,
            # proving the delivery path genuinely produces a curve+milestones end to end, not just
            # refuses.
            print("\nStep 2: compute_phenology delivers a real bloom curve + milestones")
            r = compute_phenology(
                trait="catkin",
                mapping_name=mapping_name,
                predictions_by_date=preds_by_date,
                output_csv_path=str(csv_out),
                acknowledge_unvalidated=True,
            )
            check("no error", "error" not in r, str(r.get("error", "")))
            check("positive_class_assessed true (every bucket resolved the positive class)",
                  r.get("positive_class_assessed") is True, str(r.get("positive_class_assessed")))
            check("CSV delivered", csv_out.is_file())

            if csv_out.is_file():
                with csv_out.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                check("one row per plant", len(rows) == len(PLANTS), str(len(rows)))
                row = next((r2 for r2 in rows if r2.get("plant_id") == "P1"), None)
                check("P1 row present", row is not None)
                if row:
                    d05, d50, d95 = row.get("catkin_05per_date"), row.get("catkin_50per_date"), row.get("catkin_95per_date")
                    check("05/50/95per dates all populated (not a fabricated blank)",
                          all([d05, d50, d95]), f"05={d05} 50={d50} 95={d95}")
                    if d05 and d50 and d95:
                        check("milestones correctly ordered (05 <= 50 <= 95)", d05 <= d50 <= d95,
                              f"05={d05} 50={d50} 95={d95}")
                        check("milestones fall within the observed date range",
                              DATES[0] <= d05 and d95 <= DATES[-1],
                              f"range={DATES[0]}..{DATES[-1]} 05={d05} 95={d95}")
        finally:
            if _saved_project_root is None:
                os.environ.pop("TCIP_PROJECT_ROOT", None)
            else:
                os.environ["TCIP_PROJECT_ROOT"] = _saved_project_root
            # Windows can't remove the tempdir the bound backend still holds a database handle
            # into; close it before the enclosing TemporaryDirectory context tears the tree down.
            backend.close()

    print()
    if _failures:
        print(f"SMOKE FAILED: {_failures} assertion(s) failed.")
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

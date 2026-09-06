"""Live e2e smoke: the agent's phenology pipeline on real geolocated images.

Builds a synthetic scene: geolocated JPEGs (real EXIF GPS + capture time) across three
dates plus a plant-locations CSV, then runs the exact tools and web route the agent and the
breeder compose between them:

    build_plant_mapping        images + CSV                 -> a named mapping under the project
    deliver_phenology_milestones  mapping + classified preds -> refuses (no MCP surface acknowledges)
    /api/results/export_csv    the same mapping + preds, acknowledged -> <trait>_phenology.csv

covering the seam unit tests don't: real EXIF -> real GPS matching -> a real per-plant curve and
milestones delivered end to end through the writer, the delivery gate, the delivered tail and the
delivery event, driven through the web app's own route with a TestClient rather than a second,
hand-rolled measurement. No served backend or network.

Usage (repo root, tcip-agent env):
    python tools/smoke_phenology_e2e.py

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_SRC = [_REPO_ROOT / "packages" / pkg / "src" for pkg in
           ("tcip-store", "tcip-annotation", "tcip-mcp", "tcip-web")]
for _path in (*_PKG_SRC, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from tcip_mcp.pipelines.postprocessing.export import write_predictions_json  # noqa: E402
from tcip_mcp.pipelines.resolution import operating_point_stamp, write_sidecar  # noqa: E402
from tcip_mcp.tools.phenology_tools import (  # noqa: E402
    build_plant_mapping,
    deliver_phenology_milestones,
    register_plant_registry,
)
# Imported before main()'s own bind_default(): this import binds its own backend, so the
# explicit bind_default() below must run after it to be the one every call below uses.
from tcip_web.app import app  # noqa: E402
from tcip_web.state import store  # noqa: E402

class _Plant(TypedDict):
    plot: str
    accession: str
    lat: float
    lon: float


# Two plants ~3 m apart in one row (1 deg lon ≈ 81 km at 43°N, so 3.7e-5 deg ≈ 3 m).
PLANTS: list[_Plant] = [
    {"plot": "P1", "accession": "acc-A", "lat": 43.19670, "lon": -90.058000},
    {"plot": "P2", "accession": "acc-B", "lat": 43.19670, "lon": -90.058037},
]
# Three capture dates; the open fraction rises 0.0 → 0.4 → 1.0.
DATES = ["2026-02-11", "2026-02-25", "2026-03-11"]
FRACTIONS = {"2026-02-11": 0.0, "2026-02-25": 0.4, "2026-03-11": 1.0}
# The bucket's own recorded id_map: the positive class is resolved from this on disk, never a
# pinned integer. Keyed by opening's own value names; every detection here is one subject.
SUBJECT = "bud"
ATTRIBUTE = "opening"
ID_MAP = {"closed": 0, "open": 1}
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


def _pred_result(n_open: int, n_total: int, *, width: int, height: int) -> dict:
    """One image's raw predictor-shaped result: n_total detections of one subject, the first
    n_open one-indexed to id_map['open'], the rest to id_map['closed'].

    write_predictions_json decodes each label through this run's own recorded id_map into the
    shape ground truth carries: the object class in every record's subject, the decoded value
    under attributes[attribute], never the value alone written straight into subject.
    """
    n_closed = n_total - n_open
    labels = [ID_MAP["open"] + 1] * n_open + [ID_MAP["closed"] + 1] * n_closed
    return {
        "width": width, "height": height,
        "boxes": [[1.0, 1.0, 3.0, 3.0] for _ in range(n_total)],
        "scores": [0.90] * n_total,
        "labels": labels,
    }


def _stem(plot: str, date: str) -> str:
    return f"{plot}_{date.replace('-', '')}"


def _author_bud_opening_trait_spec(root: Path) -> None:
    """Register the bud_opening trait under ``root`` and record a confirmed meaning for its delivery.

    The spec is ``tests/_trait_fixtures.BUD_OPENING`` itself rather than a copy of its field values, so
    a change to that definition cannot leave this smoke run exercising a stale one. The crossing
    door refuses a trait whose delivered number has no breeder-confirmed meaning, so this states
    one and confirms it through the same two writers a real project goes through; it also declares
    the confirmed positive class in ``root``'s own class registry, which the web export route (but
    not the MCP tool) requires reachable from the delivered dataset's own root.
    """
    import tcip_store as ts

    from tcip_mcp import traits
    from tests._operationalization_fixtures import seed_confirmed_crossing
    from tests._trait_fixtures import BUD_OPENING

    data = traits._encode_spec(BUD_OPENING)
    key = traits.trait_spec_key(traits.trait_specs_dir(root), data["name"])
    spec, reason = traits._validate_and_write_spec(key, data, expect=ts.Version.ABSENT)
    if spec is None:
        raise ValueError(f"the fixture trait spec does not clear crops.yml: {reason}")
    seed_confirmed_crossing(root, data["name"])


def main() -> int:
    # Its own process entry point, so it binds the storage backend the seam has no default for.
    from tcip_store.binding import bind_default

    backend = bind_default()

    print("Phenology e2e smoke: build_plant_mapping -> deliver_phenology_milestones -> "
          "acknowledged web export\n")
    with tempfile.TemporaryDirectory() as td:
        # A workspace/project split, its name fitting crop_subject_phenotype
        # (initialize_project holds a workspace project to that scheme).
        workspace_root = Path(td)
        root = workspace_root / "currant_bud_phenology"
        root.mkdir(parents=True)
        dataset_root = root / "dataset"        # a registered dataset the mapping is built over
        images_root = dataset_root / "images"
        preds_root = dataset_root / "predictions" / "live"  # class-carrying predictions (valid)
        mapping_name = "smoke-valley"

        # Both pinned explicitly for this run and restored after: state resolution otherwise
        # follows the process cwd, and the web app refuses a TestClient request with no workspace.
        _saved_platform_root = os.environ.get("TCIP_STATE_ROOT")
        _saved_workspace = os.environ.get("TCIP_WORKSPACE")
        os.environ["TCIP_STATE_ROOT"] = str(root)
        os.environ["TCIP_WORKSPACE"] = str(workspace_root)
        try:
            from tcip_mcp.traits import registered_crops
            from tcip_mcp.tools.project_tools import initialize_project, register_dataset

            init = initialize_project(str(root), site="smoke test orchard")
            check("project initialized", "error" not in init, init.get("error", ""))
            _author_bud_opening_trait_spec(root)

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
                    # created_by=None: prediction_producer refuses without a real checkpoint
                    # digest, and this scene has no checkpoint behind it to name one from.
                    write_predictions_json(
                        preds_root / date / f"{stem}.json",
                        _pred_result(n_elong, N_DETECTIONS, width=8, height=8), None,
                        subject=SUBJECT, attribute=ATTRIBUTE, id_map=ID_MAP,
                    )
                # The bucket's own recorded scope and id_map, the shape run_inference stamps,
                # written through the store so a database-bound backend's reader can see it.
                stamp = operating_point_stamp(
                    {"conf": {"value": 0.5, "source": "default"}},
                    validated=False, validated_by=None, tile_size_validated=None,
                    shippable_issues=[], id_map=ID_MAP, subject=SUBJECT, attribute=ATTRIBUTE,
                    trait=None, dataset_hash=None, checkpoint=None, checkpoint_sha256=None,
                    experiment_id=None, images_dir=str(images_root / date), raster_path=None,
                    produced_at=datetime.now(timezone.utc).isoformat(),
                )
                write_sidecar(preds_root / date, stamp, "operating_point")

            plant_csv = root / "plants.csv"
            with plant_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["plot_name", "accession_name", "WGS84_centroid_x", "WGS84_centroid_y"])
                for p in PLANTS:
                    w.writerow([p["plot"], p["accession"], p["lon"], p["lat"]])

            crop = sorted(registered_crops())[0]
            register_dataset(str(dataset_root), crop=crop, project_root=str(root))
            # The web export route resolves the class registry from the delivered dataset's own
            # root, never the project's; the MCP tool never checks one (it reads the bucket's id_map).
            from tcip_mcp.class_registry import copy_registry
            from tcip_mcp.dataset_layout import classes_path

            copy_registry(classes_path(root), classes_path(dataset_root))

            preds_by_date = {d: str(preds_root / d) for d in DATES}

            registry_name = "smoke-valley-plants"
            reg = register_plant_registry(
                name=registry_name, csv_paths=[str(plant_csv)], crop=crop,
                site="smoke test orchard")
            check("no error", "error" not in reg, reg.get("error", ""))

            # 2. build_plant_mapping: real EXIF GPS → plant assignments.
            print("Step 1: build_plant_mapping")
            m = build_plant_mapping(
                name=mapping_name,
                images_root=str(images_root),
                plant_registry=registry_name,
            )
            check("no error", "error" not in m, m.get("error", ""))
            check("3 dates mapped", m.get("n_dates") == 3, str(m.get("n_dates")))
            check("6 images seen", m.get("n_images") == 6, str(m.get("n_images")))
            check("all 6 images mapped to a plant", m.get("n_mapped") == 6,
                  f"n_mapped={m.get('n_mapped')} n_unattributed={m.get('n_unattributed')}")
            from tcip_mcp.pipelines.postprocessing import plant_mapping as _plant_mapping
            check("mapping persisted", bool(_plant_mapping.load_mapping(root, mapping_name)))

            # No MCP tool takes an acknowledgement; only the Results tab's export does.
            # This bucket's sidecar was never validated, so the door refuses unconditionally.
            print("\nStep 2: deliver_phenology_milestones refuses an unacknowledgeable unvalidated delivery")
            csv_out = root / "delivery" / "bud_phenology.csv"
            r = deliver_phenology_milestones(
                trait="bud_opening",
                mapping_name=mapping_name,
                predictions_by_date=preds_by_date,
                output_csv_path=str(csv_out),
            )
            check("refused (no acknowledgement route exists on this tool)", "error" in r, str(r))
            # The tool refuses on the positive class before ever reaching the classifier gate, so
            # a classifier-gate refusal (not the earlier one) proves every bucket resolved it.
            check("positive_class_assessed true (else the tool would refuse on the class instead)",
                  "validated positive-state classifier" in r.get("error", ""), str(r))
            check("no CSV written on refusal", not csv_out.is_file())

            # 3. The breeder's own route: look at the unvalidated numbers, then acknowledge and
            # export them, exercising the writer, the gate, the tail and the delivery event.
            print("\nStep 3: the web export route delivers an acknowledged, unvalidated CSV")
            store.open_project(root.resolve())
            client = TestClient(app, base_url="http://127.0.0.1")
            body = {
                "project_root": str(root), "mapping_name": mapping_name,
                "predictions_by_date": preds_by_date, "trait": "bud_opening",
            }

            screen = client.post(
                "/api/results/phenology_measurement", json={**body, "show_unvalidated": True})
            check("phenology_measurement looks at the unvalidated numbers (200)",
                  screen.status_code == 200, screen.text)
            if screen.status_code == 200:
                screen_body = screen.json()
                check("has_unvalidated_dimensions true (nothing on disk was ever validated)",
                      screen_body.get("has_unvalidated_dimensions") is True)
                milestone_rows = screen_body.get("milestones", {}).get("rows", [])
                check("one milestone row per plant", len(milestone_rows) == len(PLANTS),
                      str(len(milestone_rows)))
                row = next((r2 for r2 in milestone_rows if r2.get("plant_id") == "P1"), None)
                check("P1 row present", row is not None)
                if row:
                    d05 = row.get("bud_05per_date")
                    d50 = row.get("bud_50per_date")
                    d95 = row.get("bud_95per_date")
                    check("05/50/95per dates all populated (not a fabricated blank)",
                          all([d05, d50, d95]), f"05={d05} 50={d50} 95={d95}")
                    if d05 and d50 and d95:
                        check("milestones correctly ordered (05 <= 50 <= 95)", d05 <= d50 <= d95,
                              f"05={d05} 50={d50} 95={d95}")
                        check("milestones fall within the observed date range",
                              DATES[0] <= d05 and d95 <= DATES[-1],
                              f"range={DATES[0]}..{DATES[-1]} 05={d05} 95={d95}")

            export = client.post("/api/results/export_csv", json={
                **body, "payload": "milestones", "filename": "bud_phenology_ack.csv",
                "user": "user:smoketest",
                "acknowledgement": {"reason": "smoke run over an uncalibrated scene"},
            })
            check("export_csv delivers the acknowledged CSV (200)",
                  export.status_code == 200, export.text)
            if export.status_code == 200:
                saved_to = export.headers.get("X-TCIP-Saved-To", "")
                check("the delivery landed under the project's own results_export/",
                      saved_to == str(root / "results_export" / "bud_phenology_ack.csv"),
                      saved_to)
                delivered_rows = list(csv.DictReader(io.StringIO(export.text)))
                check("the CSV carries the delivered rows", bool(delivered_rows))
                if delivered_rows:
                    cell = delivered_rows[0]
                    check("acknowledged_by carries the acting user",
                          cell.get("acknowledged_by") == "user:smoketest", str(cell))
                    check("acknowledgement_reason carries the stated reason",
                          cell.get("acknowledgement_reason") == "smoke run over an uncalibrated scene",
                          str(cell))
                    check("operating_point_validated stamps false (never silently upgraded)",
                          cell.get("operating_point_validated") == "false", str(cell))

            events = client.get(
                "/api/results/delivery-events", params={"project_root": str(root)}).json()
            record = next(
                (e for e in events.get("records", []) if e.get("door") == "results.export_csv"),
                None)
            check("the delivery event names the acknowledging user", record is not None)
            if record:
                check("the delivery event carries acknowledged_by",
                      record.get("acknowledged_by") == "user:smoketest", str(record))
                check("the delivery event carries the reason",
                      record.get("acknowledgement_reason") == "smoke run over an uncalibrated scene",
                      str(record))
        finally:
            if _saved_platform_root is None:
                os.environ.pop("TCIP_STATE_ROOT", None)
            else:
                os.environ["TCIP_STATE_ROOT"] = _saved_platform_root
            if _saved_workspace is None:
                os.environ.pop("TCIP_WORKSPACE", None)
            else:
                os.environ["TCIP_WORKSPACE"] = _saved_workspace
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

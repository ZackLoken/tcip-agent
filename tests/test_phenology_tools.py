"""Tests for the phenology MCP tools (build_plant_mapping + compute_phenology).

The tools are the agent-facing surface for the per-plant bloom pipeline. These tests pin:
(1) build_plant_mapping wraps build + persist and reports a compact summary + error paths;
(2) compute_phenology writes the canonical column schema from classified predictions + a
persisted plant mapping; and (3) its measurement-integrity guard refuses to deliver a CSV
when the predictions carry no elongation class.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from tcip_mcp.pipelines.postprocessing import phenology
from tcip_mcp.tools.phenology_tools import build_plant_mapping, compute_phenology


def _plant_csv(path: Path) -> None:
    path.write_text(
        "plot_name,accession_name,WGS84_centroid_x,WGS84_centroid_y\n"
        "P1,acc-9,-90.058,43.197\n",
        encoding="utf-8",
    )


def test_build_plant_mapping_wraps_build_and_persists(tmp_path: Path) -> None:
    # One date folder with a plain (no-GPS) image + a plant CSV. The matcher can't map a
    # GPS-less image, so it lands unmapped — but the tool must still build, persist, and
    # summarize correctly. (Sequence/NN matching itself is covered in test_plant_mapping.)
    images_root = tmp_path / "images"
    (images_root / "2026-02-11").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(images_root / "2026-02-11" / "img1.jpg")
    csv_path = tmp_path / "plants.csv"
    _plant_csv(csv_path)
    out = tmp_path / "state" / "plant_mapping.json"

    res = build_plant_mapping(
        images_root=str(images_root),
        plant_csv_paths=[str(csv_path)],
        output_mapping_path=str(out),
    )

    assert "error" not in res
    assert res["mapping_path"] == str(out)
    assert res["n_dates"] == 1
    assert res["n_images"] == 1
    assert res["n_mapped"] + res["n_unmapped"] == 1
    assert "2026-02-11" in res["per_date"]
    assert out.is_file()
    # Persisted JSON is loadable and shaped as {date: [assignment, ...]}.
    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert list(persisted.keys()) == ["2026-02-11"]
    assert persisted["2026-02-11"][0]["stem"] == "img1"
    # The purged 'confidence' field must not reappear via the pipeline.
    assert "confidence" not in persisted["2026-02-11"][0]


def test_build_plant_mapping_missing_images_root(tmp_path: Path) -> None:
    res = build_plant_mapping(
        images_root=str(tmp_path / "nope"),
        plant_csv_paths=[str(tmp_path / "plants.csv")],
        output_mapping_path=str(tmp_path / "m.json"),
    )
    assert "error" in res
    assert "images_root not found" in res["error"]


def test_build_plant_mapping_missing_csv(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    (images_root / "2026-02-11").mkdir(parents=True)
    res = build_plant_mapping(
        images_root=str(images_root),
        plant_csv_paths=[str(tmp_path / "missing.csv")],
        output_mapping_path=str(tmp_path / "m.json"),
    )
    assert "error" in res
    assert "plant CSV" in res["error"]


def _write_mapping(path: Path, mapping: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping), encoding="utf-8")


def _write_preds(dir_path: Path, stem: str, lines: list[str]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compute_phenology_writes_canonical_csv(tmp_path: Path) -> None:
    d1, d2 = tmp_path / "2026-02-11", tmp_path / "2026-03-09"
    # P1: 0/1 elongated early, 1/1 elongated late → fraction 0.0 → 1.0.
    _write_preds(d1, "P1_a", ["0 0.9 0.5 0.5 0.1 0.1"])
    _write_preds(d2, "P1_b", ["1 0.9 0.5 0.5 0.1 0.1"])
    mapping_path = tmp_path / "state" / "plant_mapping.json"
    _write_mapping(
        mapping_path,
        {
            "2026-02-11": [
                {"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}
            ],
            "2026-03-09": [
                {"stem": "P1_b", "plot_name": "P1", "accession_name": "acc-9"}
            ],
        },
    )
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1), "2026-03-09": str(d2)},
        output_csv_path=str(out_csv),
        elongated_class_id=1,
    )

    assert "error" not in res
    assert res["elongation_classified"] is True
    assert res["n_plants"] == 1
    assert res["columns"] == phenology.PHENOLOGY_CSV_COLUMNS
    assert out_csv.is_file()

    with out_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == phenology.PHENOLOGY_CSV_COLUMNS
    assert rows[0]["plant_id"] == "P1"
    assert rows[0]["accession"] == "acc-9"
    # onset is the first non-zero fraction → the late date; the CSV must not carry
    # the internal 'series' column.
    assert rows[0]["catkin_elongation_date"] == "2026-03-09"
    assert "series" not in rows[0]


def test_compute_phenology_refuses_unclassified_predictions(tmp_path: Path) -> None:
    # Predictions carry only class 0 — no elongation class. The tool must NOT write a
    # CSV and must flag the measurement as invalid.
    d1 = tmp_path / "2026-02-11"
    _write_preds(d1, "P1_a", ["0 0.9 0.5 0.5 0.1 0.1"])
    mapping_path = tmp_path / "plant_mapping.json"
    _write_mapping(
        mapping_path,
        {"2026-02-11": [{"stem": "P1_a", "plot_name": "P1", "accession_name": "acc-9"}]},
    )
    out_csv = tmp_path / "out" / "catkin_phenology.csv"

    res = compute_phenology(
        mapping_path=str(mapping_path),
        predictions_by_date={"2026-02-11": str(d1)},
        output_csv_path=str(out_csv),
        elongated_class_id=1,
    )

    assert "error" in res
    assert res["elongation_classified"] is False
    assert res["classes_seen"] == [0]
    assert not out_csv.exists()  # nothing delivered


def test_compute_phenology_missing_mapping(tmp_path: Path) -> None:
    res = compute_phenology(
        mapping_path=str(tmp_path / "nope.json"),
        predictions_by_date={},
        output_csv_path=str(tmp_path / "out.csv"),
    )
    assert "error" in res
    assert "not found" in res["error"]

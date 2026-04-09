"""CSV export for per-plant phenotyping results."""

from __future__ import annotations

import csv
from pathlib import Path


def export_phenology_csv(
    results: list[dict],
    output_path: str,
    trait_prefix: str = "catkin",
) -> str:
    """Export per-plant phenology results to CSV.

    Args:
        results: Output from aggregate_per_plant().
        output_path: Path for the output CSV file.
        trait_prefix: Prefix for column names (e.g. 'catkin').

    Returns:
        Path to the written CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "plant_id",
        "observations",
        "max_count",
        f"{trait_prefix}_05per_date",
        f"{trait_prefix}_50per_date",
        f"{trait_prefix}_95per_date",
        "fit_steepness",
        "fit_error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            fit = r.get("fit", {})
            milestones = fit.get("milestones", {})
            row = {
                "plant_id": r["plant_id"],
                "observations": r["observations"],
                "max_count": r["max_count"],
                f"{trait_prefix}_05per_date": milestones.get("05per", ""),
                f"{trait_prefix}_50per_date": milestones.get("50per", ""),
                f"{trait_prefix}_95per_date": milestones.get("95per", ""),
                "fit_steepness": fit.get("steepness", ""),
                "fit_error": fit.get("error", ""),
            }
            writer.writerow(row)

    return output_path


def export_detection_csv(
    image_results: list[dict],
    output_path: str,
) -> str:
    """Export per-image detection counts to CSV.

    Args:
        image_results: List of dicts with 'image', 'count', 'boxes', etc.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image", "detection_count", "avg_confidence"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in image_results:
            scores = r.get("scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            writer.writerow({
                "image": Path(r.get("image", "")).name,
                "detection_count": r.get("count", len(r.get("boxes", []))),
                "avg_confidence": round(avg_conf, 4),
            })

    return output_path

"""GUARDS proofs, kept in their own file with the narrowest possible import set: neither test
imports ``register_plant_registry`` itself, only the two doors whose signature changed, so
``prove_test_fails_before.py`` can revert just ``phenology_tools.py``/``orthomosaic_tools.py``
to the pre-registry baseline without the module failing to collect on an import the baseline
predates entirely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts
from tcip_mcp.tools.phenology_tools import build_plant_mapping


def test_build_plant_mapping_no_longer_accepts_plant_csv_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUARDS: the door's signature dropped plant_csv_paths for plant_registry."""
    from tests.test_plant_mapping_binding import _init

    _init(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        build_plant_mapping(  # type: ignore[call-arg]
            name="valley", images_root=str(tmp_path), plant_csv_paths=["nope.csv"])


def test_deliver_orthomosaic_plant_counts_no_longer_accepts_plant_csv_paths() -> None:
    """GUARDS: the door's signature dropped plant_csv_paths for plant_registry."""
    with pytest.raises(TypeError):
        deliver_orthomosaic_plant_counts(  # type: ignore[call-arg]
            predictions_dir="preds", raster_path="raster.tif", plant_csv_paths=["nope.csv"],
            output_csv_path="out.csv", delivered_phenotype="stem_count")

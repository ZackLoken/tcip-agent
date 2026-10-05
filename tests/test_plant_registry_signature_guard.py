"""The two delivery doors that take a named plant registry, held to their signatures.

Nothing here imports ``register_plant_registry``: the module reaches only
``phenology_tools.py`` and ``orthomosaic_tools.py``, so it collects against any tree that
carries those two.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.tools.orthomosaic_tools import deliver_orthomosaic_plant_counts
from tcip_mcp.tools.phenology_tools import build_plant_mapping


def test_build_plant_mapping_refuses_plant_csv_paths_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard: the door's signature takes plant_registry, not plant_csv_paths."""
    from tests.test_plant_mapping_binding import _init

    _init(tmp_path)
    with pytest.raises(TypeError, match="plant_csv_paths"):
        build_plant_mapping(  # type: ignore[call-arg]
            tmp_path, name="valley", images_root=str(tmp_path), plant_csv_paths=["nope.csv"])


def test_deliver_orthomosaic_plant_counts_refuses_plant_csv_paths_argument(tmp_path: Path) -> None:
    """A guard: the door's signature takes plant_registry, not plant_csv_paths."""
    with pytest.raises(TypeError, match="plant_csv_paths"):
        deliver_orthomosaic_plant_counts(  # type: ignore[call-arg]
            tmp_path, dataset_root=str(tmp_path), bucket="preds/2026-01-01",
            plant_csv_paths=["nope.csv"],
            output_csv_path="out.csv", delivered_phenotype="stem_count", plants=[])

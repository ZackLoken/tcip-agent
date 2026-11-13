"""A run records the dataset identity it launched against in its launch record.

The id and fingerprint are written into the run's launch record when it is opened, and
compare_experiments reports whether two runs share a dataset fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

import tcip_store as ts
from tcip_mcp.dataset_layout import image_dir
from tcip_mcp.experiments import compare_experiments
from tests._chain_fixtures import (
    BUILT_DETECTOR,
    FRAME_STATISTICS,
    training_config,
)
from tests._training_values import drawn_split
from tests._verified_checkpoint_fixtures import opened_run
from tests.test_typed_store_records import STATES_NO_ID, _stating

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fastapi.testclient import TestClient

    from tcip_mcp.pipelines.data.selection import Sample

SUBJECT = "bud"


def _make_dataset(root: Path, *, shade: int = 0) -> Path:
    """Write two labeled images under one capture date and return their images directory.

    Image ``i`` is a ``labeled_frame`` on the ground ``(shade, 10 * i, 0)``.
    """
    from tcip_annotation.state import Annotation, BBox
    from tests._producer_fixtures import (
        labeled_frame,
        registry_of,
        registry_over,
    )

    images_dir = image_dir(root, "2-11-26")
    images_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        labeled_frame(
            images_dir / f"img_{i:03d}.jpg",
            [Annotation(subject=SUBJECT, geometry=BBox(1, 1, 9, 9))],
            64,
            64,
            (shade, 10 * i, 0),
        )
    registry_over(root, registry_of(SUBJECT))
    return images_dir


def _config(project: Path, images_dir: Path) -> dict:
    """Return a :data:`BUILT_DETECTOR` run's config of ``project`` over ``images_dir``."""
    return training_config(
        project,
        BUILT_DETECTOR,
        {
            "images_dir": str(images_dir),
            "scope": {"subject": SUBJECT},
            "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
        },
    )


def _fingerprints(ids: list[str], project: Path) -> list[str | None]:
    """Return the dataset fingerprint each compared run of ``ids`` carries, in order."""
    fingerprints = []
    for entry in compare_experiments(ids, project=project).experiments:
        assert entry.kind == "compared", entry
        fingerprints.append(entry.dataset.fingerprint)
    return fingerprints


def _open_loose_run(project: Path, experiment_id: str) -> Path:
    """Open a classifier run over images under no dataset root; return its directory."""
    from tests._verified_checkpoint_fixtures import table_images
    from tests.tiny_trainer_fixtures import (
        MEAN_INTENSITY_CLASSIFIER,
        TINY_TRAINER_FILE,
    )

    return opened_run(
        project,
        training_config(
            project,
            {
                "builder": MEAN_INTENSITY_CLASSIFIER,
                "source_files": [TINY_TRAINER_FILE],
                "task": "classification",
                "normalization": FRAME_STATISTICS,
            },
            {
                **table_images(project / "loose"),
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        ),
        experiment_id=experiment_id,
    )


def test_compare_experiments_surfaces_shared_fingerprint(
    tmp_path: Path,
) -> None:
    first = _config(tmp_path, _make_dataset(tmp_path / "first"))
    opened_run(tmp_path, first, experiment_id="a")
    opened_run(tmp_path, first, experiment_id="b")
    a, b = _fingerprints(["a", "b"], tmp_path)
    assert a is not None and a == b
    opened_run(
        tmp_path,
        _config(tmp_path, _make_dataset(tmp_path / "second", shade=200)),
        experiment_id="c",
    )
    a, c = _fingerprints(["a", "c"], tmp_path)
    assert None not in (a, c) and a != c


def test_compare_experiments_mixed_none_fingerprint_is_unknown_not_same(
    tmp_path: Path,
) -> None:
    """Comparing a run of known fingerprint with one of none carries the none, not a match.

    The second run is over a dataset holding no label document, and its column carries no
    fingerprint.
    """
    opened_run(
        tmp_path,
        _config(tmp_path, _make_dataset(tmp_path / "first")),
        experiment_id="a",
    )
    _open_loose_run(tmp_path, "b")
    a, b = _fingerprints(["a", "b"], tmp_path)
    assert a is not None and b is None


def test_dataset_identity_helper_registered_vs_bespoke(tmp_path: Path) -> None:
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec
    from tcip_mcp.tools.project_tools import register_dataset

    _make_dataset(tmp_path)
    reg = register_dataset(tmp_path, str(tmp_path), crop="currant", actor=None)
    ds_id, fp = dataset_identity(
        DataSpec.model_validate(
            {
                "images_dir": str(image_dir(tmp_path, "2-11-26")),
                "split": drawn_split(),
            }
        )
    )
    assert ds_id == reg["id"] and fp == reg["fingerprint"]
    # bespoke / imageless run -> no fabricated identity
    assert dataset_identity(
        DataSpec.model_validate({"split": drawn_split()})
    ) == (None, None)


def test_dataset_identity_raises_a_fingerprint_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fingerprint read failure raises out of ``dataset_identity`` by name.

    A fingerprint read raising ``OSError`` mid-scan raises that error out of the call.
    """
    import tcip_mcp.pipelines.data.dataset_fingerprint as dataset_fingerprint_mod
    from tcip_mcp.pipelines.data.split_construction import dataset_identity
    from tcip_mcp.pipelines.schemas import DataSpec
    from tests._producer_fixtures import register_dataset_at

    _make_dataset(tmp_path)
    register_dataset_at(tmp_path, tmp_path, crop="currant")

    def _raise(_root: str | Path, _admitted: Iterable[Sample]) -> NoReturn:
        raise OSError("simulated I/O error mid-scan")

    # dataset_identity reads the fingerprint through its own module at call time, so it is
    # patched at the source module.
    monkeypatch.setattr(dataset_fingerprint_mod, "dataset_fingerprint", _raise)
    with pytest.raises(OSError, match="simulated I/O error"):
        dataset_identity(
            DataSpec.model_validate(
                {
                    "images_dir": str(image_dir(tmp_path, "2-11-26")),
                    "split": drawn_split(),
                }
            )
        )


def test_a_trial_over_a_data_axis_records_the_dataset_its_own_input_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep trial over a data axis records the identity of the dataset its own input names.

    A sweep trial whose sampled point names another dataset records that dataset's id and
    fingerprint.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_run
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.tools.training_tools import open_trial
    from tests._verified_checkpoint_fixtures import opened_sweep

    base_images = _make_dataset(tmp_path / "base")
    register_dataset(
        tmp_path / "base", str(tmp_path / "base"), crop="currant", actor=None
    )
    other_images = _make_dataset(tmp_path / "other", shade=200)
    other = register_dataset(
        tmp_path / "other", str(tmp_path / "other"), crop="currant", actor=None
    )

    trial_dir = open_trial(
        opened_sweep(tmp_path, _config(tmp_path, base_images)),
        "a",
        {"data.images_dir": str(other_images)},
    )

    dataset = read_run(trial_dir).dataset
    assert (dataset.id, dataset.fingerprint) == (
        other["id"],
        other["fingerprint"],
    )


def test_a_launch_records_the_identity_of_the_dataset_it_trains_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launch records the identity of the dataset its config names.

    The run's launch record carries the registered dataset's id and fingerprint.
    """
    pytest.importorskip("torch")
    from tcip_mcp.experiments import read_run
    from tcip_mcp.tools.project_tools import register_dataset
    from tcip_mcp.tools.training_tools import RunLaunched, launch_training
    from tests._producer_fixtures import fake_popen

    fake_popen(monkeypatch, [])

    images_dir = _make_dataset(tmp_path)
    registered = register_dataset(
        tmp_path, str(tmp_path), crop="currant", actor=None
    )

    launched = launch_training(
        tmp_path,
        training_config(
            tmp_path,
            BUILT_DETECTOR,
            {
                "images_dir": str(images_dir),
                "scope": {"subject": SUBJECT},
                "split": drawn_split(val_ratio=0.15, grouping="tile_prefix"),
            },
        ),
        actor=None,
    )
    assert isinstance(launched, RunLaunched), launched

    dataset = read_run(Path(launched.output_dir)).dataset
    assert (dataset.id, dataset.fingerprint) == (
        registered["id"],
        registered["fingerprint"],
    )


def _runs_over_each_identity_state(project: Path) -> tuple[list[str], str]:
    """Open a run over an unregistered root, a registered one and none; return their ids and the id.

    The ids are ``"a"`` (unregistered, so a fingerprint beside no id), ``"b"`` (registered) and
    ``"c"`` (images under no dataset root, so neither); the string is the id ``b``'s root was
    registered with.
    """
    from tcip_mcp.tools.project_tools import register_dataset

    unregistered = _make_dataset(project / "first")
    registered_images = _make_dataset(project / "second")
    registered = register_dataset(
        project / "second", str(project / "second"), crop="currant", actor=None
    )
    opened_run(project, _config(project, unregistered), experiment_id="a")
    opened_run(project, _config(project, registered_images), experiment_id="b")
    _open_loose_run(project, "c")
    return ["a", "b", "c"], registered["id"]


def test_the_comparison_and_the_lineage_serve_the_dataset_the_run_record_serves(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A run's comparison row and lineage carry the dataset its record serves, as one object.

    Three runs are opened by the launcher's producers, over an unregistered root, a registered
    root and images under no dataset root. The comparison route's row, the lineage view of
    ``get_experiment`` and its full view's run record state equal ``dataset`` objects; the
    third run's id and fingerprint are both null.
    """
    from tcip_mcp.tools.experiment_tools import get_experiment

    ids, registered_id = _runs_over_each_identity_state(opened_project)

    answer = opened_client.post(
        "/api/training/compare", json={"experiment_ids": ids}
    )
    assert answer.status_code == 200, answer.text
    served = []
    for row, eid in zip(answer.json()["experiments"], ids, strict=True):
        record = get_experiment(opened_project, eid)["run"]["dataset"]
        lineage = get_experiment(opened_project, eid, view="lineage")[
            "lineage"
        ]["dataset"]
        assert row["dataset"] == record == lineage
        served.append(record)

    unregistered, registered, loose = served
    assert unregistered["id"] is None
    assert unregistered["fingerprint"] is not None
    assert registered["id"] == registered_id
    assert loose == {"id": None, "fingerprint": None}


def test_neither_the_comparison_nor_the_lineage_restates_the_dataset_fields(
    opened_client: TestClient, opened_project: Path
) -> None:
    """A comparison row and a lineage carry the run's dataset under ``dataset`` alone.

    The runs are opened as in the agreement test; the comparison route's JSON and the lineage
    view of ``get_experiment`` hold no ``dataset_id`` or ``dataset_fingerprint`` key.
    """
    from tcip_mcp.tools.experiment_tools import get_experiment

    ids, _ = _runs_over_each_identity_state(opened_project)

    answer = opened_client.post(
        "/api/training/compare", json={"experiment_ids": ids}
    )
    assert answer.status_code == 200, answer.text
    restated = {"dataset_id", "dataset_fingerprint"}
    for row in answer.json()["experiments"]:
        assert "dataset" in row
        assert restated.isdisjoint(row)
    for eid in ids:
        lineage = get_experiment(opened_project, eid, view="lineage")[
            "lineage"
        ]
        assert "dataset" in lineage
        assert restated.isdisjoint(lineage)


@STATES_NO_ID
def test_a_run_record_stating_no_dataset_id_reads_only_as_null(
    tmp_path: Path, stated: object
) -> None:
    """A run record whose dataset id is empty, only whitespace or absent refuses; null reads.

    The run is opened over a registered root by the launcher's producers and its ``run.json``
    then rewritten through the store's blob interface, a fault injected past its producer.
    """
    import json

    from tcip_mcp.experiments import RUN_FILE, read_run
    from tcip_mcp.tools.project_tools import register_dataset
    from tests.test_a_refused_record_reads_as_its_readers_sentence import (
        _rewritten,
    )

    images_dir = _make_dataset(tmp_path)
    register_dataset(tmp_path, str(tmp_path), crop="currant", actor=None)
    run_dir = opened_run(
        tmp_path, _config(tmp_path, images_dir), experiment_id="a"
    )
    path = run_dir / RUN_FILE
    dataset = json.loads(path.read_text(encoding="utf-8"))["dataset"]
    _rewritten(path, {"dataset": _stating(dataset, "id", stated)})

    if stated is None:
        assert read_run(run_dir).dataset.id is None
        return
    with pytest.raises(ts.ReadRefusedError, match=r"dataset\.id: ") as refused:
        read_run(run_dir)

    assert str(path) in str(refused.value), refused.value

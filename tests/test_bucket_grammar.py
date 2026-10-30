"""Which directories under ``images/`` read as buckets.

A date, ``UNDATED_BUCKET`` and a literal like a plot name are buckets; a dot-prefixed, empty or
path-separated name is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.dataset_layout import image_dir, list_dates


def test_is_bucket_name_admits_a_date_an_undated_token_and_a_literal() -> None:
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, is_bucket_name

    assert is_bucket_name("2026-03-02")
    assert is_bucket_name(UNDATED_BUCKET)
    assert is_bucket_name("plot_14")


def test_is_bucket_name_refuses_a_dot_prefixed_or_unsafe_name() -> None:
    from tcip_mcp.dataset_layout import is_bucket_name

    assert not is_bucket_name(".hidden")
    assert not is_bucket_name(".DS_Store")
    assert not is_bucket_name("..")
    assert not is_bucket_name("a/b")
    assert not is_bucket_name("")


def test_list_dates_excludes_a_hidden_directory(tmp_path: Path) -> None:
    image_dir(tmp_path, "2026-03-02").mkdir(parents=True)
    image_dir(tmp_path, ".DS_Store").mkdir(parents=True)
    assert list_dates(tmp_path) == ["2026-03-02"]


def test_list_dates_admits_a_literal_and_the_undated_bucket(
    tmp_path: Path,
) -> None:
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    for name in ("2026-03-02", UNDATED_BUCKET, "plot_14"):
        image_dir(tmp_path, name).mkdir(parents=True)
    assert list_dates(tmp_path) == sorted(
        ["2026-03-02", UNDATED_BUCKET, "plot_14"]
    )


def test_the_tree_and_the_project_summary_agree_on_dates_over_a_hidden_directory(
    client: TestClient, tmp_path: Path
) -> None:
    """The tree route and the project-picker summary agree on dates over a hidden directory.

    Beside a dated directory under ``images/``, a hidden one adds no date to either.
    """
    from tcip_mcp.project_record import record_or_problem
    from tcip_mcp.workspace import ListedProject
    from tcip_web.routes.projects import _summarize
    from tests._producer_fixtures import registry_of, registry_over

    root = tmp_path / "Valley_Farm"
    image_dir(root, "2026-03-02").mkdir(parents=True)
    image_dir(root, ".hidden").mkdir(parents=True)
    registry_over(root, registry_of("leaf"))

    tree = client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()
    picker = _summarize(ListedProject(root, *record_or_problem(root)))

    assert [c["name"] for c in tree["captures"]] == ["2026-03-02"]
    assert picker.tree.model_dump(mode="json") == tree


def test_the_tree_states_each_captures_date_and_none_where_no_date_names_it(
    client: TestClient, tmp_path: Path
) -> None:
    """A capture named by its ISO date is that date; the undated bucket and a no-date name none."""
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    root = tmp_path / "Valley_Farm"
    for capture in ("2026-03-02", "20260302", "2026-13-01", UNDATED_BUCKET):
        image_dir(root, capture).mkdir(parents=True)

    tree = client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()

    assert tree["captures"] == [
        {"name": "2026-03-02", "date": "2026-03-02"},
        {"name": "2026-13-01", "date": None},
        {"name": "20260302", "date": None},
        {"name": UNDATED_BUCKET, "date": None},
    ]


def test_a_publication_states_the_one_capture_its_documents_are_of(
    tmp_path: Path,
) -> None:
    """A bucket over one capture's images states that capture; one over two captures refuses."""
    pytest.importorskip("torch")
    from tcip_mcp.pipelines.postprocessing.export import DetectionResult
    from tests._chain_fixtures import published
    from tests._producer_fixtures import write_image

    first, second = "2026-03-02", "2026-03-09"
    results = [
        DetectionResult(
            image=str(
                write_image(image_dir(tmp_path / "ds", capture) / f"{capture}.jpg", (8, 8))
            ),
            width=8,
            height=8,
            boxes=[(1.0, 1.0, 4.0, 4.0)],
            scores=[0.9],
            cap=2,
        ).model_dump()
        for capture in (first, second)
    ]

    with pytest.raises(ValueError, match="captures"):
        published(tmp_path, "baseline/both", results, scope={"subject": "bud"})
    bucket = published(tmp_path, "baseline/first", results[:1], scope={"subject": "bud"})
    assert bucket.date == first


def test_the_tree_and_the_project_summary_list_only_published_buckets_alike(
    client: TestClient, tmp_path: Path
) -> None:
    """The tree route and the project-picker summary list a published bucket alike.

    A published bucket lists by name under its date in the tree, and the summary's buckets by
    date equal the tree's.
    """
    pytest.importorskip("torch")
    from tcip_mcp.project_record import record_or_problem
    from tcip_mcp.workspace import ListedProject
    from tcip_web.routes.projects import _summarize
    from tests._chain_fixtures import DATE, unassessed_bucket

    bucket = unassessed_bucket(tmp_path, experiment_id="exp-grammar")
    root = tmp_path / "ds"

    tree = client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()
    picker = _summarize(ListedProject(root, *record_or_problem(root)))

    assert tree["buckets_by_date"][DATE] == [bucket.name]
    assert picker.tree.model_dump(mode="json") == tree

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
    from tcip_mcp.project_record import record_fields
    from tcip_web.routes.projects import _summarize
    from tests._producer_fixtures import registry_of, registry_over

    root = tmp_path / "Valley_Farm"
    image_dir(root, "2026-03-02").mkdir(parents=True)
    image_dir(root, ".hidden").mkdir(parents=True)
    registry_over(root, registry_of("leaf"))

    tree = client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()
    picker = _summarize(root, record_fields(root))

    assert tree["dates_with_images"] == ["2026-03-02"]
    assert picker.dates == ["2026-03-02"]


def test_the_tree_and_the_project_summary_list_only_published_buckets_alike(
    client: TestClient, tmp_path: Path
) -> None:
    """The tree route and the project-picker summary list a published bucket alike.

    A published bucket lists by name under its date in the tree, and the summary's buckets by
    date equal the tree's.
    """
    pytest.importorskip("torch")
    from tcip_mcp.project_record import record_fields
    from tcip_web.routes.projects import _summarize
    from tests._chain_fixtures import DATE, unassessed_bucket

    bucket = unassessed_bucket(tmp_path, experiment_id="exp-grammar")
    root = tmp_path / "ds"

    tree = client.get(
        "/api/dataset/tree", params={"dataset_root": str(root)}
    ).json()
    picker = _summarize(root, record_fields(root))

    assert tree["buckets_by_date"][DATE] == [bucket.name]
    assert picker.buckets_by_date == tree["buckets_by_date"]

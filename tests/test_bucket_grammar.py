"""One bucket grammar, owned by the layout, for every reader of ``images/`` and ``predictions/``.

A directory name is a bucket (a date, ``UNDATED_BUCKET``, or a literal like a plot name) unless it
starts with a dot; a dot-prefixed directory is platform or editor cruft, never a capture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tcip_mcp.dataset_layout import list_dates


@pytest.fixture
def client() -> TestClient:
    from tcip_web.app import app

    return TestClient(app, base_url="http://127.0.0.1")


def test_is_bucket_name_admits_a_date_an_undated_token_and_a_literal():
    from tcip_mcp.dataset_layout import UNDATED_BUCKET, is_bucket_name

    assert is_bucket_name("2026-03-02")
    assert is_bucket_name(UNDATED_BUCKET)
    assert is_bucket_name("plot_14")


def test_is_bucket_name_refuses_a_dot_prefixed_or_unsafe_name():
    from tcip_mcp.dataset_layout import is_bucket_name

    assert not is_bucket_name(".hidden")
    assert not is_bucket_name(".DS_Store")
    assert not is_bucket_name("..")
    assert not is_bucket_name("a/b")
    assert not is_bucket_name("")


def test_list_dates_excludes_a_hidden_directory(tmp_path):
    images = tmp_path / "images"
    (images / "2026-03-02").mkdir(parents=True)
    (images / ".DS_Store").mkdir(parents=True)
    assert list_dates(tmp_path) == ["2026-03-02"]


def test_list_dates_admits_a_literal_and_the_undated_bucket(tmp_path):
    from tcip_mcp.dataset_layout import UNDATED_BUCKET

    images = tmp_path / "images"
    for name in ("2026-03-02", UNDATED_BUCKET, "plot_14"):
        (images / name).mkdir(parents=True)
    assert list_dates(tmp_path) == sorted(["2026-03-02", UNDATED_BUCKET, "plot_14"])


def test_ingest_refuses_a_dot_prefixed_literal_bucket(tmp_path):
    from tcip_mcp.tools.ingest_tools import ingest_images

    src = tmp_path / "src"
    src.mkdir()
    from PIL import Image
    Image.new("RGB", (8, 8)).save(src / "img.jpg")

    result = ingest_images(tmp_path / "proj", str(src), date_from=".hidden")

    assert "error" in result
    assert not (tmp_path / "proj" / "images" / ".hidden").exists()


def test_ingest_admits_a_literal_non_iso_bucket(tmp_path):
    from tcip_mcp.tools.ingest_tools import ingest_images

    src = tmp_path / "src"
    src.mkdir()
    from PIL import Image
    Image.new("RGB", (8, 8)).save(src / "img.jpg")

    result = ingest_images(tmp_path / "proj", str(src), date_from="plot_14")

    assert "error" not in result
    assert (tmp_path / "proj" / "images" / "plot_14" / "img.jpg").is_file()
    assert list_dates(tmp_path / "proj") == ["plot_14"]


def test_the_tree_and_the_project_summary_agree_on_dates_over_a_hidden_directory(
    client: TestClient, tmp_path: Path
) -> None:
    """The tree route and the project-picker summary both derive dates from the layout's one
    grammar, so a hidden directory under ``images/`` is invisible to both alike."""
    from tcip_mcp.subject_registry import SubjectRegistry, Subject
    from tcip_web.routes.projects import _summarize
    from tests._producer_fixtures import registry_over

    root = tmp_path / "Valley_Farm"
    (root / "images" / "2026-03-02").mkdir(parents=True)
    (root / "images" / ".hidden").mkdir(parents=True)
    registry_over(root, SubjectRegistry((Subject("leaf"),)))

    tree = client.get("/api/dataset/tree", params={"dataset_root": str(root)}).json()
    picker = _summarize(root)

    assert tree["dates_with_images"] == ["2026-03-02"]
    assert picker.dates == ["2026-03-02"]


def test_the_tree_and_the_project_summary_list_only_published_buckets_alike(
    client: TestClient, tmp_path: Path
) -> None:
    """Both read a bucket off its own record: a published bucket lists under the date its record
    states, and a directory of documents with no record lists nowhere."""
    pytest.importorskip("torch")
    from tcip_web.routes.projects import _summarize
    from tests._chain_fixtures import DATE, unassessed_bucket

    bucket = unassessed_bucket(tmp_path, experiment_id="exp-grammar")
    root = tmp_path / "ds"
    (root / "predictions" / "staged" / DATE).mkdir(parents=True)
    (root / "predictions" / "staged" / DATE / "s00.json").write_text("{}", encoding="utf-8")
    name = bucket.relative_to(root / "predictions").as_posix()

    tree = client.get("/api/dataset/tree", params={"dataset_root": str(root)}).json()
    picker = _summarize(root)

    assert tree["prediction_dirs"][DATE] == {name: str(bucket)}
    assert picker.prediction_dirs == tree["prediction_dirs"]

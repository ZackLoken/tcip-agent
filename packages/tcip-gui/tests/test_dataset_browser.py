"""Tests for the dataset browser."""

import pytest
import tempfile
from pathlib import Path

from tcip_gui.panels.dataset_browser import DatasetBrowser


class TestDatasetBrowser:
    def test_creation(self):
        browser = DatasetBrowser()
        assert browser._image_dir is None

    def test_load_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            browser = DatasetBrowser()
            browser.load_dataset(tmp)
            assert browser._name_label.text() == Path(tmp).name
            assert "Images: 0" in browser._stats_label.text()

    def test_load_nonexistent_dir(self):
        browser = DatasetBrowser()
        browser.load_dataset("/nonexistent/path")
        assert "Images: 0" in browser._stats_label.text()

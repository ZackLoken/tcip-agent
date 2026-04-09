"""Tests for registry query tools."""

from __future__ import annotations

from tcip_mcp.tools.registry_tools import (
    list_crops,
    get_crop_traits,
    get_trait_info,
    find_traits_by_task,
    find_traits_by_sensor,
    get_registry_summary,
)


def test_list_crops():
    result = list_crops()
    assert "hazelnut" in result
    assert result["hazelnut"]["total"] >= 2
    assert result["hazelnut"]["automatable"] >= 2


def test_get_crop_traits():
    result = get_crop_traits("hazelnut")
    assert result["crop"] == "hazelnut"
    assert result["automatable_count"] >= 2
    assert any(t["name"] == "catkin_05per_date" for t in result["automatable"])


def test_get_trait_info():
    result = get_trait_info("hazelnut", "catkin_05per_date")
    assert result["crop"] == "hazelnut"
    assert result["trait"] == "catkin_05per_date"
    assert result["ml_task"] == "object_detection"
    assert result["automatable"] is True


def test_get_trait_info_missing():
    result = get_trait_info("hazelnut", "nonexistent_trait")
    assert "error" in result


def test_find_traits_by_task():
    result = find_traits_by_task("hazelnut", "object_detection")
    assert result["count"] >= 2
    assert all(t["pipeline_group"] == "ground_rgb_object_detection" for t in result["traits"])


def test_find_traits_by_sensor():
    result = find_traits_by_sensor("hazelnut", "rgb")
    assert result["count"] >= 2


def test_get_registry_summary():
    result = get_registry_summary()
    assert "hazelnut" in result["crops"]
    assert result["crop_count"] >= 1
    assert result["total_traits"] >= 2
    assert "object_detection" in result["ml_tasks"]

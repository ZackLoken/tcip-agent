"""Phase 1.2 — registry query API, external/plugin registration, test isolation,
and the extended channel-compat (backbone-stage + pyramid-level) checks."""

import pytest

# registry.py is torch-free, so these run even without the ML stack installed.
from tcip_mcp.pipelines.registry import (
    ComponentRegistry,
    restore_all,
    snapshot_all,
)


def test_query_api_by_format_compatible_and_constraint():
    reg = ComponentRegistry("t_query")
    reg.register_factory("a", lambda: None, metadata={
        "input_format": "image", "output_format": "multi_scale_dict",
        "valid_tasks": ["detection"], "supported_channels": "any"})
    reg.register_factory("b", lambda: None, metadata={
        "input_format": "multi_scale_dict", "output_format": "boxes",
        "valid_tasks": ["detection", "instance_seg"]})

    assert [c["name"] for c in reg.list_by_format(output_format="boxes")] == ["b"]
    # b consumes exactly what a produces
    assert [c["name"] for c in reg.list_compatible_with(reg.describe("a"))] == ["b"]
    # list membership for list-valued metadata
    assert {c["name"] for c in reg.find_by_constraint(valid_tasks="detection")} == {"a", "b"}
    assert {c["name"] for c in reg.find_by_constraint(valid_tasks="instance_seg")} == {"b"}
    # channel support: "any" vs unspecified (RGB-only) reality
    assert reg.supports_channels("a", 4) is True
    assert reg.supports_channels("b", 4) is False
    assert reg.supports_channels("b", 3) is True


def test_register_external_with_bad_metadata_warns_but_succeeds(caplog):
    reg = ComponentRegistry("t_ext")
    with caplog.at_level("WARNING"):
        reg.register_external("ext", lambda: 7, metadata={"valid_tasks": "detection"})
    assert "ext" in reg
    assert reg.build("ext") == 7
    assert any("valid_tasks" in r.message for r in caplog.records)


def test_snapshot_restore_isolation():
    reg = ComponentRegistry("t_snap")
    reg.register_factory("base", lambda: None)
    snap = reg.snapshot()
    reg.register_factory("temp", lambda: None)
    assert "temp" in reg
    reg.restore(snap)
    assert "temp" not in reg and "base" in reg


def test_snapshot_all_restore_all_round_trip():
    from tcip_mcp.pipelines.registry import BACKBONES
    snap = snapshot_all()
    try:
        BACKBONES.register_external("__test_bb__", lambda: None)
        assert "__test_bb__" in BACKBONES
    finally:
        restore_all(snap)
    assert "__test_bb__" not in BACKBONES


def test_channel_compat_flags_single_scale_backbone_and_p2_overflow():
    pytest.importorskip("torch")
    # Import the component modules so the registries are populated (fpn/heads/backbones).
    import tcip_mcp.pipelines.components.backbones  # noqa: F401
    import tcip_mcp.pipelines.components.heads  # noqa: F401
    import tcip_mcp.pipelines.components.necks  # noqa: F401
    from tcip_mcp.pipelines.composer import validate_model_spec
    from tcip_mcp.pipelines.registry import BACKBONES, HEADS

    snap = snapshot_all()
    try:
        BACKBONES.register_external(
            "__single__", lambda **k: None, metadata={"output_format": "flat"})
        HEADS.register_external(
            "__capped__", lambda **k: None,
            metadata={"input_format": "multi_scale_dict", "max_pyramid_levels": 4})
        spec = {
            "backbone": {"name": "__single__"},
            "neck": {"name": "fpn", "add_p2": True},
            "heads": [{"name": "__capped__"}],
        }
        joined = " ".join(validate_model_spec(spec))
        assert "single-scale" in joined           # pyramid neck needs multi-scale backbone
        assert "pyramid levels" in joined          # add_p2 overflows the head's cap
    finally:
        restore_all(snap)

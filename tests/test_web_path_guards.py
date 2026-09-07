"""Confinement primitives of the web layer: origin trust, allowed roots, safe joins.

Every refusal asserted here is paired with the legitimate call the same guard must still
admit, since a rail that only refuses is as broken as one that only admits.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcip_web.paths import (
    allowed_image_roots,
    assert_path_allowed,
    safe_join,
)
from tcip_web.trust_boundary import origin_allowed

# -- Origin trust boundary ------------------------------------------------

LOCAL_ARRIVAL = {"type": "websocket", "scheme": "ws", "server": ["127.0.0.1", 8765],
                 "headers": [(b"host", b"127.0.0.1:8765")]}


def test_origin_without_a_parseable_host_is_not_local() -> None:
    """An Origin carrying no host is not same-machine and must not pass as loopback.

    A sandboxed cross-site iframe and a ``file://`` page both send an opaque origin whose
    hostname does not parse; treating that absence as loopback would open the state socket,
    which reports filesystem paths, to a page running anywhere.
    """
    assert not origin_allowed("null", LOCAL_ARRIVAL)
    assert not origin_allowed("file://", LOCAL_ARRIVAL)
    assert not origin_allowed("http://", LOCAL_ARRIVAL)
    assert not origin_allowed("http://evil.example.com", LOCAL_ARRIVAL)
    # A present but empty Origin is checked like any other, not read as absent.
    assert not origin_allowed("", LOCAL_ARRIVAL)


def test_local_browser_and_non_browser_clients_are_still_admitted() -> None:
    """Loopback pages, and only a client that sends no Origin at all (never an empty one),
    keep working."""
    assert origin_allowed("http://127.0.0.1:8765", LOCAL_ARRIVAL)
    assert origin_allowed("http://localhost:5173", LOCAL_ARRIVAL)
    assert origin_allowed("http://[::1]:8765", LOCAL_ARRIVAL)
    assert origin_allowed(None, LOCAL_ARRIVAL)


# -- TCIP_IMAGE_ROOTS containment -----------------------------------------


def test_sibling_sharing_a_root_name_prefix_is_outside_the_root(tmp_path, monkeypatch) -> None:
    """Containment is by path component, never by string prefix.

    ``<root>-secret`` sits beside the allowed root, not inside it, while ``<root>/sub``
    legitimately shares the same leading characters and must still be served.
    """
    allowed = tmp_path / "data"
    (allowed / "sub").mkdir(parents=True)
    inside = allowed / "sub" / "IMG_0042.JPG"
    inside.write_bytes(b"x")
    sibling = tmp_path / "data-secret"
    sibling.mkdir()
    secret = sibling / "keys.pem"
    secret.write_bytes(b"x")

    monkeypatch.setenv("TCIP_WORKSPACE", str(allowed))
    assert assert_path_allowed(str(inside)) == inside.resolve()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(secret))


def test_every_configured_root_is_honoured_not_only_the_first(
    tmp_path, tmp_path_factory: pytest.TempPathFactory, monkeypatch
) -> None:
    """A multi-entry ``TCIP_IMAGE_ROOTS`` admits work under each entry it names.

    The two roots differ in depth and in the file they hold, so honouring only the first
    entry, or failing to split the list, changes the outcome for the second.
    """
    first = tmp_path / "imagery_a" / "2026-02-11"
    first.mkdir(parents=True)
    img_a = first / "IMG_0001.JPG"
    img_a.write_bytes(b"x")
    second = tmp_path / "archive_b"
    second.mkdir()
    img_b = second / "scan.tif"
    img_b.write_bytes(b"x")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    leak = elsewhere / "leak.jpg"
    leak.write_bytes(b"x")

    root_a = tmp_path / "imagery_a"
    monkeypatch.setenv(
        "TCIP_IMAGE_ROOTS", os.pathsep.join([str(root_a), "  ", str(second)])
    )
    assert allowed_image_roots() == [root_a.resolve(), second.resolve()]
    assert assert_path_allowed(str(img_a)) == img_a.resolve()
    assert assert_path_allowed(str(img_b)) == img_b.resolve()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(leak))


def test_derived_default_canonicalises_the_path_it_admits(tmp_path, monkeypatch) -> None:
    """A path under the derived allow-set (here, the workspace) resolves to an absolute path.

    Callers treat the return value as the path to open, so a dot segment or a relative
    path must be collapsed. The additive ``TCIP_IMAGE_ROOTS`` list stays empty throughout:
    what admits this path is the workspace, not that list.
    """
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    assert allowed_image_roots() == []

    images = tmp_path / "project" / "images"
    images.mkdir(parents=True)
    target = images / "IMG_0007.JPG"
    target.write_bytes(b"x")

    dotted = images / ".." / "images" / "IMG_0007.JPG"
    assert assert_path_allowed(str(dotted)) == target.resolve()

    monkeypatch.chdir(tmp_path)
    relative = Path("project") / "images" / "IMG_0007.JPG"
    assert assert_path_allowed(relative) == target.resolve()

    monkeypatch.setenv("TCIP_IMAGE_ROOTS", "   ")
    assert allowed_image_roots() == []
    assert assert_path_allowed(str(dotted)) == target.resolve()


# -- safe_join ------------------------------------------------------------


def test_symlink_inside_the_root_pointing_out_is_refused(tmp_path) -> None:
    """A join that resolves out of the root through a symlink escapes and must be refused.

    The per-segment screen never sees a dot-dot or an absolute part here; only the check on
    the resolved path notices that the destination left the root.
    """
    base = tmp_path / "project"
    (base / "images").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"x")
    link = base / "images" / "shortcut"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    with pytest.raises(ValueError, match="is outside"):
        safe_join(base, "images/shortcut/secret.txt")

    real = base / "images" / "IMG_0007.JPG"
    real.write_bytes(b"x")
    assert safe_join(base, "images", "IMG_0007.JPG") == real.resolve()


# -- the workspace's own exclusions: .removed, and a project pending removal -------------


def test_a_removed_holding_directory_is_excluded_like_imports(tmp_path, monkeypatch) -> None:
    """``.removed`` sits directly under the workspace, itself an allowed root, so the exclusion
    has to be by name: nothing else keeps a moved-project archive out of a guarded route."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    holding = ws / ".removed" / "sample_plot_alpha-20260304T120000Z"
    holding.mkdir(parents=True)
    (holding / "images").mkdir()
    target = holding / "images" / "IMG_0001.JPG"
    target.write_bytes(b"x")

    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(target))


def test_a_project_pending_removal_is_excluded_by_identity(tmp_path, monkeypatch) -> None:
    """A pending project's own root, and a path nested under it, are refused from the moment
    its marker lands; the same paths were admitted before the marker existed."""
    import tcip_store as ts
    from tcip_mcp import workspace

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("TCIP_WORKSPACE", str(ws))
    proj = ws / "sample_plot_alpha"
    (proj / ".tcip").mkdir(parents=True)
    (proj / "images").mkdir()
    nested = proj / "images" / "IMG_0001.JPG"
    nested.write_bytes(b"x")

    assert assert_path_allowed(str(proj)) == proj.resolve()
    assert assert_path_allowed(str(nested)) == nested.resolve()

    ts.replace(
        workspace.pending_removal_key(proj),
        {"requested_at": "20260304T120000Z", "requested_by": "user:tester",
         "archive_path": "archive.zip", "holding_dir": "holding",
         "external_roots": [], "dependent_projects": []},
        expect=ts.Version.ABSENT,
    )

    with pytest.raises(ValueError, match="pending removal"):
        assert_path_allowed(str(proj))
    with pytest.raises(ValueError, match="pending removal"):
        assert_path_allowed(str(nested))


@pytest.mark.skipif(os.name != "nt", reason="drive-relative parts are a Windows path shape")
def test_drive_relative_part_cannot_replace_the_root_anchor(tmp_path) -> None:
    """A part naming another drive relocates the join off the root and must be refused.

    The part is neither absolute by POSIX rules nor a dot-dot segment, so the earlier
    screens pass it through and only the resolved-path check catches the relocation.
    """
    base = tmp_path / "project"
    (base / "images").mkdir(parents=True)
    other_drive = "Y" if base.drive.upper().startswith("Z") else "Z"

    with pytest.raises(ValueError, match="is outside"):
        safe_join(base, f"{other_drive}:evil.txt")

    target = base / "images" / "IMG_0007.JPG"
    target.write_bytes(b"x")
    assert safe_join(base, "images", "IMG_0007.JPG") == target.resolve()

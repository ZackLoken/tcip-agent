"""Confinement primitives of the web layer: origin trust, allowed roots, safe joins."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import image_root
from tcip_web.paths import (
    assert_path_allowed,
    image_roots_from_environment,
    safe_join,
)
from tcip_web.state import store
from tcip_web.trust_boundary import origin_allowed

# -- Origin trust boundary ------------------------------------------------

LOCAL_ARRIVAL = {
    "type": "websocket",
    "scheme": "ws",
    "server": ["127.0.0.1", 8765],
    "headers": [(b"host", b"127.0.0.1:8765")],
}


def test_origin_without_a_parseable_host_is_not_local() -> None:
    """An Origin carrying no host is not same-machine and must not pass as loopback.

    A sandboxed cross-site iframe and a ``file://`` page both send an opaque origin whose
    hostname does not parse; such origins, an empty one and a foreign host are refused.
    """
    assert not origin_allowed("null", LOCAL_ARRIVAL)
    assert not origin_allowed("file://", LOCAL_ARRIVAL)
    assert not origin_allowed("http://", LOCAL_ARRIVAL)
    assert not origin_allowed("http://evil.example.com", LOCAL_ARRIVAL)
    # A present but empty Origin is checked like any other, not read as absent.
    assert not origin_allowed("", LOCAL_ARRIVAL)


def test_local_browser_and_non_browser_clients_are_still_admitted() -> None:
    """Loopback pages and a client sending no Origin at all are admitted."""
    assert origin_allowed("http://127.0.0.1:8765", LOCAL_ARRIVAL)
    assert origin_allowed("http://127.0.0.1:5173", LOCAL_ARRIVAL)
    assert origin_allowed(None, LOCAL_ARRIVAL)


# -- containment under the backend's workspace and image roots ------------


def test_sibling_sharing_a_root_name_prefix_is_outside_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling whose name extends the allowed root's name is outside the root.

    A file under ``<root>-secret`` is refused as outside the allowed roots, while one under
    ``<root>/sub`` is admitted.
    """
    allowed = tmp_path / "data"
    (allowed / "sub").mkdir(parents=True)
    inside = allowed / "sub" / "IMG_0042.JPG"
    inside.write_bytes(b"x")
    sibling = tmp_path / "data-secret"
    sibling.mkdir()
    secret = sibling / "keys.pem"
    secret.write_bytes(b"x")

    store.configure(allowed.resolve(), ())
    assert assert_path_allowed(str(inside)) == inside.resolve()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(secret))


def test_every_configured_root_is_honored_not_only_the_first(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-entry ``TCIP_IMAGE_ROOTS`` admits work under each entry it names.

    The two roots differ in depth and in the file they hold; each one's file is admitted, and
    a path under neither is refused.
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
    roots = image_roots_from_environment()
    assert roots == (root_a, second)
    store.configure(store.workspace, roots)
    assert assert_path_allowed(str(img_a)) == img_a.resolve()
    assert assert_path_allowed(str(img_b)) == img_b.resolve()
    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(leak))


def test_a_relative_image_root_refuses_at_start_and_an_absolute_one_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative ``TCIP_IMAGE_ROOTS`` entry is refused at start, and an absolute one is read.

    Reading a list holding the relative entry ``archive`` raises naming it; the same list with
    that entry stated absolutely reads as its two directories.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "TCIP_IMAGE_ROOTS", os.pathsep.join([str(tmp_path), "archive"])
    )
    with pytest.raises(ValueError, match=r"relative entries \['archive'\]"):
        image_roots_from_environment()

    monkeypatch.setenv(
        "TCIP_IMAGE_ROOTS",
        os.pathsep.join([str(tmp_path), str(tmp_path / "archive")]),
    )
    assert image_roots_from_environment() == (tmp_path, tmp_path / "archive")


def test_derived_default_canonicalises_the_path_it_admits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path under the derived allow-set (here, the workspace) resolves to an absolute path.

    A dot segment is collapsed in the returned path, and a relative path is refused asking for
    an absolute one. The additive ``TCIP_IMAGE_ROOTS`` list stays empty throughout.
    """
    monkeypatch.delenv("TCIP_IMAGE_ROOTS", raising=False)
    assert image_roots_from_environment() == ()

    images = image_root(tmp_path / "project")
    images.mkdir(parents=True)
    target = images / "IMG_0007.JPG"
    target.write_bytes(b"x")

    dotted = images / ".." / images.name / "IMG_0007.JPG"
    assert assert_path_allowed(str(dotted)) == target.resolve()

    monkeypatch.chdir(tmp_path)
    relative = image_root("project") / "IMG_0007.JPG"
    with pytest.raises(ValueError, match="absolute path"):
        assert_path_allowed(relative)

    monkeypatch.setenv("TCIP_IMAGE_ROOTS", "   ")
    assert image_roots_from_environment() == ()
    assert assert_path_allowed(str(dotted)) == target.resolve()


# -- safe_join ------------------------------------------------------------


def test_symlink_inside_the_root_pointing_out_is_refused(
    tmp_path: Path,
) -> None:
    """A join that resolves out of the root through a symlink escapes and must be refused.

    The join holds no dot-dot or absolute part; it is refused for its resolved destination,
    and a join that stays inside the root resolves.
    """
    base = tmp_path / "project"
    images = image_root(base)
    images.mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"x")
    link = images / "shortcut"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks not available on this machine: {exc}")

    with pytest.raises(ValueError, match="is outside"):
        safe_join(base, f"{images.name}/shortcut/secret.txt")

    real = images / "IMG_0007.JPG"
    real.write_bytes(b"x")
    assert safe_join(base, images.name, "IMG_0007.JPG") == real.resolve()


# -- the workspace's own exclusion: .removed ---------------------------------------------


def test_a_removed_holding_directory_is_excluded_like_imports(
    tmp_path: Path,
) -> None:
    """A path under the workspace's ``.removed`` holding directory is refused by name.

    ``.removed`` sits directly under the workspace, itself an allowed root; a path under it is
    refused as outside the allowed roots.
    """
    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    store.configure(ws, ())
    holding = ws / ".removed" / "sample_plot_alpha-20260304T120000Z"
    holding.mkdir(parents=True)
    image_root(holding).mkdir()
    target = image_root(holding) / "IMG_0001.JPG"
    target.write_bytes(b"x")

    with pytest.raises(ValueError, match="outside the allowed roots"):
        assert_path_allowed(str(target))


@pytest.mark.skipif(
    os.name != "nt", reason="drive-relative parts are a Windows path shape"
)
def test_drive_relative_part_cannot_replace_the_root_anchor(
    tmp_path: Path,
) -> None:
    """A part naming another drive relocates the join off the root and must be refused.

    The part is neither absolute by POSIX rules nor a dot-dot segment; the join is refused for
    its resolved destination, and a join that stays inside the root resolves.
    """
    base = tmp_path / "project"
    images = image_root(base)
    images.mkdir(parents=True)
    other_drive = "Y" if base.drive.upper().startswith("Z") else "Z"

    with pytest.raises(ValueError, match="is outside"):
        safe_join(base, f"{other_drive}:evil.txt")

    target = images / "IMG_0007.JPG"
    target.write_bytes(b"x")
    assert safe_join(base, images.name, "IMG_0007.JPG") == target.resolve()

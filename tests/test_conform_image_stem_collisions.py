"""``scripts/conform_image_stem_collisions.py``: the census and conform of a bucket already
holding two logical identities under one case-folded stem key, the state
``image_utils.bucket_logical_identities`` refuses at every reader and the ingest door now refuses
at write time. Never touches the human's real workspace; every fixture is a scratch ``tmp_path``
project built directly (no ``ingest_images`` call), since the point is a bucket already collided
before this family's door landed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "conform_image_stem_collisions.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("conform_image_stem_collisions_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scaffold_project(root: Path) -> None:
    (root / ".tcip").mkdir(parents=True, exist_ok=True)


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_collect_collisions_finds_a_dated_a_flat_root_and_a_case_variant_pair(tmp_path: Path):
    module = _load_script()
    root = tmp_path / "proj"
    _scaffold_project(root)
    _write(root / "images" / "2026-02-11" / "foo.jpg")
    _write(root / "images" / "2026-02-11" / "foo.png")
    _write(root / "images" / "bar.jpg")  # flat root
    _write(root / "images" / "bar.png")
    _write(root / "images" / "undated" / "Baz.jpg")
    _write(root / "images" / "undated" / "baz.png")

    collisions = module.collect_collisions([root])

    keys = {(c.bucket, c.key) for c in collisions}
    assert keys == {("2026-02-11", "foo"), (None, "bar"), ("undated", "baz")}


def test_plan_reports_every_collision_and_creates_no_parking_directory(tmp_path: Path, capsys):
    module = _load_script()
    root = tmp_path / "proj"
    _scaffold_project(root)
    _write(root / "images" / "2026-02-11" / "foo.jpg")
    _write(root / "images" / "2026-02-11" / "foo.png")
    _write(root / "images" / "bar.jpg")
    _write(root / "images" / "bar.png")
    _write(root / "images" / "undated" / "Baz.jpg")
    _write(root / "images" / "undated" / "baz.png")

    collisions = module.collect_collisions([root])
    code = module.report_plan(collisions)

    assert code == 2
    out = capsys.readouterr().out
    assert "foo.jpg" in out and "foo.png" in out
    assert "bar.jpg" in out and "bar.png" in out
    assert "Baz.jpg" in out and "baz.png" in out
    assert not (root / ".tcip" / "collisions").exists()


def test_plan_over_a_clean_project_exits_zero(tmp_path: Path):
    module = _load_script()
    root = tmp_path / "proj"
    _scaffold_project(root)
    _write(root / "images" / "2026-02-11" / "foo.jpg")

    collisions = module.collect_collisions([root])
    assert module.report_plan(collisions) == 0


def test_served_today_is_the_first_by_sorted_filename_among_a_same_stem_duplicate(tmp_path: Path):
    module = _load_script()
    d = tmp_path / "images"
    d.mkdir()
    _write(d / "foo.jpg")
    _write(d / "foo.png")

    assert module.served_today([d / "foo.jpg", d / "foo.png"]) == d / "foo.jpg"


def test_served_today_is_none_for_a_case_variant_pair(tmp_path: Path):
    module = _load_script()
    d = tmp_path / "images"
    d.mkdir()
    _write(d / "Foo.jpg")
    _write(d / "foo.png")

    assert module.served_today([d / "Foo.jpg", d / "foo.png"]) is None


def test_served_today_is_none_when_a_manifest_matches_a_raw_files_exact_stem(tmp_path: Path):
    module = _load_script()
    d = tmp_path / "images"
    d.mkdir()
    _write(d / "cap.jpg")
    manifest = d / "cap.bandgroup"
    _write(manifest)

    assert module.served_today([d / "cap.jpg", manifest]) is None


def _project_with_dated_collision(root: Path) -> None:
    _scaffold_project(root)
    _write(root / "images" / "2026-02-11" / "foo.jpg")
    _write(root / "images" / "2026-02-11" / "foo.png")


def test_apply_with_keep_moves_the_other_and_records_the_event(tmp_path: Path):
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    module = _load_script()
    root = tmp_path / "proj"
    _project_with_dated_collision(root)
    keep = root / "images" / "2026-02-11" / "foo.jpg"
    other = root / "images" / "2026-02-11" / "foo.png"

    collisions = module.collect_collisions([root])
    code = module.apply_collisions(collisions, [keep])

    assert code == 0
    assert keep.is_file()
    assert not other.exists()
    parked = root / ".tcip" / "collisions" / "2026-02-11" / "foo.png"
    assert parked.is_file()

    from tcip_mcp.audit import audit_log_key

    entries = ts.read_log(audit_log_key(root.resolve())).records
    matching = [e for e in entries if e.get("tool") == "conform_image_stem_collisions"]
    assert len(matching) == 1
    entry = matching[0]
    assert entry["stem_key"] == "foo"
    assert entry["bucket"] == "2026-02-11"
    assert entry["kept_file"] == str(keep)
    assert entry["moved_file"] == str(other)
    assert isinstance(entry["kept_digest"], str) and isinstance(entry["moved_digest"], str)

    from tcip_mcp.pipelines.image_utils import list_logical_images

    logical = list_logical_images(root / "images" / "2026-02-11")
    assert set(logical) == {"foo"}


def test_apply_without_a_keep_for_a_key_refuses(tmp_path: Path):
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    module = _load_script()
    root = tmp_path / "proj"
    _project_with_dated_collision(root)

    collisions = module.collect_collisions([root])
    code = module.apply_collisions(collisions, [])

    assert code == 2
    assert (root / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (root / "images" / "2026-02-11" / "foo.png").is_file()


def test_apply_with_a_keep_outside_the_key_refuses(tmp_path: Path):
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    module = _load_script()
    root = tmp_path / "proj"
    _project_with_dated_collision(root)
    outsider = root / "images" / "2026-02-11" / "unrelated.jpg"
    _write(outsider)

    collisions = module.collect_collisions([root])
    code = module.apply_collisions(collisions, [outsider])

    assert code == 2
    assert (root / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (root / "images" / "2026-02-11" / "foo.png").is_file()


def test_apply_with_a_keep_of_the_non_served_file_while_a_label_exists_refuses(tmp_path: Path):
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    module = _load_script()
    root = tmp_path / "proj"
    _project_with_dated_collision(root)
    # foo.jpg sorts first, so it is the file served today; a label exists for stem "foo".
    _write(root / "annotations" / "2026-02-11" / "foo.json", b'{"annotations": []}')
    non_served = root / "images" / "2026-02-11" / "foo.png"

    collisions = module.collect_collisions([root])
    code = module.apply_collisions(collisions, [non_served])

    assert code == 2
    assert (root / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (root / "images" / "2026-02-11" / "foo.png").is_file()


def test_apply_onto_an_existing_parked_destination_refuses(tmp_path: Path):
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    module = _load_script()
    root = tmp_path / "proj"
    _project_with_dated_collision(root)
    keep = root / "images" / "2026-02-11" / "foo.jpg"
    _write(root / ".tcip" / "collisions" / "2026-02-11" / "foo.png")

    collisions = module.collect_collisions([root])
    code = module.apply_collisions(collisions, [keep])

    assert code == 2
    assert (root / "images" / "2026-02-11" / "foo.jpg").is_file()
    assert (root / "images" / "2026-02-11" / "foo.png").is_file()


def test_a_second_plan_over_a_conformed_project_exits_zero(tmp_path: Path):
    import tcip_store as ts
    from tcip_store.file_backend import FileBackend

    ts.bind(FileBackend())
    module = _load_script()
    root = tmp_path / "proj"
    _project_with_dated_collision(root)
    keep = root / "images" / "2026-02-11" / "foo.jpg"

    first = module.collect_collisions([root])
    assert module.apply_collisions(first, [keep]) == 0

    second = module.collect_collisions([root])
    assert module.report_plan(second) == 0

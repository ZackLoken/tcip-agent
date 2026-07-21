"""scripts/doctor.py — the data-state doctor catches the field-session bug family."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tcip_annotation import json_io
from tcip_annotation.state import BBox

PY_EXE = sys.executable
DOCTOR = str(Path(__file__).parent.parent / "scripts" / "doctor.py")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    det = root / "annotations" / "catkin" / "2026-02-11" / "detect"
    det.mkdir(parents=True)
    state = root / ".tcip" / "state"
    state.mkdir(parents=True)
    for name in ("IMG_A", "IMG_B", "IMG_C"):
        Image.new("RGB", (32, 32)).save(root / "images" / "2026-02-11" / f"{name}.JPG")
    # A: confirmed negative (empty + status). B: empty WITHOUT confirmation (the IMG_0150 case).
    # C: has objects but status wrongly says negative (contradiction).
    json_io.write_detect(det / "IMG_A.json", [], 32, 32, keep_empty=True)
    json_io.write_detect(det / "IMG_B.json", [], 32, 32, keep_empty=True)
    json_io.write_detect(det / "IMG_C.json", [BBox(1, 1, 9, 9, 0)], 32, 32)
    # Scoped by campaign/date — a confirmation belongs to the campaign it was made in.
    (state / "image_status.json").write_text(json.dumps(
        {"catkin/2026-02-11": {"IMG_A.JPG": "negative", "IMG_B.JPG": "unannotated",
                               "IMG_C.JPG": "negative"}}))
    return root


def _run(root: Path):
    return subprocess.run([PY_EXE, DOCTOR, str(root)], capture_output=True, text=True)


def test_doctor_flags_the_field_session_bug_family(tmp_path):
    root = _project(tmp_path)
    # registry entry pointing at a pytest temp checkpoint (the leak the field session found)
    reg = root / ".tcip" / "models"
    reg.mkdir(parents=True)
    (reg / "registry.json").write_text(json.dumps(
        [{"name": "junk", "checkpoint_path": "C:\Temp\pytest-of-x\model.pt"}]))

    res = _run(root)
    assert res.returncode == 2  # errors present
    out = res.stdout
    assert "IMG_B" in out and "not a confirmed negative" in out   # unconfirmed empty -> warn
    assert "contradictory" in out and "IMG_C" in out              # objects + negative -> error
    assert "junk" in out and "test/temp" in out                   # registry pollution -> error
    assert "IMG_A" not in out.replace("IMG_A.JPG is negative", "")  # confirmed negative is clean


def test_doctor_clean_project_exits_zero(tmp_path):
    root = tmp_path / "clean"
    (root / "images" / "d").mkdir(parents=True)
    det = root / "annotations" / "t" / "d" / "detect"
    det.mkdir(parents=True)
    (root / ".tcip" / "state").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "d" / "IMG_A.JPG")
    json_io.write_detect(det / "IMG_A.json", [BBox(1, 1, 9, 9, 0, created_by="user:zack")], 32, 32)
    res = _run(root)
    assert res.returncode == 0, res.stdout


def test_doctor_reports_legacy_confirmations_on_an_unmigrated_store(tmp_path):
    """The flat store is the state a project is in before anyone opens the GUI — the one moment
    the breeder most needs telling that their confirmations are not counted yet."""
    import subprocess
    import sys

    root = tmp_path / "proj"
    (root / "images" / "2026-02-11").mkdir(parents=True)
    det = root / "annotations" / "catkin" / "2026-02-11" / "detect"
    det.mkdir(parents=True)
    state = root / ".tcip" / "state"
    state.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(root / "images" / "2026-02-11" / "IMG_A.JPG")
    json_io.write_detect(det / "IMG_A.json", [], 32, 32, keep_empty=True)
    # Pre-scoping shape: keyed by image name alone, values are strings not dicts.
    (state / "image_status.json").write_text(json.dumps({"IMG_A.JPG": "negative"}))

    res = subprocess.run([sys.executable, "scripts/doctor.py", str(root)],
                         capture_output=True, text=True)
    assert "predate campaign scoping" in res.stdout, res.stdout

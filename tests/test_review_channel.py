"""P4 — the agent→GUI review channel: focus_review + stage_proposals.

focus_review resolves a model's predictions on a frame and posts a ``review_focus`` event (a
soft miss with no GUI, but the resolution must be right). stage_proposals writes agent-proposed
detections to the PREDICTIONS tree (never GT) for canvas sign-off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcip_mcp.dataset_layout import image_dir, prediction_dir
from tcip_mcp.tools.annotation_tools import focus_review, stage_proposals


@pytest.fixture(autouse=True)
def _stub_gui(monkeypatch):
    # No GUI backend in tests — make delivery a deterministic soft miss (else a 2s socket timeout).
    import tcip_mcp.web_client as web_client
    monkeypatch.setattr(web_client, "post_panel_event",
                        lambda *a, **k: {"delivered": False, "status": "no_subscribers"})


def _images(root: Path, date: str, names: list[str]) -> None:
    idir = Path(image_dir(root, date))
    idir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (idir / name).write_bytes(b"x")


def _pred(root: Path, model: str, date: str, stem: str, text: str) -> None:
    d = Path(prediction_dir(root, model, date, "detect"))
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.txt").write_text(text, encoding="utf-8")


def test_focus_review_lands_on_first_frame_with_predictions(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    imgs = [f"IMG_{i:04d}.JPG" for i in range(5)]
    _images(root, date, imgs)
    _pred(root, "baseline", date, "IMG_0002", "0 0.9 0.5 0.5 0.1 0.1\n")
    _pred(root, "baseline", date, "IMG_0003", "0 0.8 0.4 0.4 0.1 0.1\n")

    res = focus_review(str(root), str(root), "catkin", date, "baseline")
    assert "error" not in res
    assert res["image_index"] == 2  # first frame with predictions for this model
    assert res["image"] == "IMG_0002.JPG"
    assert res["n_with_predictions"] == 2
    assert res["filter_type"] == "all"
    assert isinstance(res["delivered"], bool)


def test_focus_review_empty_prediction_file_is_not_a_target(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, [f"IMG_{i:04d}.JPG" for i in range(3)])
    _pred(root, "baseline", date, "IMG_0000", "")  # empty (no detections) — skip
    _pred(root, "baseline", date, "IMG_0002", "0 0.9 0.5 0.5 0.1 0.1\n")

    res = focus_review(str(root), str(root), "catkin", date, "baseline")
    assert res["image_index"] == 2
    assert res["n_with_predictions"] == 1


def test_focus_review_explicit_index_and_filter(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    _images(root, date, [f"IMG_{i:04d}.JPG" for i in range(4)])
    _pred(root, "baseline", date, "IMG_0000", "0 0.9 0.5 0.5 0.1 0.1\n")

    res = focus_review(str(root), str(root), "catkin", date, "baseline",
                       image_index=3, detection_idx=2, filter_type="fp")
    assert res["image_index"] == 3
    assert res["detection_idx"] == 2
    assert res["filter_type"] == "fp"


def test_focus_review_rejects_bad_filter(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _images(root, "2026-02-11", ["IMG_0000.JPG"])
    res = focus_review(str(root), str(root), "catkin", "2026-02-11", "baseline", filter_type="bogus")
    assert "error" in res


def test_stage_proposals_writes_prediction_format_not_gt(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    date = "2026-02-11"
    boxes = [
        {"class_id": 0, "conf": 0.8, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        {"class_id": 1, "conf": 0.6, "cx": 0.25, "cy": 0.25, "w": 0.05, "h": 0.05},
    ]
    res = stage_proposals(str(root), "agent_proposals", date, "IMG_0001", boxes)
    assert res["staged"] == 2

    out = Path(prediction_dir(root, "agent_proposals", date, "detect")) / "IMG_0001.txt"
    assert out.is_file()
    lines = out.read_text().strip().splitlines()
    # prediction format: "cls conf cx cy w h" (6 fields, unlike GT's 5)
    assert lines[0].split() == ["0", "0.8000", "0.500000", "0.500000", "0.100000", "0.100000"]
    # It must NOT have written into annotations/ (GT).
    assert not (root / "annotations").exists()


def test_stage_proposals_rejects_unnormalized_coords(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    # pixel coords (>1) must be caught, not written off-canvas.
    boxes = [{"class_id": 0, "conf": 0.9, "cx": 320.0, "cy": 240.0, "w": 40.0, "h": 40.0}]
    res = stage_proposals(str(root), "agent_proposals", "2026-02-11", "IMG_0001", boxes)
    assert "error" in res and "normal" in res["error"].lower()


def test_stage_proposals_rejects_path_traversal_into_gt(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    good = [{"class_id": 0, "conf": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]
    # A malformed model_name/date/stem must never escape predictions/ into the GT tree.
    for bad_model in ("../annotations/catkin", "..", "a/b", "D:/evil"):
        res = stage_proposals(str(root), bad_model, "2026-02-11", "IMG_0001", good)
        assert "error" in res
    res = stage_proposals(str(root), "agent_proposals", "2026-02-11", "../../annotations/x/IMG", good)
    assert "error" in res
    assert not (root / "annotations").exists()  # nothing leaked into ground truth


def test_stage_proposals_rejects_non_numeric_class(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    boxes = [{"class_id": "cat", "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]  # bad class_id
    res = stage_proposals(str(root), "agent_proposals", "2026-02-11", "IMG_0001", boxes)
    assert "error" in res  # returns cleanly, doesn't crash the audited tool

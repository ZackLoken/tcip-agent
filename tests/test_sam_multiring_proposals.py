"""One contour extractor: a SAM-assisted proposal keeps every region of a split mask.

``state.Polygon`` is multi-ring because an occlusion-split object (a bud behind a branch) is
genuinely more than one region. The prediction-export path honored that; the human-in-the-loop
bootstrapping path (SAM proposal -> ``stage_proposals``) had its own contour extractor that kept only
the largest contour, so the same object was whole as a prediction and truncated as SAM-assisted GT,
and that GT is what a dataset rasterizes and a model learns from. Both paths now call
``tcip_annotation.mask_contours.mask_to_polygon_rings``.

The round-trip tests drive the real ``auto_mask`` / ``predict_from_point`` code with a fake SAM2 (the
pattern ``tests/test_vision.py::TestSamPredictorCache`` uses), so every hop this touches
(mask -> candidate dict -> staged ``Annotation``) is exercised, not just the extractor in isolation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("torch")

IMG_SIZE = 128
#: One object split into two disjoint lobes by an occluder. Each lobe alone yields a strictly smaller
#: enclosing box than the pair, so an assertion on the pair cannot pass on the largest lobe alone.
LOBE_A = (slice(20, 60), slice(20, 50))   # y 20:60, x 20:50: the larger lobe
LOBE_B = (slice(30, 50), slice(80, 100))  # y 30:50, x 80:100


def _split_mask() -> np.ndarray:
    m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    m[LOBE_A] = True
    m[LOBE_B] = True
    return m


@pytest.fixture
def sam_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-image project whose fake SAM2 always segments the two-lobe mask above.

    Fakes only the ``sam2`` package (build + predictor + auto-mask generator) and the checkpoint
    file, so ``auto_mask`` / ``predict_from_point`` themselves (including their contour extraction)
    are the real code under test.
    """
    mask = _split_mask()
    ys, xs = np.nonzero(mask)
    bbox = [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]

    class FakePredictor:
        def __init__(self, model):
            self.model = model

        def set_image(self, img):
            pass

        def predict(self, **kwargs):
            return mask[None], np.array([0.9]), None

    class FakeAutoMaskGenerator:
        def __init__(self, **kwargs):
            pass

        def generate(self, img):
            return [{"segmentation": mask, "area": int(mask.sum()), "bbox": bbox,
                     "stability_score": 0.95, "predicted_iou": 0.92}]

    build_mod = types.ModuleType("sam2.build_sam")
    build_mod.build_sam2 = lambda config, ckpt, device: object()
    predictor_mod = types.ModuleType("sam2.sam2_image_predictor")
    predictor_mod.SAM2ImagePredictor = FakePredictor
    autogen_mod = types.ModuleType("sam2.automatic_mask_generator")
    autogen_mod.SAM2AutomaticMaskGenerator = FakeAutoMaskGenerator
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_mod)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_mod)
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", autogen_mod)

    home = tmp_path / "home"
    ckpt = home / ".cache" / "tcip" / "sam2" / "sam2.1_hiera_tiny.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake")
    monkeypatch.setattr(Path, "home", lambda: home)

    from tcip_annotation import sam_wrapper

    monkeypatch.setattr(sam_wrapper, "_predictor", None)
    monkeypatch.setattr(sam_wrapper, "_current_model_type", None)
    monkeypatch.setattr(sam_wrapper, "_current_image_path", None)

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(120, 120, 120)).save(images / "occluded.jpg")
    return tmp_path


# --------------------------------------------------------------------------
# the shared extractor itself
# --------------------------------------------------------------------------

def test_shared_extractor_returns_one_ring_per_component() -> None:
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    rings = mask_to_polygon_rings(_split_mask())
    assert len(rings) == 2
    assert all(len(r) >= 3 for r in rings)


def test_shared_extractor_single_component_is_one_ring() -> None:
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    m = np.zeros((32, 32), dtype=np.uint8)
    m[5:20, 5:20] = 1
    assert len(mask_to_polygon_rings(m)) == 1


def test_shared_extractor_empty_mask_has_no_rings() -> None:
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    assert mask_to_polygon_rings(np.zeros((16, 16), dtype=np.uint8)) == []


def test_shared_extractor_orders_rings_largest_first() -> None:
    """A consumer that can only show one ring (an edit canvas) must get the dominant region."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    rings = mask_to_polygon_rings(_split_mask())
    xs_first = [x for x, _ in rings[0]]
    assert max(xs_first) < 80  # lobe A (the larger); lobe B lives at x >= 80


def test_shared_extractor_honors_a_soft_mask_threshold() -> None:
    """A probability mask binarizes at the caller's threshold; the default treats a mask as binary."""
    from tcip_annotation.mask_contours import mask_to_polygon_rings

    soft = np.zeros((32, 32), dtype=np.float32)
    soft[5:20, 5:20] = 0.4
    assert mask_to_polygon_rings(soft, threshold=0.5) == []
    assert len(mask_to_polygon_rings(soft, threshold=0.3)) == 1


def test_measurement_entry_point_delegates_to_the_shared_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mask_to_polygon_points`` must call the shared extractor, not re-implement agreement with it:
    a second implementation that merely agrees today is a latent defect."""
    from tcip_annotation import mask_contours
    from tcip_mcp.pipelines.measurement.mask_geometry import mask_to_polygon_points

    sentinel = [[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]]
    seen: dict[str, object] = {}

    def _fake(mask, *, threshold=None, epsilon_frac=0.005):
        seen["threshold"] = threshold
        return sentinel

    monkeypatch.setattr(mask_contours, "mask_to_polygon_rings", _fake)
    assert mask_to_polygon_points(_split_mask(), threshold=0.25) is sentinel
    assert seen["threshold"] == 0.25


# --------------------------------------------------------------------------
# mask -> candidate -> accepted Annotation, through the real proposal path
# --------------------------------------------------------------------------

def test_auto_mask_candidate_carries_every_ring(sam_project: Path) -> None:
    """Hop 1: the engine's candidate dict holds both lobes under ``rings``."""
    from tcip_annotation.sam_wrapper import auto_mask

    cands = auto_mask(str(sam_project / "images" / "occluded.jpg"), model_type="hiera_t")
    assert len(cands) == 1
    assert len(cands[0]["rings"]) == 2


def test_neutral_candidate_keeps_every_ring(sam_project: Path) -> None:
    """Hop 2: the engine-neutral candidate the proposal seam hands on keeps both rings."""
    from tcip_mcp.pipelines.proposal import resolve_proposer

    cands = resolve_proposer("sam").propose(
        str(sam_project / "images" / "occluded.jpg"), model_type="hiera_t")
    assert len(cands) == 1
    assert len(cands[0]["rings"]) == 2


def test_split_sam_proposal_is_accepted_as_a_multi_ring_annotation(sam_project: Path) -> None:
    """The whole point: a two-lobe SAM proposal reaches disk as one annotation with both lobes.

    The SAM path used to drop every region but the largest while extracting the contour, before
    ``stage_proposals`` ever saw the candidate, so the staged shape was silently a fragment of the
    object a breeder then confirmed.
    """
    from tcip_annotation import json_io
    from tcip_annotation.state import Polygon, bbox_of
    from tcip_mcp.tools.proposal_tools import stage_proposals, propose_annotations

    img = str(sam_project / "images" / "occluded.jpg")
    proposed = propose_annotations(image_path=img, engine_params={"model_type": "hiera_t"})
    assert "error" not in proposed, proposed
    assert proposed["candidate_count"] == 1

    accepted = stage_proposals(image_path=img,
                                assignments=[{"candidate_id": 0, "subject": "bud"}])
    assert "error" not in accepted, accepted
    assert accepted["proposal_count"] == 1

    anns = json_io.read_annotations(str(sam_project / "predictions" / "sam" / "occluded.json"))
    assert len(anns) == 1
    geom = anns[0].geometry
    assert isinstance(geom, Polygon)
    assert len(geom.rings) == 2, "the staged proposal lost a region of the mask"

    # The enclosing box spans both lobes: no single ring can produce this, so keeping only the
    # largest cannot pass.
    b = bbox_of(geom)
    assert b.x1 <= 20.0 and b.x2 >= 99.0
    assert b.y1 <= 30.0 and b.y2 >= 59.0


def test_segment_prompt_returns_every_ring_of_a_split_mask(sam_project: Path) -> None:
    """The prompted path (a breeder clicking the object) reports both regions too."""
    from tcip_mcp.tools.proposal_tools import segment_prompt

    result = segment_prompt(
        image_path=str(sam_project / "images" / "occluded.jpg"),
        points=[{"x": 35.0, "y": 40.0, "label": 1}],
        engine_params={"model_type": "hiera_t"},
    )
    assert "error" not in result, result
    assert result["ring_count"] == 2
    assert result["vertex_count"] == sum(len(r) for r in result["rings"])
    assert all(len(r) >= 3 for r in result["rings"])

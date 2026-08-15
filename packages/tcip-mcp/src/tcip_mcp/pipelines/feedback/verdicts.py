"""Reading one stored review verdict entry: the action vocabulary and the boxes it carries.

The engine writes each entry as a plain dict (``review_engine.record_detection_action``); every
consumer of that shape reads it through here, so what "the breeder affirmed this object" means and
where the affirmed box comes from are stated once. What each consumer then does with the box is its
own: the calibration reference emits COCO ``[x, y, w, h]`` scaled by the image, the materializer
denormalizes to pixel corners for a label file. Those output forms genuinely differ and stay apart;
only the reading of the stored entry is shared.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

POSITIVE_ACTIONS = frozenset({"accepted", "edited"})
"""The actions by which a breeder affirms an object exists. A rejection is not among them, and
neither is a verdict that only attests the image was swept."""

REJECTED_ACTION = "rejected"

_GT_BOX_KEY = "gt_bbox_norm"
_PRED_BOX_KEY = "pred_bbox_norm"

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Verdict:
    """One decoded verdict entry.

    Boxes are the stored normalized centre form ``(cx, cy, w, h)``, or ``None`` when the entry
    carries no usable box under that key. ``affirmed_box`` is the ground-truth box when the entry
    has one and the predicted box otherwise, which is what carries an accepted false positive (it
    has only what the model drew). ``geometry_recorded`` says whether the entry named a box at all,
    which is different from naming an unusable one: an entry with neither key is a coverage-only
    attestation that a human swept the image, and it contributes no object to anything.
    """

    action: str | None
    class_name: str
    gt_box: Box | None
    pred_box: Box | None
    affirmed_box: Box | None
    geometry_recorded: bool
    conf: float | None
    class_id: int | None
    missed_object_attested: bool

    @property
    def is_positive(self) -> bool:
        """Whether this verdict affirms the object exists."""
        return self.action in POSITIVE_ACTIONS

    @property
    def is_rejection(self) -> bool:
        return self.action == REJECTED_ACTION


def _box(raw: object) -> Box | None:
    """A stored ``[cx, cy, w, h]`` as floats, or None when it is absent or not four values."""
    if not raw or not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    return tuple(float(v) for v in raw)  # type: ignore[return-value]


def decode_verdict(entry: Mapping) -> Verdict:
    """One stored verdict entry as a :class:`Verdict`."""
    raw_gt = entry.get(_GT_BOX_KEY)
    raw_pred = entry.get(_PRED_BOX_KEY)
    conf = entry.get("conf")
    class_id = entry.get("class_id")
    return Verdict(
        action=entry.get("action"),
        class_name=str(entry.get("class_name", "")),
        gt_box=_box(raw_gt),
        pred_box=_box(raw_pred),
        affirmed_box=_box(raw_gt or raw_pred),
        geometry_recorded=raw_gt is not None or raw_pred is not None,
        conf=float(conf) if conf is not None else None,
        class_id=int(class_id) if class_id is not None else None,
        missed_object_attested=bool(entry.get("missed_object_attested")),
    )

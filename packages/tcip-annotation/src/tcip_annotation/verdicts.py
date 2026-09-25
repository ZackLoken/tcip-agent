"""The review verdict: its action vocabulary, declared once, and the one reading of a stored entry.

``VerdictAction``'s literal strings are the declaration; ``VERDICT_ACTIONS`` is derived from them.
Every stored entry is read through :func:`decode_verdict`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from tcip_annotation.json_io import iscrowd_of

VerdictAction = Literal["accepted", "rejected", "edited", "swept"]
VERDICT_ACTIONS: tuple[VerdictAction, ...] = get_args(VerdictAction)

ACCEPTED_ACTION: VerdictAction = "accepted"
REJECTED_ACTION: VerdictAction = "rejected"
EDITED_ACTION: VerdictAction = "edited"
SWEPT_ACTION: VerdictAction = "swept"

POSITIVE_ACTIONS: frozenset[VerdictAction] = frozenset({ACCEPTED_ACTION, EDITED_ACTION})
"""The actions by which a breeder affirms an object exists. A rejection is not among them, and
neither is a verdict that only attests the image was swept."""

GT_BOX_KEY = "gt_bbox_norm"
PRED_BOX_KEY = "pred_bbox_norm"

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Verdict:
    """One decoded verdict entry.

    ``reviewed_by`` is the bare name of the person who recorded this verdict, as the engine stamped
    it, or ``""`` when the store holds none.

    Boxes are the stored normalized center form ``(cx, cy, w, h)``, or ``None`` when the entry
    carries no usable box under that key. ``affirmed_box`` is the ground-truth box when the entry
    has one and the predicted box otherwise (an accepted false positive has only what the model
    drew). ``geometry_recorded`` says whether the entry named a box at all: an entry with neither
    key is a coverage-only attestation that a human swept the image, and it contributes no object
    to anything. ``iscrowd`` is the crowd flag of the ground-truth record the verdict was recorded
    on, which every entry states. ``producer_identity`` and ``conf_threshold`` are the producing
    bucket's identity and the display threshold the verdict was recorded against, each ``None``
    where the caller resolved none.
    """

    action: VerdictAction | None
    class_name: str
    reviewed_by: str
    gt_box: Box | None
    pred_box: Box | None
    affirmed_box: Box | None
    geometry_recorded: bool
    conf: float | None
    class_id: int | None
    missed_object_attested: bool
    iscrowd: bool
    producer_identity: dict | None
    conf_threshold: float | None

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


_STATED_KEYS = ("action", "reviewed_by", "class_name", GT_BOX_KEY, PRED_BOX_KEY, "conf",
                "class_id", "missed_object_attested", "iscrowd", "producer_identity",
                "conf_threshold")
"""The keys every verdict entry states (``review_engine.record_detection_action`` writes each
one, a ``null`` where it has no value)."""


def decode_verdict(entry: Mapping) -> Verdict:
    """One stored verdict entry as a :class:`Verdict`, reading every stated key as stated, or
    ``ValueError`` naming the keys an entry does not state or a crowd flag that is not one."""
    missing = [k for k in _STATED_KEYS if k not in entry]
    if missing:
        raise ValueError(f"a verdict entry states {missing}; this one carries none: {entry!r}")
    raw_gt, raw_pred = entry[GT_BOX_KEY], entry[PRED_BOX_KEY]
    conf, class_id, threshold = entry["conf"], entry["class_id"], entry["conf_threshold"]
    return Verdict(
        action=entry["action"],
        class_name=str(entry["class_name"]),
        reviewed_by=str(entry["reviewed_by"]).strip(),
        gt_box=_box(raw_gt),
        pred_box=_box(raw_pred),
        affirmed_box=_box(raw_gt or raw_pred),
        geometry_recorded=raw_gt is not None or raw_pred is not None,
        conf=float(conf) if conf is not None else None,
        class_id=int(class_id) if class_id is not None else None,
        missed_object_attested=bool(entry["missed_object_attested"]),
        iscrowd=iscrowd_of(entry["iscrowd"]),
        producer_identity=entry["producer_identity"],
        conf_threshold=float(threshold) if threshold is not None else None,
    )

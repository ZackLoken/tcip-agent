"""GUI-driving tools: push data to a panel, or drive the live Annotate tab to a frame.

Delivery goes through the tcip-web event channel (:mod:`tcip_mcp.web_client`), which delivers only
when the backend has this server's project open, and answers ``delivered: false`` naming what it
has open otherwise, or when no GUI answers there.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import Annotation, BBox
from tcip_annotation.state import polygonal
from tcip_annotation.json_io import UnreadableLabelDocument
from tcip_annotation.json_io import read_annotations as read_labels

from tcip_mcp.audit import audited
from tcip_mcp.server import tool


def _logical_image_names(images_dir) -> list[str]:
    """Every logical image's on-disk display name under ``images_dir``: a plain file's own name, or
    (for a ``.bandgroup``-grouped capture) its manifest's filename.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

    return [logical_image_name(src) for src in list_logical_images(images_dir).values()]


@tool()
@audited
def push_panel_event(project: Path, workspace: Path, panel: str, event_type: str,
                     data: dict) -> dict:
    """Push structured data to a TCIP GUI panel via the tcip-web backend serving ``workspace``,
    which delivers it only while it has this project open.

    Sends an HTTP POST to the running FastAPI server (see :mod:`tcip_mcp.web_client`); the backend
    broadcasts to any connected browsers via WebSocket, or answers ``delivered: false`` naming the
    project it has open instead. If the backend itself is not running, returns ``{"status":
    "no_subscribers"}``.

    Args:
        panel: Target panel: one per GUI tab, or 'app' for app-level events like
            annotate_focus. See ``web_client.VALID_PANELS`` for the current set.
        event_type: Any event type the panel understands, not confined to
            ``web_client.PLATFORM_PANEL_EVENTS`` (the platform's own emitters). 'banner' is the one
            example the browser renders directly: ``data['text']`` shows as a quiet note above that
            tab.
        data: Arbitrary JSON data payload.
    """
    from tcip_mcp.web_client import VALID_PANELS, post_panel_event

    if panel not in VALID_PANELS:
        return {"error": f"Unknown panel: {panel}. Valid: {sorted(VALID_PANELS)}"}

    result = post_panel_event(project, workspace, panel, event_type, data)
    result.setdefault("panel", panel)
    result.setdefault("event_type", event_type)
    return result


@tool()
@audited
def focus_human_attention(
    project: Path,
    workspace: Path,
    dataset_root: str,
    subject: str,
    date: str,
    image_index: int | None = None,
    mode: str | None = None,
    predictions_dir: str | None = None,
    proposal: int | None = None,
) -> dict:
    """Drive the live Annotate tab to a (subject, date) frame in the right mode, showing the
    proposals of the bucket at ``predictions_dir`` when one is named (emits ``annotate_focus``).

    It lands on ``image_index``, else on the first frame holding ``subject``: in the bucket's
    document for it when a bucket is named, in its label document otherwise. A backend with
    another project open, or none running, answers ``delivered: false`` rather than raising. An
    image elsewhere on the date whose document will not read is named by file name in the result's
    ``unreadable``; only the landed-on frame's own unreadable document refuses, naming that
    document's path, and an ``image_index`` outside the date's images or a ``date`` that is not
    one path segment refuses.

    Args:
        dataset_root: Dataset root holding ``images/`` and ``annotations/`` (plus
            ``predictions/``).
        subject: Annotation subject (e.g. "fruit").
        date: Capture-date bucket (e.g. "2026-03-02").
        image_index: Index into the date's sorted image list.
        mode: "box", "polygon" or "point" (default: inferred from the geometry the labels on
            that frame actually carry).
        predictions_dir: The published bucket or staged proposals whose proposals the canvas
            shows.
        proposal: The index of the proposal, in the bucket's document for the frame, to focus.
    """
    from tcip_annotation.json_io import annotations_hold_subject

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import annotation_dir, image_dir, label_filename
    from tcip_mcp.web_client import ANNOTATE_MODES, PANEL_EVENT_ANNOTATE_FOCUS, post_panel_event
    from tcip_mcp.workspace import is_valid_name

    if not is_valid_name(date):
        return {"error": f"date must be a single safe path segment (no separators/'..'), got {date!r}"}
    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    images = sorted(_logical_image_names(idir))
    if not images:
        return {"error": f"no images on {date}"}
    try:
        bucket = read_bucket(predictions_dir) if predictions_dir else None
    except ValueError as exc:
        return {"error": str(exc)}
    adir = Path(annotation_dir(dataset_root, date))

    def _document(name: str) -> Path | None:
        if bucket is not None:
            return bucket.document(name)
        label = adir / label_filename(Path(name).stem)
        return label if label.is_file() else None

    holding: list[int] = []
    unreadable: dict[str, str] = {}
    for i, name in enumerate(images):
        document = _document(name)
        try:
            if document is not None and annotations_hold_subject(read_labels(str(document)),
                                                                 subject):
                holding.append(i)
        except UnreadableLabelDocument as exc:
            unreadable[name] = str(exc)

    if image_index is None:
        image_index = holding[0] if holding else 0
    if not 0 <= image_index < len(images):
        return {"error": f"image_index {image_index} names none of the {len(images)} images on "
                         f"{date}"}
    target_name = images[image_index]
    if target_name in unreadable:
        return {"error": unreadable[target_name]}
    if mode is None:
        label = adir / label_filename(Path(target_name).stem)
        try:
            task = _subject_task(read_labels(str(label)), subject) if label.is_file() else None
        except UnreadableLabelDocument as exc:
            return {"error": str(exc)}
        mode = _TASK_MODE.get(task or "", "box")
    if mode not in ANNOTATE_MODES:
        vocabulary = ", ".join(repr(m) for m in ANNOTATE_MODES)
        return {"error": f"mode must be one of {vocabulary}, got {mode!r}"}

    payload = {
        "dataset_root": dataset_root, "subject": subject, "date": date,
        "image_index": image_index, "mode": mode, "active_subject": subject,
        "predictions_dir": predictions_dir, "proposal": proposal,
    }
    result = post_panel_event(project, workspace, "app", PANEL_EVENT_ANNOTATE_FOCUS, payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "subject": subject, "date": date, "image_index": image_index, "mode": mode,
        "predictions_dir": predictions_dir, "proposal": proposal,
        "n_images": len(images), "n_holding_subject": len(holding), "image": target_name,
        "unreadable": sorted(unreadable),
    }


def _subject_task(anns: list[Annotation], subject: str) -> str | None:
    """"segment" if ``subject`` has a polygon here, "detect" if it has a box, "point" if its only
    geometry here is a point, else None (no geometry-bearing annotation of ``subject``).
    """
    scoped = [a for a in anns if a.subject == subject and a.geometry is not None]
    if any(polygonal(a.geometry) for a in scoped):
        return "segment"
    if any(isinstance(a.geometry, BBox) for a in scoped):
        return "detect"
    if scoped:
        return "point"
    return None


# The GUI drawing mode each resolved task is edited in, the frontend's own Mode union ("box" |
# "polygon" | "point", store/types.ts); a point-only frame lands in point mode, never box mode.
_TASK_MODE = {"segment": "polygon", "detect": "box", "point": "point"}

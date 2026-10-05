"""GUI-driving tools: push data to a panel, or drive the live Annotate tab to a frame.

Delivery goes through the tcip-web event channel (:mod:`tcip_mcp.web_client`), which delivers only
when the backend has this server's project open, and answers ``delivered: false`` naming what it
has open otherwise, or when no GUI answers there.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import Annotation, BBox
from tcip_annotation.state import polygonal
from tcip_annotation.json_io import (
    UnreadableLabelDocument, read_document_versioned, read_label_document,
)

from tcip_mcp.server import tool


def _logical_image_names(images_dir) -> list[str]:
    """Every logical image's on-disk display name under ``images_dir``: a plain file's own name, or
    (for a ``.bandgroup``-grouped capture) its manifest's filename.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

    return [logical_image_name(src) for src in list_logical_images(images_dir).values()]


@tool()
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
def focus_human_attention(
    project: Path,
    workspace: Path,
    dataset_root: str,
    subject: str,
    date: str,
    image_index: int | None = None,
    mode: str | None = None,
    bucket: str | None = None,
    proposal: int | None = None,
) -> dict:
    """Drive the live Annotate tab to a (subject, date) frame in the right mode, showing the
    proposals of the bucket named ``bucket`` under ``dataset_root`` when one is named (emits
    ``annotate_focus``).

    It lands on ``image_index``, else on the first frame holding ``subject``: in the bucket's
    document for it when a bucket is named, in its label document otherwise. A backend with
    another project open, or none running, answers ``delivered: false`` rather than raising. An
    image elsewhere on the date whose document will not read is named by file name in the result's
    ``unreadable``; only the landed-on frame's own unreadable document refuses, naming that
    document, and an ``image_index`` outside the date's images or a ``date`` that is not one path
    segment refuses.

    Args:
        dataset_root: Dataset root holding ``images/``.
        subject: Annotation subject (e.g. "fruit").
        date: Capture-date bucket (e.g. "2026-03-02").
        image_index: Index into the date's sorted image list.
        mode: "box", "polygon" or "point" (default: inferred from the geometry that frame's
            document, the bucket's when one is named, actually carries).
        bucket: The name of the published bucket or staged proposals whose proposals the canvas
            shows.
        proposal: The index of the proposal, in the bucket's document for the frame, to focus.
    """
    from tcip_annotation.json_io import annotations_hold_subject

    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.dataset_layout import image_dir, label_key
    from tcip_mcp.web_client import ANNOTATE_MODES, PANEL_EVENT_ANNOTATE_FOCUS, post_panel_event

    try:
        idir = image_dir(dataset_root, date)
    except ValueError as exc:
        return {"error": str(exc)}
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    images = sorted(_logical_image_names(idir))
    if not images:
        return {"error": f"no images on {date}"}
    try:
        found = read_bucket(dataset_root, bucket) if bucket else None
    except ValueError as exc:
        return {"error": str(exc)}

    held: dict[str, list[Annotation]] = {}
    unreadable: dict[str, str] = {}
    for name in images:
        stem = Path(name).stem
        try:
            if found is None:
                held[name] = read_document_versioned(
                    label_key(dataset_root, date, stem))[0].annotations
            else:
                document = found.document_key(stem)
                held[name] = read_label_document(document).annotations if document else []
        except UnreadableLabelDocument as exc:
            unreadable[name] = str(exc)
    holding = [i for i, name in enumerate(images)
               if name in held and annotations_hold_subject(held[name], subject)]

    if image_index is None:
        image_index = holding[0] if holding else 0
    if not 0 <= image_index < len(images):
        return {"error": f"image_index {image_index} names none of the {len(images)} images on "
                         f"{date}"}
    target_name = images[image_index]
    if target_name in unreadable:
        return {"error": unreadable[target_name]}
    if mode is None:
        mode = _TASK_MODE.get(_subject_task(held[target_name], subject) or "", "box")
    if mode not in ANNOTATE_MODES:
        vocabulary = ", ".join(repr(m) for m in ANNOTATE_MODES)
        return {"error": f"mode must be one of {vocabulary}, got {mode!r}"}

    payload = {
        "dataset_root": dataset_root, "subject": subject, "date": date,
        "image_index": image_index, "mode": mode, "active_subject": subject,
        "bucket": bucket, "proposal": proposal,
    }
    result = post_panel_event(project, workspace, "app", PANEL_EVENT_ANNOTATE_FOCUS, payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "subject": subject, "date": date, "image_index": image_index, "mode": mode,
        "bucket": bucket, "proposal": proposal,
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

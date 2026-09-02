"""GUI-driving tools: push data to a panel, or drive the live Annotate/Review tab to a frame.

Both routes go through the tcip-web event channel (:mod:`tcip_mcp.web_client`), a soft miss with
``delivered: false`` if no GUI answers. Neither reads or writes an annotation or prediction file
itself beyond what it needs to resolve where to land: push_panel_data forwards an arbitrary
payload; focus resolves a (subject, date) frame through read_annotations' own reader and posts
the event the GUI honors with local view setters.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import Annotation, BBox, Polygon
from tcip_annotation.json_io import UnreadableLabelDocument
from tcip_annotation.json_io import read_annotations as read_labels

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp


def _logical_image_names(images_dir) -> list[str]:
    """Every logical image's on-disk display name under ``images_dir``, a plain file's own name,
    or (for a ``.bandgroup``-grouped capture) its manifest's filename, the file every other
    by-name reader (``image_name_map``, the dataset gallery route) treats as that capture's name.
    Folding sibling band files into one name here is what lets this tool's frame index agree with
    the frontend's own image_list, which now enumerates the same way.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

    return [logical_image_name(src) for src in list_logical_images(images_dir).values()]


@mcp.tool()
@audited
def push_panel_data(
    panel: str,
    event_type: str,
    data: dict,
) -> dict:
    """Push structured data to a TCIP GUI panel via the tcip-web backend.

    Sends an HTTP POST to the running FastAPI server (see :mod:`tcip_mcp.web_client`); the backend
    broadcasts to any connected browsers via WebSocket. If the backend is not running the call
    returns ``{"status": "no_subscribers"}`` so the agent can proceed.

    Args:
        panel: Target panel: one per GUI tab, or 'app' for app-level events like annotate_focus /
            review_focus / active_project_changed. See ``web_client.VALID_PANELS`` for the
            current set.
        event_type: Any event type the panel understands, not confined to
            ``web_client.PLATFORM_PANEL_EVENTS`` (the platform's own emitters). 'banner' is the
            one example the browser renders directly: ``data['text']`` shows as a quiet note
            above that tab.
        data: Arbitrary JSON data payload.
    """
    from tcip_mcp.web_client import VALID_PANELS, post_panel_event

    if panel not in VALID_PANELS:
        return {"error": f"Unknown panel: {panel}. Valid: {sorted(VALID_PANELS)}"}

    result = post_panel_event(panel, event_type, data)
    result.setdefault("panel", panel)
    result.setdefault("event_type", event_type)
    return result


@mcp.tool()
@audited
def focus(
    tab: str,
    project_root: str,
    dataset_root: str,
    subject: str,
    date: str,
    image_index: int | None = None,
    mode: str | None = None,
    model_name: str | None = None,
    detection_idx: int = 0,
    filter_type: str = "all",
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Drive the live GUI to a (subject, date) frame, the Annotate tab or the Review tab.

    ``tab='annotate'`` lands the Annotate tab on the first frame annotated for ``subject`` in the
    right mode (emits ``annotate_focus``); ``tab='review'`` lands the Review tab on a model's
    predictions (emits ``review_focus``). Requires the GUI to be running; returns ``delivered:
    false`` if not. On both tabs, an image elsewhere on the date whose label or prediction
    document will not read is named by file name in the result's ``unreadable`` rather than
    aborting the call; only the landed-on or explicitly named frame's own unreadable document
    refuses, naming that document's path.

    Args:
        tab: Which GUI surface to drive, 'annotate' or 'review'.
        project_root: Project root (== dataset_root for workspace projects).
        dataset_root: Dataset root holding ``images/`` and ``annotations/`` (plus ``predictions/``).
        subject: Annotation subject (e.g. "leaf", "bush").
        date: Capture-date bucket (e.g. "2026-03-02").
        image_index: Index into the date's sorted image list. Default: first frame labeled for
            ``subject`` (annotate) / with a prediction of ``subject`` for the model (review).
        mode: Annotate only, "box", "polygon" or "point" (default: inferred from the geometry the
            labels on that frame actually carry).
        model_name: Review only (required when ``tab='review'``), the model whose predictions.
        detection_idx: Review only, which detection to center in the Review navigator.
        filter_type: Review only, "all" | "tp" | "fp" | "fn" match filter.
        iou_threshold: Review only, IoU cutoff for the TP/FP/FN match classification.
        conf_threshold: Review only, confidence cutoff for showing predictions.
    """
    if tab == "annotate":
        return _focus_annotate(project_root, dataset_root, subject, date, mode=mode,
                               image_index=image_index)
    if tab == "review":
        if not model_name:
            return {"error": "tab='review' requires model_name"}
        return _focus_review(
            project_root, dataset_root, subject, date, model_name,
            image_index=image_index, detection_idx=detection_idx, filter_type=filter_type,
            iou_threshold=iou_threshold, conf_threshold=conf_threshold,
        )
    return {"error": f"tab must be 'annotate' or 'review', got {tab!r}"}


def _subject_task(anns: list[Annotation], subject: str) -> str | None:
    """"segment" if ``subject`` has a polygon here, "detect" if it has a box, "point" if its only
    geometry here is a point, else None (no geometry-bearing annotation of ``subject``).

    ``"point"`` is a real answer, not a box: callers use a non-``None`` return as "this frame is
    annotated for the subject", so collapsing a point-only frame to ``None`` would hide it from the
    Annotate tab's own frame count, while calling it ``"detect"`` would claim a box nobody drew.
    """
    scoped = [a for a in anns if a.subject == subject and a.geometry is not None]
    if any(isinstance(a.geometry, Polygon) for a in scoped):
        return "segment"
    if any(isinstance(a.geometry, BBox) for a in scoped):
        return "detect"
    if scoped:
        return "point"
    return None


# The GUI drawing mode each resolved task is edited in, the frontend's own Mode union ("box" |
# "polygon" | "point", store/types.ts); a point-only frame lands in point mode, never box mode.
_TASK_MODE = {"segment": "polygon", "detect": "box", "point": "point"}


def _focus_annotate(
    project_root: str,
    dataset_root: str,
    subject: str,
    date: str,
    mode: str | None = None,
    image_index: int | None = None,
) -> dict:
    """Drive the live Annotate tab to a (subject, date), in the right mode, on a frame labeled for
    the subject. Posts an ``annotate_focus`` event the GUI honors with local view setters.

    Refuses only when the landed-on (or explicitly named) frame's own label will not read, naming
    that document's path in the error; every other image whose label will not read is named
    instead (by image file name, not by document path) in the result's ``unreadable``, so one bad
    file elsewhere on the date never closes the agent's own navigation surface for it.

    The ``mode`` vocabulary this validates against is ``tcip_mcp.web_client.AnnotateMode``, the
    same Literal ``tcip_web.state.GuiState.mode`` holds: this package cannot import ``tcip_web``
    (the dependency runs the other way), so the vocabulary is declared here and both sides read
    the one Literal rather than restating it.
    """
    from tcip_mcp.dataset_layout import annotation_dir, image_dir, label_filename
    from tcip_mcp.web_client import ANNOTATE_MODES, PANEL_EVENT_ANNOTATE_FOCUS, post_panel_event

    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    images = sorted(_logical_image_names(idir))
    if not images:
        return {"error": f"no images on {date}"}

    adir = Path(annotation_dir(dataset_root, date))

    def _task(stem: str) -> str | None:
        f = adir / label_filename(stem)
        return _subject_task(read_labels(str(f)), subject) if f.is_file() else None

    n_annotated = 0
    first_idx: int | None = None
    tasks: dict[str, str | None] = {}
    unreadable: dict[str, str] = {}
    for i, name in enumerate(images):
        try:
            task = _task(Path(name).stem)
        except UnreadableLabelDocument as exc:
            unreadable[name] = str(exc)
            continue
        tasks[name] = task
        if task is not None:
            n_annotated += 1
            if first_idx is None:
                first_idx = i

    if image_index is None:
        image_index = first_idx if first_idx is not None else 0
    image_index = max(0, min(image_index, len(images) - 1))

    target_name = images[image_index]
    if target_name in unreadable:
        return {"error": unreadable[target_name]}
    resolved_task = tasks[target_name]
    if mode is None:
        mode = _TASK_MODE.get(resolved_task or "", "box")
    if mode not in ANNOTATE_MODES:
        vocabulary = ", ".join(repr(m) for m in ANNOTATE_MODES)
        return {"error": f"mode must be one of {vocabulary}, got {mode!r}"}

    payload = {
        "project_root": project_root, "dataset_root": dataset_root,
        "subject": subject, "date": date, "image_index": image_index, "mode": mode,
        "active_subject": subject,
    }
    result = post_panel_event("app", PANEL_EVENT_ANNOTATE_FOCUS, payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "subject": subject, "date": date, "image_index": image_index, "mode": mode,
        "n_images": len(images), "n_annotated": n_annotated, "image": images[image_index],
        "unreadable": sorted(unreadable),
    }


def _focus_review(
    project_root: str,
    dataset_root: str,
    subject: str,
    date: str,
    model_name: str,
    image_index: int | None = None,
    detection_idx: int = 0,
    filter_type: str = "all",
    iou_threshold: float = 0.5,
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Drive the live Review tab to a model's predictions of ``subject`` on a frame. Posts a
    ``review_focus`` event the GUI honors with local setters.

    Refuses only when the landed-on (or explicitly named) frame's own prediction document will
    not read, naming that document's path in the error; every other image whose prediction will
    not read is named instead (by image file name) in the result's ``unreadable``, the same
    stance ``_focus_annotate`` takes."""
    from tcip_mcp.dataset_layout import image_dir, label_filename, prediction_dir
    from tcip_mcp.web_client import PANEL_EVENT_REVIEW_FOCUS, post_panel_event
    from tcip_mcp.workspace import is_valid_name

    if filter_type not in ("all", "tp", "fp", "fn"):
        return {"error": f"filter_type must be all|tp|fp|fn, got {filter_type!r}"}
    for label, val in (("model_name", model_name), ("date", date)):
        if not is_valid_name(val):
            return {"error": f"{label} must be a single safe path segment (no separators/'..'), got {val!r}"}

    idir = Path(image_dir(dataset_root, date))
    if not idir.is_dir():
        return {"error": f"no images for date {date} under {dataset_root}"}
    images = sorted(_logical_image_names(idir))
    if not images:
        return {"error": f"no images on {date}"}

    pred_dir = Path(prediction_dir(dataset_root, model_name, date))

    def _has_pred(stem: str) -> bool:
        f = pred_dir / label_filename(stem)
        return bool(f.is_file() and any(a.subject == subject and a.geometry is not None
                                        for a in read_labels(str(f))))

    n_with_preds = 0
    first_idx: int | None = None
    unreadable: dict[str, str] = {}
    for i, name in enumerate(images):
        try:
            has_pred = _has_pred(Path(name).stem)
        except UnreadableLabelDocument as exc:
            unreadable[name] = str(exc)
            continue
        if has_pred:
            n_with_preds += 1
            if first_idx is None:
                first_idx = i

    if image_index is None:
        image_index = first_idx if first_idx is not None else 0
    image_index = max(0, min(image_index, len(images) - 1))

    target_name = images[image_index]
    if target_name in unreadable:
        return {"error": unreadable[target_name]}

    payload = {
        "project_root": project_root, "dataset_root": dataset_root,
        "subject": subject, "date": date, "model_name": model_name,
        "image_index": image_index, "detection_idx": detection_idx, "filter_type": filter_type,
        "iou_threshold": iou_threshold, "conf_threshold": conf_threshold,
    }
    result = post_panel_event("app", PANEL_EVENT_REVIEW_FOCUS, payload)
    return {
        "delivered": result.get("delivered", False),
        "status": result.get("status"),
        "subject": subject, "date": date, "model_name": model_name,
        "image_index": image_index, "detection_idx": detection_idx, "filter_type": filter_type,
        "n_images": len(images), "n_with_predictions": n_with_preds, "image": images[image_index],
        "unreadable": sorted(unreadable),
    }

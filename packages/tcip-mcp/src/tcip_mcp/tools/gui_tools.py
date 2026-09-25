"""GUI-driving tools: push data to a panel, or drive the live Annotate/Review tab to a frame.

Both refuse before posting anything when the GUI's currently open project (the
``canvas_open_binding`` record) does not name the caller's own stated project; once the binding
agrees, delivery goes through the tcip-web event channel (:mod:`tcip_mcp.web_client`) as a soft
miss with ``delivered: false`` if no GUI answers there.
"""

from __future__ import annotations

from pathlib import Path

from tcip_annotation import Annotation, BBox
from tcip_annotation.state import polygonal
from tcip_annotation.json_io import UnreadableLabelDocument
from tcip_annotation.json_io import read_annotations as read_labels

from tcip_mcp.audit import audited
from tcip_mcp.pipelines.resolution import DEFAULT_CONF
from tcip_mcp.server import mcp


def _binding_refusal(binding: dict | None, compared_root: str) -> dict:
    """Refuse driving the GUI, naming what the GUI has open, the root this call named, and the step
    that converges them (``web_client.binding_divergence``), under
    ``web_client.gui_binding_matches``. No binding at all, or one naming another project, refuses
    every time; ``delivered`` is always ``False``.
    """
    from tcip_mcp.web_client import binding_divergence

    divergence = binding_divergence(binding, compared_root)
    if binding is None:
        opened = "The GUI has no project open"
    elif divergence["bound_project"]:
        opened = f"The GUI has {divergence['bound_project']!r} open"
    else:
        opened = f"The GUI has the root {divergence['bound_root']} open"
    return {
        "error": f"{opened}; this call named {compared_root}. {divergence['converge']}.",
        "bound_project": divergence["bound_project"],
        "bound_root": divergence["bound_root"],
        "compared_root": compared_root,
        "converge": divergence["converge"],
        "delivered": False,
    }


def _logical_image_names(images_dir) -> list[str]:
    """Every logical image's on-disk display name under ``images_dir``: a plain file's own name, or
    (for a ``.bandgroup``-grouped capture) its manifest's filename.
    """
    from tcip_mcp.pipelines.image_utils import list_logical_images, logical_image_name

    return [logical_image_name(src) for src in list_logical_images(images_dir).values()]


@mcp.tool()
@audited
def push_panel_event(
    panel: str,
    event_type: str,
    data: dict,
    *,
    project_root: str,
) -> dict:
    """Push structured data to a TCIP GUI panel via the tcip-web backend, for ``project_root``.

    Refuses before posting anything when the GUI's currently open project (the
    ``canvas_open_binding`` record) does not name ``project_root``: a mismatch, or no binding at
    all, refuses every time, naming what the GUI has open, the root this call named, and the step
    that converges them. No override. The comparison is against ``project_root`` as the caller
    states it, never this process's own pinned platform-state root.

    Once the binding agrees, sends an HTTP POST to the running FastAPI server (see
    :mod:`tcip_mcp.web_client`); the backend broadcasts to any connected browsers via WebSocket. If
    the backend itself is not running, returns ``{"status": "no_subscribers"}``.

    Args:
        panel: Target panel: one per GUI tab, or 'app' for app-level events like annotate_focus /
            review_focus. See ``web_client.VALID_PANELS`` for the current set.
            ``active_project_changed`` is not sendable through this tool; ``activate_project``
            posts it through ``web_client.post_panel_event``.
        event_type: Any event type the panel understands, not confined to
            ``web_client.PLATFORM_PANEL_EVENTS`` (the platform's own emitters). 'banner' is the one
            example the browser renders directly: ``data['text']`` shows as a quiet note above that
            tab.
        data: Arbitrary JSON data payload.
        project_root: The project this push means to drive the GUI for, compared against the
            ``canvas_open_binding`` record. Required.
    """
    from tcip_mcp.web_client import (
        VALID_PANELS, GuiBindingUnreadable, gui_binding_matches, post_panel_event,
    )

    if panel not in VALID_PANELS:
        return {"error": f"Unknown panel: {panel}. Valid: {sorted(VALID_PANELS)}"}

    try:
        matches, binding = gui_binding_matches(project_root)
    except GuiBindingUnreadable as exc:
        return {"error": str(exc), "delivered": False, "panel": panel, "event_type": event_type}
    if not matches:
        refusal = _binding_refusal(binding, project_root)
        refusal.setdefault("panel", panel)
        refusal.setdefault("event_type", event_type)
        return refusal

    result = post_panel_event(panel, event_type, data)
    result.setdefault("panel", panel)
    result.setdefault("event_type", event_type)
    return result


@mcp.tool()
@audited
def focus_human_attention(
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
    predictions (emits ``review_focus``). Once the GUI's open project agrees with ``project_root``
    (see below), a backend that is not running still answers ``delivered: false`` rather than
    raising. On both tabs, an image elsewhere on the date whose label or prediction document will
    not read is named by file name in the result's ``unreadable``; only the landed-on or explicitly
    named frame's own unreadable document refuses, naming that document's path.

    Refuses before resolving anything when the GUI's currently open project (the
    ``canvas_open_binding`` record) does not name ``project_root``. No override; a mismatch, or no
    binding at all, refuses every time, naming what the GUI has open, the root this call named, and
    the step that converges them.

    Args:
        tab: Which GUI surface to drive, 'annotate' or 'review'.
        project_root: Project root (== dataset_root for workspace projects).
        dataset_root: Dataset root holding ``images/`` and ``annotations/`` (plus
            ``predictions/``).
        subject: Annotation subject (e.g. "fruit").
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
    from tcip_mcp.web_client import GuiBindingUnreadable, gui_binding_matches

    try:
        matches, binding = gui_binding_matches(project_root)
    except GuiBindingUnreadable as exc:
        return {"error": str(exc), "delivered": False}
    if not matches:
        return _binding_refusal(binding, project_root)

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
    that document's path; every other image whose label will not read is named (by image file name)
    in the result's ``unreadable``. ``mode`` is validated against
    ``tcip_mcp.web_client.AnnotateMode``.
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

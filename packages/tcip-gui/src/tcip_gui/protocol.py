"""JSON-RPC protocol message types for agent ↔ GUI communication."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class JsonRpcMessage:
    """Base JSON-RPC 2.0 message."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    jsonrpc: str = "2.0"

    def to_line(self) -> str:
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_line(cls, line: str) -> JsonRpcMessage:
        data = json.loads(line)
        return cls(
            method=data.get("method", ""),
            params=data.get("params", {}),
            id=data.get("id"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


# ── Agent → GUI messages ──


@dataclass
class TextDelta:
    text: str

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(method="assistant.text_delta", params={"text": self.text})


@dataclass
class TextDone:
    text: str

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(method="assistant.text_done", params={"text": self.text})


@dataclass
class ToolCallStart:
    tool_id: str
    name: str
    input: dict[str, Any]

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="tool.call_start",
            params={"id": self.tool_id, "name": self.name, "input": self.input},
        )


@dataclass
class ToolCallResult:
    tool_id: str
    output: str
    is_error: bool = False

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="tool.call_result",
            params={"id": self.tool_id, "output": self.output, "is_error": self.is_error},
        )


@dataclass
class PermissionRequest:
    request_id: str
    tool: str
    input: dict[str, Any]
    description: str
    level: str

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="permission.request",
            params={
                "id": self.request_id,
                "tool": self.tool,
                "input": self.input,
                "description": self.description,
                "level": self.level,
            },
        )


@dataclass
class UsageStatus:
    input_tokens: int
    output_tokens: int
    cost: float

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="status.usage",
            params={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost": self.cost,
            },
        )


@dataclass
class TurnComplete:
    turn_number: int

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="status.turn_complete",
            params={"turn_number": self.turn_number},
        )


@dataclass
class AgentError:
    code: int
    message: str

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="error",
            params={"code": self.code, "message": self.message},
        )


# ── Canvas messages (Phase 4 — stubs for protocol completeness) ──


@dataclass
class CanvasLoadImage:
    path: str
    annotations: list[Any] | None = None

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {"path": self.path}
        if self.annotations:
            params["annotations"] = self.annotations
        return JsonRpcMessage(method="canvas.load_image", params=params)


@dataclass
class CanvasShowPredictions:
    predictions_path: str
    prediction_type: str = "detection"  # "detection" or "segmentation"

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="canvas.show_predictions",
            params={
                "predictions_path": self.predictions_path,
                "prediction_type": self.prediction_type,
            },
        )


@dataclass
class CanvasClear:
    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(method="canvas.clear")


@dataclass
class CanvasHighlight:
    annotation_ids: list[str]

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="canvas.highlight",
            params={"annotation_ids": self.annotation_ids},
        )


# ── Training messages (Agent → GUI) ──


@dataclass
class TrainingStarted:
    run_name: str
    metrics_path: str
    total_epochs: int

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="training.started",
            params={
                "run_name": self.run_name,
                "metrics_path": self.metrics_path,
                "total_epochs": self.total_epochs,
            },
        )


@dataclass
class TrainingMetricsUpdate:
    epoch: float
    train_loss: float | None = None
    val_loss: float | None = None
    map50: float | None = None
    lr: float | None = None
    stage: str | None = None
    eta: str | None = None

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {"epoch": self.epoch}
        if self.train_loss is not None:
            params["train_loss"] = self.train_loss
        if self.val_loss is not None:
            params["val_loss"] = self.val_loss
        if self.map50 is not None:
            params["map50"] = self.map50
        if self.lr is not None:
            params["lr"] = self.lr
        if self.stage is not None:
            params["stage"] = self.stage
        if self.eta is not None:
            params["eta"] = self.eta
        return JsonRpcMessage(method="training.metrics_update", params=params)


@dataclass
class TrainingComplete:
    run_name: str
    best_epoch: int
    best_metric: float
    metrics: dict[str, Any] | None = None

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {
            "run_name": self.run_name,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
        }
        if self.metrics:
            params["metrics"] = self.metrics
        return JsonRpcMessage(method="training.complete", params=params)


@dataclass
class ResultsShow:
    run_name: str
    overall: dict[str, float]
    per_class: list[dict[str, Any]]
    worst_images: list[str] | None = None
    csv_path: str | None = None

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {
            "run_name": self.run_name,
            "overall": self.overall,
            "per_class": self.per_class,
        }
        if self.worst_images:
            params["worst_images"] = self.worst_images
        if self.csv_path:
            params["csv_path"] = self.csv_path
        return JsonRpcMessage(method="results.show", params=params)


# ── Dataset browsing messages (Agent → GUI) ──


@dataclass
class DatasetNavigate:
    """Tell the GUI to navigate to a specific image or filter."""
    image_path: str | None = None
    filter_type: str | None = None  # "all", "annotated", "unannotated", "class:<name>"
    sort_by: str | None = None  # "name", "date", "annotation_count"

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {}
        if self.image_path:
            params["image_path"] = self.image_path
        if self.filter_type:
            params["filter"] = self.filter_type
        if self.sort_by:
            params["sort_by"] = self.sort_by
        return JsonRpcMessage(method="dataset.navigate", params=params)


@dataclass
class DatasetSetClasses:
    """Sync class names to the GUI class selector."""
    classes: list[str]
    source: str = ""  # e.g., "data.yaml", "classes.txt"

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {"classes": self.classes}
        if self.source:
            params["source"] = self.source
        return JsonRpcMessage(method="dataset.set_classes", params=params)


# ── Inference progress messages (Agent → GUI) ──


@dataclass
class InferenceProgress:
    """Progress update during batch inference."""
    current: int
    total: int
    image_path: str = ""
    elapsed_seconds: float = 0.0

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {"current": self.current, "total": self.total}
        if self.image_path:
            params["image_path"] = self.image_path
        if self.elapsed_seconds > 0:
            params["elapsed_seconds"] = self.elapsed_seconds
        return JsonRpcMessage(method="inference.progress", params=params)


@dataclass
class InferenceComplete:
    """Inference batch finished."""
    total_images: int
    total_detections: int
    output_dir: str = ""

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="inference.complete",
            params={
                "total_images": self.total_images,
                "total_detections": self.total_detections,
                "output_dir": self.output_dir,
            },
        )


# ── Export progress messages (Agent → GUI) ──


@dataclass
class ExportProgress:
    """Progress update during model export (ONNX, TensorRT, etc)."""
    stage: str  # "preparing", "exporting", "validating", "done"
    format: str = "onnx"
    message: str = ""

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="export.progress",
            params={"stage": self.stage, "format": self.format, "message": self.message},
        )


@dataclass
class ExportComplete:
    """Model export finished."""
    format: str
    output_path: str
    file_size_bytes: int = 0

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="export.complete",
            params={
                "format": self.format,
                "output_path": self.output_path,
                "file_size_bytes": self.file_size_bytes,
            },
        )


# ── GUI → Agent messages ──


@dataclass
class UserMessage:
    text: str

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="user.message",
            params={"text": self.text},
            id=str(uuid.uuid4()),
        )


@dataclass
class PermissionResponse:
    request_id: str
    allowed: bool
    reason: str | None = None

    def to_rpc(self) -> JsonRpcMessage:
        params: dict[str, Any] = {"id": self.request_id, "allowed": self.allowed}
        if self.reason:
            params["reason"] = self.reason
        return JsonRpcMessage(method="permission.response", params=params)


@dataclass
class ControlCancel:
    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(method="control.cancel")


@dataclass
class ControlShutdown:
    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(method="control.shutdown")


# ── GUI → Agent canvas feedback ──


@dataclass
class CanvasAnnotationSaved:
    image_path: str
    count: int

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="canvas.annotation_saved",
            params={"image_path": self.image_path, "count": self.count},
        )


@dataclass
class CanvasReviewComplete:
    image_path: str
    accepted: int
    edited: int
    rejected: int

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="canvas.review_complete",
            params={
                "image_path": self.image_path,
                "accepted": self.accepted,
                "edited": self.edited,
                "rejected": self.rejected,
            },
        )


@dataclass
class CanvasBatchComplete:
    summary: dict[str, Any]

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="canvas.batch_complete",
            params={"summary": self.summary},
        )


# ── GUI → Agent training/results feedback ──


@dataclass
class TrainingPauseRequested:
    run_name: str = ""

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="training.pause",
            params={"run_name": self.run_name} if self.run_name else {},
        )


@dataclass
class TrainingStopRequested:
    run_name: str = ""

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="training.stop",
            params={"run_name": self.run_name} if self.run_name else {},
        )


@dataclass
class ResultAction:
    action: str  # "accept", "retrain", "retrain_hpo"
    run_name: str = ""

    def to_rpc(self) -> JsonRpcMessage:
        return JsonRpcMessage(
            method="results.action",
            params={"action": self.action, "run_name": self.run_name},
        )


def parse_agent_message(msg: JsonRpcMessage) -> Any:
    """Parse a JSON-RPC message from the agent into a typed dataclass."""
    m = msg.method
    p = msg.params
    if m == "assistant.text_delta":
        return TextDelta(text=p.get("text", ""))
    if m == "assistant.text_done":
        return TextDone(text=p.get("text", ""))
    if m == "tool.call_start":
        return ToolCallStart(tool_id=p["id"], name=p["name"], input=p.get("input", {}))
    if m == "tool.call_result":
        return ToolCallResult(tool_id=p["id"], output=p.get("output", ""), is_error=p.get("is_error", False))
    if m == "permission.request":
        return PermissionRequest(
            request_id=p["id"], tool=p["tool"], input=p.get("input", {}),
            description=p.get("description", ""), level=p.get("level", ""),
        )
    if m == "status.usage":
        return UsageStatus(
            input_tokens=p.get("input_tokens", 0),
            output_tokens=p.get("output_tokens", 0),
            cost=p.get("cost", 0.0),
        )
    if m == "status.turn_complete":
        return TurnComplete(turn_number=p.get("turn_number", 0))
    if m == "error":
        return AgentError(code=p.get("code", -1), message=p.get("message", "Unknown error"))
    if m == "canvas.load_image":
        return CanvasLoadImage(path=p.get("path", ""), annotations=p.get("annotations"))
    if m == "canvas.show_predictions":
        return CanvasShowPredictions(
            predictions_path=p.get("predictions_path", ""),
            prediction_type=p.get("prediction_type", "detection"),
        )
    if m == "canvas.clear":
        return CanvasClear()
    if m == "canvas.highlight":
        return CanvasHighlight(annotation_ids=p.get("annotation_ids", []))
    if m == "training.started":
        return TrainingStarted(
            run_name=p.get("run_name", ""),
            metrics_path=p.get("metrics_path", ""),
            total_epochs=p.get("total_epochs", 0),
        )
    if m == "training.metrics_update":
        return TrainingMetricsUpdate(
            epoch=p.get("epoch", 0),
            train_loss=p.get("train_loss"),
            val_loss=p.get("val_loss"),
            map50=p.get("map50"),
            lr=p.get("lr"),
            stage=p.get("stage"),
            eta=p.get("eta"),
        )
    if m == "training.complete":
        return TrainingComplete(
            run_name=p.get("run_name", ""),
            best_epoch=p.get("best_epoch", 0),
            best_metric=p.get("best_metric", 0.0),
            metrics=p.get("metrics"),
        )
    if m == "results.show":
        return ResultsShow(
            run_name=p.get("run_name", ""),
            overall=p.get("overall", {}),
            per_class=p.get("per_class", []),
            worst_images=p.get("worst_images"),
            csv_path=p.get("csv_path"),
        )
    if m == "dataset.navigate":
        return DatasetNavigate(
            image_path=p.get("image_path"),
            filter_type=p.get("filter"),
            sort_by=p.get("sort_by"),
        )
    if m == "dataset.set_classes":
        return DatasetSetClasses(
            classes=p.get("classes", []),
            source=p.get("source", ""),
        )
    if m == "inference.progress":
        return InferenceProgress(
            current=p.get("current", 0),
            total=p.get("total", 0),
            image_path=p.get("image_path", ""),
            elapsed_seconds=p.get("elapsed_seconds", 0.0),
        )
    if m == "inference.complete":
        return InferenceComplete(
            total_images=p.get("total_images", 0),
            total_detections=p.get("total_detections", 0),
            output_dir=p.get("output_dir", ""),
        )
    if m == "export.progress":
        return ExportProgress(
            stage=p.get("stage", ""),
            format=p.get("format", "onnx"),
            message=p.get("message", ""),
        )
    if m == "export.complete":
        return ExportComplete(
            format=p.get("format", ""),
            output_path=p.get("output_path", ""),
            file_size_bytes=p.get("file_size_bytes", 0),
        )
    return msg  # Unknown method, return raw

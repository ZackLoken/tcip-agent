//! Event bus for async observability — notifications, logging, GUI signals.
//!
//! Producers emit `AgentEvent`s. The bus routes them to registered sinks
//! (Discord webhook, JSON log, GUI bridge) based on each sink's filter.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Event model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    SessionStarted {
        session_id: String,
        crop: Option<String>,
    },
    SessionEnded {
        session_id: String,
        summary: String,
    },
    TurnCompleted {
        turn_id: u32,
        tool_calls: u32,
        tokens_used: u64,
    },
    TrainingStarted {
        model: String,
        dataset: String,
        epochs: u32,
    },
    EpochCompleted {
        epoch: u32,
        loss: f64,
        metrics: HashMap<String, f64>,
    },
    TrainingFinished {
        model: String,
        final_metrics: HashMap<String, f64>,
    },
    EvaluationDone {
        model: String,
        map50: f64,
    },
    AnnotationReviewNeeded {
        image_count: u32,
        pending_count: u32,
    },
    RecoveryAttempted {
        scenario: String,
        success: bool,
    },
    WorkflowStoryCompleted {
        workflow_id: String,
        story: String,
    },
    WorkflowCompleted {
        workflow_id: String,
    },
    Error {
        message: String,
        recoverable: bool,
    },
}

impl AgentEvent {
    /// Whether this event is critical and should bypass batching.
    pub fn is_critical(&self) -> bool {
        matches!(
            self,
            AgentEvent::Error { recoverable: false, .. }
                | AgentEvent::WorkflowCompleted { .. }
                | AgentEvent::TrainingFinished { .. }
        )
    }

    /// Human-readable summary for notifications.
    pub fn summary(&self) -> String {
        match self {
            Self::SessionStarted { session_id, crop } => {
                format!("Session started: {session_id} (crop: {})", crop.as_deref().unwrap_or("none"))
            }
            Self::SessionEnded { session_id, summary } => {
                format!("Session ended: {session_id} — {summary}")
            }
            Self::TurnCompleted { turn_id, tool_calls, tokens_used } => {
                format!("Turn {turn_id}: {tool_calls} tool calls, {tokens_used} tokens")
            }
            Self::TrainingStarted { model, dataset, epochs } => {
                format!("Training started: {model} on {dataset} for {epochs} epochs")
            }
            Self::EpochCompleted { epoch, loss, .. } => {
                format!("Epoch {epoch}: loss={loss:.4}")
            }
            Self::TrainingFinished { model, final_metrics } => {
                let metrics_str: Vec<String> = final_metrics
                    .iter()
                    .map(|(k, v)| format!("{k}={v:.3}"))
                    .collect();
                format!("Training finished: {model} — {}", metrics_str.join(", "))
            }
            Self::EvaluationDone { model, map50 } => {
                format!("Evaluation: {model} mAP@50={map50:.3}")
            }
            Self::AnnotationReviewNeeded { image_count, pending_count } => {
                format!("Review needed: {pending_count}/{image_count} images pending")
            }
            Self::RecoveryAttempted { scenario, success } => {
                let status = if *success { "succeeded" } else { "failed" };
                format!("Recovery {status}: {scenario}")
            }
            Self::WorkflowStoryCompleted { workflow_id, story } => {
                format!("Story complete: {story} (workflow {workflow_id})")
            }
            Self::WorkflowCompleted { workflow_id } => {
                format!("Workflow complete: {workflow_id}")
            }
            Self::Error { message, recoverable } => {
                let severity = if *recoverable { "Warning" } else { "Error" };
                format!("{severity}: {message}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Event sink trait
// ---------------------------------------------------------------------------

/// Trait for event consumers (Discord, JSON log, GUI, etc.).
pub trait EventSink: Send + Sync {
    /// Whether this sink accepts the given event.
    fn accepts(&self, event: &AgentEvent) -> bool;
    /// Send the event to the sink.
    fn send(&self, event: &AgentEvent) -> Result<(), SinkError>;
    /// Sink name for logging.
    fn name(&self) -> &str;
}

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("send error: {0}")]
    Send(String),
}

// ---------------------------------------------------------------------------
// JSON log sink
// ---------------------------------------------------------------------------

/// Appends events as structured JSON lines to a log file.
pub struct JsonLogSink {
    path: PathBuf,
}

impl JsonLogSink {
    pub fn new(workspace: &Path) -> Self {
        Self {
            path: workspace.join(".tcip").join("events.jsonl"),
        }
    }
}

impl EventSink for JsonLogSink {
    fn accepts(&self, _event: &AgentEvent) -> bool {
        true // log everything
    }

    fn send(&self, event: &AgentEvent) -> Result<(), SinkError> {
        use std::io::Write;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let record = serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "event": event,
        });
        writeln!(file, "{}", serde_json::to_string(&record).unwrap_or_default())?;
        Ok(())
    }

    fn name(&self) -> &str {
        "json_log"
    }
}

// ---------------------------------------------------------------------------
// GUI event sink — forwards events as JSON-RPC messages
// ---------------------------------------------------------------------------

/// Forwards agent events to the GUI as JSON-RPC messages over a shared writer.
///
/// This is the bridge between the async event bus and the GUI process,
/// enabling real-time training progress, inference updates, and notifications.
pub struct GuiEventSink {
    writer: Arc<std::sync::Mutex<Box<dyn std::io::Write + Send>>>,
}

impl GuiEventSink {
    pub fn new(writer: Box<dyn std::io::Write + Send>) -> Self {
        Self {
            writer: Arc::new(std::sync::Mutex::new(writer)),
        }
    }

    /// Convert an `AgentEvent` to a JSON-RPC method + params.
    fn to_jsonrpc(event: &AgentEvent) -> Option<(String, serde_json::Value)> {
        match event {
            AgentEvent::TrainingStarted {
                model,
                dataset,
                epochs,
            } => Some((
                "training.started".to_string(),
                serde_json::json!({
                    "run_name": model,
                    "dataset": dataset,
                    "total_epochs": epochs,
                }),
            )),
            AgentEvent::EpochCompleted {
                epoch,
                loss,
                metrics,
            } => {
                let mut params = serde_json::json!({
                    "epoch": epoch,
                    "train_loss": loss,
                });
                if let Some(map) = params.as_object_mut() {
                    for (k, v) in metrics {
                        map.insert(k.clone(), serde_json::json!(v));
                    }
                }
                Some(("training.metrics_update".to_string(), params))
            }
            AgentEvent::TrainingFinished {
                model,
                final_metrics,
            } => {
                let best_metric = final_metrics
                    .get("mAP50")
                    .or_else(|| final_metrics.get("best_metric"))
                    .copied()
                    .unwrap_or(0.0);
                let best_epoch = final_metrics
                    .get("best_epoch")
                    .copied()
                    .unwrap_or(0.0) as i64;
                Some((
                    "training.complete".to_string(),
                    serde_json::json!({
                        "run_name": model,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "metrics": final_metrics,
                    }),
                ))
            }
            AgentEvent::EvaluationDone { model, map50 } => Some((
                "results.show".to_string(),
                serde_json::json!({
                    "run_name": model,
                    "overall": {"mAP50": map50},
                    "per_class": [],
                }),
            )),
            AgentEvent::Error { message, recoverable } => Some((
                "error".to_string(),
                serde_json::json!({
                    "code": if *recoverable { -1 } else { -2 },
                    "message": message,
                }),
            )),
            AgentEvent::AnnotationReviewNeeded {
                image_count,
                pending_count,
            } => Some((
                "annotation.review_needed".to_string(),
                serde_json::json!({
                    "image_count": image_count,
                    "pending_count": pending_count,
                }),
            )),
            // Session and workflow events don't have GUI-facing protocol messages
            _ => None,
        }
    }
}

impl EventSink for GuiEventSink {
    fn accepts(&self, event: &AgentEvent) -> bool {
        Self::to_jsonrpc(event).is_some()
    }

    fn send(&self, event: &AgentEvent) -> Result<(), SinkError> {
        if let Some((method, params)) = Self::to_jsonrpc(event) {
            let msg = serde_json::json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            });
            let line = serde_json::to_string(&msg)
                .map_err(|e| SinkError::Send(e.to_string()))?;
            let mut w = self.writer.lock().map_err(|e| SinkError::Send(e.to_string()))?;
            use std::io::Write;
            writeln!(w, "{line}")?;
            w.flush()?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "gui_bridge"
    }
}

// ---------------------------------------------------------------------------
// Event bus
// ---------------------------------------------------------------------------

/// Async event bus that routes events to registered sinks.
pub struct EventBus {
    sender: mpsc::Sender<AgentEvent>,
}

impl EventBus {
    /// Create a new event bus and start the routing task.
    ///
    /// Returns the bus handle and a JoinHandle for the router task.
    pub fn new(sinks: Vec<Box<dyn EventSink>>) -> (Self, tokio::task::JoinHandle<()>) {
        let (sender, mut receiver) = mpsc::channel::<AgentEvent>(256);
        let sinks = Arc::new(sinks);

        let handle = tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                for sink in sinks.iter() {
                    if sink.accepts(&event) {
                        if let Err(e) = sink.send(&event) {
                            warn!("sink {} failed: {e}", sink.name());
                        }
                    }
                }
            }
            debug!("event bus router shut down");
        });

        (Self { sender }, handle)
    }

    /// Emit an event (non-blocking).
    pub fn emit(&self, event: AgentEvent) {
        if let Err(e) = self.sender.try_send(event) {
            warn!("event bus overflow: {e}");
        }
    }

    /// Get a clone of the sender for distributing to subsystems.
    pub fn sender(&self) -> mpsc::Sender<AgentEvent> {
        self.sender.clone()
    }
}

/// A lightweight handle that can be cloned and passed to subsystems.
#[derive(Clone)]
pub struct EventEmitter {
    sender: mpsc::Sender<AgentEvent>,
}

impl EventEmitter {
    pub fn new(sender: mpsc::Sender<AgentEvent>) -> Self {
        Self { sender }
    }

    pub fn emit(&self, event: AgentEvent) {
        if let Err(e) = self.sender.try_send(event) {
            warn!("event emit overflow: {e}");
        }
    }

    /// Create a no-op emitter (for testing or when events are disabled).
    pub fn noop() -> Self {
        let (sender, _receiver) = mpsc::channel(1);
        Self { sender }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_summary_format() {
        let event = AgentEvent::TrainingFinished {
            model: "yolov8n".to_string(),
            final_metrics: HashMap::from([("mAP50".to_string(), 0.752)]),
        };
        let summary = event.summary();
        assert!(summary.contains("yolov8n"));
        assert!(summary.contains("0.752"));
    }

    #[test]
    fn critical_events() {
        assert!(AgentEvent::Error {
            message: "OOM".to_string(),
            recoverable: false
        }
        .is_critical());

        assert!(!AgentEvent::TurnCompleted {
            turn_id: 1,
            tool_calls: 2,
            tokens_used: 500
        }
        .is_critical());
    }

    #[test]
    fn gui_sink_writes_jsonrpc() {
        let buffer = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let writer: Box<dyn std::io::Write + Send> = Box::new(BufferWriter(buffer.clone()));
        let sink = GuiEventSink::new(writer);

        let event = AgentEvent::EpochCompleted {
            epoch: 5,
            loss: 0.123,
            metrics: HashMap::from([("mAP50".to_string(), 0.85)]),
        };
        assert!(sink.accepts(&event));
        sink.send(&event).unwrap();

        let data = buffer.lock().unwrap();
        let line = String::from_utf8_lossy(&data);
        let parsed: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["method"], "training.metrics_update");
        assert_eq!(parsed["params"]["epoch"], 5);
    }

    #[test]
    fn gui_sink_ignores_session_events() {
        let writer: Box<dyn std::io::Write + Send> = Box::new(Vec::<u8>::new());
        let sink = GuiEventSink::new(writer);

        let event = AgentEvent::SessionStarted {
            session_id: "s1".to_string(),
            crop: None,
        };
        assert!(!sink.accepts(&event));
    }

    /// Helper to capture sink output in tests.
    struct BufferWriter(Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn json_log_sink_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let sink = JsonLogSink::new(tmp.path());

        let event = AgentEvent::SessionStarted {
            session_id: "test-123".to_string(),
            crop: Some("hazelnut".to_string()),
        };

        assert!(sink.accepts(&event));
        sink.send(&event).unwrap();

        let log_path = tmp.path().join(".tcip").join("events.jsonl");
        let content = std::fs::read_to_string(log_path).unwrap();
        assert!(content.contains("test-123"));
        assert!(content.contains("hazelnut"));
    }

    #[tokio::test]
    async fn event_bus_routes_to_sinks() {
        let tmp = tempfile::tempdir().unwrap();
        let sink = JsonLogSink::new(tmp.path());
        let sinks: Vec<Box<dyn EventSink>> = vec![Box::new(sink)];

        let (bus, handle) = EventBus::new(sinks);
        bus.emit(AgentEvent::TurnCompleted {
            turn_id: 1,
            tool_calls: 3,
            tokens_used: 1000,
        });

        // Drop sender to close the channel
        drop(bus);
        handle.await.unwrap();

        let log_path = tmp.path().join(".tcip").join("events.jsonl");
        let content = std::fs::read_to_string(log_path).unwrap();
        assert!(content.contains("TurnCompleted"));
    }

    #[test]
    fn noop_emitter_does_not_panic() {
        let emitter = EventEmitter::noop();
        emitter.emit(AgentEvent::SessionStarted {
            session_id: "x".to_string(),
            crop: None,
        });
    }
}

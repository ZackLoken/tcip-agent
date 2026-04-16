//! JSON-RPC transport for GUI ↔ Agent communication.
//!
//! Reads newline-delimited JSON-RPC from stdin, writes responses and events to stdout.
//! Designed to be driven by the PyQt6 GUI's AgentBridge.

use serde::Deserialize;
use serde_json::Value;
use std::io::{self, BufRead, Write};
use tcip_api::ApiClient;
use tcip_runtime::checkpoint::CheckpointResolver;
use tcip_runtime::ConversationRuntime;
use tcip_tools::ToolExecutor;
use tracing::{debug, error};

#[derive(Deserialize)]
struct IncomingMessage {
    method: String,
    params: Option<Value>,
    #[allow(dead_code)]
    id: Option<Value>,
}

fn emit(method: &str, params: Value) {
    let msg = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    });
    let line = serde_json::to_string(&msg).expect("serialize");
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    let _ = writeln!(handle, "{line}");
    let _ = handle.flush();
}

fn emit_error(code: i64, message: &str) {
    emit("error", serde_json::json!({"code": code, "message": message}));
}

/// Run the JSON-RPC transport loop.
pub async fn run<C, T, R>(
    mut runtime: ConversationRuntime<C, T, R>,
) -> Result<(), Box<dyn std::error::Error>>
where
    C: ApiClient,
    T: ToolExecutor,
    R: CheckpointResolver,
{
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut turn_number: u32 = 0;

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                error!("stdin read error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let msg: IncomingMessage = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(e) => {
                emit_error(-32700, &format!("Parse error: {e}"));
                continue;
            }
        };

        match msg.method.as_str() {
            "user.message" => {
                let text = msg
                    .params
                    .as_ref()
                    .and_then(|p| p.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("");

                if text.is_empty() {
                    emit_error(-32602, "Empty message text");
                    continue;
                }

                debug!("user.message: {text}");

                match runtime.run_turn(text).await {
                    Ok(summary) => {
                        debug!("run_turn OK: text_len={}, tool_calls={}",
                               summary.assistant_text.len(), summary.tool_calls.len());
                        // Emit text done
                        if !summary.assistant_text.is_empty() {
                            emit(
                                "assistant.text_done",
                                serde_json::json!({"text": summary.assistant_text}),
                            );
                        }

                        // Emit tool call results
                        for tc in &summary.tool_calls {
                            emit(
                                "tool.call_start",
                                serde_json::json!({
                                    "id": tc.tool_use_id,
                                    "name": tc.name,
                                    "input": tc.input
                                }),
                            );
                            emit(
                                "tool.call_result",
                                serde_json::json!({
                                    "id": tc.tool_use_id,
                                    "output": tc.output,
                                    "is_error": tc.is_error
                                }),
                            );
                        }

                        // Emit usage
                        turn_number += 1;
                        let model = runtime.model();
                        let cost = runtime.usage().estimated_cost_usd(model);
                        emit(
                            "status.usage",
                            serde_json::json!({
                                "input_tokens": summary.input_tokens,
                                "output_tokens": summary.output_tokens,
                                "cost": cost
                            }),
                        );
                        emit(
                            "status.turn_complete",
                            serde_json::json!({"turn_number": turn_number}),
                        );
                    }
                    Err(e) => {
                        error!("run_turn FAILED: {e}");
                        emit_error(-32000, &format!("Runtime error: {e}"));
                    }
                }
            }
            "permission.response" => {
                // TODO: Wire to PermissionEnforcer when async approval is implemented
                debug!("permission.response received (not yet wired)");
            }
            "session.fork" => {
                let branch_name = msg
                    .params
                    .as_ref()
                    .and_then(|p| p.get("branch_name"))
                    .and_then(Value::as_str)
                    .unwrap_or("unnamed");

                let forked = runtime.session.fork(branch_name);
                emit(
                    "session.forked",
                    serde_json::json!({
                        "session_id": forked.id(),
                        "parent_session_id": runtime.session.id(),
                        "branch_name": branch_name
                    }),
                );
                debug!("session.fork → {} (branch: {branch_name})", forked.id());
            }
            "control.cancel" => {
                debug!("cancel requested (not yet implemented)");
            }
            "control.shutdown" => {
                debug!("shutdown requested");
                break;
            }
            other => {
                emit_error(-32601, &format!("Unknown method: {other}"));
            }
        }
    }

    Ok(())
}

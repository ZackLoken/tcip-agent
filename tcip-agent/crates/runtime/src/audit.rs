//! Audit log — append-only JSONL record of all tool calls and agent actions.
//!
//! Writes to `.tcip/audit.jsonl` in the workspace root.
//! Each line is a JSON object with: timestamp, action, tool_name, tool_input (sanitized),
//! result_status, user_id, and session_id.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: String,
    pub action: AuditAction,
    pub tool_name: Option<String>,
    /// Sanitized input (no secrets, truncated large values)
    pub tool_input_preview: Option<String>,
    pub result: AuditResult,
    pub session_id: Option<String>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    ToolCall,
    ToolResult,
    PolicyDeny,
    PolicyAsk,
    CheckpointApproved,
    CheckpointDenied,
    SessionStart,
    SessionEnd,
    ModeSwitch,
    FileWrite,
    FileRead,
    BashCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditResult {
    Success,
    Error,
    Denied,
    Pending,
}

pub struct AuditLogger {
    log_path: PathBuf,
    session_id: Option<String>,
}

impl AuditLogger {
    /// Create a new audit logger writing to `.tcip/audit.jsonl` in the workspace.
    pub fn new(workspace_root: &Path) -> Self {
        let tcip_dir = workspace_root.join(".tcip");
        let _ = fs::create_dir_all(&tcip_dir);
        Self {
            log_path: tcip_dir.join("audit.jsonl"),
            session_id: None,
        }
    }

    pub fn set_session_id(&mut self, id: String) {
        self.session_id = Some(id);
    }

    /// Log a tool call.
    pub fn log_tool_call(&self, tool_name: &str, input: &serde_json::Value, result: AuditResult) {
        let preview = sanitize_input_preview(input);
        self.append(AuditEntry {
            timestamp: Utc::now().to_rfc3339(),
            action: AuditAction::ToolCall,
            tool_name: Some(tool_name.to_string()),
            tool_input_preview: Some(preview),
            result,
            session_id: self.session_id.clone(),
            detail: None,
        });
    }

    /// Log a policy decision.
    pub fn log_policy_decision(&self, tool_name: &str, action: AuditAction, reason: &str) {
        self.append(AuditEntry {
            timestamp: Utc::now().to_rfc3339(),
            action,
            tool_name: Some(tool_name.to_string()),
            tool_input_preview: None,
            result: AuditResult::Denied,
            session_id: self.session_id.clone(),
            detail: Some(reason.to_string()),
        });
    }

    /// Log a bash command execution.
    pub fn log_bash_command(&self, command: &str, result: AuditResult) {
        // Truncate long commands
        let cmd_preview = if command.len() > 500 {
            format!("{}...(truncated)", &command[..500])
        } else {
            command.to_string()
        };
        self.append(AuditEntry {
            timestamp: Utc::now().to_rfc3339(),
            action: AuditAction::BashCommand,
            tool_name: Some("bash".to_string()),
            tool_input_preview: Some(cmd_preview),
            result,
            session_id: self.session_id.clone(),
            detail: None,
        });
    }

    /// Log a session lifecycle event.
    pub fn log_session_event(&self, action: AuditAction, detail: Option<String>) {
        self.append(AuditEntry {
            timestamp: Utc::now().to_rfc3339(),
            action,
            tool_name: None,
            tool_input_preview: None,
            result: AuditResult::Success,
            session_id: self.session_id.clone(),
            detail,
        });
    }

    fn append(&self, entry: AuditEntry) {
        match serde_json::to_string(&entry) {
            Ok(line) => {
                if let Err(e) = self.write_line(&line) {
                    warn!("audit log write failed: {e}");
                }
            }
            Err(e) => {
                warn!("audit log serialization failed: {e}");
            }
        }
    }

    fn write_line(&self, line: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Read recent audit entries (last N lines).
    pub fn recent_entries(&self, n: usize) -> Vec<AuditEntry> {
        let content = match fs::read_to_string(&self.log_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        content
            .lines()
            .rev()
            .take(n)
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }
}

/// Sanitize tool input for logging — remove sensitive fields, truncate large values.
fn sanitize_input_preview(input: &serde_json::Value) -> String {
    let mut sanitized = input.clone();

    if let Some(obj) = sanitized.as_object_mut() {
        // Remove potentially sensitive fields
        for key in &["api_key", "token", "password", "secret", "credential", "content"] {
            if obj.contains_key(*key) {
                obj.insert(key.to_string(), serde_json::Value::String("[REDACTED]".to_string()));
            }
        }
        // Truncate long string values
        for (_, val) in obj.iter_mut() {
            if let Some(s) = val.as_str() {
                if s.len() > 200 {
                    *val = serde_json::Value::String(format!("{}...({}B)", &s[..200], s.len()));
                }
            }
        }
    }

    let s = sanitized.to_string();
    if s.len() > 1000 {
        format!("{}...(truncated)", &s[..1000])
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn audit_log_writes_and_reads() {
        let tmp = TempDir::new().unwrap();
        let logger = AuditLogger::new(tmp.path());

        logger.log_tool_call(
            "read_file",
            &serde_json::json!({"path": "test.txt"}),
            AuditResult::Success,
        );
        logger.log_tool_call(
            "bash",
            &serde_json::json!({"command": "ls -la"}),
            AuditResult::Success,
        );

        let entries = logger.recent_entries(10);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn sanitize_redacts_sensitive_fields() {
        let input = serde_json::json!({
            "path": "model.pt",
            "api_key": "sk-12345678",
            "token": "secret-token",
        });
        let preview = sanitize_input_preview(&input);
        assert!(preview.contains("[REDACTED]"));
        assert!(!preview.contains("sk-12345678"));
        assert!(!preview.contains("secret-token"));
    }
}

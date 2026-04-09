use crate::dispatcher::{ToolError, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

/// MCP client bridge — communicates with an external Python MCP server via stdio JSON-RPC.
pub struct McpBridge {
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    child: Mutex<Option<Child>>,
    next_id: Mutex<u64>,
}

#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: u64,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

impl McpBridge {
    /// Spawn a new MCP server process and perform the initialize handshake.
    pub fn spawn(command: &str, args: &[&str]) -> Result<Self, ToolError> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| ToolError::Mcp(format!("failed to spawn MCP server: {e}")))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| ToolError::Mcp("failed to take stdin".to_string()))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| ToolError::Mcp("failed to take stdout".to_string()))?;

        let bridge = Self {
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            child: Mutex::new(Some(child)),
            next_id: Mutex::new(1),
        };

        // Initialize handshake
        let _init_result = bridge.send_request(
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "tcip-agent",
                    "version": "0.1.0"
                }
            })),
        )?;

        // Send initialized notification
        bridge.send_notification("notifications/initialized", None)?;

        Ok(bridge)
    }

    /// Create from pre-existing stdin/stdout handles (for testing or external process).
    pub fn from_handles(stdin: ChildStdin, stdout: ChildStdout, child: Child) -> Self {
        Self {
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
            child: Mutex::new(Some(child)),
            next_id: Mutex::new(1),
        }
    }

    /// Gracefully shut down the MCP server.
    pub fn shutdown(&self) {
        // Send shutdown notification
        let _ = self.send_notification("notifications/cancelled", None);
        // Kill the child process
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    fn next_id(&self) -> u64 {
        let mut id = self.next_id.lock().expect("id lock poisoned");
        let current = *id;
        *id += 1;
        current
    }

    fn send_request(&self, method: &str, params: Option<Value>) -> Result<Value, ToolError> {
        let id = self.next_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };

        let mut payload = serde_json::to_string(&request)
            .map_err(|e| ToolError::Mcp(format!("serialize error: {e}")))?;
        payload.push('\n');

        {
            let mut stdin = self.stdin.lock().expect("stdin lock poisoned");
            stdin
                .write_all(payload.as_bytes())
                .map_err(|e| ToolError::Mcp(format!("write error: {e}")))?;
            stdin
                .flush()
                .map_err(|e| ToolError::Mcp(format!("flush error: {e}")))?;
        }

        // Read response
        let mut line = String::new();
        {
            let mut stdout = self.stdout.lock().expect("stdout lock poisoned");
            // Skip empty lines and notifications
            loop {
                line.clear();
                let bytes_read = stdout
                    .read_line(&mut line)
                    .map_err(|e| ToolError::Mcp(format!("read error: {e}")))?;
                if bytes_read == 0 {
                    return Err(ToolError::Mcp("MCP server closed connection".to_string()));
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // Check if it's a response (has "id" field)
                if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
                    if v.get("id").is_some() {
                        break;
                    }
                    // Otherwise it's a notification, skip
                }
            }
        }

        let response: JsonRpcResponse = serde_json::from_str(line.trim())
            .map_err(|e| ToolError::Mcp(format!("parse response error: {e}")))?;

        if let Some(error) = response.error {
            return Err(ToolError::Mcp(error.message));
        }

        response
            .result
            .ok_or_else(|| ToolError::Mcp("empty result".to_string()))
    }

    fn send_notification(&self, method: &str, params: Option<Value>) -> Result<(), ToolError> {
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(Value::Object(serde_json::Map::new()))
        });

        let mut payload = serde_json::to_string(&request)
            .map_err(|e| ToolError::Mcp(format!("serialize error: {e}")))?;
        payload.push('\n');

        let mut stdin = self.stdin.lock().expect("stdin lock poisoned");
        stdin
            .write_all(payload.as_bytes())
            .map_err(|e| ToolError::Mcp(format!("write error: {e}")))?;
        stdin
            .flush()
            .map_err(|e| ToolError::Mcp(format!("flush error: {e}")))?;

        Ok(())
    }

    /// List all tools available on the MCP server.
    pub async fn list_tools(&self) -> Result<Vec<ToolSpec>, ToolError> {
        let result = self.send_request("tools/list", None)?;

        let tools = result["tools"]
            .as_array()
            .ok_or_else(|| ToolError::Mcp("invalid tools/list response".to_string()))?;

        tools
            .iter()
            .map(|t| {
                let name = t["name"]
                    .as_str()
                    .ok_or_else(|| ToolError::Mcp("tool missing name".to_string()))?
                    .to_string();
                let perm = infer_mcp_permission(&name);
                Ok(ToolSpec {
                    name,
                    description: t["description"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                    input_schema: t["inputSchema"].clone(),
                    required_permission: perm,
                })
            })
            .collect()
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(&self, name: &str, arguments: &Value) -> Result<String, ToolError> {
        let result = self.send_request(
            "tools/call",
            Some(serde_json::json!({
                "name": name,
                "arguments": arguments
            })),
        )?;

        // Check MCP isError flag (per MCP spec, tools/call can set isError: true)
        let is_error = result["isError"].as_bool().unwrap_or(false);

        // MCP tools/call returns { content: [{type: "text", text: "..."}], isError?: bool }
        let output = if let Some(content) = result["content"].as_array() {
            let texts: Vec<&str> = content
                .iter()
                .filter_map(|c| c["text"].as_str())
                .collect();
            texts.join("\n")
        } else {
            serde_json::to_string_pretty(&result)
                .unwrap_or_else(|_| result.to_string())
        };

        if is_error {
            return Err(ToolError::Mcp(output));
        }

        // Also detect dict-style errors from Python tools: {"error": "..."}
        if let Ok(parsed) = serde_json::from_str::<Value>(&output) {
            if let Some(err_msg) = parsed.get("error").and_then(Value::as_str) {
                return Err(ToolError::Mcp(err_msg.to_string()));
            }
        }

        Ok(output)
    }
}

/// Infer permission level for an MCP tool based on its name.
///
/// Read/query/list/get tools → ReadOnly
/// Train/annotate/write/create/delete/update → WorkspaceWrite
/// Unknown → ReadOnly (conservative; policy engine can escalate)
fn infer_mcp_permission(name: &str) -> crate::dispatcher::PermissionLevel {
    use crate::dispatcher::PermissionLevel;
    let lower = name.to_lowercase();

    // Explicitly write-level operations
    let write_patterns = [
        "train", "write", "create", "delete", "update", "annotate",
        "save", "export", "run_pipeline", "start_training", "modify",
    ];
    for pattern in &write_patterns {
        if lower.contains(pattern) {
            return PermissionLevel::WorkspaceWrite;
        }
    }

    // Default to ReadOnly — policy engine handles escalation
    PermissionLevel::ReadOnly
}

impl Drop for McpBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

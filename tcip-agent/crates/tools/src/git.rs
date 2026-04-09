//! Git utilities — lightweight wrapper around git CLI for project tracking.
//!
//! Provides init, auto-commit, status, and diff operations for tracking
//! agent-generated changes. Uses the git CLI binary rather than libgit2
//! to minimize dependencies.

use crate::dispatcher::ToolError;
use std::path::Path;
use std::process::Command;

/// Check if git is available on the system.
pub fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if a directory is inside a git repository.
pub fn is_git_repo(workspace: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(workspace)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Initialize a git repository in the workspace if one doesn't exist.
pub fn init_repo(workspace: &Path) -> Result<String, ToolError> {
    if is_git_repo(workspace) {
        return Ok("Repository already initialized".to_string());
    }

    let output = Command::new("git")
        .args(["init"])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git init failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ToolError::Execution(format!("git init failed: {stderr}")));
    }

    // Create .gitignore with sensible defaults
    let gitignore = workspace.join(".gitignore");
    if !gitignore.exists() {
        let defaults = concat!(
            "# Python\n",
            "__pycache__/\n",
            "*.pyc\n",
            ".venv/\n",
            "*.egg-info/\n",
            "\n",
            "# Rust\n",
            "target/\n",
            "\n",
            "# ML artifacts\n",
            "*.pt\n",
            "*.pth\n",
            "*.onnx\n",
            "checkpoints/\n",
            "runs/\n",
            "\n",
            "# Data\n",
            "data/images/\n",
            "\n",
            "# System\n",
            ".tcip/\n",
        );
        std::fs::write(&gitignore, defaults)
            .map_err(|e| ToolError::Execution(format!("failed to write .gitignore: {e}")))?;
    }

    Ok("Git repository initialized".to_string())
}

/// Get git status summary.
pub fn status(workspace: &Path) -> Result<String, ToolError> {
    if !is_git_repo(workspace) {
        return Ok("Not a git repository".to_string());
    }

    let output = Command::new("git")
        .args(["status", "--short"])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git status failed: {e}")))?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Auto-commit all staged and unstaged changes with a descriptive message.
pub fn auto_commit(workspace: &Path, message: &str) -> Result<String, ToolError> {
    if !is_git_repo(workspace) {
        return Err(ToolError::Execution("Not a git repository".to_string()));
    }

    // Stage all changes
    let add_output = Command::new("git")
        .args(["add", "-A"])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git add failed: {e}")))?;

    if !add_output.status.success() {
        let stderr = String::from_utf8_lossy(&add_output.stderr);
        return Err(ToolError::Execution(format!("git add failed: {stderr}")));
    }

    // Check if there's anything to commit
    let diff_output = Command::new("git")
        .args(["diff", "--cached", "--stat"])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git diff failed: {e}")))?;

    let diff_stat = String::from_utf8_lossy(&diff_output.stdout);
    if diff_stat.trim().is_empty() {
        return Ok("Nothing to commit".to_string());
    }

    // Commit
    let commit_output = Command::new("git")
        .args(["commit", "-m", message, "--author", "tcip-agent <tcip@automated>"])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git commit failed: {e}")))?;

    if !commit_output.status.success() {
        let stderr = String::from_utf8_lossy(&commit_output.stderr);
        return Err(ToolError::Execution(format!("git commit failed: {stderr}")));
    }

    let stdout = String::from_utf8_lossy(&commit_output.stdout);
    Ok(stdout.to_string())
}

/// Get diff of working tree changes.
pub fn diff(workspace: &Path) -> Result<String, ToolError> {
    if !is_git_repo(workspace) {
        return Err(ToolError::Execution("Not a git repository".to_string()));
    }

    let output = Command::new("git")
        .args(["diff", "--stat"])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git diff failed: {e}")))?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Get recent commit log (last N commits).
pub fn log_recent(workspace: &Path, count: usize) -> Result<String, ToolError> {
    if !is_git_repo(workspace) {
        return Err(ToolError::Execution("Not a git repository".to_string()));
    }

    let output = Command::new("git")
        .args(["log", "--oneline", &format!("-{count}")])
        .current_dir(workspace)
        .output()
        .map_err(|e| ToolError::Execution(format!("git log failed: {e}")))?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_available_check() {
        // This should work on any dev machine
        let available = git_available();
        // Don't assert true — CI might not have git
        // Just verify the function doesn't panic
        let _ = available;
    }
}

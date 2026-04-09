use crate::dispatcher::{PermissionLevel, ToolError, ToolSpec};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "read_file".to_string(),
            description: "Read the contents of a file. Specify start_line and end_line for partial reads.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to workspace)"},
                    "start_line": {"type": "integer", "description": "Start line (1-based, optional)"},
                    "end_line": {"type": "integer", "description": "End line (1-based, inclusive, optional)"}
                },
                "required": ["path"]
            }),
            required_permission: PermissionLevel::ReadOnly,
        },
        ToolSpec {
            name: "write_file".to_string(),
            description: "Write content to a file, creating directories as needed.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to workspace)"},
                    "content": {"type": "string", "description": "File content to write"}
                },
                "required": ["path", "content"]
            }),
            required_permission: PermissionLevel::WorkspaceWrite,
        },
        ToolSpec {
            name: "edit_file".to_string(),
            description: "Replace an exact string in a file with a new string.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to workspace)"},
                    "old_string": {"type": "string", "description": "Exact text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_string", "new_string"]
            }),
            required_permission: PermissionLevel::WorkspaceWrite,
        },
    ]
}

fn resolve_path(path_str: &str, workspace: &Path) -> Result<PathBuf, ToolError> {
    let path = Path::new(path_str);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    };

    // Security: ensure path stays within workspace
    let canonical_workspace = workspace
        .canonicalize()
        .map_err(|e| ToolError::Execution(format!("cannot resolve workspace: {e}")))?;

    // Try to canonicalize the full path first (file exists)
    let canonical_path = if let Ok(p) = resolved.canonicalize() {
        p
    } else {
        // File doesn't exist yet — canonicalize parent + append filename
        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).ok();
            if let Ok(canonical_parent) = parent.canonicalize() {
                if let Some(filename) = resolved.file_name() {
                    canonical_parent.join(filename)
                } else {
                    resolved.clone()
                }
            } else {
                resolved.clone()
            }
        } else {
            resolved.clone()
        }
    };

    if !canonical_path.starts_with(&canonical_workspace) {
        return Err(ToolError::Execution(format!(
            "path escapes workspace: {}",
            path_str
        )));
    }

    Ok(canonical_path)
}

pub fn read_file(input: &Value, workspace: &Path) -> Result<String, ToolError> {
    let path_str = input["path"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("path is required".to_string()))?;
    let path = resolve_path(path_str, workspace)?;

    let content =
        std::fs::read_to_string(&path).map_err(|e| ToolError::Execution(e.to_string()))?;

    let start = input["start_line"].as_u64().map(|n| n as usize);
    let end = input["end_line"].as_u64().map(|n| n as usize);

    match (start, end) {
        (Some(s), Some(e)) => {
            let lines: Vec<&str> = content.lines().collect();
            let s = s.saturating_sub(1).min(lines.len());
            let e = e.min(lines.len());
            Ok(lines[s..e].join("\n"))
        }
        (Some(s), None) => {
            let lines: Vec<&str> = content.lines().collect();
            let s = s.saturating_sub(1).min(lines.len());
            Ok(lines[s..].join("\n"))
        }
        _ => Ok(content),
    }
}

pub fn write_file(input: &Value, workspace: &Path) -> Result<String, ToolError> {
    let path_str = input["path"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("path is required".to_string()))?;
    let content = input["content"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("content is required".to_string()))?;

    let path = resolve_path(path_str, workspace)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ToolError::Execution(e.to_string()))?;
    }

    std::fs::write(&path, content).map_err(|e| ToolError::Execution(e.to_string()))?;

    Ok(format!("wrote {} bytes to {}", content.len(), path_str))
}

pub fn edit_file(input: &Value, workspace: &Path) -> Result<String, ToolError> {
    let path_str = input["path"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("path is required".to_string()))?;
    let old_string = input["old_string"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("old_string is required".to_string()))?;
    let new_string = input["new_string"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("new_string is required".to_string()))?;

    let path = resolve_path(path_str, workspace)?;
    let content =
        std::fs::read_to_string(&path).map_err(|e| ToolError::Execution(e.to_string()))?;

    let count = content.matches(old_string).count();
    if count == 0 {
        return Err(ToolError::Execution(
            "old_string not found in file".to_string(),
        ));
    }
    if count > 1 {
        return Err(ToolError::Execution(format!(
            "old_string matches {count} locations; must be unique"
        )));
    }

    let new_content = content.replacen(old_string, new_string, 1);
    std::fs::write(&path, new_content).map_err(|e| ToolError::Execution(e.to_string()))?;

    Ok(format!("edited {path_str}"))
}

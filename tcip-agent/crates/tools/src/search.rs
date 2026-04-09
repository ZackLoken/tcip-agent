use crate::dispatcher::{PermissionLevel, ToolError, ToolSpec};
use serde_json::{json, Value};
use std::path::Path;

pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "grep_search".to_string(),
            description: "Search for a pattern in files. Returns matching lines with context."
                .to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex)"},
                    "path": {"type": "string", "description": "Directory or file to search (relative to workspace)"},
                    "include": {"type": "string", "description": "Glob pattern to filter files (e.g. *.py)"}
                },
                "required": ["pattern"]
            }),
            required_permission: PermissionLevel::ReadOnly,
        },
        ToolSpec {
            name: "glob_search".to_string(),
            description: "Find files matching a glob pattern.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.rs)"}
                },
                "required": ["pattern"]
            }),
            required_permission: PermissionLevel::ReadOnly,
        },
    ]
}

pub fn grep_search(input: &Value, workspace: &Path) -> Result<String, ToolError> {
    let pattern = input["pattern"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("pattern is required".to_string()))?;

    let search_path = input["path"]
        .as_str()
        .map(|p| workspace.join(p))
        .unwrap_or_else(|| workspace.to_path_buf());

    let include_pattern = input["include"].as_str();

    let regex = regex::Regex::new(pattern)
        .map_err(|e| ToolError::InvalidInput(format!("invalid regex: {e}")))?;

    let mut results = Vec::new();
    let max_results = 100;

    search_dir(&search_path, &regex, include_pattern, &mut results, max_results)
        .map_err(|e| ToolError::Execution(e.to_string()))?;

    if results.is_empty() {
        Ok("No matches found.".to_string())
    } else {
        Ok(results.join("\n"))
    }
}

fn search_dir(
    dir: &Path,
    regex: &regex::Regex,
    include: Option<&str>,
    results: &mut Vec<String>,
    max: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= max {
        return Ok(());
    }

    if dir.is_file() {
        return search_file(dir, regex, results, max);
    }

    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Skip hidden directories and common ignores
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') || name == "node_modules" || name == "target" || name == "__pycache__" {
                continue;
            }
        }

        if path.is_dir() {
            search_dir(&path, regex, include, results, max)?;
        } else if path.is_file() {
            if let Some(pattern) = include {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let glob_ext = pattern.trim_start_matches("*.");
                    if ext != glob_ext {
                        continue;
                    }
                }
            }
            search_file(&path, regex, results, max)?;
        }
    }
    Ok(())
}

fn search_file(
    path: &Path,
    regex: &regex::Regex,
    results: &mut Vec<String>,
    max: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= max {
        return Ok(());
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()), // Skip binary / unreadable files
    };

    for (line_num, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            let display_path = path.display();
            results.push(format!("{display_path}:{}: {}", line_num + 1, line.trim()));
            if results.len() >= max {
                break;
            }
        }
    }

    Ok(())
}

pub fn glob_search(input: &Value, workspace: &Path) -> Result<String, ToolError> {
    let pattern = input["pattern"]
        .as_str()
        .ok_or_else(|| ToolError::InvalidInput("pattern is required".to_string()))?;

    let full_pattern = workspace.join(pattern).display().to_string();
    let paths: Vec<String> = glob::glob(&full_pattern)
        .map_err(|e| ToolError::InvalidInput(format!("invalid glob: {e}")))?
        .filter_map(Result::ok)
        .take(200)
        .map(|p| {
            p.strip_prefix(workspace)
                .unwrap_or(&p)
                .display()
                .to_string()
        })
        .collect();

    if paths.is_empty() {
        Ok("No files found.".to_string())
    } else {
        Ok(paths.join("\n"))
    }
}

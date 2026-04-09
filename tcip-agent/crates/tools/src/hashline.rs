//! Hashline edit tool — content-hash-tagged line editing for reliable file modifications.
//!
//! Instead of matching exact old_text/new_text, lines are identified by line number + content hash.
//! This avoids stale-context errors when the agent hallucinates surrounding text.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// A single hashline-tagged line.
#[derive(Debug, Clone)]
pub struct HashLine {
    pub line_number: u32,
    pub hash: String,
    pub content: String,
}

/// An edit targeting a specific line by its hash.
#[derive(Debug, Clone)]
pub struct HashLineEdit {
    pub line: u32,
    pub hash: String,
    pub new_content: String,
}

/// Compute a 4-character content hash for a line.
///
/// Hashes the trimmed content (ignoring leading whitespace) for stability
/// when indentation changes.
pub fn line_hash(content: &str) -> String {
    let mut hasher = DefaultHasher::new();
    content.trim_start().hash(&mut hasher);
    let h = hasher.finish();
    format!("{:04x}", h & 0xFFFF)
}

/// Read a file and return hashline-tagged output.
pub fn hashline_read(path: &Path, start_line: u32, end_line: u32) -> Result<Vec<HashLine>, HashLineError> {
    let content = fs::read_to_string(path).map_err(HashLineError::Io)?;
    let lines: Vec<&str> = content.lines().collect();

    let start = (start_line as usize).saturating_sub(1);
    let end = (end_line as usize).min(lines.len());

    if start >= lines.len() {
        return Err(HashLineError::OutOfRange {
            requested: start_line,
            total: lines.len() as u32,
        });
    }

    let mut result = Vec::new();
    for (i, line) in lines[start..end].iter().enumerate() {
        let line_num = start as u32 + i as u32 + 1;
        result.push(HashLine {
            line_number: line_num,
            hash: line_hash(line),
            content: line.to_string(),
        });
    }
    Ok(result)
}

/// Format hashline output for display.
pub fn format_hashlines(lines: &[HashLine]) -> String {
    lines
        .iter()
        .map(|l| format!("{}#{}| {}", l.line_number, l.hash, l.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Apply hashline edits atomically. All edits are validated before any are applied.
pub fn hashline_edit(path: &Path, edits: &[HashLineEdit]) -> Result<String, HashLineError> {
    let content = fs::read_to_string(path).map_err(HashLineError::Io)?;
    let mut lines: Vec<String> = content.lines().map(String::from).collect();

    // Phase 1: Validate all edits
    for edit in edits {
        let idx = edit.line as usize - 1;
        if idx >= lines.len() {
            return Err(HashLineError::OutOfRange {
                requested: edit.line,
                total: lines.len() as u32,
            });
        }

        let current_hash = line_hash(&lines[idx]);
        if current_hash != edit.hash {
            return Err(HashLineError::HashMismatch {
                line: edit.line,
                expected: edit.hash.clone(),
                actual: current_hash,
            });
        }
    }

    // Phase 2: Apply all edits
    for edit in edits {
        let idx = edit.line as usize - 1;
        lines[idx] = edit.new_content.clone();
    }

    let new_content = lines.join("\n");
    // Preserve trailing newline if original had one
    let new_content = if content.ends_with('\n') {
        format!("{new_content}\n")
    } else {
        new_content
    };

    fs::write(path, &new_content).map_err(HashLineError::Io)?;

    Ok(format!("Applied {} edit(s) to {}", edits.len(), path.display()))
}

#[derive(Debug, thiserror::Error)]
pub enum HashLineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Line {requested} out of range (file has {total} lines)")]
    OutOfRange { requested: u32, total: u32 },
    #[error("Hash mismatch on line {line}: expected #{expected}, got #{actual}. Re-read with hashline_read.")]
    HashMismatch {
        line: u32,
        expected: String,
        actual: String,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_file(dir: &Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut f = fs::File::create(&path).unwrap();
        write!(f, "{}", content).unwrap();
        path
    }

    #[test]
    fn line_hash_ignores_leading_whitespace() {
        assert_eq!(line_hash("  hello"), line_hash("    hello"));
        assert_ne!(line_hash("hello"), line_hash("world"));
    }

    #[test]
    fn hashline_read_returns_tagged_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_test_file(tmp.path(), "line1\nline2\nline3\nline4\n");

        let result = hashline_read(&path, 2, 3).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].line_number, 2);
        assert_eq!(result[0].content, "line2");
        assert_eq!(result[1].line_number, 3);
    }

    #[test]
    fn hashline_edit_applies_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_test_file(tmp.path(), "alpha\nbeta\ngamma\n");

        let lines = hashline_read(&path, 1, 3).unwrap();
        let beta_hash = lines[1].hash.clone();

        hashline_edit(
            &path,
            &[HashLineEdit {
                line: 2,
                hash: beta_hash,
                new_content: "BETA_MODIFIED".to_string(),
            }],
        )
        .unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("BETA_MODIFIED"));
        assert!(content.contains("alpha"));
        assert!(content.contains("gamma"));
    }

    #[test]
    fn hashline_edit_rejects_mismatched_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_test_file(tmp.path(), "alpha\nbeta\ngamma\n");

        let result = hashline_edit(
            &path,
            &[HashLineEdit {
                line: 2,
                hash: "xxxx".to_string(),
                new_content: "new".to_string(),
            }],
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Hash mismatch"));
    }

    #[test]
    fn hashline_edit_atomic_all_or_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_test_file(tmp.path(), "alpha\nbeta\ngamma\n");

        let lines = hashline_read(&path, 1, 3).unwrap();

        // First edit valid, second has wrong hash
        let result = hashline_edit(
            &path,
            &[
                HashLineEdit {
                    line: 1,
                    hash: lines[0].hash.clone(),
                    new_content: "ALPHA_NEW".to_string(),
                },
                HashLineEdit {
                    line: 2,
                    hash: "xxxx".to_string(),
                    new_content: "BETA_NEW".to_string(),
                },
            ],
        );
        assert!(result.is_err());

        // File should be unchanged (atomic — nothing applied)
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("alpha"));
        assert!(!content.contains("ALPHA_NEW"));
    }

    #[test]
    fn format_hashlines_output() {
        let lines = vec![
            HashLine {
                line_number: 11,
                hash: "aB3f".to_string(),
                content: "data:".to_string(),
            },
            HashLine {
                line_number: 12,
                hash: "kQ9x".to_string(),
                content: "  train: ./data/train".to_string(),
            },
        ];
        let output = format_hashlines(&lines);
        assert!(output.contains("11#aB3f| data:"));
        assert!(output.contains("12#kQ9x|   train: ./data/train"));
    }
}

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::warn;

/// Tracks consecutive tool calls to detect infinite loops.
///
/// Detection rules:
/// - 3+ consecutive identical (tool_name, input_hash) → hard break with warning
/// - 5+ consecutive same tool_name (any input) → advisory warning
pub struct LoopDetector {
    /// Sliding window of recent tool call signatures.
    window: VecDeque<ToolSignature>,
    /// Number of identical consecutive calls before breaking.
    identical_threshold: usize,
    /// Number of same-tool consecutive calls before advisory.
    same_tool_threshold: usize,
    /// Max window size retained.
    max_window: usize,
}

#[derive(Debug, Clone)]
struct ToolSignature {
    tool_name: String,
    input_hash: u64,
}

/// Result of checking a tool call against the loop detector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopCheckResult {
    /// No loop detected, proceed normally.
    Ok,
    /// Advisory: same tool called many times, inject hint to the model.
    Advisory { message: String },
    /// Hard break: identical calls detected, must stop the inner loop.
    Break { message: String },
}

impl LoopDetector {
    /// Create a new loop detector with default thresholds.
    pub fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(10),
            identical_threshold: 3,
            same_tool_threshold: 5,
            max_window: 10,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(identical_threshold: usize, same_tool_threshold: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(same_tool_threshold + 2),
            identical_threshold,
            same_tool_threshold,
            max_window: same_tool_threshold + 5,
        }
    }

    /// Record a tool call and check for loops.
    ///
    /// Call this before executing each tool. If the result is `Break`,
    /// the caller should stop the inner loop and inject a warning.
    pub fn check(&mut self, tool_name: &str, input: &serde_json::Value) -> LoopCheckResult {
        let sig = ToolSignature {
            tool_name: tool_name.to_string(),
            input_hash: hash_input(input),
        };

        self.window.push_back(sig.clone());
        if self.window.len() > self.max_window {
            self.window.pop_front();
        }

        // Check for identical consecutive calls (same tool + same input hash)
        let identical_count = self
            .window
            .iter()
            .rev()
            .take_while(|s| s.tool_name == sig.tool_name && s.input_hash == sig.input_hash)
            .count();

        if identical_count >= self.identical_threshold {
            let msg = format!(
                "Loop detected: tool '{}' called {} times with identical input. \
                 Breaking loop — try a different approach or tool.",
                tool_name, identical_count
            );
            warn!("{msg}");
            return LoopCheckResult::Break { message: msg };
        }

        // Check for same tool name (any input)
        let same_tool_count = self
            .window
            .iter()
            .rev()
            .take_while(|s| s.tool_name == sig.tool_name)
            .count();

        if same_tool_count >= self.same_tool_threshold {
            let msg = format!(
                "Advisory: tool '{}' called {} consecutive times. \
                 Consider whether a different tool would be more effective.",
                tool_name, same_tool_count
            );
            warn!("{msg}");
            return LoopCheckResult::Advisory { message: msg };
        }

        LoopCheckResult::Ok
    }

    /// Reset the detector (e.g. at turn boundaries).
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn hash_input(input: &serde_json::Value) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Use canonical JSON string for hashing
    let canonical = serde_json::to_string(input).unwrap_or_default();
    canonical.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn no_loop_for_different_tools() {
        let mut detector = LoopDetector::new();
        assert_eq!(detector.check("read_file", &json!({"path": "a.rs"})), LoopCheckResult::Ok);
        assert_eq!(detector.check("write_file", &json!({"path": "b.rs"})), LoopCheckResult::Ok);
        assert_eq!(detector.check("grep_search", &json!({"pattern": "foo"})), LoopCheckResult::Ok);
    }

    #[test]
    fn detects_identical_loop() {
        let mut detector = LoopDetector::new();
        let input = json!({"path": "same.rs"});
        assert_eq!(detector.check("read_file", &input), LoopCheckResult::Ok);
        assert_eq!(detector.check("read_file", &input), LoopCheckResult::Ok);
        match detector.check("read_file", &input) {
            LoopCheckResult::Break { .. } => {}
            other => panic!("expected Break, got {other:?}"),
        }
    }

    #[test]
    fn advisory_for_same_tool_different_inputs() {
        let mut detector = LoopDetector::new();
        for i in 0..4 {
            assert_eq!(
                detector.check("read_file", &json!({"path": format!("file_{i}.rs")})),
                LoopCheckResult::Ok
            );
        }
        match detector.check("read_file", &json!({"path": "file_4.rs"})) {
            LoopCheckResult::Advisory { .. } => {}
            other => panic!("expected Advisory, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut detector = LoopDetector::new();
        let input = json!({"path": "same.rs"});
        detector.check("read_file", &input);
        detector.check("read_file", &input);
        detector.reset();
        // After reset, counter should be back to 1
        assert_eq!(detector.check("read_file", &input), LoopCheckResult::Ok);
    }

    #[test]
    fn custom_thresholds() {
        let mut detector = LoopDetector::with_thresholds(2, 3);
        let input = json!({"x": 1});
        assert_eq!(detector.check("tool_a", &input), LoopCheckResult::Ok);
        match detector.check("tool_a", &input) {
            LoopCheckResult::Break { .. } => {}
            other => panic!("expected Break at threshold 2, got {other:?}"),
        }
    }
}

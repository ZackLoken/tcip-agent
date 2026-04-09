use serde::{Deserialize, Serialize};
use tracing::debug;

/// Action taken when a policy rule matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleAction {
    Allow,
    Deny,
    Ask,
}

/// A single permission rule matching tool and argument patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRule {
    /// Glob pattern for tool name (e.g. "write_file", "mcp__*", "bash").
    pub tool_pattern: String,
    /// Optional glob pattern for tool input (e.g. "rm -rf *", "data/predictions/**").
    pub arg_pattern: Option<String>,
    /// Action to take when this rule matches.
    pub action: RuleAction,
    /// Human-readable reason (displayed on deny/ask).
    pub reason: Option<String>,
}

/// Result of evaluating a tool call against the policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyResult {
    /// Policy explicitly allows this operation.
    Allow,
    /// Policy explicitly denies this operation.
    Deny { reason: String },
    /// Policy requires human confirmation.
    Ask { reason: String },
    /// No policy rule matched — fall through to mode ceiling.
    NoMatch,
}

/// Rule-based permission policy engine.
///
/// Rules are evaluated in order; first match wins.
/// If no rule matches, the result is `NoMatch` and the caller
/// should fall back to the existing mode-based permission ceiling.
pub struct PermissionPolicy {
    rules: Vec<PermissionRule>,
}

impl PermissionPolicy {
    /// Create an empty policy (everything falls through to mode ceiling).
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create a policy with the default ML/CV safety rules.
    pub fn with_defaults() -> Self {
        Self {
            rules: default_rules(),
        }
    }

    /// Add a rule to the policy.
    pub fn add_rule(&mut self, rule: PermissionRule) {
        self.rules.push(rule);
    }

    /// Evaluate a tool call against the policy.
    pub fn evaluate(&self, tool_name: &str, tool_input: &str) -> PolicyResult {
        for rule in &self.rules {
            if !glob_match(&rule.tool_pattern, tool_name) {
                continue;
            }

            if let Some(ref arg_pat) = rule.arg_pattern {
                if !glob_match(arg_pat, tool_input) {
                    continue;
                }
            }

            let reason = rule
                .reason
                .clone()
                .unwrap_or_else(|| format!("matched rule: {}", rule.tool_pattern));

            debug!("policy match: tool={tool_name} rule={} → {:?}", rule.tool_pattern, rule.action);

            return match rule.action {
                RuleAction::Allow => PolicyResult::Allow,
                RuleAction::Deny => PolicyResult::Deny { reason },
                RuleAction::Ask => PolicyResult::Ask { reason },
            };
        }

        PolicyResult::NoMatch
    }

    /// Load policy rules from a TOML file.
    ///
    /// Expected format:
    /// ```toml
    /// [[rules]]
    /// tool_pattern = "bash"
    /// arg_pattern = "rm -rf *"
    /// action = "Deny"
    /// reason = "Destructive command blocked"
    /// ```
    pub fn load_from_toml(path: &std::path::Path) -> Result<Self, PolicyError> {
        let content = std::fs::read_to_string(path).map_err(PolicyError::Io)?;
        let config: PolicyConfig =
            toml::from_str(&content).map_err(PolicyError::Parse)?;
        Ok(Self {
            rules: config.rules,
        })
    }
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[derive(Deserialize)]
struct PolicyConfig {
    #[serde(default)]
    rules: Vec<PermissionRule>,
}

/// Simple glob matching: `*` matches any sequence of characters.
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    // Split pattern on '*' and match segments
    let parts: Vec<&str> = pattern.split('*').collect();

    if parts.len() == 1 {
        // No wildcards — exact match
        return pattern == text;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            // First segment must match at start
            if !text.starts_with(part) {
                return false;
            }
            pos = part.len();
        } else if i == parts.len() - 1 {
            // Last segment must match at end
            if !text[pos..].ends_with(part) {
                return false;
            }
            pos = text.len();
        } else {
            // Middle segments must appear in order
            match text[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }

    true
}

/// Default safety rules for ML/CV workflows.
fn default_rules() -> Vec<PermissionRule> {
    vec![
        // ── Source-code protection: deny writes to agent's own codebase ──
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*crates/*.rs".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot overwrite agent Rust source code".to_string()),
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*Cargo.toml".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot overwrite Cargo.toml".to_string()),
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*Cargo.lock".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot overwrite Cargo.lock".to_string()),
        },
        PermissionRule {
            tool_pattern: "edit_file".to_string(),
            arg_pattern: Some("*crates/*.rs".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot edit agent Rust source code".to_string()),
        },
        PermissionRule {
            tool_pattern: "edit_file".to_string(),
            arg_pattern: Some("*Cargo.toml".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot edit Cargo.toml".to_string()),
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*packages/*.py".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot overwrite Python package source code".to_string()),
        },
        PermissionRule {
            tool_pattern: "edit_file".to_string(),
            arg_pattern: Some("*packages/*.py".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot edit Python package source code".to_string()),
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*pyproject.toml".to_string()),
            action: RuleAction::Deny,
            reason: Some("Cannot overwrite pyproject.toml".to_string()),
        },
        // ── Protect skill files ──
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("skills/*.md".to_string()),
            action: RuleAction::Ask,
            reason: Some("Modifying skill files requires approval".to_string()),
        },
        // ── Allow writes to designated output areas ──
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*data/predictions*".to_string()),
            action: RuleAction::Allow,
            reason: None,
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*data/labels*".to_string()),
            action: RuleAction::Allow,
            reason: None,
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*.tcip/*".to_string()),
            action: RuleAction::Allow,
            reason: None,
        },
        PermissionRule {
            tool_pattern: "write_file".to_string(),
            arg_pattern: Some("*projects/*".to_string()),
            action: RuleAction::Allow,
            reason: None,
        },
        // ── Block destructive shell commands ──
        PermissionRule {
            tool_pattern: "bash".to_string(),
            arg_pattern: Some("*rm -rf*".to_string()),
            action: RuleAction::Deny,
            reason: Some("Destructive command blocked by policy".to_string()),
        },
        PermissionRule {
            tool_pattern: "bash".to_string(),
            arg_pattern: Some("*format*".to_string()),
            action: RuleAction::Deny,
            reason: Some("Potentially destructive formatting command blocked".to_string()),
        },
        // ── Training/HPO require approval ──
        PermissionRule {
            tool_pattern: "mcp__launch_training".to_string(),
            arg_pattern: None,
            action: RuleAction::Ask,
            reason: Some("Training requires GPU approval".to_string()),
        },
        PermissionRule {
            tool_pattern: "mcp__run_hpo".to_string(),
            arg_pattern: None,
            action: RuleAction::Ask,
            reason: Some("HPO requires resource approval".to_string()),
        },
    ]
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_exact_match() {
        assert!(glob_match("bash", "bash"));
        assert!(!glob_match("bash", "bash2"));
    }

    #[test]
    fn glob_star_match() {
        assert!(glob_match("mcp__*", "mcp__launch_training"));
        assert!(glob_match("mcp__*", "mcp__anything"));
        assert!(!glob_match("mcp__*", "bash"));
    }

    #[test]
    fn glob_contains_match() {
        assert!(glob_match("*rm -rf*", "sudo rm -rf /tmp/data"));
        assert!(!glob_match("*rm -rf*", "ls -la"));
    }

    #[test]
    fn glob_star_only() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*", ""));
    }

    #[test]
    fn default_policy_blocks_rm_rf() {
        let policy = PermissionPolicy::with_defaults();
        match policy.evaluate("bash", "rm -rf /important/data") {
            PolicyResult::Deny { reason } => assert!(reason.contains("Destructive")),
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[test]
    fn default_policy_asks_for_training() {
        let policy = PermissionPolicy::with_defaults();
        match policy.evaluate("mcp__launch_training", "{}") {
            PolicyResult::Ask { reason } => assert!(reason.contains("GPU")),
            other => panic!("expected Ask, got {other:?}"),
        }
    }

    #[test]
    fn default_policy_allows_predictions() {
        let policy = PermissionPolicy::with_defaults();
        match policy.evaluate("write_file", "data/predictions/out.json") {
            PolicyResult::Allow => {}
            other => panic!("expected Allow, got {other:?}"),
        }
    }

    #[test]
    fn no_match_falls_through() {
        let policy = PermissionPolicy::with_defaults();
        match policy.evaluate("read_file", "src/main.rs") {
            PolicyResult::NoMatch => {}
            other => panic!("expected NoMatch, got {other:?}"),
        }
    }

    #[test]
    fn custom_rule_overrides() {
        let mut policy = PermissionPolicy::new();
        policy.add_rule(PermissionRule {
            tool_pattern: "read_file".to_string(),
            arg_pattern: Some("*secrets*".to_string()),
            action: RuleAction::Deny,
            reason: Some("No reading secrets".to_string()),
        });

        match policy.evaluate("read_file", "config/secrets.toml") {
            PolicyResult::Deny { .. } => {}
            other => panic!("expected Deny, got {other:?}"),
        }

        // Non-secret file falls through
        match policy.evaluate("read_file", "config/app.toml") {
            PolicyResult::NoMatch => {}
            other => panic!("expected NoMatch, got {other:?}"),
        }
    }

    #[test]
    fn first_match_wins() {
        let mut policy = PermissionPolicy::new();
        policy.add_rule(PermissionRule {
            tool_pattern: "bash".to_string(),
            arg_pattern: Some("*echo*".to_string()),
            action: RuleAction::Allow,
            reason: None,
        });
        policy.add_rule(PermissionRule {
            tool_pattern: "bash".to_string(),
            arg_pattern: None,
            action: RuleAction::Deny,
            reason: Some("bash blocked".to_string()),
        });

        // "echo hello" matches first rule → Allow
        assert_eq!(policy.evaluate("bash", "echo hello"), PolicyResult::Allow);
        // "ls" matches second rule → Deny
        match policy.evaluate("bash", "ls") {
            PolicyResult::Deny { .. } => {}
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[test]
    fn workspace_isolation_blocks_source_writes() {
        let policy = PermissionPolicy::with_defaults();

        // Cannot overwrite Rust source
        match policy.evaluate("write_file", "tcip-agent/crates/tools/src/bash.rs") {
            PolicyResult::Deny { reason } => assert!(reason.contains("Rust source")),
            other => panic!("expected Deny for .rs write, got {other:?}"),
        }

        // Cannot edit Cargo.toml
        match policy.evaluate("edit_file", "tcip-agent/Cargo.toml") {
            PolicyResult::Deny { reason } => assert!(reason.contains("Cargo.toml")),
            other => panic!("expected Deny for Cargo.toml edit, got {other:?}"),
        }

        // Cannot overwrite Python package source
        match policy.evaluate("write_file", "packages/tcip-mcp/src/tcip_mcp/tools/training_tools.py") {
            PolicyResult::Deny { reason } => assert!(reason.contains("Python package")),
            other => panic!("expected Deny for .py write, got {other:?}"),
        }
    }

    #[test]
    fn workspace_isolation_allows_output_dirs() {
        let policy = PermissionPolicy::with_defaults();

        // Can write to data/labels
        match policy.evaluate("write_file", "data/labels/detect/IMG_0133.txt") {
            PolicyResult::Allow => {}
            other => panic!("expected Allow for labels, got {other:?}"),
        }

        // Can write to .tcip directory
        match policy.evaluate("write_file", ".tcip/sessions/abc.jsonl") {
            PolicyResult::Allow => {}
            other => panic!("expected Allow for .tcip, got {other:?}"),
        }

        // Can write to projects directory
        match policy.evaluate("write_file", "projects/exp01/config.yaml") {
            PolicyResult::Allow => {}
            other => panic!("expected Allow for projects, got {other:?}"),
        }
    }
}

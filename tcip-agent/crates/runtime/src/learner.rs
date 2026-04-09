//! Self-learning: extract reusable patterns from completed sessions, match them to new tasks.
//!
//! Learned skills are persisted as `.md` files with YAML frontmatter in `.tcip/learned_skills/`.
//! On each session start, the best-matching learned skills are injected via `ContextCollector`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// A learned skill extracted from a successful session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedSkill {
    pub name: String,
    pub skill_type: PatternType,
    pub triggers: Vec<String>,
    pub confidence: f64,
    pub learned_from: String,
    pub learned_at: DateTime<Utc>,
    pub body: String,
    /// Number of times this skill has been injected.
    pub use_count: u32,
    /// Last time this skill was injected.
    pub last_used: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatternType {
    ProblemSolution,
    Technique,
    Optimization,
    WorkflowSequence,
}

impl std::fmt::Display for PatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProblemSolution => write!(f, "problem-solution"),
            Self::Technique => write!(f, "technique"),
            Self::Optimization => write!(f, "optimization"),
            Self::WorkflowSequence => write!(f, "workflow-sequence"),
        }
    }
}

/// Summary of a completed session, used as input to the extractor.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub session_id: String,
    pub tool_calls: Vec<ToolCallSummary>,
    pub had_errors: bool,
    pub error_count: usize,
    pub final_text: String,
}

#[derive(Debug, Clone)]
pub struct ToolCallSummary {
    pub name: String,
    pub input_keys: Vec<String>,
    pub succeeded: bool,
}

// ---------------------------------------------------------------------------
// Skill Extractor
// ---------------------------------------------------------------------------

/// Extracts reusable patterns from completed sessions.
pub struct SkillExtractor {
    /// Minimum tool calls for a session to be worth extracting from.
    min_tool_calls: usize,
    /// Maximum error ratio allowed.
    max_error_ratio: f64,
}

impl SkillExtractor {
    pub fn new() -> Self {
        Self {
            min_tool_calls: 3,
            max_error_ratio: 0.25,
        }
    }

    /// Check if a session is worth extracting skills from.
    pub fn is_extractable(&self, summary: &SessionSummary) -> bool {
        let total = summary.tool_calls.len();
        if total < self.min_tool_calls {
            return false;
        }
        let error_ratio = summary.error_count as f64 / total as f64;
        error_ratio <= self.max_error_ratio
    }

    /// Extract a learned skill from a session summary.
    ///
    /// Returns `None` if the session doesn't meet extraction criteria.
    pub fn extract(&self, summary: &SessionSummary) -> Option<LearnedSkill> {
        if !self.is_extractable(summary) {
            return None;
        }

        let tool_sequence: Vec<&str> = summary
            .tool_calls
            .iter()
            .filter(|tc| tc.succeeded)
            .map(|tc| tc.name.as_str())
            .collect();

        // Determine pattern type from tool sequence
        let pattern_type = classify_pattern(&tool_sequence);

        // Build triggers from tool names and input keys
        let mut triggers: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for tc in &summary.tool_calls {
            if seen.insert(tc.name.clone()) {
                triggers.push(tc.name.clone());
            }
            for key in &tc.input_keys {
                if seen.insert(key.clone()) {
                    triggers.push(key.clone());
                }
            }
        }
        triggers.truncate(15); // Keep it manageable

        // Build name slug from first few tools
        let slug: String = tool_sequence
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("-");
        let name = if slug.is_empty() {
            format!("learned-{}", &summary.session_id[..8.min(summary.session_id.len())])
        } else {
            slug
        };

        // Build body from tool sequence description
        let body = format!(
            "Learned workflow ({}):\n{}",
            pattern_type,
            tool_sequence
                .iter()
                .enumerate()
                .map(|(i, t)| format!("{}. {}", i + 1, t))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Some(LearnedSkill {
            name,
            skill_type: pattern_type,
            triggers,
            confidence: 0.8,
            learned_from: summary.session_id.clone(),
            learned_at: Utc::now(),
            body,
            use_count: 0,
            last_used: None,
        })
    }
}

fn classify_pattern(tools: &[&str]) -> PatternType {
    let has_train = tools.iter().any(|t| t.contains("train"));
    let has_eval = tools.iter().any(|t| t.contains("eval") || t.contains("metric"));
    let has_annotate = tools.iter().any(|t| t.contains("annot") || t.contains("label"));

    if has_train && has_eval {
        PatternType::Optimization
    } else if has_annotate {
        PatternType::WorkflowSequence
    } else if tools.len() >= 5 {
        PatternType::WorkflowSequence
    } else {
        PatternType::ProblemSolution
    }
}

// ---------------------------------------------------------------------------
// Skill Matcher
// ---------------------------------------------------------------------------

/// Scores learned skills against a new task description.
pub struct SkillMatcher {
    /// Maximum number of skills to return.
    top_n: usize,
    /// Minimum score to include a skill.
    min_score: f64,
}

impl SkillMatcher {
    pub fn new() -> Self {
        Self {
            top_n: 3,
            min_score: 0.2,
        }
    }

    /// Score all learned skills against a user message and return the top N.
    pub fn match_skills<'a>(
        &self,
        skills: &'a [LearnedSkill],
        user_message: &str,
    ) -> Vec<(&'a LearnedSkill, f64)> {
        let words: Vec<&str> = user_message.split_whitespace().collect();
        let lower_msg = user_message.to_lowercase();

        let mut scored: Vec<(&LearnedSkill, f64)> = skills
            .iter()
            .map(|skill| {
                let score = self.score_skill(skill, &words, &lower_msg);
                (skill, score)
            })
            .filter(|(_, score)| *score >= self.min_score)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.top_n);
        scored
    }

    fn score_skill(&self, skill: &LearnedSkill, words: &[&str], lower_msg: &str) -> f64 {
        if skill.triggers.is_empty() {
            return 0.0;
        }

        let mut exact_matches = 0usize;
        let mut partial_matches = 0usize;

        for trigger in &skill.triggers {
            let lower_trigger = trigger.to_lowercase();
            if lower_msg.contains(&lower_trigger) {
                exact_matches += 1;
            } else {
                // Check word-level partial match
                for word in words {
                    if word.to_lowercase().starts_with(&lower_trigger)
                        || lower_trigger.starts_with(&word.to_lowercase())
                    {
                        partial_matches += 1;
                        break;
                    }
                }
            }
        }

        let trigger_count = skill.triggers.len() as f64;
        let exact_ratio = exact_matches as f64 / trigger_count;
        let partial_ratio = partial_matches as f64 / trigger_count;

        // Blend: exact match weighted higher
        let match_score = exact_ratio * 0.7 + partial_ratio * 0.3;

        // Confidence weighting
        match_score * skill.confidence
    }
}

// ---------------------------------------------------------------------------
// Skill Store (filesystem persistence)
// ---------------------------------------------------------------------------

/// Manages learned skills on disk.
pub struct SkillStore {
    dir: PathBuf,
    skills: Vec<LearnedSkill>,
}

impl SkillStore {
    pub fn new(workspace: &Path) -> Self {
        let dir = workspace.join(".tcip").join("learned_skills");
        Self {
            dir,
            skills: Vec::new(),
        }
    }

    /// Load all learned skills from disk.
    pub fn load(&mut self) -> &[LearnedSkill] {
        self.skills.clear();
        if !self.dir.exists() {
            return &self.skills;
        }

        let entries = match fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) => {
                warn!("failed to read learned_skills dir: {e}");
                return &self.skills;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match fs::read_to_string(&path) {
                Ok(content) => match serde_json::from_str::<LearnedSkill>(&content) {
                    Ok(skill) => {
                        debug!("loaded learned skill: {}", skill.name);
                        self.skills.push(skill);
                    }
                    Err(e) => warn!("failed to parse {}: {e}", path.display()),
                },
                Err(e) => warn!("failed to read {}: {e}", path.display()),
            }
        }

        info!("loaded {} learned skills", self.skills.len());
        &self.skills
    }

    /// Save a learned skill to disk.
    pub fn save(&mut self, skill: LearnedSkill) -> Result<(), std::io::Error> {
        fs::create_dir_all(&self.dir)?;
        let filename = format!("{}.json", sanitize_filename(&skill.name));
        let path = self.dir.join(&filename);
        let json = serde_json::to_string_pretty(&skill)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&path, json)?;
        info!("saved learned skill: {} → {}", skill.name, path.display());
        self.skills.push(skill);
        Ok(())
    }

    /// Delete a learned skill by name.
    pub fn forget(&mut self, name: &str) -> bool {
        let filename = format!("{}.json", sanitize_filename(name));
        let path = self.dir.join(&filename);
        if path.exists() {
            if let Err(e) = fs::remove_file(&path) {
                warn!("failed to delete {}: {e}", path.display());
                return false;
            }
        }
        let before = self.skills.len();
        self.skills.retain(|s| s.name != name);
        self.skills.len() < before
    }

    /// Get all loaded skills.
    pub fn skills(&self) -> &[LearnedSkill] {
        &self.skills
    }

    /// Decay confidence for skills unused in the given number of days.
    pub fn decay_unused(&mut self, stale_days: i64, decay_amount: f64) {
        let cutoff = Utc::now() - chrono::Duration::days(stale_days);
        let mut decayed = Vec::new();

        for skill in &mut self.skills {
            let last = skill.last_used.unwrap_or(skill.learned_at);
            if last < cutoff && skill.confidence > 0.0 {
                skill.confidence = (skill.confidence - decay_amount).max(0.0);
                decayed.push(skill.name.clone());
            }
        }

        if !decayed.is_empty() {
            info!("decayed confidence for {} stale skills", decayed.len());
        }
    }

    /// Mark a skill as used (bump use_count and last_used).
    pub fn mark_used(&mut self, name: &str) {
        if let Some(skill) = self.skills.iter_mut().find(|s| s.name == name) {
            skill.use_count += 1;
            skill.last_used = Some(Utc::now());
        }
    }
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Top-level learner that ties extractor + matcher + store together
// ---------------------------------------------------------------------------

/// The top-level learner that ties extraction, matching, and persistence together.
pub struct Learner {
    pub extractor: SkillExtractor,
    pub matcher: SkillMatcher,
    pub store: SkillStore,
}

impl Learner {
    pub fn new(workspace: &Path) -> Self {
        let mut store = SkillStore::new(workspace);
        store.load();
        Self {
            extractor: SkillExtractor::new(),
            matcher: SkillMatcher::new(),
            store,
        }
    }

    /// After a session ends, try to extract and save a learned skill.
    pub fn on_session_end(&mut self, summary: &SessionSummary) -> Option<String> {
        match self.extractor.extract(summary) {
            Some(skill) => {
                let name = skill.name.clone();
                match self.store.save(skill) {
                    Ok(()) => {
                        info!("learned new skill: {name}");
                        Some(name)
                    }
                    Err(e) => {
                        warn!("failed to save learned skill: {e}");
                        None
                    }
                }
            }
            None => None,
        }
    }

    /// Get learned skills relevant to a user message, formatted for context injection.
    pub fn get_relevant_context(&mut self, user_message: &str) -> Option<String> {
        let matches = self.matcher.match_skills(self.store.skills(), user_message);
        if matches.is_empty() {
            return None;
        }

        // Collect names first, then mark used (avoids borrow conflict)
        let results: Vec<(String, f64, f64, String)> = matches
            .iter()
            .map(|(skill, score)| {
                (skill.name.clone(), skill.confidence, *score, skill.body.clone())
            })
            .collect();

        let mut parts = Vec::new();
        for (name, confidence, score, body) in &results {
            self.store.mark_used(name);
            parts.push(format!(
                "### Learned: {} (confidence: {:.0}%, relevance: {:.0}%)\n{}",
                name,
                confidence * 100.0,
                score * 100.0,
                body,
            ));
        }

        Some(parts.join("\n\n"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_summary(tool_names: &[&str], errors: usize) -> SessionSummary {
        SessionSummary {
            session_id: "test-abc123".to_string(),
            tool_calls: tool_names
                .iter()
                .enumerate()
                .map(|(i, name)| ToolCallSummary {
                    name: name.to_string(),
                    input_keys: vec!["crop".to_string()],
                    succeeded: i >= errors,
                })
                .collect(),
            had_errors: errors > 0,
            error_count: errors,
            final_text: "Done.".to_string(),
        }
    }

    #[test]
    fn extractor_rejects_too_few_tools() {
        let ext = SkillExtractor::new();
        let summary = make_summary(&["read_file", "write_file"], 0);
        assert!(!ext.is_extractable(&summary));
        assert!(ext.extract(&summary).is_none());
    }

    #[test]
    fn extractor_rejects_too_many_errors() {
        let ext = SkillExtractor::new();
        let summary = make_summary(&["a", "b", "c", "d"], 3); // 75% errors
        assert!(!ext.is_extractable(&summary));
    }

    #[test]
    fn extractor_extracts_valid_session() {
        let ext = SkillExtractor::new();
        let summary = make_summary(&["annotate", "train_model", "eval_metrics", "export"], 0);
        let skill = ext.extract(&summary).unwrap();
        assert_eq!(skill.skill_type, PatternType::Optimization);
        assert!(!skill.triggers.is_empty());
        assert_eq!(skill.confidence, 0.8);
    }

    #[test]
    fn matcher_scores_exact_trigger() {
        let matcher = SkillMatcher::new();
        let skill = LearnedSkill {
            name: "test".to_string(),
            skill_type: PatternType::Technique,
            triggers: vec!["hazelnut".to_string(), "catkin".to_string(), "annotation".to_string()],
            confidence: 0.9,
            learned_from: "s1".to_string(),
            learned_at: Utc::now(),
            body: "test body".to_string(),
            use_count: 0,
            last_used: None,
        };
        let skills = [skill];
        let results = matcher.match_skills(&skills, "annotate hazelnut catkins");
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.3); // Should have decent score
    }

    #[test]
    fn matcher_filters_low_scores() {
        let matcher = SkillMatcher::new();
        let skill = LearnedSkill {
            name: "test".to_string(),
            skill_type: PatternType::Technique,
            triggers: vec!["completely".to_string(), "unrelated".to_string()],
            confidence: 0.5,
            learned_from: "s1".to_string(),
            learned_at: Utc::now(),
            body: "test".to_string(),
            use_count: 0,
            last_used: None,
        };
        let skills = [skill];
        let results = matcher.match_skills(&skills, "train hazelnut detector");
        assert!(results.is_empty());
    }

    #[test]
    fn store_save_load_forget() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = SkillStore::new(tmp.path());
        let skill = LearnedSkill {
            name: "test-skill".to_string(),
            skill_type: PatternType::ProblemSolution,
            triggers: vec!["test".to_string()],
            confidence: 0.8,
            learned_from: "s1".to_string(),
            learned_at: Utc::now(),
            body: "test body".to_string(),
            use_count: 0,
            last_used: None,
        };
        store.save(skill).unwrap();
        assert_eq!(store.skills().len(), 1);

        // Reload from disk
        let mut store2 = SkillStore::new(tmp.path());
        store2.load();
        assert_eq!(store2.skills().len(), 1);
        assert_eq!(store2.skills()[0].name, "test-skill");

        // Forget
        assert!(store2.forget("test-skill"));
        assert_eq!(store2.skills().len(), 0);
    }

    #[test]
    fn decay_reduces_confidence() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = SkillStore::new(tmp.path());
        let skill = LearnedSkill {
            name: "old".to_string(),
            skill_type: PatternType::Technique,
            triggers: vec!["x".to_string()],
            confidence: 0.8,
            learned_from: "s1".to_string(),
            learned_at: Utc::now() - chrono::Duration::days(100),
            body: "old body".to_string(),
            use_count: 0,
            last_used: None,
        };
        store.save(skill).unwrap();
        store.decay_unused(90, 0.1);
        assert!((store.skills()[0].confidence - 0.7).abs() < 0.001);
    }
}

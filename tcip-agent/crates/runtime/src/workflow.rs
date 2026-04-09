//! PRD-driven workflow contracts with acceptance criteria and FSM progression.
//!
//! Each workflow has stories with acceptance criteria. Stories progress through
//! a state machine: Pending → InProgress → Verifying → Complete/Failed.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowContract {
    pub id: String,
    pub title: String,
    pub stories: Vec<Story>,
    pub current_story_idx: usize,
    pub max_fix_attempts: u32,
    pub status: WorkflowStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Active,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Story {
    pub title: String,
    pub acceptance_criteria: Vec<AcceptanceCriterion>,
    pub status: StoryStatus,
    pub fix_attempts: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StoryStatus {
    Pending,
    InProgress,
    Verifying,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceCriterion {
    pub description: String,
    pub verification_type: VerificationType,
    pub threshold: Option<f64>,
    pub passed: bool,
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationType {
    AutoTest,
    MetricThreshold,
    ManualReview,
}

// ---------------------------------------------------------------------------
// WorkflowContract implementation
// ---------------------------------------------------------------------------

impl WorkflowContract {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            title: title.into(),
            stories: Vec::new(),
            current_story_idx: 0,
            max_fix_attempts: 3,
            status: WorkflowStatus::Active,
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a story with its acceptance criteria.
    pub fn add_story(&mut self, title: impl Into<String>, criteria: Vec<AcceptanceCriterion>) {
        self.stories.push(Story {
            title: title.into(),
            acceptance_criteria: criteria,
            status: StoryStatus::Pending,
            fix_attempts: 0,
        });
    }

    /// Get the current story, if any.
    pub fn current_story(&self) -> Option<&Story> {
        self.stories.get(self.current_story_idx)
    }

    /// Get the current story mutably.
    pub fn current_story_mut(&mut self) -> Option<&mut Story> {
        self.stories.get_mut(self.current_story_idx)
    }

    /// Begin work on the current story (Pending → InProgress).
    pub fn start_current_story(&mut self) -> bool {
        if let Some(story) = self.stories.get_mut(self.current_story_idx) {
            if story.status == StoryStatus::Pending {
                story.status = StoryStatus::InProgress;
                self.updated_at = Utc::now();
                debug!("story '{}' → InProgress", story.title);
                return true;
            }
        }
        false
    }

    /// Submit current story for verification (InProgress → Verifying).
    pub fn submit_for_verification(&mut self) -> bool {
        if let Some(story) = self.stories.get_mut(self.current_story_idx) {
            if story.status == StoryStatus::InProgress {
                story.status = StoryStatus::Verifying;
                self.updated_at = Utc::now();
                debug!("story '{}' → Verifying", story.title);
                return true;
            }
        }
        false
    }

    /// Pass a specific acceptance criterion by index, with optional evidence.
    pub fn pass_criterion(&mut self, story_idx: usize, criterion_idx: usize, evidence: Option<String>) -> bool {
        if let Some(story) = self.stories.get_mut(story_idx) {
            if let Some(ac) = story.acceptance_criteria.get_mut(criterion_idx) {
                ac.passed = true;
                ac.evidence = evidence;
                self.updated_at = Utc::now();
                return true;
            }
        }
        false
    }

    /// Check if all criteria for the current story are passing.
    pub fn all_criteria_passing(&self) -> bool {
        self.current_story()
            .map(|s| s.acceptance_criteria.iter().all(|ac| ac.passed))
            .unwrap_or(false)
    }

    /// Resolve verification: if all criteria pass → Complete and advance; else retry or fail.
    pub fn resolve_verification(&mut self) -> VerificationOutcome {
        let idx = self.current_story_idx;
        let max_attempts = self.max_fix_attempts;

        let story = match self.stories.get_mut(idx) {
            Some(s) if s.status == StoryStatus::Verifying => s,
            _ => return VerificationOutcome::NoAction,
        };

        if story.acceptance_criteria.iter().all(|ac| ac.passed) {
            story.status = StoryStatus::Complete;
            info!("story '{}' → Complete", story.title);
            self.updated_at = Utc::now();

            // Advance to next story
            if idx + 1 < self.stories.len() {
                self.current_story_idx = idx + 1;
                VerificationOutcome::StoryComplete { next_story: idx + 1 }
            } else {
                // All stories done — check for final regression
                self.status = WorkflowStatus::Complete;
                info!("workflow '{}' → Complete", self.title);
                VerificationOutcome::WorkflowComplete
            }
        } else {
            story.fix_attempts += 1;
            if story.fix_attempts >= max_attempts {
                story.status = StoryStatus::Failed;
                self.status = WorkflowStatus::Failed;
                warn!("story '{}' → Failed after {} attempts", story.title, story.fix_attempts);
                VerificationOutcome::StoryFailed {
                    story_idx: idx,
                    attempts: story.fix_attempts,
                }
            } else {
                story.status = StoryStatus::InProgress;
                let failing: Vec<String> = story
                    .acceptance_criteria
                    .iter()
                    .filter(|ac| !ac.passed)
                    .map(|ac| ac.description.clone())
                    .collect();
                debug!("story '{}' → InProgress (retry {})", story.title, story.fix_attempts);
                VerificationOutcome::Retry {
                    story_idx: idx,
                    attempt: story.fix_attempts,
                    failing_criteria: failing,
                }
            }
        }
    }

    /// Cancel the workflow.
    pub fn cancel(&mut self) {
        self.status = WorkflowStatus::Cancelled;
        self.updated_at = Utc::now();
    }

    /// Build a status summary for context injection.
    pub fn status_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("## Workflow: {} [{}]", self.title, format_status(&self.status)));

        for (i, story) in self.stories.iter().enumerate() {
            let marker = if i == self.current_story_idx && self.status == WorkflowStatus::Active {
                "→"
            } else {
                " "
            };
            let passing = story.acceptance_criteria.iter().filter(|ac| ac.passed).count();
            let total = story.acceptance_criteria.len();
            lines.push(format!(
                "{marker} Story {}: {} [{}, {passing}/{total} criteria]",
                i + 1,
                story.title,
                format_story_status(&story.status),
            ));
        }
        lines.join("\n")
    }

    /// Save to disk.
    pub fn save(&self, workspace: &Path) -> Result<PathBuf, std::io::Error> {
        let dir = workspace.join(".tcip").join("workflows");
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", self.id));
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(&path, json)?;
        Ok(path)
    }

    /// Load from disk.
    pub fn load(workspace: &Path, id: &str) -> Result<Self, std::io::Error> {
        let path = workspace.join(".tcip").join("workflows").join(format!("{id}.json"));
        let content = fs::read_to_string(&path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerificationOutcome {
    StoryComplete { next_story: usize },
    WorkflowComplete,
    Retry { story_idx: usize, attempt: u32, failing_criteria: Vec<String> },
    StoryFailed { story_idx: usize, attempts: u32 },
    NoAction,
}

fn format_status(s: &WorkflowStatus) -> &'static str {
    match s {
        WorkflowStatus::Active => "ACTIVE",
        WorkflowStatus::Complete => "COMPLETE",
        WorkflowStatus::Failed => "FAILED",
        WorkflowStatus::Cancelled => "CANCELLED",
    }
}

fn format_story_status(s: &StoryStatus) -> &'static str {
    match s {
        StoryStatus::Pending => "pending",
        StoryStatus::InProgress => "in-progress",
        StoryStatus::Verifying => "verifying",
        StoryStatus::Complete => "complete",
        StoryStatus::Failed => "failed",
    }
}

/// Helper to build an acceptance criterion.
pub fn criterion(desc: impl Into<String>, vtype: VerificationType, threshold: Option<f64>) -> AcceptanceCriterion {
    AcceptanceCriterion {
        description: desc.into(),
        verification_type: vtype,
        threshold,
        passed: false,
        evidence: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_workflow() -> WorkflowContract {
        let mut wf = WorkflowContract::new("wf-001", "Train hazelnut catkin detector");
        wf.add_story("Dataset preparation", vec![
            criterion("≥500 images with valid YOLO labels", VerificationType::AutoTest, None),
            criterion("Class distribution within 3:1 ratio", VerificationType::MetricThreshold, Some(3.0)),
        ]);
        wf.add_story("Model training", vec![
            criterion("Training completes without error", VerificationType::AutoTest, None),
            criterion("mAP@50 ≥ 0.70 on validation set", VerificationType::MetricThreshold, Some(0.70)),
        ]);
        wf.add_story("Delivery", vec![
            criterion("CSV output matches expected schema", VerificationType::AutoTest, None),
        ]);
        wf
    }

    #[test]
    fn story_fsm_pending_to_complete() {
        let mut wf = sample_workflow();
        assert_eq!(wf.current_story().unwrap().status, StoryStatus::Pending);

        assert!(wf.start_current_story());
        assert_eq!(wf.current_story().unwrap().status, StoryStatus::InProgress);

        assert!(wf.submit_for_verification());
        assert_eq!(wf.current_story().unwrap().status, StoryStatus::Verifying);

        // Pass all criteria for story 0
        wf.pass_criterion(0, 0, Some("520 images found".into()));
        wf.pass_criterion(0, 1, Some("ratio = 2.1".into()));

        let outcome = wf.resolve_verification();
        assert_eq!(outcome, VerificationOutcome::StoryComplete { next_story: 1 });
        assert_eq!(wf.current_story_idx, 1);
    }

    #[test]
    fn retry_on_failing_criteria() {
        let mut wf = sample_workflow();
        wf.start_current_story();
        wf.submit_for_verification();
        // Don't pass any criteria
        let outcome = wf.resolve_verification();
        match outcome {
            VerificationOutcome::Retry { attempt, failing_criteria, .. } => {
                assert_eq!(attempt, 1);
                assert_eq!(failing_criteria.len(), 2);
            }
            other => panic!("expected Retry, got {other:?}"),
        }
        assert_eq!(wf.current_story().unwrap().status, StoryStatus::InProgress);
    }

    #[test]
    fn fails_after_max_attempts() {
        let mut wf = sample_workflow();
        wf.max_fix_attempts = 2;

        // First attempt
        wf.start_current_story();
        wf.submit_for_verification();
        wf.resolve_verification(); // → Retry (attempt 1), back to InProgress

        // Second attempt (already InProgress from retry)
        wf.submit_for_verification();
        let outcome = wf.resolve_verification(); // → Failed (attempt 2 >= max 2)
        match outcome {
            VerificationOutcome::StoryFailed { attempts, .. } => assert_eq!(attempts, 2),
            other => panic!("expected StoryFailed, got {other:?}"),
        }
        assert_eq!(wf.status, WorkflowStatus::Failed);
    }

    #[test]
    fn workflow_completes_when_all_stories_done() {
        let mut wf = sample_workflow();

        // Story 0
        wf.start_current_story();
        wf.submit_for_verification();
        wf.pass_criterion(0, 0, None);
        wf.pass_criterion(0, 1, None);
        wf.resolve_verification();

        // Story 1
        wf.start_current_story();
        wf.submit_for_verification();
        wf.pass_criterion(1, 0, None);
        wf.pass_criterion(1, 1, None);
        wf.resolve_verification();

        // Story 2 (last)
        wf.start_current_story();
        wf.submit_for_verification();
        wf.pass_criterion(2, 0, None);
        let outcome = wf.resolve_verification();
        assert_eq!(outcome, VerificationOutcome::WorkflowComplete);
        assert_eq!(wf.status, WorkflowStatus::Complete);
    }

    #[test]
    fn status_summary_format() {
        let wf = sample_workflow();
        let summary = wf.status_summary();
        assert!(summary.contains("Train hazelnut catkin detector"));
        assert!(summary.contains("Dataset preparation"));
        assert!(summary.contains("Model training"));
    }

    #[test]
    fn save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let wf = sample_workflow();
        let path = wf.save(tmp.path()).unwrap();
        assert!(path.exists());

        let loaded = WorkflowContract::load(tmp.path(), "wf-001").unwrap();
        assert_eq!(loaded.title, wf.title);
        assert_eq!(loaded.stories.len(), 3);
    }

    #[test]
    fn cancel_workflow() {
        let mut wf = sample_workflow();
        wf.cancel();
        assert_eq!(wf.status, WorkflowStatus::Cancelled);
    }
}

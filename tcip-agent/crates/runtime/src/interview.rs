//! Deep interview protocol for requirement clarification before expensive ML work.
//!
//! Scores ambiguity across 6 ML/CV-specific dimensions. When ambiguity is high,
//! the agent asks targeted clarifying questions before proceeding.

use serde::{Deserialize, Serialize};
use tracing::debug;

/// The 6 ML/CV-specific dimensions for ambiguity scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityProfile {
    /// What ML task? (detection, segmentation, counting, classification)
    pub intent_clarity: f64,
    /// Which crop? Which traits? Which growth stage?
    pub crop_trait_specificity: f64,
    /// What sensor? Resolution? Image count? Existing labels?
    pub data_specification: f64,
    /// mAP threshold? Accuracy target? Throughput?
    pub success_criteria: f64,
    /// GPU budget? Timeline? Annotation workforce?
    pub constraint_clarity: f64,
    /// Brownfield vs greenfield?
    pub context_clarity: f64,
}

impl AmbiguityProfile {
    /// All dimensions at zero (maximum ambiguity).
    pub fn empty() -> Self {
        Self {
            intent_clarity: 0.0,
            crop_trait_specificity: 0.0,
            data_specification: 0.0,
            success_criteria: 0.0,
            constraint_clarity: 0.0,
            context_clarity: 0.0,
        }
    }

    /// Compute overall ambiguity score (0.0 = fully clear, 1.0 = fully ambiguous).
    pub fn ambiguity(&self) -> f64 {
        let weighted = self.intent_clarity * 0.25
            + self.crop_trait_specificity * 0.20
            + self.data_specification * 0.20
            + self.success_criteria * 0.15
            + self.constraint_clarity * 0.10
            + self.context_clarity * 0.10;
        1.0 - weighted.clamp(0.0, 1.0)
    }

    /// Find the weakest dimension (lowest score) for targeted questioning.
    pub fn weakest_dimension(&self) -> (&'static str, f64) {
        let dims = [
            ("intent_clarity", self.intent_clarity),
            ("crop_trait_specificity", self.crop_trait_specificity),
            ("data_specification", self.data_specification),
            ("success_criteria", self.success_criteria),
            ("constraint_clarity", self.constraint_clarity),
            ("context_clarity", self.context_clarity),
        ];
        dims.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
    }
}

/// Manages a multi-round interview to reduce ambiguity.
pub struct InterviewSession {
    pub profile: AmbiguityProfile,
    pub round: u32,
    pub max_rounds: u32,
    pub threshold: f64,
    pub questions_asked: Vec<InterviewQuestion>,
    pub status: InterviewStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterviewQuestion {
    pub round: u32,
    pub dimension: String,
    pub question: String,
    pub answer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterviewStatus {
    /// Interview is in progress.
    Active,
    /// Ambiguity is below threshold — ready to proceed.
    Resolved,
    /// Max rounds reached — proceed with assumptions.
    MaxRoundsReached,
    /// User skipped the interview.
    Skipped,
}

impl InterviewSession {
    pub fn new() -> Self {
        Self {
            profile: AmbiguityProfile::empty(),
            round: 0,
            max_rounds: 10,
            threshold: 0.20,
            questions_asked: Vec::new(),
            status: InterviewStatus::Active,
        }
    }

    /// Check if a user message should trigger an interview.
    pub fn should_trigger(message: &str) -> bool {
        let word_count = message.split_whitespace().count();
        if word_count > 50 {
            return false;
        }
        let vague_verbs = [
            "build", "create", "help", "make", "set up", "need", "want",
            "train", "detect", "count", "analyze", "classify",
        ];
        let lower = message.to_lowercase();
        vague_verbs.iter().any(|v| lower.contains(v))
    }

    /// Score the initial user message against all dimensions.
    pub fn score_initial_message(&mut self, message: &str) {
        let lower = message.to_lowercase();

        // Intent clarity — check for ML task keywords
        let intent_keywords = ["detect", "segment", "classif", "count", "track", "regress"];
        self.profile.intent_clarity = keyword_score(&lower, &intent_keywords);

        // Crop/trait specificity
        let crop_keywords = ["hazelnut", "chestnut", "persimmon", "black_locust", "currant", "elderberry"];
        let trait_score = if lower.contains("trait") || lower.contains("phenotyp") { 0.3 } else { 0.0 };
        self.profile.crop_trait_specificity = keyword_score(&lower, &crop_keywords) + trait_score;

        // Data specification
        let data_keywords = ["image", "photo", "rgb", "multispectral", "lidar", "resolution", "label"];
        self.profile.data_specification = keyword_score(&lower, &data_keywords);

        // Success criteria
        let metric_keywords = ["map", "accuracy", "f1", "precision", "recall", "iou", "threshold"];
        self.profile.success_criteria = keyword_score(&lower, &metric_keywords);

        // Constraint clarity
        let constraint_keywords = ["gpu", "budget", "time", "deadline", "a100", "rtx", "hour"];
        self.profile.constraint_clarity = keyword_score(&lower, &constraint_keywords);

        // Context clarity
        let context_keywords = ["existing", "from scratch", "continue", "improve", "retrain", "new"];
        self.profile.context_clarity = keyword_score(&lower, &context_keywords);

        self.profile.intent_clarity = self.profile.intent_clarity.clamp(0.0, 1.0);
        self.profile.crop_trait_specificity = self.profile.crop_trait_specificity.clamp(0.0, 1.0);
        self.profile.data_specification = self.profile.data_specification.clamp(0.0, 1.0);
        self.profile.success_criteria = self.profile.success_criteria.clamp(0.0, 1.0);
        self.profile.constraint_clarity = self.profile.constraint_clarity.clamp(0.0, 1.0);
        self.profile.context_clarity = self.profile.context_clarity.clamp(0.0, 1.0);

        debug!(
            "initial ambiguity: {:.0}% (intent={:.2}, crop={:.2}, data={:.2}, success={:.2}, constraint={:.2}, context={:.2})",
            self.profile.ambiguity() * 100.0,
            self.profile.intent_clarity,
            self.profile.crop_trait_specificity,
            self.profile.data_specification,
            self.profile.success_criteria,
            self.profile.constraint_clarity,
            self.profile.context_clarity,
        );
    }

    /// Generate the next interview question targeting the weakest dimension.
    pub fn next_question(&mut self) -> Option<InterviewQuestion> {
        if self.status != InterviewStatus::Active {
            return None;
        }

        let ambiguity = self.profile.ambiguity();
        if ambiguity <= self.threshold {
            self.status = InterviewStatus::Resolved;
            return None;
        }

        if self.round >= self.max_rounds {
            self.status = InterviewStatus::MaxRoundsReached;
            return None;
        }

        self.round += 1;
        let (dim_name, _score) = self.profile.weakest_dimension();

        let question_text = match dim_name {
            "intent_clarity" => "What specific ML task do you need? (e.g., object detection, instance segmentation, counting, classification, regression)",
            "crop_trait_specificity" => "Which crop and trait(s) are you targeting? (e.g., hazelnut catkin detection, chestnut blight classification)",
            "data_specification" => "What type of imagery do you have? (sensor type, resolution, approximate image count, any existing labels?)",
            "success_criteria" => "What accuracy target do you need? (e.g., mAP@50 ≥ 0.70, or F1 ≥ 0.85)",
            "constraint_clarity" => "What are your compute/time constraints? (GPU type, training time budget, annotation workforce)",
            "context_clarity" => "Are you building from scratch or improving an existing pipeline?",
            _ => "Could you provide more details about your requirements?",
        };

        let q = InterviewQuestion {
            round: self.round,
            dimension: dim_name.to_string(),
            question: question_text.to_string(),
            answer: None,
        };
        self.questions_asked.push(q.clone());
        Some(q)
    }

    /// Process a user's answer to an interview question.
    pub fn process_answer(&mut self, dimension: &str, answer: &str) {
        // Mark the question as answered
        if let Some(q) = self.questions_asked.iter_mut().rev().find(|q| q.dimension == dimension) {
            q.answer = Some(answer.to_string());
        }

        // Re-score the relevant dimension based on answer length and content
        let boost = (answer.split_whitespace().count() as f64 / 20.0).clamp(0.2, 0.8);

        match dimension {
            "intent_clarity" => self.profile.intent_clarity = (self.profile.intent_clarity + boost).clamp(0.0, 1.0),
            "crop_trait_specificity" => self.profile.crop_trait_specificity = (self.profile.crop_trait_specificity + boost).clamp(0.0, 1.0),
            "data_specification" => self.profile.data_specification = (self.profile.data_specification + boost).clamp(0.0, 1.0),
            "success_criteria" => self.profile.success_criteria = (self.profile.success_criteria + boost).clamp(0.0, 1.0),
            "constraint_clarity" => self.profile.constraint_clarity = (self.profile.constraint_clarity + boost).clamp(0.0, 1.0),
            "context_clarity" => self.profile.context_clarity = (self.profile.context_clarity + boost).clamp(0.0, 1.0),
            _ => {}
        }
    }

    /// Skip the interview.
    pub fn skip(&mut self) {
        self.status = InterviewStatus::Skipped;
    }

    /// Build a status line for display.
    pub fn status_line(&self) -> String {
        let ambiguity = self.profile.ambiguity();
        let (dim, score) = self.profile.weakest_dimension();
        format!(
            "Round {} | Targeting: {} ({:.1}/1.0) | Ambiguity: {:.0}%",
            self.round, dim, score, ambiguity * 100.0
        )
    }

    /// Build an execution spec from the completed interview.
    pub fn build_spec(&self) -> String {
        let mut spec = String::from("## Execution Specification\n\n");
        for q in &self.questions_asked {
            if let Some(ref answer) = q.answer {
                spec.push_str(&format!("**{}**: {}\n\n", q.dimension, answer));
            }
        }
        spec.push_str(&format!(
            "**Ambiguity**: {:.0}%\n**Rounds**: {}\n**Status**: {:?}\n",
            self.profile.ambiguity() * 100.0,
            self.round,
            self.status,
        ));
        spec
    }
}

fn keyword_score(text: &str, keywords: &[&str]) -> f64 {
    let hits = keywords.iter().filter(|k| text.contains(**k)).count();
    if hits == 0 { 0.0 } else { (hits as f64 * 0.4).clamp(0.0, 1.0) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguity_starts_at_100_percent() {
        let profile = AmbiguityProfile::empty();
        assert!((profile.ambiguity() - 1.0).abs() < 0.001);
    }

    #[test]
    fn full_clarity_is_zero_ambiguity() {
        let profile = AmbiguityProfile {
            intent_clarity: 1.0,
            crop_trait_specificity: 1.0,
            data_specification: 1.0,
            success_criteria: 1.0,
            constraint_clarity: 1.0,
            context_clarity: 1.0,
        };
        assert!(profile.ambiguity().abs() < 0.001);
    }

    #[test]
    fn weakest_dimension_finds_minimum() {
        let profile = AmbiguityProfile {
            intent_clarity: 0.8,
            crop_trait_specificity: 0.1,
            data_specification: 0.5,
            success_criteria: 0.6,
            constraint_clarity: 0.9,
            context_clarity: 0.7,
        };
        let (name, score) = profile.weakest_dimension();
        assert_eq!(name, "crop_trait_specificity");
        assert!((score - 0.1).abs() < 0.001);
    }

    #[test]
    fn should_trigger_on_vague_request() {
        assert!(InterviewSession::should_trigger("I need to count hazelnuts"));
        assert!(InterviewSession::should_trigger("help me build a detector"));
        assert!(!InterviewSession::should_trigger(&"word ".repeat(60))); // Too long
    }

    #[test]
    fn interview_flow() {
        let mut session = InterviewSession::new();
        session.score_initial_message("build hazelnut detector");

        // Should have some ambiguity
        assert!(session.profile.ambiguity() > 0.3);

        // Get first question
        let q = session.next_question().unwrap();
        assert!(!q.question.is_empty());

        // Answer it
        session.process_answer(&q.dimension, "I want to detect catkins using RGB drone imagery at 1cm resolution");

        // Ambiguity should decrease
        let after = session.profile.ambiguity();
        assert!(after < 1.0);
    }

    #[test]
    fn interview_resolves_below_threshold() {
        let mut session = InterviewSession::new();
        session.profile = AmbiguityProfile {
            intent_clarity: 0.9,
            crop_trait_specificity: 0.9,
            data_specification: 0.9,
            success_criteria: 0.8,
            constraint_clarity: 0.8,
            context_clarity: 0.8,
        };
        // Already below threshold — should resolve immediately
        assert!(session.next_question().is_none());
        assert_eq!(session.status, InterviewStatus::Resolved);
    }

    #[test]
    fn skip_ends_interview() {
        let mut session = InterviewSession::new();
        session.skip();
        assert_eq!(session.status, InterviewStatus::Skipped);
        assert!(session.next_question().is_none());
    }

    #[test]
    fn spec_captures_answers() {
        let mut session = InterviewSession::new();
        session.score_initial_message("detect something");
        let q = session.next_question().unwrap();
        session.process_answer(&q.dimension, "Object detection on hazelnuts");
        let spec = session.build_spec();
        assert!(spec.contains("Object detection on hazelnuts"));
    }
}

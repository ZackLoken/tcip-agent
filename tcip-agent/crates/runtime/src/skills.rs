use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tracing::debug;

/// Subagent modes — each filters skills by mode tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubagentMode {
    PipelineDesigner,
    CodeGenerator,
    TrainingOrchestrator,
    ResultsAnalyzer,
}

impl SubagentMode {
    /// Preferred model for this mode.
    #[must_use]
    pub fn preferred_model(&self) -> &str {
        match self {
            Self::PipelineDesigner => "claude-opus-4-20250514",
            _ => "claude-sonnet-4-20250514",
        }
    }
}

impl std::fmt::Display for SubagentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PipelineDesigner => write!(f, "PipelineDesigner"),
            Self::CodeGenerator => write!(f, "CodeGenerator"),
            Self::TrainingOrchestrator => write!(f, "TrainingOrchestrator"),
            Self::ResultsAnalyzer => write!(f, "ResultsAnalyzer"),
        }
    }
}

impl std::str::FromStr for SubagentMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "PipelineDesigner" => Ok(Self::PipelineDesigner),
            "CodeGenerator" => Ok(Self::CodeGenerator),
            "TrainingOrchestrator" => Ok(Self::TrainingOrchestrator),
            "ResultsAnalyzer" => Ok(Self::ResultsAnalyzer),
            _ => Err(format!("unknown mode: {s}")),
        }
    }
}

/// Parsed YAML frontmatter from a skill file.
#[derive(Debug, Clone)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub triggers: Vec<String>,
    pub modes: Vec<String>,
    pub priority: SkillPriority,
    pub max_chars: usize,
}

/// Skill priority affects injection order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkillPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// A loaded skill file with metadata and content.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    pub metadata: SkillMetadata,
    pub content: String,
    pub filename: String,
}

/// Loads and scores skill files for dynamic injection.
pub struct SkillInjector {
    skills_dir: PathBuf,
    skills: Vec<LoadedSkill>,
    max_skills_per_prompt: usize,
    loaded: bool,
}

impl SkillInjector {
    #[must_use]
    pub fn new(skills_dir: PathBuf) -> Self {
        Self {
            skills_dir,
            skills: Vec::new(),
            max_skills_per_prompt: 4,
            loaded: false,
        }
    }

    /// Set the maximum number of skills injected per prompt.
    pub fn set_max_skills(&mut self, max: usize) {
        self.max_skills_per_prompt = max;
    }

    /// Discover and load all skill files from the skills directory.
    fn ensure_loaded(&mut self) {
        if self.loaded {
            return;
        }
        self.loaded = true;

        let dir = match std::fs::read_dir(&self.skills_dir) {
            Ok(d) => d,
            Err(_) => return,
        };

        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            if let Ok(raw) = std::fs::read_to_string(&path) {
                if let Some(skill) = parse_skill_file(&raw, &path) {
                    debug!("loaded skill: {} ({})", skill.metadata.name, skill.filename);
                    self.skills.push(skill);
                }
            }
        }
    }

    /// Build the skill injection section for a given mode and optional user message.
    pub fn build_skills_section(
        &mut self,
        mode: &SubagentMode,
        user_message: Option<&str>,
    ) -> String {
        self.ensure_loaded();

        let mode_str = mode.to_string();

        // Filter skills that match the current mode
        let mut candidates: Vec<(&LoadedSkill, f32)> = self
            .skills
            .iter()
            .filter(|s| s.metadata.modes.iter().any(|m| m == &mode_str))
            .map(|s| {
                let trigger_score = user_message
                    .map(|msg| score_triggers(&s.metadata.triggers, msg))
                    .unwrap_or(0.0);
                (s, trigger_score)
            })
            .collect();

        // Sort: priority descending, then trigger score descending
        candidates.sort_by(|a, b| {
            b.0.metadata
                .priority
                .cmp(&a.0.metadata.priority)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Take top N within token budget
        let mut sections = Vec::new();

        for (skill, _score) in candidates.iter().take(self.max_skills_per_prompt) {
            let content = if skill.content.len() > skill.metadata.max_chars {
                format!(
                    "{}...(truncated)",
                    &skill.content[..skill.metadata.max_chars]
                )
            } else {
                skill.content.clone()
            };
            sections.push(format!("### Skill: {}\n\n{content}", skill.filename));
        }

        if sections.is_empty() {
            // Fall back: no mode-matched skills, take highest priority globally
            let mut fallback: Vec<&LoadedSkill> = self.skills.iter().collect();
            fallback.sort_by(|a, b| b.metadata.priority.cmp(&a.metadata.priority));
            for skill in fallback.into_iter().take(2) {
                let content = if skill.content.len() > skill.metadata.max_chars {
                    format!(
                        "{}...(truncated)",
                        &skill.content[..skill.metadata.max_chars]
                    )
                } else {
                    skill.content.clone()
                };
                sections.push(format!("### Skill: {}\n\n{content}", skill.filename));
            }
        }

        if sections.is_empty() {
            String::new()
        } else {
            format!("## Active Skills\n\n{}", sections.join("\n\n---\n\n"))
        }
    }

    /// Get all loaded skill metadata (for diagnostics).
    pub fn skill_metadata(&mut self) -> Vec<&SkillMetadata> {
        self.ensure_loaded();
        self.skills.iter().map(|s| &s.metadata).collect()
    }

    /// Force reload skills from disk.
    pub fn reload(&mut self) {
        self.loaded = false;
        self.skills.clear();
        self.ensure_loaded();
    }
}

/// Parse a skill file with YAML frontmatter.
fn parse_skill_file(raw: &str, path: &Path) -> Option<LoadedSkill> {
    let filename = path.file_name()?.to_str()?.to_string();

    // Parse YAML frontmatter between --- delimiters
    let (metadata, content) = if raw.starts_with("---") {
        let rest = &raw[3..];
        if let Some(end) = rest.find("---") {
            let yaml_str = &rest[..end];
            let body = rest[end + 3..].trim_start();
            let meta = parse_frontmatter(yaml_str, &filename)?;
            (meta, body.to_string())
        } else {
            return None;
        }
    } else {
        // No frontmatter — create default metadata from filename
        let name = filename.trim_end_matches(".md").to_string();
        let meta = SkillMetadata {
            name: name.clone(),
            description: String::new(),
            triggers: Vec::new(),
            modes: vec!["PipelineDesigner".to_string()],
            priority: SkillPriority::Normal,
            max_chars: 4000,
        };
        (meta, raw.to_string())
    };

    Some(LoadedSkill {
        metadata,
        content,
        filename,
    })
}

/// Parse YAML frontmatter string into `SkillMetadata`.
fn parse_frontmatter(yaml_str: &str, filename: &str) -> Option<SkillMetadata> {
    let mut map = BTreeMap::new();
    let mut current_key: Option<String> = None;
    let mut current_list: Vec<String> = Vec::new();

    for line in yaml_str.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check for list continuation
        if trimmed.starts_with("- ") {
            if current_key.is_some() {
                current_list.push(trimmed[2..].trim().to_string());
            }
            continue;
        }

        // Save previous list if any
        if let Some(ref key) = current_key {
            if !current_list.is_empty() {
                map.insert(key.clone(), FrontmatterValue::List(current_list.clone()));
                current_list.clear();
            }
        }

        // Parse key: value
        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim().to_string();
            let value = value.trim();

            if value.is_empty() {
                // Start of a list
                current_key = Some(key);
                current_list.clear();
            } else if value.starts_with('[') && value.ends_with(']') {
                // Inline list: [A, B, C]
                let items: Vec<String> = value[1..value.len() - 1]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                map.insert(key.clone(), FrontmatterValue::List(items));
                current_key = None;
            } else {
                // Scalar value — strip quotes
                let clean = value.trim_matches('"').to_string();
                map.insert(key.clone(), FrontmatterValue::Scalar(clean));
                current_key = None;
            }
        }
    }

    // Save final list
    if let Some(ref key) = current_key {
        if !current_list.is_empty() {
            map.insert(key.clone(), FrontmatterValue::List(current_list));
        }
    }

    // Convert to SkillMetadata
    let name = match map.get("name") {
        Some(FrontmatterValue::Scalar(s)) => s.clone(),
        _ => filename.trim_end_matches(".md").to_string(),
    };
    let description = match map.get("description") {
        Some(FrontmatterValue::Scalar(s)) => s.clone(),
        _ => String::new(),
    };
    let triggers = match map.get("triggers") {
        Some(FrontmatterValue::List(l)) => l.clone(),
        _ => Vec::new(),
    };
    let modes = match map.get("modes") {
        Some(FrontmatterValue::List(l)) => l.clone(),
        _ => vec!["PipelineDesigner".to_string()],
    };
    let priority = match map.get("priority") {
        Some(FrontmatterValue::Scalar(s)) => match s.as_str() {
            "high" => SkillPriority::High,
            "low" => SkillPriority::Low,
            _ => SkillPriority::Normal,
        },
        _ => SkillPriority::Normal,
    };
    let max_chars = match map.get("max_chars") {
        Some(FrontmatterValue::Scalar(s)) => s.parse().unwrap_or(4000),
        _ => 4000,
    };

    Some(SkillMetadata {
        name,
        description,
        triggers,
        modes,
        priority,
        max_chars,
    })
}

#[derive(Debug)]
enum FrontmatterValue {
    Scalar(String),
    List(Vec<String>),
}

/// Score how well trigger keywords match a user message.
fn score_triggers(triggers: &[String], message: &str) -> f32 {
    if triggers.is_empty() {
        return 0.0;
    }

    let message_lower = message.to_lowercase();
    let matches = triggers
        .iter()
        .filter(|t| message_lower.contains(&t.to_lowercase()))
        .count();

    matches as f32 / triggers.len() as f32
}

/// Build the full system prompt following claw-code's `SystemPromptBuilder` pattern.
pub fn build_system_prompt(
    skill_injector: &mut SkillInjector,
    mode: &SubagentMode,
    project_context: Option<&ProjectContext>,
    environment: &EnvironmentInfo,
) -> String {
    build_system_prompt_with_context(skill_injector, mode, project_context, environment, None, None)
}

/// Build system prompt with optional user message for trigger matching and injected context.
pub fn build_system_prompt_with_context(
    skill_injector: &mut SkillInjector,
    mode: &SubagentMode,
    project_context: Option<&ProjectContext>,
    environment: &EnvironmentInfo,
    user_message: Option<&str>,
    injected_context: Option<&str>,
) -> String {
    let mut sections = Vec::new();

    // Section 1: Base identity
    sections.push(BASE_SYSTEM_PROMPT.to_string());

    // Section 2: System rules
    sections.push(SYSTEM_RULES.to_string());

    // Section 3: Dynamic boundary
    sections.push("--- DYNAMIC CONTEXT BELOW ---".to_string());

    // Section 4: Skills (mode-dependent, trigger-scored)
    let skills = skill_injector.build_skills_section(mode, user_message);
    if !skills.is_empty() {
        sections.push(skills);
    }

    // Section 5: Injected context (from ContextCollector)
    if let Some(ctx) = injected_context {
        if !ctx.is_empty() {
            sections.push(format!("## Injected Context\n\n{ctx}"));
        }
    }

    // Section 6: Project context
    if let Some(ctx) = project_context {
        sections.push(format!(
            "## Project Context\n\n\
             Crop: {}\n\
             Traits: {}\n\
             Pipeline stage: {}",
            ctx.crop.as_deref().unwrap_or("not set"),
            if ctx.traits.is_empty() {
                "not set".to_string()
            } else {
                ctx.traits.join(", ")
            },
            ctx.pipeline_stage.as_deref().unwrap_or("not started"),
        ));
    }

    // Section 7: Environment
    sections.push(format!(
        "## Environment\n\n\
         Model: {}\n\
         Mode: {mode}\n\
         Working directory: {}\n\
         Date: {}\n\
         GPU: {}",
        environment.model,
        environment.cwd.display(),
        chrono::Utc::now().format("%Y-%m-%d"),
        environment.gpu_info.as_deref().unwrap_or("unknown"),
    ));

    sections.join("\n\n")
}

/// Project-level context from `.tcip/state.toml`.
pub struct ProjectContext {
    pub crop: Option<String>,
    pub traits: Vec<String>,
    pub pipeline_stage: Option<String>,
}

/// Runtime environment info.
pub struct EnvironmentInfo {
    pub model: String,
    pub cwd: PathBuf,
    pub gpu_info: Option<String>,
}

impl ProjectContext {
    /// Load from `.tcip/state.toml` if it exists.
    pub fn load(workspace: &Path) -> Option<Self> {
        let path = workspace.join(".tcip").join("state.toml");
        let content = std::fs::read_to_string(path).ok()?;
        let table: toml::Table = toml::from_str(&content).ok()?;

        Some(Self {
            crop: table
                .get("crop")
                .and_then(|v| v.as_str())
                .map(String::from),
            traits: table
                .get("traits")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            pipeline_stage: table
                .get("pipeline_stage")
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    }
}

const BASE_SYSTEM_PROMPT: &str = "\
You are an ML/CV engineering agent for tree crop breeding programs (TCIP).
You automate: pipeline design, code generation, training/HPO, annotation review,
inference, and delivery of per-plant CSV outputs.

You work on 6 crops: hazelnut, chestnut, persimmon, black_locust, currant, elderberry.
You use a two-layer pipeline paradigm: isolation model → task model → post-processing.
All models are pure PyTorch (nn.Module + torchvision detection). No Ultralytics/MMDet/HuggingFace.";

const SYSTEM_RULES: &str = "\
## Rules

1. Always query the crop registry before making assumptions about traits or pipelines.
2. Propose pipeline designs via HITL checkpoints — never launch training without approval.
3. Use progressive unfreezing with multi-stage training schedules.
4. All data in YOLO format. Convert on input/output as needed.
5. Track all artifacts (models, configs, metrics) through the artifact manager.
6. When uncertain, ask the user — don't guess about crop biology or trait definitions.
7. For annotation review, always compute IoU-based matching before presenting results.
8. Export final results as per-plant CSV with quality metadata (R², n_obs_dates).";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_frontmatter_basic() {
        let raw = r#"---
name: test-skill
description: "A test skill"
triggers:
  - detect
  - train
modes: [PipelineDesigner, CodeGenerator]
priority: high
max_chars: 3000
---

# Test Skill Content

This is the body."#;

        let skill = parse_skill_file(raw, Path::new("test-skill.md")).unwrap();
        assert_eq!(skill.metadata.name, "test-skill");
        assert_eq!(skill.metadata.triggers, vec!["detect", "train"]);
        assert_eq!(
            skill.metadata.modes,
            vec!["PipelineDesigner", "CodeGenerator"]
        );
        assert_eq!(skill.metadata.priority, SkillPriority::High);
        assert_eq!(skill.metadata.max_chars, 3000);
        assert!(skill.content.contains("# Test Skill Content"));
    }

    #[test]
    fn parse_no_frontmatter() {
        let raw = "# Just Content\n\nNo frontmatter here.";
        let skill = parse_skill_file(raw, Path::new("raw.md")).unwrap();
        assert_eq!(skill.metadata.name, "raw");
        assert_eq!(skill.metadata.priority, SkillPriority::Normal);
        assert!(skill.content.contains("Just Content"));
    }

    #[test]
    fn trigger_scoring() {
        let triggers = vec!["pipeline".to_string(), "design".to_string(), "detection".to_string()];
        // All 3 match
        assert!((score_triggers(&triggers, "design a pipeline for detection") - 1.0).abs() < f32::EPSILON);
        // 1 of 3 match
        assert!((score_triggers(&triggers, "train the model") - 0.0).abs() < f32::EPSILON);
        // 2 of 3 match
        let score = score_triggers(&triggers, "pipeline for classification, not detection");
        assert!((score - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn trigger_scoring_empty() {
        assert!((score_triggers(&[], "anything")).abs() < f32::EPSILON);
    }

    #[test]
    fn mode_from_str() {
        assert_eq!(
            "PipelineDesigner".parse::<SubagentMode>().unwrap(),
            SubagentMode::PipelineDesigner
        );
        assert!("Unknown".parse::<SubagentMode>().is_err());
    }
}

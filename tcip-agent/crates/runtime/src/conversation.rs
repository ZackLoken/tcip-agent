use crate::checkpoint::{Checkpoint, CheckpointResolution, CheckpointResolver};
use crate::compact::{CompactionConfig, Compactor};
use crate::context_injection::{ContextCollector, ContextPriority};
use crate::events::{AgentEvent, EventEmitter};
use crate::hooks::HookRunner;
use crate::interview::InterviewSession;
use crate::learner::Learner;
use crate::loop_detector::{LoopCheckResult, LoopDetector};
use crate::permission::{PermissionEnforcer, PermissionResult};
use crate::policy::{PermissionPolicy, PolicyResult};
use crate::recovery::{EscalationPolicy, RecoveryEngine, RecoveryResult};
use crate::session::{ContentBlock, ConversationMessage, MessageRole, Session, TokenUsageRecord};
use crate::skills::{EnvironmentInfo, ProjectContext, SkillInjector, SubagentMode};
use crate::usage::UsageTracker;
use crate::workflow::WorkflowContract;
use tcip_api::{ApiClient, ApiError, AssistantEvent, MessageRequest, ToolDefinition};
use tcip_tools::ToolExecutor;
use tracing::{debug, info, warn};

/// Summary of a single conversation turn.
#[derive(Debug)]
pub struct TurnSummary {
    pub assistant_text: String,
    pub tool_calls: Vec<ToolCallRecord>,
    pub checkpoint_requested: bool,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug)]
pub struct ToolCallRecord {
    pub tool_use_id: String,
    pub name: String,
    pub input: serde_json::Value,
    pub output: String,
    pub is_error: bool,
}

/// The core agentic conversation loop, adapted from claw-code's `ConversationRuntime`.
///
/// Generic over API client and tool executor for testability.
pub struct ConversationRuntime<C, T, R> {
    pub session: Session,
    api_client: C,
    tool_executor: T,
    checkpoint_resolver: R,
    permission_enforcer: PermissionEnforcer,
    skill_injector: SkillInjector,
    usage_tracker: UsageTracker,
    loop_detector: LoopDetector,
    recovery_engine: RecoveryEngine,
    compactor: Compactor,
    hook_runner: HookRunner,
    context_collector: ContextCollector,
    policy: PermissionPolicy,
    learner: Learner,
    interview: Option<InterviewSession>,
    workflow: Option<WorkflowContract>,
    active_mode: SubagentMode,
    max_iterations: usize,
    model: String,
    workspace_root: std::path::PathBuf,
    event_emitter: EventEmitter,
}

impl<C, T, R> ConversationRuntime<C, T, R>
where
    C: ApiClient,
    T: ToolExecutor,
    R: CheckpointResolver,
{
    /// Create a new runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: Session,
        api_client: C,
        tool_executor: T,
        checkpoint_resolver: R,
        permission_enforcer: PermissionEnforcer,
        skill_injector: SkillInjector,
        model: String,
        workspace_root: std::path::PathBuf,
    ) -> Self {
        Self {
            session,
            api_client,
            tool_executor,
            checkpoint_resolver,
            permission_enforcer,
            skill_injector,
            usage_tracker: UsageTracker::new(),
            loop_detector: LoopDetector::new(),
            recovery_engine: RecoveryEngine::new(),
            compactor: Compactor::default(),
            hook_runner: HookRunner::new(),
            context_collector: ContextCollector::new(),
            policy: PermissionPolicy::with_defaults(),
            learner: Learner::new(&workspace_root),
            interview: None,
            workflow: None,
            active_mode: SubagentMode::PipelineDesigner,
            max_iterations: 25,
            model,
            workspace_root,
            event_emitter: EventEmitter::noop(),
        }
    }

    /// Attach an event emitter for broadcasting agent events to sinks (GUI, log, etc.).
    pub fn set_event_emitter(&mut self, emitter: EventEmitter) {
        self.event_emitter = emitter;
    }

    /// Get a mutable reference to the hook runner for registering hooks.
    pub fn hooks_mut(&mut self) -> &mut HookRunner {
        &mut self.hook_runner
    }

    /// Configure compaction settings.
    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.compactor = Compactor::new(config);
    }

    /// Configure loop detection thresholds.
    pub fn set_loop_thresholds(&mut self, identical: usize, same_tool: usize) {
        self.loop_detector = LoopDetector::with_thresholds(identical, same_tool);
    }

    /// Get a mutable reference to the context collector.
    pub fn context_collector_mut(&mut self) -> &mut ContextCollector {
        &mut self.context_collector
    }

    /// Replace the permission policy.
    pub fn set_policy(&mut self, policy: PermissionPolicy) {
        self.policy = policy;
    }

    /// Get a mutable reference to the learner.
    pub fn learner_mut(&mut self) -> &mut Learner {
        &mut self.learner
    }

    /// Start a deep interview session.
    pub fn start_interview(&mut self) {
        self.interview = Some(InterviewSession::new());
    }

    /// Get the current interview session.
    pub fn interview(&self) -> Option<&InterviewSession> {
        self.interview.as_ref()
    }

    /// Get the current interview session mutably.
    pub fn interview_mut(&mut self) -> Option<&mut InterviewSession> {
        self.interview.as_mut()
    }

    /// Set the active workflow contract.
    pub fn set_workflow(&mut self, wf: WorkflowContract) {
        self.workflow = Some(wf);
    }

    /// Get the current workflow.
    pub fn workflow(&self) -> Option<&WorkflowContract> {
        self.workflow.as_ref()
    }

    /// Get the current workflow mutably.
    pub fn workflow_mut(&mut self) -> Option<&mut WorkflowContract> {
        self.workflow.as_mut()
    }

    /// Switch the active subagent mode.
    pub fn switch_mode(&mut self, mode: SubagentMode) {
        info!("switching mode: {} → {mode}", self.active_mode);
        self.event_emitter.emit(AgentEvent::RecoveryAttempted {
            scenario: format!("mode_switch: {} → {mode}", self.active_mode),
            success: true,
        });
        self.active_mode = mode;
        self.model = mode.preferred_model().to_string();
    }

    /// Run a single conversation turn: user input → (possibly multiple) API calls + tool executions.
    pub async fn run_turn(&mut self, user_input: &str) -> Result<TurnSummary, RuntimeError> {
        self.session
            .push_user_text(user_input)
            .map_err(RuntimeError::Session)?;

        // Reset per-turn state
        self.loop_detector.reset();
        self.recovery_engine.reset();

        // Inject learned skills context
        if let Some(learned_ctx) = self.learner.get_relevant_context(user_input) {
            self.context_collector.register(
                "learned_skills",
                "learned_matches",
                learned_ctx,
                ContextPriority::Normal,
            );
        }

        // Inject workflow status if active
        if let Some(ref wf) = self.workflow {
            if wf.status == crate::workflow::WorkflowStatus::Active {
                self.context_collector.register(
                    "workflow",
                    "workflow_status",
                    wf.status_summary(),
                    ContextPriority::Critical,
                );
            }
        }

        let mut full_text = String::new();
        let mut tool_records = Vec::new();
        let mut checkpoint_requested = false;
        let mut total_input_tokens: u32 = 0;
        let mut total_output_tokens: u32 = 0;

        for iteration in 0..self.max_iterations {
            debug!("turn iteration {iteration}");

            // Build system prompt
            let project_ctx = ProjectContext::load(&self.workspace_root);
            let env_info = EnvironmentInfo {
                model: self.model.clone(),
                cwd: self.workspace_root.clone(),
                gpu_info: None,
            };
            let injected_context = self.context_collector.consume_formatted();
            let system_prompt = crate::skills::build_system_prompt_with_context(
                &mut self.skill_injector,
                &self.active_mode,
                project_ctx.as_ref(),
                &env_info,
                Some(user_input),
                if injected_context.is_empty() { None } else { Some(&injected_context) },
            );

            // Build tool definitions
            let tool_defs: Vec<ToolDefinition> = self
                .tool_executor
                .list_tools()
                .into_iter()
                .map(|spec| ToolDefinition {
                    name: spec.name,
                    description: spec.description,
                    input_schema: spec.input_schema,
                })
                .collect();

            // Build API request
            let request = MessageRequest {
                model: self.model.clone(),
                max_tokens: 4096,
                messages: self.session.to_api_messages(),
                system: Some(system_prompt),
                tools: if tool_defs.is_empty() {
                    None
                } else {
                    Some(tool_defs)
                },
                tool_choice: None,
                stream: true,
            };

            // Call API
            let events = self
                .api_client
                .stream(request)
                .await
                .map_err(RuntimeError::Api)?;

            // Process events
            let mut assistant_blocks = Vec::new();
            let mut pending_tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut turn_usage = (0u32, 0u32);

            for event in events {
                match event {
                    AssistantEvent::TextDone(text) => {
                        full_text.push_str(&text);
                        assistant_blocks.push(ContentBlock::Text { text });
                    }
                    AssistantEvent::ToolUse { id, name, input } => {
                        assistant_blocks.push(ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                        pending_tool_uses.push((id, name, input));
                    }
                    AssistantEvent::Usage(u) => {
                        turn_usage = (u.input_tokens, u.output_tokens);
                    }
                    _ => {}
                }
            }

            total_input_tokens += turn_usage.0;
            total_output_tokens += turn_usage.1;
            self.usage_tracker.record(turn_usage.0, turn_usage.1);

            // Push assistant message
            self.session
                .push_message(ConversationMessage {
                    role: MessageRole::Assistant,
                    blocks: assistant_blocks,
                    usage: Some(TokenUsageRecord {
                        input_tokens: turn_usage.0,
                        output_tokens: turn_usage.1,
                    }),
                    timestamp: chrono::Utc::now(),
                })
                .map_err(RuntimeError::Session)?;

            // If no tool calls, turn is done
            if pending_tool_uses.is_empty() {
                break;
            }

            // Execute tool calls with loop detection, hooks, and recovery
            let mut loop_broken = false;
            for (tool_id, tool_name, tool_input) in pending_tool_uses {
                // Loop detection
                match self.loop_detector.check(&tool_name, &tool_input) {
                    LoopCheckResult::Break { message } => {
                        warn!("loop break: {message}");
                        // Inject warning as tool result
                        self.session
                            .push_message(ConversationMessage {
                                role: MessageRole::Tool,
                                blocks: vec![ContentBlock::ToolResult {
                                    tool_use_id: tool_id.clone(),
                                    tool_name: tool_name.clone(),
                                    output: message.clone(),
                                    is_error: true,
                                }],
                                usage: None,
                                timestamp: chrono::Utc::now(),
                            })
                            .map_err(RuntimeError::Session)?;
                        tool_records.push(ToolCallRecord {
                            tool_use_id: tool_id,
                            name: tool_name,
                            input: tool_input,
                            output: message,
                            is_error: true,
                        });
                        loop_broken = true;
                        break;
                    }
                    LoopCheckResult::Advisory { message } => {
                        // Log advisory but continue
                        info!("loop advisory: {message}");
                    }
                    LoopCheckResult::Ok => {}
                }

                // Pre-hooks
                let hook_result = self.hook_runner.run_pre_hooks(&tool_name, &tool_input);
                if !hook_result.proceed {
                    let block_msg = hook_result.messages.join("; ");
                    self.session
                        .push_message(ConversationMessage {
                            role: MessageRole::Tool,
                            blocks: vec![ContentBlock::ToolResult {
                                tool_use_id: tool_id.clone(),
                                tool_name: tool_name.clone(),
                                output: format!("Blocked by hook: {block_msg}"),
                                is_error: true,
                            }],
                            usage: None,
                            timestamp: chrono::Utc::now(),
                        })
                        .map_err(RuntimeError::Session)?;
                    tool_records.push(ToolCallRecord {
                        tool_use_id: tool_id,
                        name: tool_name,
                        input: tool_input,
                        output: format!("Blocked by hook: {block_msg}"),
                        is_error: true,
                    });
                    continue;
                }

                // Check policy first (rule-based), then fall back to mode-based permissions
                let input_str = tool_input.to_string();
                let policy_result = self.policy.evaluate(&tool_name, &input_str);

                let (output, is_error) = match policy_result {
                    PolicyResult::Deny { reason } => {
                        (format!("Policy denied: {reason}"), true)
                    }
                    PolicyResult::Ask { reason } => {
                        checkpoint_requested = true;
                        let checkpoint = Checkpoint::TrainingLaunch {
                            tool_name: tool_name.clone(),
                            tool_input: tool_input.clone(),
                            estimated_cost: None,
                        };
                        info!("policy asks for confirmation: {reason}");
                        match self.checkpoint_resolver.resolve(&checkpoint).await {
                            CheckpointResolution::Approved => {
                                self.execute_with_recovery(&tool_name, &tool_input).await
                            }
                            CheckpointResolution::Denied { reason } => {
                                (format!("Denied: {reason}"), true)
                            }
                            CheckpointResolution::Modified { changes } => {
                                self.execute_with_recovery(&tool_name, &changes).await
                            }
                        }
                    }
                    PolicyResult::Allow => {
                        self.execute_with_recovery(&tool_name, &tool_input).await
                    }
                    PolicyResult::NoMatch => {
                        // Fall back to existing mode-based permission enforcer
                        let perm_result = self.permission_enforcer.check(&tool_name);
                        match perm_result {
                            PermissionResult::Allowed => {
                                self.execute_with_recovery(&tool_name, &tool_input).await
                            }
                            PermissionResult::NeedsApproval { .. } => {
                                checkpoint_requested = true;
                                let checkpoint = Checkpoint::TrainingLaunch {
                                    tool_name: tool_name.clone(),
                                    tool_input: tool_input.clone(),
                                    estimated_cost: None,
                                };
                                match self.checkpoint_resolver.resolve(&checkpoint).await {
                                    CheckpointResolution::Approved => {
                                        self.execute_with_recovery(&tool_name, &tool_input).await
                                    }
                                    CheckpointResolution::Denied { reason } => {
                                        (format!("Denied: {reason}"), true)
                                    }
                                    CheckpointResolution::Modified { changes } => {
                                        self.execute_with_recovery(&tool_name, &changes).await
                                    }
                                }
                            }
                            PermissionResult::Denied { reason } => {
                                (format!("Permission denied: {reason}"), true)
                            }
                        }
                    }
                };

                // Post-hooks
                if !is_error {
                    let post_result =
                        self.hook_runner
                            .run_post_hooks(&tool_name, &tool_input, &output);
                    for feedback in &post_result.feedback {
                        info!("hook feedback: {feedback}");
                    }
                }

                tool_records.push(ToolCallRecord {
                    tool_use_id: tool_id.clone(),
                    name: tool_name.clone(),
                    input: tool_input,
                    output: output.clone(),
                    is_error,
                });

                // Push tool result message
                self.session
                    .push_message(ConversationMessage {
                        role: MessageRole::Tool,
                        blocks: vec![ContentBlock::ToolResult {
                            tool_use_id: tool_id,
                            tool_name,
                            output,
                            is_error,
                        }],
                        usage: None,
                        timestamp: chrono::Utc::now(),
                    })
                    .map_err(RuntimeError::Session)?;
            }

            if loop_broken {
                // After a loop break, continue to next iteration so the model
                // sees the error and can try a different approach
                continue;
            }

            // Check if compaction is needed after this iteration
            if self.compactor.needs_compaction(
                self.usage_tracker.total_input_tokens + self.usage_tracker.total_output_tokens,
            ) {
                info!("token threshold exceeded, triggering compaction");
                let messages = self.session.messages().to_vec();
                match self
                    .compactor
                    .compact(&messages, &self.api_client, &self.model)
                    .await
                {
                    Ok((compacted, result)) => {
                        info!(
                            "compaction complete: {} messages → {}, ~{} → ~{} tokens",
                            result.messages_summarized,
                            compacted.len(),
                            result.tokens_before,
                            result.tokens_after,
                        );
                        self.session.replace_messages(compacted);
                    }
                    Err(e) => {
                        warn!("compaction failed: {e}, continuing without compaction");
                    }
                }
            }

            // Loop back — API will see tool results and may call more tools
        }

        // Emit turn-completed event
        self.event_emitter.emit(AgentEvent::TurnCompleted {
            turn_id: self.session.messages().len() as u32,
            tool_calls: tool_records.len() as u32,
            tokens_used: (total_input_tokens + total_output_tokens) as u64,
        });

        Ok(TurnSummary {
            assistant_text: full_text,
            tool_calls: tool_records,
            checkpoint_requested,
            input_tokens: total_input_tokens,
            output_tokens: total_output_tokens,
        })
    }

    /// Execute a tool with recovery on failure.
    async fn execute_with_recovery(
        &mut self,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) -> (String, bool) {
        match self.tool_executor.execute(tool_name, tool_input).await {
            Ok(result) => (result, false),
            Err(e) => {
                let error_str = format!("{e}");

                // Emit error event
                self.event_emitter.emit(AgentEvent::Error {
                    message: format!("Tool '{tool_name}' failed: {error_str}"),
                    recoverable: true,
                });
                // Run failure hooks
                let failure_result =
                    self.hook_runner
                        .run_failure_hooks(tool_name, tool_input, &error_str);

                // Try recovery
                match self.recovery_engine.try_recover(tool_name, &error_str) {
                    RecoveryResult::Retry { hint } => {
                        let combined_hint = if let Some(hook_hint) = failure_result.recovery_hint {
                            format!("{hint} | Hook hint: {hook_hint}")
                        } else {
                            hint
                        };
                        (
                            format!(
                                "Error: {error_str}\n\nRecovery hint: {combined_hint}"
                            ),
                            true,
                        )
                    }
                    RecoveryResult::Escalate { scenario, policy } => match policy {
                        EscalationPolicy::Abort => {
                            (format!("Error: {error_str}\n\nRecovery exhausted for {scenario:?}. Aborting."), true)
                        }
                        EscalationPolicy::AlertHuman => {
                            (format!("Error: {error_str}\n\nRecovery exhausted for {scenario:?}. Human intervention needed."), true)
                        }
                        EscalationPolicy::LogAndContinue => {
                            warn!("recovery exhausted for {scenario:?}, continuing");
                            (format!("Error: {error_str}"), true)
                        }
                    },
                    RecoveryResult::NoRecipe => {
                        (format!("Error: {error_str}"), true)
                    }
                }
            }
        }
    }

    /// Get cumulative usage stats.
    #[must_use]
    pub fn usage(&self) -> &UsageTracker {
        &self.usage_tracker
    }

    /// Get the active subagent mode.
    #[must_use]
    pub fn mode(&self) -> SubagentMode {
        self.active_mode
    }

    /// Get the active model name.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("API error: {0}")]
    Api(#[from] ApiError),
    #[error("session error: {0}")]
    Session(#[from] crate::session::SessionError),
    #[error("tool error: {0}")]
    Tool(#[from] tcip_tools::ToolError),
}

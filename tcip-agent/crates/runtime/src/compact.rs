use crate::session::{ContentBlock, ConversationMessage, MessageRole};
use tcip_api::{ApiClient, MessageRequest};
use tracing::info;

/// Configuration for context window auto-compaction.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Number of recent messages to preserve verbatim.
    pub preserve_recent: usize,
    /// Token threshold that triggers compaction.
    pub max_estimated_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            preserve_recent: 4,
            max_estimated_tokens: 100_000,
        }
    }
}

/// Result of a compaction operation.
#[derive(Debug)]
pub struct CompactionResult {
    /// How many messages were summarized.
    pub messages_summarized: usize,
    /// Estimated tokens before compaction.
    pub tokens_before: usize,
    /// Estimated tokens after compaction.
    pub tokens_after: usize,
}

/// Manages context window compaction to prevent hitting token limits.
///
/// When estimated token usage exceeds the configured threshold, older messages
/// are summarized into a compact `<summary>` block, preserving recent context.
pub struct Compactor {
    config: CompactionConfig,
}

impl Compactor {
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    /// Check whether compaction is needed based on current token usage.
    pub fn needs_compaction(&self, total_tokens: u64) -> bool {
        total_tokens as usize >= self.config.max_estimated_tokens
    }

    /// Perform compaction on a message history.
    ///
    /// Splits messages into [old_prefix, recent_tail], summarizes the prefix
    /// via an API call, and returns the compacted message list.
    pub async fn compact<C: ApiClient>(
        &self,
        messages: &[ConversationMessage],
        api_client: &C,
        model: &str,
    ) -> Result<(Vec<ConversationMessage>, CompactionResult), CompactionError> {
        let total = messages.len();

        if total <= self.config.preserve_recent {
            return Err(CompactionError::TooFewMessages);
        }

        let split_point = total.saturating_sub(self.config.preserve_recent);
        let old_prefix = &messages[..split_point];
        let recent_tail = &messages[split_point..];

        let tokens_before = estimate_tokens(messages);

        // Check if there's an existing summary to merge
        let existing_summary = extract_existing_summary(old_prefix);

        // Build summarization prompt
        let summary_prompt = build_summary_prompt(old_prefix, existing_summary.as_deref());

        // Call API for summarization
        let summary_text = call_summary_api(api_client, model, &summary_prompt).await?;

        // Build compacted message list
        let summary_message = ConversationMessage {
            role: MessageRole::User,
            blocks: vec![ContentBlock::Text {
                text: format!("<summary>\n{summary_text}\n</summary>\n\nContinue from where we left off."),
            }],
            usage: None,
            timestamp: chrono::Utc::now(),
        };

        let mut compacted = vec![summary_message];
        compacted.extend_from_slice(recent_tail);

        let tokens_after = estimate_tokens(&compacted);

        info!(
            "compaction: {split_point} messages summarized, ~{tokens_before} → ~{tokens_after} tokens"
        );

        Ok((
            compacted,
            CompactionResult {
                messages_summarized: split_point,
                tokens_before,
                tokens_after,
            },
        ))
    }
}

impl Default for Compactor {
    fn default() -> Self {
        Self::new(CompactionConfig::default())
    }
}

/// Extract an existing `<summary>` block from the first message, if present.
fn extract_existing_summary(messages: &[ConversationMessage]) -> Option<String> {
    messages.first().and_then(|msg| {
        msg.blocks.iter().find_map(|block| {
            if let ContentBlock::Text { text } = block {
                if text.contains("<summary>") {
                    let start = text.find("<summary>")? + "<summary>".len();
                    let end = text.find("</summary>")?;
                    Some(text[start..end].trim().to_string())
                } else {
                    None
                }
            } else {
                None
            }
        })
    })
}

/// Build the summarization prompt from conversation history.
fn build_summary_prompt(messages: &[ConversationMessage], existing_summary: Option<&str>) -> String {
    let mut prompt = String::from(
        "Summarize the following conversation context. Preserve:\n\
         - Active crop and trait names\n\
         - Current pipeline stage and design decisions\n\
         - Pending tool results or errors\n\
         - Key configuration values (model, batch size, LR, etc.)\n\
         - File paths that were created or modified\n\n\
         Be concise — use bullet points, no prose.\n\n",
    );

    if let Some(prev) = existing_summary {
        prompt.push_str("Previous summary to merge with:\n");
        prompt.push_str(prev);
        prompt.push_str("\n\n---\n\nNew messages to incorporate:\n\n");
    }

    for msg in messages {
        let role_label = match msg.role {
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::Tool => "Tool",
        };

        for block in &msg.blocks {
            match block {
                ContentBlock::Text { text } => {
                    // Truncate very long text blocks to avoid summary prompt itself being huge
                    let truncated = if text.len() > 500 {
                        format!("{}... [truncated]", &text[..500])
                    } else {
                        text.clone()
                    };
                    prompt.push_str(&format!("[{role_label}]: {truncated}\n"));
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    let input_str = serde_json::to_string(input).unwrap_or_default();
                    let truncated = if input_str.len() > 200 {
                        format!("{}...", &input_str[..200])
                    } else {
                        input_str
                    };
                    prompt.push_str(&format!("[{role_label} → {name}]: {truncated}\n"));
                }
                ContentBlock::ToolResult {
                    tool_name, output, is_error, ..
                } => {
                    let status = if *is_error { "ERROR" } else { "OK" };
                    let truncated = if output.len() > 300 {
                        format!("{}...", &output[..300])
                    } else {
                        output.clone()
                    };
                    prompt.push_str(&format!("[{role_label} ← {tool_name} ({status})]: {truncated}\n"));
                }
            }
        }
    }

    prompt
}

/// Call the API to produce a summary.
async fn call_summary_api<C: ApiClient>(
    api_client: &C,
    model: &str,
    prompt: &str,
) -> Result<String, CompactionError> {
    use tcip_api::{AssistantEvent, ContentBlock as ApiContentBlock, InputMessage, MessageRole as ApiRole};

    let request = MessageRequest {
        model: model.to_string(),
        max_tokens: 1024,
        messages: vec![InputMessage {
            role: ApiRole::User,
            content: vec![ApiContentBlock::Text {
                text: prompt.to_string(),
            }],
        }],
        system: Some("You are a conversation summarizer. Output only the summary, nothing else.".to_string()),
        tools: None,
        tool_choice: None,
        stream: true,
    };

    let events = api_client.stream(request).await.map_err(CompactionError::Api)?;

    let mut summary = String::new();
    for event in events {
        if let AssistantEvent::TextDone(text) = event {
            summary.push_str(&text);
        }
    }

    if summary.is_empty() {
        return Err(CompactionError::EmptySummary);
    }

    Ok(summary)
}

/// Rough token estimation: ~4 chars per token.
fn estimate_tokens(messages: &[ConversationMessage]) -> usize {
    let total_chars: usize = messages
        .iter()
        .flat_map(|m| &m.blocks)
        .map(|block| match block {
            ContentBlock::Text { text } => text.len(),
            ContentBlock::ToolUse { input, .. } => {
                serde_json::to_string(input).unwrap_or_default().len()
            }
            ContentBlock::ToolResult { output, .. } => output.len(),
        })
        .sum();

    total_chars / 4
}

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("too few messages to compact")]
    TooFewMessages,
    #[error("API error during summarization: {0}")]
    Api(#[from] tcip_api::ApiError),
    #[error("summarization returned empty result")]
    EmptySummary,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_msg(role: MessageRole, text: &str) -> ConversationMessage {
        ConversationMessage {
            role,
            blocks: vec![ContentBlock::Text { text: text.to_string() }],
            usage: None,
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn needs_compaction_threshold() {
        let compactor = Compactor::new(CompactionConfig {
            preserve_recent: 4,
            max_estimated_tokens: 1000,
        });
        assert!(!compactor.needs_compaction(999));
        assert!(compactor.needs_compaction(1000));
        assert!(compactor.needs_compaction(2000));
    }

    #[test]
    fn extract_summary_from_message() {
        let msg = text_msg(
            MessageRole::User,
            "<summary>\n- Working on hazelnut catkins\n- Using FCOS model\n</summary>\n\nContinue.",
        );
        let summary = extract_existing_summary(&[msg]);
        assert_eq!(summary.unwrap(), "- Working on hazelnut catkins\n- Using FCOS model");
    }

    #[test]
    fn estimate_tokens_rough() {
        let msgs = vec![
            text_msg(MessageRole::User, "a".repeat(400).as_str()),
            text_msg(MessageRole::Assistant, "b".repeat(400).as_str()),
        ];
        // 800 chars / 4 = 200 tokens
        assert_eq!(estimate_tokens(&msgs), 200);
    }

    #[test]
    fn build_prompt_includes_messages() {
        let msgs = vec![
            text_msg(MessageRole::User, "design a pipeline for hazelnut"),
            text_msg(MessageRole::Assistant, "I'll use FCOS for detection"),
        ];
        let prompt = build_summary_prompt(&msgs, None);
        assert!(prompt.contains("[User]: design a pipeline"));
        assert!(prompt.contains("[Assistant]: I'll use FCOS"));
    }

    #[test]
    fn build_prompt_merges_existing_summary() {
        let msgs = vec![text_msg(MessageRole::User, "next step")];
        let prompt = build_summary_prompt(&msgs, Some("- Previous context here"));
        assert!(prompt.contains("Previous summary to merge with:"));
        assert!(prompt.contains("- Previous context here"));
    }
}

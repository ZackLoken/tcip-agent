use crate::types::{ApiClient, ApiError, AssistantEvent, MessageRequest, TokenUsage};

/// Offline stub client used when no API key is available.
/// Echoes back a message so the GUI can still be tested.
pub struct EchoClient;

impl EchoClient {
    pub fn new() -> Self {
        Self
    }
}

impl ApiClient for EchoClient {
    async fn stream(
        &self,
        request: MessageRequest,
    ) -> Result<Vec<AssistantEvent>, ApiError> {
        // Extract the last user message text for the echo
        let user_text = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == crate::types::MessageRole::User)
            .and_then(|m| {
                m.content.iter().find_map(|c| match c {
                    crate::types::ContentBlock::Text { text } => Some(text.clone()),
                    crate::types::ContentBlock::ToolResult { content, .. } => {
                        Some(content.clone())
                    }
                    _ => None,
                })
            })
            .unwrap_or_default();

        let reply = format!(
            "**[Offline Mode]** No ANTHROPIC_API_KEY set — running without an LLM backend.\n\n\
             Your message: \"{user_text}\"\n\n\
             The GUI, tools, and MCP server are fully functional. \
             Set `ANTHROPIC_API_KEY` to enable the AI agent."
        );

        Ok(vec![
            AssistantEvent::TextDone(reply),
            AssistantEvent::Usage(TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }),
            AssistantEvent::MessageStop,
        ])
    }
}

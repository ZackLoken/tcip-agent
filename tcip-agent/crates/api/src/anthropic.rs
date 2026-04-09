use crate::types::{
    ApiClient, ApiError, AssistantEvent, ContentBlockDelta, ContentBlockStartData,
    MessageRequest, StreamEvent, TokenUsage,
};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use std::time::Duration;
use tracing::{debug, warn};

/// Anthropic Messages API client with SSE streaming and retry.
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    max_retries: u32,
    initial_backoff: Duration,
}

impl AnthropicClient {
    /// Create a new client. Reads `ANTHROPIC_API_KEY` from env if `api_key` is None.
    pub fn new(api_key: Option<String>, base_url: Option<String>) -> Result<Self, ApiError> {
        let api_key = api_key
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .ok_or(ApiError::MissingCredentials)?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(ApiError::Http)?;

        Ok(Self {
            http,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            max_retries: 2,
            initial_backoff: Duration::from_millis(200),
        })
    }

    async fn do_stream(
        &self,
        request: &MessageRequest,
    ) -> Result<Vec<AssistantEvent>, ApiError> {
        let url = format!("{}/v1/messages", self.base_url);

        let response = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .json(request)
            .send()
            .await
            .map_err(ApiError::Http)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();

            // Parse error type from body
            let (error_type, message) = if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                let et = v["error"]["type"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                let msg = v["error"]["message"]
                    .as_str()
                    .unwrap_or(&body)
                    .to_string();
                (et, msg)
            } else {
                ("unknown".to_string(), body)
            };

            let retryable = status.as_u16() == 429
                || status.as_u16() == 529
                || status.is_server_error();

            return Err(ApiError::Api {
                status: status.as_u16(),
                error_type,
                message,
                retryable,
            });
        }

        let body = response.text().await.map_err(ApiError::Http)?;
        self.parse_sse_body(&body)
    }

    fn parse_sse_body(&self, body: &str) -> Result<Vec<AssistantEvent>, ApiError> {
        let mut events = Vec::new();
        let mut current_text = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_json = String::new();
        let mut in_tool = false;
        let mut usage = TokenUsage::default();

        for line in body.lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }

            let event: StreamEvent = match serde_json::from_str(data) {
                Ok(e) => e,
                Err(e) => {
                    debug!("skipping unparseable SSE line: {e}");
                    continue;
                }
            };

            match event {
                StreamEvent::MessageStart { message } => {
                    usage = message.usage;
                }
                StreamEvent::ContentBlockStart { content_block, .. } => match content_block {
                    ContentBlockStartData::Text { text } => {
                        current_text = text;
                        in_tool = false;
                    }
                    ContentBlockStartData::ToolUse { id, name } => {
                        // Flush pending text
                        if !current_text.is_empty() {
                            events.push(AssistantEvent::TextDone(
                                std::mem::take(&mut current_text),
                            ));
                        }
                        current_tool_id = id;
                        current_tool_name = name;
                        current_tool_json.clear();
                        in_tool = true;
                    }
                },
                StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                    ContentBlockDelta::TextDelta { text } => {
                        events.push(AssistantEvent::TextDelta(text.clone()));
                        current_text.push_str(&text);
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        current_tool_json.push_str(&partial_json);
                    }
                },
                StreamEvent::ContentBlockStop { .. } => {
                    if in_tool {
                        let input: serde_json::Value =
                            serde_json::from_str(&current_tool_json).unwrap_or_default();
                        events.push(AssistantEvent::ToolUse {
                            id: std::mem::take(&mut current_tool_id),
                            name: std::mem::take(&mut current_tool_name),
                            input,
                        });
                        current_tool_json.clear();
                        in_tool = false;
                    } else if !current_text.is_empty() {
                        events.push(AssistantEvent::TextDone(
                            std::mem::take(&mut current_text),
                        ));
                    }
                }
                StreamEvent::MessageDelta {
                    usage: Some(u),
                    ..
                } => {
                    usage.output_tokens = u.output_tokens;
                }
                StreamEvent::MessageStop => {
                    events.push(AssistantEvent::Usage(usage.clone()));
                    events.push(AssistantEvent::MessageStop);
                }
                _ => {}
            }
        }

        Ok(events)
    }
}

impl ApiClient for AnthropicClient {
    async fn stream(
        &self,
        request: MessageRequest,
    ) -> Result<Vec<AssistantEvent>, ApiError> {
        let mut last_error: Option<ApiError> = None;
        let mut backoff = self.initial_backoff;

        for attempt in 0..=self.max_retries {
            match self.do_stream(&request).await {
                Ok(events) => return Ok(events),
                Err(e) => {
                    let retryable = matches!(
                        &e,
                        ApiError::Api { retryable: true, .. } | ApiError::Http(_)
                    );

                    if !retryable || attempt == self.max_retries {
                        if attempt > 0 {
                            return Err(ApiError::RetriesExhausted {
                                attempts: attempt + 1,
                                last_error: Box::new(e),
                            });
                        }
                        return Err(e);
                    }

                    warn!(
                        attempt = attempt + 1,
                        max = self.max_retries,
                        "retryable API error, backing off: {e}"
                    );
                    last_error = Some(e);
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(2));
                }
            }
        }

        Err(ApiError::RetriesExhausted {
            attempts: self.max_retries + 1,
            last_error: Box::new(last_error.unwrap_or(ApiError::MissingCredentials)),
        })
    }
}

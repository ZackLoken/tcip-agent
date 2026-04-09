#[cfg(test)]
mod tests {
    use tcip_api::*;
    use serde_json::json;

    // ── Mock API client ──

    struct MockApiClient {
        responses: Vec<Vec<AssistantEvent>>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockApiClient {
        fn new(responses: Vec<Vec<AssistantEvent>>) -> Self {
            Self {
                responses,
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn text_response(text: &str) -> Vec<AssistantEvent> {
            vec![
                AssistantEvent::TextDone(text.to_string()),
                AssistantEvent::Usage(TokenUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                }),
                AssistantEvent::MessageStop,
            ]
        }

        fn tool_then_text(tool_name: &str, tool_input: serde_json::Value, text: &str) -> Vec<Vec<AssistantEvent>> {
            vec![
                // First call: tool use
                vec![
                    AssistantEvent::ToolUse {
                        id: "tool_01".to_string(),
                        name: tool_name.to_string(),
                        input: tool_input,
                    },
                    AssistantEvent::Usage(TokenUsage {
                        input_tokens: 200,
                        output_tokens: 100,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    }),
                    AssistantEvent::MessageStop,
                ],
                // Second call: text response after tool result
                Self::text_response(text),
            ]
        }
    }

    impl ApiClient for MockApiClient {
        async fn stream(
            &self,
            _request: MessageRequest,
        ) -> Result<Vec<AssistantEvent>, ApiError> {
            let idx = self.call_count.load(std::sync::atomic::Ordering::SeqCst);
            self.call_count.store(idx + 1, std::sync::atomic::Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                Ok(MockApiClient::text_response("(no more responses)"))
            }
        }
    }

    // ── Mock tool executor ──

    struct MockToolExecutor {
        results: std::collections::HashMap<String, String>,
    }

    impl MockToolExecutor {
        fn new() -> Self {
            Self {
                results: std::collections::HashMap::new(),
            }
        }

        fn with_tool(mut self, name: &str, result: &str) -> Self {
            self.results.insert(name.to_string(), result.to_string());
            self
        }
    }

    impl tcip_tools::ToolExecutor for MockToolExecutor {
        async fn execute(
            &mut self,
            tool_name: &str,
            _input: &serde_json::Value,
        ) -> Result<String, tcip_tools::ToolError> {
            self.results
                .get(tool_name)
                .cloned()
                .ok_or_else(|| tcip_tools::ToolError::UnknownTool(tool_name.to_string()))
        }

        fn list_tools(&self) -> Vec<tcip_tools::ToolSpec> {
            self.results
                .keys()
                .map(|name| tcip_tools::ToolSpec {
                    name: name.clone(),
                    description: format!("Mock {name}"),
                    input_schema: json!({"type": "object"}),
                    required_permission: tcip_tools::PermissionLevel::ReadOnly,
                })
                .collect()
        }
    }

    // ── Mock checkpoint resolver ──

    struct AutoApproveResolver;

    impl tcip_runtime::checkpoint::CheckpointResolver for AutoApproveResolver {
        async fn resolve(
            &self,
            _checkpoint: &tcip_runtime::checkpoint::Checkpoint,
        ) -> tcip_runtime::checkpoint::CheckpointResolution {
            tcip_runtime::checkpoint::CheckpointResolution::Approved
        }
    }

    #[allow(dead_code)]
    struct AutoDenyResolver;

    impl tcip_runtime::checkpoint::CheckpointResolver for AutoDenyResolver {
        async fn resolve(
            &self,
            _checkpoint: &tcip_runtime::checkpoint::Checkpoint,
        ) -> tcip_runtime::checkpoint::CheckpointResolution {
            tcip_runtime::checkpoint::CheckpointResolution::Denied {
                reason: "test denied".to_string(),
            }
        }
    }

    // ── Helper ──

    fn make_runtime<C: ApiClient, T: tcip_tools::ToolExecutor>(
        api: C,
        tools: T,
    ) -> tcip_runtime::ConversationRuntime<C, T, AutoApproveResolver> {
        let session = tcip_runtime::Session::new("test-session".to_string());
        let enforcer = tcip_runtime::PermissionEnforcer::new(tcip_runtime::PermissionMode::WorkspaceWrite);
        let skill_injector = tcip_runtime::SkillInjector::new(std::path::PathBuf::from("nonexistent"));

        tcip_runtime::ConversationRuntime::new(
            session,
            api,
            tools,
            AutoApproveResolver,
            enforcer,
            skill_injector,
            "test-model".to_string(),
            std::path::PathBuf::from("."),
        )
    }

    // ═══ TESTS ═══

    #[tokio::test]
    async fn test_simple_text_response() {
        let api = MockApiClient::new(vec![MockApiClient::text_response("Hello, I am TCIP Agent.")]);
        let tools = MockToolExecutor::new();
        let mut runtime = make_runtime(api, tools);

        let summary = runtime.run_turn("Hi").await.unwrap();
        assert_eq!(summary.assistant_text, "Hello, I am TCIP Agent.");
        assert!(summary.tool_calls.is_empty());
        assert_eq!(summary.input_tokens, 100);
        assert_eq!(summary.output_tokens, 50);
    }

    #[tokio::test]
    async fn test_tool_call_and_response() {
        let api = MockApiClient::new(MockApiClient::tool_then_text(
            "mcp__list_crops",
            json!({}),
            "The available crops are: hazelnut, chestnut, persimmon.",
        ));
        let tools = MockToolExecutor::new()
            .with_tool("mcp__list_crops", r#"["hazelnut","chestnut","persimmon"]"#);
        let mut runtime = make_runtime(api, tools);

        let summary = runtime.run_turn("What crops are available?").await.unwrap();
        assert_eq!(summary.tool_calls.len(), 1);
        assert_eq!(summary.tool_calls[0].name, "mcp__list_crops");
        assert!(!summary.tool_calls[0].is_error);
        assert!(summary.assistant_text.contains("hazelnut"));
    }

    #[tokio::test]
    async fn test_session_message_tracking() {
        let api = MockApiClient::new(vec![MockApiClient::text_response("Got it.")]);
        let tools = MockToolExecutor::new();
        let mut runtime = make_runtime(api, tools);

        assert_eq!(runtime.session.message_count(), 0);
        runtime.run_turn("Hello").await.unwrap();
        // Should have: 1 user message + 1 assistant message
        assert_eq!(runtime.session.message_count(), 2);
    }

    #[tokio::test]
    async fn test_usage_tracking() {
        let api = MockApiClient::new(vec![
            MockApiClient::text_response("First"),
            MockApiClient::text_response("Second"),
        ]);
        let tools = MockToolExecutor::new();
        let mut runtime = make_runtime(api, tools);

        runtime.run_turn("one").await.unwrap();
        runtime.run_turn("two").await.unwrap();

        let usage = runtime.usage();
        assert_eq!(usage.turn_count, 2);
        assert_eq!(usage.total_input_tokens, 200);
        assert_eq!(usage.total_output_tokens, 100);
    }

    #[tokio::test]
    async fn test_mode_switching() {
        let api = MockApiClient::new(vec![]);
        let tools = MockToolExecutor::new();
        let mut runtime = make_runtime(api, tools);

        assert_eq!(
            runtime.mode(),
            tcip_runtime::skills::SubagentMode::PipelineDesigner
        );

        runtime.switch_mode(tcip_runtime::skills::SubagentMode::TrainingOrchestrator);
        assert_eq!(
            runtime.mode(),
            tcip_runtime::skills::SubagentMode::TrainingOrchestrator
        );
    }
}

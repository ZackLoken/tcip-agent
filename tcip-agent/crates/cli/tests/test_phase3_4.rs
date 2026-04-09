//! Integration tests for Phase 3-4 subsystems:
//! learner, interview, workflow, subagent, events, hashline.

#[cfg(test)]
mod tests {
    use tcip_runtime::learner::{Learner, SessionSummary, ToolCallSummary};
    use tcip_runtime::interview::{InterviewSession, InterviewStatus};
    use tcip_runtime::workflow::{
        VerificationType, VerificationOutcome,
        WorkflowContract, WorkflowStatus, criterion,
    };
    use tcip_runtime::subagent::{SubagentConfig, SubagentSpawner, SubagentResult, TaskStatus};
    use tcip_runtime::events::{AgentEvent, EventBus, EventEmitter, JsonLogSink, EventSink};
    use tcip_runtime::context_injection::{ContextCollector, ContextPriority};

    // ═══ LEARNER INTEGRATION ═══

    #[test]
    fn learner_end_to_end_learn_and_recall() {
        let tmp = tempfile::tempdir().unwrap();
        let mut learner = Learner::new(tmp.path());

        // Session with a successful annotation workflow
        let summary = SessionSummary {
            session_id: "session-001".to_string(),
            tool_calls: vec![
                ToolCallSummary { name: "list_images".to_string(), input_keys: vec!["hazelnut".to_string()], succeeded: true },
                ToolCallSummary { name: "annotate_batch".to_string(), input_keys: vec!["catkin".to_string()], succeeded: true },
                ToolCallSummary { name: "train_model".to_string(), input_keys: vec!["detect".to_string()], succeeded: true },
                ToolCallSummary { name: "eval_metrics".to_string(), input_keys: vec!["mAP".to_string()], succeeded: true },
            ],
            had_errors: false,
            error_count: 0,
            final_text: "Training complete. mAP@50 = 0.78".to_string(),
        };

        // Learn from session
        let skill_name = learner.on_session_end(&summary);
        assert!(skill_name.is_some());

        // Now a new session asks about similar work
        let context = learner.get_relevant_context("I need to annotate hazelnut catkins and train a detector");
        assert!(context.is_some());
        let ctx = context.unwrap();
        assert!(ctx.contains("Learned"));
    }

    #[test]
    fn learner_persists_across_instances() {
        let tmp = tempfile::tempdir().unwrap();

        // First instance learns
        {
            let mut learner = Learner::new(tmp.path());
            let summary = SessionSummary {
                session_id: "s1".to_string(),
                tool_calls: vec![
                    ToolCallSummary { name: "read_file".to_string(), input_keys: vec![], succeeded: true },
                    ToolCallSummary { name: "write_file".to_string(), input_keys: vec![], succeeded: true },
                    ToolCallSummary { name: "run_command".to_string(), input_keys: vec![], succeeded: true },
                ],
                had_errors: false,
                error_count: 0,
                final_text: "Done".to_string(),
            };
            learner.on_session_end(&summary);
        }

        // Second instance should see the skill
        let learner2 = Learner::new(tmp.path());
        assert!(!learner2.store.skills().is_empty());
    }

    // ═══ INTERVIEW INTEGRATION ═══

    #[test]
    fn interview_full_flow_to_spec() {
        let mut session = InterviewSession::new();

        // Initial vague request
        session.score_initial_message("help me count hazelnuts");
        assert!(session.profile.ambiguity() > 0.3);

        // Simulate multi-round Q&A
        let mut rounds = 0;
        while session.status == InterviewStatus::Active && rounds < 10 {
            if let Some(q) = session.next_question() {
                let answer = match q.dimension.as_str() {
                    "intent_clarity" => "I need object detection to count individual hazelnut clusters",
                    "crop_trait_specificity" => "Hazelnut, catkin clusters, early flowering stage",
                    "data_specification" => "RGB drone images, 1cm GSD, ~2000 images, no existing labels",
                    "success_criteria" => "mAP@50 >= 0.75, must handle partial occlusion",
                    "constraint_clarity" => "Single RTX 4090, 48 hours training max",
                    "context_clarity" => "From scratch, no existing models",
                    _ => "Not sure",
                };
                session.process_answer(&q.dimension, answer);
                rounds += 1;
            } else {
                break;
            }
        }

        // Should be resolved or close to it
        assert!(
            session.status == InterviewStatus::Resolved || session.profile.ambiguity() < 0.5,
            "ambiguity should decrease: {:.0}%",
            session.profile.ambiguity() * 100.0
        );

        // Build spec
        let spec = session.build_spec();
        assert!(spec.contains("hazelnut") || spec.contains("detection") || spec.contains("Execution"));
    }

    // ═══ WORKFLOW INTEGRATION ═══

    #[test]
    fn workflow_full_lifecycle_with_retry() {
        let mut wf = WorkflowContract::new("wf-test", "Train elderberry detector");
        wf.add_story("Data prep", vec![
            criterion("500+ labeled images", VerificationType::AutoTest, None),
        ]);
        wf.add_story("Training", vec![
            criterion("mAP@50 >= 0.70", VerificationType::MetricThreshold, Some(0.70)),
        ]);

        // Story 1: pass on first try
        wf.start_current_story();
        wf.submit_for_verification();
        wf.pass_criterion(0, 0, Some("523 images".into()));
        let outcome = wf.resolve_verification();
        assert_eq!(outcome, VerificationOutcome::StoryComplete { next_story: 1 });

        // Story 2: fail first, then pass
        wf.start_current_story();
        wf.submit_for_verification();
        // Don't pass criterion → retry
        match wf.resolve_verification() {
            VerificationOutcome::Retry { attempt, .. } => assert_eq!(attempt, 1),
            other => panic!("expected Retry, got {other:?}"),
        }

        // Retry with passing criterion
        wf.submit_for_verification();
        wf.pass_criterion(1, 0, Some("mAP@50 = 0.74".into()));
        let outcome = wf.resolve_verification();
        assert_eq!(outcome, VerificationOutcome::WorkflowComplete);
        assert_eq!(wf.status, WorkflowStatus::Complete);
    }

    #[test]
    fn workflow_save_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut wf = WorkflowContract::new("wf-persist", "Test workflow");
        wf.add_story("Story 1", vec![
            criterion("Test criterion", VerificationType::AutoTest, None),
        ]);
        wf.start_current_story();

        let path = wf.save(tmp.path()).unwrap();
        assert!(path.exists());

        let loaded = WorkflowContract::load(tmp.path(), "wf-persist").unwrap();
        assert_eq!(loaded.title, "Test workflow");
        assert_eq!(loaded.stories.len(), 1);
    }

    #[test]
    fn workflow_status_injects_to_context() {
        let mut wf = WorkflowContract::new("wf-ctx", "Detection pipeline");
        wf.add_story("Annotate", vec![criterion("Labels exist", VerificationType::AutoTest, None)]);
        wf.add_story("Train", vec![criterion("Model trained", VerificationType::AutoTest, None)]);
        wf.start_current_story();

        let mut collector = ContextCollector::new();
        collector.register("workflow", "status", wf.status_summary(), ContextPriority::Critical);

        let formatted = collector.consume_formatted();
        assert!(formatted.contains("Detection pipeline"));
        assert!(formatted.contains("Annotate"));
    }

    // ═══ SUBAGENT INTEGRATION ═══

    #[test]
    fn subagent_lifecycle() {
        let mut spawner = SubagentSpawner::new(SubagentConfig {
            max_concurrent: 2,
            ..Default::default()
        });

        assert!(spawner.can_spawn());

        let task = spawner.create_task(
            tcip_runtime::skills::SubagentMode::CodeGenerator,
            "Write augmentation pipeline",
            "parent-session-001",
        );
        assert!(!task.id.is_empty());

        spawner.on_start();
        spawner.on_start();
        assert!(!spawner.can_spawn()); // at capacity

        spawner.on_complete(SubagentResult {
            task_id: task.id.clone(),
            status: TaskStatus::Completed,
            output: "Pipeline written to augment.py".to_string(),
            tool_calls: vec!["write_file".to_string()],
            tokens_used: 3000,
            completed_at: chrono::Utc::now(),
        });

        assert!(spawner.can_spawn());
        assert_eq!(spawner.completed().len(), 1);
        assert_eq!(spawner.completed()[0].status, TaskStatus::Completed);
    }

    // ═══ EVENT BUS INTEGRATION ═══

    #[tokio::test]
    async fn event_bus_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let sink = JsonLogSink::new(tmp.path());
        let sinks: Vec<Box<dyn EventSink>> = vec![Box::new(sink)];

        let (bus, handle) = EventBus::new(sinks);

        // Emit several events
        bus.emit(AgentEvent::SessionStarted {
            session_id: "s-test".to_string(),
            crop: Some("chestnut".to_string()),
        });
        bus.emit(AgentEvent::TrainingStarted {
            model: "fasterrcnn".to_string(),
            dataset: "chestnut_v3".to_string(),
            epochs: 50,
        });
        bus.emit(AgentEvent::TrainingFinished {
            model: "fasterrcnn".to_string(),
            final_metrics: std::collections::HashMap::from([("mAP50".to_string(), 0.82)]),
        });

        drop(bus);
        handle.await.unwrap();

        let log_path = tmp.path().join(".tcip").join("events.jsonl");
        let content = std::fs::read_to_string(log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(content.contains("chestnut"));
        assert!(content.contains("fasterrcnn"));
    }

    #[test]
    fn event_emitter_noop_safe() {
        let emitter = EventEmitter::noop();
        // Should not panic
        emitter.emit(AgentEvent::Error {
            message: "test".to_string(),
            recoverable: true,
        });
    }

    // ═══ HASHLINE INTEGRATION ═══

    #[test]
    fn hashline_read_edit_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.yaml");
        std::fs::write(&path, "data:\n  train: ./train\n  val: ./val\n  nc: 6\n").unwrap();

        // Read
        let lines = tcip_tools::hashline::hashline_read(&path, 1, 4).unwrap();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[3].content, "  nc: 6");

        // Edit nc from 6 to 3
        let nc_hash = lines[3].hash.clone();
        tcip_tools::hashline::hashline_edit(
            &path,
            &[tcip_tools::hashline::HashLineEdit {
                line: 4,
                hash: nc_hash,
                new_content: "  nc: 3".to_string(),
            }],
        ).unwrap();

        // Verify
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("nc: 3"));
        assert!(!content.contains("nc: 6"));
    }

    #[test]
    fn hashline_rejects_stale_edit() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("data.txt");
        std::fs::write(&path, "line1\nline2\n").unwrap();

        let result = tcip_tools::hashline::hashline_edit(
            &path,
            &[tcip_tools::hashline::HashLineEdit {
                line: 1,
                hash: "0000".to_string(),
                new_content: "modified".to_string(),
            }],
        );

        assert!(result.is_err());
        // File unchanged
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("line1"));
    }
}

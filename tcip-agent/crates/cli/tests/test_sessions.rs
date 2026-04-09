#[cfg(test)]
mod tests {
    use tcip_runtime::session::{ContentBlock, ConversationMessage, MessageRole, Session};

    #[test]
    fn test_new_session() {
        let session = Session::new("test-123".to_string());
        assert_eq!(session.id(), "test-123");
        assert_eq!(session.message_count(), 0);
    }

    #[test]
    fn test_push_user_text() {
        let mut session = Session::new("test".to_string());
        session.push_user_text("Hello").unwrap();
        assert_eq!(session.message_count(), 1);
        assert_eq!(session.messages[0].role, MessageRole::User);
    }

    #[test]
    fn test_session_persistence_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");

        // Create and persist
        {
            let mut session = Session::new("persist-test".to_string());
            session.enable_persistence(path.clone()).unwrap();
            session.push_user_text("Hello agent").unwrap();
            session
                .push_message(ConversationMessage {
                    role: MessageRole::Assistant,
                    blocks: vec![ContentBlock::Text {
                        text: "Hi there!".to_string(),
                    }],
                    usage: None,
                    timestamp: chrono::Utc::now(),
                })
                .unwrap();
            assert_eq!(session.message_count(), 2);
        }

        // Reload
        let loaded = Session::load(path).unwrap();
        assert_eq!(loaded.id(), "persist-test");
        assert_eq!(loaded.message_count(), 2);
        assert_eq!(loaded.messages[0].role, MessageRole::User);
        assert_eq!(loaded.messages[1].role, MessageRole::Assistant);
    }

    #[test]
    fn test_to_api_messages() {
        let mut session = Session::new("api-test".to_string());
        session.push_user_text("What crops?").unwrap();
        session
            .push_message(ConversationMessage {
                role: MessageRole::Assistant,
                blocks: vec![ContentBlock::Text {
                    text: "Six crops.".to_string(),
                }],
                usage: None,
                timestamp: chrono::Utc::now(),
            })
            .unwrap();

        let api_msgs = session.to_api_messages();
        assert_eq!(api_msgs.len(), 2);
        assert_eq!(api_msgs[0].role, tcip_api::MessageRole::User);
        assert_eq!(api_msgs[1].role, tcip_api::MessageRole::Assistant);
    }

    #[test]
    fn test_session_metadata() {
        let mut session = Session::new("meta-test".to_string());
        session.meta.crop = Some("hazelnut".to_string());
        session.meta.traits = vec!["catkin_05per_date".to_string()];
        session.meta.pipeline_stage = Some("annotation".to_string());

        assert_eq!(session.meta.crop.as_deref(), Some("hazelnut"));
        assert_eq!(session.meta.traits.len(), 1);
    }
}

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

/// A conversation message (user, assistant, or tool result).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: MessageRole,
    pub blocks: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsageRecord>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        tool_name: String,
        output: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsageRecord {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Session metadata (persisted as first JSONL record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub crop: Option<String>,
    pub traits: Vec<String>,
    pub pipeline_stage: Option<String>,
    /// Parent session if this was forked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    /// Branch name for forked sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_name: Option<String>,
}

/// JSONL record types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "record_type")]
enum SessionRecord {
    #[serde(rename = "meta")]
    Meta(SessionMeta),
    #[serde(rename = "message")]
    Message(ConversationMessage),
}

/// Session: conversation state with JSONL persistence.
pub struct Session {
    pub meta: SessionMeta,
    pub messages: Vec<ConversationMessage>,
    persistence_path: Option<PathBuf>,
}

impl Session {
    /// Create a new session.
    pub fn new(session_id: String) -> Self {
        Self {
            meta: SessionMeta {
                session_id,
                created_at: Utc::now(),
                crop: None,
                traits: Vec::new(),
                pipeline_stage: None,
                parent_session_id: None,
                branch_name: None,
            },
            messages: Vec::new(),
            persistence_path: None,
        }
    }

    /// Fork this session into a new branch.
    ///
    /// Creates a new session with all current messages copied,
    /// linked to this session as its parent.
    pub fn fork(&self, branch_name: &str) -> Self {
        let new_id = uuid::Uuid::new_v4().to_string();
        Self {
            meta: SessionMeta {
                session_id: new_id,
                created_at: Utc::now(),
                crop: self.meta.crop.clone(),
                traits: self.meta.traits.clone(),
                pipeline_stage: self.meta.pipeline_stage.clone(),
                parent_session_id: Some(self.meta.session_id.clone()),
                branch_name: Some(branch_name.to_string()),
            },
            messages: self.messages.clone(),
            persistence_path: None,
        }
    }

    /// Enable JSONL persistence to the given path.
    pub fn enable_persistence(&mut self, path: PathBuf) -> Result<(), SessionError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(SessionError::Io)?;
        }

        // Write session meta as first record
        let record = SessionRecord::Meta(self.meta.clone());
        let mut line = serde_json::to_string(&record).map_err(SessionError::Serialize)?;
        line.push('\n');

        fs::write(&path, &line).map_err(SessionError::Io)?;
        self.persistence_path = Some(path);
        Ok(())
    }

    /// Load a session from a JSONL file.
    pub fn load(path: PathBuf) -> Result<Self, SessionError> {
        let content = fs::read_to_string(&path).map_err(SessionError::Io)?;
        let mut meta: Option<SessionMeta> = None;
        let mut messages = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let record: SessionRecord =
                serde_json::from_str(line).map_err(SessionError::Deserialize)?;
            match record {
                SessionRecord::Meta(m) => meta = Some(m),
                SessionRecord::Message(msg) => messages.push(msg),
            }
        }

        let meta = meta.ok_or_else(|| {
            SessionError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "session file missing meta record",
            ))
        })?;

        Ok(Self {
            meta,
            messages,
            persistence_path: Some(path),
        })
    }

    /// Push a message and persist it.
    pub fn push_message(&mut self, message: ConversationMessage) -> Result<(), SessionError> {
        if let Some(path) = &self.persistence_path {
            let record = SessionRecord::Message(message.clone());
            let mut line = serde_json::to_string(&record).map_err(SessionError::Serialize)?;
            line.push('\n');

            let mut file = OpenOptions::new()
                .append(true)
                .open(path)
                .map_err(SessionError::Io)?;
            file.write_all(line.as_bytes()).map_err(SessionError::Io)?;
        }

        self.messages.push(message);
        Ok(())
    }

    /// Push a user text message.
    pub fn push_user_text(&mut self, text: &str) -> Result<(), SessionError> {
        self.push_message(ConversationMessage {
            role: MessageRole::User,
            blocks: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            usage: None,
            timestamp: Utc::now(),
        })
    }

    /// Convert session messages to API format.
    #[must_use]
    pub fn to_api_messages(&self) -> Vec<tcip_api::InputMessage> {
        self.messages
            .iter()
            .filter_map(|msg| {
                let role = match msg.role {
                    MessageRole::User => tcip_api::MessageRole::User,
                    MessageRole::Assistant => tcip_api::MessageRole::Assistant,
                    MessageRole::Tool => tcip_api::MessageRole::User, // Tool results go as user messages
                };

                let content: Vec<tcip_api::ContentBlock> = msg
                    .blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => tcip_api::ContentBlock::Text {
                            text: text.clone(),
                        },
                        ContentBlock::ToolUse { id, name, input } => {
                            tcip_api::ContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: input.clone(),
                            }
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                            ..
                        } => tcip_api::ContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: output.clone(),
                            is_error: Some(*is_error),
                        },
                    })
                    .collect();

                if content.is_empty() {
                    None
                } else {
                    Some(tcip_api::InputMessage { role, content })
                }
            })
            .collect()
    }

    /// Get the session ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.meta.session_id
    }

    /// Get the parent session ID (if this is a forked session).
    #[must_use]
    pub fn parent_session_id(&self) -> Option<&str> {
        self.meta.parent_session_id.as_deref()
    }

    /// Get the branch name (if this is a forked session).
    #[must_use]
    pub fn branch_name(&self) -> Option<&str> {
        self.meta.branch_name.as_deref()
    }

    /// Get the message count.
    #[must_use]
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Get a slice of all messages.
    #[must_use]
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// Replace the entire message history (used by compaction).
    pub fn replace_messages(&mut self, messages: Vec<ConversationMessage>) {
        self.messages = messages;
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(serde_json::Error),
    #[error("deserialization error: {0}")]
    Deserialize(serde_json::Error),
}

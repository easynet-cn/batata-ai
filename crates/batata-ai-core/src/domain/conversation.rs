use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub tenant_id: String,
    pub title: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub id: String,
    pub conversation_id: String,
    pub tenant_id: String,
    pub role: String,
    pub content: String,
    pub model: Option<String>,
    pub usage: Option<serde_json::Value>,
    pub latency_ms: Option<i64>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
}

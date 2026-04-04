use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptDefinition {
    pub id: String,
    pub tenant_id: Option<String>,
    pub name: String,
    pub description: String,
    pub template: String,
    pub variables: serde_json::Value,
    pub category: Option<String>,
    pub version: i32,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVersion {
    pub id: String,
    pub prompt_id: String,
    pub version: i32,
    pub template: String,
    pub variables: serde_json::Value,
    pub description: String,
    pub change_message: Option<String>,
    pub changed_by: Option<String>,
    pub created_at: NaiveDateTime,
}

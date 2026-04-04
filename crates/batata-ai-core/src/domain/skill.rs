use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub parameters_schema: serde_json::Value,
    pub skill_type: String,
    pub config: Option<serde_json::Value>,
    pub enabled: bool,
    pub version: i32,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillVersion {
    pub id: String,
    pub skill_id: String,
    pub version: i32,
    pub description: String,
    pub parameters_schema: serde_json::Value,
    pub config: Option<serde_json::Value>,
    pub change_message: Option<String>,
    pub changed_by: Option<String>,
    pub created_at: NaiveDateTime,
}

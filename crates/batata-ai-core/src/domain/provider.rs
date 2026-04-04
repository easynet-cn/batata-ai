use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDefinition {
    pub id: String,
    pub name: String,
    pub provider_type: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub config: Option<serde_json::Value>,
    pub enabled: bool,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

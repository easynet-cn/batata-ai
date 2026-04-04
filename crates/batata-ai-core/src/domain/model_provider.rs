use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvider {
    pub model_id: String,
    pub provider_id: String,
    pub model_identifier: String,
    pub is_default: bool,
    pub priority: i32,
    pub enabled: bool,
    pub created_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

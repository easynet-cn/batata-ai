use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    pub key_hash: String,
    pub key_prefix: String,
    pub scopes: serde_json::Value,
    pub rate_limit: Option<i32>,
    pub expires_at: Option<NaiveDateTime>,
    pub enabled: bool,
    pub last_used_at: Option<NaiveDateTime>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

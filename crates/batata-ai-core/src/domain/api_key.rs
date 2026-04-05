use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub tenant_id: String,
    pub name: String,
    /// SHA-256 hash of the Bearer token (`sk-...`).
    pub key_hash: String,
    /// Display prefix of the Bearer token (first 10 chars).
    pub key_prefix: String,
    /// Public app key for dual-key auth (`bat_...`), unique.
    pub app_key: Option<String>,
    /// SHA-256 hash of the app secret key (`bsk_...`).
    pub app_secret_hash: Option<String>,
    pub scopes: serde_json::Value,
    pub rate_limit: Option<i32>,
    pub expires_at: Option<NaiveDateTime>,
    pub enabled: bool,
    pub last_used_at: Option<NaiveDateTime>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

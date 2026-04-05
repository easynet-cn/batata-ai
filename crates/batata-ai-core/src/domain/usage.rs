use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantUsage {
    pub id: String,
    pub tenant_id: String,
    pub period: String,            // "2026-04" (monthly) or "2026-04-04" (daily)
    pub total_requests: i64,
    pub total_prompt_tokens: i64,
    pub total_completion_tokens: i64,
    pub total_tokens: i64,
    pub estimated_cost: f64,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPolicyDefinition {
    pub id: String,
    pub name: String,
    pub policy_type: String,
    pub config: serde_json::Value,
    pub enabled: bool,
    pub priority: i32,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub deleted_at: Option<NaiveDateTime>,
}

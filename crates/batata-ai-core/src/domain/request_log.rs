use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLog {
    pub id: String,
    pub provider_id: String,
    pub provider_name: String,
    pub model_identifier: String,
    pub routing_policy: String,
    pub status: RequestStatus,
    pub latency_ms: u64,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub estimated_cost: Option<f64>,
    pub error_message: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestStatus {
    Success,
    Failed,
    Timeout,
    RateLimited,
}

impl std::fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::Failed => write!(f, "failed"),
            Self::Timeout => write!(f, "timeout"),
            Self::RateLimited => write!(f, "rate_limited"),
        }
    }
}

impl std::str::FromStr for RequestStatus {
    type Err = crate::error::BatataError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "success" => Ok(Self::Success),
            "failed" => Ok(Self::Failed),
            "timeout" => Ok(Self::Timeout),
            "rate_limited" => Ok(Self::RateLimited),
            other => Err(crate::error::BatataError::Config(format!(
                "unknown request status: {other}"
            ))),
        }
    }
}

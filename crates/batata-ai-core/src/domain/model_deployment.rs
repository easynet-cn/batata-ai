use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

/// Deployment status for a model version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Staged but not yet serving traffic.
    Staged,
    /// Serving a percentage of traffic (canary/A-B testing).
    Canary,
    /// Serving all traffic.
    Active,
    /// Previously active, now replaced.
    Retired,
}

impl std::fmt::Display for DeploymentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Staged => write!(f, "staged"),
            Self::Canary => write!(f, "canary"),
            Self::Active => write!(f, "active"),
            Self::Retired => write!(f, "retired"),
        }
    }
}

impl std::str::FromStr for DeploymentStatus {
    type Err = crate::error::BatataError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "staged" => Ok(Self::Staged),
            "canary" => Ok(Self::Canary),
            "active" => Ok(Self::Active),
            "retired" => Ok(Self::Retired),
            other => Err(crate::error::BatataError::Config(format!(
                "unknown deployment status: {other}"
            ))),
        }
    }
}

/// A versioned deployment of a model, supporting canary/A-B rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeployment {
    pub id: String,
    /// Reference to the model definition.
    pub model_id: String,
    /// Semantic version tag (e.g. "v1.2", "2024-04-01").
    pub version_tag: String,
    /// The provider-side model identifier (e.g. "gpt-4o-2024-08-06").
    pub provider_model_id: String,
    /// Provider that serves this version.
    pub provider_id: String,
    pub status: DeploymentStatus,
    /// Traffic weight (0-100). Used for canary / A-B splits.
    /// Active deployments for the same model_id should sum to 100.
    pub traffic_weight: i32,
    /// Optional tenant scope. NULL = platform-wide.
    pub tenant_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
}

/// Performance snapshot for a model deployment over a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmark {
    pub id: String,
    pub deployment_id: String,
    /// Time window label (e.g. "2026-04-12T00:00").
    pub period: String,
    pub total_requests: i64,
    pub error_count: i64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// p50 latency.
    pub p50_latency_ms: f64,
    /// p99 latency.
    pub p99_latency_ms: f64,
    pub avg_prompt_tokens: f64,
    pub avg_completion_tokens: f64,
    /// Average estimated cost per request.
    pub avg_cost: f64,
    pub created_at: NaiveDateTime,
}

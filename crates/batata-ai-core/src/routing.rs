use std::collections::HashMap;

use async_trait::async_trait;
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::message::ChatRequest;

// ---------------------------------------------------------------------------
// RoutingPriority — caller's preference
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingPriority {
    Cost,
    Performance,
    Quality,
}

impl Default for RoutingPriority {
    fn default() -> Self {
        Self::Quality
    }
}

// ---------------------------------------------------------------------------
// RoutingContext — input to routing decisions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RoutingContext {
    pub request: ChatRequest,
    pub required_model: Option<String>,
    pub required_capabilities: Vec<String>,
    pub priority: RoutingPriority,
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// RouteCandidate — a provider+model pair that can serve the request
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RouteCandidate {
    pub provider_id: String,
    pub provider_name: String,
    pub model_identifier: String,
    pub priority: i32,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// RoutingPolicy — strategy trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait RoutingPolicy: Send + Sync {
    fn name(&self) -> &str;

    /// Select and rank candidates. Returns sorted candidates (best first).
    async fn select(
        &self,
        ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>>;
}

// ---------------------------------------------------------------------------
// ProviderStatus — runtime health of a provider+model pair
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatus {
    pub provider_id: String,
    pub model_identifier: String,
    pub healthy: bool,
    pub latency_p50_ms: u64,
    pub latency_p99_ms: u64,
    pub error_rate: f64,
    pub rate_limit_remaining: Option<u32>,
    pub last_checked: NaiveDateTime,
}

// ---------------------------------------------------------------------------
// StatusStore — trait abstraction for status backend (memory / redis / etc.)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait StatusStore: Send + Sync {
    /// Get status for a specific provider+model pair.
    async fn get(&self, provider_id: &str, model_identifier: &str) -> Result<Option<ProviderStatus>>;

    /// Update status for a provider+model pair with optional TTL.
    async fn set(&self, status: &ProviderStatus, ttl_secs: Option<u64>) -> Result<()>;

    /// Remove status entry.
    async fn remove(&self, provider_id: &str, model_identifier: &str) -> Result<()>;

    /// List all unhealthy provider+model pairs.
    async fn list_unhealthy(&self) -> Result<Vec<ProviderStatus>>;

    /// List all tracked statuses.
    async fn list_all(&self) -> Result<Vec<ProviderStatus>>;
}

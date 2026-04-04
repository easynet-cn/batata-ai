use std::sync::Arc;

use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPolicy, StatusStore};

/// Routes by lowest observed latency (p50).
pub struct LatencyPolicy {
    status_store: Arc<dyn StatusStore>,
}

impl LatencyPolicy {
    pub fn new(status_store: Arc<dyn StatusStore>) -> Self {
        Self { status_store }
    }
}

#[async_trait]
impl RoutingPolicy for LatencyPolicy {
    fn name(&self) -> &str {
        "latency"
    }

    async fn select(
        &self,
        _ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>> {
        let mut scored = Vec::with_capacity(candidates.len());
        for c in candidates {
            let latency = self
                .status_store
                .get(&c.provider_id, &c.model_identifier)
                .await?
                .map(|s| s.latency_p50_ms)
                .unwrap_or(u64::MAX);
            let mut c = c.clone();
            c.score = latency as f64;
            scored.push(c);
        }
        scored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

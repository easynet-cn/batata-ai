use std::sync::Arc;

use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPolicy, StatusStore};

/// Filters out unhealthy candidates, preserves existing order.
pub struct FallbackPolicy {
    status_store: Arc<dyn StatusStore>,
}

impl FallbackPolicy {
    pub fn new(status_store: Arc<dyn StatusStore>) -> Self {
        Self { status_store }
    }
}

#[async_trait]
impl RoutingPolicy for FallbackPolicy {
    fn name(&self) -> &str {
        "fallback"
    }

    async fn select(
        &self,
        _ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>> {
        let mut healthy = Vec::with_capacity(candidates.len());
        for c in candidates {
            let is_healthy = self
                .status_store
                .get(&c.provider_id, &c.model_identifier)
                .await?
                .map(|s| s.healthy)
                .unwrap_or(true); // no status = assume healthy
            if is_healthy {
                healthy.push(c.clone());
            }
        }
        Ok(healthy)
    }
}

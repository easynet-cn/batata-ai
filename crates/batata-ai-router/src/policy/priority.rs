use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPolicy};

/// Routes by static priority from model_providers table.
pub struct PriorityPolicy;

#[async_trait]
impl RoutingPolicy for PriorityPolicy {
    fn name(&self) -> &str {
        "priority"
    }

    async fn select(
        &self,
        _ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by_key(|c| c.priority);
        Ok(sorted)
    }
}

use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPolicy};

/// Routes by lowest cost (uses score field, lower = cheaper).
pub struct CostPolicy;

#[async_trait]
impl RoutingPolicy for CostPolicy {
    fn name(&self) -> &str {
        "cost"
    }

    async fn select(
        &self,
        _ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(sorted)
    }
}

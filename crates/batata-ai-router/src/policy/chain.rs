use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPolicy};

/// Chains multiple policies: each policy filters/sorts the output of the previous.
pub struct ChainPolicy {
    policies: Vec<Box<dyn RoutingPolicy>>,
}

impl ChainPolicy {
    pub fn new(policies: Vec<Box<dyn RoutingPolicy>>) -> Self {
        Self { policies }
    }
}

#[async_trait]
impl RoutingPolicy for ChainPolicy {
    fn name(&self) -> &str {
        "chain"
    }

    async fn select(
        &self,
        ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>> {
        let mut current = candidates.to_vec();
        for policy in &self.policies {
            current = policy.select(ctx, &current).await?;
            if current.is_empty() {
                break;
            }
        }
        Ok(current)
    }
}

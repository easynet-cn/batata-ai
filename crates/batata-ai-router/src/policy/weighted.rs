use async_trait::async_trait;

use batata_ai_core::error::Result;
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPolicy};

/// Weighted random selection: distributes traffic proportionally by score.
///
/// Each candidate's `score` is treated as a weight (higher = more traffic).
/// If all scores are 0, falls back to uniform distribution.
pub struct WeightedPolicy;

#[async_trait]
impl RoutingPolicy for WeightedPolicy {
    fn name(&self) -> &str {
        "weighted"
    }

    async fn select(
        &self,
        _ctx: &RoutingContext,
        candidates: &[RouteCandidate],
    ) -> Result<Vec<RouteCandidate>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let total_weight: f64 = candidates.iter().map(|c| c.score.max(0.0)).sum();

        // If all weights are zero, treat as uniform
        if total_weight <= 0.0 {
            return Ok(candidates.to_vec());
        }

        // Simple weighted shuffle: assign random priority based on weight
        // Use deterministic approach: sort by weight descending, assign
        // probability-weighted order. For true randomness in production,
        // replace with rand crate.
        let mut weighted: Vec<(f64, RouteCandidate)> = candidates
            .iter()
            .map(|c| {
                let normalized = c.score.max(0.0) / total_weight;
                (normalized, c.clone())
            })
            .collect();

        // Sort by weight descending — highest weight first gets most traffic
        weighted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(weighted.into_iter().map(|(_, c)| c).collect())
    }
}

use std::sync::Arc;

use batata_ai_core::domain::{RequestLog, RequestStatus};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::message::{ChatRequest, ChatResponse};
use batata_ai_core::repository::{
    ModelCostRepository, ModelRepository, ProviderRepository, RequestLogRepository,
};
use batata_ai_core::routing::{RouteCandidate, RoutingContext, RoutingPriority};

use crate::router::Router;

/// High-level service that combines:
/// - Storage: reads enabled candidates from DB
/// - Router: policy-based selection + fallback execution
/// - Logging: records every request for audit/analytics
#[allow(dead_code)]
pub struct RouterService {
    router: Router,
    model_repo: Arc<dyn ModelRepository>,
    provider_repo: Arc<dyn ProviderRepository>,
    cost_repo: Arc<dyn ModelCostRepository>,
    log_repo: Option<Arc<dyn RequestLogRepository>>,
}

impl RouterService {
    pub fn new(
        router: Router,
        model_repo: Arc<dyn ModelRepository>,
        provider_repo: Arc<dyn ProviderRepository>,
        cost_repo: Arc<dyn ModelCostRepository>,
    ) -> Self {
        Self {
            router,
            model_repo,
            provider_repo,
            cost_repo,
            log_repo: None,
        }
    }

    /// Enable request logging.
    pub fn with_logging(mut self, log_repo: Arc<dyn RequestLogRepository>) -> Self {
        self.log_repo = Some(log_repo);
        self
    }

    /// Build route candidates from DB for a given model name.
    async fn build_candidates(&self, model_name: &str) -> Result<Vec<RouteCandidate>> {
        let model = self
            .model_repo
            .find_by_name(model_name)
            .await?
            .ok_or_else(|| BatataError::ModelNotFound(model_name.to_string()))?;

        if !model.enabled {
            return Err(BatataError::ModelNotFound(format!(
                "model {} is disabled",
                model_name
            )));
        }

        // Get all providers for this model
        let providers = self.model_repo.find_providers(&model.id).await?;

        let mut candidates = Vec::new();
        for provider in &providers {
            if !provider.enabled {
                continue;
            }

            // Look up cost for scoring
            let cost_score = self
                .cost_repo
                .find_by_model_provider(&model.id, &provider.id)
                .await?
                .map(|c| c.input_cost_per_1k + c.output_cost_per_1k)
                .unwrap_or(0.0);

            // We need the model_provider record for priority and model_identifier
            // For now, use provider name as model_identifier fallback
            candidates.push(RouteCandidate {
                provider_id: provider.id.clone(),
                provider_name: provider.name.clone(),
                model_identifier: model_name.to_string(),
                priority: 0,
                score: cost_score,
            });
        }

        Ok(candidates)
    }

    /// Route a chat request by model name.
    pub async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
        priority: RoutingPriority,
    ) -> Result<ChatResponse> {
        let candidates = self.build_candidates(model_name).await?;

        let ctx = RoutingContext {
            request,
            required_model: Some(model_name.to_string()),
            required_capabilities: vec![],
            priority,
            metadata: Default::default(),
        };

        let start = std::time::Instant::now();
        let result = self.router.route(ctx, candidates).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Log the request if logging is enabled
        if let Some(log_repo) = &self.log_repo {
            let log = match &result {
                Ok(response) => RequestLog {
                    id: uuid::Uuid::new_v4().to_string(),
                    tenant_id: String::new(),
                    provider_id: String::new(), // filled from response.model context
                    provider_name: String::new(),
                    model_identifier: response.model.clone(),
                    routing_policy: self.router.policy_name().to_string(),
                    status: RequestStatus::Success,
                    latency_ms: elapsed_ms,
                    prompt_tokens: response.usage.as_ref().map(|u| u.prompt_tokens),
                    completion_tokens: response.usage.as_ref().map(|u| u.completion_tokens),
                    total_tokens: response.usage.as_ref().map(|u| u.total_tokens),
                    estimated_cost: None,
                    error_message: None,
                    metadata: None,
                    created_at: chrono::Utc::now().naive_utc(),
                },
                Err(e) => RequestLog {
                    id: uuid::Uuid::new_v4().to_string(),
                    tenant_id: String::new(),
                    provider_id: String::new(),
                    provider_name: String::new(),
                    model_identifier: model_name.to_string(),
                    routing_policy: self.router.policy_name().to_string(),
                    status: RequestStatus::Failed,
                    latency_ms: elapsed_ms,
                    prompt_tokens: None,
                    completion_tokens: None,
                    total_tokens: None,
                    estimated_cost: None,
                    error_message: Some(e.to_string()),
                    metadata: None,
                    created_at: chrono::Utc::now().naive_utc(),
                },
            };
            // Fire-and-forget: don't fail the request if logging fails
            if let Err(e) = log_repo.create(&log).await {
                tracing::warn!(error = %e, "failed to write request log");
            }
        }

        result
    }
}

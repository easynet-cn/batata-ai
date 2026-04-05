use std::sync::Arc;

use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::message::{ChatResponse, ChatStream};
use batata_ai_core::provider::Provider;
use batata_ai_core::routing::{
    ProviderStatus, RouteCandidate, RoutingContext, RoutingPolicy, StatusStore,
};

/// The core router that combines policy-based selection with fallback execution.
pub struct Router {
    policy: Box<dyn RoutingPolicy>,
    providers: Vec<(String, Arc<dyn Provider>)>, // (provider_id, provider)
    status_store: Arc<dyn StatusStore>,
}

impl Router {
    pub fn new(
        policy: Box<dyn RoutingPolicy>,
        status_store: Arc<dyn StatusStore>,
    ) -> Self {
        Self {
            policy,
            providers: Vec::new(),
            status_store,
        }
    }

    /// Register a provider with its ID.
    pub fn register_provider(&mut self, id: impl Into<String>, provider: Arc<dyn Provider>) {
        self.providers.push((id.into(), provider));
    }

    fn find_provider(&self, provider_id: &str) -> Option<&Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|(id, _)| id == provider_id)
            .map(|(_, p)| p)
    }

    /// List registered providers as (id, name) pairs.
    pub fn list_providers(&self) -> Vec<(String, String)> {
        self.providers
            .iter()
            .map(|(id, p)| (id.clone(), p.name().to_string()))
            .collect()
    }

    /// Get the name of the active routing policy.
    pub fn policy_name(&self) -> &str {
        self.policy.name()
    }

    /// Route a request: select best candidate via policy, execute with fallback.
    pub async fn route(
        &self,
        ctx: RoutingContext,
        candidates: Vec<RouteCandidate>,
    ) -> Result<ChatResponse> {
        let selected = self.policy.select(&ctx, &candidates).await?;

        if selected.is_empty() {
            return Err(BatataError::Provider(
                "no available provider for this request".into(),
            ));
        }

        let mut last_error = None;

        for candidate in &selected {
            let provider = match self.find_provider(&candidate.provider_id) {
                Some(p) => p,
                None => continue,
            };

            let mut req = ctx.request.clone();
            req.model = Some(candidate.model_identifier.clone());

            let start = std::time::Instant::now();
            match provider.chat(req).await {
                Ok(response) => {
                    // Update healthy status
                    let elapsed = start.elapsed().as_millis() as u64;
                    let status = ProviderStatus {
                        provider_id: candidate.provider_id.clone(),
                        model_identifier: candidate.model_identifier.clone(),
                        healthy: true,
                        latency_p50_ms: elapsed,
                        latency_p99_ms: elapsed,
                        error_rate: 0.0,
                        rate_limit_remaining: None,
                        last_checked: chrono::Utc::now().naive_utc(),
                    };
                    let _ = self.status_store.set(&status, Some(300)).await;
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %candidate.provider_name,
                        model = %candidate.model_identifier,
                        error = %e,
                        "provider failed, trying next candidate"
                    );
                    // Mark unhealthy
                    let status = ProviderStatus {
                        provider_id: candidate.provider_id.clone(),
                        model_identifier: candidate.model_identifier.clone(),
                        healthy: false,
                        latency_p50_ms: 0,
                        latency_p99_ms: 0,
                        error_rate: 1.0,
                        rate_limit_remaining: None,
                        last_checked: chrono::Utc::now().naive_utc(),
                    };
                    let _ = self.status_store.set(&status, Some(60)).await;
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            BatataError::Provider("all providers failed".into())
        }))
    }

    /// Route a streaming request with fallback.
    pub async fn route_stream(
        &self,
        ctx: RoutingContext,
        candidates: Vec<RouteCandidate>,
    ) -> Result<ChatStream> {
        let selected = self.policy.select(&ctx, &candidates).await?;

        if selected.is_empty() {
            return Err(BatataError::Provider(
                "no available provider for this request".into(),
            ));
        }

        let mut last_error = None;

        for candidate in &selected {
            let provider = match self.find_provider(&candidate.provider_id) {
                Some(p) => p,
                None => continue,
            };

            let mut req = ctx.request.clone();
            req.model = Some(candidate.model_identifier.clone());

            match provider.stream_chat(req).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    tracing::warn!(
                        provider = %candidate.provider_name,
                        model = %candidate.model_identifier,
                        error = %e,
                        "provider stream failed, trying next candidate"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            BatataError::Provider("all providers failed".into())
        }))
    }
}

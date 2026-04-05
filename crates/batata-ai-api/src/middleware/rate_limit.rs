use std::collections::HashMap;
use std::future::{ready, Ready};
use std::sync::Arc;
use std::time::{Duration, Instant};

use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::error::ErrorTooManyRequests;
use actix_web::{web, Error, HttpMessage};
use futures::future::LocalBoxFuture;
use tokio::sync::Mutex;

use super::auth::AuthContext;

/// Per-key rate limit state using a token bucket algorithm.
struct RateLimitState {
    tokens: i32,
    last_refill: Instant,
}

/// Shared rate limiter that can be registered as app_data.
#[derive(Clone)]
pub struct RateLimiter {
    states: Arc<Mutex<HashMap<String, RateLimitState>>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            states: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Check whether the given key is allowed to proceed.
    ///
    /// `rate_limit` is the maximum number of tokens (requests) per minute.
    /// Returns `true` if the request is allowed, `false` if rate-limited.
    pub async fn check(&self, key: &str, rate_limit: i32) -> bool {
        let mut states = self.states.lock().await;
        let now = Instant::now();

        let state = states.entry(key.to_string()).or_insert(RateLimitState {
            tokens: rate_limit,
            last_refill: now,
        });

        // Refill tokens based on elapsed time (1 token per 60/rate_limit seconds)
        let elapsed = now.duration_since(state.last_refill);
        if elapsed >= Duration::from_secs(60) {
            // Full refill every minute
            state.tokens = rate_limit;
            state.last_refill = now;
        } else if rate_limit > 0 {
            let refill_interval = Duration::from_secs(60) / rate_limit as u32;
            let refill_count = (elapsed.as_millis() / refill_interval.as_millis()) as i32;
            if refill_count > 0 {
                state.tokens = (state.tokens + refill_count).min(rate_limit);
                state.last_refill += refill_interval * refill_count as u32;
            }
        }

        if state.tokens > 0 {
            state.tokens -= 1;
            true
        } else {
            false
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

/// Rate limiting middleware factory.
pub struct RateLimit;

impl<S, B> Transform<S, ServiceRequest> for RateLimit
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RateLimitMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RateLimitMiddleware { service }))
    }
}

pub struct RateLimitMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for RateLimitMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        // Skip rate limiting for health check
        if req.path() == "/health" {
            let fut = self.service.call(req);
            return Box::pin(async move { fut.await });
        }

        // Extract AuthContext and RateLimiter from request
        let auth_ctx = req.extensions().get::<AuthContext>().cloned();
        let limiter = req
            .app_data::<web::Data<RateLimiter>>()
            .cloned();

        let fut = self.service.call(req);

        Box::pin(async move {
            // If we have both an AuthContext and a RateLimiter, enforce rate limits
            if let (Some(ctx), Some(limiter)) = (auth_ctx, limiter) {
                // Use rate_limit from the AuthContext; 0 means unlimited
                if ctx.rate_limit > 0 {
                    let allowed = limiter.check(&ctx.api_key_id, ctx.rate_limit).await;
                    if !allowed {
                        return Err(ErrorTooManyRequests("rate limit exceeded"));
                    }
                }
            }

            fut.await
        })
    }
}

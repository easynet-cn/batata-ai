use std::cell::RefCell;
use std::future::{ready, Ready};
use std::rc::Rc;
use std::sync::Arc;

use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::error::ErrorUnauthorized;
use actix_web::{web, Error, HttpMessage};
use futures::future::LocalBoxFuture;
use sha2::{Digest, Sha256};

use batata_ai_core::domain::ApiKey;
use batata_ai_core::repository::ApiKeyRepository;

/// Authenticated context, injected into request extensions.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub api_key_id: String,
    pub scopes: serde_json::Value,
    /// Requests per minute; 0 means unlimited.
    pub rate_limit: i32,
}

/// API Key authentication middleware factory.
pub struct ApiKeyAuth;

impl<S, B> Transform<S, ServiceRequest> for ApiKeyAuth
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = ApiKeyAuthMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(ApiKeyAuthMiddleware {
            service: Rc::new(RefCell::new(service)),
        }))
    }
}

pub struct ApiKeyAuthMiddleware<S> {
    service: Rc<RefCell<S>>,
}

impl<S, B> Service<ServiceRequest> for ApiKeyAuthMiddleware<S>
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
        // Skip auth for health check
        if req.path() == "/health" {
            let fut = self.service.borrow_mut().call(req);
            return Box::pin(async move { fut.await });
        }

        let api_key_repo = req
            .app_data::<web::Data<Arc<dyn ApiKeyRepository>>>()
            .cloned();

        let auth_header = req
            .headers()
            .get("Authorization")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Extract token before async block
        let token = auth_header
            .as_ref()
            .and_then(|h| h.strip_prefix("Bearer ").map(|s| s.to_string()));

        let svc = self.service.clone();

        Box::pin(async move {
            let token = token.ok_or_else(|| {
                ErrorUnauthorized("missing or invalid Authorization header")
            })?;

            let repo = api_key_repo
                .ok_or_else(|| ErrorUnauthorized("internal: api key repo not configured"))?;

            // Hash the provided key
            let key_hash = hash_api_key(&token);

            let api_key: ApiKey = repo
                .find_by_key_hash(&key_hash)
                .await
                .map_err(|_| ErrorUnauthorized("internal error"))?
                .ok_or_else(|| ErrorUnauthorized("invalid API key"))?;

            // Check if key is enabled
            if !api_key.enabled {
                return Err(ErrorUnauthorized("API key is disabled"));
            }

            // Check expiration
            if let Some(expires_at) = api_key.expires_at {
                if chrono::Utc::now().naive_utc() > expires_at {
                    return Err(ErrorUnauthorized("API key has expired"));
                }
            }

            // Update last_used_at (fire-and-forget)
            let _ = repo.touch_last_used(&api_key.id).await;

            // Insert AuthContext into request extensions so downstream
            // middleware (CasbinAuthz, RateLimit) and handlers can access it.
            let auth_ctx = AuthContext {
                tenant_id: api_key.tenant_id,
                user_id: None, // TODO: resolve from api_key.user_id once column is populated
                api_key_id: api_key.id,
                scopes: api_key.scopes,
                rate_limit: api_key.rate_limit.unwrap_or(0),
            };
            req.extensions_mut().insert(auth_ctx);

            svc.borrow_mut().call(req).await
        })
    }
}

/// Hash an API key using SHA-256.
pub fn hash_api_key(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hex::encode(hasher.finalize())
}

/// Generate an API key with a prefix.
pub fn generate_api_key() -> (String, String) {
    let key = format!("sk-{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let prefix = key[..10].to_string();
    (key, prefix)
}

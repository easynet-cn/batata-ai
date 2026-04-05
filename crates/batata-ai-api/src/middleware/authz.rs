use std::future::{ready, Ready};
use std::sync::Arc;

use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::error::ErrorForbidden;
use actix_web::{web, Error, HttpMessage};
use casbin::CoreApi;
use futures::future::LocalBoxFuture;
use tokio::sync::RwLock;

use crate::middleware::auth::AuthContext;

/// Casbin-based authorization middleware factory.
pub struct CasbinAuthz;

impl<S, B> Transform<S, ServiceRequest> for CasbinAuthz
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = CasbinAuthzMiddleware<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(CasbinAuthzMiddleware { service }))
    }
}

pub struct CasbinAuthzMiddleware<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for CasbinAuthzMiddleware<S>
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
        // Skip authz for health check
        if req.path() == "/health" {
            let fut = self.service.call(req);
            return Box::pin(async move { fut.await });
        }

        let enforcer = req
            .app_data::<web::Data<Arc<RwLock<casbin::Enforcer>>>>()
            .cloned();

        let auth_ctx = req
            .extensions()
            .get::<AuthContext>()
            .cloned();

        let path = req.path().to_string();
        let method = req.method().clone();

        let fut = self.service.call(req);

        Box::pin(async move {
            // If no auth context, skip authz (let auth middleware handle rejection)
            let Some(auth) = auth_ctx else {
                return fut.await;
            };

            // If no enforcer configured, skip authz
            let Some(enforcer_data) = enforcer else {
                return fut.await;
            };

            let action = match method {
                actix_web::http::Method::GET | actix_web::http::Method::HEAD => "read",
                actix_web::http::Method::POST => "write",
                actix_web::http::Method::PUT | actix_web::http::Method::PATCH => "write",
                actix_web::http::Method::DELETE => "delete",
                _ => "read",
            };

            // Use user_id if available, otherwise use api_key_id as subject
            let subject = auth.user_id.as_deref().unwrap_or(&auth.api_key_id);

            let enforcer = enforcer_data.read().await;
            let allowed = enforcer
                .enforce((subject, &auth.tenant_id, path.as_str(), action))
                .unwrap_or(false);

            if !allowed {
                return Err(ErrorForbidden("permission denied"));
            }

            fut.await
        })
    }
}

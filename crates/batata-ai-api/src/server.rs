use std::sync::Arc;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use tokio::sync::RwLock;

use batata_ai_core::repository::ApiKeyRepository;

use crate::handler;
use crate::middleware::{ApiKeyAuth, CasbinAuthz, RateLimit, RateLimiter, RequestTracing};
use crate::state::AppState;

/// Start the HTTP API server.
pub async fn start(state: AppState, bind: &str) -> std::io::Result<()> {
    // Extract shared resources that middleware needs as separate app_data.
    let api_key_repo: web::Data<Arc<dyn ApiKeyRepository>> =
        web::Data::new(state.api_key_repo.clone());
    let enforcer: web::Data<Arc<RwLock<casbin::Enforcer>>> =
        web::Data::new(state.enforcer.clone());
    let rate_limiter: web::Data<RateLimiter> = web::Data::new(state.rate_limiter.clone());

    let state = web::Data::new(state);

    tracing::info!("starting API server on {}", bind);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(api_key_repo.clone())
            .app_data(enforcer.clone())
            .app_data(rate_limiter.clone())
            // Middleware — actix-web wraps last-added outermost.
            // Execution order: CORS → Tracing → ApiKeyAuth → CasbinAuthz → RateLimit → Handler
            .wrap(RateLimit)
            .wrap(CasbinAuthz)
            .wrap(ApiKeyAuth)
            .wrap(RequestTracing)
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
                    .max_age(3600),
            )
            // Health
            .service(handler::health::health_check)
            // Models
            .service(handler::model::list_models)
            // Chat
            .service(handler::chat::chat_completions)
            // Usage
            .service(handler::usage::get_usage)
            // Conversations
            .service(handler::conversation::create_conversation)
            .service(handler::conversation::list_conversations)
            .service(handler::conversation::get_conversation)
            .service(handler::conversation::delete_conversation)
            .service(handler::conversation::list_messages)
            // Admin
            .service(handler::admin::tenant::create_tenant)
            .service(handler::admin::tenant::list_tenants)
            .service(handler::admin::tenant::get_tenant)
            .service(handler::admin::tenant::update_tenant)
            .service(handler::admin::tenant::delete_tenant)
            .service(handler::admin::api_key::create_api_key)
            .service(handler::admin::api_key::list_api_keys)
            .service(handler::admin::api_key::delete_api_key)
            .service(handler::admin::provider::create_provider)
            .service(handler::admin::provider::list_providers)
            .service(handler::admin::provider::get_provider)
            .service(handler::admin::provider::update_provider)
            .service(handler::admin::provider::delete_provider)
            .service(handler::admin::user::create_user)
            .service(handler::admin::user::list_users)
            .service(handler::admin::user::get_user)
            .service(handler::admin::user::update_user)
            .service(handler::admin::user::change_password)
            .service(handler::admin::user::delete_user)
            .service(handler::admin::role::assign_role)
            .service(handler::admin::role::get_roles)
            .service(handler::admin::role::remove_role)
            .service(handler::admin::model::create_model)
            .service(handler::admin::model::list_models_admin)
            .service(handler::admin::model::update_model)
            .service(handler::admin::model::delete_model)
    })
    .bind(bind)?
    .run()
    .await
}

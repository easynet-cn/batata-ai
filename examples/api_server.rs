//! Example: Start the batata-ai API server with SQLite.
//!
//! ```bash
//! cargo run --example api_server
//! ```
//!
//! Then:
//! ```bash
//! # Health check
//! curl http://localhost:8080/health
//!
//! # List models
//! curl http://localhost:8080/v1/models
//! ```

use std::sync::Arc;

use casbin::CoreApi;
use tokio::sync::RwLock;

use batata_ai_core::cache::DefaultCacheKeyStrategy;
use batata_ai_core::crypto::Encryptor;
use batata_ai_router::{InMemoryCache, InMemoryStatusStore, PriorityPolicy, Router};
use batata_ai_storage::{
    connect_and_migrate, SeaOrmApiKeyRepository, SeaOrmConversationMessageRepository,
    SeaOrmConversationRepository, SeaOrmModelRepository, SeaOrmPromptRepository,
    SeaOrmProviderRepository, SeaOrmRequestLogRepository, SeaOrmTenantRepository,
    SeaOrmUserRepository,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,batata_ai=debug".into()),
        )
        .init();

    tracing::info!("starting batata-ai API server...");

    // Connect to SQLite and run migrations
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "sqlite://batata-ai.db?mode=rwc".to_string());
    let db = connect_and_migrate(&db_url).await?;

    // Initialize encryptor for sensitive fields
    let encryptor = Encryptor::from_env()?;

    // Initialize router with priority policy
    let status_store = Arc::new(InMemoryStatusStore::new());
    let router = Router::new(Box::new(PriorityPolicy), status_store);

    // Initialize Casbin enforcer
    let casbin_model = casbin::DefaultModel::from_str(include_str!("../config/rbac_model.conf")).await?;
    let casbin_adapter = sea_orm_adapter::SeaOrmAdapter::new(db.clone()).await?;
    let enforcer = casbin::Enforcer::new(casbin_model, casbin_adapter).await?;
    let enforcer = Arc::new(RwLock::new(enforcer));

    // Initialize cache
    let cache = Arc::new(InMemoryCache::new());

    // Build AppState
    let state = batata_ai_api::state::AppState {
        db: db.clone(),
        router,
        enforcer,
        rate_limiter: batata_ai_api::middleware::RateLimiter::new(),
        tenant_repo: Arc::new(SeaOrmTenantRepository::new(db.clone())),
        user_repo: Arc::new(SeaOrmUserRepository::new(db.clone())),
        api_key_repo: Arc::new(SeaOrmApiKeyRepository::new(db.clone())),
        model_repo: Arc::new(SeaOrmModelRepository::new(db.clone())),
        provider_repo: Arc::new(SeaOrmProviderRepository::new(db.clone(), encryptor)),
        prompt_repo: Arc::new(SeaOrmPromptRepository::new(db.clone())),
        conversation_repo: Arc::new(SeaOrmConversationRepository::new(db.clone())),
        message_repo: Arc::new(SeaOrmConversationMessageRepository::new(db.clone())),
        log_repo: Arc::new(SeaOrmRequestLogRepository::new(db.clone())),
        cache: Some(cache),
        cache_key_strategy: Arc::new(DefaultCacheKeyStrategy),
    };

    // Start server
    let bind = std::env::var("BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    batata_ai_api::server::start(state, &bind).await?;

    Ok(())
}

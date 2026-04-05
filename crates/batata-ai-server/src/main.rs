use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use tokio::sync::RwLock;

use casbin::CoreApi;

use batata_ai_api::middleware::RateLimiter;
use batata_ai_api::provider_factory;
use batata_ai_core::cache::DefaultCacheKeyStrategy;
use batata_ai_core::repository::ProviderRepository;
use batata_ai_router::{InMemoryCache, InMemoryStatusStore, PriorityPolicy, Router};
use batata_ai_storage::{
    connect_and_migrate, SeaOrmApiKeyRepository, SeaOrmConversationMessageRepository,
    SeaOrmConversationRepository, SeaOrmModelRepository, SeaOrmPromptRepository,
    SeaOrmProviderRepository, SeaOrmRequestLogRepository, SeaOrmTenantRepository,
    SeaOrmUserRepository,
};

#[derive(Parser, Debug)]
#[command(name = "batata-ai-server", about = "batata-ai API server")]
struct Args {
    /// Server bind address
    #[arg(long, env = "BIND", default_value = "0.0.0.0:8080")]
    bind: String,

    /// Database URL (SQLite/MySQL/PostgreSQL)
    #[arg(long, env = "DATABASE_URL", default_value = "sqlite://batata-ai.db?mode=rwc")]
    database_url: String,

    /// Log level
    #[arg(long, env = "RUST_LOG", default_value = "info,batata_ai=debug")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .init();

    tracing::info!("batata-ai-server v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("bind: {}", args.bind);
    tracing::info!("database: {}", args.database_url);

    // Connect to database and run migrations
    let db = connect_and_migrate(&args.database_url).await?;

    // Initialize router with priority policy
    let status_store = Arc::new(InMemoryStatusStore::new());
    let mut router = Router::new(Box::new(PriorityPolicy), status_store);

    // Load enabled providers from database and register with router
    let provider_repo = SeaOrmProviderRepository::new(db.clone());
    match provider_repo.find_enabled().await {
        Ok(providers) => {
            let mut ok_count = 0u32;
            let mut err_count = 0u32;
            for def in &providers {
                match provider_factory::create_provider(def) {
                    Ok(provider) => {
                        tracing::info!(
                            id = %def.id,
                            name = %def.name,
                            provider_type = %def.provider_type,
                            "registered provider"
                        );
                        router.register_provider(&def.id, provider);
                        ok_count += 1;
                    }
                    Err(e) => {
                        tracing::warn!(
                            id = %def.id,
                            name = %def.name,
                            provider_type = %def.provider_type,
                            error = %e,
                            "failed to create provider, skipping"
                        );
                        err_count += 1;
                    }
                }
            }
            tracing::info!(
                total = providers.len(),
                ok = ok_count,
                failed = err_count,
                "provider loading complete"
            );
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load providers from database, starting with empty router");
        }
    }

    // Initialize Casbin enforcer
    let casbin_model = casbin::DefaultModel::from_str(include_str!("../../../config/rbac_model.conf")).await?;
    let casbin_adapter = sea_orm_adapter::SeaOrmAdapter::new(db.clone()).await?;
    let mut enforcer = casbin::Enforcer::new(casbin_model, casbin_adapter).await?;
    // Seed default role policies (idempotent — duplicates are ignored by casbin)
    if let Err(e) = batata_ai_api::handler::admin::role::init_default_policies(&mut enforcer).await {
        tracing::warn!(error = %e, "failed to seed default policies (may already exist)");
    }
    let enforcer = Arc::new(RwLock::new(enforcer));
    tracing::info!("casbin enforcer initialized");

    // Initialize cache
    let cache = Arc::new(InMemoryCache::new());

    // Build AppState
    let state = batata_ai_api::state::AppState {
        db: db.clone(),
        router,
        enforcer,
        rate_limiter: RateLimiter::new(),
        tenant_repo: Arc::new(SeaOrmTenantRepository::new(db.clone())),
        user_repo: Arc::new(SeaOrmUserRepository::new(db.clone())),
        api_key_repo: Arc::new(SeaOrmApiKeyRepository::new(db.clone())),
        model_repo: Arc::new(SeaOrmModelRepository::new(db.clone())),
        provider_repo: Arc::new(SeaOrmProviderRepository::new(db.clone())),
        prompt_repo: Arc::new(SeaOrmPromptRepository::new(db.clone())),
        conversation_repo: Arc::new(SeaOrmConversationRepository::new(db.clone())),
        message_repo: Arc::new(SeaOrmConversationMessageRepository::new(db.clone())),
        log_repo: Arc::new(SeaOrmRequestLogRepository::new(db.clone())),
        cache: Some(cache),
        cache_key_strategy: Arc::new(DefaultCacheKeyStrategy),
    };

    // Start server
    batata_ai_api::server::start(state, &args.bind).await?;

    Ok(())
}

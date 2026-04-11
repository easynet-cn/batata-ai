use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use tokio::sync::RwLock;

use casbin::CoreApi;

use std::path::PathBuf;

use batata_ai_api::middleware::RateLimiter;
use batata_ai_api::provider_factory;
use batata_ai_core::cache::DefaultCacheKeyStrategy;
use batata_ai_core::config::{
    BatataConfig, ChunkerConfig, RagConfig, RagDevice, RagEmbedderKind, RagStoreKind,
};
use batata_ai_core::crypto::Encryptor;
use batata_ai_core::repository::ProviderRepository;
use batata_ai_router::{InMemoryCache, InMemoryStatusStore, PriorityPolicy, Router};
use batata_ai_storage::{
    connect_and_migrate, SeaOrmApiKeyRepository, SeaOrmConversationMessageRepository,
    SeaOrmConversationRepository, SeaOrmKbDocumentRepository, SeaOrmKnowledgeBaseRepository,
    SeaOrmModelRepository, SeaOrmPromptRepository, SeaOrmProviderRepository,
    SeaOrmRequestLogRepository, SeaOrmTenantRepository, SeaOrmUserRepository,
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

    /// Optional TOML config file (e.g. config/batata-ai.toml).
    /// Env vars still override individual fields.
    #[arg(long, env = "BATATA_AI_CONFIG")]
    config: Option<PathBuf>,
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

    // Initialize encryptor for sensitive fields
    let encryptor = Encryptor::from_env()?;

    // Initialize router with priority policy
    let status_store = Arc::new(InMemoryStatusStore::new());
    let mut router = Router::new(Box::new(PriorityPolicy), status_store);

    // Load enabled providers from database and register with router
    let provider_repo = SeaOrmProviderRepository::new(db.clone(), encryptor.clone());
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

    // --- Load config file (if any) and apply env overrides ---
    let file_cfg = BatataConfig::load(args.config.as_deref())
        .map_err(|e| anyhow::anyhow!("config load: {e}"))?;
    if args.config.is_some() {
        tracing::info!(path = ?args.config, "loaded config file");
    }

    // --- RAG pipeline (config-driven) ---
    let rag_cfg = rag_config_with_env_overrides(file_cfg.rag.clone());
    let rag_pipeline = if rag_cfg.enabled {
        tracing::info!(embedder = ?rag_cfg.embedder, device = ?rag_cfg.device, "initializing RAG pipeline");
        match batata_ai_rag::build_pipeline(&rag_cfg, Some(&db)) {
            Ok(p) => {
                tracing::info!("RAG pipeline ready");
                p
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to build RAG pipeline; continuing without RAG");
                None
            }
        }
    } else {
        tracing::info!("RAG disabled (set BATATA_AI_RAG_ENABLED=1 to enable)");
        None
    };

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
        provider_repo: Arc::new(SeaOrmProviderRepository::new(db.clone(), encryptor)),
        prompt_repo: Arc::new(SeaOrmPromptRepository::new(db.clone())),
        conversation_repo: Arc::new(SeaOrmConversationRepository::new(db.clone())),
        message_repo: Arc::new(SeaOrmConversationMessageRepository::new(db.clone())),
        log_repo: Arc::new(SeaOrmRequestLogRepository::new(db.clone())),
        cache: Some(cache),
        cache_key_strategy: Arc::new(DefaultCacheKeyStrategy),
        rag_pipeline,
        kb_repo: Some(Arc::new(SeaOrmKnowledgeBaseRepository::new(db.clone()))),
        kb_document_repo: Some(Arc::new(SeaOrmKbDocumentRepository::new(db.clone()))),
        rag_object_store: rag_object_store_from_env(),
        rag_upload_prefix: std::env::var("BATATA_AI_RAG_UPLOAD_PREFIX")
            .unwrap_or_else(|_| "rag-uploads".to_string()),
    };

    // Start server
    batata_ai_api::server::start(state, &args.bind).await?;

    Ok(())
}

/// Apply environment-variable overrides on top of a file-loaded [`RagConfig`].
///
/// Env precedence: an unset variable keeps the file value, a set variable
/// replaces it. This lets users ship a baseline config file and still tweak
/// individual knobs per deployment without a rebuild.
/// Build an optional ObjectStore for RAG uploads from env vars.
/// Currently only a local filesystem backend is wired here; S3/OSS can
/// be added alongside without changing the AppState layout.
fn rag_object_store_from_env() -> Option<Arc<dyn batata_ai_core::object_store::ObjectStore>> {
    if let Ok(path) = std::env::var("BATATA_AI_RAG_UPLOAD_DIR") {
        let store = batata_ai_object_store::LocalFileStore::new(
            std::path::PathBuf::from(path),
            "rag-local".to_string(),
        );
        return Some(Arc::new(store));
    }
    None
}

fn rag_config_with_env_overrides(mut cfg: RagConfig) -> RagConfig {
    if let Some(v) = std::env::var("BATATA_AI_RAG_ENABLED").ok() {
        cfg.enabled = matches!(v.as_str(), "1" | "true" | "yes" | "on");
    }
    if let Some(v) = std::env::var("BATATA_AI_RAG_EMBEDDER").ok() {
        cfg.embedder = match v.as_str() {
            "clip" => RagEmbedderKind::Clip,
            "bge" => RagEmbedderKind::Bge,
            other => {
                tracing::warn!(value = other, "unknown BATATA_AI_RAG_EMBEDDER, keeping existing");
                cfg.embedder
            }
        };
    }
    if let Some(v) = std::env::var("BATATA_AI_RAG_BGE_MODEL").ok() {
        cfg.bge_model = v;
    }
    if let Some(v) = std::env::var("BATATA_AI_RAG_DEVICE").ok() {
        cfg.device = match v.as_str() {
            "gpu" => RagDevice::Gpu,
            "cpu" => RagDevice::Cpu,
            other => {
                tracing::warn!(value = other, "unknown BATATA_AI_RAG_DEVICE, keeping existing");
                cfg.device
            }
        };
    }
    if let Some(v) = std::env::var("BATATA_AI_RAG_STORE").ok() {
        cfg.store = match v.as_str() {
            "in-memory" | "memory" => RagStoreKind::InMemory,
            "database" | "db" => RagStoreKind::Database,
            "hnsw" => RagStoreKind::Hnsw,
            other => {
                tracing::warn!(value = other, "unknown BATATA_AI_RAG_STORE, keeping existing");
                cfg.store
            }
        };
    }
    if let Some(v) = std::env::var("BATATA_AI_RAG_CHUNK_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        cfg.chunker.window = v;
    }
    if let Some(v) = std::env::var("BATATA_AI_RAG_CHUNK_OVERLAP")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        cfg.chunker.overlap = v;
    }
    // Let ChunkerConfig own its invariant — avoid panics in `FixedWindowChunker::new`.
    if cfg.chunker.overlap >= cfg.chunker.window {
        let fallback = ChunkerConfig::default();
        tracing::warn!(
            window = cfg.chunker.window,
            overlap = cfg.chunker.overlap,
            "invalid chunker window/overlap, falling back to defaults"
        );
        cfg.chunker = fallback;
    }
    cfg
}

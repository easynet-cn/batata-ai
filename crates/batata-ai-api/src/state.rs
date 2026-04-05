use std::sync::Arc;

use sea_orm::DatabaseConnection;
use tokio::sync::RwLock;

use batata_ai_core::cache::{CacheKeyStrategy, CacheStore};
use batata_ai_core::repository::{
    ApiKeyRepository, ConversationMessageRepository, ConversationRepository, ModelRepository,
    PromptRepository, ProviderRepository, RequestLogRepository, TenantRepository, UserRepository,
};
use crate::middleware::RateLimiter;
use batata_ai_router::Router;

/// Shared application state injected into all handlers.
pub struct AppState {
    pub db: DatabaseConnection,
    pub router: Router,
    pub enforcer: Arc<RwLock<casbin::Enforcer>>,
    pub rate_limiter: RateLimiter,
    pub tenant_repo: Arc<dyn TenantRepository>,
    pub user_repo: Arc<dyn UserRepository>,
    pub api_key_repo: Arc<dyn ApiKeyRepository>,
    pub model_repo: Arc<dyn ModelRepository>,
    pub provider_repo: Arc<dyn ProviderRepository>,
    pub prompt_repo: Arc<dyn PromptRepository>,
    pub conversation_repo: Arc<dyn ConversationRepository>,
    pub message_repo: Arc<dyn ConversationMessageRepository>,
    pub log_repo: Arc<dyn RequestLogRepository>,
    pub cache: Option<Arc<dyn CacheStore>>,
    pub cache_key_strategy: Arc<dyn CacheKeyStrategy>,
}

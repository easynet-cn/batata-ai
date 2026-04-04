use std::sync::Arc;

use sea_orm::DatabaseConnection;

use batata_ai_core::repository::{
    ApiKeyRepository, ConversationMessageRepository, ConversationRepository, ModelRepository,
    PromptRepository, ProviderRepository, TenantRepository,
};
use batata_ai_router::Router;

/// Shared application state injected into all handlers.
pub struct AppState {
    pub db: DatabaseConnection,
    pub router: Router,
    pub tenant_repo: Arc<dyn TenantRepository>,
    pub api_key_repo: Arc<dyn ApiKeyRepository>,
    pub model_repo: Arc<dyn ModelRepository>,
    pub provider_repo: Arc<dyn ProviderRepository>,
    pub prompt_repo: Arc<dyn PromptRepository>,
    pub conversation_repo: Arc<dyn ConversationRepository>,
    pub message_repo: Arc<dyn ConversationMessageRepository>,
}

use async_trait::async_trait;

use chrono::NaiveDateTime;

use crate::domain::{
    ApiKey, Conversation, ConversationMessage, ModelCost, ModelDefinition, ModelProvider,
    ModelType, ObjectStoreBucket, ObjectStoreConfig, PromptDefinition, PromptVersion,
    ProviderDefinition, RequestLog, RoutingPolicyDefinition, SkillDefinition, SkillVersion,
    StoredObject, Tenant, TenantUsage, User,
};
use crate::error::Result;

// ---------------------------------------------------------------------------
// Generic CRUD
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Repository<T: Send + Sync>: Send + Sync {
    async fn find_by_id(&self, id: &str) -> Result<Option<T>>;
    async fn find_all(&self) -> Result<Vec<T>>;
    async fn create(&self, entity: &T) -> Result<T>;
    async fn update(&self, entity: &T) -> Result<T>;
    /// Soft delete: set deleted_at timestamp.
    async fn delete(&self, id: &str) -> Result<bool>;
    /// Permanently remove from database.
    async fn hard_delete(&self, id: &str) -> Result<bool>;
    /// Restore a soft-deleted record.
    async fn restore(&self, id: &str) -> Result<bool>;
}

// ---------------------------------------------------------------------------
// Tenant
// ---------------------------------------------------------------------------

#[async_trait]
pub trait TenantRepository: Repository<Tenant> {
    async fn find_by_slug(&self, slug: &str) -> Result<Option<Tenant>>;
    async fn find_enabled(&self) -> Result<Vec<Tenant>>;
}

// ---------------------------------------------------------------------------
// User
// ---------------------------------------------------------------------------

#[async_trait]
pub trait UserRepository: Repository<User> {
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<User>>;
    async fn find_by_username(&self, tenant_id: &str, username: &str) -> Result<Option<User>>;
}

// ---------------------------------------------------------------------------
// API Key
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ApiKeyRepository: Repository<ApiKey> {
    /// Look up by key hash (for auth).
    async fn find_by_key_hash(&self, key_hash: &str) -> Result<Option<ApiKey>>;
    /// Look up by key prefix (for display).
    async fn find_by_prefix(&self, prefix: &str) -> Result<Option<ApiKey>>;
    /// Find all keys for a tenant.
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<ApiKey>>;
    /// Update last_used_at timestamp.
    async fn touch_last_used(&self, id: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ModelRepository: Repository<ModelDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<ModelDefinition>>;
    async fn find_by_type(&self, model_type: &ModelType) -> Result<Vec<ModelDefinition>>;
    async fn find_providers(&self, model_id: &str) -> Result<Vec<ProviderDefinition>>;
    async fn add_provider(&self, rel: &ModelProvider) -> Result<()>;
    async fn remove_provider(&self, model_id: &str, provider_id: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ProviderRepository: Repository<ProviderDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<ProviderDefinition>>;
    async fn find_enabled(&self) -> Result<Vec<ProviderDefinition>>;
    async fn find_models(&self, provider_id: &str) -> Result<Vec<ModelDefinition>>;
}

// ---------------------------------------------------------------------------
// Prompt (mixed: platform + tenant)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait PromptRepository: Repository<PromptDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<PromptDefinition>>;
    async fn find_by_category(&self, category: &str) -> Result<Vec<PromptDefinition>>;
    /// Find prompts visible to a tenant (platform + tenant's own).
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<PromptDefinition>>;
    async fn update_with_version(
        &self,
        entity: &PromptDefinition,
        change_message: Option<&str>,
        changed_by: Option<&str>,
    ) -> Result<PromptDefinition>;
    async fn find_versions(&self, prompt_id: &str) -> Result<Vec<PromptVersion>>;
    async fn find_version(
        &self,
        prompt_id: &str,
        version: i32,
    ) -> Result<Option<PromptVersion>>;
    async fn rollback_to_version(&self, prompt_id: &str, version: i32) -> Result<PromptDefinition>;
}

// ---------------------------------------------------------------------------
// Skill (mixed: platform + tenant)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait SkillRepository: Repository<SkillDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<SkillDefinition>>;
    async fn find_enabled(&self) -> Result<Vec<SkillDefinition>>;
    /// Find skills visible to a tenant (platform + tenant's own).
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<SkillDefinition>>;
    async fn update_with_version(
        &self,
        entity: &SkillDefinition,
        change_message: Option<&str>,
        changed_by: Option<&str>,
    ) -> Result<SkillDefinition>;
    async fn find_versions(&self, skill_id: &str) -> Result<Vec<SkillVersion>>;
    async fn find_version(
        &self,
        skill_id: &str,
        version: i32,
    ) -> Result<Option<SkillVersion>>;
    async fn rollback_to_version(&self, skill_id: &str, version: i32) -> Result<SkillDefinition>;
}

// ---------------------------------------------------------------------------
// Routing Policy (mixed: platform + tenant)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait RoutingPolicyRepository: Repository<RoutingPolicyDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<RoutingPolicyDefinition>>;
    async fn find_enabled(&self) -> Result<Vec<RoutingPolicyDefinition>>;
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<RoutingPolicyDefinition>>;
}

// ---------------------------------------------------------------------------
// Model Cost
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ModelCostRepository: Repository<ModelCost> {
    async fn find_by_model_provider(
        &self,
        model_id: &str,
        provider_id: &str,
    ) -> Result<Option<ModelCost>>;
    async fn find_by_model(&self, model_id: &str) -> Result<Vec<ModelCost>>;
}

// ---------------------------------------------------------------------------
// Request Log (tenant-level)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait RequestLogRepository: Send + Sync {
    async fn create(&self, log: &RequestLog) -> Result<RequestLog>;
    async fn find_by_id(&self, id: &str) -> Result<Option<RequestLog>>;
    async fn find_by_time_range(
        &self,
        tenant_id: &str,
        from: NaiveDateTime,
        to: NaiveDateTime,
        limit: u64,
    ) -> Result<Vec<RequestLog>>;
    async fn find_by_provider(
        &self,
        tenant_id: &str,
        provider_id: &str,
        limit: u64,
    ) -> Result<Vec<RequestLog>>;
}

// ---------------------------------------------------------------------------
// Object Store Config (platform-level)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ObjectStoreConfigRepository: Repository<ObjectStoreConfig> {
    async fn find_by_name(&self, name: &str) -> Result<Option<ObjectStoreConfig>>;
    async fn find_enabled(&self) -> Result<Vec<ObjectStoreConfig>>;
}

// ---------------------------------------------------------------------------
// Object Store Bucket (mixed: platform + tenant)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ObjectStoreBucketRepository: Repository<ObjectStoreBucket> {
    async fn find_by_name(&self, name: &str) -> Result<Option<ObjectStoreBucket>>;
    async fn find_by_config(&self, config_id: &str) -> Result<Vec<ObjectStoreBucket>>;
    async fn find_enabled(&self) -> Result<Vec<ObjectStoreBucket>>;
    async fn find_default(&self) -> Result<Option<ObjectStoreBucket>>;
    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<ObjectStoreBucket>>;
}

// ---------------------------------------------------------------------------
// Stored Object (tenant-level)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait StoredObjectRepository: Send + Sync {
    async fn create(&self, obj: &StoredObject) -> Result<StoredObject>;
    async fn find_by_id(&self, id: &str) -> Result<Option<StoredObject>>;
    async fn find_by_key(&self, key: &str) -> Result<Option<StoredObject>>;
    async fn find_by_keys(&self, keys: &[String]) -> Result<Vec<StoredObject>>;
    async fn delete(&self, id: &str) -> Result<bool>;
    async fn find_by_bucket(
        &self,
        bucket_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<StoredObject>>;
    async fn count_by_bucket(&self, bucket_id: &str) -> Result<u64>;
    async fn find_by_content_type(
        &self,
        bucket_id: &str,
        content_type_prefix: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<StoredObject>>;
}

// ---------------------------------------------------------------------------
// Conversation (tenant-level)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ConversationRepository: Repository<Conversation> {
    async fn find_by_tenant(
        &self,
        tenant_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<Conversation>>;
    async fn count_by_tenant(&self, tenant_id: &str) -> Result<u64>;
}

// ---------------------------------------------------------------------------
// Conversation Message (tenant-level, no soft delete)
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ConversationMessageRepository: Send + Sync {
    async fn create(&self, msg: &ConversationMessage) -> Result<ConversationMessage>;
    async fn find_by_id(&self, id: &str) -> Result<Option<ConversationMessage>>;
    async fn find_by_conversation(
        &self,
        conversation_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<ConversationMessage>>;
    async fn count_by_conversation(&self, conversation_id: &str) -> Result<u64>;
}

// ---------------------------------------------------------------------------
// Tenant Usage
// ---------------------------------------------------------------------------

#[async_trait]
pub trait TenantUsageRepository: Send + Sync {
    async fn find_or_create(&self, tenant_id: &str, period: &str) -> Result<TenantUsage>;
    async fn increment(
        &self,
        tenant_id: &str,
        period: &str,
        requests: i64,
        prompt_tokens: i64,
        completion_tokens: i64,
        cost: f64,
    ) -> Result<TenantUsage>;
    async fn find_by_tenant_period(&self, tenant_id: &str, period: &str) -> Result<Option<TenantUsage>>;
    async fn find_by_tenant(&self, tenant_id: &str, limit: u64) -> Result<Vec<TenantUsage>>;
}

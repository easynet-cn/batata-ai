use async_trait::async_trait;

use chrono::NaiveDateTime;

use crate::domain::{
    ModelCost, ModelDefinition, ModelProvider, ModelType, ObjectStoreBucket, ObjectStoreConfig,
    PromptDefinition, PromptVersion, ProviderDefinition, RequestLog, RoutingPolicyDefinition,
    SkillDefinition, SkillVersion, StoredObject,
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
// Prompt
// ---------------------------------------------------------------------------

#[async_trait]
pub trait PromptRepository: Repository<PromptDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<PromptDefinition>>;
    async fn find_by_category(&self, category: &str) -> Result<Vec<PromptDefinition>>;
    /// Update prompt and auto-snapshot the previous version to history.
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
    /// Rollback to a specific version.
    async fn rollback_to_version(&self, prompt_id: &str, version: i32) -> Result<PromptDefinition>;
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

#[async_trait]
pub trait SkillRepository: Repository<SkillDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<SkillDefinition>>;
    async fn find_enabled(&self) -> Result<Vec<SkillDefinition>>;
    /// Update skill and auto-snapshot the previous version to history.
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
// Routing Policy
// ---------------------------------------------------------------------------

#[async_trait]
pub trait RoutingPolicyRepository: Repository<RoutingPolicyDefinition> {
    async fn find_by_name(&self, name: &str) -> Result<Option<RoutingPolicyDefinition>>;
    async fn find_enabled(&self) -> Result<Vec<RoutingPolicyDefinition>>;
}

// ---------------------------------------------------------------------------
// Model Cost
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ModelCostRepository: Repository<ModelCost> {
    /// Find the current cost for a specific model-provider pair.
    async fn find_by_model_provider(
        &self,
        model_id: &str,
        provider_id: &str,
    ) -> Result<Option<ModelCost>>;
    /// Find all costs for a model (across all providers).
    async fn find_by_model(&self, model_id: &str) -> Result<Vec<ModelCost>>;
}

// ---------------------------------------------------------------------------
// Request Log
// ---------------------------------------------------------------------------

#[async_trait]
pub trait RequestLogRepository: Send + Sync {
    async fn create(&self, log: &RequestLog) -> Result<RequestLog>;
    async fn find_by_id(&self, id: &str) -> Result<Option<RequestLog>>;
    /// Query logs within a time range.
    async fn find_by_time_range(
        &self,
        from: NaiveDateTime,
        to: NaiveDateTime,
        limit: u64,
    ) -> Result<Vec<RequestLog>>;
    /// Query logs by provider.
    async fn find_by_provider(
        &self,
        provider_id: &str,
        limit: u64,
    ) -> Result<Vec<RequestLog>>;
}

// ---------------------------------------------------------------------------
// Object Store Config
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ObjectStoreConfigRepository: Repository<ObjectStoreConfig> {
    async fn find_by_name(&self, name: &str) -> Result<Option<ObjectStoreConfig>>;
    async fn find_enabled(&self) -> Result<Vec<ObjectStoreConfig>>;
}

// ---------------------------------------------------------------------------
// Object Store Bucket
// ---------------------------------------------------------------------------

#[async_trait]
pub trait ObjectStoreBucketRepository: Repository<ObjectStoreBucket> {
    async fn find_by_name(&self, name: &str) -> Result<Option<ObjectStoreBucket>>;
    async fn find_by_config(&self, config_id: &str) -> Result<Vec<ObjectStoreBucket>>;
    async fn find_enabled(&self) -> Result<Vec<ObjectStoreBucket>>;
    async fn find_default(&self) -> Result<Option<ObjectStoreBucket>>;
}

// ---------------------------------------------------------------------------
// Stored Object
// ---------------------------------------------------------------------------

#[async_trait]
pub trait StoredObjectRepository: Send + Sync {
    async fn create(&self, obj: &StoredObject) -> Result<StoredObject>;
    async fn find_by_id(&self, id: &str) -> Result<Option<StoredObject>>;
    async fn find_by_key(&self, key: &str) -> Result<Option<StoredObject>>;
    /// Batch fetch by multiple keys.
    async fn find_by_keys(&self, keys: &[String]) -> Result<Vec<StoredObject>>;
    async fn delete(&self, id: &str) -> Result<bool>;
    /// Paginated query by bucket.
    async fn find_by_bucket(
        &self,
        bucket_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<StoredObject>>;
    /// Count objects in a bucket.
    async fn count_by_bucket(&self, bucket_id: &str) -> Result<u64>;
    /// Paginated query by content type prefix (e.g., "image/", "video/").
    async fn find_by_content_type(
        &self,
        bucket_id: &str,
        content_type_prefix: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<StoredObject>>;
}

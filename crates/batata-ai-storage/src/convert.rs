use batata_ai_core::domain::{
    BucketAccessPolicy, ModelCost, ModelDefinition, ModelProvider, ModelType, ObjectStoreBackend,
    ObjectStoreBucket, ObjectStoreConfig, PromptDefinition, PromptVersion, ProviderDefinition,
    RequestLog, RequestStatus, RoutingPolicyDefinition, SkillDefinition, SkillVersion,
    StoredObject,
};

use crate::entity;

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

impl From<entity::provider::Model> for ProviderDefinition {
    fn from(m: entity::provider::Model) -> Self {
        Self {
            id: m.id,
            name: m.name,
            provider_type: m.provider_type,
            api_key: m.api_key,
            base_url: m.base_url,
            config: m.config,
            enabled: m.enabled,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

impl From<entity::model::Model> for ModelDefinition {
    fn from(m: entity::model::Model) -> Self {
        Self {
            id: m.id,
            name: m.name,
            model_type: m.model_type.parse().unwrap_or(ModelType::Chat),
            context_length: m.context_length,
            description: m.description,
            metadata: m.metadata,
            enabled: m.enabled,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelProvider
// ---------------------------------------------------------------------------

impl From<entity::model_provider::Model> for ModelProvider {
    fn from(m: entity::model_provider::Model) -> Self {
        Self {
            model_id: m.model_id,
            provider_id: m.provider_id,
            model_identifier: m.model_identifier,
            is_default: m.is_default,
            priority: m.priority,
            enabled: m.enabled,
            created_at: m.created_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

impl From<entity::prompt::Model> for PromptDefinition {
    fn from(m: entity::prompt::Model) -> Self {
        Self {
            id: m.id,
            name: m.name,
            description: m.description,
            template: m.template,
            variables: m.variables,
            category: m.category,
            version: m.version,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// Skill
// ---------------------------------------------------------------------------

impl From<entity::skill::Model> for SkillDefinition {
    fn from(m: entity::skill::Model) -> Self {
        Self {
            id: m.id,
            name: m.name,
            description: m.description,
            parameters_schema: m.parameters_schema,
            skill_type: m.skill_type,
            config: m.config,
            enabled: m.enabled,
            version: m.version,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// PromptVersion
// ---------------------------------------------------------------------------

impl From<entity::prompt_version::Model> for PromptVersion {
    fn from(m: entity::prompt_version::Model) -> Self {
        Self {
            id: m.id,
            prompt_id: m.prompt_id,
            version: m.version,
            template: m.template,
            variables: m.variables,
            description: m.description,
            change_message: m.change_message,
            changed_by: m.changed_by,
            created_at: m.created_at,
        }
    }
}

// ---------------------------------------------------------------------------
// SkillVersion
// ---------------------------------------------------------------------------

impl From<entity::skill_version::Model> for SkillVersion {
    fn from(m: entity::skill_version::Model) -> Self {
        Self {
            id: m.id,
            skill_id: m.skill_id,
            version: m.version,
            description: m.description,
            parameters_schema: m.parameters_schema,
            config: m.config,
            change_message: m.change_message,
            changed_by: m.changed_by,
            created_at: m.created_at,
        }
    }
}

// ---------------------------------------------------------------------------
// RoutingPolicyDefinition
// ---------------------------------------------------------------------------

impl From<entity::routing_policy::Model> for RoutingPolicyDefinition {
    fn from(m: entity::routing_policy::Model) -> Self {
        Self {
            id: m.id,
            name: m.name,
            policy_type: m.policy_type,
            config: m.config,
            enabled: m.enabled,
            priority: m.priority,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// ModelCost
// ---------------------------------------------------------------------------

impl From<entity::model_cost::Model> for ModelCost {
    fn from(m: entity::model_cost::Model) -> Self {
        Self {
            id: m.id,
            model_id: m.model_id,
            provider_id: m.provider_id,
            input_cost_per_1k: m.input_cost_per_1k,
            output_cost_per_1k: m.output_cost_per_1k,
            currency: m.currency,
            effective_from: m.effective_from,
            created_at: m.created_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// RequestLog
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ObjectStoreConfig
// ---------------------------------------------------------------------------

impl From<entity::object_store_config::Model> for ObjectStoreConfig {
    fn from(m: entity::object_store_config::Model) -> Self {
        Self {
            id: m.id,
            name: m.name,
            backend: m.backend.parse().unwrap_or(ObjectStoreBackend::Local),
            endpoint: m.endpoint,
            region: m.region,
            access_key: m.access_key,
            secret_key: m.secret_key,
            enabled: m.enabled,
            config: m.config,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// ObjectStoreBucket
// ---------------------------------------------------------------------------

impl From<entity::object_store_bucket::Model> for ObjectStoreBucket {
    fn from(m: entity::object_store_bucket::Model) -> Self {
        Self {
            id: m.id,
            config_id: m.config_id,
            name: m.name,
            bucket: m.bucket,
            root_path: m.root_path,
            access_policy: m.access_policy.parse().unwrap_or(BucketAccessPolicy::Private),
            custom_domain: m.custom_domain,
            is_default: m.is_default,
            enabled: m.enabled,
            config: m.config,
            created_at: m.created_at,
            updated_at: m.updated_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// StoredObject
// ---------------------------------------------------------------------------

impl From<entity::stored_object::Model> for StoredObject {
    fn from(m: entity::stored_object::Model) -> Self {
        Self {
            id: m.id,
            bucket_id: m.bucket_id,
            key: m.key,
            original_name: m.original_name,
            content_type: m.content_type,
            size: m.size,
            checksum: m.checksum,
            metadata: m.metadata,
            created_at: m.created_at,
            deleted_at: m.deleted_at,
        }
    }
}

// ---------------------------------------------------------------------------
// RequestLog
// ---------------------------------------------------------------------------

impl From<entity::request_log::Model> for RequestLog {
    fn from(m: entity::request_log::Model) -> Self {
        Self {
            id: m.id,
            provider_id: m.provider_id,
            provider_name: m.provider_name,
            model_identifier: m.model_identifier,
            routing_policy: m.routing_policy,
            status: m.status.parse().unwrap_or(RequestStatus::Failed),
            latency_ms: m.latency_ms as u64,
            prompt_tokens: m.prompt_tokens.map(|v| v as u32),
            completion_tokens: m.completion_tokens.map(|v| v as u32),
            total_tokens: m.total_tokens.map(|v| v as u32),
            estimated_cost: m.estimated_cost,
            error_message: m.error_message,
            metadata: m.metadata,
            created_at: m.created_at,
        }
    }
}

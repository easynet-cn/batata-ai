use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::{ModelDefinition, ModelProvider, ModelType, ProviderDefinition};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ModelRepository, Repository};

use crate::entity::{model, model_provider, provider};

pub struct SeaOrmModelRepository {
    db: DatabaseConnection,
}

impl SeaOrmModelRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<ModelDefinition> for SeaOrmModelRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<ModelDefinition>> {
        model::Entity::find_by_id(id)
            .filter(model::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<ModelDefinition>> {
        model::Entity::find()
            .filter(model::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &ModelDefinition) -> Result<ModelDefinition> {
        let active = model::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            model_type: Set(entity.model_type.to_string()),
            context_length: Set(entity.context_length),
            description: Set(entity.description.clone()),
            metadata: Set(entity.metadata.clone()),
            enabled: Set(entity.enabled),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = model::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &ModelDefinition) -> Result<ModelDefinition> {
        let active = model::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            model_type: Set(entity.model_type.to_string()),
            context_length: Set(entity.context_length),
            description: Set(entity.description.clone()),
            metadata: Set(entity.metadata.clone()),
            enabled: Set(entity.enabled),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = model::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<model::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<model::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<model::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ModelRepository for SeaOrmModelRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<ModelDefinition>> {
        model::Entity::find()
            .filter(model::Column::Name.eq(name))
            .filter(model::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_type(&self, model_type: &ModelType) -> Result<Vec<ModelDefinition>> {
        model::Entity::find()
            .filter(model::Column::ModelType.eq(model_type.to_string()))
            .filter(model::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_providers(&self, model_id: &str) -> Result<Vec<ProviderDefinition>> {
        let provider_ids: Vec<String> = model_provider::Entity::find()
            .filter(model_provider::Column::ModelId.eq(model_id))
            .filter(model_provider::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map_err(map_db_err)?
            .into_iter()
            .map(|mp| mp.provider_id)
            .collect();

        if provider_ids.is_empty() {
            return Ok(vec![]);
        }

        provider::Entity::find()
            .filter(provider::Column::Id.is_in(provider_ids))
            .filter(provider::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn add_provider(&self, rel: &ModelProvider) -> Result<()> {
        let active = model_provider::ActiveModel {
            model_id: Set(rel.model_id.clone()),
            provider_id: Set(rel.provider_id.clone()),
            model_identifier: Set(rel.model_identifier.clone()),
            is_default: Set(rel.is_default),
            priority: Set(rel.priority),
            enabled: Set(rel.enabled),
            created_at: Set(rel.created_at),
            deleted_at: Set(None),
        };
        model_provider::Entity::insert(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(())
    }

    async fn remove_provider(&self, model_id: &str, provider_id: &str) -> Result<()> {
        model_provider::Entity::delete_many()
            .filter(model_provider::Column::ModelId.eq(model_id))
            .filter(model_provider::Column::ProviderId.eq(provider_id))
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(())
    }
}

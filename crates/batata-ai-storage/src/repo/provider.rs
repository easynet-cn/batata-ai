use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::{ModelDefinition, ProviderDefinition};
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ProviderRepository, Repository};

use crate::entity::{model, model_provider, provider};

pub struct SeaOrmProviderRepository {
    db: DatabaseConnection,
}

impl SeaOrmProviderRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<ProviderDefinition> for SeaOrmProviderRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<ProviderDefinition>> {
        provider::Entity::find_by_id(id)
            .filter(provider::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<ProviderDefinition>> {
        provider::Entity::find()
            .filter(provider::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &ProviderDefinition) -> Result<ProviderDefinition> {
        let active = provider::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            provider_type: Set(entity.provider_type.clone()),
            api_key: Set(entity.api_key.clone()),
            base_url: Set(entity.base_url.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = provider::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &ProviderDefinition) -> Result<ProviderDefinition> {
        let active = provider::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            provider_type: Set(entity.provider_type.clone()),
            api_key: Set(entity.api_key.clone()),
            base_url: Set(entity.base_url.clone()),
            config: Set(entity.config.clone()),
            enabled: Set(entity.enabled),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = provider::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<provider::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<provider::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<provider::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ProviderRepository for SeaOrmProviderRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<ProviderDefinition>> {
        provider::Entity::find()
            .filter(provider::Column::Name.eq(name))
            .filter(provider::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_enabled(&self) -> Result<Vec<ProviderDefinition>> {
        provider::Entity::find()
            .filter(provider::Column::Enabled.eq(true))
            .filter(provider::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_models(&self, provider_id: &str) -> Result<Vec<ModelDefinition>> {
        let model_ids: Vec<String> = model_provider::Entity::find()
            .filter(model_provider::Column::ProviderId.eq(provider_id))
            .filter(model_provider::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map_err(map_db_err)?
            .into_iter()
            .map(|mp| mp.model_id)
            .collect();

        if model_ids.is_empty() {
            return Ok(vec![]);
        }

        model::Entity::find()
            .filter(model::Column::Id.is_in(model_ids))
            .filter(model::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}

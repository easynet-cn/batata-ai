use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::ObjectStoreConfig;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ObjectStoreConfigRepository, Repository};

use crate::entity::object_store_config;

pub struct SeaOrmObjectStoreConfigRepository {
    db: DatabaseConnection,
}

impl SeaOrmObjectStoreConfigRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<ObjectStoreConfig> for SeaOrmObjectStoreConfigRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<ObjectStoreConfig>> {
        object_store_config::Entity::find_by_id(id)
            .filter(object_store_config::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<ObjectStoreConfig>> {
        object_store_config::Entity::find()
            .filter(object_store_config::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &ObjectStoreConfig) -> Result<ObjectStoreConfig> {
        let active = object_store_config::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            backend: Set(entity.backend.to_string()),
            endpoint: Set(entity.endpoint.clone()),
            region: Set(entity.region.clone()),
            access_key: Set(entity.access_key.clone()),
            secret_key: Set(entity.secret_key.clone()),
            enabled: Set(entity.enabled),
            config: Set(entity.config.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = object_store_config::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &ObjectStoreConfig) -> Result<ObjectStoreConfig> {
        let active = object_store_config::ActiveModel {
            id: Set(entity.id.clone()),
            name: Set(entity.name.clone()),
            backend: Set(entity.backend.to_string()),
            endpoint: Set(entity.endpoint.clone()),
            region: Set(entity.region.clone()),
            access_key: Set(entity.access_key.clone()),
            secret_key: Set(entity.secret_key.clone()),
            enabled: Set(entity.enabled),
            config: Set(entity.config.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = object_store_config::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<object_store_config::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<object_store_config::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<object_store_config::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ObjectStoreConfigRepository for SeaOrmObjectStoreConfigRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<ObjectStoreConfig>> {
        object_store_config::Entity::find()
            .filter(object_store_config::Column::Name.eq(name))
            .filter(object_store_config::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_enabled(&self) -> Result<Vec<ObjectStoreConfig>> {
        object_store_config::Entity::find()
            .filter(object_store_config::Column::Enabled.eq(true))
            .filter(object_store_config::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}

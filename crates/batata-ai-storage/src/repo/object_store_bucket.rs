use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::ObjectStoreBucket;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ObjectStoreBucketRepository, Repository};

use crate::entity::object_store_bucket;

pub struct SeaOrmObjectStoreBucketRepository {
    db: DatabaseConnection,
}

impl SeaOrmObjectStoreBucketRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<ObjectStoreBucket> for SeaOrmObjectStoreBucketRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<ObjectStoreBucket>> {
        object_store_bucket::Entity::find_by_id(id)
            .filter(object_store_bucket::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<ObjectStoreBucket>> {
        object_store_bucket::Entity::find()
            .filter(object_store_bucket::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &ObjectStoreBucket) -> Result<ObjectStoreBucket> {
        let active = object_store_bucket::ActiveModel {
            id: Set(entity.id.clone()),
            config_id: Set(entity.config_id.clone()),
            name: Set(entity.name.clone()),
            bucket: Set(entity.bucket.clone()),
            root_path: Set(entity.root_path.clone()),
            access_policy: Set(entity.access_policy.to_string()),
            custom_domain: Set(entity.custom_domain.clone()),
            is_default: Set(entity.is_default),
            enabled: Set(entity.enabled),
            config: Set(entity.config.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = object_store_bucket::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &ObjectStoreBucket) -> Result<ObjectStoreBucket> {
        let active = object_store_bucket::ActiveModel {
            id: Set(entity.id.clone()),
            config_id: Set(entity.config_id.clone()),
            name: Set(entity.name.clone()),
            bucket: Set(entity.bucket.clone()),
            root_path: Set(entity.root_path.clone()),
            access_policy: Set(entity.access_policy.to_string()),
            custom_domain: Set(entity.custom_domain.clone()),
            is_default: Set(entity.is_default),
            enabled: Set(entity.enabled),
            config: Set(entity.config.clone()),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = object_store_bucket::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<object_store_bucket::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<object_store_bucket::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<object_store_bucket::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ObjectStoreBucketRepository for SeaOrmObjectStoreBucketRepository {
    async fn find_by_name(&self, name: &str) -> Result<Option<ObjectStoreBucket>> {
        object_store_bucket::Entity::find()
            .filter(object_store_bucket::Column::Name.eq(name))
            .filter(object_store_bucket::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_config(&self, config_id: &str) -> Result<Vec<ObjectStoreBucket>> {
        object_store_bucket::Entity::find()
            .filter(object_store_bucket::Column::ConfigId.eq(config_id))
            .filter(object_store_bucket::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_enabled(&self) -> Result<Vec<ObjectStoreBucket>> {
        object_store_bucket::Entity::find()
            .filter(object_store_bucket::Column::Enabled.eq(true))
            .filter(object_store_bucket::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn find_default(&self) -> Result<Option<ObjectStoreBucket>> {
        object_store_bucket::Entity::find()
            .filter(object_store_bucket::Column::IsDefault.eq(true))
            .filter(object_store_bucket::Column::Enabled.eq(true))
            .filter(object_store_bucket::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }
}

use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::StoredObject;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::StoredObjectRepository;

use crate::entity::stored_object;

pub struct SeaOrmStoredObjectRepository {
    db: DatabaseConnection,
}

impl SeaOrmStoredObjectRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl StoredObjectRepository for SeaOrmStoredObjectRepository {
    async fn create(&self, obj: &StoredObject) -> Result<StoredObject> {
        let active = stored_object::ActiveModel {
            id: Set(obj.id.clone()),
            bucket_id: Set(obj.bucket_id.clone()),
            tenant_id: Set(obj.tenant_id.clone()),
            key: Set(obj.key.clone()),
            original_name: Set(obj.original_name.clone()),
            content_type: Set(obj.content_type.clone()),
            size: Set(obj.size),
            checksum: Set(obj.checksum.clone()),
            metadata: Set(obj.metadata.clone()),
            created_at: Set(obj.created_at),
            deleted_at: Set(None),
        };
        let result = stored_object::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn find_by_id(&self, id: &str) -> Result<Option<StoredObject>> {
        stored_object::Entity::find_by_id(id)
            .filter(stored_object::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_key(&self, key: &str) -> Result<Option<StoredObject>> {
        stored_object::Entity::find()
            .filter(stored_object::Column::Key.eq(key))
            .filter(stored_object::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_keys(&self, keys: &[String]) -> Result<Vec<StoredObject>> {
        if keys.is_empty() {
            return Ok(vec![]);
        }
        stored_object::Entity::find()
            .filter(stored_object::Column::Key.is_in(keys.to_vec()))
            .filter(stored_object::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<stored_object::Entity>(&self.db, id).await
    }

    async fn find_by_bucket(
        &self,
        bucket_id: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<StoredObject>> {
        stored_object::Entity::find()
            .filter(stored_object::Column::BucketId.eq(bucket_id))
            .filter(stored_object::Column::DeletedAt.is_null())
            .order_by_desc(stored_object::Column::CreatedAt)
            .offset(page * page_size)
            .limit(page_size)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn count_by_bucket(&self, bucket_id: &str) -> Result<u64> {
        stored_object::Entity::find()
            .filter(stored_object::Column::BucketId.eq(bucket_id))
            .filter(stored_object::Column::DeletedAt.is_null())
            .count(&self.db)
            .await
            .map_err(map_db_err)
    }

    async fn find_by_content_type(
        &self,
        bucket_id: &str,
        content_type_prefix: &str,
        page: u64,
        page_size: u64,
    ) -> Result<Vec<StoredObject>> {
        stored_object::Entity::find()
            .filter(stored_object::Column::BucketId.eq(bucket_id))
            .filter(stored_object::Column::ContentType.starts_with(content_type_prefix))
            .filter(stored_object::Column::DeletedAt.is_null())
            .order_by_desc(stored_object::Column::CreatedAt)
            .offset(page * page_size)
            .limit(page_size)
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }
}

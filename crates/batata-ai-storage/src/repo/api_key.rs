use async_trait::async_trait;
use sea_orm::*;

use batata_ai_core::domain::ApiKey;
use batata_ai_core::error::{BatataError, Result};
use batata_ai_core::repository::{ApiKeyRepository, Repository};

use crate::entity::api_key;

pub struct SeaOrmApiKeyRepository {
    db: DatabaseConnection,
}

impl SeaOrmApiKeyRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

fn map_db_err(e: DbErr) -> BatataError {
    BatataError::Storage(e.to_string())
}

#[async_trait]
impl Repository<ApiKey> for SeaOrmApiKeyRepository {
    async fn find_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        api_key::Entity::find_by_id(id)
            .filter(api_key::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_all(&self) -> Result<Vec<ApiKey>> {
        api_key::Entity::find()
            .filter(api_key::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn create(&self, entity: &ApiKey) -> Result<ApiKey> {
        let active = api_key::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            key_hash: Set(entity.key_hash.clone()),
            key_prefix: Set(entity.key_prefix.clone()),
            app_key: Set(entity.app_key.clone()),
            app_secret_hash: Set(entity.app_secret_hash.clone()),
            scopes: Set(entity.scopes.clone()),
            rate_limit: Set(entity.rate_limit),
            expires_at: Set(entity.expires_at),
            enabled: Set(entity.enabled),
            last_used_at: Set(entity.last_used_at),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(None),
        };
        let result = api_key::Entity::insert(active)
            .exec_with_returning(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn update(&self, entity: &ApiKey) -> Result<ApiKey> {
        let active = api_key::ActiveModel {
            id: Set(entity.id.clone()),
            tenant_id: Set(entity.tenant_id.clone()),
            name: Set(entity.name.clone()),
            key_hash: Set(entity.key_hash.clone()),
            key_prefix: Set(entity.key_prefix.clone()),
            app_key: Set(entity.app_key.clone()),
            app_secret_hash: Set(entity.app_secret_hash.clone()),
            scopes: Set(entity.scopes.clone()),
            rate_limit: Set(entity.rate_limit),
            expires_at: Set(entity.expires_at),
            enabled: Set(entity.enabled),
            last_used_at: Set(entity.last_used_at),
            created_at: Set(entity.created_at),
            updated_at: Set(entity.updated_at),
            deleted_at: Set(entity.deleted_at),
        };
        let result = api_key::Entity::update(active)
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(result.into())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::soft_delete::<api_key::Entity>(&self.db, id).await
    }

    async fn hard_delete(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::hard_delete::<api_key::Entity>(&self.db, id).await
    }

    async fn restore(&self, id: &str) -> Result<bool> {
        crate::repo::soft_delete::restore::<api_key::Entity>(&self.db, id).await
    }
}

#[async_trait]
impl ApiKeyRepository for SeaOrmApiKeyRepository {
    async fn find_by_key_hash(&self, key_hash: &str) -> Result<Option<ApiKey>> {
        api_key::Entity::find()
            .filter(api_key::Column::KeyHash.eq(key_hash))
            .filter(api_key::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_prefix(&self, prefix: &str) -> Result<Option<ApiKey>> {
        api_key::Entity::find()
            .filter(api_key::Column::KeyPrefix.eq(prefix))
            .filter(api_key::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_app_key(&self, app_key: &str) -> Result<Option<ApiKey>> {
        api_key::Entity::find()
            .filter(api_key::Column::AppKey.eq(app_key))
            .filter(api_key::Column::DeletedAt.is_null())
            .one(&self.db)
            .await
            .map(|opt| opt.map(Into::into))
            .map_err(map_db_err)
    }

    async fn find_by_tenant(&self, tenant_id: &str) -> Result<Vec<ApiKey>> {
        api_key::Entity::find()
            .filter(api_key::Column::TenantId.eq(tenant_id))
            .filter(api_key::Column::DeletedAt.is_null())
            .all(&self.db)
            .await
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(map_db_err)
    }

    async fn touch_last_used(&self, id: &str) -> Result<()> {
        let now = chrono::Utc::now().naive_utc();
        let col_last_used: api_key::Column = "last_used_at".parse().map_err(|e| {
            BatataError::Storage(format!("cannot parse 'last_used_at' column: {:?}", e))
        })?;
        let col_id: api_key::Column = "id".parse().map_err(|e| {
            BatataError::Storage(format!("cannot parse 'id' column: {:?}", e))
        })?;

        api_key::Entity::update_many()
            .col_expr(
                col_last_used,
                sea_orm::sea_query::Expr::value(sea_orm::Value::ChronoDateTime(Some(Box::new(
                    now,
                )))),
            )
            .filter(col_id.eq(id))
            .exec(&self.db)
            .await
            .map_err(map_db_err)?;
        Ok(())
    }
}
